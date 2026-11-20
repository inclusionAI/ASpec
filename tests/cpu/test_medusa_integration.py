# Copyright 2026 Ant Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Medusa integration contracts: registration, Plan, features, dispatch and resume."""

import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from aspec.config.draft_algorithm import draft_model_semantics, resolve_draft_algorithm
from aspec.config.draft_attention import resolve_draft_attention, resolve_draft_head_layout
from aspec.config.draft_structure import resolve_draft_structure
from aspec.config.train_config import load_config
from aspec.feature_contract import FeatureContractError, build_target_feature_contract
from aspec.models.draft.auto import AutoDraftModelConfig, AutoEagle3DraftModel
from aspec.models.draft.medusa import MedusaConfig, MedusaDraftModel
from aspec.models.medusa import MedusaModel
from aspec.planning import (
    PlanCompileError,
    _load_draft_descriptor,
    _runtime_values,
    compile_execution_plan,
)
from aspec.training.checkpoint import (
    _logical_training_topology,
    _validate_exact_resume_topology,
)
from aspec.training.medusa_trainer import MedusaTrainer
from aspec.training.trainer_actor import _trainer_class_for_config

_REPO_ROOT = Path(__file__).parents[2]
_EXAMPLE = _REPO_ROOT / "configs" / "examples" / "medusa_qwen3_8b_4k_tcp.yaml"
_DRAFT_JSON = _REPO_ROOT / "aspec" / "config" / "medusa_qwen3_8b_draft_config.json"

_ILLEGAL_OVERRIDES = [
    ("training.dflash_packing=true", PlanCompileError),
    # CP>1 is rejected by the shared draft-execution contract inside load_config;
    # the family guard is covered directly in test_direct_compile_rejects_medusa_cp.
    ("training.draft_context_parallel_size=2", NotImplementedError),
    ("training.micro_batch_size=2", PlanCompileError),
    ("training.fsdp_strategy=FULL_SHARD", PlanCompileError),
    ("training.attention_backend=fa4", PlanCompileError),
    ("training.attention_backend=cudnn_flex_attention", PlanCompileError),
    ("inference.store_last_hidden_states=true", PlanCompileError),
    ("training.compile_model=true", PlanCompileError),
    ("model.freeze_lm_head=true", PlanCompileError),
    ("training.ploss_weights=[1.0,1.0]", PlanCompileError),
    ("training.eagle3_num_anchors=1", PlanCompileError),
]


def _tiny_config(**overrides):
    raw = dict(architectures=["MedusaDraftModel"], model_type="aspec_medusa",
               hidden_size=16, vocab_size=32, target_hidden_size=16,
               target_num_hidden_layers=4, medusa_num_heads=3, medusa_num_layers=1)
    raw.update(overrides)
    return AutoDraftModelConfig.from_dict(raw)


def _fake_qwen_target(tmp_path, vocab_size=151936):
    target = tmp_path / "Qwen3-8B-test"
    target.mkdir(parents=True, exist_ok=True)
    (target / "config.json").write_text(json.dumps({
        "model_type": "qwen3",
        "architectures": ["Qwen3ForCausalLM"],
        "hidden_size": 4096,
        "vocab_size": vocab_size,
        "num_hidden_layers": 36,
        "max_position_embeddings": 40960,
    }), encoding="utf-8")
    return target


def _config(tmp_path, *overrides):
    args = [f"model.target_model_path={_fake_qwen_target(tmp_path)}", *overrides]
    return load_config(config_path=str(_EXAMPLE), cli_args=args), args


def test_registration_resolves_medusa():
    assert resolve_draft_algorithm({"architectures": ["MedusaDraftModel"]}) == "medusa"
    shipped = AutoDraftModelConfig.from_file(str(_DRAFT_JSON))
    assert isinstance(shipped, MedusaConfig)
    assert shipped.medusa_num_heads == 5
    assert shipped.medusa_num_layers == 1
    assert shipped.target_layer_ids == [35]
    assert shipped.num_aux_hidden_states == 1

    tiny = _tiny_config()
    model = AutoEagle3DraftModel.from_config(tiny, attention_backend="sdpa")
    assert isinstance(model, MedusaDraftModel)
    assert len(model.medusa_head) == 3
    with pytest.raises(NotImplementedError):
        AutoEagle3DraftModel.from_config(tiny, attention_backend="fa4")
    assert _trainer_class_for_config(tiny) is MedusaTrainer
    semantics = draft_model_semantics(tiny)
    assert semantics["medusa_num_heads"] == 3
    assert semantics["medusa_num_layers"] == 1


def test_loader_does_not_rewrite_heads_only_validation():
    raw = {
        "architectures": ["MedusaDraftModel"], "hidden_size": 16, "vocab_size": 32,
        "target_hidden_size": 16, "target_num_hidden_layers": 4,
    }
    with pytest.raises(ValueError):
        AutoDraftModelConfig.from_dict({**raw, "tie_word_embeddings": True})
    for override in (
        {"tie_word_embeddings": False, "sliding_window": 128},
        {"tie_word_embeddings": False, "layer_types": ["sliding_attention"]},
        {"tie_word_embeddings": False, "use_sliding_window": True},
        {"tie_word_embeddings": False, "num_experts": 8},
        {"tie_word_embeddings": False, "num_experts_per_tok": 2},
        {"tie_word_embeddings": False, "moe_intermediate_size": 64},
    ):
        with pytest.raises(ValueError):
            AutoDraftModelConfig.from_dict({**raw, **override})


def test_heads_only_policies():
    raw = _tiny_config().to_dict()
    attention = resolve_draft_attention(raw)
    assert attention["layer_types"] == []
    assert attention["context_windows"] == []
    assert attention["draft_block_attention"] == "parallel_heads"
    layout = resolve_draft_head_layout(raw)
    assert layout["type"] == "heads_only"
    assert layout["head_dim"] is None
    assert resolve_draft_structure(raw) == {"ffn_type": "heads_only"}
    for override in (
        {"compress_ratios": [0, 0, 4]},
        {"csa_training_stage": "dense"},
        {"qsa_training_stage": "dense"},
        {"dsa_training_stage": "dense"},
    ):
        with pytest.raises(ValueError, match="compressed or sparse attention"):
            resolve_draft_attention({**raw, **override})
    for override in (
        {"router_z_loss_coef": 0.1},
        {"moe_router_enable_expert_bias": True},
    ):
        with pytest.raises(ValueError, match="expert routing"):
            resolve_draft_structure({**raw, **override})


def test_feature_contract_requests_one_post_norm_state():
    contract = build_target_feature_contract(
        drafter_family="medusa", target_hidden_size=4096, target_num_hidden_layers=36,
        aux_hidden_state_layers=[35], store_last_hidden_states=False)
    hidden = contract["tensors"]["hidden_states"]
    assert hidden["semantic"] == "target_final_hidden_state"
    assert hidden["value_stage"] == "post_norm"
    assert "last_hidden_states" not in contract["tensors"]
    with pytest.raises(FeatureContractError):
        build_target_feature_contract(
            drafter_family="medusa", target_hidden_size=4096, target_num_hidden_layers=36,
            aux_hidden_state_layers=[35], store_last_hidden_states=True)
    with pytest.raises(FeatureContractError):
        build_target_feature_contract(
            drafter_family="medusa", target_hidden_size=4096, target_num_hidden_layers=36,
            aux_hidden_state_layers=[34, 35], store_last_hidden_states=False)
    with pytest.raises(FeatureContractError):
        build_target_feature_contract(
            drafter_family="medusa", target_hidden_size=4096, target_num_hidden_layers=36,
            aux_hidden_state_layers=[34], store_last_hidden_states=False)


def test_plan_compiles_with_medusa_objective(tmp_path):
    config, args = _config(tmp_path)
    plan = compile_execution_plan(config, config_path=str(_EXAMPLE), cli_args=args).to_dict()
    draft = plan["workload"]["models"]["draft"]
    assert draft["family"] == "medusa"
    assert draft["structure"] == {"ffn_type": "heads_only"}
    assert draft["attention"]["head_layout"]["type"] == "heads_only"
    objective = plan["workload"]["objective"]
    assert objective["num_heads"] == 5
    assert objective["num_layers"] == 1
    assert objective["loss"] == "parallel_future_token_ce"
    assert objective["head_weighting"] == "sum_of_per_head_mean_ce"
    assert objective["prediction_start"] == 2
    features = plan["workload"]["features"]
    assert features["aux_hidden_state_layers"] == [35]
    assert features["tensors"]["hidden_states"]["semantic"] == "target_final_hidden_state"
    assert "last_hidden_states" not in features["tensors"]

    runtime = _runtime_values(config, _load_draft_descriptor(config))
    assert runtime["draft_algorithm"] == "medusa"
    assert runtime["ttt_length"] == 5
    assert runtime["ploss_weights"] == [1.0] * 5
    assert runtime["freeze_lm_head"] is False


@pytest.mark.parametrize("override, error", _ILLEGAL_OVERRIDES)
def test_plan_rejects_unsupported_medusa_combinations(tmp_path, override, error):
    target = f"model.target_model_path={_fake_qwen_target(tmp_path)}"
    args = [target, override]
    with pytest.raises(error):
        config = load_config(config_path=str(_EXAMPLE), cli_args=args)
        compile_execution_plan(config, config_path=str(_EXAMPLE), cli_args=args)


def test_direct_compile_rejects_medusa_cp(tmp_path):
    # load_config normally rejects CP>1 first; keep the family guard effective
    # for callers that hand a config straight to plan compilation.
    config, _ = _config(tmp_path)
    config.training.draft_context_parallel_size = 2
    with pytest.raises(PlanCompileError, match="draft_context_parallel_size=1"):
        _load_draft_descriptor(config)


def test_plan_rejects_draft_target_vocabulary_mismatch(tmp_path):
    target = f"model.target_model_path={_fake_qwen_target(tmp_path, vocab_size=151935)}"
    args = [target]
    with pytest.raises(PlanCompileError, match="Target vocabulary"):
        config = load_config(config_path=str(_EXAMPLE), cli_args=args)
        compile_execution_plan(config, config_path=str(_EXAMPLE), cli_args=args)


@pytest.mark.parametrize("key, value, message", [
    ("compress_ratios", [0, 0, 0, 4], "compressed or sparse attention"),
    ("dsa_training_stage", "dense", "compressed or sparse attention"),
    ("router_z_loss_coef", 0.1, "FFN or expert routing"),
])
def test_plan_rejects_heads_only_ignored_settings(tmp_path, key, value, message):
    raw = json.loads(_DRAFT_JSON.read_text(encoding="utf-8"))
    raw[key] = value
    draft = tmp_path / "medusa_ignored_settings.json"
    draft.write_text(json.dumps(raw), encoding="utf-8")
    args = [
        f"model.target_model_path={_fake_qwen_target(tmp_path)}",
        f"model.draft_model_config={draft}",
    ]
    with pytest.raises(PlanCompileError, match=message):
        config = load_config(config_path=str(_EXAMPLE), cli_args=args)
        compile_execution_plan(config, config_path=str(_EXAMPLE), cli_args=args)


@pytest.mark.parametrize("key, value", [
    ("training_num_nodes", 2),
    ("cluster", "static_multi_node"),
])
def test_direct_compile_rejects_medusa_multi_node(tmp_path, key, value):
    # The shared cluster contract rejects multi-node configs before family
    # gates; mutate the loaded config to cover the medusa guard itself.
    config, _ = _config(tmp_path)
    if key == "cluster":
        config.cluster.mode = value
    else:
        setattr(config.training, key, value)
    with pytest.raises(PlanCompileError, match="single node"):
        _load_draft_descriptor(config)


def test_producer_publishes_final_state_for_medusa(monkeypatch):
    for name in ("sglang", "sglang.srt", "sglang.srt.model_executor"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    sys.modules["sglang.srt.model_executor"].hook_manager = SimpleNamespace(
        SUPPORTS_MODEL_HOOK_SETUP=True)
    from aspec.inference.sglang_v0521 import training_engine_kwargs

    args = SimpleNamespace(
        train_with_decode=False, draft_context_parallel_size=1,
        draft_context_shard_alignment=1, max_seq_length=4096,
        store_last_hidden_states=False, target_prefill_memory_optimized=False,
        attention_backend="sdpa", mooncake_hidden_dim=4096, draft_algorithm="medusa")
    kwargs = {"aux_hidden_state_layer_ids": [35]}
    published = training_engine_kwargs(kwargs, args)["forward_hooks"][-1]["config"]
    assert published["final_hidden_as_aux"] is True
    args.draft_algorithm = "mtp"
    assert training_engine_kwargs(kwargs, args)["forward_hooks"][-1]["config"][
        "final_hidden_as_aux"] is True
    args.draft_algorithm = "dspark"
    assert training_engine_kwargs(kwargs, args)["forward_hooks"][-1]["config"][
        "final_hidden_as_aux"] is False


def _trainer_args(config, **overrides):
    values = dict(
        draft_algorithm="medusa", ttt_length=3, ploss_weights=[1.0, 1.0, 1.0],
        draft_model_config_obj=config, world_size=2, freeze_lm_head=False,
        max_seq_length=4096, micro_batch_size=1, draft_accumulation_steps=1,
        max_grad_norm=1.0, seed=42, draft_context_parallel_size=1,
        training_num_nodes=1, training_num_gpus_per_node=2, attention_backend="sdpa",
        fsdp_strategy="REPLICATE", sp_ulysses_size=1, sp_ring_size=1,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def test_target_head_binding_is_rng_free_and_reuses_an_existing_head():
    from aspec.training.medusa_trainer import MedusaTrainer

    trainer = object.__new__(MedusaTrainer)
    draft = MedusaDraftModel(_tiny_config())
    trainer.draft_model = draft
    before = torch.get_rng_state().clone()
    trainer._init_target_lm_head("unused")
    assert trainer.target_lm_head.lm_head is draft.medusa_head[0][-1]
    assert trainer.target_lm_head.norm is None
    assert torch.equal(torch.get_rng_state(), before)


def test_frozen_head_is_rejected():
    from aspec.training.medusa_trainer import MedusaTrainer

    trainer = object.__new__(MedusaTrainer)
    trainer.args = SimpleNamespace(freeze_lm_head=True, ttt_length=3)
    with pytest.raises(ValueError):
        trainer._prepare_algorithm_training(MedusaDraftModel(_tiny_config()))
    # Bypassing Plan: freeze_lm_head() is the first call, and it must reject
    # rather than raise AttributeError on the missing embedding-owned head.
    with pytest.raises(ValueError, match="trainable vocabulary projections"):
        MedusaDraftModel(_tiny_config()).freeze_lm_head()


def test_head_count_drift_is_rejected():
    from aspec.training.medusa_trainer import MedusaTrainer

    trainer = object.__new__(MedusaTrainer)
    trainer.args = SimpleNamespace(freeze_lm_head=False, ttt_length=5)
    with pytest.raises(ValueError, match="drifted"):
        trainer._prepare_algorithm_training(MedusaDraftModel(_tiny_config()))


def test_target_vocabulary_is_bound_at_startup(tmp_path):
    from aspec.training.medusa_trainer import MedusaTrainer

    trainer = object.__new__(MedusaTrainer)
    trainer.args = SimpleNamespace(trust_remote_code=False)
    draft = MedusaDraftModel(_tiny_config())
    trainer._load_algorithm_weights(draft, str(_fake_qwen_target(tmp_path, vocab_size=32)))
    mismatched = _fake_qwen_target(tmp_path / "mismatch", vocab_size=31)
    with pytest.raises(ValueError, match="full Target vocabulary"):
        trainer._load_algorithm_weights(draft, str(mismatched))


def test_startup_vocabulary_honors_sglang_overrides(tmp_path):
    from aspec.training.medusa_trainer import MedusaTrainer

    trainer = object.__new__(MedusaTrainer)
    trainer.args = SimpleNamespace(trust_remote_code=False, sglang_extra_args={})
    draft = MedusaDraftModel(_tiny_config())
    target = str(_fake_qwen_target(tmp_path, vocab_size=33))
    with pytest.raises(ValueError, match="full Target vocabulary"):
        trainer._load_algorithm_weights(draft, target)
    for overrides in ('{"vocab_size": 32}', {"vocab_size": 32}):
        trainer.args.sglang_extra_args = {"json_model_override_args": overrides}
        trainer._load_algorithm_weights(draft, target)
    trainer.args.sglang_extra_args = {"json_model_override_args": {"vocab_size": 34}}
    with pytest.raises(ValueError, match="full Target vocabulary"):
        trainer._load_algorithm_weights(draft, target)


def test_metric_semantics_drop_unmeasured_acceptance():
    from aspec.training.medusa_trainer import MedusaTrainer

    trainer = object.__new__(MedusaTrainer)
    metrics = {
        "train/avg_loss": 1.0, "train/avg_acc": 0.5, "train/avg_alpha": 0.0,
        "train/simulated_acc_len": 0.25, "train/alpha_0": 0.0,
        "train/ploss_0": 1.0, "train/acc_0": 0.5,
    }
    cleaned = trainer._metric_semantics(dict(metrics), "train")
    assert set(cleaned) == {"train/avg_loss", "train/avg_acc", "train/ploss_0", "train/acc_0"}


def test_training_model_forward_and_feature_status(monkeypatch):
    trainer = object.__new__(MedusaTrainer)
    trainer.args = SimpleNamespace(
        gradient_checkpointing=True, attention_backend="sdpa",
        draft_context_parallel_size=1, ttt_length=3)
    trainer.data_fetcher = None
    trainer.dp_rank = 0
    trainer.draft_model = MedusaDraftModel(_tiny_config())
    trainer.model = trainer._create_training_model(trainer.draft_model)
    assert isinstance(trainer.model, MedusaModel)
    assert trainer.model.gradient_checkpointing is True

    monkeypatch.setattr(torch.Tensor, "cuda", lambda self, *args, **kwargs: self)
    losses, detached, accuracy, counts, pad = trainer._forward({
        "input_ids": torch.arange(8).reshape(1, 8),
        "hidden_states": torch.randn(1, 8, 16),
        "loss_mask": torch.ones(1, 8, 1),
        "attention_mask": torch.ones(1, 8),
    })
    assert len(losses) == len(detached) == len(accuracy) == len(pad) == 3
    # Head i predicts t+2+i, so the tail shortens by the offset.
    assert [int(count) for count in counts] == [6, 5, 4]

    status = trainer.get_feature_contract_status()
    assert status["state"] == "not_ready"
    assert status["medusa_num_heads"] == 3
    assert status["medusa_num_layers"] == 1


def test_checkpoint_topology_pins_head_count():
    config = _tiny_config()
    args = _trainer_args(config)
    current = _logical_training_topology(args, 2)
    objective = current["training_objective"]
    assert objective["ttt_length"] == 3
    assert objective["ploss_weights"] == [1.0, 1.0, 1.0]
    assert objective["head_weighting"] == "sum_of_per_head_mean_ce"
    assert current["draft_ffn_policy"] == {"ffn_type": "heads_only"}

    metadata = {"world_size": 2,
                "training_topology": json.loads(json.dumps(current))}
    _validate_exact_resume_topology(args, metadata)
    changed = json.loads(json.dumps(metadata))
    changed["training_topology"]["training_objective"]["ttt_length"] = 5
    with pytest.raises(RuntimeError):
        _validate_exact_resume_topology(args, changed)
