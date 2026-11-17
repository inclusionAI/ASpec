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
    ("training.draft_context_parallel_size=2", NotImplementedError),
    ("training.micro_batch_size=2", PlanCompileError),
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


def _fake_qwen_target(tmp_path):
    target = tmp_path / "Qwen3-8B-test"
    target.mkdir()
    (target / "config.json").write_text(json.dumps({
        "model_type": "qwen3",
        "architectures": ["Qwen3ForCausalLM"],
        "hidden_size": 4096,
        "vocab_size": 151936,
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


def test_producer_publishes_final_state_for_medusa(monkeypatch):
    for name in ("sglang", "sglang.srt", "sglang.srt.model_executor",
                 "mooncake", "mooncake.store"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    sys.modules["sglang.srt.model_executor"].hook_manager = SimpleNamespace(
        SUPPORTS_MODEL_HOOK_SETUP=True)
    sys.modules["mooncake.store"].MooncakeDistributedStore = object
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
    trainer.args = SimpleNamespace(freeze_lm_head=True)
    with pytest.raises(ValueError):
        trainer._prepare_algorithm_training(MedusaDraftModel(_tiny_config()))


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
