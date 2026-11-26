# Copyright 2026 Ant Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""JetSpec integration contracts: registration, Plan, dispatch and resume."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from aspec.config.draft_algorithm import draft_model_semantics, resolve_draft_algorithm
from aspec.config.train_config import load_config
from aspec.models.draft.auto import AutoDraftModelConfig, AutoEagle3DraftModel
from aspec.models.draft.jetspec import JetSpecConfig, JetSpecDraftModel
from aspec.models.jetspec import JetSpecModel
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
from aspec.training.dflash_trainer import DFlashTrainer
from aspec.training.jetspec_trainer import JetSpecTrainer
from aspec.training.trainer_actor import _trainer_class_for_config
from tools.convert_to_hf import _serving_config

_REPO_ROOT = Path(__file__).parents[2]
_EXAMPLE = _REPO_ROOT / "configs" / "examples" / "jetspec_qwen3_8b_4k_tcp.yaml"
_DRAFT_JSON = _REPO_ROOT / "aspec" / "config" / "jetspec_qwen3_8b_draft_config.json"

_ILLEGAL_OVERRIDES = [
    ("training.dflash_packing=true", PlanCompileError),
    ("training.draft_context_parallel_size=2", NotImplementedError),
    ("training.attention_backend=sdpa", PlanCompileError),
    ("training.attention_backend=fa4", PlanCompileError),
    ("training.attention_backend=cudnn_flex_attention", PlanCompileError),
    ("training.compile_model=true", PlanCompileError),
    ("training.dflash_loss_objective=dpace", PlanCompileError),
    ("training.dflash_ce_loss_alpha=0.5", PlanCompileError),
    ("training.dflash_l1_loss_alpha=0.1", PlanCompileError),
    ("training.dflash_block_size=8", PlanCompileError),
    ("training.jetspec_distillation_temperature=0.0", PlanCompileError),
    ("training.jetspec_distillation_temperature=2.0e19", PlanCompileError),
    ("training.jetspec_kl_chunk_rows=0", PlanCompileError),
    ("training.jetspec_loss_decay_gamma=-1.0", PlanCompileError),
    ("inference.store_last_hidden_states=false", PlanCompileError),
]


def _tiny_config(**overrides):
    raw = dict(
        architectures=["JetSpecDraftModel"], model_type="jetspec",
        hidden_size=16, intermediate_size=32, num_hidden_layers=1,
        num_attention_heads=2, num_key_value_heads=1, head_dim=8, vocab_size=32,
        target_hidden_size=16, target_num_hidden_layers=4, target_layer_ids=[1],
        num_target_layers=1, mask_token_id=31, max_position_embeddings=128,
        block_size=5, is_causal=True,
    )
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


def test_registration_resolves_jetspec():
    assert resolve_draft_algorithm({"architectures": ["JetSpecDraftModel"]}) == "jetspec"
    shipped = AutoDraftModelConfig.from_file(str(_DRAFT_JSON))
    assert isinstance(shipped, JetSpecConfig)
    assert shipped.block_size == 16
    assert shipped.is_causal is True
    assert shipped.target_layer_ids == [1, 9, 17, 25, 33]

    tiny = _tiny_config()
    model = AutoEagle3DraftModel.from_config(tiny)
    assert isinstance(model, JetSpecDraftModel)
    assert _trainer_class_for_config(tiny) is JetSpecTrainer
    semantics = draft_model_semantics(tiny)
    assert semantics["architectures"] == ["JetSpecDraftModel"]
    assert semantics["is_causal"] is True
    assert semantics["block_size"] == 5


def test_sglang_serving_export_uses_causal_dflash_layout():
    raw = json.loads(_DRAFT_JSON.read_text())
    config = _serving_config(raw)
    assert config["architectures"] == ["DFlashDraftModel"]
    assert config["is_causal"] is True
    assert config["dflash_config"]["causal_head"] is True



def test_plan_accepts_published_jetspec_layout(tmp_path):
    from aspec.models.draft.jetspec import official_jetspec_config

    raw = json.loads(_DRAFT_JSON.read_text())
    published = official_jetspec_config(raw)
    config_path = tmp_path / "published_draft_config.json"
    config_path.write_text(json.dumps(published))
    config, args = _config(tmp_path, f"model.draft_model_config={config_path}")

    plan = compile_execution_plan(
        config, config_path=str(_EXAMPLE), cli_args=args
    ).to_dict()

    assert plan["workload"]["models"]["draft"]["family"] == "jetspec"
    assert plan["workload"]["features"]["aux_hidden_state_layers"] == raw["target_layer_ids"]

def test_plan_compiles_with_jetspec_objective(tmp_path):
    config, args = _config(tmp_path)
    plan = compile_execution_plan(config, config_path=str(_EXAMPLE), cli_args=args).to_dict()
    draft = plan["workload"]["models"]["draft"]
    assert draft["family"] == "jetspec"
    objective = plan["workload"]["objective"]
    assert objective["loss"] == "temperature_scaled_forward_kl"
    assert objective["block_size"] == 16
    assert objective["prediction_start"] == 1
    assert objective["teacher_feature"] == "last_hidden_states"
    assert objective["distillation_temperature"] == 1.0
    assert objective["kl_chunk_rows"] == 256
    assert objective["loss_decay_gamma"] == 0.0
    assert objective["ce_loss_alpha"] == 0.0
    assert objective["l1_loss_alpha"] == 0.0
    assert objective["frozen_parts"] == ["target_lm_head"]
    features = plan["workload"]["features"]
    assert features["aux_hidden_state_layers"] == [1, 9, 17, 25, 33]
    assert "last_hidden_states" in features["tensors"]
    attention = next(
        item for item in plan["capabilities"] if item["name"] == "draft_attention_execution"
    )
    assert attention["requirement"] == "required"

    runtime = _runtime_values(config, _load_draft_descriptor(config))
    assert runtime["draft_algorithm"] == "jetspec"
    assert runtime["jetspec_distillation_temperature"] == 1.0
    assert runtime["jetspec_kl_chunk_rows"] == 256
    assert runtime["jetspec_loss_decay_gamma"] == 0.0


@pytest.mark.parametrize("temperature, accepted", [(1.0, True), (1.8446744e19, False)])
def test_plan_matches_trainer_temperature_limit(tmp_path, temperature, accepted):
    override = f"training.jetspec_distillation_temperature={temperature}"
    if accepted:
        config, args = _config(tmp_path, override)
        plan = compile_execution_plan(
            config, config_path=str(_EXAMPLE), cli_args=args
        ).to_dict()
        assert plan["workload"]["objective"]["distillation_temperature"] == temperature
    else:
        with pytest.raises(PlanCompileError, match="jetspec_distillation_temperature"):
            config, args = _config(tmp_path, override)
            compile_execution_plan(
                config, config_path=str(_EXAMPLE), cli_args=args
            )


@pytest.mark.parametrize("override, error", _ILLEGAL_OVERRIDES)
def test_plan_rejects_unsupported_jetspec_combinations(tmp_path, override, error):
    target = f"model.target_model_path={_fake_qwen_target(tmp_path)}"
    args = [target, override]
    with pytest.raises(error):
        config = load_config(config_path=str(_EXAMPLE), cli_args=args)
        compile_execution_plan(config, config_path=str(_EXAMPLE), cli_args=args)


def _trainer_args():
    return SimpleNamespace(
        draft_algorithm="jetspec", draft_model_config_obj=_tiny_config(), world_size=2,
        dflash_block_size=5, dflash_num_anchors=2, dflash_num_target_layers=1,
        dflash_loss_objective="decay", dflash_dpace_alpha=0.5,
        dflash_loss_decay_gamma=7.0, dflash_ce_loss_alpha=0.0, dflash_l1_loss_alpha=0.0,
        jetspec_distillation_temperature=1.0, jetspec_kl_chunk_rows=64,
        jetspec_loss_decay_gamma=0.0, anchor_sampling_scope="row",
        deterministic_anchor_sampling=False, max_seq_length=128, micro_batch_size=1,
        draft_accumulation_steps=1, max_grad_norm=1.0, seed=42,
        draft_context_parallel_size=1, training_num_nodes=1, training_num_gpus_per_node=2,
        attention_backend="flex_attention", fsdp_strategy="REPLICATE",
        sp_ulysses_size=1, sp_ring_size=1,
    )


def test_trainer_constructor_propagates_jetspec_objective(
    monkeypatch,
):
    def base_init(self, args):
        self.args = args
        self.block_size = args.dflash_block_size
        self.num_anchors = args.dflash_num_anchors
        self.num_target_layers = args.dflash_num_target_layers
        self.loss_objective = args.dflash_loss_objective
        self.dpace_alpha = args.dflash_dpace_alpha
        self.loss_decay_gamma = args.dflash_loss_decay_gamma
        self.ce_loss_alpha = args.dflash_ce_loss_alpha
        self.l1_loss_alpha = args.dflash_l1_loss_alpha
        self._anchor_slot_offset = 1

    monkeypatch.setattr(DFlashTrainer, "__init__", base_init)
    args = _trainer_args()
    args.jetspec_distillation_temperature = 0.5
    args.jetspec_kl_chunk_rows = 32
    args.jetspec_loss_decay_gamma = 0.25
    trainer = JetSpecTrainer(args)

    assert trainer.loss_decay_gamma == 0.25
    assert trainer.loss_decay_gamma != args.dflash_loss_decay_gamma
    wrapper = trainer._build_training_wrapper(JetSpecDraftModel(_tiny_config()))
    assert isinstance(wrapper, JetSpecModel)
    assert wrapper.distillation_temperature == 0.5
    assert wrapper.kl_chunk_rows == 32
    assert wrapper.loss_decay_gamma == 0.25
    assert wrapper.prediction_start == 1
    assert trainer._anchor_slot_offset == 1

    objective = _logical_training_topology(args, 2)["training_objective"]
    assert objective["distillation_temperature"] == wrapper.distillation_temperature
    assert objective["kl_chunk_rows"] == wrapper.kl_chunk_rows
    assert objective["loss_decay_gamma"] == wrapper.loss_decay_gamma

    trainer.block_size = 4
    with pytest.raises(ValueError, match="block_size"):
        trainer._build_draft_model(_tiny_config())
    trainer.block_size = 5
    trainer.num_target_layers = 2
    with pytest.raises(ValueError, match="num_target_layers"):
        trainer._build_draft_model(_tiny_config())


@pytest.mark.parametrize(
    "missing",
    [
        "empty", "distillation_temperature", "kl_chunk_rows", "loss_decay_gamma",
        "is_causal",
    ],
)
def test_exact_resume_rejects_incomplete_jetspec_topology(missing):
    args = _trainer_args()
    topology = json.loads(json.dumps(_logical_training_topology(args, 2)))
    if missing == "empty":
        topology = {}
    elif missing in {
        "distillation_temperature", "kl_chunk_rows", "loss_decay_gamma",
    }:
        del topology["training_objective"][missing]
    else:
        del topology["draft_model_semantics"][missing]
    with pytest.raises(RuntimeError, match="complete JetSpec training_topology"):
        _validate_exact_resume_topology(
            args, {"world_size": 2, "training_topology": topology}
        )


def test_exact_resume_requires_topology_for_jetspec():
    args = _trainer_args()
    with pytest.raises(RuntimeError, match="training_topology for JetSpec"):
        _validate_exact_resume_topology(args, {"world_size": 2})


def test_checkpoint_topology_pins_jetspec_objective():
    args = _trainer_args()
    current = _logical_training_topology(args, 2)
    objective = current["training_objective"]
    assert objective["loss"] == "temperature_scaled_forward_kl"
    assert objective["distillation_temperature"] == 1.0
    assert objective["kl_chunk_rows"] == 64
    assert objective["block_size"] == 5
    assert objective["loss_decay_gamma"] == 0.0
    assert current["draft_model_semantics"]["is_causal"] is True
    assert current["draft_model_semantics"]["block_size"] == 5

    metadata = {"world_size": 2, "training_topology": json.loads(json.dumps(current))}
    _validate_exact_resume_topology(args, metadata)
    for key, value in (
        ("distillation_temperature", 0.5),
        ("kl_chunk_rows", 32),
        ("loss_decay_gamma", 1.0),
    ):
        changed = json.loads(json.dumps(metadata))
        changed["training_topology"]["training_objective"][key] = value
        with pytest.raises(RuntimeError):
            _validate_exact_resume_topology(args, changed)
