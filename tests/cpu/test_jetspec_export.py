# Copyright 2026 Ant Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""JetSpec native export preserves causal semantics and strict weights."""

import json

import pytest
import torch

from aspec.draft_artifact import load_native_draft, verify_native_export
from aspec.models.draft import AutoDraftModelConfig, AutoEagle3DraftModel
from aspec.training.checkpoint import load_initial_draft_weights
from tools import convert_to_hf


def _raw_config(**overrides):
    raw = {
        "architectures": ["JetSpecDraftModel"],
        "model_type": "jetspec",
        "hidden_size": 16,
        "intermediate_size": 32,
        "num_hidden_layers": 1,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 8,
        "vocab_size": 32,
        "target_hidden_size": 16,
        "num_target_layers": 2,
        "target_num_hidden_layers": 6,
        "target_layer_ids": [1, 4],
        "max_position_embeddings": 128,
        "mask_token_id": 31,
        "block_size": 5,
        "is_causal": True,
    }
    raw.update(overrides)
    return raw


def _prepare_source(tmp_path, raw, *, block_size=None, semantics=None):
    start = "iter_0000002"
    source = tmp_path / start
    (source / "model").mkdir(parents=True)
    (source / "model/.metadata").write_bytes(b"stub")
    (source / "draft_config.json").write_text(json.dumps(raw))
    (source / "meta.json").write_text(json.dumps({"training_topology": {
        "draft_algorithm": "jetspec",
        "training_objective": {
            "block_size": raw["block_size"] if block_size is None else block_size,
            "loss": "temperature_scaled_forward_kl",
            "distillation_temperature": 1.0,
            "kl_chunk_rows": 64,
            "loss_decay_gamma": 0.0,
        },
        "draft_model_semantics": semantics or raw,
    }}))
    return source


def _model(raw, dtype=torch.bfloat16):
    config = AutoDraftModelConfig.from_dict(raw)
    return AutoEagle3DraftModel.from_config(config, torch_dtype=dtype)


def _monkeypatch_checkpoint(monkeypatch, model):
    wrapped = {
        "model_state.model.draft_model." + key: value.detach().clone()
        for key, value in model.state_dict().items()
    }
    monkeypatch.setattr(convert_to_hf, "_load_fsdp_state_dict", lambda _: wrapped)


def test_jetspec_native_export_roundtrips_every_tensor(tmp_path, monkeypatch):
    raw = _raw_config()
    model = _model(raw)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(0.0001234)
    _monkeypatch_checkpoint(monkeypatch, model)
    source = _prepare_source(tmp_path, raw)
    output = tmp_path / "asset"

    manifest = convert_to_hf.convert_checkpoint(
        input_dir=source, output_dir=output, artifact_format="aspec"
    )

    assert manifest["storage_format"] == "aspec_native"
    assert manifest["algorithm"] == "jetspec"
    assert manifest["training_contract"]["block_size"] == raw["block_size"]
    loader = manifest["draft_structure"]["loader_contract"]
    assert loader["proposal_entrypoint"] == "aspec.draft_artifact.JetSpecConsumer"
    assert loader["borrowed_tensors"] == ["Target output head"]
    exported = json.loads((output / "config.json").read_text())
    for key in (
        "architectures", "model_type", "block_size", "is_causal",
        "target_layer_ids", "num_target_layers", "target_num_hidden_layers",
    ):
        assert exported[key] == raw[key]

    verify_native_export(output, expected=model.state_dict())
    loaded = load_native_draft(output)
    assert set(loaded.state_dict()) == set(model.state_dict())
    for key, value in model.state_dict().items():
        assert value.dtype == loaded.state_dict()[key].dtype
        torch.testing.assert_close(value, loaded.state_dict()[key], atol=0, rtol=0)

    warm = _model(raw)
    load_initial_draft_weights(warm, str(output))
    for key, value in model.state_dict().items():
        torch.testing.assert_close(warm.state_dict()[key], value, atol=0, rtol=0)



def test_jetspec_native_export_accepts_published_layout_config(tmp_path, monkeypatch):
    from aspec.models.draft.jetspec import official_jetspec_config

    raw = _raw_config()
    model = _model(raw)
    _monkeypatch_checkpoint(monkeypatch, model)
    source = _prepare_source(tmp_path, raw)
    published = official_jetspec_config(raw)
    # The published config carries these defaults explicitly.
    published["hidden_act"] = "silu"
    published["rope_scaling"] = None
    config_path = tmp_path / "published_draft_config.json"
    config_path.write_text(json.dumps(published))

    manifest = convert_to_hf.convert_checkpoint(
        input_dir=source,
        output_dir=tmp_path / "asset",
        config_path=config_path,
        artifact_format="aspec",
    )

    assert manifest["algorithm"] == "jetspec"
    assert manifest["training_contract"]["block_size"] == raw["block_size"]
    assert manifest["draft_structure"]["loader_contract"]["proposal_entrypoint"] == (
        "aspec.draft_artifact.JetSpecConsumer"
    )

def test_jetspec_serving_export_rejects_non_silu_activation():
    with pytest.raises(NotImplementedError, match="SiLU"):
        convert_to_hf._serving_config(_raw_config(hidden_act="gelu"))


def test_published_export_harness_rejects_overlapping_output(tmp_path):
    from tools.validation.jetspec_published_export import _validate_output_path

    checkpoint = (tmp_path / "checkpoint").resolve()
    target = (tmp_path / "target").resolve()
    checkpoint.mkdir()
    target.mkdir()
    for output in (checkpoint, checkpoint / "asset", target, target / "asset"):
        with pytest.raises(ValueError, match="must not overlap"):
            _validate_output_path(output, checkpoint, str(target))


def test_jetspec_native_export_rejects_semantics_that_disagree_with_checkpoint(
    tmp_path,
):
    raw = _raw_config()
    changed = dict(raw, target_layer_ids=[2, 5])
    source = _prepare_source(tmp_path, raw, semantics=raw)
    with pytest.raises(ValueError, match="semantics conflict"):
        convert_to_hf.convert_checkpoint(
            input_dir=source,
            output_dir=tmp_path / "asset",
            config_path=_write_config(tmp_path, changed),
            artifact_format="aspec",
        )


def test_jetspec_native_export_rejects_block_size_that_disagrees_with_checkpoint(
    tmp_path, monkeypatch
):
    raw = _raw_config()
    _monkeypatch_checkpoint(monkeypatch, _model(raw))
    source = _prepare_source(tmp_path, raw, block_size=raw["block_size"] + 1)
    with pytest.raises(ValueError, match="block_size conflicts"):
        convert_to_hf.convert_checkpoint(
            input_dir=source, output_dir=tmp_path / "asset", artifact_format="aspec"
        )


def test_jetspec_serving_export_uses_causal_dflash_layout():
    config = convert_to_hf._serving_config(_raw_config())

    assert config["architectures"] == ["DFlashDraftModel"]
    assert config["model_type"] == "qwen3"
    assert config["is_causal"] is True
    assert config["block_size"] == 5
    assert config["num_target_layers"] == 6
    assert config["dflash_config"]["causal_head"] is True
    assert config["dflash_config"]["target_layer_ids"] == [1, 4]
    assert config["dflash_config"]["mask_token_id"] == 31


def test_jetspec_sglang_export_manifest_uses_dflash_runtime(tmp_path, monkeypatch):
    raw = _raw_config()
    _monkeypatch_checkpoint(monkeypatch, _model(raw))
    source = _prepare_source(tmp_path, raw)
    output = tmp_path / "sglang-asset"

    manifest = convert_to_hf.convert_checkpoint(
        input_dir=source, output_dir=output, artifact_format="sglang"
    )

    assert manifest["storage_format"] == "sglang"
    assert manifest["algorithm"] == "jetspec"
    assert manifest["draft_structure"]["loader_contract"]["speculative_algorithm"] == "DFLASH"
    exported = json.loads((output / "config.json").read_text())
    assert exported["architectures"] == ["DFlashDraftModel"]
    assert exported["is_causal"] is True
    assert exported["dflash_config"]["causal_head"] is True


def _write_config(tmp_path, raw):
    path = tmp_path / "changed_draft_config.json"
    path.write_text(json.dumps(raw))
    return path
