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


def test_jetspec_has_no_serving_export():
    with pytest.raises(NotImplementedError, match="--format aspec"):
        convert_to_hf._serving_config(_raw_config())


def _write_config(tmp_path, raw):
    path = tmp_path / "changed_draft_config.json"
    path.write_text(json.dumps(raw))
    return path
