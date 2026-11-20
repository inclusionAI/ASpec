# Copyright 2026 Ant Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Domino native export preserves correction-head semantics and strict weights."""

import json

import pytest
import torch

from aspec.draft_artifact import load_native_draft
from aspec.models.draft import AutoDraftModelConfig, AutoEagle3DraftModel
from aspec.models.draft.domino import official_domino_config
from aspec.training.checkpoint import load_initial_draft_weights
from tools import convert_to_hf


def _raw_config(**overrides):
    raw = {
        "architectures": ["DominoDraftModel"],
        "model_type": "domino",
        "projector_type": "domino",
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
        "emb_dim": 6,
        "gru_hidden_dim": 8,
        "shift_label": True,
        "pure_draft_prefix_len": 1,
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
        "draft_algorithm": "domino",
        "training_objective": {
            "block_size": raw["block_size"] if block_size is None else block_size,
            "loss": "mixed_corrected_backbone_ce",
            "loss_decay_gamma": 7.0,
            "lambda_base": 0.3,
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


def test_domino_native_export_roundtrips_every_tensor(tmp_path, monkeypatch):
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
    assert manifest["algorithm"] == "domino"
    assert manifest["training_contract"]["block_size"] == raw["block_size"]
    loader = manifest["draft_structure"]["loader_contract"]
    assert loader["proposal_entrypoint"] == "aspec.draft_artifact.DominoConsumer"
    assert loader["borrowed_tensors"] == ["Target output head"]
    exported = json.loads((output / "config.json").read_text())
    for key in (
        "architectures", "model_type", "projector_type", "block_size", "emb_dim",
        "gru_hidden_dim", "shift_label", "pure_draft_prefix_len", "target_layer_ids",
        "num_target_layers", "target_num_hidden_layers",
    ):
        assert exported[key] == raw[key]

    loaded = load_native_draft(output)
    assert set(loaded.state_dict()) == set(model.state_dict())
    for key, value in model.state_dict().items():
        assert value.dtype == loaded.state_dict()[key].dtype
        torch.testing.assert_close(value, loaded.state_dict()[key], atol=0, rtol=0)
    for key, value in model.named_buffers():
        actual = dict(loaded.named_buffers())[key]
        assert value.dtype == actual.dtype
        torch.testing.assert_close(value, actual, atol=0, rtol=0)

    warm = _model(raw)
    load_initial_draft_weights(warm, str(output))
    for key, value in model.state_dict().items():
        torch.testing.assert_close(warm.state_dict()[key], value, atol=0, rtol=0)


@pytest.mark.parametrize("field", ["shift_label", "pure_draft_prefix_len", "gru_hidden_dim"])
def test_domino_native_export_rejects_semantics_that_disagree_with_checkpoint(
    tmp_path, field
):
    raw = _raw_config()
    changed = dict(raw)
    if field == "shift_label":
        changed[field] = not raw[field]
        changed["pure_draft_prefix_len"] = 0
    elif field == "pure_draft_prefix_len":
        changed[field] = 0
    else:
        changed[field] = raw[field] + 2
    source = _prepare_source(tmp_path, raw, semantics=raw)
    with pytest.raises(ValueError, match="semantics conflict"):
        convert_to_hf.convert_checkpoint(
            input_dir=source,
            output_dir=tmp_path / "asset",
            config_path=_write_config(tmp_path, changed),
            artifact_format="aspec",
        )


def test_domino_native_export_rejects_block_size_that_disagrees_with_checkpoint(
    tmp_path, monkeypatch
):
    raw = _raw_config()
    _monkeypatch_checkpoint(monkeypatch, _model(raw))
    source = _prepare_source(tmp_path, raw, block_size=raw["block_size"] + 1)
    with pytest.raises(ValueError, match="block_size conflicts"):
        convert_to_hf.convert_checkpoint(
            input_dir=source, output_dir=tmp_path / "asset", artifact_format="aspec"
        )


def test_domino_serving_config_uses_the_published_layout():
    raw = _raw_config()
    exported = convert_to_hf._serving_config(raw)

    assert exported["architectures"] == ["DFlashDraftModel"]
    assert exported["model_type"] == "qwen3"
    assert exported["block_size"] == raw["block_size"]
    assert exported["num_target_layers"] == raw["target_num_hidden_layers"]
    assert exported["dflash_config"] == {
        "projector_type": "domino",
        "mask_token_id": raw["mask_token_id"],
        "shift_label": True,
        "target_layer_ids": raw["target_layer_ids"],
        "pure_draft_prefix_len": 1,
        "gru_hidden_dim": raw["gru_hidden_dim"],
        "emb_dim": raw["emb_dim"],
    }


@pytest.mark.parametrize("change, message", [
    ({"pure_draft_prefix_len": 0}, "pure_draft_prefix_len=1"),
    ({"shift_label": 1}, "boolean shift_label"),
])
def test_domino_serving_export_rejects_unsupported_layout(change, message):
    with pytest.raises((NotImplementedError, ValueError), match=message):
        convert_to_hf._serving_config({**_raw_config(), **change})


def test_domino_sglang_export_writes_published_config(tmp_path, monkeypatch):
    raw = _raw_config()
    model = _model(raw)
    _monkeypatch_checkpoint(monkeypatch, model)
    source = _prepare_source(tmp_path, raw)
    output = tmp_path / "serving"

    manifest = convert_to_hf.convert_checkpoint(
        input_dir=source, output_dir=output, artifact_format="sglang"
    )

    assert manifest["storage_format"] == "sglang"
    assert manifest["algorithm"] == "domino"
    assert manifest["training_contract"]["block_size"] == raw["block_size"]
    loader = manifest["draft_structure"]["loader_contract"]
    assert loader["runtime"] == "sglang"
    assert loader["speculative_algorithm"] == "DFLASH"
    exported = json.loads((output / "config.json").read_text())
    assert exported["architectures"] == ["DFlashDraftModel"]
    assert exported["model_type"] == "qwen3"
    assert exported["block_size"] == raw["block_size"]
    assert exported["num_target_layers"] == raw["target_num_hidden_layers"]
    assert exported["dflash_config"]["projector_type"] == "domino"
    assert exported["dflash_config"]["pure_draft_prefix_len"] == 1
    assert (output / "model.safetensors").is_file()


def _write_config(tmp_path, raw):
    path = tmp_path / "changed_draft_config.json"
    path.write_text(json.dumps(raw))
    return path


def test_native_export_accepts_the_published_domino_config(tmp_path, monkeypatch):
    flat = _raw_config()
    model = _model(flat)
    _monkeypatch_checkpoint(monkeypatch, model)
    published = official_domino_config(flat)
    source = _prepare_source(tmp_path, published)
    output = tmp_path / "asset"

    manifest = convert_to_hf.convert_checkpoint(
        input_dir=source, output_dir=output, artifact_format="aspec"
    )

    assert manifest["algorithm"] == "domino"
    assert manifest["training_contract"]["block_size"] == flat["block_size"]
    exported = json.loads((output / "config.json").read_text())
    assert exported["architectures"] == ["DominoDraftModel"]
    assert exported["model_type"] == "domino"
    assert exported["num_target_layers"] == flat["num_target_layers"]
    assert exported["target_num_hidden_layers"] == flat["target_num_hidden_layers"]
