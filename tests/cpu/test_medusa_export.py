# Copyright 2026 Ant Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Native Medusa export, strict reload and manifest contracts."""

import json

import pytest
import torch

from aspec.draft_artifact import load_native_draft, verify_native_export
from aspec.models.draft import AutoDraftModelConfig, AutoEagle3DraftModel
from tools import convert_to_hf

_RAW = {
    "architectures": ["MedusaDraftModel"], "model_type": "aspec_medusa",
    "hidden_size": 16, "vocab_size": 32, "target_hidden_size": 16,
    "target_num_hidden_layers": 4, "medusa_num_heads": 2, "medusa_num_layers": 1,
    "tie_word_embeddings": False,
}


def _checkpoint(tmp_path, monkeypatch):
    config = AutoDraftModelConfig.from_dict(_RAW)
    torch.manual_seed(0)
    source = AutoEagle3DraftModel.from_config(config, torch_dtype=torch.float32)
    checkpoint = tmp_path / "iter_0000002"
    (checkpoint / "model").mkdir(parents=True)
    (checkpoint / "model" / ".metadata").write_bytes(b"stub")
    (checkpoint / "draft_config.json").write_text(
        json.dumps(config.to_dict()), encoding="utf-8")
    wrapped = {
        f"model_state.model.draft_model.{key}": value.detach().clone()
        for key, value in source.state_dict().items()
    }
    monkeypatch.setattr(convert_to_hf, "_load_fsdp_state_dict", lambda _: wrapped)
    return source, checkpoint


def test_native_export_roundtrips_every_head(tmp_path, monkeypatch):
    source, checkpoint = _checkpoint(tmp_path, monkeypatch)
    output = tmp_path / "asset"
    manifest = convert_to_hf.convert_checkpoint(
        input_dir=checkpoint, output_dir=output, artifact_format="aspec", verify=True)

    assert manifest["algorithm"] == "medusa"
    assert manifest["tensor_count"] == len(source.state_dict())
    assert manifest["verified"]["tensors"] == len(source.state_dict())
    assert manifest["draft_structure"]["loader_contract"]["runtime"] == "aspec"
    assert "candidate-tree acceptance" in manifest["draft_structure"]["loader_contract"]["scope"]

    exported_config = json.loads((output / "config.json").read_text())
    assert exported_config["architectures"] == ["MedusaDraftModel"]
    assert exported_config["target_layer_ids"] == [3]
    assert exported_config["num_aux_hidden_states"] == 1
    assert exported_config["draft_vocab_size"] == 32

    imported = load_native_draft(output)
    assert len(imported.medusa_head) == 2
    for key, value in source.state_dict().items():
        assert torch.equal(imported.state_dict()[key], value)


def test_verify_detects_a_changed_tensor(tmp_path, monkeypatch):
    _, checkpoint = _checkpoint(tmp_path, monkeypatch)
    output = tmp_path / "asset"
    convert_to_hf.convert_checkpoint(
        input_dir=checkpoint, output_dir=output, artifact_format="aspec")

    from safetensors.torch import load_file, save_file

    tensors = load_file(str(output / "model.safetensors"))
    first = next(iter(tensors))
    tensors[first] = tensors[first] + 1
    save_file(tensors, str(output / "model.safetensors"))
    with pytest.raises(ValueError, match="checksum mismatch"):
        verify_native_export(output)


def test_verify_compares_reload_with_the_exported_tensors(tmp_path, monkeypatch):
    source, checkpoint = _checkpoint(tmp_path, monkeypatch)
    output = tmp_path / "asset"
    convert_to_hf.convert_checkpoint(
        input_dir=checkpoint, output_dir=output, artifact_format="aspec")

    from safetensors.torch import load_file, save_file

    tensors = load_file(str(output / "model.safetensors"))
    first = next(iter(tensors))
    tensors[first] = tensors[first] + 1
    save_file(tensors, str(output / "model.safetensors"))
    # Keep the artifact internally consistent so only the exported values differ.
    manifest_path = output / "export_manifest.json"
    recorded = json.loads(manifest_path.read_text())
    recorded["model_sha256"] = convert_to_hf._sha256_file(output / "model.safetensors")
    manifest_path.write_text(
        json.dumps(recorded, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="changed .* values"):
        verify_native_export(output, expected=source.state_dict())


def test_verify_requires_native_format_before_writing(tmp_path, monkeypatch):
    _, checkpoint = _checkpoint(tmp_path, monkeypatch)
    output = tmp_path / "serving"
    with pytest.raises(ValueError, match="--verify requires --format aspec"):
        convert_to_hf.convert_checkpoint(
            input_dir=checkpoint, output_dir=output, artifact_format="sglang", verify=True)
    assert not output.exists()


def test_export_verify_compares_with_the_tensors_it_wrote(tmp_path, monkeypatch):
    source, checkpoint = _checkpoint(tmp_path, monkeypatch)
    from aspec import draft_artifact

    captured = {}

    def fake_verify(path, *, expected=None):
        captured["expected"] = expected
        return {"method": "stub", "tensors": len(expected or {})}

    monkeypatch.setattr(draft_artifact, "verify_native_export", fake_verify)
    manifest = convert_to_hf.convert_checkpoint(
        input_dir=checkpoint, output_dir=tmp_path / "asset", artifact_format="aspec",
        verify=True)

    assert set(captured["expected"]) == set(source.state_dict())
    assert manifest["verified"] == {"method": "stub", "tensors": len(source.state_dict())}


def test_sglang_format_is_rejected_for_medusa(tmp_path, monkeypatch):
    _, checkpoint = _checkpoint(tmp_path, monkeypatch)
    with pytest.raises(NotImplementedError, match="format aspec"):
        convert_to_hf.convert_checkpoint(
            input_dir=checkpoint, output_dir=tmp_path / "serving", artifact_format="sglang")
