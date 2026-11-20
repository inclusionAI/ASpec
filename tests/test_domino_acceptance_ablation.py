# Copyright 2026 Ant Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json

import pytest
import torch
from safetensors.torch import load_file, save_file

from tools.validation import domino_acceptance_ablation as ablation


def _write_asset(root, *, algorithm="domino", storage_format="sglang"):
    root.mkdir(parents=True)
    tensors = {
        "fc.weight": torch.ones(4, 4),
        "embed_proj.0.weight": torch.ones(4, 4),
        "embed_proj.2.weight": torch.ones(8, 4),
    }
    save_file(tensors, str(root / "model.safetensors"), metadata={"format": "pt"})
    (root / "config.json").write_text(json.dumps({"model_type": "qwen3"}) + "\n")
    manifest = {
        "schema_version": "aspec.serving-export/v1",
        "storage_format": storage_format,
        "algorithm": algorithm,
        "model_sha256": ablation._sha256(root / "model.safetensors"),
        "config_sha256": hashlib.sha256((root / "config.json").read_bytes()).hexdigest(),
    }
    (root / "export_manifest.json").write_text(json.dumps(manifest) + "\n")
    return tensors


def test_backbone_ablation_zeroes_only_the_correction_output(tmp_path):
    corrected = tmp_path / "corrected"
    original = _write_asset(corrected)
    output = tmp_path / "backbone"

    manifest = ablation.build_backbone_only(corrected, output)

    actual = load_file(str(output / "model.safetensors"))
    assert torch.count_nonzero(actual["embed_proj.2.weight"]) == 0
    torch.testing.assert_close(actual["fc.weight"], original["fc.weight"])
    torch.testing.assert_close(actual["embed_proj.0.weight"], original["embed_proj.0.weight"])
    assert manifest["ablation"]["kind"] == ablation._ABLATION_KIND
    assert manifest["ablation"]["zeroed_tensors"] == ["embed_proj.2.weight"]
    assert manifest["model_sha256"] == ablation._sha256(output / "model.safetensors")
    assert (output / "config.json").read_bytes() == (corrected / "config.json").read_bytes()


@pytest.mark.parametrize("algorithm, storage_format", [
    ("dflash", "sglang"),
    ("domino", "aspec_native"),
])
def test_backbone_ablation_requires_a_sglang_domino_asset(tmp_path, algorithm, storage_format):
    corrected = tmp_path / "corrected"
    _write_asset(corrected, algorithm=algorithm, storage_format=storage_format)
    with pytest.raises(ValueError, match="SGLang-format Domino"):
        ablation.build_backbone_only(corrected, tmp_path / "backbone")


def test_backbone_ablation_rejects_overlapping_output(tmp_path):
    corrected = tmp_path / "assets" / "corrected"
    _write_asset(corrected)
    for output in (corrected.parent, corrected / "backbone"):
        with pytest.raises(ValueError, match="must not overlap"):
            ablation.build_backbone_only(corrected, output, force=True)


def _record(length, *, data="d", prompt="p", verify=10, target="Qwen3-8B", settings=None):
    summary = {
        "acceptance_length": length,
        "verify_count": verify,
        "acceptance_convention": "SGLang completion_tokens / spec_verify_ct; includes target tokens",
    }
    return {
        "schema": "aspec.decode-evaluation/v1",
        "algorithm": "DFLASH",
        "data_sha256": data,
        "prompt_tokens_sha256": prompt,
        "target": target,
        "settings": settings or {
            "samples": 16, "max_new_tokens": 32, "repeats": 2, "tp": 1,
            "domino_candidate_pool_size": 0, "disable_thinking": False,
            "trust_remote_code": False, "deterministic_inference": False,
        },
        "greedy_equal": True,
        "target_repeat_stable": True,
        "draft_repeat_stable": True,
        "runs": {"speculative": {"summary": summary}},
    }


def test_ablation_summary_requires_matching_prompts_and_reports_delta():
    result = ablation.summarize(_record(2.4), _record(1.6))
    assert result["corrected"]["acceptance_length"] == 2.4
    assert result["backbone"]["acceptance_length"] == 1.6
    assert result["corrected_minus_backbone"] == pytest.approx(0.8)
    with pytest.raises(ValueError, match="prompt_tokens_sha256"):
        ablation.summarize(_record(2.4), _record(1.6, prompt="other"))
    with pytest.raises(ValueError, match="target differs"):
        ablation.summarize(_record(2.4), _record(1.6, target="other"))
    with pytest.raises(ValueError, match="decoding settings differ"):
        ablation.summarize(_record(2.4), _record(1.6, settings={"samples": 8}))
