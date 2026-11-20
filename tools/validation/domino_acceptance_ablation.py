# Copyright 2026 Ant Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Paired real-decode ablation: corrected Domino versus its backbone-only branch.

The backbone-only artifact keeps the trained DFlash weights and zeroes only the
correction head's output projection, so SGLang still loads the same asset schema
and produces the backbone logits exactly. The two artifacts must be evaluated on
the same prompts and decoding settings; the summary rejects mismatched evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

_CORRECTION_OUTPUT = "embed_proj.2."
_ABLATION_KIND = "backbone_only_zero_correction_output"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_backbone_only(corrected: Path, output: Path, *, force: bool = False) -> dict:
    """Write a SGLang asset whose correction delta is exactly zero."""
    corrected = Path(corrected).resolve()
    manifest_path = corrected / "export_manifest.json"
    config_path = corrected / "config.json"
    model_path = corrected / "model.safetensors"
    for path in (manifest_path, config_path, model_path):
        if not path.is_file():
            raise FileNotFoundError(f"missing SGLang Domino asset file: {path}")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("algorithm") != "domino" or manifest.get("storage_format") != "sglang":
        raise ValueError("requires a SGLang-format Domino export")

    output = Path(output).resolve()
    if (output == corrected or corrected.is_relative_to(output)
            or output.is_relative_to(corrected)):
        raise ValueError("backbone output must not overlap the corrected asset")
    if output.exists() and any(output.iterdir()):
        if not force:
            raise FileExistsError(f"output directory is not empty: {output}")
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)

    state = load_file(str(model_path))
    keys = sorted(key for key in state if key.startswith(_CORRECTION_OUTPUT))
    if not keys:
        raise ValueError("Domino asset has no correction output projection")
    for key in keys:
        state[key] = torch.zeros_like(state[key])
    save_file(state, str(output / "model.safetensors"), metadata={"format": "pt"})
    shutil.copy2(config_path, output / "config.json")

    ablation = {
        "kind": _ABLATION_KIND,
        "source_manifest_sha256": _sha256(manifest_path),
        "zeroed_tensors": keys,
        "correction_delta": "exactly zero; backbone logits unchanged",
    }
    manifest["model_sha256"] = _sha256(output / "model.safetensors")
    manifest["ablation"] = ablation
    (output / "export_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    return manifest


_PAIRED_SETTINGS = (
    "samples", "max_new_tokens", "repeats", "tp", "domino_candidate_pool_size",
    "disable_thinking", "trust_remote_code", "deterministic_inference",
)


def _paired_settings(record: dict) -> dict:
    settings = record.get("settings") or {}
    return {key: settings.get(key) for key in _PAIRED_SETTINGS}


def summarize(corrected: dict, backbone: dict) -> dict:
    """Combine two evaluate records after checking they describe the same prompts."""
    for name, record in (("corrected", corrected), ("backbone", backbone)):
        if record.get("schema") != "aspec.decode-evaluation/v1":
            raise ValueError(f"{name} is not an ASpec decode evaluation")
        if record.get("algorithm") != "DFLASH":
            raise ValueError(f"{name} did not run DFLASH verification")
    for key in ("data_sha256", "prompt_tokens_sha256"):
        if corrected.get(key) != backbone.get(key):
            raise ValueError(f"{key} differs between corrected and backbone runs")
    if corrected.get("target") != backbone.get("target"):
        raise ValueError("target differs between corrected and backbone runs")
    if _paired_settings(corrected) != _paired_settings(backbone):
        raise ValueError("decoding settings differ between corrected and backbone runs")
    corrected_summary = corrected["runs"]["speculative"]["summary"]
    backbone_summary = backbone["runs"]["speculative"]["summary"]
    corrected_length = corrected_summary.get("acceptance_length")
    backbone_length = backbone_summary.get("acceptance_length")
    if corrected_length is None or backbone_length is None:
        raise ValueError("both runs require measured speculative acceptance")
    return {
        "schema": "aspec.domino-acceptance-ablation/v1",
        "data_sha256": corrected["data_sha256"],
        "prompt_tokens_sha256": corrected["prompt_tokens_sha256"],
        "corrected": {
            "acceptance_length": corrected_length,
            "verify_count": corrected_summary["verify_count"],
            "greedy_equal": corrected["greedy_equal"],
            "target_repeat_stable": corrected["target_repeat_stable"],
            "draft_repeat_stable": corrected["draft_repeat_stable"],
        },
        "backbone": {
            "acceptance_length": backbone_length,
            "verify_count": backbone_summary["verify_count"],
            "greedy_equal": backbone["greedy_equal"],
            "target_repeat_stable": backbone["target_repeat_stable"],
            "draft_repeat_stable": backbone["draft_repeat_stable"],
        },
        "corrected_minus_backbone": corrected_length - backbone_length,
        "acceptance_convention": corrected_summary["acceptance_convention"],
        "claim": "bounded paired real decode; not throughput or convergence evidence",
    }


def _evaluate(draft: Path, output: Path, args) -> dict:
    if output.exists():
        raise FileExistsError(f"evaluation output already exists: {output}")
    command = [
        sys.executable, "-m", "aspec.commands.evaluate_draft",
        "--target", args.target,
        "--draft", str(draft),
        "--data", args.data,
        "--output", str(output),
        "--samples", str(args.samples),
        "--max-new-tokens", str(args.max_new_tokens),
        "--repeats", str(args.repeats),
        "--tp", str(args.tp),
        "--domino-candidate-pool-size", str(args.candidate_pool_size),
    ]
    if args.trust_remote_code:
        command.append("--trust-remote-code")
    if args.disable_thinking:
        command.append("--disable-thinking")
    subprocess.run(command, check=True)
    return json.loads(output.read_text())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corrected", type=Path, required=True,
                        help="SGLang-format Domino export to ablate")
    parser.add_argument("--backbone", type=Path, required=True,
                        help="Output directory for the backbone-only artifact")
    parser.add_argument("--target", required=True)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output", type=Path, required=True,
                        help="Combined ablation summary JSON")
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--candidate-pool-size", type=int, default=0)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--disable-thinking", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if min(args.samples, args.max_new_tokens, args.repeats, args.tp) < 1:
        raise ValueError("use positive sample, token, repeat and TP budgets")
    if args.candidate_pool_size < 0:
        raise ValueError("candidate pool size must be non-negative")

    build_backbone_only(args.corrected, args.backbone, force=args.force)
    corrected_path = args.output.with_suffix(".corrected.json")
    backbone_path = args.output.with_suffix(".backbone.json")
    corrected = _evaluate(args.corrected, corrected_path, args)
    backbone = _evaluate(args.backbone, backbone_path, args)
    summary = summarize(corrected, backbone)
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
