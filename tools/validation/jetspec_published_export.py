# Copyright 2026 Ant Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Export the published JetSpec head through the ASpec checkpoint exporter.

The published ``JetSpec/jetspec-qwen3-8b`` repository is an HF draft-head asset,
not an ASpec FSDP training checkpoint. This validation harness loads its tensors
into an ASpec draft model, writes the minimal distributed-checkpoint layout the
exporter expects, and then runs the normal ``convert_checkpoint`` path.

This is a reproducibility tool for the bounded published-checkpoint comparison,
not a training or serving entry point.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import torch

from aspec.config.draft_algorithm import draft_model_semantics
from aspec.models.draft import AutoDraftModelConfig, AutoEagle3DraftModel
from aspec.training.checkpoint import load_initial_draft_weights
from tools import convert_to_hf


def _validate_output_path(output: Path, checkpoint: Path, target_model: str | None) -> None:
    output = output.expanduser().resolve()
    protected = [("published checkpoint", checkpoint)]
    if target_model is not None:
        target = Path(target_model).expanduser()
        if target.exists():
            protected.append(("Target model", target.resolve()))
    for label, path in protected:
        if output == path or output.is_relative_to(path) or path.is_relative_to(output):
            raise ValueError(
                f"--output must not overlap the {label} directory: {path}"
            )


def export_published_jetspec(
    checkpoint: Path,
    output: Path,
    *,
    target_model: str | None = None,
    artifact_format: str = "sglang",
    force: bool = False,
) -> dict:
    """Load a published JetSpec head and export it through ASpec's converter."""
    if target_model is None:
        raise ValueError(
            "published JetSpec checkpoints omit embed_tokens; pass --target-model "
            "so the export borrows the matching Target embedding"
        )
    checkpoint = checkpoint.expanduser().resolve()
    output = output.expanduser().resolve()
    _validate_output_path(output, checkpoint, target_model)
    config_path = checkpoint / "config.json"
    weights_path = checkpoint / "model.safetensors"
    if not config_path.is_file() or not weights_path.is_file():
        raise FileNotFoundError(
            "published checkpoint must contain config.json and model.safetensors"
        )

    raw = json.loads(config_path.read_text(encoding="utf-8"))
    config = AutoDraftModelConfig.from_dict(raw)
    model = AutoEagle3DraftModel.from_config(config, torch_dtype=torch.bfloat16)
    load_initial_draft_weights(model, str(checkpoint), allow_target_embedding=True)
    state = {
        "model_state.model.draft_model." + key: value.detach().clone()
        for key, value in model.state_dict().items()
        if key != "embed_tokens.weight"
    }

    with tempfile.TemporaryDirectory(prefix="aspec-jetspec-published-") as directory:
        source = Path(directory) / "iter_0000001"
        (source / "model").mkdir(parents=True)
        (source / "model/.metadata").write_bytes(b"validation-stub")
        (source / "draft_config.json").write_text(json.dumps(raw))
        (source / "meta.json").write_text(json.dumps({
            "training_topology": {
                "draft_algorithm": "jetspec",
                "training_objective": {"block_size": config.block_size},
                "draft_model_semantics": draft_model_semantics(config),
            }
        }))

        original = convert_to_hf._load_fsdp_state_dict
        convert_to_hf._load_fsdp_state_dict = lambda _: state
        try:
            return convert_to_hf.convert_checkpoint(
                input_dir=source,
                output_dir=output,
                target_model_path=target_model,
                artifact_format=artifact_format,
                force=force,
            )
        finally:
            convert_to_hf._load_fsdp_state_dict = original


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--target-model", default=None,
                        help="optional Target path for a borrowed embedding")
    parser.add_argument("--format", choices=("sglang", "aspec"), default="sglang",
                        dest="artifact_format")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    manifest = export_published_jetspec(
        args.checkpoint,
        args.output,
        target_model=args.target_model,
        artifact_format=args.artifact_format,
        force=args.force,
    )
    print(json.dumps({
        "algorithm": manifest["algorithm"],
        "storage_format": manifest["storage_format"],
        "tensor_count": manifest["tensor_count"],
        "output": str(args.output.resolve()),
    }, indent=2))


if __name__ == "__main__":
    main()
