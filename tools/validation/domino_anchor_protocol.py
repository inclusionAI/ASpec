# Copyright 2026 Ant Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Materialize and verify one canonical anchor stream for matched Domino runs.

ASpec and SpecForge use different anchor samplers. A comparison should consume
the same explicit positions instead of asking two RNGs to agree. This tool turns
tokenized samples into a hashed replay file and proves that regenerating it from
the same seed and supervision is bit-identical.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from aspec.data.utils import resolve_loss_mask
from aspec.models.ops.anchors import (
    sample_anchor_positions,
    validate_injected_anchors,
)

_SCHEMA = "aspec.domino-anchor-protocol/v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_samples(path: Path, limit: int | None = None) -> list[dict]:
    if path.suffix == ".pt":
        payload = torch.load(path, map_location="cpu", weights_only=False)
        rows = payload.get("prompts") if isinstance(payload, dict) else payload
        if not rows:
            raise ValueError("tokenized cache contains no prompts")
    else:
        rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    if limit is not None:
        rows = rows[:limit]
    samples = []
    for index, row in enumerate(rows):
        input_ids = row.get("input_ids")
        if input_ids is None:
            raise ValueError(f"sample {index} has no input_ids")
        input_ids = torch.as_tensor(input_ids).reshape(-1).long()
        normalized = dict(row)
        normalized["input_ids"] = input_ids
        if row.get("packed_loss_mask") is not None:
            mask = resolve_loss_mask(normalized)
        elif row.get("loss_mask") is not None:
            mask = torch.as_tensor(row["loss_mask"]).reshape(-1)
        else:
            mask = None
        if mask is None or mask.shape != input_ids.shape:
            raise ValueError(f"sample {index} has no aligned loss_mask")
        samples.append({
            "data_id": str(row.get("data_id") or row.get("id") or f"sample_{index}"),
            "input_ids": input_ids,
            "loss_mask": mask.long(),
        })
    if len({sample["data_id"] for sample in samples}) != len(samples):
        raise ValueError("duplicate data_id in anchor source")
    return samples


def build_anchors(samples, *, num_anchors: int, block_size: int, seed: int,
                  max_gap: int | None = None) -> dict:
    """Generate one deterministic anchor stream for the given samples."""
    if min(num_anchors, block_size) < 1:
        raise ValueError("num_anchors and block_size must be positive")
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    records = []
    for sample in samples:
        input_ids, loss_mask = sample["input_ids"], sample["loss_mask"]
        if input_ids.shape != loss_mask.shape or input_ids.ndim != 1:
            raise ValueError("input_ids and loss_mask must be aligned 1-D tensors")
        anchors, keep = sample_anchor_positions(
            int(input_ids.numel()), loss_mask.unsqueeze(0), int(num_anchors),
            int(block_size), torch.device("cpu"), max_gap=max_gap, generator=generator,
        )
        records.append({
            "data_id": sample["data_id"],
            "anchors": anchors[0].tolist(),
            "keep_mask": keep[0].tolist(),
        })
    return {
        "schema_version": _SCHEMA,
        "num_anchors": int(num_anchors),
        "block_size": int(block_size),
        "seed": int(seed),
        "max_gap": max_gap,
        "sample_count": len(records),
        "records": records,
    }


def _validate_record(record: dict) -> None:
    if record.get("schema_version") != _SCHEMA:
        raise ValueError("not a Domino anchor protocol file")
    records = record.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("anchor protocol file has no records")
    if len({row["data_id"] for row in records}) != len(records):
        raise ValueError("anchor protocol file has duplicate data_id")


def verify_anchors(record: dict, samples) -> None:
    """Recompute the stream and validate every kept anchor against its supervision."""
    _validate_record(record)
    expected = build_anchors(
        samples,
        num_anchors=record["num_anchors"],
        block_size=record["block_size"],
        seed=record["seed"],
        max_gap=record.get("max_gap"),
    )
    if len(expected["records"]) != len(record["records"]):
        raise ValueError("anchor protocol sample count changed")
    by_id = {sample["data_id"]: sample for sample in samples}
    for actual, replay in zip(record["records"], expected["records"], strict=True):
        if actual["data_id"] != replay["data_id"]:
            raise ValueError("anchor protocol sample order changed")
        if actual["anchors"] != replay["anchors"] or actual["keep_mask"] != replay["keep_mask"]:
            raise ValueError(f"anchor replay differs for {actual['data_id']}")
        sample = by_id[actual["data_id"]]
        validate_injected_anchors(
            torch.tensor([actual["anchors"]], dtype=torch.long),
            torch.tensor([actual["keep_mask"]], dtype=torch.bool),
            loss_mask=sample["loss_mask"].unsqueeze(0),
            num_anchors=record["num_anchors"],
            block_size=record["block_size"],
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("build", "verify"))
    parser.add_argument("--source", type=Path, required=True,
                        help="tokenized .pt cache or JSONL with input_ids/loss_mask")
    parser.add_argument("--anchors", type=Path, required=True)
    parser.add_argument("--num-anchors", type=int, default=128)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-gap", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    samples = _load_samples(args.source, args.limit)
    if args.action == "build":
        if args.anchors.exists():
            raise FileExistsError(args.anchors)
        record = build_anchors(
            samples,
            num_anchors=args.num_anchors,
            block_size=args.block_size,
            seed=args.seed,
            max_gap=args.max_gap,
        )
        record["source_sha256"] = _sha256(args.source)
        args.anchors.write_text(json.dumps(record, sort_keys=True) + "\n")
    else:
        record = json.loads(args.anchors.read_text())
        if record.get("source_sha256") != _sha256(args.source):
            raise ValueError("anchor source changed; refusing to verify a different dataset")
        verify_anchors(record, samples)
    print(f"PASS {args.action} {len(samples)} samples -> {args.anchors}")


if __name__ == "__main__":
    main()
