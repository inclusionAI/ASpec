# Copyright 2026 Ant Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Report per-head held-out loss and token accuracy for an exported Medusa Draft."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from aspec.data.utils import unpack_loss_mask
from aspec.draft_artifact import load_native_draft
from aspec.models.medusa import MedusaModel


def _record_batch(record):
    ids = record["input_ids"].reshape(-1).long()
    hidden = record["hidden_states"]
    hidden = hidden.reshape(-1, hidden.shape[-1])
    packed = record.get("packed_loss_mask")
    if packed is None:
        raise ValueError("Held-out Medusa records require packed_loss_mask")
    mask = packed.reshape(-1) if isinstance(packed, torch.Tensor) else unpack_loss_mask(packed)
    if ids.numel() == 0 or hidden.shape[0] != ids.numel() or mask.numel() != ids.numel():
        raise ValueError("Held-out record tensors disagree on sequence length")
    return ids.unsqueeze(0), hidden.unsqueeze(0), mask.unsqueeze(0)


def evaluate_records(draft, records, *, device="cpu"):
    """Aggregate each head's token-weighted CE and accuracy over held-out records."""
    model = MedusaModel(draft).to(device).eval()
    dtype = next(model.parameters()).dtype
    heads = draft.medusa_num_heads
    loss_sums = torch.zeros(heads, dtype=torch.float64)
    correct = torch.zeros(heads, dtype=torch.float64)
    counts = torch.zeros(heads, dtype=torch.long)
    samples = 0
    with torch.no_grad():
        for record in records:
            ids, hidden, mask = _record_batch(record)
            ids, hidden, mask = ids.to(device), hidden.to(device, dtype), mask.to(device)
            losses, _, accuracy, head_counts, _ = model(
                ids, hidden, mask, torch.ones_like(ids))
            for head in range(heads):
                tokens = int(head_counts[head])
                if tokens == 0:
                    continue
                loss_sums[head] += float(losses[head]) * tokens
                correct[head] += float(accuracy[head]) * tokens
                counts[head] += tokens
            samples += 1
    if samples == 0:
        raise ValueError("No held-out records were evaluated")
    per_head = []
    for head in range(heads):
        tokens = int(counts[head])
        per_head.append({
            "head": head,
            "prediction_offset": head + 2,
            "tokens": tokens,
            "loss": float(loss_sums[head]) / tokens if tokens else None,
            "accuracy": float(correct[head]) / tokens if tokens else None,
        })
    measured = [item for item in per_head if item["tokens"]]
    return {
        "samples": samples,
        "per_head": per_head,
        "aggregate": {
            "sum_of_head_mean_ce": (
                sum(item["loss"] for item in measured) if measured else None),
            "measured_heads": len(measured),
        },
    }


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--draft", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--split", default="eval")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main():
    args = _parse_args()
    from aspec.offline.dataset import OfflineDataset

    dataset = OfflineDataset(args.dataset)
    rows = dataset.rows(args.split)
    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("--limit must be positive")
        rows = rows[: args.limit]
    draft = load_native_draft(args.draft, device=args.device)
    report = evaluate_records(
        draft, (dataset.load(row) for row in rows), device=args.device)
    report.update({
        "draft_manifest_sha256": hashlib.sha256(
            (args.draft / "export_manifest.json").read_bytes()).hexdigest(),
        "dataset_identity": dataset.identity,
        "split": args.split,
        "medusa_num_heads": draft.medusa_num_heads,
        "medusa_num_layers": draft.medusa_num_layers,
    })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
