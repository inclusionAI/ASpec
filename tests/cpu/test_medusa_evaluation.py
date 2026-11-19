# Copyright 2026 Ant Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Per-head held-out evaluation for exported Medusa heads."""

import math

import pytest
import torch
from torch.nn import functional as F

from aspec.commands.evaluate_medusa import _record_batch, evaluate_records
from aspec.models.draft import AutoDraftModelConfig, AutoEagle3DraftModel


def _draft():
    config = AutoDraftModelConfig.from_dict({
        "architectures": ["MedusaDraftModel"], "model_type": "aspec_medusa",
        "hidden_size": 8, "vocab_size": 12, "target_hidden_size": 8,
        "target_num_hidden_layers": 4, "medusa_num_heads": 2, "medusa_num_layers": 1,
    })
    torch.manual_seed(1)
    return AutoEagle3DraftModel.from_config(config, torch_dtype=torch.float32)


def _record(ids, hidden, packed):
    return {"input_ids": ids, "hidden_states": hidden, "packed_loss_mask": packed}


def _expected(draft, ids, hidden, mask):
    rows = []
    for head in range(draft.medusa_num_heads):
        offset = head + 2
        keep = mask[offset:].bool()
        if not keep.any():
            rows.append((0, None, None))
            continue
        logits = draft.medusa_head[head](hidden[:-offset].float())[keep]
        targets = ids[offset:][keep]
        rows.append((
            int(keep.sum()),
            F.cross_entropy(logits, targets, reduction="mean").item(),
            (logits.argmax(-1) == targets).float().mean().item(),
        ))
    return rows


def _assert_report(report, expected_rows):
    assert report["samples"] == len(expected_rows)
    measured = []
    for item, rows in zip(report["per_head"], zip(*expected_rows), strict=True):
        tokens = sum(row[0] for row in rows)
        assert item["tokens"] == tokens
        if tokens == 0:
            assert item["loss"] is None and item["accuracy"] is None
            continue
        loss = sum(count * value for count, value, _ in rows) / tokens
        accuracy = sum(count * value for count, _, value in rows) / tokens
        assert math.isclose(item["loss"], loss, rel_tol=1e-6, abs_tol=1e-7)
        assert math.isclose(item["accuracy"], accuracy, rel_tol=1e-6, abs_tol=1e-7)
        measured.append(item["loss"])
    assert math.isclose(
        report["aggregate"]["sum_of_head_mean_ce"], sum(measured), rel_tol=1e-6)


def test_per_head_metrics_are_token_weighted_across_records():
    draft = _draft()
    torch.manual_seed(2)
    ids_a = torch.randint(0, 12, (7,))
    hidden_a = torch.randn(7, 8)
    mask_a = torch.tensor([0, 1, 1, 1, 1, 0, 1])
    ids_b = torch.randint(0, 12, (6,))
    hidden_b = torch.randn(6, 8)
    records = [
        _record(ids_a, hidden_a, mask_a),
        _record(ids_b, hidden_b, [1, 4, 1]),  # packed: 0,1,1,1,1,0
    ]
    report = evaluate_records(draft, records)
    _assert_report(report, [
        _expected(draft, ids_a, hidden_a, mask_a),
        _expected(draft, ids_b, hidden_b, torch.tensor([0, 1, 1, 1, 1, 0])),
    ])


def test_record_validation_rejects_missing_mask_and_length_drift():
    with pytest.raises(ValueError, match="packed_loss_mask"):
        _record_batch({"input_ids": torch.arange(4), "hidden_states": torch.randn(4, 8)})
    with pytest.raises(ValueError, match="sequence length"):
        _record_batch({
            "input_ids": torch.arange(4), "hidden_states": torch.randn(5, 8),
            "packed_loss_mask": torch.ones(4),
        })
