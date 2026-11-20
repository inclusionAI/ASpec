# Copyright 2026 Ant Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

import json
import sys

import pytest
import torch

from tools.validation import domino_anchor_protocol as protocol


def _samples():
    return [
        {"data_id": "a", "input_ids": torch.arange(40) % 31,
         "loss_mask": torch.tensor([0] * 5 + [1] * 30 + [0] * 5)},
        {"data_id": "b", "input_ids": torch.arange(23) % 31,
         "loss_mask": torch.tensor([0] * 3 + [1] * 18 + [0] * 2)},
    ]


def test_anchor_stream_is_reproducible_and_supervision_valid():
    samples = _samples()
    first = protocol.build_anchors(samples, num_anchors=4, block_size=5, seed=7)
    second = protocol.build_anchors(samples, num_anchors=4, block_size=5, seed=7)
    assert first == second
    protocol.verify_anchors(first, samples)
    assert first["schema_version"] == protocol._SCHEMA
    assert [row["data_id"] for row in first["records"]] == ["a", "b"]


def test_changed_anchor_stream_fails_replay_verification():
    samples = _samples()
    record = protocol.build_anchors(samples, num_anchors=2, block_size=5, seed=7)
    record["records"][0]["anchors"][0] += 1
    with pytest.raises(ValueError, match="anchor replay differs"):
        protocol.verify_anchors(record, samples)


def test_anchor_protocol_rejects_noncanonical_schema():
    samples = _samples()
    record = protocol.build_anchors(samples, num_anchors=2, block_size=5, seed=7)
    record["schema_version"] = "other"
    with pytest.raises(ValueError, match="not a Domino anchor protocol"):
        protocol.verify_anchors(record, samples)


def test_anchor_cli_roundtrip_and_source_identity(tmp_path, monkeypatch):
    source = tmp_path / "samples.jsonl"
    rows = [
        {"data_id": "a", "input_ids": list(range(40)), "loss_mask": [0] * 5 + [1] * 30 + [0] * 5},
        {"data_id": "b", "input_ids": list(range(23)), "loss_mask": [0] * 3 + [1] * 18 + [0] * 2},
    ]
    source.write_text("".join(json.dumps(row) + "\n" for row in rows))
    anchors = tmp_path / "anchors.json"
    common = ["--source", str(source), "--anchors", str(anchors),
              "--num-anchors", "4", "--block-size", "5", "--seed", "7"]
    monkeypatch.setattr(sys, "argv", ["domino_anchor_protocol", "build", *common])
    protocol.main()
    monkeypatch.setattr(sys, "argv", ["domino_anchor_protocol", "verify", *common])
    protocol.main()
    source.write_text(source.read_text() + "\n")
    with pytest.raises(ValueError, match="source changed"):
        protocol.main()


def test_anchor_loader_accepts_jsonl_packed_loss_mask(tmp_path):
    source = tmp_path / "samples.jsonl"
    source.write_text(json.dumps({
        "data_id": "a", "input_ids": [1, 2, 3, 4, 5], "packed_loss_mask": "2,3",
    }) + "\n")
    samples = protocol._load_samples(source)
    assert samples[0]["loss_mask"].tolist() == [0, 0, 1, 1, 1]
    record = protocol.build_anchors(samples, num_anchors=1, block_size=2, seed=1)
    protocol.verify_anchors(record, samples)
