# Copyright 2026 Ant Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Guard numerical reports against hidden normalization/near-zero errors."""

import json
import math

import pytest
import torch

from tools.validation.megatron_torch_alignment import compare_vectors, scalar_diff, selected_steps


def test_scalar_rel_diff_preserves_sign_and_exposes_zero_floor():
    assert scalar_diff(9, 10)["signed_rel_diff"] == -.1
    assert scalar_diff(9, 10)["abs_rel_diff"] == .1
    row = scalar_diff(1e-14, 0)
    assert row["denominator_floored"]
    assert row["absolute_difference"] == 1e-14
    assert row["signed_rel_diff"] == .01
    with pytest.raises(ValueError, match="Nonfinite"):
        scalar_diff(float("nan"), 1)


def test_tensor_rel_diff_uses_logical_elements_and_reports_each_parameter():
    expected = {"large": torch.tensor([1., 2., 3.]), "small": torch.tensor([4.])}
    actual = {"large": torch.tensor([2., 2., 3.]), "small": torch.tensor([2.])}
    row = compare_vectors(actual, expected, device="cpu")
    assert row["relative_l2"] == pytest.approx(math.sqrt(5 / 30))
    assert row["relative_l1"] == pytest.approx(3 / 10)
    assert row["relative_max"] == .5
    assert row["mean_absolute"] == .75
    assert row["pointwise_rel_diff_mean"] == .375
    assert row["pointwise_rel_diff_quantiles"]["p50"] == .25
    assert row["pointwise_exceed_counts"]["0.1"] == 2
    assert row["different_elements"] == 2
    assert row["numel"] == 4
    assert {r["parameter"] for r in row["per_parameter"]} == set(expected)


def test_tensor_rel_diff_zero_reference_is_explicit_and_does_not_drop_elements():
    row = compare_vectors({"w": torch.tensor([0., 1e-12])},
                          {"w": torch.zeros(2)}, device="cpu")
    assert row["relative_l2"] is None
    assert row["zero_reference"]
    assert row["denominator_floored_count"] == 2
    assert row["pointwise_rel_diff_mean"] == pytest.approx(.5)
    equal = compare_vectors({"w": torch.zeros(2)}, {"w": torch.zeros(2)}, device="cpu")
    assert equal["relative_l2"] == 0
    assert equal["bitwise_equal"]
    with pytest.raises(ValueError, match="Nonfinite"):
        compare_vectors({"w": torch.tensor([float("inf")])},
                        {"w": torch.ones(1)}, device="cpu")


def test_declared_observation_steps_do_not_extend_the_trajectory():
    assert selected_steps("1,2,10,100", 100) == {1, 2, 10, 100}
    assert selected_steps("", 100) == set()
    with pytest.raises(ValueError, match="within"):
        selected_steps("101", 100)


def test_online_objective_pools_cp_supervision_and_checks_packed_rows(tmp_path):
    from tools.validation.megatron_packed_online import collect

    baseline, candidate = tmp_path / "torch", tmp_path / "megatron"
    baseline.mkdir()
    candidate.mkdir()
    for rank in range(2):
        rows = [
            {"step": 1, "documents": [13, 17], "features": {"input_ids": "same", "hidden_states": str(rank)},
             "work_ids": [f"doc-{2 * micro}", f"doc-{2 * micro + 1}"],
             "anchor_seed": micro, "numerator": 2 + rank, "denominator": 1 + 2 * rank}
            for micro in range(2)
        ]
        for root in (baseline, candidate):
            (root / f"recipe-packed-cp{rank}.json").write_text(json.dumps({"rows": rows}))
    expected = collect(baseline, 2, 1)
    assert expected["steps"] == [1.25]
    (baseline / "objectives.json").write_text(json.dumps(expected))
    assert collect(candidate, 2, 1, baseline)["rel_diff"] == [0.0]

    rank_one = candidate / "recipe-packed-cp1.json"
    rows[0]["features"]["hidden_states"] = "different"
    rank_one.write_text(json.dumps({"rows": rows}))
    with pytest.raises(ValueError, match="CP rank 1, step 1"):
        collect(candidate, 2, 1, baseline)

    rows[0]["features"]["hidden_states"] = "1"
    rows[0]["work_ids"].reverse()
    rank_one.write_text(json.dumps({"rows": rows}))
    with pytest.raises(ValueError, match="Context ranks consumed different packed rows"):
        collect(candidate, 2, 1)

    rank_zero = candidate / "recipe-packed-cp0.json"
    rank_zero_rows = json.loads(rank_zero.read_text())["rows"]
    rank_zero_rows[0]["work_ids"].reverse()
    rank_zero.write_text(json.dumps({"rows": rank_zero_rows}))
    with pytest.raises(ValueError, match="CP rank 0, step 1"):
        collect(candidate, 2, 1, baseline)

    rank_zero_rows[0]["documents"].pop()
    rank_zero.write_text(json.dumps({"rows": rank_zero_rows}))
    with pytest.raises(ValueError, match="two documents per packed row"):
        collect(candidate, 2, 1)


def test_online_run_requires_four_committed_documents_per_step(tmp_path):
    from tools.validation.megatron_packed_online import check_online_run

    checkpoint = tmp_path / "checkpoints" / "iter_0000001"
    checkpoint.mkdir(parents=True)
    (checkpoint.parent / "latest_checkpointed_iteration.txt").write_text("1")
    (checkpoint / "complete.json").write_text(json.dumps({"optimizer_step": 1, "committed_work_count": 3}))
    evidence = {
        "state": "completed",
        "observations": {
            "cleanup": {
                "work_lifecycle": {"state": "validated", "optimizer_steps_committed": 1, "committed_work": 3},
            },
            "training": {"feature_contract": {"transport": {
                "state": "validated", "protocol": "nvlink_intra", "consumer_endpoints": 1,
                "destination_reads": 3, "destination_releases": 3, "transfer_failures": 0,
            }}},
        },
    }
    (tmp_path / "evidence.json").write_text(json.dumps(evidence))
    with pytest.raises(ValueError, match="four committed documents per optimizer step"):
        check_online_run(tmp_path, 1, 1)
