# Copyright 2026 Ant Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Guard numerical reports against hidden normalization/near-zero errors."""

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
