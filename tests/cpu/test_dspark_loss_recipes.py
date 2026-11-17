# Copyright 2026 Ant Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""DSpark loss contracts for the Red Hat Speculators b9c51bc training objective."""

import copy
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from aspec.config import load_config
from aspec.config.train_config import config_to_flat_args
from aspec.models.dspark import DSparkModel
from aspec.models.ops.loss import speculators_token_losses
from aspec.planning import PlanCompileError, compile_execution_plan
from aspec.training.checkpoint import _logical_training_topology, _validate_exact_resume_topology
from aspec.training.dspark_trainer import DSparkTrainer
from tests.test_dspark_loss_alignment import TinyDraft, objective_fixture

RECIPE = "speculators_redhat_v1"
CONFIG = Path(__file__).resolve().parents[2] / "configs/examples/sglang_qwen3_8b_dspark.yaml"


def reference_row(draft, batch, row, anchors, keep):
    """Independent dense reference, one upstream-style forward per packed row."""
    positions = anchors[:, None] + torch.arange(4)
    prev = draft.markov_head.markov_w1(batch["input_ids"][row, positions])
    hidden = draft.hidden.view(-1, 4, 8)
    head = batch["lm_head_weight"]
    logits = F.linear(hidden, head) + draft.markov_head.markov_w2(prev)
    teacher = F.linear(batch["last_hidden_states"][row, positions], head)
    mask = batch["loss_mask"][row, positions] * keep[:, None]
    decay = torch.exp(-torch.arange(4) / 4)
    labels = teacher.argmax(-1)
    ce = F.cross_entropy(logits.flatten(0, 1), labels.flatten(), reduction="none").view_as(mask)
    overlap = torch.minimum(logits.softmax(-1), teacher.softmax(-1)).sum(-1)
    confidence = draft.confidence_head(torch.cat([hidden, prev], dim=-1))
    bce = F.binary_cross_entropy_with_logits(confidence, overlap.detach(), reduction="none")
    return {
        key: (value * mask * decay).sum() / (mask.sum() + eps)
        for key, value, eps in (
            ("ce_loss", ce, 1e-5),
            ("tv_loss", 1 - overlap, 1e-5),
            ("confidence_loss", bce, 1e-8),
        )
    }, mask


@pytest.mark.parametrize("empty_second_row", [False, True])
def test_speculators_rows_components_and_gradients(monkeypatch, empty_second_row):
    torch.manual_seed(91)
    draft = TinyDraft()
    reference = copy.deepcopy(draft)
    # Two blocks per head chunk exercises compaction and checkpoint replay.
    model = DSparkModel(draft, block_size=4, num_anchors=2, loss_recipe=RECIPE)
    batch, anchors, keep = objective_fixture()
    anchors = anchors.masked_fill(~keep, 0).repeat(2, 1)
    keep = keep.repeat(2, 1)
    keep[1, 1:] = False
    if empty_second_row:
        keep[1].zero_()
    for key in ("input_ids", "loss_mask", "ctx_doc_ids", "last_hidden_states", "base_position_ids"):
        batch[key] = torch.cat([batch[key], batch[key].clone()])
    batch["loss_mask"][1, 4] = 0
    monkeypatch.setattr(
        model,
        "_draft_backbone",
        lambda *a, **kw: (
            draft.hidden.expand(2, -1, -1),
            anchors,
            keep,
            anchors.shape[1],
        ),
    )
    # Run the tensor operations eagerly on the CPU CI lane.
    monkeypatch.setattr(
        "aspec.models.dspark.speculators_token_losses", speculators_token_losses.__wrapped__
    )
    actual = model(**batch)
    rows = [reference_row(reference, batch, i, anchors[i], keep[i]) for i in range(2)]
    expected_terms = {key: (rows[0][0][key] + rows[1][0][key]) / 2 for key in rows[0][0]}
    expected = (
        0.1 * expected_terms["ce_loss"]
        + 0.9 * expected_terms["tv_loss"]
        + expected_terms["confidence_loss"]
    )
    torch.testing.assert_close(actual[0], expected)
    assert actual[6][1].item() == 2
    assert "l1_loss" not in actual[5]
    for key, value in expected_terms.items():
        numerator, denominator = actual[5][key]
        torch.testing.assert_close(numerator / denominator, value)
    torch.testing.assert_close(actual[4], sum(mask.sum(0) for _, mask in rows))
    assert rows[0][1][1].tolist() == [1, 1, 0, 1]  # a later slot survives a supervision gap
    actual[0].backward()
    expected.backward()
    for param, ref in zip(draft.parameters(), reference.parameters(), strict=True):
        torch.testing.assert_close(param.grad, ref.grad, atol=1e-6, rtol=1e-5)


def test_speculators_fused_tv_subgradient_at_partial_tie():
    # A permutation keeps the softmax normalizer equal, with one tied probability.
    draft = torch.tensor([[0.0, 1.0, 2.0]], requires_grad=True)
    teacher = torch.tensor([[1.0, 0.0, 2.0]], requires_grad=True)
    ce, tv, labels = speculators_token_losses.__wrapped__(draft, teacher)
    q, p = draft.detach().softmax(-1), teacher.detach().softmax(-1)
    indicator = (q <= p).float()
    tv.sum().backward()
    torch.testing.assert_close(draft.grad, -q * (indicator - (q * indicator).sum(-1, keepdim=True)))
    assert teacher.grad is None
    torch.testing.assert_close(ce, F.cross_entropy(draft, labels, reduction="none"))


def test_cp_shards_contribute_one_row_across_accumulation_steps(monkeypatch):
    model = DSparkModel(TinyDraft(), block_size=4, loss_recipe=RECIPE)
    model.draft_context_parallel_size = 2
    model.direct_cp_group = object()
    coefficient = torch.tensor(1.0, requires_grad=True)
    numerator = denominator = 0
    expected_rows = []
    for valid in ([1, 0, 0, 0, 1, 1, 1, 1], [0, 0, 0, 0, 1, 0, 0, 0]):
        mask = torch.tensor(valid).float().view(1, 2, 4)
        decay = torch.exp(-torch.arange(4) / 4)
        values = torch.arange(1.0, 9.0).view_as(mask)
        expected_rows.append((values * mask * decay).sum() / (mask.sum() + 1e-5))
        for rank in range(2):
            local = mask[:, rank : rank + 1]

            def all_reduce(count, group):
                assert group is model.direct_cp_group
                count.add_(mask[:, 1 - rank].sum())

            monkeypatch.setattr("aspec.models.dspark.dist.all_reduce", all_reduce)
            weights, _, rows = model._speculators_row_weights(local, local * decay)
            numerator += (coefficient * values[:, rank : rank + 1] * weights).sum()
            denominator += rows
    assert denominator == 2
    (numerator / denominator).backward()
    torch.testing.assert_close(coefficient.grad, torch.stack(expected_rows).mean())


def test_ce_only_speculators_requires_teacher():
    model = DSparkModel(TinyDraft(), l1_loss_alpha=0, confidence_head_alpha=0, loss_recipe=RECIPE)
    assert model.uses_target_hidden_states


def test_recipe_plan_wrapper_and_exact_resume(monkeypatch):
    overrides = ["training.attention_backend=flex_attention"]
    base = load_config(str(CONFIG), cli_args=overrides)
    config = load_config(
        str(CONFIG), cli_args=[*overrides, f"training.dspark_loss_recipe={RECIPE}"]
    )
    default = compile_execution_plan(base)
    plan = compile_execution_plan(config)
    assert default.workload.objective["loss_recipe"] == "deepspec"
    assert plan.workload.objective["loss_recipe"] == RECIPE
    assert plan.plan_id != default.plan_id
    args = config_to_flat_args(config)
    args.draft_algorithm, args.world_size = "dspark", 1
    monkeypatch.setattr(
        "aspec.training.trainer.Trainer.__init__",
        lambda trainer, args: setattr(trainer, "args", args),
    )
    trainer = DSparkTrainer(args)
    wrapper = trainer._build_training_wrapper(TinyDraft())
    assert wrapper.loss_recipe == RECIPE
    assert "tv_loss" in trainer._extra_loss_component_keys
    metadata = {"world_size": 1, "training_topology": _logical_training_topology(args, 1)}
    _validate_exact_resume_topology(args, metadata)
    legacy = copy.deepcopy(metadata)
    del legacy["training_topology"]["training_objective"]["loss_recipe"]
    with pytest.raises(RuntimeError, match="dspark_loss_recipe"):
        _validate_exact_resume_topology(args, legacy)
    args.dspark_loss_recipe = "deepspec"
    _validate_exact_resume_topology(args, legacy)
    with pytest.raises(RuntimeError, match="dspark_loss_recipe"):
        _validate_exact_resume_topology(args, metadata)


@pytest.mark.parametrize(
    "override", ["training.dspark_loss_recipe=typo", "training.dspark_l1_target_top_k=8"]
)
def test_invalid_recipe_config(override):
    config = load_config(str(CONFIG), cli_args=[f"training.dspark_loss_recipe={RECIPE}", override])
    with pytest.raises(PlanCompileError, match="dspark_loss_recipe|unfiltered"):
        compile_execution_plan(config)
