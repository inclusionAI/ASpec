# Copyright 2026 Ant Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""CPU-side fallback and validation for the fused Medusa CE epilogue."""

import copy

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from aspec.models.draft.medusa import MedusaConfig, MedusaDraftModel
from aspec.models.medusa import MedusaModel
from aspec.models.ops import medusa_ce


def _tensors(rows=128, hidden=8, vocab=64, dtype=torch.bfloat16):
    return (
        torch.randn(rows, hidden, dtype=dtype),
        torch.randn(vocab, hidden, dtype=dtype),
    )


def test_predicate_rejects_cpu_tensors():
    hidden, weight = _tensors()
    assert medusa_ce.use_fused_medusa_ce(hidden, weight) is False


def test_predicate_rejects_non_bf16_and_shape_mismatch():
    hidden, weight = _tensors(dtype=torch.float32)
    assert medusa_ce.use_fused_medusa_ce(hidden, weight) is False
    hidden, weight = _tensors()
    assert medusa_ce.use_fused_medusa_ce(hidden[:, :4], weight) is False
    assert medusa_ce.use_fused_medusa_ce(hidden.transpose(0, 1), weight) is False


def test_predicate_honors_num_rows():
    hidden, weight = _tensors(rows=1)
    assert medusa_ce.use_fused_medusa_ce(hidden, weight, num_rows=1) is False
    assert medusa_ce.use_fused_medusa_ce(hidden, weight, num_rows=1024) is False


def test_predicate_small_selection_threshold(monkeypatch):
    class Fake:
        is_cuda = True
        dtype = torch.bfloat16
        ndim = 2
        shape = (64, 8)
        device = "fake"

        @staticmethod
        def is_contiguous():
            return True

    hidden, weight = Fake(), Fake()
    monkeypatch.setattr(medusa_ce, "_TRITON_IMPORT_ERROR", None)
    monkeypatch.setattr(medusa_ce, "triton", object())
    monkeypatch.setattr(medusa_ce, "_device_capability", lambda device: (8, 9))
    assert medusa_ce.use_fused_medusa_ce(hidden, weight, num_rows=63) is False
    assert medusa_ce.use_fused_medusa_ce(hidden, weight, num_rows=64) is True


def test_predicate_rejects_vocab_beyond_exact_argmax(monkeypatch):
    class Fake:
        is_cuda = True
        dtype = torch.bfloat16
        ndim = 2
        device = "fake"

        def __init__(self, shape):
            self.shape = shape

        @staticmethod
        def is_contiguous():
            return True

    hidden = Fake((64, 8))
    weight = Fake((medusa_ce._MAX_EXACT_ARGMAX_VOCAB + 1, 8))
    monkeypatch.setattr(medusa_ce, "_TRITON_IMPORT_ERROR", None)
    monkeypatch.setattr(medusa_ce, "triton", object())
    monkeypatch.setattr(medusa_ce, "_device_capability", lambda device: (8, 9))
    assert medusa_ce.use_fused_medusa_ce(hidden, weight) is False


def test_missing_triton_falls_back(monkeypatch):
    hidden, weight = _tensors()
    monkeypatch.setattr(medusa_ce, "_TRITON_IMPORT_ERROR", ImportError("missing"))
    assert medusa_ce.use_fused_medusa_ce(hidden, weight) is False
    with pytest.raises(RuntimeError, match="requires Triton"):
        medusa_ce.medusa_head_ce(hidden, weight, torch.zeros(128, dtype=torch.int64))


def test_medusa_head_ce_requires_cuda():
    hidden, weight = _tensors()
    targets = torch.zeros(128, dtype=torch.int64)
    with pytest.raises((ValueError, RuntimeError)):
        medusa_ce.medusa_head_ce(hidden, weight, targets)


def test_resolve_row_chunk_defaults():
    assert medusa_ce._resolve_row_chunk(151936, None) == 512
    assert medusa_ce._resolve_row_chunk(1_000_000, None) == 128
    assert medusa_ce._resolve_row_chunk(10, 7) == 7
    with pytest.raises(ValueError):
        medusa_ce._resolve_row_chunk(10, 0)


def _tiny_model():
    config = MedusaConfig(
        hidden_size=8,
        vocab_size=16,
        target_hidden_size=8,
        target_num_hidden_layers=4,
        medusa_num_heads=2,
        medusa_num_layers=1,
    )
    draft = MedusaDraftModel(config)
    with torch.no_grad():
        for head in draft.medusa_head:
            nn.init.normal_(head[0].linear.weight, std=0.05)
            nn.init.normal_(head[-1].weight, std=0.05)
    return MedusaModel(draft)


def test_save_fused_logits_budget():
    model = _tiny_model()
    model.gradient_checkpointing = False
    assert model._save_fused_logits(4094, 151936) is True
    model.gradient_checkpointing = True
    assert model._save_fused_logits(4094, 151936) is True
    assert model._save_fused_logits(100_000, 151936) is False


def test_two_layer_wiring_matches_reference(monkeypatch):
    config = MedusaConfig(
        hidden_size=8,
        vocab_size=16,
        target_hidden_size=8,
        target_num_hidden_layers=4,
        medusa_num_heads=2,
        medusa_num_layers=2,
    )
    draft = MedusaDraftModel(config)
    with torch.no_grad():
        for head in draft.medusa_head:
            nn.init.normal_(head[0].linear.weight, std=0.05)
            nn.init.normal_(head[1].linear.weight, std=0.05)
            nn.init.normal_(head[-1].weight, std=0.05)
    fused = MedusaModel(copy.deepcopy(draft))
    reference = MedusaModel(copy.deepcopy(draft))
    input_ids = torch.randint(0, 16, (1, 32))
    hidden_states = torch.randn(1, 32, 8)
    loss_mask = (torch.rand(1, 32) > 0.2).long()
    attention_mask = torch.ones(1, 32, dtype=torch.long)

    def fake_fused_ce(hidden, weight, targets, **kwargs):
        logits = F.linear(hidden, weight).float()
        return (
            F.cross_entropy(logits, targets, reduction="none"),
            (logits.argmax(-1) == targets).float(),
        )

    monkeypatch.setattr("aspec.models.medusa.use_fused_medusa_ce", lambda *args, **kwargs: True)
    monkeypatch.setattr("aspec.models.medusa.medusa_head_ce", fake_fused_ce)
    fused_losses, _, fused_accs, fused_counts, _ = fused(
        input_ids, hidden_states, loss_mask, attention_mask
    )

    monkeypatch.setattr("aspec.models.medusa.use_fused_medusa_ce", lambda *args, **kwargs: False)
    ref_losses, _, ref_accs, ref_counts, _ = reference(
        input_ids, hidden_states, loss_mask, attention_mask
    )

    for fused_loss, ref_loss in zip(fused_losses, ref_losses, strict=True):
        torch.testing.assert_close(fused_loss, ref_loss, rtol=2e-5, atol=2e-6)
    for fused_acc, ref_acc in zip(fused_accs, ref_accs, strict=True):
        torch.testing.assert_close(fused_acc, ref_acc, rtol=2e-5, atol=2e-6)
    assert [count.item() for count in fused_counts] == [count.item() for count in ref_counts]

    sum(fused_losses).backward()
    sum(ref_losses).backward()
    for (name, parameter), (ref_name, ref_parameter) in zip(
        fused.named_parameters(), reference.named_parameters(), strict=True
    ):
        assert name == ref_name
        torch.testing.assert_close(parameter.grad, ref_parameter.grad, rtol=2e-5, atol=2e-6)


def test_reference_path_zero_mask_keeps_zero_grads():
    model = _tiny_model()
    input_ids = torch.randint(0, 16, (1, 12))
    hidden_states = torch.randn(1, 12, 8)
    loss_mask = torch.zeros(1, 12, dtype=torch.long)
    attention_mask = torch.ones(1, 12, dtype=torch.long)

    losses, _, _, counts, _ = model(input_ids, hidden_states, loss_mask, attention_mask)
    assert all(count.item() == 0 for count in counts)
    sum(losses).backward()
    for parameter in model.parameters():
        assert parameter.grad is not None
        assert torch.count_nonzero(parameter.grad) == 0


def test_reference_path_finite_on_cpu():
    model = _tiny_model()
    input_ids = torch.randint(0, 16, (1, 16))
    hidden_states = torch.randn(1, 16, 8)
    loss_mask = torch.ones(1, 16, dtype=torch.long)
    attention_mask = torch.ones(1, 16, dtype=torch.long)

    losses, _, accs, counts, _ = model(input_ids, hidden_states, loss_mask, attention_mask)
    assert len(losses) == 2
    assert all(torch.isfinite(loss) for loss in losses)
    assert all(torch.isfinite(acc) for acc in accs)
    assert all(count > 0 for count in counts)
