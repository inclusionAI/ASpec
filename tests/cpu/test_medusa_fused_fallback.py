# Copyright 2026 Ant Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""CPU-side fallback and validation for the fused Medusa CE epilogue."""

import copy

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from aspec.models.draft.medusa import MedusaConfig, MedusaDraftModel
from aspec.models.medusa import MedusaModel, should_save_fused_logits
from aspec.models.ops import medusa_ce


class _FakeTensor:
    def __init__(
        self,
        shape=(64, 8),
        *,
        dtype=torch.bfloat16,
        is_cuda=True,
        contiguous=True,
        ndim=2,
        device="fake",
        requires_grad=False,
    ):
        self.shape = shape
        self.dtype = dtype
        self.is_cuda = is_cuda
        self._contiguous = contiguous
        self.ndim = ndim
        self.device = device
        self.requires_grad = requires_grad

    def is_contiguous(self):
        return self._contiguous


def _enable_fake_fused(monkeypatch):
    monkeypatch.setattr(medusa_ce, "_TRITON_IMPORT_ERROR", None)
    monkeypatch.setattr(medusa_ce, "triton", object())
    monkeypatch.setattr(medusa_ce, "_device_capability", lambda device: (8, 9))


def _tensors(rows=128, hidden=8, vocab=64, dtype=torch.bfloat16):
    return (
        torch.randn(rows, hidden, dtype=dtype),
        torch.randn(vocab, hidden, dtype=dtype),
    )


def test_predicate_rejects_cpu_tensors():
    hidden, weight = _tensors()
    assert medusa_ce.use_fused_medusa_ce(hidden, weight) is False


def test_predicate_rejects_non_bf16_and_shape_mismatch(monkeypatch):
    _enable_fake_fused(monkeypatch)
    assert medusa_ce.use_fused_medusa_ce(_FakeTensor(), _FakeTensor()) is True
    assert medusa_ce.use_fused_medusa_ce(
        _FakeTensor(dtype=torch.float32), _FakeTensor()
    ) is False
    assert medusa_ce.use_fused_medusa_ce(
        _FakeTensor(ndim=3), _FakeTensor()
    ) is False
    assert medusa_ce.use_fused_medusa_ce(
        _FakeTensor(shape=(64, 4)), _FakeTensor()
    ) is False
    assert medusa_ce.use_fused_medusa_ce(
        _FakeTensor(contiguous=False), _FakeTensor()
    ) is False


def test_predicate_small_selection_threshold(monkeypatch):
    _enable_fake_fused(monkeypatch)
    hidden, weight = _FakeTensor(), _FakeTensor()
    assert medusa_ce.use_fused_medusa_ce(hidden, weight, num_rows=63) is False
    assert medusa_ce.use_fused_medusa_ce(hidden, weight, num_rows=64) is True


def test_predicate_rejects_vocab_beyond_exact_argmax(monkeypatch):
    _enable_fake_fused(monkeypatch)
    hidden = _FakeTensor()
    weight = _FakeTensor(shape=(medusa_ce._MAX_EXACT_ARGMAX_VOCAB + 1, 8))
    assert medusa_ce.use_fused_medusa_ce(hidden, weight) is False


def test_missing_triton_falls_back(monkeypatch):
    hidden, weight = _tensors()
    monkeypatch.setattr(medusa_ce, "_TRITON_IMPORT_ERROR", ImportError("missing"))
    assert medusa_ce.use_fused_medusa_ce(hidden, weight) is False
    with pytest.raises(RuntimeError, match="requires Triton"):
        medusa_ce.medusa_head_ce(hidden, weight, torch.zeros(128, dtype=torch.int64))


def test_medusa_head_ce_requires_cuda(monkeypatch):
    _enable_fake_fused(monkeypatch)
    hidden = _FakeTensor(is_cuda=False)
    weight = _FakeTensor()
    targets = torch.zeros(64, dtype=torch.int64)
    with pytest.raises(ValueError, match="requires CUDA"):
        medusa_ce.medusa_head_ce(hidden, weight, targets)


def test_resolve_row_chunk_defaults():
    assert medusa_ce._resolve_row_chunk(151936, None) == 512
    assert medusa_ce._resolve_row_chunk(1_000_000, None) == 128
    assert medusa_ce._resolve_row_chunk(10, 7) == 7
    with pytest.raises(ValueError):
        medusa_ce._resolve_row_chunk(10, 0)


def test_bounded_rows_respects_byte_budget():
    assert medusa_ce._bounded_rows(8192, 151936, 2 * 1024**3) == 7067
    assert medusa_ce._bounded_rows(8192, 151936, 256 * 1024**2) == 883
    assert medusa_ce._bounded_rows(1, 151936, 1) == 1


def test_should_save_fused_logits_pins_k5_boundary():
    assert should_save_fused_logits(4094, 151936, 5, True) is True
    assert should_save_fused_logits(8192, 151936, 5, True) is False
    assert should_save_fused_logits(100_000, 151936, 5, False) is True
    model = _tiny_model()
    assert model._save_fused_logits(4094, 151936) == should_save_fused_logits(
        4094, 151936, model.num_heads, model.gradient_checkpointing
    )


def test_save_logits_gate():
    class Requiring:
        def __init__(self, requires_grad):
            self.requires_grad = requires_grad

    trainable = Requiring(True)
    frozen = Requiring(False)
    with torch.no_grad():
        assert medusa_ce._save_logits_for_backward(True, trainable, trainable) is False
    assert medusa_ce._save_logits_for_backward(True, trainable, frozen) is True
    assert medusa_ce._save_logits_for_backward(True, frozen, frozen) is False
    assert medusa_ce._save_logits_for_backward(False, trainable, trainable) is False


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
    calls = []

    def fake_fused_ce(hidden, weight, targets, **kwargs):
        calls.append(hidden.shape)
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

    assert len(calls) == 2
    assert all(shape[1] == config.hidden_size for shape in calls)
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


def test_benchmark_reference_head_matches_model():
    from tools.performance.bench_medusa_loss import _reference_head

    model = _tiny_model()
    head = model.draft_model.medusa_head[0]
    targets = torch.randint(0, 16, (40,))
    features = torch.randn(40, 8)
    positions = torch.arange(40)
    model_loss, model_hits = model._reference_head_loss(
        head, features, targets, positions, positions, positions
    )
    bench_loss, bench_hits = _reference_head(
        head, features, targets, model.gradient_checkpointing
    )
    torch.testing.assert_close(model_loss, bench_loss, rtol=0, atol=0)
    torch.testing.assert_close(model_hits, bench_hits, rtol=0, atol=0)


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
