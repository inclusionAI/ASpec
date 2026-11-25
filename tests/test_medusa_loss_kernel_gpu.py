# Copyright 2026 Ant Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""GPU parity and edge cases for the fused Medusa CE epilogue.

Run with ``ASPEC_RUN_GPU_INTEGRATION=1`` on a CUDA device.  The reference is
Torch's BF16 linear followed by FP32 cross-entropy, matching the production
Medusa objective.
"""

import os

import pytest
import torch
import torch.nn.functional as F
from torch import nn

if os.environ.get("ASPEC_RUN_GPU_INTEGRATION") != "1":
    pytest.skip("set ASPEC_RUN_GPU_INTEGRATION=1 for GPU integration", allow_module_level=True)
if not torch.cuda.is_available():
    pytest.skip("CUDA is required", allow_module_level=True)
pytest.importorskip("triton")

from aspec.models.draft.medusa import MedusaConfig, MedusaDraftModel
from aspec.models.medusa import MedusaModel
from aspec.models.ops import medusa_ce
from aspec.models.ops.medusa_ce import medusa_head_ce, use_fused_medusa_ce


def _tensors(tokens, hidden, vocab, seed=0):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(tokens, hidden, device="cuda", dtype=torch.bfloat16, generator=generator)
    w = torch.randn(vocab, hidden, device="cuda", dtype=torch.bfloat16,
                    generator=generator) * 0.05
    targets = torch.randint(0, vocab, (tokens,), device="cuda", generator=generator)
    return x, w, targets


def _reference(x, w, targets):
    logits = F.linear(x, w).float()
    nll = F.cross_entropy(logits, targets, reduction="none")
    hits = (logits.argmax(-1) == targets).float()
    return nll, hits


def _manual_grads(x, w, targets):
    x_ref = x.detach().clone().requires_grad_(True)
    w_ref = w.detach().clone().requires_grad_(True)
    loss = F.cross_entropy(F.linear(x_ref, w_ref).float(), targets, reduction="sum")
    loss.backward()
    return x_ref.grad, w_ref.grad


def _fused_grads(x, w, targets, *, save_logits):
    x_fused = x.detach().clone().requires_grad_(True)
    w_fused = w.detach().clone().requires_grad_(True)
    nll, hits = medusa_head_ce(x_fused, w_fused, targets, save_logits=save_logits)
    nll.sum().backward()
    return nll.detach(), hits, x_fused.grad, w_fused.grad


def _assert_gradient_close(actual, expected, name):
    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-3, msg=name)
    error = (actual - expected).norm() / expected.norm().clamp_min(1e-6)
    assert error < 1e-2, f"{name} relative norm error {error.item():.4f}"


def _assert_chunked_gradient_close(actual, expected, name):
    """BF16 chunk accumulation is compared by scale, not near-zero atol."""
    assert torch.isfinite(actual).all(), f"{name} has non-finite values"
    error = ((actual.float() - expected.float()).norm()
             / expected.float().norm().clamp_min(1e-6))
    assert error < 1e-2, f"{name} relative norm error {error.item():.4f}"
    max_abs = (actual.float() - expected.float()).abs().max()
    scale = expected.float().abs().max().clamp_min(1.0)
    assert max_abs <= 1e-2 * scale, (
        f"{name} max abs error {max_abs.item():.6f} exceeds 1% of scale "
        f"{scale.item():.6f}"
    )


@pytest.mark.parametrize(
    "vocab,tokens",
    [(127, 1), (4096, 7), (4096, 128), (151936, 1024)],
)
def test_nll_and_hits_match_torch(vocab, tokens):
    x, w, targets = _tensors(tokens, 64, vocab)
    nll, hits = medusa_head_ce(x, w, targets, save_logits=True)
    ref_nll, ref_hits = _reference(x, w, targets)
    torch.testing.assert_close(nll, ref_nll, rtol=1e-4, atol=1e-5)
    assert torch.equal(hits, ref_hits)


@pytest.mark.parametrize("vocab", [3, 127, 4097])
def test_vocab_not_divisible_by_block(vocab):
    x, w, targets = _tensors(11, 32, vocab)
    nll, hits = medusa_head_ce(x, w, targets, save_logits=True)
    ref_nll, ref_hits = _reference(x, w, targets)
    torch.testing.assert_close(nll, ref_nll, rtol=1e-4, atol=1e-5)
    assert torch.equal(hits, ref_hits)


def test_target_at_tile_boundary():
    vocab = 4096
    targets = torch.tensor([0, 2047, 2048, 4095], device="cuda", dtype=torch.int64)
    x = torch.randn(4, 32, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(vocab, 32, device="cuda", dtype=torch.bfloat16) * 0.05
    nll, hits = medusa_head_ce(x, w, targets, save_logits=True)
    ref_nll, ref_hits = _reference(x, w, targets)
    torch.testing.assert_close(nll, ref_nll, rtol=1e-4, atol=1e-5)
    assert torch.equal(hits, ref_hits)


@pytest.mark.parametrize("vocab", [4096, 151936])
def test_hits_positive_with_argmax_targets(vocab):
    x, w, _ = _tensors(128, 64, vocab)
    targets = F.linear(x, w).float().argmax(-1).to(torch.int64)
    nll, hits = medusa_head_ce(x, w, targets, save_logits=True)
    ref_nll, ref_hits = _reference(x, w, targets)
    assert hits.sum() > 0
    torch.testing.assert_close(nll, ref_nll, rtol=1e-4, atol=1e-5)
    assert torch.equal(hits, ref_hits)


def test_dx_dw_match_reference():
    x, w, targets = _tensors(128, 64, 4096)
    _, _, dx, dw = _fused_grads(x, w, targets, save_logits=True)
    dx_ref, dw_ref = _manual_grads(x, w, targets)
    _assert_gradient_close(dx, dx_ref, "dx")
    _assert_gradient_close(dw, dw_ref, "dw")


def test_save_logits_modes_match():
    x, w, targets = _tensors(128, 64, 4096)
    nll_true, hits_true, dx_true, dw_true = _fused_grads(x, w, targets, save_logits=True)
    nll_false, hits_false, dx_false, dw_false = _fused_grads(x, w, targets, save_logits=False)
    torch.testing.assert_close(nll_true, nll_false, rtol=0, atol=0)
    assert torch.equal(hits_true, hits_false)
    torch.testing.assert_close(dx_true, dx_false, rtol=0, atol=0)
    torch.testing.assert_close(dw_true, dw_false, rtol=0, atol=0)


def test_chunked_backward_matches_full(monkeypatch):
    x, w, targets = _tensors(1024, 64, 4096)
    nll_full, hits_full, dx_full, dw_full = _fused_grads(x, w, targets, save_logits=True)

    monkeypatch.setattr(medusa_ce, "_DLOGITS_BUDGET_BYTES", 1)
    nll_chunk, hits_chunk, dx_chunk, dw_chunk = _fused_grads(x, w, targets, save_logits=True)
    torch.testing.assert_close(nll_full, nll_chunk, rtol=0, atol=0)
    assert torch.equal(hits_full, hits_chunk)
    _assert_chunked_gradient_close(dx_chunk, dx_full, "chunked saved dx")
    _assert_chunked_gradient_close(dw_chunk, dw_full, "chunked saved dw")

    _, _, dx_recompute, dw_recompute = _fused_grads(x, w, targets, save_logits=False)
    _assert_chunked_gradient_close(dx_recompute, dx_full, "chunked recompute dx")
    _assert_chunked_gradient_close(dw_recompute, dw_full, "chunked recompute dw")


def test_ties_break_at_lowest_index():
    vocab = 256
    x = torch.zeros(3, 16, device="cuda", dtype=torch.bfloat16)
    w = torch.zeros(vocab, 16, device="cuda", dtype=torch.bfloat16)
    targets = torch.tensor([0, 5, vocab - 1], device="cuda", dtype=torch.int64)
    nll, hits = medusa_head_ce(x, w, targets, save_logits=True)
    expected = torch.log(torch.tensor(float(vocab), device="cuda")).expand_as(nll)
    torch.testing.assert_close(nll, expected, rtol=1e-6, atol=1e-6)
    assert hits.tolist() == [1.0, 0.0, 0.0]


def test_nonfinite_input_produces_nonfinite_nll():
    x, w, targets = _tensors(128, 64, 4096)
    x = x.clone()
    x[0, 0] = float("nan")
    nll, _ = medusa_head_ce(x, w, targets, save_logits=True)
    ref_nll, _ = _reference(x, w, targets)
    assert torch.isnan(nll[0])
    assert torch.isnan(ref_nll[0])


def _small_model(gradient_checkpointing):
    config = MedusaConfig(
        hidden_size=32,
        vocab_size=128,
        target_hidden_size=32,
        target_num_hidden_layers=4,
        medusa_num_heads=3,
        medusa_num_layers=1,
    )
    draft = MedusaDraftModel(config).cuda().bfloat16()
    with torch.no_grad():
        for head in draft.medusa_head:
            nn.init.normal_(head[0].linear.weight, std=0.05)
            nn.init.normal_(head[-1].weight, std=0.05)
    return MedusaModel(draft, gradient_checkpointing=gradient_checkpointing)


def _model_inputs(seq_len=300):
    input_ids = torch.randint(0, 128, (1, seq_len), device="cuda")
    hidden_states = torch.randn(1, seq_len, 32, device="cuda", dtype=torch.bfloat16)
    loss_mask = (torch.rand(1, seq_len, device="cuda") > 0.2).long()
    attention_mask = torch.ones(1, seq_len, dtype=torch.long, device="cuda")
    return input_ids, hidden_states, loss_mask, attention_mask


def _require_fused_path(model, hidden_states, seq_len=128):
    weight = model.draft_model.medusa_head[0][-1].weight
    if not use_fused_medusa_ce(
        hidden_states[0], weight, num_rows=min(seq_len, hidden_states.shape[1])
    ):
        pytest.skip("fused Medusa CE is not supported on this GPU/runtime")


@pytest.mark.parametrize("gradient_checkpointing", [False, True])
def test_end_to_end_parity_with_python_path(gradient_checkpointing, monkeypatch):
    torch.manual_seed(1234)
    fused = _small_model(gradient_checkpointing)
    torch.manual_seed(1234)
    reference = _small_model(gradient_checkpointing)
    reference.load_state_dict(
        {key: value.clone() for key, value in fused.state_dict().items()}
    )
    input_ids, hidden_states, loss_mask, attention_mask = _model_inputs()
    _require_fused_path(fused, hidden_states)

    fused_losses, _, fused_accs, fused_counts, _ = fused(
        input_ids, hidden_states, loss_mask, attention_mask
    )
    monkeypatch.setattr("aspec.models.medusa.use_fused_medusa_ce", lambda *args, **kwargs: False)
    ref_losses, _, ref_accs, ref_counts, _ = reference(
        input_ids, hidden_states, loss_mask, attention_mask
    )

    for fused_loss, ref_loss in zip(fused_losses, ref_losses, strict=True):
        torch.testing.assert_close(fused_loss, ref_loss, rtol=1e-4, atol=1e-5)
    for fused_acc, ref_acc in zip(fused_accs, ref_accs, strict=True):
        torch.testing.assert_close(fused_acc, ref_acc, rtol=1e-5, atol=1e-6)
    assert [count.item() for count in fused_counts] == [count.item() for count in ref_counts]

    sum(fused_losses).backward()
    sum(ref_losses).backward()
    for (name, parameter), (ref_name, ref_parameter) in zip(
        fused.named_parameters(), reference.named_parameters(), strict=True
    ):
        assert name == ref_name
        assert parameter.grad is not None and ref_parameter.grad is not None
        _assert_gradient_close(parameter.grad, ref_parameter.grad, name)


def test_zero_selected_tokens_skips_kernel(monkeypatch):
    model = _small_model(True)
    calls = []
    monkeypatch.setattr("aspec.models.medusa.medusa_head_ce", lambda *args, **kwargs: calls.append(1))
    input_ids, hidden_states, _, attention_mask = _model_inputs()
    loss_mask = torch.zeros_like(input_ids)
    losses, _, _, counts, _ = model(input_ids, hidden_states, loss_mask, attention_mask)
    assert calls == []
    assert all(count.item() == 0 for count in counts)
    sum(losses).backward()
    for parameter in model.parameters():
        assert parameter.grad is not None
        assert torch.count_nonzero(parameter.grad) == 0


def test_state_dict_unchanged():
    model = _small_model(True)
    before = {key: (value.shape, value.dtype) for key, value in model.state_dict().items()}
    input_ids, hidden_states, loss_mask, attention_mask = _model_inputs()
    losses, _, _, _, _ = model(input_ids, hidden_states, loss_mask, attention_mask)
    sum(losses).backward()
    after = {key: (value.shape, value.dtype) for key, value in model.state_dict().items()}
    assert before == after
