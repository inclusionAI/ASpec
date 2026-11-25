# Copyright 2026 Ant Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""GPU checks for the fused Triton absorbed-MLA attention core.

These tests compare the fused kernel against the dense PyTorch reference in
``aspec.models.ops.absorbed_mla``. They require a CUDA device and Triton; the
CPU suite skips them.
"""

import math

import pytest
import torch

from aspec.models.ops import absorbed_mla_fused as fused_ops
from aspec.models.ops.absorbed_mla import absorbed_mla_attention

pytestmark = pytest.mark.skipif(
    not fused_ops._HAS_TRITON or not torch.cuda.is_available(),
    reason="fused absorbed MLA requires Triton and CUDA",
)


def _make_tensors(*, batch=1, query=8, key=41, heads=4, nope=16, rope=16, value=16, latent=32):
    generator = torch.Generator().manual_seed(41)
    return {
        "q_nope": torch.randn(
            batch, query, heads, nope, generator=generator
        ).cuda().half(),
        "q_rope": torch.randn(
            batch, query, heads, rope, generator=generator
        ).cuda().half(),
        "compressed_kv": torch.randn(
            batch, key, latent, generator=generator
        ).cuda().half(),
        "k_rope": torch.randn(batch, key, rope, generator=generator).cuda().half(),
        "kv_b_proj_weight": torch.randn(
            heads * (nope + value), latent, generator=generator
        ).cuda().half(),
    }


def _relative_error(actual, expected):
    expected = expected.float()
    return (
        (actual.float() - expected).norm() / expected.norm().clamp_min(1e-6)
    ).item()


def _run_grads(fn, tensors):
    values = {name: value.detach().clone().requires_grad_(True) for name, value in tensors.items()}
    output = fn(values)
    grads = torch.autograd.grad(output.float().pow(2).sum(), list(values.values()))
    return output, grads


@pytest.mark.parametrize(
    "shape",
    [
        dict(batch=1, query=8, key=41, heads=2),
        dict(batch=1, query=7, key=41, heads=3),
        dict(batch=2, query=33, key=100, heads=8, latent=64),
        dict(batch=1, query=70, key=200, heads=4, latent=128),
    ],
)
def test_fused_forward_and_gradients_match_reference(shape):
    tensors = _make_tensors(**shape)
    reference = absorbed_mla_attention(**tensors)
    fused = fused_ops.fused_absorbed_mla_attention(**tensors)
    assert torch.isfinite(fused).all()
    assert _relative_error(fused, reference) < 2e-2

    _, reference_grads = _run_grads(
        lambda values: absorbed_mla_attention(**values), tensors
    )
    _, fused_grads = _run_grads(
        lambda values: fused_ops.fused_absorbed_mla_attention(**values), tensors
    )
    for name, expected, actual in zip(
        tensors, reference_grads, fused_grads, strict=True
    ):
        assert torch.isfinite(actual).all(), name
        assert _relative_error(actual, expected) < 3e-2, name


def test_fused_default_scale_matches_materialized():
    tensors = _make_tensors()
    fused = fused_ops.fused_absorbed_mla_attention(**tensors)
    explicit = fused_ops.fused_absorbed_mla_attention(
        **tensors,
        softmax_scale=1.0 / math.sqrt(16 + 16),
    )
    torch.testing.assert_close(fused, explicit, rtol=0, atol=0)


def test_fused_core_matches_float32_reference_at_latent_512():
    torch.manual_seed(7)
    query, key, heads, latent, rope = 8, 64, 2, 512, 64
    q_lat = torch.randn(1, query, heads, latent, device="cuda", dtype=torch.bfloat16)
    q_rope = torch.randn(1, query, heads, rope, device="cuda", dtype=torch.bfloat16)
    c_kv = torch.randn(1, key, latent, device="cuda", dtype=torch.bfloat16)
    k_rope = torch.randn(1, key, rope, device="cuda", dtype=torch.bfloat16)
    scale = 1.0 / math.sqrt(128 + 64)

    def reference(q_lat, q_rope, c_kv, k_rope):
        scores = torch.einsum("bshk,btk->bhst", q_lat.float(), c_kv.float())
        scores = scores + torch.einsum("bshr,btr->bhst", q_rope.float(), k_rope.float())
        probs = torch.softmax(scores * scale, dim=-1)
        return torch.einsum("bhst,btk->bshk", probs, c_kv.float())

    values = [q_lat, q_rope, c_kv, k_rope]
    expected = reference(*values)
    actual = fused_ops.fused_absorbed_mla_core(
        q_lat, q_rope, c_kv, k_rope, softmax_scale=scale
    )
    assert torch.isfinite(actual).all()
    assert _relative_error(actual, expected) < 1e-2

    fused_values = [value.detach().clone().requires_grad_(True) for value in values]
    ref_values = [value.detach().clone().requires_grad_(True) for value in values]
    ref_out = reference(*ref_values)
    fused_out = fused_ops.fused_absorbed_mla_core(
        *fused_values, softmax_scale=scale
    )
    ref_grads = torch.autograd.grad(ref_out.pow(2).sum(), ref_values)
    fused_grads = torch.autograd.grad(fused_out.float().pow(2).sum(), fused_values)
    for expected_grad, actual_grad in zip(ref_grads, fused_grads, strict=True):
        assert torch.isfinite(actual_grad).all()
        assert _relative_error(actual_grad, expected_grad) < 3e-2


def test_fused_core_rejects_cpu_tensors():
    if not fused_ops._HAS_TRITON:
        pytest.skip("triton is not installed")
    tensor = torch.zeros(1, 2, 1, 16)
    with pytest.raises(RuntimeError, match="CUDA"):
        fused_ops.fused_absorbed_mla_core(
            tensor, tensor, torch.zeros(1, 2, 16), torch.zeros(1, 2, 16),
            softmax_scale=0.1,
        )


def test_fused_rejects_dense_mask():
    tensors = _make_tensors()
    mask = torch.ones(1, 8, 4, 41, dtype=torch.bool, device="cuda")
    with pytest.raises(NotImplementedError, match="attn_mask"):
        fused_ops.fused_absorbed_mla_attention(**tensors, attn_mask=mask)


def _make_core_tensors(*, device="cuda", batch=1, query=8, key=41, heads=4, latent=32, rope=16):
    generator = torch.Generator(device=device).manual_seed(41)
    return (
        torch.randn(batch, query, heads, latent, generator=generator, device=device, dtype=torch.float16),
        torch.randn(batch, query, heads, rope, generator=generator, device=device, dtype=torch.float16),
        torch.randn(batch, key, latent, generator=generator, device=device, dtype=torch.float16),
        torch.randn(batch, key, rope, generator=generator, device=device, dtype=torch.float16),
    )


def test_fused_core_rejects_mismatched_compressed_kv():
    q_lat, q_rope, _, k_rope = _make_core_tensors()
    with pytest.raises(ValueError, match="compressed_kv"):
        fused_ops.fused_absorbed_mla_core(
            q_lat,
            q_rope,
            torch.zeros(2, 41, 32, device="cuda", dtype=torch.float16),
            torch.zeros(2, 41, 16, device="cuda", dtype=torch.float16),
            softmax_scale=0.1,
        )
    with pytest.raises(ValueError, match="compressed_kv"):
        fused_ops.fused_absorbed_mla_core(
            q_lat,
            q_rope,
            torch.zeros(1, 41, 16, device="cuda", dtype=torch.float16),
            k_rope,
            softmax_scale=0.1,
        )


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two CUDA devices")
def test_fused_core_rejects_cross_device_inputs():
    q_lat, q_rope, compressed_kv, k_rope = _make_core_tensors(device="cuda:0")
    with pytest.raises(ValueError, match="same device"):
        fused_ops.fused_absorbed_mla_core(
            q_lat,
            q_rope,
            compressed_kv.to("cuda:1"),
            k_rope.to("cuda:1"),
            softmax_scale=0.1,
        )


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two CUDA devices")
def test_fused_core_runs_on_input_device_when_not_current():
    previous = torch.cuda.current_device()
    try:
        torch.cuda.set_device(0)
        q_lat, q_rope, compressed_kv, k_rope = _make_core_tensors(device="cuda:1")
        for tensor in (q_lat, q_rope, compressed_kv, k_rope):
            tensor.requires_grad_(True)
        output = fused_ops.fused_absorbed_mla_core(
            q_lat, q_rope, compressed_kv, k_rope, softmax_scale=0.1
        )
        assert output.device == torch.device("cuda:1")
        output.float().sum().backward()
        for tensor in (q_lat, q_rope, compressed_kv, k_rope):
            assert tensor.grad is not None
            assert tensor.grad.device == torch.device("cuda:1")
    finally:
        torch.cuda.set_device(previous)
