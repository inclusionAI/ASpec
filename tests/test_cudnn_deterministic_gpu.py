# Copyright 2026 Ant Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Native cuDNN MHA64 forward and backward repeatability."""

import os

import pytest
import torch

pytestmark = pytest.mark.skipif(
    os.getenv("ASPEC_RUN_CUDNN_FLEX_INTEGRATION") != "1" or not torch.cuda.is_available(),
    reason="requires the native CUDA cuDNN integration environment",
)


@pytest.mark.parametrize("context_length", [8192, 262144])
def test_native_mha64_deterministic_forward_and_backward(context_length):
    from aspec.models.ops.cudnn_flex_attention import build_dspark_mla_mask, cudnn_flex_attention

    generator = torch.Generator(device="cuda").manual_seed(93)
    queries, heads, dim = 16, 4, 64
    anchors = torch.tensor([[7, context_length - 9]], device="cuda")
    mask = build_dspark_mla_mask(
        anchor_positions=anchors, block_keep_mask=torch.ones_like(anchors, dtype=torch.bool),
        context_length=context_length, block_size=8, is_causal=True, context_window=2048,
    )
    inputs = [torch.randn(1, length, heads, dim, device="cuda", dtype=torch.bfloat16,
                          generator=generator) for length in (queries, context_length + queries,
                                                             context_length + queries)]
    gradient = torch.randn(inputs[0].shape, device="cuda", dtype=torch.bfloat16,
                           generator=generator)
    enabled = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        torch.use_deterministic_algorithms(True)
        with torch.backends.cudnn.flags(deterministic=True, benchmark=False):
            repeats = []
            for _ in range(2):
                q, k, v = [value.detach().clone().requires_grad_() for value in inputs]
                output = cudnn_flex_attention(q, k, v, mask=mask, scale=dim ** -0.5,
                                              gqa_mode="native")
                grads = torch.autograd.grad(output, (q, k, v), gradient)
                assert torch.isfinite(output).all()
                assert all(torch.isfinite(value).all() for value in grads)
                repeats.append((output.detach(), *grads))
            for first, second in zip(*repeats, strict=True):
                assert torch.equal(first, second)
    finally:
        torch.use_deterministic_algorithms(enabled, warn_only=warn_only)
