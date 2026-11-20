# Copyright 2026 Ant Group Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Reference absorbed-MLA attention for feasibility checks.

The standard DeepSeek-V2/V3 identity re-associates the materialized MLA
projections:

    q_nope @ (c_kv @ W_UK^T)^T  ==  (q_nope @ W_UK) @ c_kv^T
    sum_t p_t (c_kv_t @ W_UV^T) ==  (sum_t p_t c_kv_t) @ W_UV^T

so the attention boundary can consume the normalized latent ``c_kv`` directly
instead of materializing per-head ``k_nope`` and ``v``. RoPE is still applied to
the small ``qk_rope_head_dim`` side, and the two score terms are added before
the softmax.

This module is a dense-mask reference implementation used by numerical and
feasibility checks. It is not wired into training, and it does not implement the
FlexAttention, cuDNN THD, Direct-CP or fused-kernel paths needed for a
production claim. Dense boolean masks are broadcast against ``[B, H, S, T]``.
"""

from __future__ import annotations

import math

import torch


def split_kv_b_proj(
    weight: torch.Tensor,
    *,
    num_heads: int,
    qk_nope_head_dim: int,
    v_head_dim: int,
    kv_lora_rank: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Split ``kv_b_proj.weight`` into per-head ``W_UK`` and ``W_UV`` views."""
    expected = (num_heads * (qk_nope_head_dim + v_head_dim), kv_lora_rank)
    if tuple(weight.shape) != expected:
        raise ValueError(
            "kv_b_proj.weight must have shape "
            f"{expected}, got {tuple(weight.shape)}"
        )
    view = weight.view(num_heads, qk_nope_head_dim + v_head_dim, kv_lora_rank)
    return view[:, :qk_nope_head_dim, :], view[:, qk_nope_head_dim:, :]


def absorbed_mla_attention(
    q_nope: torch.Tensor,
    q_rope: torch.Tensor,
    compressed_kv: torch.Tensor,
    k_rope: torch.Tensor,
    kv_b_proj_weight: torch.Tensor,
    *,
    softmax_scale: float | None = None,
    attn_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Absorbed MLA attention with a dense mask.

    Args:
        q_nope: ``[B, S, H, Dn]`` query without RoPE.
        q_rope: ``[B, S, H, Dr]`` query after RoPE.
        compressed_kv: ``[B, T, K]`` normalized latent KV before ``kv_b_proj``.
        k_rope: ``[B, T, Dr]`` shared MQA key after RoPE.
        kv_b_proj_weight: ``[H * (Dn + Dv), K]`` materialized projection weight.
        softmax_scale: score scale; defaults to ``1 / sqrt(Dn + Dr)``.
        attn_mask: optional boolean mask broadcastable to ``[B, H, S, T]``.

    Returns:
        ``[B, S, H * Dv]`` attention output before ``o_proj``.
    """
    if q_nope.ndim != 4 or q_rope.ndim != 4:
        raise ValueError("q_nope and q_rope must be [B, S, H, D]")
    batch, query_len, num_heads, qk_nope_head_dim = q_nope.shape
    if q_rope.shape[:3] != q_nope.shape[:3]:
        raise ValueError("q_rope must match q_nope's batch, query and head axes")
    qk_rope_head_dim = q_rope.size(-1)
    if compressed_kv.ndim != 3:
        raise ValueError("compressed_kv must be [B, T, K]")
    if compressed_kv.shape[0] != batch:
        raise ValueError("compressed_kv batch size must match the queries")
    key_len, kv_lora_rank = compressed_kv.shape[1], compressed_kv.shape[2]
    if k_rope.shape != (batch, key_len, qk_rope_head_dim):
        raise ValueError(
            "k_rope must be [B, T, qk_rope_head_dim] with the compressed_kv "
            "batch and key length"
        )
    if kv_b_proj_weight.ndim != 2:
        raise ValueError("kv_b_proj.weight must be a matrix")
    inferred_v_head_dim = (
        kv_b_proj_weight.shape[0] // num_heads - qk_nope_head_dim
    )
    if inferred_v_head_dim <= 0 or kv_b_proj_weight.shape[0] % num_heads:
        raise ValueError("kv_b_proj.weight rows do not describe per-head K/V")
    w_uk, w_uv = split_kv_b_proj(
        kv_b_proj_weight,
        num_heads=num_heads,
        qk_nope_head_dim=qk_nope_head_dim,
        v_head_dim=inferred_v_head_dim,
        kv_lora_rank=kv_lora_rank,
    )

    dtype = q_nope.dtype
    for name, tensor in (
        ("q_rope", q_rope),
        ("compressed_kv", compressed_kv),
        ("k_rope", k_rope),
        ("kv_b_proj.weight", kv_b_proj_weight),
    ):
        if tensor.dtype != dtype:
            raise ValueError(f"{name} dtype must match q_nope ({dtype})")
    if not dtype.is_floating_point:
        raise ValueError("absorbed MLA requires floating-point tensors")
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(qk_nope_head_dim + qk_rope_head_dim)
    if (
        isinstance(softmax_scale, bool)
        or not math.isfinite(softmax_scale)
        or softmax_scale <= 0
    ):
        raise ValueError("softmax_scale must be positive and finite")

    q_latent = torch.einsum("bshn,hnk->bshk", q_nope, w_uk)
    scores = torch.einsum("bshk,btk->bhst", q_latent, compressed_kv)
    scores = scores + torch.einsum("bshr,btr->bhst", q_rope, k_rope)
    scores = scores * softmax_scale
    if attn_mask is not None:
        if attn_mask.dtype != torch.bool:
            raise ValueError("attn_mask must be boolean")
        try:
            mask = attn_mask.expand(scores.shape)
        except RuntimeError as exc:
            raise ValueError(
                "attn_mask must broadcast against [B, H, S, T]"
            ) from exc
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
    probs = torch.softmax(scores.float(), dim=-1)
    if attn_mask is not None:
        # Match fused SDPA: rows with no visible key contribute zero, not a
        # uniform average over the finite masked scores.
        probs = probs * mask.any(dim=-1, keepdim=True)
    probs = probs.to(dtype)
    out_latent = torch.einsum("bhst,btk->bshk", probs, compressed_kv)
    output = torch.einsum("bshk,hvk->bshv", out_latent, w_uv)
    return output.reshape(batch, query_len, num_heads * inferred_v_head_dim)
