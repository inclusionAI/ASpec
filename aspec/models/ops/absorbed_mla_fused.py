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

"""Fused absorbed-MLA attention with a FlashAttention-style Triton kernel.

The reference implementation in :mod:`aspec.models.ops.absorbed_mla` keeps the
logits dense and materializes them with ``einsum``. This module keeps the same
absorbed identity but fuses the score, online softmax and latent-output
accumulation, so it never materializes ``[B, H, S, T]`` scores or per-head
``k_nope``/``v`` tensors:

    q_lat  = q_nope @ W_UK            # [B, S, H, K]
    logits = q_lat @ c_kv^T + q_rope @ k_rope^T
    out    = softmax(logits) @ c_kv    # [B, S, H, K]
    output = out @ W_UV^T              # [B, S, H * Dv]

``W_UK``/``W_UV`` are per-head blocks of ``kv_b_proj.weight``; the latent
projection before and after the attention core stay in PyTorch so the gradient
for ``q_nope``/``kv_b_proj.weight`` flows through normal autograd.

This is a feasibility kernel, not a production path: it supports dense,
unmasked attention on contiguous BSHD/BTD tensors, and dense masks remain on the
reference implementation. Backward uses two deterministic kernels (query-side
and key/value-side) and never needs atomics.
"""

from __future__ import annotations

import math

import torch

from aspec.models.ops.absorbed_mla import split_kv_b_proj

try:  # pragma: no cover - import guard for CPU-only installs
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except ImportError:  # pragma: no cover
    triton = None
    tl = None
    _HAS_TRITON = False


def _next_power_of_2(value: int) -> int:
    return 1 << max(0, int(value) - 1).bit_length()


# The latent output forces a [BLOCK_N, BLOCK_K] FP32 accumulator in the
# key/value-side backward, so the key tile stays smaller than the query tile.
# ``_DKDV_GROUP`` bounds the head-reduction workspace: each program owns
# ``num_heads / GROUP`` partial latent-gradient tensors instead of one per head.
_FWD_TILE_M = 32
_FWD_TILE_N = 64
_FWD_WARPS = 4
_FWD_STAGES = 2
_DQ_TILE_M = 16
_DQ_TILE_N = 64
_DQ_WARPS = 4
_DQ_STAGES = 2
_DKDV_TILE_M = 64
_DKDV_TILE_N = 32
_DKDV_WARPS = 4
_DKDV_STAGES = 2
_DKDV_GROUP = 8


def _tile_sizes(
    key_dim: int,
    rope_dim: int,
    block_m: int,
    block_n: int,
    query_len: int,
) -> tuple[int, int, int, int]:
    block_k = max(16, _next_power_of_2(key_dim))
    block_dr = max(16, _next_power_of_2(rope_dim))
    # A long configured query tile is wasted on short draft blocks.
    block_m = max(16, min(block_m, _next_power_of_2(query_len)))
    return block_m, block_n, block_k, block_dr


def _dkdv_group(num_heads: int, preferred: int | None = None) -> int:
    if preferred is None:
        preferred = _DKDV_GROUP
    group = max(1, min(int(preferred), int(num_heads)))
    while num_heads % group:
        group -= 1
    return group


if _HAS_TRITON:

    @triton.jit
    def _absorbed_mla_fwd_kernel(
        q_lat_ptr,
        q_rope_ptr,
        c_kv_ptr,
        k_rope_ptr,
        out_ptr,
        lse_ptr,
        S,
        T,
        H,
        K,
        DR,
        scale,
        stride_ql_b,
        stride_ql_s,
        stride_ql_h,
        stride_qr_b,
        stride_qr_s,
        stride_qr_h,
        stride_c_b,
        stride_c_t,
        stride_kr_b,
        stride_kr_t,
        stride_o_b,
        stride_o_s,
        stride_o_h,
        stride_l_b,
        stride_l_h,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        BLOCK_DR: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_bh = tl.program_id(1)
        b = pid_bh // H
        h = pid_bh % H

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_k = tl.arange(0, BLOCK_K)
        offs_dr = tl.arange(0, BLOCK_DR)
        mask_m = offs_m < S
        mask_k = offs_k < K
        mask_dr = offs_dr < DR

        q_lat = tl.load(
            q_lat_ptr
            + b * stride_ql_b
            + offs_m[:, None] * stride_ql_s
            + h * stride_ql_h
            + offs_k[None, :],
            mask=mask_m[:, None] & mask_k[None, :],
            other=0.0,
        )
        q_rope = tl.load(
            q_rope_ptr
            + b * stride_qr_b
            + offs_m[:, None] * stride_qr_s
            + h * stride_qr_h
            + offs_dr[None, :],
            mask=mask_m[:, None] & mask_dr[None, :],
            other=0.0,
        )

        m_i = tl.full([BLOCK_M], float("-inf"), tl.float32)
        l_i = tl.zeros([BLOCK_M], tl.float32)
        acc = tl.zeros([BLOCK_M, BLOCK_K], tl.float32)

        for start_n in range(0, T, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N)
            mask_n = offs_n < T
            c = tl.load(
                c_kv_ptr
                + b * stride_c_b
                + offs_n[:, None] * stride_c_t
                + offs_k[None, :],
                mask=mask_n[:, None] & mask_k[None, :],
                other=0.0,
            )
            kr = tl.load(
                k_rope_ptr
                + b * stride_kr_b
                + offs_n[:, None] * stride_kr_t
                + offs_dr[None, :],
                mask=mask_n[:, None] & mask_dr[None, :],
                other=0.0,
            )

            scores = tl.dot(q_lat, tl.trans(c))
            scores += tl.dot(q_rope, tl.trans(kr))
            scores = scores * scale
            scores = tl.where(mask_n[None, :], scores, -1.0e30)

            m_new = tl.maximum(m_i, tl.max(scores, 1))
            alpha = tl.exp(m_i - m_new)
            probs = tl.exp(scores - m_new[:, None])
            l_i = l_i * alpha + tl.sum(probs, 1)
            acc = acc * alpha[:, None] + tl.dot(probs.to(c.dtype), c)
            m_i = m_new

        l_safe = tl.where(l_i > 0, l_i, 1.0)
        acc = acc / l_safe[:, None]
        lse = m_i + tl.log(l_safe)
        tl.store(
            out_ptr
            + b * stride_o_b
            + offs_m[:, None] * stride_o_s
            + h * stride_o_h
            + offs_k[None, :],
            acc.to(out_ptr.dtype.element_ty),
            mask=mask_m[:, None] & mask_k[None, :],
        )
        tl.store(lse_ptr + b * stride_l_b + h * stride_l_h + offs_m, lse, mask=mask_m)

    @triton.jit
    def _absorbed_mla_bwd_dq_kernel(
        q_lat_ptr,
        q_rope_ptr,
        c_kv_ptr,
        k_rope_ptr,
        dout_ptr,
        lse_ptr,
        d_ptr,
        dq_lat_ptr,
        dq_rope_ptr,
        S,
        T,
        H,
        K,
        DR,
        scale,
        stride_ql_b,
        stride_ql_s,
        stride_ql_h,
        stride_qr_b,
        stride_qr_s,
        stride_qr_h,
        stride_c_b,
        stride_c_t,
        stride_kr_b,
        stride_kr_t,
        stride_do_b,
        stride_do_s,
        stride_do_h,
        stride_l_b,
        stride_l_h,
        stride_d_b,
        stride_d_s,
        stride_d_h,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        BLOCK_DR: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_bh = tl.program_id(1)
        b = pid_bh // H
        h = pid_bh % H

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_k = tl.arange(0, BLOCK_K)
        offs_dr = tl.arange(0, BLOCK_DR)
        mask_m = offs_m < S
        mask_k = offs_k < K
        mask_dr = offs_dr < DR

        q_lat = tl.load(
            q_lat_ptr
            + b * stride_ql_b
            + offs_m[:, None] * stride_ql_s
            + h * stride_ql_h
            + offs_k[None, :],
            mask=mask_m[:, None] & mask_k[None, :],
            other=0.0,
        )
        q_rope = tl.load(
            q_rope_ptr
            + b * stride_qr_b
            + offs_m[:, None] * stride_qr_s
            + h * stride_qr_h
            + offs_dr[None, :],
            mask=mask_m[:, None] & mask_dr[None, :],
            other=0.0,
        )
        dout = tl.load(
            dout_ptr
            + b * stride_do_b
            + offs_m[:, None] * stride_do_s
            + h * stride_do_h
            + offs_k[None, :],
            mask=mask_m[:, None] & mask_k[None, :],
            other=0.0,
        )
        lse = tl.load(
            lse_ptr + b * stride_l_b + h * stride_l_h + offs_m,
            mask=mask_m,
            other=0.0,
        )
        d_i = tl.load(
            d_ptr + b * stride_d_b + offs_m * stride_d_s + h * stride_d_h,
            mask=mask_m,
            other=0.0,
        )

        dq_lat = tl.zeros([BLOCK_M, BLOCK_K], tl.float32)
        dq_rope = tl.zeros([BLOCK_M, BLOCK_DR], tl.float32)

        for start_n in range(0, T, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N)
            mask_n = offs_n < T
            c = tl.load(
                c_kv_ptr
                + b * stride_c_b
                + offs_n[:, None] * stride_c_t
                + offs_k[None, :],
                mask=mask_n[:, None] & mask_k[None, :],
                other=0.0,
            )
            kr = tl.load(
                k_rope_ptr
                + b * stride_kr_b
                + offs_n[:, None] * stride_kr_t
                + offs_dr[None, :],
                mask=mask_n[:, None] & mask_dr[None, :],
                other=0.0,
            )

            scores = tl.dot(q_lat, tl.trans(c))
            scores += tl.dot(q_rope, tl.trans(kr))
            scores = scores * scale
            scores = tl.where(mask_n[None, :], scores, -1.0e30)
            probs = tl.exp(scores - lse[:, None])
            probs = tl.where(mask_m[:, None] & mask_n[None, :], probs, 0.0)

            dp = tl.dot(dout, tl.trans(c))
            ds = probs * (dp - d_i[:, None]) * scale
            dq_lat += tl.dot(ds.to(c.dtype), c)
            dq_rope += tl.dot(ds.to(kr.dtype), kr)

        tl.store(
            dq_lat_ptr
            + b * stride_ql_b
            + offs_m[:, None] * stride_ql_s
            + h * stride_ql_h
            + offs_k[None, :],
            dq_lat.to(dq_lat_ptr.dtype.element_ty),
            mask=mask_m[:, None] & mask_k[None, :],
        )
        tl.store(
            dq_rope_ptr
            + b * stride_qr_b
            + offs_m[:, None] * stride_qr_s
            + h * stride_qr_h
            + offs_dr[None, :],
            dq_rope.to(dq_rope_ptr.dtype.element_ty),
            mask=mask_m[:, None] & mask_dr[None, :],
        )

    @triton.jit
    def _absorbed_mla_bwd_dkdv_kernel(
        q_lat_ptr,
        q_rope_ptr,
        c_kv_ptr,
        k_rope_ptr,
        dout_ptr,
        lse_ptr,
        d_ptr,
        dc_part_ptr,
        dkr_part_ptr,
        S,
        T,
        H,
        K,
        DR,
        scale,
        stride_ql_b,
        stride_ql_s,
        stride_ql_h,
        stride_qr_b,
        stride_qr_s,
        stride_qr_h,
        stride_c_b,
        stride_c_t,
        stride_kr_b,
        stride_kr_t,
        stride_do_b,
        stride_do_s,
        stride_do_h,
        stride_l_b,
        stride_l_h,
        stride_d_b,
        stride_d_s,
        stride_d_h,
        stride_dc_g,
        stride_dc_b,
        stride_dc_t,
        stride_dk_g,
        stride_dk_b,
        stride_dk_t,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        BLOCK_DR: tl.constexpr,
        GROUP: tl.constexpr,
    ):
        pid_n = tl.program_id(0)
        b = tl.program_id(1)
        pid_g = tl.program_id(2)
        head0 = pid_g * GROUP

        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)
        offs_dr = tl.arange(0, BLOCK_DR)
        mask_n = offs_n < T
        mask_k = offs_k < K
        mask_dr = offs_dr < DR

        c = tl.load(
            c_kv_ptr
            + b * stride_c_b
            + offs_n[:, None] * stride_c_t
            + offs_k[None, :],
            mask=mask_n[:, None] & mask_k[None, :],
            other=0.0,
        )
        kr = tl.load(
            k_rope_ptr
            + b * stride_kr_b
            + offs_n[:, None] * stride_kr_t
            + offs_dr[None, :],
            mask=mask_n[:, None] & mask_dr[None, :],
            other=0.0,
        )
        dc = tl.zeros([BLOCK_N, BLOCK_K], tl.float32)
        dkr = tl.zeros([BLOCK_N, BLOCK_DR], tl.float32)

        # Each program owns a disjoint head group, so the head reduction is
        # written to a per-group workspace instead of racing through atomics.
        for hi in range(0, GROUP):
            h = head0 + hi
            for start_m in range(0, S, BLOCK_M):
                offs_m = start_m + tl.arange(0, BLOCK_M)
                mask_m = offs_m < S
                q_lat = tl.load(
                    q_lat_ptr
                    + b * stride_ql_b
                    + offs_m[:, None] * stride_ql_s
                    + h * stride_ql_h
                    + offs_k[None, :],
                    mask=mask_m[:, None] & mask_k[None, :],
                    other=0.0,
                )
                q_rope = tl.load(
                    q_rope_ptr
                    + b * stride_qr_b
                    + offs_m[:, None] * stride_qr_s
                    + h * stride_qr_h
                    + offs_dr[None, :],
                    mask=mask_m[:, None] & mask_dr[None, :],
                    other=0.0,
                )
                dout = tl.load(
                    dout_ptr
                    + b * stride_do_b
                    + offs_m[:, None] * stride_do_s
                    + h * stride_do_h
                    + offs_k[None, :],
                    mask=mask_m[:, None] & mask_k[None, :],
                    other=0.0,
                )
                lse = tl.load(
                    lse_ptr + b * stride_l_b + h * stride_l_h + offs_m,
                    mask=mask_m,
                    other=0.0,
                )
                d_i = tl.load(
                    d_ptr + b * stride_d_b + offs_m * stride_d_s + h * stride_d_h,
                    mask=mask_m,
                    other=0.0,
                )

                scores = tl.dot(q_lat, tl.trans(c))
                scores += tl.dot(q_rope, tl.trans(kr))
                scores = scores * scale
                scores = tl.where(mask_n[None, :], scores, -1.0e30)
                probs = tl.exp(scores - lse[:, None])
                probs = tl.where(mask_m[:, None] & mask_n[None, :], probs, 0.0)

                dp = tl.dot(dout, tl.trans(c))
                ds = probs * (dp - d_i[:, None]) * scale
                dc += tl.dot(tl.trans(ds.to(c.dtype)), q_lat)
                dc += tl.dot(tl.trans(probs.to(c.dtype)), dout)
                dkr += tl.dot(tl.trans(ds.to(kr.dtype)), q_rope)

        base_dc = dc_part_ptr + pid_g * stride_dc_g + b * stride_dc_b
        base_dk = dkr_part_ptr + pid_g * stride_dk_g + b * stride_dk_b
        tl.store(
            base_dc + offs_n[:, None] * stride_dc_t + offs_k[None, :],
            dc,
            mask=mask_n[:, None] & mask_k[None, :],
        )
        tl.store(
            base_dk + offs_n[:, None] * stride_dk_t + offs_dr[None, :],
            dkr,
            mask=mask_n[:, None] & mask_dr[None, :],
        )


def _require_fused_support(tensor: torch.Tensor) -> None:
    if not _HAS_TRITON:
        raise RuntimeError("fused absorbed MLA requires triton")
    if not tensor.is_cuda:
        raise RuntimeError("fused absorbed MLA requires CUDA tensors")
    if tensor.dtype not in (torch.bfloat16, torch.float16):
        raise RuntimeError(
            "fused absorbed MLA supports bfloat16/float16, got "
            f"{tensor.dtype}"
        )


class _AbsorbedMLAFusedCore(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q_lat, q_rope, compressed_kv, k_rope, softmax_scale):
        _require_fused_support(q_lat)
        for name, tensor in (
            ("q_rope", q_rope),
            ("compressed_kv", compressed_kv),
            ("k_rope", k_rope),
        ):
            if tensor.dtype != q_lat.dtype:
                raise ValueError(f"{name} dtype must match q_lat")
            if tensor.device != q_lat.device:
                raise ValueError(f"{name} must be on the same device as q_lat")
        if q_lat.ndim != 4 or q_rope.ndim != 4 or compressed_kv.ndim != 3:
            raise ValueError("fused absorbed MLA expects [B,S,H,K], [B,S,H,Dr], [B,T,K]")
        if q_rope.shape[:3] != q_lat.shape[:3]:
            raise ValueError("q_rope must match q_lat batch/query/head axes")
        if k_rope.shape != (compressed_kv.size(0), compressed_kv.size(1), q_rope.size(-1)):
            raise ValueError("k_rope must be [B,T,Dr]")
        if compressed_kv.size(0) != q_lat.size(0) or compressed_kv.size(2) != q_lat.size(3):
            raise ValueError("compressed_kv must match q_lat batch and latent dimensions")

        q_lat = q_lat.contiguous()
        q_rope = q_rope.contiguous()
        compressed_kv = compressed_kv.contiguous()
        k_rope = k_rope.contiguous()
        batch, query_len, num_heads, key_dim = q_lat.shape
        key_len = compressed_kv.size(1)
        rope_dim = q_rope.size(-1)

        output = torch.empty_like(q_lat)
        lse = torch.empty(
            (batch, num_heads, query_len),
            device=q_lat.device,
            dtype=torch.float32,
        )
        block_m, block_n, block_k, block_dr = _tile_sizes(
            key_dim, rope_dim, _FWD_TILE_M, _FWD_TILE_N, query_len
        )
        grid = (triton.cdiv(query_len, block_m), batch * num_heads)
        with torch.cuda.device(q_lat.device):
            _absorbed_mla_fwd_kernel[grid](
                q_lat,
                q_rope,
                compressed_kv,
                k_rope,
                output,
                lse,
                query_len,
                key_len,
                num_heads,
                key_dim,
                rope_dim,
                softmax_scale,
                q_lat.stride(0),
                q_lat.stride(1),
                q_lat.stride(2),
                q_rope.stride(0),
                q_rope.stride(1),
                q_rope.stride(2),
                compressed_kv.stride(0),
                compressed_kv.stride(1),
                k_rope.stride(0),
                k_rope.stride(1),
                output.stride(0),
                output.stride(1),
                output.stride(2),
                lse.stride(0),
                lse.stride(1),
                BLOCK_M=block_m,
                BLOCK_N=block_n,
                BLOCK_K=block_k,
                BLOCK_DR=block_dr,
                num_warps=_FWD_WARPS,
                num_stages=_FWD_STAGES,
            )
        ctx.save_for_backward(q_lat, q_rope, compressed_kv, k_rope, output, lse)
        ctx.softmax_scale = float(softmax_scale)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        q_lat, q_rope, compressed_kv, k_rope, output, lse = ctx.saved_tensors
        if grad_output.device != q_lat.device:
            raise ValueError("grad_output must be on q_lat.device")
        grad_output = grad_output.contiguous()
        batch, query_len, num_heads, key_dim = q_lat.shape
        key_len = compressed_kv.size(1)
        rope_dim = q_rope.size(-1)
        scale = ctx.softmax_scale

        d_i = (grad_output.float() * output.float()).sum(-1)
        grad_q_lat = torch.empty_like(q_lat)
        grad_q_rope = torch.empty_like(q_rope)
        dq_m, dq_n, dq_k, dq_dr = _tile_sizes(
            key_dim, rope_dim, _DQ_TILE_M, _DQ_TILE_N, query_len
        )

        grid_m = (triton.cdiv(query_len, dq_m), batch * num_heads)
        with torch.cuda.device(q_lat.device):
            _absorbed_mla_bwd_dq_kernel[grid_m](
                q_lat,
                q_rope,
                compressed_kv,
                k_rope,
                grad_output,
                lse,
                d_i,
                grad_q_lat,
                grad_q_rope,
                query_len,
                key_len,
                num_heads,
                key_dim,
                rope_dim,
                scale,
                q_lat.stride(0),
                q_lat.stride(1),
                q_lat.stride(2),
                q_rope.stride(0),
                q_rope.stride(1),
                q_rope.stride(2),
                compressed_kv.stride(0),
                compressed_kv.stride(1),
                k_rope.stride(0),
                k_rope.stride(1),
                grad_output.stride(0),
                grad_output.stride(1),
                grad_output.stride(2),
                lse.stride(0),
                lse.stride(1),
                d_i.stride(0),
                d_i.stride(1),
                d_i.stride(2),
                BLOCK_M=dq_m,
                BLOCK_N=dq_n,
                BLOCK_K=dq_k,
                BLOCK_DR=dq_dr,
                num_warps=_DQ_WARPS,
                num_stages=_DQ_STAGES,
            )

        dk_m, dk_n, dk_k, dk_dr = _tile_sizes(
            key_dim, rope_dim, _DKDV_TILE_M, _DKDV_TILE_N, query_len
        )
        group = _dkdv_group(num_heads)
        num_groups = num_heads // group
        dc_part = torch.zeros(
            (num_groups, batch, key_len, key_dim),
            device=q_lat.device,
            dtype=torch.float32,
        )
        dkr_part = torch.zeros(
            (num_groups, batch, key_len, rope_dim),
            device=q_lat.device,
            dtype=torch.float32,
        )
        grid_n = (triton.cdiv(key_len, dk_n), batch, num_groups)
        with torch.cuda.device(q_lat.device):
            _absorbed_mla_bwd_dkdv_kernel[grid_n](
                q_lat,
                q_rope,
                compressed_kv,
                k_rope,
                grad_output,
                lse,
                d_i,
                dc_part,
                dkr_part,
                query_len,
                key_len,
                num_heads,
                key_dim,
                rope_dim,
                scale,
                q_lat.stride(0),
                q_lat.stride(1),
                q_lat.stride(2),
                q_rope.stride(0),
                q_rope.stride(1),
                q_rope.stride(2),
                compressed_kv.stride(0),
                compressed_kv.stride(1),
                k_rope.stride(0),
                k_rope.stride(1),
                grad_output.stride(0),
                grad_output.stride(1),
                grad_output.stride(2),
                lse.stride(0),
                lse.stride(1),
                d_i.stride(0),
                d_i.stride(1),
                d_i.stride(2),
                dc_part.stride(0),
                dc_part.stride(1),
                dc_part.stride(2),
                dkr_part.stride(0),
                dkr_part.stride(1),
                dkr_part.stride(2),
                BLOCK_M=dk_m,
                BLOCK_N=dk_n,
                BLOCK_K=dk_k,
                BLOCK_DR=dk_dr,
                GROUP=group,
                num_warps=_DKDV_WARPS,
                num_stages=_DKDV_STAGES,
            )
        grad_c_kv = dc_part.sum(dim=0).to(compressed_kv.dtype)
        grad_k_rope = dkr_part.sum(dim=0).to(k_rope.dtype)
        return grad_q_lat, grad_q_rope, grad_c_kv, grad_k_rope, None


def fused_absorbed_mla_core(
    q_lat: torch.Tensor,
    q_rope: torch.Tensor,
    compressed_kv: torch.Tensor,
    k_rope: torch.Tensor,
    *,
    softmax_scale: float,
) -> torch.Tensor:
    """Fused absorbed attention core: ``[B,S,H,K] -> [B,S,H,K]``.

    ``softmax_scale`` is required because the core cannot recover
    ``qk_nope_head_dim`` from the latent width alone.
    """
    return _AbsorbedMLAFusedCore.apply(
        q_lat, q_rope, compressed_kv, k_rope, float(softmax_scale)
    )


def fused_absorbed_mla_attention(
    q_nope: torch.Tensor,
    q_rope: torch.Tensor,
    compressed_kv: torch.Tensor,
    k_rope: torch.Tensor,
    kv_b_proj_weight: torch.Tensor,
    *,
    softmax_scale: float | None = None,
    attn_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Absorbed MLA attention with a fused Triton score/softmax/latent core.

    ``softmax_scale`` defaults to ``1 / sqrt(Dn + Dr)`` where ``Dn`` is the
    ``q_nope`` head dim and ``Dr`` the RoPE head dim, matching the materialized
    MLA path. The fused core does not implement dense masks; a non-``None``
    ``attn_mask`` is rejected rather than silently ignored.
    """
    if attn_mask is not None:
        raise NotImplementedError(
            "fused absorbed MLA does not support attn_mask; use "
            "absorbed_mla_attention as the dense-mask reference"
        )
    if q_nope.ndim != 4 or q_rope.ndim != 4:
        raise ValueError("q_nope and q_rope must be [B, S, H, D]")
    for name, tensor in (
        ("q_rope", q_rope),
        ("compressed_kv", compressed_kv),
        ("k_rope", k_rope),
        ("kv_b_proj_weight", kv_b_proj_weight),
    ):
        if tensor.device != q_nope.device:
            raise ValueError(f"{name} must be on the same device as q_nope")
    batch, query_len, num_heads, qk_nope_head_dim = q_nope.shape
    qk_rope_head_dim = q_rope.size(-1)
    if compressed_kv.ndim != 3 or compressed_kv.size(0) != batch:
        raise ValueError("compressed_kv must be [B, T, K]")
    if k_rope.shape != (batch, compressed_kv.size(1), qk_rope_head_dim):
        raise ValueError("k_rope must be [B, T, qk_rope_head_dim]")
    kv_lora_rank = compressed_kv.size(-1)
    inferred_v_head_dim = kv_b_proj_weight.shape[0] // num_heads - qk_nope_head_dim
    if inferred_v_head_dim <= 0 or kv_b_proj_weight.shape[0] % num_heads:
        raise ValueError("kv_b_proj.weight rows do not describe per-head K/V")
    w_uk, w_uv = split_kv_b_proj(
        kv_b_proj_weight,
        num_heads=num_heads,
        qk_nope_head_dim=qk_nope_head_dim,
        v_head_dim=inferred_v_head_dim,
        kv_lora_rank=kv_lora_rank,
    )
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(qk_nope_head_dim + qk_rope_head_dim)

    q_lat = torch.einsum("bshn,hnk->bshk", q_nope, w_uk)
    out_latent = _AbsorbedMLAFusedCore.apply(
        q_lat, q_rope, compressed_kv, k_rope, float(softmax_scale)
    )
    output = torch.einsum("bshk,hvk->bshv", out_latent, w_uv)
    return output.reshape(batch, query_len, num_heads * inferred_v_head_dim)
