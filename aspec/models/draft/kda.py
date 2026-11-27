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

"""Kimi Delta Attention (KDA) for DSpark/DFlash Draft architectures.

Implements the KDA layer and state-branching dual-source semantics inspired by
Kimi Linear (arXiv:2510.26692) and Flash Linear Attention's ``fla/layers/kda.py``
(MIT License):

  - Fine-grained channel-wise decay:
      g_t = -exp(A_log) * softplus(f_t + dt_bias)
  - Delta-rule rank-1 state update:
      S_t = (I - beta_t k_t k_t^T) Diag(exp(g_t)) S_{t-1} + beta_t k_t v_t^T
      o_t = scale * S_t^T q_t
  - Document-isolated causal context pass to extract anchor prefix states S_{a-1}
    (and optional short-convolution history).
  - Independent causal B-step Draft block rollout starting from each anchor's
    branched prefix state, mathematically identical to stateful proposal
    generation where verified Target context advances the persistent state and
    each Draft proposal branches without mutating it.

Execution backends are explicit (no silent fallback):
  - ``fla``: calls ``fla.ops.kda.chunk_kda`` / ``fused_recurrent_kda`` on CUDA.
  - ``reference``: independent pure-PyTorch recurrent and chunkwise reference
    implementations for mathematical cross-checking and CPU testing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class KDAForwardMetadata:
    """Anchor and document-boundary coordinates shared by KDA Draft layers."""

    anchor_positions: torch.Tensor
    block_keep_mask: torch.Tensor
    block_size: int
    context_doc_ids: Optional[torch.Tensor] = None


@dataclass(frozen=True)
class KDAPrefixState:
    """Persistent KDA prefix state for incremental proposal generation.

    ``recurrent_state`` has shape ``[B, H, D_k, D_v]`` (FP32).
    ``k_conv_state`` and ``v_conv_state`` have shapes ``[B, conv_size - 1, key_dim]``
    and ``[B, conv_size - 1, value_dim]`` when ``use_short_conv=True`` and
    ``conv_size > 1``, or ``None`` otherwise.
    """

    recurrent_state: torch.Tensor
    k_conv_state: Optional[torch.Tensor] = None
    v_conv_state: Optional[torch.Tensor] = None


def _import_fla_kda():
    """Import FLA KDA kernels strictly; fail closed when unavailable."""
    try:
        from fla.ops.kda import chunk_kda, fused_recurrent_kda
    except ImportError as exc:
        raise RuntimeError(
            "kda_backend='fla' requires flash-linear-attention (fla.ops.kda); "
            "install flash-linear-attention in the Draft environment or select "
            "kda_backend='reference' explicitly for CPU/reference execution"
        ) from exc
    return chunk_kda, fused_recurrent_kda


def _l2_normalize(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return x / (torch.linalg.vector_norm(x.float(), dim=-1, keepdim=True) + eps).to(x.dtype)


def kda_gate_reference(
    f_raw: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
) -> torch.Tensor:
    """Compute fine-grained KDA log-decay ``g_t`` in FP32.

    Args:
        f_raw: ``[B, T, H, D_k]`` raw forget-gate projections.
        A_log: ``[H]`` log decay base parameter.
        dt_bias: ``[H * D_k]`` or ``[H, D_k]`` per-channel bias.

    Returns:
        ``g`` of shape ``[B, T, H, D_k]`` in FP32 with ``g <= 0``.
    """
    bias = dt_bias.float().view(1, 1, f_raw.shape[2], f_raw.shape[3])
    neg_exp_a = -torch.exp(A_log.float()).view(1, 1, -1, 1)
    return neg_exp_a * F.softplus(f_raw.float() + bias)


def recurrent_kda_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    *,
    scale: Optional[float] = None,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = False,
    doc_ids: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Pure-PyTorch step-by-step KDA recurrence (exact mathematical reference).

    Args:
        q: ``[B, T, H, D_k]``
        k: ``[B, T, H, D_k]``
        v: ``[B, T, H, D_v]``
        g: ``[B, T, H, D_k]`` log-decay (``<= 0``)
        beta: ``[B, T, H]`` update gate
        scale: query scaling factor (defaults to ``D_k ** -0.5``)
        initial_state: optional ``[B, H, D_k, D_v]`` state before ``t=0``
        output_final_state: whether to return the final ``[B, H, D_k, D_v]`` state
        doc_ids: optional ``[B, T]`` integer document IDs; state resets to zero
            whenever ``doc_ids[:, t] != doc_ids[:, t - 1]`` or ``doc_ids[:, t] < 0``.
    """
    bsz, seq_len, num_heads, head_k_dim = q.shape
    head_v_dim = v.shape[-1]
    if scale is None:
        scale = head_k_dim ** -0.5

    q_f = q.float() * scale
    k_f = k.float()
    v_f = v.float()
    g_f = g.float()
    beta_f = beta.float()

    if initial_state is None:
        state = q.new_zeros((bsz, num_heads, head_k_dim, head_v_dim), dtype=torch.float32)
    else:
        state = initial_state.float()

    outputs = []
    for t in range(seq_len):
        valid_mask = None
        if doc_ids is not None:
            valid = doc_ids[:, t] >= 0
            valid_mask = valid.to(state.dtype).view(bsz, 1, 1, 1)
            keep = valid if t == 0 else (valid & (doc_ids[:, t] == doc_ids[:, t - 1]))
            state = state * keep.to(state.dtype).view(bsz, 1, 1, 1)

        decay = torch.exp(g_f[:, t]).unsqueeze(-1)  # [B, H, D_k, 1]
        decayed_state = state * decay
        k_t = k_f[:, t]  # [B, H, D_k]
        v_t = v_f[:, t]  # [B, H, D_v]
        beta_t = beta_f[:, t].unsqueeze(-1)  # [B, H, 1]

        # Delta-rule update: S_t = S_tilde + beta_t * k_t * (v_t - k_t^T S_tilde)^T
        pred_v = torch.einsum("bhk,bhkv->bhv", k_t, decayed_state)
        delta_v = (v_t - pred_v) * beta_t
        state = decayed_state + torch.einsum("bhk,bhv->bhkv", k_t, delta_v)

        if valid_mask is not None:
            state = state * valid_mask

        o_t = torch.einsum("bhk,bhkv->bhv", q_f[:, t], state)
        outputs.append(o_t)

    out = (
        torch.stack(outputs, dim=1).to(q.dtype)
        if outputs
        else q.new_empty((bsz, 0, num_heads, head_v_dim))
    )
    return out, (state if output_final_state else None)


def _run_per_doc_span(
    span_fn,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    *,
    initial_state: Optional[torch.Tensor],
    output_final_state: bool,
    doc_ids: torch.Tensor,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Run ``span_fn`` on each contiguous same-document span, resetting state at boundaries."""
    bsz, seq_len, num_heads, head_k_dim = q.shape
    head_v_dim = v.shape[-1]
    batch_outs = []
    batch_states = []
    for b in range(bsz):
        row_docs = doc_ids[b].tolist()
        row_out = q.new_zeros((1, seq_len, num_heads, head_v_dim))
        state = (
            initial_state[b : b + 1].float()
            if initial_state is not None
            else q.new_zeros((1, num_heads, head_k_dim, head_v_dim), dtype=torch.float32)
        )
        start = 0
        while start < seq_len:
            doc = row_docs[start]
            end = start + 1
            while end < seq_len and row_docs[end] == doc:
                end += 1
            if doc < 0:
                state = torch.zeros_like(state)
            else:
                seg_init = state if start == 0 else torch.zeros_like(state)
                seg_out, state = span_fn(
                    q[b : b + 1, start:end],
                    k[b : b + 1, start:end],
                    v[b : b + 1, start:end],
                    g[b : b + 1, start:end],
                    beta[b : b + 1, start:end],
                    seg_init,
                )
                row_out[:, start:end] = seg_out
            start = end
        batch_outs.append(row_out)
        batch_states.append(state)
    out = torch.cat(batch_outs, dim=0)
    final_state = torch.cat(batch_states, dim=0) if output_final_state else None
    return out, final_state


def chunk_kda_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    *,
    scale: Optional[float] = None,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = False,
    chunk_size: int = 64,
    doc_ids: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Chunkwise parallel KDA reference using WY representation.

    Produces mathematically equivalent outputs and final state to
    :func:`recurrent_kda_reference` while parallelizing within each chunk of
    length ``chunk_size`` (splitting at document boundaries when ``doc_ids`` is
    provided).
    """
    bsz, seq_len, num_heads, head_k_dim = q.shape
    head_v_dim = v.shape[-1]
    if scale is None:
        scale = head_k_dim ** -0.5
    if seq_len == 0:
        final = (
            initial_state.float()
            if initial_state is not None
            else q.new_zeros((bsz, num_heads, head_k_dim, head_v_dim), dtype=torch.float32)
        )
        return q.new_empty((bsz, 0, num_heads, head_v_dim)), (final if output_final_state else None)

    if doc_ids is not None:
        return _run_per_doc_span(
            lambda qs, ks, vs, gs, bs, init: _chunk_kda_single_span(
                qs, ks, vs, gs, bs,
                scale=scale,
                initial_state=init,
                output_final_state=True,
                chunk_size=chunk_size,
            ),
            q, k, v, g, beta,
            initial_state=initial_state,
            output_final_state=output_final_state,
            doc_ids=doc_ids,
        )

    return _chunk_kda_single_span(
        q,
        k,
        v,
        g,
        beta,
        scale=scale,
        initial_state=initial_state,
        output_final_state=output_final_state,
        chunk_size=chunk_size,
    )


def _chunk_kda_single_span(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    *,
    scale: float,
    initial_state: Optional[torch.Tensor],
    output_final_state: bool,
    chunk_size: int,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Chunkwise WY KDA on a contiguous single-document span."""
    bsz, seq_len, num_heads, head_k_dim = q.shape
    head_v_dim = v.shape[-1]
    pad_len = (chunk_size - (seq_len % chunk_size)) % chunk_size

    q_f = q.float() * scale
    k_f = k.float()
    v_f = v.float()
    g_f = g.float()
    beta_f = beta.float()

    if pad_len > 0:
        q_f = F.pad(q_f, (0, 0, 0, 0, 0, pad_len))
        k_f = F.pad(k_f, (0, 0, 0, 0, 0, pad_len))
        v_f = F.pad(v_f, (0, 0, 0, 0, 0, pad_len))
        g_f = F.pad(g_f, (0, 0, 0, 0, 0, pad_len))
        beta_f = F.pad(beta_f, (0, 0, 0, pad_len))

    num_chunks = (seq_len + pad_len) // chunk_size
    # Transpose to [B, H, N, C, D]
    q_c = q_f.transpose(1, 2).reshape(bsz, num_heads, num_chunks, chunk_size, head_k_dim)
    k_c = k_f.transpose(1, 2).reshape(bsz, num_heads, num_chunks, chunk_size, head_k_dim)
    v_c = v_f.transpose(1, 2).reshape(bsz, num_heads, num_chunks, chunk_size, head_v_dim)
    g_c = g_f.transpose(1, 2).reshape(bsz, num_heads, num_chunks, chunk_size, head_k_dim)
    b_c = beta_f.transpose(1, 2).reshape(bsz, num_heads, num_chunks, chunk_size)

    # Cumulative log-decay within each chunk: G[i] = sum_{j<=i} g[j]
    g_cum = g_c.cumsum(dim=-2)

    # Build strictly lower-triangular transition matrix A[i, j] = beta[i] * <k[i], exp(G[i] - G[j]) * k[j]> for j < i
    k_beta = k_c * b_c.unsqueeze(-1)
    # Pairwise decay difference G[i] - G[j] (only used for i > j so <= 0)
    tril_strict = torch.tril(
        torch.ones((chunk_size, chunk_size), device=q.device, dtype=torch.bool),
        diagonal=-1,
    )
    tril_incl = torch.tril(
        torch.ones((chunk_size, chunk_size), device=q.device, dtype=torch.bool),
        diagonal=0,
    )
    # [B, H, N, C, C, D_k]: clamp positive upper-triangle diffs before exp for safety
    g_diff = (g_cum.unsqueeze(-2) - g_cum.unsqueeze(-3)).clamp(max=0.0)
    decay_pair = torch.exp(g_diff)

    # A_mat[i, j] = sum_d k_beta[i, d] * decay_pair[i, j, d] * k_c[j, d] for i > j
    a_mat = torch.einsum("bhnid,bhnijd,bhnjd->bhnij", k_beta, decay_pair, k_c)
    a_mat = a_mat.masked_fill(~tril_strict, 0.0)

    # Solve (I + A) * U = beta * V and (I + A) * W = beta * (exp(G) * K)
    eye = torch.eye(chunk_size, device=q.device, dtype=torch.float32)
    lhs = eye + a_mat
    rhs_v = v_c * b_c.unsqueeze(-1)
    rhs_k = k_beta * torch.exp(g_cum)
    u_c = torch.linalg.solve_triangular(lhs, rhs_v, upper=False, unitriangular=True)
    w_c = torch.linalg.solve_triangular(lhs, rhs_k, upper=False, unitriangular=True)

    # Intra-chunk causal attention: M[i, j] = sum_d q[i, d] * exp(G[i, d] - G[j, d]) * k[j, d] for j <= i
    attn_intra = torch.einsum("bhnid,bhnijd,bhnjd->bhnij", q_c, decay_pair, k_c)
    attn_intra = attn_intra.masked_fill(~tril_incl, 0.0)

    state = (
        initial_state.float()
        if initial_state is not None
        else q.new_zeros((bsz, num_heads, head_k_dim, head_v_dim), dtype=torch.float32)
    )
    chunk_outputs = []
    for c in range(num_chunks):
        q_i = q_c[:, :, c]  # [B, H, C, D_k]
        k_i = k_c[:, :, c]  # [B, H, C, D_k]
        g_i = g_cum[:, :, c]  # [B, H, C, D_k]
        u_i = u_c[:, :, c]  # [B, H, C, D_v]
        w_i = w_c[:, :, c]  # [B, H, C, D_k]
        m_i = attn_intra[:, :, c]  # [B, H, C, C]

        # Effective corrected values inside chunk c given incoming state
        v_new = u_i - torch.einsum("bhck,bhkv->bhcv", w_i, state)
        o_inter = torch.einsum("bhck,bhkv->bhcv", q_i * torch.exp(g_i), state)
        o_intra = torch.einsum("bhij,bhjv->bhiv", m_i, v_new)
        chunk_outputs.append(o_inter + o_intra)

        # Update recurrent state at the end of chunk c
        # (for the last chunk with padding, stop at the true last token)
        valid_c = chunk_size - pad_len if (c == num_chunks - 1 and pad_len > 0) else chunk_size
        g_last = g_i[:, :, valid_c - 1, :]  # [B, H, D_k]
        g_to_end = (g_last.unsqueeze(-2) - g_i[:, :, :valid_c, :]).clamp(max=0.0)
        k_weighted = k_i[:, :, :valid_c, :] * torch.exp(g_to_end)
        state = state * torch.exp(g_last).unsqueeze(-1) + torch.einsum(
            "bhck,bhcv->bhkv", k_weighted, v_new[:, :, :valid_c, :]
        )

    out = torch.cat(chunk_outputs, dim=2)[:, :, :seq_len, :].transpose(1, 2).contiguous()
    return out.to(q.dtype), (state if output_final_state else None)


def _doc_isolated_causal_conv1d(
    x: torch.Tensor,
    conv: nn.Conv1d,
    *,
    doc_ids: Optional[torch.Tensor] = None,
    initial_state: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Depthwise causal 1D convolution + SiLU with optional doc reset and state.

    Args:
        x: ``[B, T, D]``
        conv: depthwise ``nn.Conv1d(D, D, kernel_size=K, groups=D)``
        doc_ids: optional ``[B, T]`` document IDs; positions belonging to a
            different document (or ``< 0``) are never mixed into position ``t``.
        initial_state: optional ``[B, K - 1, D]`` pre-conv history before ``t=0``.

    Returns:
        ``(y, final_state)`` where ``y`` is ``[B, T, D]`` after SiLU and
        ``final_state`` is ``[B, K - 1, D]`` (or ``None`` if ``K == 1``).
    """
    bsz, seq_len, dim = x.shape
    kernel_size = conv.kernel_size[0]
    hist_len = kernel_size - 1
    if seq_len == 0:
        return x, initial_state

    if hist_len == 0:
        y = F.conv1d(x.transpose(1, 2), conv.weight, conv.bias, groups=dim).transpose(1, 2)
        if doc_ids is not None:
            y = y * (doc_ids >= 0).unsqueeze(-1).to(y.dtype)
        return F.silu(y), None

    if initial_state is None:
        prefix = x.new_zeros((bsz, hist_len, dim))
    else:
        prefix = initial_state.to(dtype=x.dtype, device=x.device)

    padded = torch.cat([prefix, x], dim=1)  # [B, hist_len + T, D]
    if doc_ids is None:
        y = F.conv1d(padded.transpose(1, 2), conv.weight, conv.bias, groups=dim).transpose(1, 2)
        final_state = padded[:, -hist_len:, :].contiguous()
        return F.silu(y), final_state

    # Document-isolated causal conv: zero out taps whose source position is in
    # a different document or padding (< 0).
    pad_docs = torch.full((bsz, hist_len), -1, device=doc_ids.device, dtype=doc_ids.dtype)
    padded_docs = torch.cat([pad_docs, doc_ids], dim=1)  # [B, hist_len + T]
    cur_docs = doc_ids  # [B, T]
    valid_cur = cur_docs >= 0

    weight = conv.weight.squeeze(1)  # [D, K]
    acc = x.new_zeros((bsz, seq_len, dim), dtype=torch.float32)
    for tap in range(kernel_size):
        # tap=0 is oldest (offset -(K-1)), tap=K-1 is current token (offset 0)
        src = padded[:, tap : tap + seq_len, :]
        src_docs = padded_docs[:, tap : tap + seq_len]
        same_doc = valid_cur & (src_docs == cur_docs)
        acc = acc + (src.float() * same_doc.unsqueeze(-1).float()) * weight[:, tap].float().view(1, 1, dim)
    if conv.bias is not None:
        acc = acc + conv.bias.float().view(1, 1, dim) * valid_cur.unsqueeze(-1).float()
    y = F.silu(acc.to(x.dtype))

    # Build final_state for the last token's document
    last_doc = doc_ids[:, -1:]
    tail_docs = padded_docs[:, -hist_len:]
    tail_keep = (last_doc >= 0) & (tail_docs == last_doc)
    final_state = padded[:, -hist_len:, :] * tail_keep.unsqueeze(-1).to(x.dtype)
    return y, final_state


def _gather_anchor_conv_states(
    raw_context: torch.Tensor,
    anchor_positions: torch.Tensor,
    block_keep_mask: torch.Tensor,
    *,
    hist_len: int,
    doc_ids: Optional[torch.Tensor] = None,
) -> Optional[torch.Tensor]:
    """Gather the ``hist_len`` pre-anchor raw context tokens for each anchor.

    Returns ``[B, N_anchors, hist_len, D]`` with out-of-bounds, cross-document,
    or dropped-anchor entries zeroed out.
    """
    if hist_len <= 0:
        return None
    bsz, ctx_len, dim = raw_context.shape
    num_anchors = anchor_positions.shape[1]
    device = raw_context.device

    offsets = torch.arange(-hist_len, 0, device=device, dtype=torch.long).view(1, 1, hist_len)
    pos = anchor_positions.unsqueeze(-1) + offsets  # [B, N, hist_len]
    in_bounds = (pos >= 0) & (pos < ctx_len)
    safe_pos = pos.clamp(0, max(ctx_len - 1, 0))

    flat_idx = safe_pos.reshape(bsz, num_anchors * hist_len).unsqueeze(-1).expand(-1, -1, dim)
    gathered = torch.gather(raw_context, 1, flat_idx).view(bsz, num_anchors, hist_len, dim)

    valid = in_bounds & block_keep_mask.unsqueeze(-1)
    if doc_ids is not None:
        anchor_docs = torch.gather(doc_ids, 1, anchor_positions.clamp(0, max(ctx_len - 1, 0)))
        src_docs = torch.gather(doc_ids, 1, safe_pos.reshape(bsz, num_anchors * hist_len)).view(
            bsz, num_anchors, hist_len
        )
        valid = valid & (anchor_docs.unsqueeze(-1) >= 0) & (src_docs == anchor_docs.unsqueeze(-1))

    return gathered * valid.unsqueeze(-1).to(gathered.dtype)


class KDAGatedRMSNorm(nn.Module):
    """Gated RMSNorm matching Kimi Linear / ``fla``'s ``FusedRMSNormGated`` (sigmoid)."""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = float(eps)

    def forward(self, x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        x_f = x.float()
        normed = x_f * torch.rsqrt(x_f.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        out = normed * self.weight.float() * torch.sigmoid(gate.float())
        return out.to(input_dtype)


class KDALayer(nn.Module):
    """Kimi Delta Attention layer for DSpark/DFlash dual-source Drafts.

    Parameter layout follows ``fla/layers/kda.py``:
      - ``q_proj``, ``k_proj``, ``v_proj``
      - optional depthwise causal ``q_conv1d``, ``k_conv1d``, ``v_conv1d``
      - low-rank ``f_proj`` (forget gate) + ``A_log`` (FP32) + ``dt_bias`` (FP32)
      - ``b_proj`` (delta-rule beta gate)
      - low-rank ``g_proj`` (output gate) + ``o_norm`` (gated RMSNorm) + ``o_proj``
    """

    def __init__(self, config, layer_idx: int = 0):
        super().__init__()
        self.layer_idx = int(layer_idx)
        self.hidden_size = int(config.hidden_size)
        self.num_heads = int(config.kda_num_heads)
        self.head_dim = int(config.kda_head_dim)
        self.expand_v = float(config.kda_expand_v)
        self.v_head_dim = int(round(self.head_dim * self.expand_v))
        self.key_dim = self.num_heads * self.head_dim
        self.value_dim = self.num_heads * self.v_head_dim
        self.use_short_conv = bool(config.kda_use_short_conv)
        self.conv_size = int(config.kda_conv_size)
        self.conv_bias = bool(config.kda_conv_bias)
        self.allow_neg_eigval = bool(config.kda_allow_neg_eigval)
        self.kda_backend = str(config.kda_backend)
        self.chunk_size = int(config.kda_chunk_size)
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(self.hidden_size, self.key_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.key_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.value_dim, bias=False)

        if self.use_short_conv:
            self.q_conv1d = nn.Conv1d(
                self.key_dim, self.key_dim, kernel_size=self.conv_size,
                groups=self.key_dim, bias=self.conv_bias,
            )
            self.k_conv1d = nn.Conv1d(
                self.key_dim, self.key_dim, kernel_size=self.conv_size,
                groups=self.key_dim, bias=self.conv_bias,
            )
            self.v_conv1d = nn.Conv1d(
                self.value_dim, self.value_dim, kernel_size=self.conv_size,
                groups=self.value_dim, bias=self.conv_bias,
            )
        else:
            self.q_conv1d = None
            self.k_conv1d = None
            self.v_conv1d = None

        self.f_proj = nn.Sequential(
            nn.Linear(self.hidden_size, self.head_dim, bias=False),
            nn.Linear(self.head_dim, self.key_dim, bias=False),
        )
        self.b_proj = nn.Linear(self.hidden_size, self.num_heads, bias=False)
        self.g_proj = nn.Sequential(
            nn.Linear(self.hidden_size, self.head_dim, bias=False),
            nn.Linear(self.head_dim, self.value_dim, bias=False),
        )

        # Keep A_log and dt_bias in FP32 for decay stability, matching FLA KDA.
        self.A_log = nn.Parameter(
            torch.log(torch.empty(self.num_heads, dtype=torch.float32).uniform_(1.0, 16.0))
        )
        self.dt_bias = nn.Parameter(torch.zeros(self.key_dim, dtype=torch.float32))

        self.o_norm = KDAGatedRMSNorm(self.v_head_dim, eps=getattr(config, "rms_norm_eps", 1e-6))
        self.o_proj = nn.Linear(self.value_dim, self.hidden_size, bias=False)

    def keep_gate_precision_fp32(self) -> None:
        """Restore FP32 precision on ``A_log`` and ``dt_bias`` after model casts."""
        with torch.no_grad():
            if self.A_log.dtype != torch.float32:
                self.A_log.data = self.A_log.data.float()
            if self.dt_bias.dtype != torch.float32:
                self.dt_bias.data = self.dt_bias.data.float()

    def _run_kda_kernel(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        *,
        initial_state: Optional[torch.Tensor] = None,
        output_final_state: bool = False,
        doc_ids: Optional[torch.Tensor] = None,
        prefer_recurrent: bool = False,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Dispatch to the configured KDA backend without silent fallback."""
        if self.kda_backend == "fla":
            if not q.is_cuda:
                raise RuntimeError(
                    "kda_backend='fla' requires CUDA tensors; use kda_backend='reference' "
                    "explicitly for CPU execution"
                )
            chunk_kda, fused_recurrent_kda = _import_fla_kda()
            if doc_ids is not None:
                return _run_per_doc_span(
                    lambda qs, ks, vs, gs, bs, init: chunk_kda(
                        q=qs, k=ks, v=vs, g=gs, beta=bs,
                        scale=self.scale,
                        initial_state=init,
                        output_final_state=True,
                        use_qk_l2norm_in_kernel=False,
                    ),
                    q, k, v, g, beta,
                    initial_state=initial_state,
                    output_final_state=output_final_state,
                    doc_ids=doc_ids,
                )
            if prefer_recurrent and q.shape[1] <= 16:
                return fused_recurrent_kda(
                    q=q,
                    k=k,
                    v=v,
                    g=g,
                    beta=beta,
                    scale=self.scale,
                    initial_state=initial_state,
                    output_final_state=output_final_state,
                    use_qk_l2norm_in_kernel=False,
                )
            return chunk_kda(
                q=q,
                k=k,
                v=v,
                g=g,
                beta=beta,
                scale=self.scale,
                initial_state=initial_state,
                output_final_state=output_final_state,
                use_qk_l2norm_in_kernel=False,
            )

        if self.kda_backend == "reference":
            if prefer_recurrent or q.shape[1] <= 16:
                return recurrent_kda_reference(
                    q, k, v, g, beta,
                    scale=self.scale,
                    initial_state=initial_state,
                    output_final_state=output_final_state,
                    doc_ids=doc_ids,
                )
            return chunk_kda_reference(
                q, k, v, g, beta,
                scale=self.scale,
                initial_state=initial_state,
                output_final_state=output_final_state,
                chunk_size=self.chunk_size,
                doc_ids=doc_ids,
            )

        raise ValueError(f"Unsupported kda_backend={self.kda_backend!r}")

    def _project_kv_gates(
        self,
        hidden: torch.Tensor,
        *,
        doc_ids: Optional[torch.Tensor] = None,
        k_conv_state: Optional[torch.Tensor] = None,
        v_conv_state: Optional[torch.Tensor] = None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        Optional[torch.Tensor],
        Optional[torch.Tensor],
    ]:
        """Project K, V, log-decay g, beta, and retain pre-conv K/V for state bridging."""
        bsz, seq_len, _ = hidden.shape
        k_raw = self.k_proj(hidden)
        v_raw = self.v_proj(hidden)
        if self.use_short_conv:
            k_conv, next_k_conv = _doc_isolated_causal_conv1d(
                k_raw, self.k_conv1d, doc_ids=doc_ids, initial_state=k_conv_state
            )
            v_conv, next_v_conv = _doc_isolated_causal_conv1d(
                v_raw, self.v_conv1d, doc_ids=doc_ids, initial_state=v_conv_state
            )
        else:
            k_conv = F.silu(k_raw)
            v_conv = F.silu(v_raw)
            next_k_conv = None
            next_v_conv = None

        k = _l2_normalize(k_conv.view(bsz, seq_len, self.num_heads, self.head_dim))
        v = v_conv.view(bsz, seq_len, self.num_heads, self.v_head_dim)
        f_raw = self.f_proj(hidden).view(bsz, seq_len, self.num_heads, self.head_dim)
        g = kda_gate_reference(f_raw, self.A_log, self.dt_bias)
        beta = torch.sigmoid(self.b_proj(hidden).float())
        if self.allow_neg_eigval:
            beta = beta * 2.0
        return k, v, g, beta, k_raw, v_raw, next_k_conv, next_v_conv

    def _extract_anchor_prefix_states(
        self,
        k_ctx: torch.Tensor,
        v_ctx: torch.Tensor,
        g_ctx: torch.Tensor,
        beta_ctx: torch.Tensor,
        anchor_positions: torch.Tensor,
        block_keep_mask: torch.Tensor,
        *,
        doc_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute exact KDA prefix states ``S_{a - 1}`` for every sampled anchor.

        For standard padded batches (``doc_ids is None``), uses a two-stage
        chunk-boundary + single batched intra-chunk advance so kernel launches
        scale with ``ctx_len / chunk_size`` rather than ``num_anchors``:
          1. Compute chunk-boundary states ``h_chunks[:, c]`` at multiples of
             ``self.chunk_size`` up to ``max_anchor``.
          2. Advance all ``B * N_anchors`` anchors from their chunk boundary to
             ``a - 1`` in one batched KDA call (with ``g=0, beta=0`` no-op
             padding past each anchor's intra-chunk offset).

        When ``doc_ids is not None`` (packed sequences), falls back to
        document-aware segmented extraction.
        """
        if doc_ids is None:
            return self._extract_anchor_prefix_states_chunked(
                k_ctx, v_ctx, g_ctx, beta_ctx, anchor_positions, block_keep_mask
            )
        return self._extract_anchor_prefix_states_segmented(
            k_ctx, v_ctx, g_ctx, beta_ctx, anchor_positions, block_keep_mask, doc_ids=doc_ids
        )

    def _extract_anchor_prefix_states_chunked(
        self,
        k_ctx: torch.Tensor,
        v_ctx: torch.Tensor,
        g_ctx: torch.Tensor,
        beta_ctx: torch.Tensor,
        anchor_positions: torch.Tensor,
        block_keep_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Two-stage chunk-boundary + batched intra-chunk anchor state extraction."""
        bsz, ctx_len, num_heads, head_k_dim = k_ctx.shape
        head_v_dim = v_ctx.shape[-1]
        num_anchors = anchor_positions.shape[1]
        device = k_ctx.device

        safe_anchors = torch.where(
            block_keep_mask,
            anchor_positions.clamp(0, ctx_len),
            torch.zeros_like(anchor_positions),
        )
        max_anchor = int(safe_anchors.max().item()) if safe_anchors.numel() > 0 else 0
        if max_anchor == 0:
            return k_ctx.new_zeros(
                (bsz, num_anchors, num_heads, head_k_dim, head_v_dim), dtype=torch.float32
            )

        chunk_len = self.chunk_size
        num_chunks = max_anchor // chunk_len

        # Stage 1: compute chunk-boundary states h_chunks[:, c] = S_{c * chunk_len - 1}
        running = k_ctx.new_zeros((bsz, num_heads, head_k_dim, head_v_dim), dtype=torch.float32)
        chunk_states = [running]
        for c in range(num_chunks):
            start = c * chunk_len
            stop = start + chunk_len
            k_seg = k_ctx[:, start:stop]
            _, running = self._run_kda_kernel(
                k_seg,
                k_seg,
                v_ctx[:, start:stop],
                g_ctx[:, start:stop],
                beta_ctx[:, start:stop],
                initial_state=running,
                output_final_state=True,
                prefer_recurrent=False,
            )
            chunk_states.append(running)
        h_chunks = torch.stack(chunk_states, dim=1)  # [B, num_chunks + 1, H, D_k, D_v]

        chunk_idx = safe_anchors // chunk_len  # [B, N]
        offsets = safe_anchors % chunk_len  # [B, N]
        gather_c = chunk_idx.view(bsz, num_anchors, 1, 1, 1).expand(
            -1, -1, num_heads, head_k_dim, head_v_dim
        )
        h_init = torch.gather(h_chunks, 1, gather_c)  # [B, N, H, D_k, D_v]

        max_offset = int(offsets.max().item())
        if max_offset == 0:
            return h_init * block_keep_mask.to(h_init.dtype).view(bsz, num_anchors, 1, 1, 1)

        # Stage 2: advance all B * N anchors within their chunk in one batched call.
        # Positions >= offset are masked to g=0, beta=0, k=0, v=0 (exact identity).
        step_idx = torch.arange(max_offset, device=device, dtype=torch.long).view(1, 1, max_offset)
        pos = (chunk_idx * chunk_len).unsqueeze(-1) + step_idx  # [B, N, max_offset]
        valid = block_keep_mask.unsqueeze(-1) & (step_idx < offsets.unsqueeze(-1)) & (pos < ctx_len)
        safe_pos = pos.clamp(0, max(ctx_len - 1, 0)).reshape(bsz, num_anchors * max_offset)

        flat_batch = bsz * num_anchors
        k_sub = torch.gather(
            k_ctx, 1, safe_pos.view(bsz, -1, 1, 1).expand(-1, -1, num_heads, head_k_dim)
        ).view(flat_batch, max_offset, num_heads, head_k_dim)
        v_sub = torch.gather(
            v_ctx, 1, safe_pos.view(bsz, -1, 1, 1).expand(-1, -1, num_heads, head_v_dim)
        ).view(flat_batch, max_offset, num_heads, head_v_dim)
        g_sub = torch.gather(
            g_ctx, 1, safe_pos.view(bsz, -1, 1, 1).expand(-1, -1, num_heads, head_k_dim)
        ).view(flat_batch, max_offset, num_heads, head_k_dim)
        b_sub = torch.gather(
            beta_ctx, 1, safe_pos.view(bsz, -1, 1).expand(-1, -1, num_heads)
        ).view(flat_batch, max_offset, num_heads)

        valid_flat = valid.view(flat_batch, max_offset)
        valid_4d = valid_flat.unsqueeze(-1).unsqueeze(-1)
        valid_3d = valid_flat.unsqueeze(-1)
        k_sub = k_sub.masked_fill(~valid_4d, 0.0)
        v_sub = v_sub.masked_fill(~valid_4d, 0.0)
        g_sub = g_sub.masked_fill(~valid_4d, 0.0)
        b_sub = b_sub.masked_fill(~valid_3d, 0.0)

        _, states_flat = self._run_kda_kernel(
            k_sub,
            k_sub,
            v_sub,
            g_sub,
            b_sub,
            initial_state=h_init.reshape(flat_batch, num_heads, head_k_dim, head_v_dim),
            output_final_state=True,
            prefer_recurrent=(max_offset <= 16),
        )
        states = states_flat.view(bsz, num_anchors, num_heads, head_k_dim, head_v_dim)
        return states * block_keep_mask.to(states.dtype).view(bsz, num_anchors, 1, 1, 1)

    def _extract_anchor_prefix_states_segmented(
        self,
        k_ctx: torch.Tensor,
        v_ctx: torch.Tensor,
        g_ctx: torch.Tensor,
        beta_ctx: torch.Tensor,
        anchor_positions: torch.Tensor,
        block_keep_mask: torch.Tensor,
        *,
        doc_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Document-aware segmented prefix-state extraction for packed sequences."""
        bsz, ctx_len, num_heads, head_k_dim = k_ctx.shape
        head_v_dim = v_ctx.shape[-1]
        zero_state = k_ctx.new_zeros((num_heads, head_k_dim, head_v_dim), dtype=torch.float32)

        batch_anchor_states = []
        for b in range(bsz):
            anchors_list = anchor_positions[b].tolist()
            keep_list = block_keep_mask[b].tolist()
            row_docs = doc_ids[b].tolist() if doc_ids is not None else None

            # Collect sorted unique anchor cut points > 0 that need a non-empty prefix.
            needed = sorted({
                int(pos)
                for pos, keep in zip(anchors_list, keep_list, strict=True)
                if keep and 0 < int(pos) <= ctx_len
            })

            state_by_pos: dict[int, torch.Tensor] = {0: zero_state}
            if needed:
                running = zero_state.unsqueeze(0)  # [1, H, D_k, D_v]
                prev_pos = 0
                prev_doc = row_docs[0] if row_docs is not None and ctx_len > 0 else 0
                for cut in needed:
                    seg_docs = (
                        doc_ids[b : b + 1, prev_pos:cut]
                        if doc_ids is not None
                        else None
                    )
                    # If the first token of this segment starts a new document
                    # relative to prev_pos - 1, reset running state before calling.
                    if (
                        row_docs is not None
                        and prev_pos > 0
                        and row_docs[prev_pos] != row_docs[prev_pos - 1]
                    ):
                        running = torch.zeros_like(running)
                    dummy_q = k_ctx[b : b + 1, prev_pos:cut]
                    _, running = self._run_kda_kernel(
                        dummy_q,
                        k_ctx[b : b + 1, prev_pos:cut],
                        v_ctx[b : b + 1, prev_pos:cut],
                        g_ctx[b : b + 1, prev_pos:cut],
                        beta_ctx[b : b + 1, prev_pos:cut],
                        initial_state=running,
                        output_final_state=True,
                        doc_ids=seg_docs,
                        prefer_recurrent=False,
                    )
                    prev_pos = cut
                    if row_docs is not None:
                        prev_doc = row_docs[cut - 1]
                    # If the anchor token itself starts a brand-new document (or is padding),
                    # its prefix state within its own document is zero!
                    if row_docs is not None and cut < ctx_len:
                        anchor_doc = row_docs[cut]
                        if anchor_doc < 0 or anchor_doc != prev_doc:
                            state_by_pos[cut] = zero_state
                        else:
                            state_by_pos[cut] = running.squeeze(0)
                    else:
                        state_by_pos[cut] = running.squeeze(0)

            row_states = [
                state_by_pos[int(pos)] if (keep and int(pos) in state_by_pos) else zero_state
                for pos, keep in zip(anchors_list, keep_list, strict=True)
            ]
            batch_anchor_states.append(torch.stack(row_states, dim=0))  # [N, H, D_k, D_v]

        return torch.stack(batch_anchor_states, dim=0)  # [B, N, H, D_k, D_v]

    @torch.no_grad()
    def update_prefix_state(
        self,
        context_hidden: torch.Tensor,
        prefix_state: Optional[KDAPrefixState] = None,
    ) -> KDAPrefixState:
        """Advance persistent KDA prefix state over newly verified Target context."""
        bsz, ctx_len, _ = context_hidden.shape
        init_rec = (
            prefix_state.recurrent_state
            if prefix_state is not None
            else context_hidden.new_zeros(
                (bsz, self.num_heads, self.head_dim, self.v_head_dim), dtype=torch.float32
            )
        )
        init_k_conv = prefix_state.k_conv_state if prefix_state is not None else None
        init_v_conv = prefix_state.v_conv_state if prefix_state is not None else None
        if ctx_len == 0:
            return KDAPrefixState(
                recurrent_state=init_rec,
                k_conv_state=init_k_conv,
                v_conv_state=init_v_conv,
            )

        k_ctx, v_ctx, g_ctx, beta_ctx, _, _, next_k_conv, next_v_conv = self._project_kv_gates(
            context_hidden,
            k_conv_state=init_k_conv,
            v_conv_state=init_v_conv,
        )
        _, next_rec = self._run_kda_kernel(
            k_ctx,
            k_ctx,
            v_ctx,
            g_ctx,
            beta_ctx,
            initial_state=init_rec,
            output_final_state=True,
            prefer_recurrent=(ctx_len <= 16),
        )
        return KDAPrefixState(
            recurrent_state=next_rec,
            k_conv_state=next_k_conv,
            v_conv_state=next_v_conv,
        )

    def propose_block(
        self,
        draft_hidden: torch.Tensor,
        prefix_state: Optional[KDAPrefixState],
    ) -> torch.Tensor:
        """Run one Draft block ``[B, block_size, D]`` branched from ``prefix_state``.

        Does not mutate ``prefix_state``.
        """
        bsz, block_size, _ = draft_hidden.shape
        init_rec = (
            prefix_state.recurrent_state
            if prefix_state is not None
            else draft_hidden.new_zeros(
                (bsz, self.num_heads, self.head_dim, self.v_head_dim), dtype=torch.float32
            )
        )
        init_k_conv = prefix_state.k_conv_state if prefix_state is not None else None
        init_v_conv = prefix_state.v_conv_state if prefix_state is not None else None

        q_raw = self.q_proj(draft_hidden)
        if self.use_short_conv:
            q_conv, _ = _doc_isolated_causal_conv1d(q_raw, self.q_conv1d)
        else:
            q_conv = F.silu(q_raw)
        q_draft = _l2_normalize(q_conv.view(bsz, block_size, self.num_heads, self.head_dim))

        k_draft, v_draft, g_draft, beta_draft, _, _, _, _ = self._project_kv_gates(
            draft_hidden,
            k_conv_state=init_k_conv,
            v_conv_state=init_v_conv,
        )
        out, _ = self._run_kda_kernel(
            q_draft,
            k_draft,
            v_draft,
            g_draft,
            beta_draft,
            initial_state=init_rec,
            output_final_state=False,
            prefer_recurrent=True,
        )
        g_out = self.g_proj(draft_hidden).view(bsz, block_size, self.num_heads, self.v_head_dim)
        out = self.o_norm(out, g_out).reshape(bsz, block_size, self.value_dim)
        return self.o_proj(out)

    def forward(
        self,
        draft_hidden: torch.Tensor,
        context_hidden: torch.Tensor,
        metadata: KDAForwardMetadata,
    ) -> tuple[torch.Tensor, dict]:
        """Dual-source DSpark/DFlash KDA forward pass.

        Args:
            draft_hidden: ``[B, N_anchors * block_size, D]``
            context_hidden: ``[B, ctx_len, D]``
            metadata: :class:`KDAForwardMetadata` with anchor positions, keep mask,
                block size, and optional ``context_doc_ids``.
        """
        bsz, draft_len, hidden_size = draft_hidden.shape
        block_size = metadata.block_size
        num_anchors = metadata.anchor_positions.shape[1]
        if draft_len != num_anchors * block_size:
            raise ValueError(
                f"KDA draft_hidden length ({draft_len}) must equal "
                f"num_anchors * block_size ({num_anchors} * {block_size})"
            )

        # 1. Context pass: compute K/V/gates + extract prefix states S_{a - 1}
        k_ctx, v_ctx, g_ctx, beta_ctx, k_raw_ctx, v_raw_ctx, _, _ = self._project_kv_gates(
            context_hidden,
            doc_ids=metadata.context_doc_ids,
        )
        anchor_states = self._extract_anchor_prefix_states(
            k_ctx,
            v_ctx,
            g_ctx,
            beta_ctx,
            metadata.anchor_positions,
            metadata.block_keep_mask,
            doc_ids=metadata.context_doc_ids,
        )  # [B, N, H, D_k, D_v]

        hist_len = self.conv_size - 1 if self.use_short_conv else 0
        k_conv_anchors = _gather_anchor_conv_states(
            k_raw_ctx,
            metadata.anchor_positions,
            metadata.block_keep_mask,
            hist_len=hist_len,
            doc_ids=metadata.context_doc_ids,
        )
        v_conv_anchors = _gather_anchor_conv_states(
            v_raw_ctx,
            metadata.anchor_positions,
            metadata.block_keep_mask,
            hist_len=hist_len,
            doc_ids=metadata.context_doc_ids,
        )

        # 2. Draft block pass: branch from each anchor's prefix state and roll B steps
        flat_batch = bsz * num_anchors
        draft_blocks = draft_hidden.view(flat_batch, block_size, hidden_size)
        prefix_state = KDAPrefixState(
            recurrent_state=anchor_states.reshape(
                flat_batch, self.num_heads, self.head_dim, self.v_head_dim
            ),
            k_conv_state=(
                k_conv_anchors.reshape(flat_batch, hist_len, self.key_dim)
                if k_conv_anchors is not None
                else None
            ),
            v_conv_state=(
                v_conv_anchors.reshape(flat_batch, hist_len, self.value_dim)
                if v_conv_anchors is not None
                else None
            ),
        )
        block_out = self.propose_block(draft_blocks, prefix_state)
        out = block_out.view(bsz, draft_len, hidden_size)

        observation = {
            "layer": self.layer_idx,
            "type": "kimi_delta_attention",
            "variant": "kda",
            "backend": self.kda_backend,
            "num_heads": self.num_heads,
            "head_dim": self.head_dim,
            "value_head_dim": self.v_head_dim,
            "use_short_conv": self.use_short_conv,
            "conv_size": self.conv_size,
            "allow_neg_eigval": self.allow_neg_eigval,
            "draft_block_semantics": "causal_anchor_state_branch",
        }
        return out, observation
