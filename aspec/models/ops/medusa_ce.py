# Copyright 2026 Ant Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Fused full-vocabulary cross-entropy for Medusa heads-only training.

The reference Medusa objective computes, per head, the mean CE over the
selected assistant tokens.  A naive implementation materialises FP32 logits,
runs a separate argmax pass, and lets autograd accumulate one ``[V, H]``
weight gradient per row chunk.  This module keeps the PyTorch objective as
the numerical reference while providing a fused Triton epilogue:

- ``_ce_partials`` streams each vocabulary tile once, capturing tile max,
  sum-exp, argmax and the target logit.
- ``_merge_stats`` merges tile statistics into a per-row log-sum-exp, NLL and
  exact argmax (lowest vocabulary index on ties).
- ``_dlogits`` writes ``(softmax - onehot) * upstream`` in BF16.
- ``_MedusaCE`` recomputes logits chunk-wise in backward when requested. It
  issues one ``dX``/``dW`` GEMM per head when the full BF16 ``dlogits`` buffer
  fits the backward budget, and otherwise accumulates ``dW`` per token chunk
  so long selections stay bounded.

The vocabulary projection itself stays on cuBLAS: on sm_89 a Triton matmul is
not expected to beat cuBLAS, and backward needs the logits again to compute
``dlogits``.  This follows the same division of work as ``tiled_l1``.

Independent implementation; Torch's CE remains the numerical reference and
the fallback for CPU, unsupported devices/dtypes, or unavailable Triton.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.autograd.function import once_differentiable

try:  # Triton is optional for CPU planning/tests.
    import triton
    import triton.language as tl

    _TRITON_IMPORT_ERROR: BaseException | None = None
except Exception as exc:  # pragma: no cover - exercised by CPU fallback tests
    triton = None
    tl = None
    _TRITON_IMPORT_ERROR = exc

_BLOCK_N = 2048
_MAX_ROW_CHUNK = 512
_MIN_ROW_CHUNK = 64
_SAVE_BYTES = 256 * 1024 * 1024
_DLOGITS_BUDGET_BYTES = 2 * 1024**3
_MAX_EXACT_ARGMAX_VOCAB = 2**24
_MIN_TOKENS = 64
_NUM_WARPS_N = 8
_NUM_WARPS_MERGE = 4

_CAPABILITY_CACHE: dict[int, tuple[int, int]] = {}


def _triton_unavailable() -> bool:
    return _TRITON_IMPORT_ERROR is not None or triton is None


def _device_capability(device: torch.device) -> tuple[int, int]:
    index = device.index if device.index is not None else torch.cuda.current_device()
    cached = _CAPABILITY_CACHE.get(index)
    if cached is None:
        cached = torch.cuda.get_device_capability(index)
        _CAPABILITY_CACHE[index] = cached
    return cached


def _is_compiling() -> bool:
    compiler = getattr(torch, "compiler", None)
    is_compiling = getattr(compiler, "is_compiling", None)
    return bool(is_compiling()) if is_compiling is not None else False


def _resolve_row_chunk(vocab: int, requested: int | None) -> int:
    if requested is not None:
        if requested <= 0:
            raise ValueError("Medusa CE row_chunk must be positive")
        return int(requested)
    budget_rows = max(1, _SAVE_BYTES // (2 * vocab))
    chunk = min(_MAX_ROW_CHUNK, max(_MIN_ROW_CHUNK, budget_rows))
    return max(_MIN_ROW_CHUNK, (chunk // _MIN_ROW_CHUNK) * _MIN_ROW_CHUNK)


def _save_logits_for_backward(
    save_logits: bool,
    hidden: torch.Tensor,
    weight: torch.Tensor,
) -> bool:
    """Retain logits only when a backward pass can actually consume them."""
    return bool(
        save_logits
        and torch.is_grad_enabled()
        and (hidden.requires_grad or weight.requires_grad)
    )


def use_fused_medusa_ce(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    *,
    num_rows: int | None = None,
) -> bool:
    """Return whether the fused epilogue can run for this head.

    ``hidden`` may be the full row feature tensor; ``num_rows`` overrides the
    selected-token count when the caller has not gathered it yet.  This is a
    cheap predicate: no allocation, synchronization or host read of values.
    """
    if _triton_unavailable() or _is_compiling():
        return False
    if not hidden.is_cuda or not weight.is_cuda or hidden.device != weight.device:
        return False
    if hidden.dtype is not torch.bfloat16 or weight.dtype is not torch.bfloat16:
        return False
    if hidden.ndim != 2 or weight.ndim != 2:
        return False
    if hidden.shape[1] != weight.shape[1]:
        return False
    if weight.shape[0] == 0:
        return False
    if weight.shape[0] > _MAX_EXACT_ARGMAX_VOCAB:
        return False
    rows = hidden.shape[0] if num_rows is None else int(num_rows)
    if rows < _MIN_TOKENS:
        return False
    if not hidden.is_contiguous() or not weight.is_contiguous():
        return False
    return _device_capability(hidden.device) >= (8, 0)


if not _triton_unavailable():

    @triton.jit(do_not_specialize=["V"])
    def _ce_partials(
        Z,
        Targets,
        Partials,
        V,
        TILES: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        row = tl.program_id(0).to(tl.int64)
        tile = tl.program_id(1).to(tl.int64)
        cols = tile * BLOCK_N + tl.arange(0, BLOCK_N).to(tl.int64)
        mask = cols < V
        z = tl.load(Z + row * V + cols, mask=mask, other=-float("inf")).to(tl.float32)
        tmax = tl.max(z, 0)
        tsum = tl.sum(tl.exp(z - tmax), 0)
        tsum = tl.where(tmax == -float("inf"), 0.0, tsum)
        local = tl.arange(0, BLOCK_N)
        arg_col = tile * BLOCK_N + tl.min(
            tl.where(z == tmax, local, BLOCK_N).to(tl.int32), 0
        ).to(tl.int64)
        target = tl.load(Targets + row).to(tl.int64)
        target_val = tl.max(tl.where(cols == target, z, -float("inf")), 0)
        base = (row * TILES + tile) * 4
        tl.store(Partials + base, tmax)
        tl.store(Partials + base + 1, tsum)
        tl.store(Partials + base + 2, arg_col.to(tl.float32))
        tl.store(Partials + base + 3, target_val)

    @triton.jit
    def _merge_stats(
        Partials,
        Lse,
        Nll,
        Hits,
        Targets,
        TILES: tl.constexpr,
        BLOCK_TILES: tl.constexpr,
        V: tl.constexpr,
    ):
        row = tl.program_id(0).to(tl.int64)
        tile = tl.arange(0, BLOCK_TILES)
        valid = tile < TILES
        base = (row * TILES + tile) * 4
        tmax = tl.load(Partials + base, mask=valid, other=-float("inf"))
        tsum = tl.load(Partials + base + 1, mask=valid, other=0.0)
        arg_col = tl.load(Partials + base + 2, mask=valid, other=float(V))
        target_val = tl.load(Partials + base + 3, mask=valid, other=-float("inf"))

        row_max = tl.max(tmax, 0)
        total = tl.sum(
            tl.where(tmax == -float("inf"), 0.0, tsum * tl.exp(tmax - row_max)), 0
        )
        lse = row_max + tl.log(total)

        best_val = tl.max(tmax, 0)
        best_col = tl.min(tl.where(tmax == best_val, arg_col, float(V)), 0)
        target = tl.load(Targets + row).to(tl.int64)
        tl.device_assert(
            (target >= 0) & (target < V),
            "Medusa CE target index out of vocabulary",
        )
        target_logit = tl.max(target_val, 0)
        nll = lse - target_logit
        hit = (best_col.to(tl.int64) == target).to(tl.float32)
        tl.store(Lse + row, lse)
        tl.store(Nll + row, nll)
        tl.store(Hits + row, hit)

    @triton.jit(do_not_specialize=["V"])
    def _dlogits(
        Z,
        Lse,
        Targets,
        Scale,
        DZ,
        V,
        TILES: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        row = tl.program_id(0).to(tl.int64)
        tile = tl.program_id(1).to(tl.int64)
        cols = tile * BLOCK_N + tl.arange(0, BLOCK_N).to(tl.int64)
        mask = cols < V
        z = tl.load(Z + row * V + cols, mask=mask, other=-float("inf")).to(tl.float32)
        lse = tl.load(Lse + row)
        scale = tl.load(Scale + row)
        target = tl.load(Targets + row).to(tl.int64)
        probs = tl.exp(z - lse)
        probs = tl.where(cols == target, probs - 1.0, probs)
        tl.store(
            DZ + row * V + cols,
            (probs * scale).to(tl.bfloat16),
            mask=mask,
        )

else:  # pragma: no cover - CPU import path
    _ce_partials = _merge_stats = _dlogits = None


class _MedusaCE(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        hidden: torch.Tensor,
        weight: torch.Tensor,
        targets: torch.Tensor,
        row_chunk: int,
        save_logits: bool,
    ):
        rows, vocab = hidden.shape[0], weight.shape[0]
        device = hidden.device
        tiles = triton.cdiv(vocab, _BLOCK_N)
        block_tiles = triton.next_power_of_2(tiles)
        merge_warps = _NUM_WARPS_MERGE if block_tiles >= 32 else 1
        partials = torch.empty((rows, tiles, 4), device=device, dtype=torch.float32)
        lse = torch.empty((rows,), device=device, dtype=torch.float32)
        nll = torch.empty((rows,), device=device, dtype=torch.float32)
        hits = torch.empty((rows,), device=device, dtype=torch.float32)
        saved_logits = (
            torch.empty((rows, vocab), device=device, dtype=torch.bfloat16)
            if save_logits
            else None
        )
        scratch = (
            None
            if save_logits
            else torch.empty(
                (min(row_chunk, rows), vocab), device=device, dtype=torch.bfloat16
            )
        )

        for start in range(0, rows, row_chunk):
            stop = min(start + row_chunk, rows)
            count = stop - start
            if saved_logits is not None:
                logits_chunk = saved_logits[start:stop]
            else:
                logits_chunk = scratch[:count]
            logits_chunk.copy_(F.linear(hidden[start:stop], weight))
            _ce_partials[(count, tiles)](
                logits_chunk,
                targets[start:stop],
                partials[start:stop],
                vocab,
                tiles,
                _BLOCK_N,
                num_warps=_NUM_WARPS_N,
                enable_fp_fusion=False,
            )
            _merge_stats[(count,)](
                partials[start:stop],
                lse[start:stop],
                nll[start:stop],
                hits[start:stop],
                targets[start:stop],
                tiles,
                block_tiles,
                vocab,
                num_warps=merge_warps,
                debug=True,
                enable_fp_fusion=False,
            )

        if saved_logits is not None:
            ctx.save_for_backward(hidden, weight, targets, lse, saved_logits)
        else:
            ctx.save_for_backward(
                hidden,
                weight,
                targets,
                lse,
                torch.empty(0, device=device, dtype=torch.bfloat16),
            )
        ctx.row_chunk = row_chunk
        ctx.save_logits = save_logits
        ctx.mark_non_differentiable(hits)
        return nll, hits

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_nll, grad_hits):
        del grad_hits
        hidden, weight, targets, lse, saved_logits = ctx.saved_tensors
        rows, vocab = hidden.shape[0], weight.shape[0]
        if grad_nll is None:
            return None, None, None, None, None
        needs_dx = ctx.needs_input_grad[0]
        needs_dw = ctx.needs_input_grad[1]
        if not needs_dx and not needs_dw:
            return None, None, None, None, None
        grad_nll = grad_nll.contiguous()
        tiles = triton.cdiv(vocab, _BLOCK_N)
        with torch.no_grad():
            scratch = (
                None
                if ctx.save_logits
                else torch.empty(
                    (min(ctx.row_chunk, rows), vocab),
                    device=hidden.device,
                    dtype=torch.bfloat16,
                )
            )
            if rows * vocab * 2 <= _DLOGITS_BUDGET_BYTES:
                dlogits = torch.empty(
                    (rows, vocab), device=hidden.device, dtype=torch.bfloat16
                )
                for start in range(0, rows, ctx.row_chunk):
                    stop = min(start + ctx.row_chunk, rows)
                    count = stop - start
                    if ctx.save_logits:
                        logits_chunk = saved_logits[start:stop]
                    else:
                        logits_chunk = scratch[:count]
                        logits_chunk.copy_(F.linear(hidden[start:stop], weight))
                    _dlogits[(count, tiles)](
                        logits_chunk,
                        lse[start:stop],
                        targets[start:stop],
                        grad_nll[start:stop],
                        dlogits[start:stop],
                        vocab,
                        tiles,
                        _BLOCK_N,
                        num_warps=_NUM_WARPS_N,
                        enable_fp_fusion=False,
                    )
                dx = torch.mm(dlogits, weight) if needs_dx else None
                dw = torch.mm(dlogits.t(), hidden) if needs_dw else None
            else:
                # Long selections never keep a full [rows, vocab] dlogits
                # buffer; accumulate dW per token chunk instead.
                dx = torch.empty_like(hidden) if needs_dx else None
                dw = torch.zeros_like(weight) if needs_dw else None
                for start in range(0, rows, ctx.row_chunk):
                    stop = min(start + ctx.row_chunk, rows)
                    count = stop - start
                    if ctx.save_logits:
                        logits_chunk = saved_logits[start:stop]
                    else:
                        logits_chunk = scratch[:count]
                        logits_chunk.copy_(F.linear(hidden[start:stop], weight))
                    dlogits_chunk = torch.empty(
                        (count, vocab), device=hidden.device, dtype=torch.bfloat16
                    )
                    _dlogits[(count, tiles)](
                        logits_chunk,
                        lse[start:stop],
                        targets[start:stop],
                        grad_nll[start:stop],
                        dlogits_chunk,
                        vocab,
                        tiles,
                        _BLOCK_N,
                        num_warps=_NUM_WARPS_N,
                        enable_fp_fusion=False,
                    )
                    if needs_dx:
                        torch.mm(dlogits_chunk, weight, out=dx[start:stop])
                    if needs_dw:
                        dw.addmm_(dlogits_chunk.t(), hidden[start:stop])
        return dx, dw, None, None, None


def medusa_head_ce(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    targets: torch.Tensor,
    *,
    row_chunk: int | None = None,
    save_logits: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-token Medusa CE for one head, with the fused epilogue.

    Returns ``(nll, hits)`` where ``nll`` is FP32 and differentiable and
    ``hits`` is FP32 ``{0, 1}`` and detached.  The caller owns the per-head
    mean/count normalization; this op only sums per-token terms.
    """
    if _triton_unavailable():
        raise RuntimeError(
            "fused Medusa CE requires Triton; use the reference loss path"
        ) from _TRITON_IMPORT_ERROR
    if not hidden.is_cuda or not weight.is_cuda:
        raise ValueError("fused Medusa CE requires CUDA tensors")
    if hidden.device != weight.device or targets.device != hidden.device:
        raise ValueError("fused Medusa CE tensors must share one CUDA device")
    if hidden.dtype is not torch.bfloat16 or weight.dtype is not torch.bfloat16:
        raise ValueError("fused Medusa CE requires BF16 hidden states and weights")
    if hidden.ndim != 2 or weight.ndim != 2:
        raise ValueError("fused Medusa CE requires 2-D hidden states and weights")
    if hidden.shape[1] != weight.shape[1]:
        raise ValueError("hidden and weight dimensions disagree")
    if hidden.shape[0] == 0 or weight.shape[0] == 0:
        raise ValueError("fused Medusa CE requires nonempty rows and vocabulary")
    if weight.shape[0] > _MAX_EXACT_ARGMAX_VOCAB:
        raise ValueError(
            "fused Medusa CE supports vocabulary <= 2^24 for exact argmax accuracy"
        )
    if targets.ndim != 1 or targets.numel() != hidden.shape[0]:
        raise ValueError("fused Medusa CE requires one target per row")
    if targets.dtype != torch.int64:
        raise ValueError("fused Medusa CE requires int64 targets")
    if not hidden.is_contiguous() or not weight.is_contiguous():
        raise ValueError("fused Medusa CE requires contiguous tensors")
    if not targets.is_contiguous():
        targets = targets.contiguous()
    save_logits = _save_logits_for_backward(save_logits, hidden, weight)
    return _MedusaCE.apply(
        hidden,
        weight,
        targets,
        _resolve_row_chunk(weight.shape[0], row_chunk),
        save_logits,
    )
