# Copyright 2026 Ant Group Co., Ltd.
# Copyright (c) 2025, Wentao Guo, Ted Zadouri, Tri Dao (QuACK tile layout).
# SPDX-License-Identifier: Apache-2.0
"""Hopper/Blackwell joint hard CE and overlap TV. Reduction layout adapted from QuACK 0.6.5
softmax; objective and BF16 gradient addition reimplement Speculators b9c51bc.
"""

import math
from functools import lru_cache

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import BFloat16, Float32, Int32, Int64, const_expr
from quack import copy_utils
from quack.compile_utils import make_fake_tensor
from quack.epilogue.frontend import F2, gemm_epilogue
from quack.reduce import row_reduce
from quack.reduction_base import ReductionBase


@cute.jit
def _lane_inputs(d, t, indices, tile, row, part, copy, vocab: cutlass.Constexpr):
    teacher_row = row
    if const_expr(indices is not None):
        teacher_row = Int32(indices[row])
    ids = cute.make_identity_tensor(d.shape)
    dx = copy.partition_S(cute.local_tile(d, tile, (row, part)))
    tx = copy.partition_S(cute.local_tile(t, tile, (teacher_row, part)))
    ix = copy.partition_S(cute.local_tile(ids, tile, (row, part)))
    rd, rt = [cute.make_rmem_tensor_like(a) for a in (dx, tx)]
    rd.fill(-BFloat16.inf)
    rt.fill(-BFloat16.inf)
    pred = copy_utils.predicate_k(ix, limit=vocab)
    copy_utils.copy(dx, rd, pred=pred)
    copy_utils.copy(tx, rt, pred=pred)
    columns = cute.make_rmem_tensor(rd.shape, Int32)
    for i in cutlass.range_constexpr(cute.size(columns)):
        columns[i] = ix[i][1]
    return rd.load().to(Float32), rt.load().to(Float32), columns.load(), pred


class _Grad:
    def __init__(self, vocab):
        self.vocab = vocab

    @cute.jit
    def __call__(
        self,
        d: cute.Tensor,
        t: cute.Tensor,
        indices,
        stats: cute.Tensor,
        labels: cute.Tensor,
        gc: cute.Tensor,
        gt: cute.Tensor,
        out: cute.Tensor,
        stream: cuda.CUstream,
    ):
        vec = math.gcd(self.vocab, 8)
        copy = copy_utils.tiled_copy_2d(BFloat16, 128, 128, vec)
        tile = (1, 128 * vec)
        self.kernel(d, t, indices, stats, labels, gc, gt, out, copy, tile).launch(
            grid=(d.shape[0], cute.ceil_div(self.vocab, tile[1]), 1),
            block=(128, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        d: cute.Tensor,
        t: cute.Tensor,
        indices,
        stats: cute.Tensor,
        labels: cute.Tensor,
        gc: cute.Tensor,
        gt: cute.Tensor,
        out: cute.Tensor,
        copy: cute.TiledCopy,
        tile: cute.Shape,
    ):
        row, part, _ = cute.arch.block_idx()
        thr = copy.get_slice(cute.arch.thread_idx()[0])
        x, y, col, pred = _lane_inputs(d, t, indices, tile, row, part, thr, self.vocab)
        q = cute.math.exp(x - stats[row, 0], fastmath=True)
        p = cute.math.exp(y - stats[row, 1], fastmath=True)
        dc = gc[row] * (q - (col == Int32(labels[row])).to(Float32))
        dt = gt[row] * (q * (stats[row, 2] - (q <= p).to(Float32)))
        # Upstream writes two BF16 gradients before autograd adds them.
        dest = thr.partition_D(cute.local_tile(out, tile, (row, part)))
        reg = cute.make_rmem_tensor_like(dest)
        reg.store((dc.to(BFloat16).to(Float32) + dt.to(BFloat16).to(Float32)).to(BFloat16))
        copy_utils.copy(reg, dest, pred=pred)


_TILE = 2048


class _Overlap(ReductionBase):
    """Stream independent vocabulary tiles, then combine FP32 row partials."""

    def __init__(self, vocab, merge):
        super().__init__(BFloat16, vocab, stage=4)
        self.merge_partials = merge
        self.cluster_n = 1
        self.width = _TILE
        self.tiles = (vocab + self.width - 1) // self.width

    @cute.jit
    def __call__(
        self,
        d: cute.Tensor,
        t: cute.Tensor,
        idx,
        parts: cute.Tensor,
        norm: cute.Tensor,
        stats: cute.Tensor,
        labels: cute.Tensor,
        ce: cute.Tensor,
        tv: cute.Tensor,
        correct: cute.Tensor,
        stream: cuda.CUstream,
    ):
        if const_expr(not self.merge_partials):
            copy = copy_utils.tiled_copy_2d(BFloat16, 128, 128, math.gcd(self.N, 8))
            self.scan(d, t, idx, parts, norm, copy).launch(
                grid=(d.shape[0] * self.tiles, 1, 1), block=(128, 1, 1), stream=stream
            )
        else:
            self.merge(d, parts, norm, stats, labels, ce, tv, correct).launch(
                grid=(d.shape[0], 1, 1), block=(128, 1, 1), stream=stream
            )

    @cute.kernel
    def scan(
        self,
        d: cute.Tensor,
        t: cute.Tensor,
        idx,
        parts: cute.Tensor,
        norm: cute.Tensor,
        copy: cute.TiledCopy,
    ):
        tid = cute.arch.thread_idx()[0]
        bid = cute.arch.block_idx()[0]
        row, part = bid // self.tiles, bid % self.tiles
        x, y, c, _ = _lane_inputs(
            d, t, idx, (1, self.width), row, part, copy.get_slice(tid), self.N
        )
        smem = cutlass.utils.SmemAllocator()
        buf, _ = self._allocate_reduction_buffer_and_mbar(smem, copy.layout_tv_tiled)
        add, minimum = cute.ReductionOp.ADD, cute.ReductionOp.MIN
        q = cute.math.exp(x - norm[row, 0], fastmath=True)
        p = cute.math.exp(y - norm[row, 1], fastmath=True)
        take = q <= p
        ov, mass = row_reduce(
            (cute.where(take, q, p), cute.where(take, q, 0.0)),
            add,
            128,
            (buf[None, None, 0], buf[None, None, 1]),
        )
        i, j = row_reduce(
            (
                cute.where(x == norm[row, 2], c.to(Float32), Float32(self.N)),
                cute.where(y == norm[row, 3], c.to(Float32), Float32(self.N)),
            ),
            minimum,
            128,
            (buf[None, None, 2], buf[None, None, 3]),
            init_val=Float32(self.N),
        )
        if tid == 0:
            parts[row, 0, part] = ov
            parts[row, 1, part] = mass
            parts[row, 2, part] = i
            parts[row, 3, part] = j

    @cute.kernel
    def merge(
        self,
        d: cute.Tensor,
        parts: cute.Tensor,
        norm: cute.Tensor,
        stats: cute.Tensor,
        labels: cute.Tensor,
        ce: cute.Tensor,
        tv: cute.Tensor,
        correct: cute.Tensor,
    ):
        row = cute.arch.block_idx()[0]
        tid = cute.arch.thread_idx()[0]
        smem = cutlass.utils.SmemAllocator()
        layout = cute.make_layout((1, (4, 1), 4))
        buf = smem.allocate_tensor(Float32, layout)
        a, b, c, e = Float32(0), Float32(0), Float32(0), Float32(0)
        c, e = Float32(self.N), Float32(self.N)
        if tid < self.tiles:
            a, b, c, e = (
                parts[row, 0, tid],
                parts[row, 1, tid],
                parts[row, 2, tid],
                parts[row, 3, tid],
            )
        ov, mass = row_reduce(
            (a, b), cute.ReductionOp.ADD, 128, (buf[None, None, 0], buf[None, None, 1])
        )
        i, j = row_reduce(
            (c, e),
            cute.ReductionOp.MIN,
            128,
            (buf[None, None, 2], buf[None, None, 3]),
            init_val=Float32(self.N),
        )
        if tid == 0:
            label = min(Int32(j), self.N - 1)
            labels[row] = Int64(label)
            correct[row] = i == j
            ce[row] = norm[row, 0] - Float32(d[row, label])
            tv[row] = 1.0 - ov
            stats[row, 0] = norm[row, 0]
            stats[row, 1] = norm[row, 1]
            stats[row, 2] = mass


@lru_cache(maxsize=32)
def _overlap_compile(vocab, indexed):
    rows = cute.sym_int()
    x = make_fake_tensor(BFloat16, (rows, vocab), math.gcd(vocab, 8))
    y = make_fake_tensor(BFloat16, (cute.sym_int(), vocab), math.gcd(vocab, 8)) if indexed else x
    ids = make_fake_tensor(Int64, (rows,))
    p = make_fake_tensor(Float32, (rows, 4, (vocab + _TILE - 1) // _TILE))
    n = make_fake_tensor(Float32, (rows, 4))
    f = make_fake_tensor(Float32, (rows,))
    s = make_fake_tensor(Float32, (rows, 3))
    correct = make_fake_tensor(cutlass.Boolean, (rows,))
    return [
        cute.compile(
            _Overlap(vocab, merge),
            x,
            y,
            ids if indexed else None,
            p,
            n,
            s,
            ids,
            f,
            f,
            correct,
            cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
            options="--enable-tvm-ffi",
        )
        for merge in (False, True)
    ]


class _PinnedNorm(ReductionBase):
    """Partition the upstream logical CTA across a thread-block cluster. Each logical
    lane keeps the same FP32 summation order, including the online tail tile."""

    def __init__(self, vocab):
        super().__init__(BFloat16, vocab, stage=4)
        block = min(131072, 1 << (vocab - 1).bit_length())
        warps = 32 if block >= 32768 else 16 if block >= 8192 else 8 if block >= 2048 else 4
        self.cluster_n = max(1, warps // 4)
        self.block = block
        self.vec = min(math.gcd(8, vocab), max(1, block // (32 * warps)))
        self.passes = max(1, block // (128 * self.cluster_n * self.vec))
        self.groups = (vocab + block - 1) // block

    def _threads_per_row(self):
        return 128

    @cute.jit
    def __call__(
        self, d: cute.Tensor, t: cute.Tensor, idx, norm: cute.Tensor, stream: cuda.CUstream
    ):
        cp = copy_utils.tiled_copy_2d(BFloat16, 128, 128, self.vec)
        cluster = (1, self.cluster_n, 1) if const_expr(self.cluster_n > 1) else None
        self.kernel(d, None, norm, cp, 0).launch(
            grid=(d.shape[0], self.cluster_n, 1), block=(128, 1, 1), cluster=cluster, stream=stream
        )
        self.kernel(t, idx, norm, cp, 1).launch(
            grid=(d.shape[0], self.cluster_n, 1), block=(128, 1, 1), cluster=cluster, stream=stream
        )

    @cute.kernel
    def kernel(
        self, d: cute.Tensor, idx, norm: cute.Tensor, cp: cute.TiledCopy, side: cutlass.Constexpr
    ):
        tid = cute.arch.thread_idx()[0]
        row, part, _ = cute.arch.block_idx()
        smem = cutlass.utils.SmemAllocator()
        buf, mb = self._allocate_reduction_buffer_and_mbar(smem, cp.layout_tv_tiled)
        self._initialize_cluster(tid, mb, 4)
        yrow = row
        if const_expr(idx is not None):
            yrow = Int32(idx[row])
        thr = cp.get_slice(tid)
        gmax = -Float32.inf
        denom = Float32(0)
        for group in cutlass.range_constexpr(self.groups):
            tile = (1, 128 * self.vec)
            # Omit padded tail columns to preserve the reference FP32 summation order.
            passes = min(
                self.passes,
                (self.N - group * self.block + 128 * self.cluster_n * self.vec - 1)
                // (128 * self.cluster_n * self.vec),
            )
            cache = cute.make_rmem_tensor((self.vec, passes), Float32)
            for k in cutlass.range_constexpr(passes):
                shard = (group * self.passes + k) * self.cluster_n + part
                dx = thr.partition_S(cute.local_tile(d, tile, (yrow, shard)))
                ix = thr.partition_S(
                    cute.local_tile(cute.make_identity_tensor(d.shape), tile, (row, shard))
                )
                pred = copy_utils.predicate_k(ix, limit=self.N)
                reg = cute.make_rmem_tensor_like(dx)
                reg.fill(-BFloat16.inf)
                copy_utils.copy(dx, reg, pred=pred)
                for j in cutlass.range_constexpr(self.vec):
                    cache[j, k] = Float32(reg[j])
            m = row_reduce(
                cache.load(),
                cute.ReductionOp.MAX,
                128,
                buf[None, None, 2 * group],
                mb + 2 * group if const_expr(self.cluster_n > 1) else None,
                init_val=-Float32.inf,
                hook_fn=cute.arch.cluster_wait
                if const_expr(self.cluster_n > 1 and group == 0)
                else None,
            )
            newmax = cute.arch.fmax(gmax, m)
            ex = cute.make_rmem_tensor_like(cache)
            ex.store(cute.math.exp(cache.load() - newmax, fastmath=True))
            total = Float32(0)
            for k in cutlass.range_constexpr(passes):
                for j in cutlass.range_constexpr(self.vec):
                    total = total + ex[j, k]
            z = row_reduce(
                total,
                cute.ReductionOp.ADD,
                128,
                buf[None, None, 2 * group + 1],
                mb + 2 * group + 1 if const_expr(self.cluster_n > 1) else None,
            )
            if const_expr(group == 0):
                denom = z
            else:
                denom = cute.math.fma(denom, cute.math.exp(gmax - newmax, fastmath=True), z)
            gmax = newmax
        if (tid == 0) & (part == 0):
            norm[row, side] = gmax + cute.math.log(denom)
            norm[row, side + 2] = gmax


@lru_cache(maxsize=32)
def _norm_compile(vocab, indexed):
    rows = cute.sym_int()
    x = make_fake_tensor(BFloat16, (rows, vocab), math.gcd(vocab, 8))
    y = make_fake_tensor(BFloat16, (cute.sym_int(), vocab), math.gcd(vocab, 8)) if indexed else x
    ids = make_fake_tensor(Int64, (rows,)) if indexed else None
    n = make_fake_tensor(Float32, (rows, 4))
    return cute.compile(
        _PinnedNorm(vocab),
        x,
        y,
        ids,
        n,
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi",
    )


@lru_cache(maxsize=32)
def _grad_compile(vocab, indexed):
    rows = cute.sym_int()
    x = make_fake_tensor(BFloat16, (rows, vocab), math.gcd(vocab, 8))
    y = make_fake_tensor(BFloat16, (cute.sym_int(), vocab), math.gcd(vocab, 8)) if indexed else x
    ids = make_fake_tensor(Int64, (rows,))
    f = make_fake_tensor(Float32, (rows,))
    stats = make_fake_tensor(Float32, (rows, 3))
    return cute.compile(
        _Grad(vocab),
        x,
        y,
        ids if indexed else None,
        stats,
        ids,
        f,
        f,
        x,
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi",
    )


class _TokenLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, teacher, indices):
        vocab = logits.shape[-1]
        x = logits.view(-1, vocab)
        y = teacher.view(-1, vocab)
        rows = x.shape[0]
        ce = x.new_empty(rows, dtype=torch.float32)
        tv = torch.empty_like(ce)
        ids = x.new_empty(rows, dtype=torch.int64)
        stats = x.new_empty((rows, 3), dtype=torch.float32)
        correct = x.new_empty(rows, dtype=torch.bool)
        partial = stats.new_empty((rows, 4, (vocab + _TILE - 1) // _TILE))
        norm = stats.new_empty((rows, 4))
        _norm_compile(vocab, indices is not None)(x, y, indices, norm)
        for kernel in _overlap_compile(vocab, indices is not None):
            kernel(x, y, indices, partial, norm, stats, ids, ce, tv, correct)
        ctx.save_for_backward(logits, teacher, indices, stats, ids)
        ctx.set_materialize_grads(False)
        shape = logits.shape[:-1]
        labels = ids.view(shape)
        correct = correct.view(shape)
        ctx.mark_non_differentiable(labels, correct)
        return ce.view(shape), tv.view(shape), labels, correct

    @staticmethod
    def backward(ctx, grad_ce, grad_tv, _labels, _correct):
        logits, teacher, indices, stats, ids = ctx.saved_tensors
        x = logits.view(-1, logits.shape[-1])
        dx = torch.empty_like(x)

        def upstream(g):
            return stats.new_zeros(x.shape[0]) if g is None else g.reshape(-1).contiguous()

        _grad_compile(x.shape[-1], indices is not None)(
            x,
            teacher.view(-1, x.shape[-1]),
            indices,
            stats,
            ids,
            upstream(grad_ce),
            upstream(grad_tv),
            dx,
        )
        return dx.view_as(logits), None, None


def speculators_cutedsl_loss(logits, teacher, indices=None):
    return _TokenLoss.apply(logits, teacher, indices)


@gemm_epilogue()
def _markov_residual(acc, c):
    # Match BF16 linear -> BF16 add. A standard FP32 GEMM residual epilogue
    # drops the intermediate rounding and can change the TV subgradient.
    if const_expr(isinstance(acc, F2)):
        rounded = F2(Float32(BFloat16(acc.lo)), Float32(BFloat16(acc.hi)))
    else:
        rounded = Float32(BFloat16(acc))
    return {"D": rounded + c}


class _MarkovProjection(torch.autograd.Function):
    @staticmethod
    def forward(ctx, base, embedding, weight):
        a = embedding.reshape(-1, embedding.shape[-1])
        out = _markov_residual(a, weight.T, base.view(-1, base.shape[-1]), tuned=False)["D"]
        ctx.save_for_backward(a, weight)
        ctx.embedding_shape = embedding.shape
        return out.view_as(base)

    @staticmethod
    def backward(ctx, grad):
        a, weight = ctx.saved_tensors
        g = grad.reshape(-1, grad.shape[-1])
        return (
            grad,
            (g @ weight).view(ctx.embedding_shape) if ctx.needs_input_grad[1] else None,
            g.T @ a if ctx.needs_input_grad[2] else None,
        )


markov_residual_projection = _MarkovProjection.apply
