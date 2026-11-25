# Absorbed MLA fused kernel (follow-up to #46)

This note is the fused-kernel follow-up to
[absorbed-mla.md](absorbed-mla.md) and [ASpec #19](https://github.com/inclusionAI/ASpec/issues/19).
The dense reference in `aspec.models.ops.absorbed_mla` keeps `[B, H, S, T]` logits
materialized; the fused core keeps the same absorbed identity but never
materializes them:

```text
q_lat  = q_nope @ W_UK            # [B, S, H, K]
logits = q_lat @ c_kv^T + q_rope @ k_rope^T
out    = (softmax(logits) @ c_kv) @ W_UV^T
```

`aspec.models.ops.absorbed_mla_fused` adds a FlashAttention-style Triton
forward (online softmax accumulating the latent output) and a deterministic
backward. It is a prototype: dense masks, packed/THD layouts, Direct-CP and
`o_proj` fusion are not covered, and the naive reference remains the numerical
oracle in `tests/cpu/test_absorbed_mla.py`. The wrapper rejects a non-`None`
`attn_mask` with `NotImplementedError` instead of silently ignoring it.

## Kernel design

- Forward: one program per `[BLOCK_M, H]` query tile walks all keys, computes
  `q_lat @ c_kv^T + q_rope @ k_rope^T`, and maintains `m`/`l`/`out_lat` in FP32.
  Saved state is `out_lat` and `lse` only.
- Backward: two kernels, both recomputing logits from `lse`.
  - query-side: `dq_lat`/`dq_rope` per query tile, no atomics;
  - key/value-side: `dc_kv`/`dk_rope` per key tile and head group. `c_kv` is
    shared across heads, so each program owns `H / GROUP` heads and writes a
    per-group FP32 partial; a deterministic reduction sums the groups. This
    avoids both the serial all-heads loop and atomics.
- The input projections (`q_nope @ W_UK`, `out_lat @ W_UV`) stay in PyTorch so
  their gradients use normal autograd.
- BF16/FP16, contiguous BSHD/BTD inputs. The configured query tile is capped at
  the draft block length to avoid wasted work on short blocks.

Tiles were selected on the measurement host: forward `32x64`, query-side
`16x64`, key/value-side `64x32`, 4 warps; head group 8.

## Measurement

`tools.performance.bench_absorbed_mla` now includes `absorbed_fused` next to
`materialized_sdpa`, `materialized_explicit` and `absorbed`. It measures one
layer forward+backward at the attention boundary: BF16, `B=1`, `H=32`,
`Dn=128`, `Dr=64`, `Dv=128`, `K=512`, block 7, no mask, `o_proj`, Target
capture or optimizer step. Peak memory is `torch.cuda.max_memory_allocated()`.

H100 PCIe 80GB, driver 580.126.09, PyTorch 2.13.0+cu130, Triton 3.7.1, warmup 2,
5 runs, median:

| context | anchors | `S` | `T` | materialized SDPA | absorbed (PyTorch) | absorbed fused |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4096 | 8 | 56 | 4152 | 3.37 ms / 0.245 GiB | 4.48 ms / 0.225 GiB | 3.86 ms / 0.137 GiB |
| 4096 | 128 | 896 | 4992 | 2.60 ms / 0.348 GiB | 10.30 ms / 2.799 GiB | 9.18 ms / 0.370 GiB |
| 4096 | 512 | 3584 | 7680 | 9.20 ms / 0.668 GiB | 63.64 ms / 16.688 GiB | 52.27 ms / 1.245 GiB |
| 16384 | 128 | 896 | 17280 | 6.39 ms / 0.830 GiB | 34.70 ms / 9.387 GiB | 28.59 ms / 0.449 GiB |
| 16384 | 256 | 1792 | 18176 | 11.04 ms / 0.937 GiB | 73.12 ms / 19.621 GiB | 59.30 ms / 0.675 GiB |

## Result

- The fused kernel is consistently faster than the naive PyTorch absorbed form
  at every measured anchor budget (about 1.1–1.25x), and the gap grows with
  `S` because the naive form materializes `[B, H, S, T]` logits.
- Peak memory collapses: 13.4x less than the naive absorbed form at the default
  512-anchor shape (1.245 vs 16.688 GiB) and 29x less at 16k context with 256
  anchors (0.675 vs 19.621 GiB). That is the main result: absorbed execution
  trades extra FLOPs for activation memory, and a fused kernel makes the trade
  usable.
- It is still slower than materialized SDPA (1.15–5.7x). The absorbed form does
  about 3.4x more attention math at this geometry (`K+Dr` scores and `K` latent
  output versus `192`/`128` materialized), and backward recomputes logits in two
  passes. A kernel cannot remove that algorithmic gap on the default workload;
  the memory reduction is the reason to consider it.

Reproduce:

```bash
env -u LD_LIBRARY_PATH CUDA_VISIBLE_DEVICES=1 \
  python -m tools.performance.bench_absorbed_mla \
  --context-length 4096 --num-anchors 512 --block-size 7 \
  --warmup 2 --runs 5
```

Keep raw JSON outside the source tree. `LD_LIBRARY_PATH` was unset only to
select this PyTorch build's bundled cuDNN on the measurement host.

## Remaining work

The first gap versus the dense reference is boolean `attn_mask` support,
including fully masked rows; the fused wrapper currently fails closed on it.
Beyond that: FlexAttention/cuDNN masks, packed THD and Direct-CP layouts,
`o_proj` fusion, shape-based tile dispatch/autotuning, end-to-end memory/time
with Target capture, exact resume and export. None of these are established by
this note.
