# Absorbed MLA feasibility (negative training result)

This note is the feasibility milestone for [ASpec #19](https://github.com/inclusionAI/ASpec/issues/19).
It checks the absorbed-MLA identity on the DSpark draft geometry and measures whether
the naive absorbed form benefits draft training. It does **not** add a selectable
training path, a fused kernel or an export claim.

## Derivation

The retained materialized path computes, per draft layer,

```text
k_nope = c_kv @ W_UK^T          # [B, T, H, Dn]
v      = c_kv @ W_UV^T          # [B, T, H, Dv]
logits = q_nope @ k_nope^T + q_rope @ k_rope^T
out    = softmax(logits) @ v
```

where `c_kv` is the post-norm latent KV (`kv_lora_rank`), `q_rope`/`k_rope` are the
RoPE side (`qk_rope_head_dim`) and `W_UK`/`W_UV` are the per-head blocks of
`kv_b_proj.weight`. Re-associating both products gives the absorbed form:

```text
q_lat  = q_nope @ W_UK          # [B, S, H, K]
logits = q_lat @ c_kv^T + q_rope @ k_rope^T
out    = (softmax(logits) @ c_kv) @ W_UV^T
```

RoPE still applies to the small side only, and both score terms are added before
the softmax. The identity is exact for the dual-source `cat(context, draft)` KV and
for `q_lora_rank` on/off: those projections happen before this boundary.

## Scope of the prototype

`aspec.models.ops.absorbed_mla.absorbed_mla_attention` implements the dense,
unmasked-or-boolean-mask case in PyTorch. Empty mask rows produce zero output and
zero gradient, matching fused SDPA. `tests/cpu/test_absorbed_mla.py` checks:

- forward parity against an explicitly materialized reference;
- gradients of both query sides, `c_kv`, `k_rope` and `kv_b_proj.weight`;
- dense masks (including broadcast masks that hide keys and empty rows) and
  short sequences;
- parity with the retained `K3DSparkMLAAttention` projections and softmax scale.

FlexAttention/cuDNN masks, packed THD layouts, Direct-CP, `o_proj` fusion and
numerical policies for a fused kernel are not covered.

## Measurement

`tools/performance/bench_absorbed_mla.py` measures forward+backward at the attention
boundary for one layer: BF16, `B=1`, `H=32`, `Dn=128`, `Dr=64`, `Dv=128`,
`K=512`, context 4096, block 7. It compares the retained fused-SDPA path, an
explicit-softmax materialized path (same logits materialization as absorbed) and
the absorbed path. No mask, `o_proj`, Target capture or optimizer step is included.

Measured on 8x H100 PCIe 80GB, driver 580.126.09, PyTorch 2.13.0+cu130, BF16,
warmup 1, 3 runs, median. Peak memory is `torch.cuda.max_memory_allocated()` for
forward+backward. Variants run in rotating order per round; the tool records each
round's order and the per-round timings and peaks in the JSON result.

| anchors | `S` | `T` | materialized SDPA | materialized explicit | absorbed |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 7 | 4103 | 3.38 ms / 0.240 GiB | 4.02 ms / 0.184 GiB | 4.32 ms / 0.101 GiB |
| 8 | 56 | 4152 | 3.26 ms / 0.245 GiB | 3.60 ms / 0.258 GiB | 3.91 ms / 0.225 GiB |
| 128 | 896 | 4992 | 3.02 ms / 0.348 GiB | 9.04 ms / 2.315 GiB | 10.49 ms / 2.799 GiB |
| 512 | 3584 | 7680 | 9.14 ms / 0.668 GiB | 51.50 ms / 13.411 GiB | 63.11 ms / 16.688 GiB |

`S` is `num_anchors * block_size`; `T = context_length + S`.

Reproduce with:

```bash
env -u LD_LIBRARY_PATH CUDA_VISIBLE_DEVICES=1 \
  python -m tools.performance.bench_absorbed_mla \
  --context-length 4096 --num-anchors 512 --block-size 7 \
  --warmup 1 --runs 3 --result /tmp/absorbed-mla.json
```

Keep the raw JSON outside the source tree. `LD_LIBRARY_PATH` was unset only to
select this PyTorch build's bundled cuDNN on the measurement host.

## Result and recommendation

For the default draft anchor budget (512 anchors, 3584 queries), the naive absorbed
path is both slower and larger than the retained fused-SDPA path: explicit logits
dominate memory, and `q_lat` has `kv_lora_rank=512` versus the materialized
`qk_nope_head_dim=128`, so the score matmul does more work. The tiny-query
regimes (1 or 8 anchors) do reduce peak activation memory (0.101 vs 0.240 GiB at
one anchor, 0.225 vs 0.245 GiB at eight), but both are slower.

**Recommendation:** do not integrate the naive PyTorch absorbed form into maintained
draft training, and do not claim a training speedup. A useful follow-up would need
a fused absorbed kernel (score + softmax + latent output) or a workload where the
query/key ratio is genuinely tiny; both are separate from this prototype. The
identity and gradient checks remain useful for validating such a kernel.

## Remaining work

Fused forward/backward kernels, FlexAttention/cuDNN mask support, packing and
Direct-CP layouts, `o_proj` fusion, end-to-end memory/time including target feature
capture, exact resume and export. None of these are established by this note.
