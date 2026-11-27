# Kimi Delta Attention (KDA) Hybrid Drafts

ASpec supports Kimi Delta Attention (`kimi_delta_attention`) layers inside the
DSpark/DFlash Draft family, matching the fine-grained channel-wise decay and
delta-rule recurrence of [Kimi Linear](https://arxiv.org/abs/2510.26692) and
`fla/layers/kda.py`.

## Architecture and visibility semantics

KDA layers are selected per decoder layer through `layer_types`, enabling both
pure-KDA and hybrid KDA + Full Attention Drafts. The shipped
`Ling-3.0-tiny` configuration
(`aspec/config/dspark_ling3_tiny_kda_draft_config.json`) follows
`Ling-3.0-tiny`'s 3:1 ratio with four Draft layers:

```json
"layer_types": [
  "kimi_delta_attention",
  "kimi_delta_attention",
  "kimi_delta_attention",
  "full_attention"
]
```

Each KDA layer preserves exact equivalence between training and stateful
proposal generation:

1. **Context pass (causal, document-isolated).** Projected Target features
   `context_hidden` run causally through the shared `k_proj`, `v_proj`,
   `f_proj`, `b_proj`, optional depthwise short convolutions (`k_conv1d`,
   `v_conv1d`) and KDA state recurrence. Document boundaries (`ctx_doc_ids`)
   reset both the recurrent state and short-convolution history. For each
   sampled anchor `a`, the layer extracts the prefix state `S_{a-1}` and the
   pre-anchor `conv_size - 1` context taps.
2. **Draft block pass (causal anchor-state branch).** Each anchor's block of
   length `B` branches independently from its `(S_{a-1}, k_conv_state,
   v_conv_state)` and rolls `B` causal KDA steps using the same `k/v/f/b`
   weights, followed by gated RMSNorm (`o_norm`) and `o_proj`. Full-attention
   layers in a hybrid schedule keep their standard DSpark bidirectional
   intra-block visibility.
3. **Proposal generation.** Verified Target tokens advance a persistent
   `KDAPrefixState` via `update_prefix_state(...)`; each draft proposal
   branches from that state via `propose_block(...)` without mutating it.

## Backends and runtime requirements

KDA execution backends are explicit and fail closed (no silent fallback):

- `kda_backend: "fla"` (default in the Ling-3.0-tiny recipe): uses
  `fla.ops.kda.chunk_kda` and `fused_recurrent_kda` on CUDA. Install
  `flash-linear-attention` in the Draft virtual environment (e.g.
  `pip install -e ".[kda]"`).
- `kda_backend: "reference"`: pure-PyTorch recurrent and chunkwise WY
  reference implementations for CPU tests and numerical verification.

Gate parameters `A_log` and `dt_bias` remain FP32 across training casts,
exact recovery, native export and strict reloading.

## Training, recovery, export and proposal consumption

```bash
export ASPEC_TARGET_MODEL=/path/to/Ling-3.0-tiny
export ASPEC_TRAIN_DATA=/path/to/ling-train.jsonl
export ASPEC_EVAL_DATA=/path/to/ling-eval.jsonl

aspec --config configs/examples/ling3_tiny_kda_dspark.yaml --dry-run
RAY_ADDRESS=local aspec --config configs/examples/ling3_tiny_kda_dspark.yaml

python -m tools.convert_to_hf --format aspec --verify \
  --input-dir outputs/ling3-tiny-kda-dspark/checkpoints/iter_0000100 \
  --output-dir outputs/ling3-tiny-kda-dspark-asset
```

Exact resume checks the KDA layer schedule and semantics (allowing
`kda_chunk_size` to change as an execution tile bound). Export requires
`--format aspec`; strict loading (`load_native_draft`) and block proposal
(`BlockDraftConsumer`) verify the KDA policy and preserve mixed FP32/BF16
dtypes.

## Support boundaries

- Initial scope: DSpark/DFlash GQA Drafts, padded inputs (`dflash_packing=false`),
  `draft_context_parallel_size=1`, `compile_model=false`, single node.
- Mixing KDA with CSA/QSA/DSA, materialized MLA (`K3DSparkModel`), DFlash2,
  and SGLang `--format sglang` serving export fail closed.
- Full-scale Target-generated training, long-run convergence, and SGLang
  serving acceleration remain outside the bounded functional validation below.

## Bounded validation record

Measured on NVIDIA B300 SXM6 AC with isolated Draft (`torch 2.15.0+cu132`,
`fla 0.6.0`, `triton 3.9.0`, `transformers 5.24.1`) and Target
(`transformers 4.56.2`, `fla 0.6.0`) virtual environments:

- **Unit, numerical, gradient & CUDA `fla` parity**: `tests/cpu/test_draft_kda.py`
  (9/9 passed, including CUDA `fla` vs pure-PyTorch reference parity, chunkwise
  vs recurrent gradient parity, document-boundary isolation, and training vs
  incremental `update_prefix_state` + `propose_block` equivalence) plus
  `tests/test_native_draft_artifact.py`, `tests/cpu/test_draft_dsa.py`, and
  `tests/test_evidence.py` (158/158 passed).
- **Real `Ling-3.0-tiny` bounded training, exact resume & held-out metrics**:
  4-layer 3:1 hybrid KDA Draft (`aspec/config/dspark_ling3_tiny_kda_draft_config.json`,
  `kda_backend="fla"`, `block_size=7`, `num_anchors=64`) trained for 240 steps
  (120 steps → exact resume → 120 steps) on 80 bundled conversations from
  `examples/data/sample_conversations.jsonl` using real `Ling-3.0-tiny`
  features (`target_layer_ids=[1, 6, 11, 16, 21]`):
  - Held-out (16 conversations) before → after: loss `4.0776 → 2.0435`,
    CE `13.5259 → 2.8815`, L1 `1.9908 → 1.3903`, top-1 accuracy
    `0.00% → 25.74%` (`[29.42%, 38.98%, 33.89%, 29.63%, 15.80%, 15.80%, 15.07%]`
    across block positions `0..6`).
  - Native `--format aspec` export, `verify_native_export`, and strict reload
    (`load_native_draft`) passed with FP32 `A_log`/`dt_bias` preserved.
- **Bounded verifier acceptance**: Strictly reloaded KDA native asset via
  `BlockDraftConsumer` verified against real `Ling-3.0-tiny` on 6 held-out
  prompts (32 new tokens each, 192 tokens / 180 rounds) measured mean accepted
  length `1.0667` (off-distribution bundled data without Target-generated
  answers; greedy Target-only vs multi-token verify parity was `0/6` under
  default non-deterministic BF16 verification). Real Target-generated data and
  deterministic verification are required for quality/decode-equality claims.
