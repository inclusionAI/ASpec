# Medusa heads-only Draft

Heads-only Medusa-1 training on frozen Target features, implemented from
[FasterDecoding/Medusa](https://github.com/FasterDecoding/Medusa) at
`e2a5d20c048a9b0a4092e6933c34313687422518` (Apache-2.0; the adapted source is based on
`tatsu-lab/stanford_alpaca` and adapts FastChat code, with those notices retained). The model and
loss landed first; this page describes the current training scope.

## Reference and configuration

- Recipe: `configs/recipes/medusa_qwen3_8b.yaml` (three passes, held-out eval, checkpoints); use
  `aspec recipes plan medusa` / `aspec recipes train medusa --output outputs/qwen3-8b-medusa`.
- Draft config: `aspec/config/medusa_qwen3_8b_draft_config.json` (Qwen3-8B Target, `K=5` heads).
- Example: `configs/examples/medusa_qwen3_8b_4k_tcp.yaml` (one Target GPU + one Draft GPU, 4K,
  SDPA, CP1, non-packing, REPLICATE, Store/TCP transport; no NVLink required). Set
  `ASPEC_TARGET_MODEL` to the Qwen3-8B model ID or directory (or override
  `model.target_model_path`) and run from the repository root so the relative draft-config path
  resolves.
- The heads are the only trainable parameters; the Target is frozen and its post-norm final
  state is the only feature the Draft consumes. At startup the trainer resolves the Target
  config and rejects a draft projection width that differs from its `vocab_size`, including for
  Hub IDs that Plan could not read locally. It applies
  `inference.sglang.extra_args.json_model_override_args` exactly as Plan does.

## Inputs and prediction semantics

- Each head is `medusa_num_layers` residual blocks (`x + SiLU(Linear(x))` with a
  zero-initialized weight and the default bias) followed by a bias-free
  `hidden_size -> vocab_size` projection. The reference repeats one shared residual block per
  head depth; that weight sharing is preserved.
- Head `i` at position `t` is supervised by the token at `t+2+i` (the reference trainer's
  offset). Supervision is assistant-only through the existing loss mask.
- The objective is the sum of per-head mean cross-entropies over the full Target vocabulary.
  Each head's CE is averaged over its supervised assistant tokens within a microbatch, and the
  per-head means are summed with equal weights; gradient accumulation weights microbatches
  equally rather than pooling their token counts. Leave `training.ttt_length` and
  `training.ploss_weights` unset: the Plan derives both from `medusa_num_heads` and rejects
  explicit weights.
- No token embedding, no Draft attention, no teacher logits. The shared-embedding and
  shared-head conventions of native MTP do not apply: every head owns its vocabulary projection.

## Supported scope

The current path fails closed outside the following:

- `training.attention_backend: sdpa`; CP1 (`draft_context_parallel_size: 1`);
  `micro_batch_size: 1`; `dflash_packing: false`; `deterministic_anchor_sampling` is unused.
- `inference.store_last_hidden_states: false` (the single post-norm slot is the feature).
- `fsdp_strategy: REPLICATE`, one node. Anchored training, MTP conditioning and whole-model
  compile are rejected.

## Loss kernel and measured execution

The objective uses a Medusa-specific fused epilogue when the selected rows are
CUDA BF16 and Triton is available; otherwise it keeps the reference path. The
fused path gathers one head's selected rows once, applies the residual block
once, runs one wide vocabulary GEMM per row chunk, and replaces per-chunk FP32
logits, `argmax` and CE with a tiled Triton CE/argmax/dlogits epilogue. It
issues one `dW` GEMM per head when the full BF16 dlogits buffer fits the
backward budget; longer selections accumulate `dW` per token chunk instead of
per 128-token chunk. CPU, unsupported dtypes/devices/shapes,
unavailable Triton, whole-model compile and very short selections use the
reference path; `tests/cpu/test_medusa.py` pins that path unchanged.

`gradient_checkpointing` still bounds the dominant vocabulary-logit memory
through an internal 8 GiB budget; the residual block is applied once and its
smaller `[T, H]` activations are retained. When the all-head BF16 logits fit
the budget (Qwen3-8B `K=5`, 4094 selected tokens and `V=151936` is about
6.2 GiB), the fused path retains them and skips one projection pass in
backward; longer selections recompute. The backward pass keeps one full BF16
`dlogits` buffer only when it fits a 2 GiB budget; otherwise it accumulates
`dX`/`dW` per token chunk. The fused path requires `V <= 2^24` because the
tile argmax column is carried in FP32; larger vocabularies use the reference
path. No configuration field is added.

Measured on one NVIDIA L20 48GB (sm_89, driver 580.126.20, CUDA 13.0, torch
2.13.0+cu130, Triton 3.7.1), Qwen3-8B head shape `K=5`, `H=4096`,
`V=151936`, 4094 selected tokens, loss-only forward+backward, 10 warmup and
30 measured iterations in one process with shared inputs and seeds:

| Cell | Median | p90 | / reference | Peak extra |
| --- | ---: | ---: | ---: | ---: |
| Current reference (128-token chunks) | 2456.74 ms | 2457.03 ms | 1.00× | 4.98 GB |
| Whole-head restructure (diagnostic only) | 913.61 ms | 914.79 ms | 2.69× | 11.36 GB |
| Fused epilogue + one `dW` per head | **761.95 ms** | **762.41 ms** | **3.22×** | **4.98 GB** |

Peak extra is the delta after resident model weights and inputs; it excludes
optimizer state. Accuracy uses the lowest vocabulary index on argmax ties;
PyTorch CUDA argmax tie order is unspecified, so ties are the only metric
difference.

The restructure is a script-only attribution cell, not a shipped path. The
reference and fused loss fingerprints are close but not bitwise equal:
`250777.75` vs `250777.95`, gradient norms `13286.11` vs `13274.47` over the
same 3,195,555,840 gradient elements, and the same accuracy count; the
difference comes from whole-head versus chunked reduction order. The
vocabulary projection remains on cuBLAS: a fully fused Triton matmul was not
adopted because sm_89 matmul quality is not expected to beat cuBLAS and the
backward pass still needs logits for `dlogits`. Forcing recompute on the
4K/`K=5` cell adds one projection pass and was rejected; the retained-logits
budget selects the faster mode when it fits. A scaled
`BF16Optimizer` smoke (`K=5`, `H=1024`, `V=32768`, 512 tokens, 20 steps)
finished with final loss `2.1982379` vs `2.1974640` and gradient norm
`1.92420` vs `1.92306`.

`torch.profiler` attributes the reference's 2.46 s to 785 vectorized
elementwise calls (874.85 ms, the repeated gradient-buffer accumulation),
about 1.36 s of GEMMs and 114 ms of softmax. In the profiled fused cell the
three Triton kernels total about 30 ms, the GEMMs about 0.66 s, and the
785-call accumulation set disappears. Nsight Systems 2023.1 did not capture
CUDA kernel data on this sm_89 device, and Nsight Compute was denied by the
cluster (`ERR_NVGPUCTRPERM`); the profiler table is the available attribution.

These are loss-only Draft-head measurements with real Qwen3-8B dimensions.
They are not end-to-end Target+transport training throughput, convergence,
acceptance or SGLang-serving claims. The full recipe's ~3.2B trainable heads
need about 51 GB of optimizer state and do not fit one L20; the scaled smoke is
an integration check, not a quality result. Reproduce with:

```bash
python -m tools.performance.bench_medusa_loss --mode both \
  --heads 5 --tokens 4094 --hidden 4096 --vocab 151936 \
  --warmup 10 --iters 30 --out medusa-loss.json
```

`--fused-logits auto|save|recompute` exposes the retained-logits decision for
diagnosis; the trainer itself uses `auto`.

## Export and strict reload

Export a trained checkpoint as an ASpec-native asset:

```bash
python -m tools.convert_to_hf \
  --input-dir outputs/qwen3-8b-medusa/checkpoints/iter_0000101 \
  --output-dir outputs/medusa-export --format aspec --verify
```

`--format aspec` is required: the SGLang serving format is rejected because no verified
candidate-tree Medusa loader exists. The artifact contains `config.json`,
`model.safetensors` and `export_manifest.json`; `--verify` reloads it through
`aspec.draft_artifact.load_native_draft` (strict, no dropped tensors) and compares every
reloaded tensor with the values the export wrote.

## Held-out per-head observations

The training eval loop reports `eval/ploss_{i}` and `eval/acc_{i}` when `dataset.eval_data_path`
is set. To report the same observations from an exported asset against a materialized offline
feature dataset:

```bash
python -m tools.evaluate_medusa \
  --draft outputs/medusa-export --dataset /path/to/offline-dataset \
  --split eval --output outputs/medusa-eval.json
```

The report records the artifact and dataset identities, per-head token counts, token-weighted
mean CE and token accuracy, and the sum of the per-head means. These are held-out training
observations, not candidate-tree acceptance.

## Candidate-tree acceptance

`aspec.inference.medusa_consumer` builds the reference Medusa tree (one Target bonus token plus
per-head top-k paths), verifies it, and accounts for accepted tokens. Greedy acceptance and
decode equality can be measured against a Hugging Face Target:

```bash
python -m tools.evaluate_medusa_acceptance \
  --draft outputs/medusa-export --target Qwen/Qwen3-8B \
  --prompts /path/to/prompts.jsonl --output outputs/medusa-acceptance.json
```

The report records `acceptance_length = generated_tokens / verifier_calls` (includes the bonus
token), greedy decode equality against Target-only decoding, and repeat stability, together with
the artifact and tree-config hashes. EOS handling uses the Target's full
`generation_config.eos_token_id` list (falling back to the tokenizer), and generation is capped
by the model position limit and the remaining token budget.

The bundled verifier batches teacher-forced candidate paths through the Target and re-runs the
prefix on every step: it is a reference evaluator, not a serving runtime, and it makes no
throughput or SGLang-compatibility claim. It uses the Target's full `model(...).logits`, so
Target-side output transformations are honored. `--sampling` accepts `greedy` (default),
`typical` (fast deterministic threshold) or `nucleus` (seeded top-p); `decode_equal` is only
reported for greedy, while sampling reports acceptance length and repeat stability with the same
seed.
