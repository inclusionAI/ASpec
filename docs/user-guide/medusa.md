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
the artifact and tree-config hashes. The bundled verifier batches teacher-forced candidate paths
through the Target and re-runs the prefix on every step: it is a reference evaluator, not a
serving runtime, and it makes no throughput or SGLang-compatibility claim. `--sampling` accepts
`greedy` (default), `typical` (fast deterministic threshold) or `nucleus` (seeded top-p);
`decode_equal` is only reported for greedy, while sampling reports acceptance length and
repeat stability with the same seed.
