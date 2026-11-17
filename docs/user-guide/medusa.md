# Medusa heads-only Draft

Heads-only Medusa-1 training on frozen Target features, implemented from
[FasterDecoding/Medusa](https://github.com/FasterDecoding/Medusa) at
`e2a5d20c048a9b0a4092e6933c34313687422518` (Apache-2.0; the adapted source is based on
`tatsu-lab/stanford_alpaca` and adapts FastChat code, with those notices retained). The model and
loss landed first; this page describes the current training scope.

## Reference and configuration

- Draft config: `aspec/config/medusa_qwen3_8b_draft_config.json` (Qwen3-8B Target, `K=5` heads).
- Example: `configs/examples/medusa_qwen3_8b_4k_tcp.yaml` (one Target GPU + one Draft GPU, 4K,
  SDPA, CP1, non-packing, REPLICATE, Store/TCP transport; no NVLink required).
- The heads are the only trainable parameters; the Target is frozen and its post-norm final
  state is the only feature the Draft consumes.

## Inputs and prediction semantics

- Each head is `medusa_num_layers` residual blocks (`x + SiLU(Linear(x))` with a
  zero-initialized weight and the default bias) followed by a bias-free
  `hidden_size -> vocab_size` projection. The reference repeats one shared residual block per
  head depth; that weight sharing is preserved.
- Head `i` at position `t` is supervised by the token at `t+2+i` (the reference trainer's
  offset). Supervision is assistant-only through the existing loss mask.
- The objective is the sum of per-head mean cross-entropies over the full Target vocabulary.
  The trainer applies uniform per-head weights (`training.ttt_length = K`,
  `ploss_weights = [1, 1, ...]`), so the aggregate equals the reference sum.
- No token embedding, no Draft attention, no teacher logits. The shared-embedding and
  shared-head conventions of native MTP do not apply: every head owns its vocabulary projection.

## Supported scope

The current path fails closed outside the following:

- `training.attention_backend: sdpa`; CP1 (`draft_context_parallel_size: 1`);
  `micro_batch_size: 1`; `dflash_packing: false`; `deterministic_anchor_sampling` is unused.
- `inference.store_last_hidden_states: false` (the single post-norm slot is the feature).
- `fsdp_strategy: REPLICATE`, one node. Anchored training, MTP conditioning and whole-model
  compile are rejected.

## Capacity

One head holds `hidden_size^2 + hidden_size * vocab_size` parameters (about 0.64B for Qwen3-8B),
so `K=5` is about 3.2B trainable parameters. Bfloat16 weights, gradients, f32 master weights and
Adam state are roughly 51 GB in REPLICATE, which does not fit one 48 GB device for the listed
example. Size the device count for your hardware, use `fsdp_strategy: SHARD` across independent
Draft GPUs when memory requires it, and measure the actual fit. Export, a maintained recipe,
per-head held-out observations and candidate-tree acceptance are not part of this scope.
