# JetSpec Draft training

This implements [ASpec #15](https://github.com/inclusionAI/ASpec/issues/15) for the
model/loss and initial training path through
`aspec.models.jetspec.JetSpecModel`,
`aspec.models.draft.jetspec.JetSpecConfig`,
`aspec.models.draft.jetspec.JetSpecDraftModel` and
`aspec.training.jetspec_trainer.JetSpecTrainer`. The ASpec training CLI selects this
algorithm through `aspec/config/jetspec_qwen3_8b_draft_config.json`; the
[formal recipe](../../configs/recipes/jetspec_qwen3_8b.yaml) and the
[smoke workflow](../../configs/examples/jetspec_qwen3_8b_4k_tcp.yaml) run one
Target GPU plus two Draft GPUs at 4K (three GPUs total).

## Training selection

- Draft config: `aspec/config/jetspec_qwen3_8b_draft_config.json` (Qwen3-8B,
  `block_size=16`, `is_causal=true`).
- Formal recipe: `configs/recipes/jetspec_qwen3_8b.yaml` (FlexAttention, CP1,
  non-packing, REPLICATE, Store/TCP transport, held-out evaluation and
  `inference.store_last_hidden_states=true`). It is catalogued as `jetspec`; its
  export boundary is native-only.
- Smoke example: `configs/examples/jetspec_qwen3_8b_4k_tcp.yaml` (shorter
  functional run with bundled data).
- The supported path fails closed outside dense full attention, padded inputs,
  `attention_backend=flex_attention`, `draft_context_parallel_size=1` and
  `dflash_packing=false`. MoE, sliding-window layers, MTP conditioning, packing,
  CP>1 and whole-model compile are rejected during Plan compilation or
  configuration. The CPU dense fallback builds the same causal block mask for
  contract tests; the training path requires FlexAttention.
- Exact recovery records the architecture, causal block size, temperature, KL chunk
  size and uniform/decay weighting. Changing any of them requires a weights-only
  new stage.

## Running

Prepare the Target in a separate environment and keep it running on GPU 0:

```bash
aspec patch-sglang
ASPEC_TARGET_MODEL=/path/to/Qwen3-8B CUDA_VISIBLE_DEVICES=0 \
  aspec target --config configs/examples/jetspec_qwen3_8b_4k_tcp.yaml
```

Compile the example, then train with two Draft GPUs (GPUs 1 and 2):

```bash
aspec --config configs/examples/jetspec_qwen3_8b_4k_tcp.yaml --dry-run \
  model.target_model_path=/path/to/Qwen3-8B
CUDA_VISIBLE_DEVICES=1,2 RAY_ADDRESS=local \
  aspec --config configs/examples/jetspec_qwen3_8b_4k_tcp.yaml \
  model.target_model_path=/path/to/Qwen3-8B
```

For exact recovery, stop the first run at step 4 while keeping `lr_total_steps=8`,
then resume that checkpoint into a fresh directory:

```bash
CUDA_VISIBLE_DEVICES=1,2 RAY_ADDRESS=local \
  aspec --config configs/examples/jetspec_qwen3_8b_4k_tcp.yaml \
  model.target_model_path=/path/to/Qwen3-8B \
  training.num_train_steps=4 training.lr_total_steps=8 \
  output_dir=outputs/jetspec_qwen3_8b_4k_tcp_step4

CUDA_VISIBLE_DEVICES=1,2 RAY_ADDRESS=local \
  aspec --config configs/examples/jetspec_qwen3_8b_4k_tcp.yaml \
  model.target_model_path=/path/to/Qwen3-8B \
  training.load_path=outputs/jetspec_qwen3_8b_4k_tcp_step4/checkpoints \
  training.num_train_steps=8 training.lr_total_steps=8 \
  training.continual_training=false \
  output_dir=outputs/jetspec_qwen3_8b_4k_tcp_resumed
```

The example is a short functional smoke-test recipe, not evidence of convergence,
throughput or decoding acceptance. Public CI is CPU-only; GPU end-to-end training
and exact resume still need a hardware run before making quality claims.

## Export and strict reload

Export a trained checkpoint with the native ASpec artifact format:

```bash
aspec export \
  --input-dir outputs/jetspec_qwen3_8b_4k_tcp_resumed/checkpoints/iter_0000009 \
  --output-dir outputs/jetspec-qwen3-8b-asset --format aspec
```

JetSpec deliberately rejects the SGLang artifact format: SGLang 0.5.21 has no
validated JetSpec loader, so a DFlash-style serving export could silently lose the
causal block contract. The native manifest records the JetSpec algorithm, trained
`block_size`, Target feature layers, temperature, KL chunk size and weighting.
`aspec.draft_artifact.load_native_draft` validates the checksums and semantics, then
loads every tensor strictly and preserves dtypes.

### Official checkpoint mapping

The published `JetSpec/jetspec-qwen3-8b` draft head uses
`architectures=["DFlashDraftModel"]`, `model_type="qwen3"` and nests
`causal_head=true`, `mask_token_id` and `target_layer_ids` under `dflash_config`.
In that layout `num_target_layers` is the Target decoder depth (36), not ASpec's
feature count (5). `AutoDraftModelConfig.from_dict` detects `causal_head=true` and
produces the flat `JetSpecDraftModel`/`model_type="jetspec"` config. The published
weight layout matches ASpec's export names; `load_initial_draft_weights` loads the
draft tensors strictly and borrows the missing Target embedding.

### Native proposal component

`aspec.draft_artifact.JetSpecConsumer` loads a native JetSpec asset and proposes one
block. It runs the same causal intra-block attention used in training, excludes the
anchor position from predictions and applies the frozen Target output head. On CPU
it materializes the same dense causal block mask as training; on CUDA it uses the
masked FlexAttention path. Verification and acceptance remain caller-owned.

## Validation status

Registration, Plan compilation, trainer construction, checkpoint topology and the
CPU numerical contracts are covered by `tests/cpu`. The CPU checks cover the
configuration contract, causal mask construction, bounded forward KL and gradients
against independent equations, explicit-mask and packed fail-closed paths,
anchor/loss-mask exclusion, empty supervision and next-token label/teacher
alignment. Native export, strict reload and the proposal consumer have focused CPU
coverage, including causal visibility and anchor-feature isolation. GPU attention
execution, convergence, acceptance length and serving performance are not certified
by those checks.

## Reference and selected method

The implementation targets **JetSpec**, not JetSpec++. The selected reference is
[JetSpec: Breaking the Scaling Ceiling of Speculative Decoding with Parallel Tree
Drafting](https://arxiv.org/abs/2606.18394) (Hu et al., 2026), Section 3.3 and
Appendix D. The upstream [implementation](https://github.com/hao-ai-lab/JetSpec) is
MIT-licensed; this is an independent ASpec implementation of the published
objective, and no JetSpec source file is copied.

JetSpec's Draft head injects fused features from selected frozen Target layers as
contextual K/V state and predicts a whole block in one pass. ASpec already retains
that head as the DFlash block head, including dual-source KV and anchor sampling.
The JetSpec core therefore adapts the retained DFlash path instead of duplicating
it, and adds the two pieces that distinguish JetSpec:

- **causal intra-block attention** (`is_causal=true`), so each Draft position sees
  the verified prefix, its anchor and earlier positions in the same block, but no
  future Draft position and no other block; and
- a **temperature-scaled forward-KL distillation objective** against the frozen
  Target, rather than CE or L1.

The DFlash2 selector, MoE Draft FFNs, sliding-window Draft layers, native-MTP
conditioning and packed/CP execution are outside this core.

FlexAttention is used on CUDA. On the CPU SDPA fallback, the same block-causal
visibility is materialized explicitly so the isolation contract is exercised by
the default tests. Draft configurations that are not causal keep the previous
unmasked fallback behavior.

## Prediction and supervision contract

A block is anchored at an original sequence position `a`. Position `0` keeps the
real token `x[a]` as block context and is excluded from the loss. Position `k >= 1`
predicts `x[a+k]` and distills the Target distribution obtained from Target hidden
state `h[a+k-1]` and the frozen Target LM head. This is the next-token alignment
used by the retained DFlash path.

The loss is

```text
p = softmax(target_logits / T)
q = softmax(draft_logits   / T)
L = T^2 * sum(w * KL(p || q)) / sum(w)
```

where `w` includes block validity, sequence bounds, `loss_mask`, the anchor
exclusion, and optional depth weighting `exp(-(k-1)/gamma)`. The default
`distillation_temperature=1.0` and `loss_decay_gamma=0.0` (uniform weights) match
the main paper configuration; both are configurable. Target logits are produced
under `torch.no_grad()`. The Target hidden states must come from
`inference.store_last_hidden_states=true`, as for the existing L1 distillation path.

`JetSpecModel` fails closed when the Draft config is not a causal
`JetSpecDraftModel`, when CE/L1 alphas are nonzero, when the training block size
disagrees with the Draft config, or when the temperature is not finite and inside
the range where the FP32 `T^2` scale stays finite. `JetSpecConfig` fixes the architecture
to `JetSpecDraftModel`, forces `is_causal=true`, and rejects non-dense FFNs,
sliding-window layers and MTP conditioning.

The full-vocabulary Target projection and FP32 KL intermediates are processed in
bounded token chunks (`kl_chunk_rows`, default 256). Only rows with positive
objective weight are projected, and a custom backward recomputes each chunk
instead of retaining every chunk's log-probability tensors until backward. An
entire masked window keeps an exact differentiable zero without projecting a
teacher row.

The training wrapper always supplies explicit visibility. A direct
`JetSpecDraftModel.forward` call must pass `block_mask`, `draft_attention_mask` or
`direct_cp_metadata`; per-layer lists must cover every Draft layer without `None`
entries. Otherwise it fails closed instead of silently running bidirectional SDPA. The CPU dense-mask path also rejects packed `ctx_doc_ids`,
because the dense helper cannot preserve document isolation.

## Minimal numerical use

Inside the [isolated environment](installation.md):

```python
import torch
from aspec.models.draft.jetspec import JetSpecConfig, JetSpecDraftModel
from aspec.models.jetspec import JetSpecModel

config = JetSpecConfig(
    hidden_size=16, intermediate_size=32, num_hidden_layers=1,
    num_attention_heads=2, num_key_value_heads=1, head_dim=8, vocab_size=32,
    num_target_layers=1, target_hidden_size=16, target_num_hidden_layers=4,
    target_layer_ids=[1], mask_token_id=31, max_position_embeddings=128,
    block_size=5,
)
draft = JetSpecDraftModel(config)
draft.embed_tokens.requires_grad_(False)
model = JetSpecModel(
    draft, block_size=5, num_anchors=2, attention_backend="sdpa",
    ce_loss_alpha=0.0, l1_loss_alpha=0.0, distillation_temperature=1.0,
)

ids = torch.arange(12)[None]
outputs = model(
    input_ids=ids,
    hidden_states_list=torch.randn(1, 12, 16),   # selected Target layers
    loss_mask=torch.ones_like(ids),
    lm_head_weight=torch.randn(32, 16),          # frozen Target head
    last_hidden_states=torch.randn(1, 12, 16),   # frozen Target final state
    injected_anchors=torch.tensor([[3, 5]]),
    injected_keep_mask=torch.ones(1, 2, dtype=torch.bool),
)
outputs[0].backward()
```

The return value follows `DFlashModel.forward`: scalar loss, accuracy,
per-position loss/accuracy/count, extra components and the objective numerator /
denominator. Per-position loss now reports KL rather than CE.

## Boundaries

- Native export, strict reload and a native proposal consumer are supported; the
  published Qwen3-8B config/weight layout maps into ASpec.
- This integration does not provide tree construction, SGLang serving export, a
  real verifier-acceptance run or a same-budget comparison with the official
  implementation. Those remain follow-up work under the issue's expected outcomes.
- The default tests are in `tests/cpu/test_jetspec.py`,
  `tests/cpu/test_jetspec_integration.py`, `tests/cpu/test_jetspec_export.py`,
  `tests/cpu/test_jetspec_consumer.py` and `tests/cpu/test_jetspec_official.py`.
