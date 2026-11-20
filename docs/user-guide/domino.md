# Domino Draft training

This implements [ASpec #38](https://github.com/inclusionAI/ASpec/issues/38) through
`aspec.models.draft.domino.DominoDraftModel` and
`aspec.models.domino.DominoModel`: a block-parallel DFlash backbone with a
GRU-conditioned causal correction head. The ASpec training CLI now selects this
algorithm through `aspec/config/domino_qwen3_8b_draft_config.json`. The
[formal recipe](../../configs/recipes/domino_qwen3_8b.yaml) and the
[smoke workflow](../../configs/examples/domino_qwen3_8b_4k_tcp.yaml) both run one
Target GPU plus two Draft GPUs at 4K (three GPUs total).

## Training selection

- Draft config: `aspec/config/domino_qwen3_8b_draft_config.json` (Qwen3-8B,
  `block_size=16`, `shift_label=true`, `pure_draft_prefix_len=1`).
- Formal recipe: `configs/recipes/domino_qwen3_8b.yaml` (FlexAttention, CP1,
  non-packing, REPLICATE, Store/TCP transport, held-out eval and the linear
  `lambda_base` curriculum). It is catalogued as `domino`; SGLang and native export
  are both validated. `domino_qwen3_8b_4k_tcp.yaml` remains a shorter smoke test.
- `training.domino_lambda_base` is the constant fallback. Set both
  `training.domino_lambda_base_start` and `training.domino_lambda_base_decay_ratio`
  to apply the reference linear schedule. Exact recovery records the constant,
  schedule, effective curriculum horizon, label layout and correction-head
  dimensions.
- Set `dataset.eval_data_path` and `training.domino_eval_backbone=true` to report
  held-out corrected and backbone-only observations per prediction position.
- The supported path fails closed outside dense full attention, padded inputs,
  `attention_backend=flex_attention`, `draft_context_parallel_size=1` and
  `dflash_packing=false`. The CUDA FlexAttention mask and the CPU dense fallback
  share the same block visibility; the CPU mask is built for contract tests only. MoE, sparse attention, MTP conditioning, packing, CP>1
  and whole-model compile are rejected during Plan compilation or configuration.

## Running

Prepare the Target in a separate environment and keep it running on GPU 0:

```bash
aspec patch-sglang
CUDA_VISIBLE_DEVICES=0 aspec target \
  --config configs/examples/domino_qwen3_8b_4k_tcp.yaml \
  model.target_model_path=/path/to/Qwen3-8B
```

For held-out data and the curriculum, set `ASPEC_TRAIN_DATA` and `ASPEC_EVAL_DATA`
and use `configs/recipes/domino_qwen3_8b.yaml`. Compile the smoke example,
then train with two Draft GPUs (GPUs 1 and 2):

```bash
aspec --config configs/examples/domino_qwen3_8b_4k_tcp.yaml --dry-run \
  model.target_model_path=/path/to/Qwen3-8B
CUDA_VISIBLE_DEVICES=1,2 RAY_ADDRESS=local \
  aspec --config configs/examples/domino_qwen3_8b_4k_tcp.yaml \
  model.target_model_path=/path/to/Qwen3-8B
```

For exact recovery, stop the first run at step 4 while keeping `lr_total_steps=8`,
then resume that checkpoint into a fresh directory:

```bash
CUDA_VISIBLE_DEVICES=1,2 RAY_ADDRESS=local \
  aspec --config configs/examples/domino_qwen3_8b_4k_tcp.yaml \
  model.target_model_path=/path/to/Qwen3-8B \
  training.num_train_steps=4 training.lr_total_steps=8 \
  output_dir=outputs/domino_qwen3_8b_4k_tcp_step4

CUDA_VISIBLE_DEVICES=1,2 RAY_ADDRESS=local \
  aspec --config configs/examples/domino_qwen3_8b_4k_tcp.yaml \
  model.target_model_path=/path/to/Qwen3-8B \
  training.load_path=outputs/domino_qwen3_8b_4k_tcp_step4/checkpoints \
  training.num_train_steps=8 training.lr_total_steps=8 \
  training.continual_training=false \
  output_dir=outputs/domino_qwen3_8b_4k_tcp_resumed
```

The example is a short functional smoke test, not convergence, throughput or
decoding-acceptance evidence. A GPU run with one Target plus two Draft GPUs has been
reported for training and exact resume; it does not establish model quality. Supply
licensed data, a full learning-rate horizon and the recipe for a real run.

## Export and strict reload

Export a trained checkpoint either for SGLang serving (default) or as a native ASpec
artifact:

```bash
aspec export --input-dir outputs/domino-qwen3-8b/checkpoints/iter_0000009 \
  --output-dir outputs/domino-qwen3-8b-sglang

aspec export --input-dir outputs/domino-qwen3-8b/checkpoints/iter_0000009 \
  --output-dir outputs/domino-qwen3-8b-asset --format aspec
```

SGLang export maps Domino to the published layout
(`architectures=["DFlashDraftModel"]`, `model_type="qwen3"`, nested
`dflash_config.projector_type="domino"`) and preserves the trained `block_size`,
`shift_label`, `pure_draft_prefix_len=1`, `emb_dim`, `gru_hidden_dim` and Target
feature layers. Other label layouts fail closed and require the native format. The
manifest records runtime `DFLASH`; SGLang 0.5.21 loads the exported
`prefix_gru`/`embed_proj` tensors.

The native manifest records the Domino algorithm, trained `block_size`, label layout,
Target feature layers and the full training objective.
`aspec.draft_artifact.load_native_draft` validates the checksums and semantics, then
loads every tensor strictly. A real 8-step checkpoint has been roundtripped through
this path (63 tensors, exact values and dtypes).

Run real decode acceptance against the SGLang export with `aspec evaluate`. Domino
defaults to the full-vocabulary candidate pool (`--domino-candidate-pool-size 0`),
reports measured acceptance length (including Target tokens) and checks greedy
equality and repeat stability:

```bash
aspec evaluate --target /path/to/Qwen3-8B \
  --draft outputs/domino-qwen3-8b-sglang \
  --data /path/to/held-out-prompts.jsonl --output outputs/domino-decode.json
```

### Official checkpoint mapping

The published `Huang2020/Qwen3-8B-Domino-b16` checkpoint uses
`architectures=["DFlashDraftModel"]`, `model_type="qwen3"`, and nests the Domino
fields under `dflash_config`. In that layout `num_target_layers` is the Target
decoder depth (36), not ASpec's feature count (5). Convert the config once before
training:

```python
import json
from aspec.models.draft import AutoDraftModelConfig

raw = json.load(open("Qwen3-8B-Domino-b16/config.json"))
json.dump(AutoDraftModelConfig.from_dict(raw).to_dict(), open("aspec_domino.json", "w"))
```

`AutoDraftModelConfig.from_dict` detects `dflash_config.projector_type="domino"`
and produces the flat `DominoDraftModel`/`model_type="domino"` config. The published
weight names already match ASpec's export names (`fc`, `hidden_norm`, `norm`,
`layers.*`, `prefix_gru.*`, `embed_proj.*`); `load_initial_draft_weights` maps them
back and borrows the missing `embed_tokens.weight` from the Target. Verified against
the published config and all 62 tensor names/shapes.

### Native proposal component

`aspec.draft_artifact.DominoConsumer` loads a native Domino asset and rolls out one
block: the DFlash backbone stays block-parallel, the pure prefix keeps backbone
logits, and the suffix is generated sequentially with `prefix_gru` + `embed_proj`
using already generated tokens. It mirrors the reference `spec_generate` indexing for
both `shift_label=true` and `shift_label=false`. The manifest advertises this
entrypoint; verification and acceptance remain caller-owned.

## Validation status

Plan compilation, registration, trainer construction, checkpoint topology and the
CPU numerical contracts are covered by `tests/cpu`. A GPU run with one Target and two
Draft GPUs verified 8-step training and step-4 exact resume: losses, model tensors,
optimizer state and RNG state matched the uninterrupted run. The public CI lane is
CPU-only; no throughput or convergence claim is made here.

The SGLang export was loaded by official SGLang 0.5.21 and evaluated with
`aspec evaluate` on 16 held-out prompts (32 new tokens, greedy, two repeats):
speculative output matched target-only greedy output, both repeats were stable, and
the measured acceptance length was 2.016 tokens/verification including Target
tokens. This is a bounded decode integration check, not convergence or throughput
evidence.

`tools.validation.domino_acceptance_ablation` runs a paired corrected-versus-backbone
decode by zeroing only the correction output projection in a copy of the same SGLang
asset. On the published checkpoint (16 prompts x 32 greedy tokens, two repeats) the
corrected branch measured 2.829 and backbone-only 2.473 tokens/verification
(+0.355). On the 450-step synthetic ASpec checkpoint the same comparison measured
2.016 versus 2.065 (-0.049), so the correction benefit is checkpoint/data-dependent
rather than unconditional. Both runs kept greedy equality and repeat stability; the
tool rejects mismatched prompts.

A bounded comparison against SpecForge at its Domino merge commit used the same
published Qwen3-8B Domino checkpoint, 16 train / 16 held-out rows, 8 steps, block 16,
128 anchors, FlexAttention and the same linear `lambda_base` schedule. Both trainers
kept corrected acceptance above the backbone-only branch. Anchor sampling, batch
aggregation and acceptance naming differ (SpecForge reports accepted tokens including
the root), so the comparison is directional rather than a strict parity test.

## Reference and selected scope

The implementation is **adapted** from the merged
[SpecForge Domino training path](https://github.com/sgl-project/SpecForge/pull/571)
(`specforge/core/domino.py` and the DFlash projector addition, merge
`7de39e32fc845caf59bca9841595145d24e1796f`, MIT; see
`licenses/SpecForge-MIT.txt`) and follows the public
[Domino method](https://arxiv.org/abs/2605.29707) at
[jianuo-huang/Domino @ `44e1e6912ba9d301899739a54bfda2ce8826980a`](https://github.com/jianuo-huang/Domino/tree/44e1e6912ba9d301899739a54bfda2ce8826980a)
for the architecture and label layout. It is not a reproduction of the paper's
training data, budget or acceptance results, and it does not copy the unlicensed
reference repository's source.

The selected first scope is one dense full-attention DFlash backbone such as Qwen3
with a frozen Target LM head and features. Dense FFNs and full intra-block attention
are enforced. Native MTP conditioning, sparse-attention training stages and MoE
FFNs fail closed.

## Correction head

The head owns two parameter groups, named as in the reference for future export:

- `prefix_gru`: a bias-free GRU over the block's real tokens, producing a causal
  predecessor state per suffix position;
- `embed_proj`: `Linear(hidden + gru_hidden) -> SiLU -> Linear(vocab)` producing the
  logit delta.

The block backbone itself stays block-parallel: attention visibility, anchor
sampling, noise construction and Target-feature fusion are unchanged from DFlash.
Only the correction head is causal, and its inputs are the **ground-truth block
tokens**, not Target features.

## Label-shift layouts

`DominoConfig` makes the alignment explicit instead of assuming one checkpoint
layout. `shift_label` and `pure_draft_prefix_len` select the reference conventions:

| Field | `shift_label=true` (official Qwen3) | `shift_label=false` |
| --- | --- | --- |
| Supervised label for block position `k` | `x[anchor + 1 + k]` | `x[anchor + k]` |
| First supervised block position | `0` | `1` (anchor is context only) |
| `suffix_start` | `pure_draft_prefix_len` | `1 + pure_draft_prefix_len` |
| GRU input tokens | `x[anchor .. anchor + block_size - 1]`, states used from `suffix_start` | `x[anchor .. anchor + block_size - 2]`, states used from `suffix_start - 1` |

`pure_draft_prefix_len` must leave at least one corrected position in the block.
The official `Qwen3-8B-Domino-b16` configuration uses `shift_label=true` and
`pure_draft_prefix_len=1`, so block position 0 keeps the backbone logits and
positions 1..`block_size-1` receive the GRU delta. The unshifted layout matches
plain DFlash supervision: position 0 is the anchor and excluded from the loss.

## Objective

`DominoModel` mixes the corrected and backbone-only cross-entropies:

```text
loss = (1 - lambda_base) * CE(corrected logits, labels)
     + lambda_base     * CE(backbone logits, labels)
```

With both curriculum fields set, the trainer evaluates the value at the 1-based
microbatch counter `m`:
`start * max(0, 1 - m / max(1, int(lr_total_steps * decay_ratio)))`.
`lr_total_steps` is measured in optimizer updates; because the counter advances per
microbatch, accumulation greater than one reaches zero before the last update,
matching the reference SpecForge Domino trainer. Otherwise the value stays at
`domino_lambda_base`.
`lambda_base=1`
removes the correction contribution and keeps the selected label layout (the
correction head receives no useful gradient). It equals the plain DFlash objective
only for the unshifted layout (`shift_label=false`) with otherwise matching
weighting settings; the shifted layout supervises different slots. Anchor sampling,
block masks, label bounds, `loss_mask` and DFlash-style exponential depth decay are
reused; `ce_loss_alpha=1` and `l1_loss_alpha=0` are enforced. Reporting accuracy and per-position loss use the
corrected logits. The trainer reports `mixed_ce` as `train/avg_loss`/`eval/avg_loss`
(the objective actually used for backward) and exposes `mixed_ce`, `backbone_ce` and
`corrected_ce` as separate token-pooled component curves. With
`training.domino_eval_backbone=true`, the same evaluation forward and the same
sampled anchors produce both branches:
`eval/ploss_i`/`eval/acc_i` remain corrected, while
`eval/backbone_ploss_i`/`eval/backbone_acc_i` (plus
`eval/backbone_avg_loss`, `eval/backbone_avg_acc` and
`eval/backbone_simulated_acc_len`) describe the backbone-only branch. There is no
second evaluation pass, and the mixed objective and trained correction head are
unchanged by this reporting mode.

## Minimal numerical use

Inside the [isolated Linux environment](installation.md):

```python
import torch
from aspec.models.domino import DominoModel
from aspec.models.draft.domino import DominoConfig, DominoDraftModel

config = DominoConfig(
    hidden_size=16, intermediate_size=32, num_hidden_layers=1,
    num_attention_heads=2, num_key_value_heads=1, head_dim=8, vocab_size=32,
    num_target_layers=1, target_hidden_size=16, target_num_hidden_layers=4,
    target_layer_ids=[1], mask_token_id=31, max_position_embeddings=128,
    block_size=5, emb_dim=6, gru_hidden_dim=8,
    shift_label=True, pure_draft_prefix_len=1,
)
draft = DominoDraftModel(config)
draft.embed_tokens.requires_grad_(False)
model = DominoModel(
    draft, block_size=5, num_anchors=2, attention_backend="sdpa",
    lambda_base=0.3,
)
result = model(
    input_ids=torch.arange(12).reshape(1, 12) % 32,
    hidden_states_list=torch.randn(1, 12, 16),
    loss_mask=torch.ones(1, 12),
    lm_head_weight=torch.randn(32, 16),
    last_hidden_states=torch.randn(1, 12, 16),
    injected_anchors=torch.tensor([[2, 7]]),
    injected_keep_mask=torch.ones(1, 2, dtype=torch.bool),
)
result[0].backward()
```

Run `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -m pytest -q tests/cpu/test_domino.py`
for configuration fail-closed checks, independent forward/gradient equations for both
label layouts, `lambda_base` identity, anchor exclusion, loss-mask and short-sequence
handling, empty supervision, and frozen Target/embedding behavior.

## Remaining work

SGLang and native export with strict reloading work for ASpec FSDP checkpoints, the
official Qwen3-8B Domino config/weight layout maps into ASpec, the native
`DominoConsumer` produces proposal tokens for both label layouts, bounded real SGLang
decode reports corrected and backbone-only acceptance, and
`tools.validation.domino_anchor_protocol` can materialize and verify a hashed
canonical anchor stream. Remaining work is harder/real data, longer convergence and a
formal anchor-replay training mode (the aligned comparison here used a local validation
harness). CPU numerical and Plan checks do not establish Qwen3-8B training quality
or throughput.
