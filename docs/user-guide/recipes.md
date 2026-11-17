# Choose a recipe

Choose a recipe for your Target, Draft algorithm and available resources. EAGLE4 uses two successive
stages. See [supported combinations](../support.md) before combining algorithms and structures.

Qwen and Ling DSpark/DFlash paths are the main optimization and recipe priorities. EAGLE3/4 and
native MTP have their own documented training and asset contracts; DFlash2 remains an experimental
example outside this recipe catalog. A listed recipe is a maintained starting configuration, not
a claim that every algorithm/structure combination has the same quality or performance evidence.

The table shows each recipe's configured starting allocation, not a minimum requirement or optimal
Target/Draft split. T = Target GPU; D = Draft GPU. Check memory fit and the required node-local NVLink
connections on your hardware before training.

| Launcher ID | Target / Draft | Starting resources | Use case |
| --- | --- | --- | --- |
| `quickstart` | Qwen3-8B / DSpark | 1 node, 2 GPUs: 1T + 1D; 1K | Try training, recovery and export with bundled data |
| `dspark` | Qwen3-8B / Dense DSpark | 1 node, 8 GPUs: 4T + 4D; 4K | Train on your data with packing and NVLink |
| `dflash` | Qwen3-8B / DFlash | 1 node, 3 GPUs: 1T + 2D; 4K | Train a block-parallel DFlash Draft |
| `eagle3` | Qwen3-8B / EAGLE3 | 1 node, 3 GPUs: 1T + 2D; 4K | Train a recurrent EAGLE3 Draft |
| `eagle4` | Qwen3-8B / EAGLE4 | 1 node, 3 GPUs: 1T + 2D; 4K; sequential stages | Train a Draft, then candidate retrieval projections |
| `dmtp` | Qwen3.5-9B / MTP-conditioned DSpark | 1 node, 8 GPUs: 4T + 4D; 4K, CP1 | Train a Draft conditioned on the frozen native predictor; native assets |
| `native-mtp` | Qwen3.5-9B / native predictor | 1 node, 3 GPUs: 1T + 2D; 4K | Fine-tune the Target's native predictor |
| `ling-tiny` | Ling-3.0-tiny / Dense DSpark | 1 node, 2 GPUs: 1T + 1D; 4K | Train a small Ling Draft |
| `ling-flash` | Ling-3.0-flash / Dense DSpark | 1 node, 3 GPUs: 2T (TP2) + 1D; 4K | Train against a larger Ling Target |
| `ling-flash31` | Ling-3.1-flash / MLA + LingMoE DSpark | 1 node, 8 GPUs: 4T (TP4/EP4) + 4D; 4K | Train a structured Draft against the FP8 Ling Target |
| `glm5-flash` | GLM-5.3-Flash FP8 / DSpark | 1 node, 4T (TP4) + 1D, 8K | Train a text Draft from scratch or continue from Red Hat weights. [GLM training guide](#glm-53-flash-dspark) |
| `shared-target` | Qwen3-8B / two independent DSpark Drafts | 1 node, 8 GPUs: 4T + 2D × 2; 4K | Compare independent Drafts sharing Target features |
| `qwen27b-long` | Qwen3.8-27B / DSpark | 1 node, 8 GPUs: 3 × TP2 Target + 2D (CP2); 256K-class | Distribute long-context training with CP |
| `multinode` | Qwen3-8B / DSpark | 2 nodes × 8 GPUs; each 4T + 4D; 4K | Keep features node-local and synchronize gradients across nodes |

The [27B recipe](../../configs/recipes/qwen3_8_27b_256k_cudnn_cp2_nvlink.yaml)
uses a 261,888-token cap, native cuDNN THD attention and CP2. Each TP2 Target replica admits one request,
with two source slots. `inference.target_prefill_memory_optimized` stays disabled because the direct
prefill path currently requires TP1. Size feature buffers and backend token capacity for your inputs;
see [long-context validation](../contributing/validation.md#algorithm-capacity-and-fault-checks) for
capacity checks and verification commands. Additional examples are under
[configs/examples](../../configs/examples/README.md).

For a short single-node DSpark run with cuDNN attention, CP1 and no packing, use the
[Qwen3.5-9B / 4K](../../configs/examples/qwen3_5_9b_4k_cudnn_nvlink.yaml) or
[Qwen3.8-27B / 4K](../../configs/examples/qwen3_8_27b_4k_cudnn_nvlink.yaml) example.
Both start with four TP1 Target engines and four Draft GPUs and require a matching local Target
checkpoint plus prepared token IDs/masks. The 9B DSpark example is separate from native MTP training.

MTP-conditioned DSpark is available as `aspec recipes plan dmtp`. See the [Qwen3.5-9B workflow](../../examples/qwen3.5-9b-dmtp/README.md) for supervision timing, recovery and native export.

## Inputs and normal training

Install using [the environment setup](installation.md). Download the exact Target checkpoint or use
its declared public Hub ID. For private/local mirrors set `ASPEC_TARGET_MODEL` to an absolute path
with the same model identity. Qwen3.8-27B and multi-node training require that variable explicitly;
multi-node training uses the same absolute local model path on every node.

Start Target in its own environment before training. For DSpark on eight GPUs, run
`CUDA_VISIBLE_DEVICES=4,5,6,7 aspec target` there and use GPUs 0–3 in Draft:

```bash
aspec recipes list
export ASPEC_TARGET_MODEL=/models/Qwen3-8B
export ASPEC_TRAIN_DATA=/data/train.parquet
export ASPEC_EVAL_DATA=/data/eval.parquet
aspec recipes plan dspark
CUDA_VISIBLE_DEVICES=0,1,2,3 aspec recipes train dspark --output outputs/dspark-run-001
```

Prepare disjoint inputs using [data preparation](data.md). The launcher validates conversation JSONL
or converts already-tokenized JSONL to typed Parquet:

```bash
aspec recipes prepare \
  --input data/train-tokenized.jsonl --output data/train.parquet
aspec recipes prepare \
  --input data/eval-tokenized.jsonl --output data/eval.parquet
```

For conversation JSONL, choose a `.jsonl` output; the selected recipe's tokenizer/template processes
it during training. Configs with `chat_template: null` expect prepared token IDs/masks. Preparation
neither generates Target answers nor silently re-tokenizes data. Never reuse Qwen token IDs for Ling;
see [Ling data and binding](ling.md). Provide enough usable supervised rows for the planned optimizer
batches, including after packing and drop-last. Different files are checked by hash, but the user must
ensure their examples are disjoint.

Training defaults to three dataset passes with evaluation and periodic/final checkpoints.
This is a starting schedule, not a convergence claim. Adjust with repeated `--set KEY=VALUE` options.
For segmented training, set the full `training.lr_total_steps` before segment one. Resume into a new
output using `--set training.load_path=/absolute/path/to/checkpoints`; see [recovery](training.md).
The optional [runtime validation command](../contributing/validation.md) runs a short training,
recovery and export check without changing normal training defaults.

## DSpark performance settings

The `dspark` recipe is a general FA4/4T+4D training starting point. For online DSpark full attention,
the pinned runtime also supports cuDNN packed attention and direct TP1 prefill:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 aspec recipes train dspark --output outputs/dspark-cudnn-001 \
  --set training.attention_backend=cudnn_flex_attention \
  --set inference.target_prefill_memory_optimized=true
```

These options retain the recipe's batch, schedule and GPU split. The [published benchmarks](../benchmarks/results.md)
use cuDNN and workload-specific Target/Draft splits, batches and buffers; their YAMLs are not general
training defaults. Use [topology selection](topology.md) or observed supply/training times to choose
a split for your workload. Do not assume 4T+4D is optimal or that enabling one backend reproduces a
published multiplier.

For DSpark/DFlash GQA 4:1 with head dimension 128, BF16 and CP1, set
`"cudnn_gqa_mode": "internal"` in the **Draft model JSON config** to enable
ASpec's joint-head backward on SM90 or SM100. Keep
`training.attention_backend=cudnn_flex_attention`. The backward source is shipped
with ASpec; the installed cuDNN Frontend package is not patched. The default
`auto` policy is unchanged. Explicit `native` and `external` modes allow controlled
comparisons. See [the execution contract and validation](../../aspec/kernels/cutedsl/cudnn_gqa/README.md)
before selecting this opt-in route; unsupported requests fail instead of falling back.

## EAGLE4: Draft then candidate retrieval

`train eagle4` runs stage one. Export its checkpoint with `aspec export --format aspec`,
set `ASPEC_INITIAL_DRAFT` to that native asset, then use `train eagle4 --stage 2 --output ...`.
The acceptance command automates both stages, their exports and stage-local resume. Exact resume
never switches stages. The candidate stage freezes the whole Draft/head and trains two retrieval
projections using the documented Top-K forward-KL objective; it is not an unpublished official
recipe or serving speed claim. See [EAGLE4 boundaries](algorithms.md#eagle4-two-stages).

## Qwen3-8B DSpark quality recipe

The separate [Target-generated configuration](../../configs/examples/qwen3_8b_dspark_target_generated.yaml)
retains the recorded quality protocol: ten passes, GBS512 and held-out Target-generated Parquet.
Set the same train/eval environment variables, then run this YAML directly with `aspec --config`.
See [decode reproducibility](../support.md#decode-reproducibility) before interpreting acceptance
or inference timing.

## GLM-5.3-Flash DSpark

The recipe uses the official FP8 Target on four GPUs and trains a BF16 Draft with FA4 attention
on a fifth GPU. Set `training.attention_backend=flex_attention` to use FlexAttention instead;
this choice is independent of the DSpark loss recipe.
Keep the attention backend fixed when reproducing or resuming a run: changing it can alter
BF16 rounding and the training trajectory.
Use the checkpoint's chat template with text-only `system`, `user` and `assistant` messages
in disjoint conversation JSONL files. Native assistant messages must use `reasoning_content`
for reasoning text; reasoning alias fields are rejected.
[Official Target](https://modelscope.cn/models/ZhipuAI/GLM-5.3-Flash) · [Draft configuration](../../aspec/config/dspark_glm5_3_flash_draft_config.json)

Activate the patched Target environment and start the service. DeepGEMM compiles kernels for
the shapes used by the workload instead of precompiling its full shape range:

```bash
SGLANG_JIT_DEEPGEMM_PRECOMPILE=0 CUDA_VISIBLE_DEVICES=0,1,2,3 aspec target
```

In the Draft environment, set the inputs and choose one initialization. The first command trains
with the default DeepSpec loss. The second loads the published BF16 Draft, verifies its frozen
embedding and output head against the Target, and starts a new optimizer with Red Hat's loss.

```bash
export ASPEC_TARGET_MODEL=/models/GLM-5.3-Flash
export ASPEC_TRAIN_DATA=/data/glm-train.jsonl
export ASPEC_EVAL_DATA=/data/glm-eval.jsonl
export CUDA_VISIBLE_DEVICES=4

aspec recipes train glm5-flash --output outputs/glm-scratch
aspec recipes train glm5-flash --output outputs/glm-warm \
  --set model.initial_draft_model_path=/models/GLM-5.3-Flash-speculator.dspark \
  --set training.dspark_loss_recipe=speculators_redhat_v1 \
  --set training.learning_rate=1e-6 --set training.min_lr=1e-7
```

The published Draft directory must contain its original `config.json` and `model.safetensors`.
Its six pre-layer captures (20, 28, 32, 36, 40, 44) correspond to ASpec's zero-based layer
outputs (19, 27, 31, 35, 39, 43). The five Draft layers use a 2,048-token anchor-relative
context window and causal attention within each eight-slot block.
[Red Hat checkpoint and configuration](https://huggingface.co/RedHatAI/GLM-5.3-Flash-speculator.dspark/tree/738c283a7e860d7dafce5785161466bc556e5cae) · [Loss definitions](training.md#dspark-loss-recipes)

To resume an interrupted ASpec run, use the same loss recipe, dataset and LR horizon. Pass
`--set training.load_path=/absolute/path/to/previous-run/checkpoints` with a fresh output path.
The checkpoint restores the optimizer, scheduler and committed work. To export one completed
iteration, run the following from the Draft environment:

```bash
aspec export --format aspec --input-dir /path/to/checkpoints/iter_0000005 \
  --output-dir /path/to/glm-draft-asset --target-model-path "$ASPEC_TARGET_MODEL" \
  --embedding-key model.language_model.embed_tokens.weight
python -c 'import sys; from aspec.draft_artifact import load_native_draft; load_native_draft(sys.argv[1])' /path/to/glm-draft-asset
```

The maintained GLM path uses text-only 8K inputs, CP1 and one Draft GPU. Warm-start
initialization loads model weights and starts a new optimizer; it does not restore the published
Draft's original optimizer state. Native exports preserve the Draft tensors and frozen Target
bindings. Full-corpus convergence, reproduction of the published Draft's quality, integrated
speculative-serving throughput, multimodal inputs and training the Target's native MTP layer
remain outside the validated scope.

## Additional structures

SWA/CSA/QSA/DSA require their documented structure and stage contracts. MLA materializes K/V.
See [structure examples](structures.md), [sharing](sharing.md) and [multi-node setup](multinode.md).
Use their documented stage transitions, input formats and export boundaries.
