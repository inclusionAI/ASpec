# Megatron Target and Draft

ASpec supports the public Megatron Core 0.20.0 API
(`core_v0.20.0`, revision `685fb8bd970185e9ac3283fe32a5d6667bf4b268`).
Select the Target and Draft independently. Qwen3-8B prefill and dense DSpark
use BF16, microbatch size one and single-node PP1. At CP1 the Target and
unpacked Draft support TP1 or TP2. The Megatron Draft supports DP replicas.
Packed DSpark training uses Draft TP1 with `training.dflash_packing=true`,
`training.anchor_sampling_scope=sample`, and a THD attention backend
(`cudnn_flex_attention` or `fa4`). The controller assigns complete documents
to each row. The collator resets positions at document boundaries and retains
each document's supervision mask and work ID. Fixed-feature and live SGLang
comparisons cover CP1 and CP2 with cuDNN. For CP2, select the SGLang Target configuration and
set `training.draft_context_parallel_size=2`. Two Draft GPUs form one replica.
Each rank reads its own Target feature shard over NVLink.
Shared Target, decode and MTP conditioning are outside
this configuration.

## Dependency profiles

| Distribution | Environment |
| --- | --- |
| `ling-aspec-runtime` | Shared ASpec code, CLI and service protocol |
| `ling-aspec` | Existing ASpec Draft dependencies |
| `ling-aspec-target` | SGLang Target dependencies |
| `ling-aspec-megatron` | Megatron dependencies for either role |

Install each role in its own isolated venv. All ASpec distributions in a run use
the same release version. Install Megatron with the common runtime directly;
combining its profile with either default environment package creates unsupported
dependency mixtures. Backend imports remain lazy.

Megatron requires Python >=3.12. Its profile pins Torch 2.13.0, CUDA toolkit 13.0.3,
Transformer Engine 2.20.2, FA4 b34, CUTLASS DSL 4.7.1 and Quack 0.6.5. The feature-only Target uses FlashInfer 0.6.18
and the standalone sglang-kernel 0.4.7 package on SM90. The ordinary
Draft and SGLang profiles retain their independent versions.

Transformer Engine's PyTorch extension must be built against the execution
environment's Torch ABI. An unconstrained build environment can select a different
Torch. Source setup installs the build prerequisites first, then builds with
`--no-build-isolation`, selecting the venv's CUDA 13 toolkit and NCCL/cuDNN headers
explicitly. Including all NVIDIA wheel headers can put the transport's CUDA 12
runtime headers before CUDA 13; the setup helper avoids this mixture.

From a source checkout:

```bash
bash setup.sh --megatron --python python3.12 --venv /absolute/path/to/draft-venv
bash setup.sh --megatron --target --python python3.12 --venv /absolute/path/to/target-venv
```

For a built release directory, create a Python 3.12 venv, activate it, and install:

```bash
pip install -r /path/to/release/runtime/megatron-build.txt
aspec_nvidia=$(python -c 'import pathlib,sysconfig; print(pathlib.Path(sysconfig.get_paths()["purelib"]) / "nvidia")')
CUDA_HOME="$aspec_nvidia/cu13" \
  NVTE_CUDA_INCLUDE_PATH="$aspec_nvidia/cu13/include" \
  CPLUS_INCLUDE_PATH="$aspec_nvidia/cudnn/include:$aspec_nvidia/nccl/include" \
  NVTE_BUILD_USE_NVIDIA_WHEELS=0 MAX_JOBS=4 pip install --no-build-isolation \
  --find-links /path/to/release ling-aspec-megatron
aspec doctor --backend megatron --gpu
```

The host needs a compatible CUDA driver and the native prerequisites in the
[installation guide](installation-advanced.md).

Each process prepares native-library paths from its own interpreter. The optional
local supervisor removes automatically selected Draft library paths before starting
Target. A separate Target service requires a service address; managed-local mode
also requires an explicit Target interpreter. Neither role imports the other's
Python dependency graph or transfers Python model objects across the boundary.

## Target and Draft adapters

The frozen Target uses Core's `HybridModel` and `HybridStack`. Each logical
Qwen3 block contributes an attention position (`*`) and an MLP position (`-`).
For Qwen3-8B the stack executes 72 positions. The adapter maps the grouped QKV
and gate/up weights into the Core TP projections and loads the final norm.
The feature-only model omits the vocabulary head.

ASpec supplies two Target-specific layer specs for the single-sequence prefill.
They preserve the BF16 residual stream across fused add/RMSNorm boundaries,
the Q/K normalization reduction order, RoPE, causal FA3 and fused SwiGLU.
At an auxiliary MLP boundary the TP leader adds the pending branch and residual
directly into the transport slot. The final feature uses the checkpoint's RMSNorm.
Each request goes through the same Core model forward. Each engine processes one sequence at a time and publishes the batch together. For one Draft dispatch of `D` samples across `E` Target engines, configure at least `inference_batch_size - 1 + ceil(D / E)` source ring slots per engine. This leaves room for a complete Draft dispatch while an unfinished Target batch holds slots.

Megatron Draft partitions attention heads and dense FFN channels using public
Core TP projections, and uses Core DDP gradient buffers with ASpec's FP32-master
AdamW update and schedule. The retained DSpark graph still owns attention/QK
kernels, block compaction, bounded projection caches, tiled/indexed L1, statistics,
selective recomputation and deferred metrics. Vocabulary, Markov and confidence
computations remain replicated across TP. This preserves the existing global
softmax/L1 semantics; vocabulary sharding is not a claimed capability.

Megatron Draft gradient precision is selected through the normal training
configuration:

```yaml
training:
  train_backend: megatron
  megatron_grad_dtype: bf16  # fp32 (default) or bf16
```

`fp32` preserves the existing default. `bf16` passes
`grad_reduce_in_fp32=False` to Core's ordinary DDP configuration, selecting BF16
main-gradient accumulation and DP reduction for the BF16 model parameters. The dedicated TP-partial Q/K norm-gradient
reduction, optimizer masters and Adam moments remain FP32 under both settings.
This option applies only to Megatron Draft; the existing Torch/REPLICATE BF16
gradient path is unchanged.

The Plan records accumulation, DP-reduction and TP-partial-reduction precision
separately. Runtime execution observations include actual main-gradient dtypes
and the Core DDP flag. Exact resume preserves this setting; checkpoints without
the field are interpreted as FP32. Changing gradient precision requires
a weights-only new stage rather than an exact continuation.

For unpacked training, one TP leader prefetches source GETs into bounded, preallocated destination rings.
TP broadcasts and collation run on the consumer thread between microbatches, in
the same order as model collectives. Availability and tensor metadata use a CPU
Gloo group; no background reader launches NCCL work. Training and evaluation own
separate input groups. With source prefetch enabled, destination slots must cover
`prefetch_depth + 2` samples, including the active consumer and an in-progress GET.
The destination lease remains attached until the consumer finishes. Source release
occurs once per DP replica. Anchors/keep masks are shared from the TP leader;
commit collection checks matching work identity and reports it once per replica.

Gradients are normalized by the complete optimizer window's supervision weight,
pooled across data and context ranks. TP-partial Q/K norm gradients are reduced once, and clipping
counts each replicated parameter once. Checkpoints describe logical TP shards and
restore model, FP32 masters, moments, schedule and both ASpec and Core RNG streams.
Exact resume requires the same Draft backend, TP topology and world size. Cross-backend weights-only
initialization is a separate stage. Native export gathers the logical Draft weights
for the existing strict loader.

## Run Qwen3-8B with DSpark

Start with SGLang Target and Megatron Draft in separate environments. Activate
the Megatron Draft venv and select two allocated GPUs:

```bash
source /absolute/path/to/megatron-draft-venv/bin/activate
CUDA_VISIBLE_DEVICES=0,1 aspec quickstart qwen3-8b \
  --target /absolute/path/to/Qwen3-8B --output outputs/megatron-draft-001 \
  --managed-target /absolute/path/to/sglang-target-venv/bin/python \
  --set training.train_backend=megatron \
  --set training.megatron_grad_dtype=bf16 \
  --set mooncake.destination_gpu_ring_slots=3
```

This exercises training, fresh-process recovery, native export and strict loading.
The two-GPU Quickstart budget covers TP1 for both roles. Megatron Draft TP2 uses
`training.draft_tensor_parallel_size=2` and two Draft GPUs per DP replica.
Megatron Target is selected independently with `model.target_model_backend=megatron`
and `inference.inference_engine_type=megatron`; its TP2 settings are
`inference.megatron.tp_size=2` and `inference.inference_num_gpus_per_engine=2`.

## Verification

Run the fixed-feature comparison in each Draft's own environment using the same
initial weights, SGLang feature cache, anchors, schedule and gradient precision.
Prepare the initial weights with `megatron_torch_alignment.py --backend fsdp
--phase prepare`. For Torch TP1 use `--backend fsdp`. For Torch TP2 the
verification tool uses a native PyTorch DTensor adapter around the existing
DSpark trainer and optimizer. This adapter is available only in the validation
tool. The public Torch training entry point supports TP1.

For a TP1 packed comparison, pass `--packed-pairs` to both runs of
`megatron_torch_alignment.py`. Supply an even number of complete, unpadded
documents in the teacher cache. Each adjacent pair must fit within `--length`.
For CP2, add `--cp 2` and launch two workers per data replica on both backends.
Each worker reads its context shard from each document before collation.
For a strict cuDNN comparison, set `CUBLAS_WORKSPACE_CONFIG=:4096:8` and pass
`--deterministic` to both training runs.
The driver checks that each document is committed once, compares anchors,
loss denominators and optimizer updates, and records THD backward evidence.

Run the following Bash commands from the checkout. Supply the role venvs, model,
SGLang feature cache and an empty output root. The same backend that wrote a
checkpoint also supplies its uninterrupted replay reference.

```bash
src=$PWD
torch_env=/absolute/path/to/torch-draft-venv
meg_env=/absolute/path/to/megatron-draft-venv
model=/absolute/path/to/Qwen3-8B
teacher=/absolute/path/to/aspec-eval-cache.pt
out=/absolute/path/to/fresh-fixed-feature-runs
export CUDA_DEVICE_MAX_CONNECTIONS=1
unset PYTHONPATH PYTHONHOME

fixed_run() {
  local env_dir=$1 devices=$2 workers=$3
  shift 3
  VIRTUAL_ENV="$env_dir" PATH="$env_dir/bin:$PATH" CUDA_VISIBLE_DEVICES="$devices" \
    "$env_dir/bin/python" -m aspec._environment \
    "$env_dir/bin/torchrun" --standalone --nproc_per_node="$workers" \
    "$src/tools/validation/megatron_torch_alignment.py" \
    --source "$src" --model "$model" --teacher "$teacher" "$@"
}
fixed_run "$torch_env" 0 1 --backend fsdp --tp 1 --phase prepare --output "$out/initial"
```

For the packed Draft comparison, use a cache whose adjacent document pairs
fit within 1,024 tokens. Run each backend on the same devices in sequence.

```bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8
for cp in 1 2; do
  devices=0
  if [[ $cp == 2 ]]; then devices=0,1; fi
  common=(--tp 1 --cp "$cp" --packed-pairs --length 1024 --anchors 32
          --steps 100 --deterministic --phase uninterrupted
          --tensor-steps 1,50,100 --state-steps 100 --initial "$out/initial/model.safetensors")
  reference="$out/torch-packed-cp$cp"
  fixed_run "$torch_env" "$devices" "$cp" --backend fsdp "${common[@]}" --output "$reference"
  fixed_run "$meg_env" "$devices" "$cp" --backend megatron --megatron-grad-dtype bf16 \
    "${common[@]}" --reference "$reference" --no-save-reference-tensors --output "$out/megatron-packed-cp$cp"
done
```

For the unpacked TP2 checkpoint and resume comparison, run:

```bash
for dp in 1 2; do
  devices=0,1
  milestones=1,2,5,10,20,30,40,50,60,70,80,90,100
  if [[ $dp == 2 ]]; then
    devices=0,1,2,3
    milestones=1,2,5,10,11,20,30,40,50,60,70,80,90,100
  fi
  workers=$((2 * dp))
  common=(--tp 2 --steps 100 --split-step 50 --attention-backward-deterministic
          --tensor-steps "$milestones" --state-steps 1,10,25,50,75,100
          --initial "$out/initial/model.safetensors")
  torch_ref="$out/torch-tp2-dp$dp"
  meg_ref="$out/megatron-tp2-dp$dp"
  fixed_run "$torch_env" "$devices" "$workers" --backend torch-tp "${common[@]}" \
    --phase checkpoint-control --output "$torch_ref"
  fixed_run "$meg_env" "$devices" "$workers" --backend megatron --megatron-grad-dtype bf16 \
    "${common[@]}" --phase checkpoint-control --reference "$torch_ref" --output "$meg_ref"
  fixed_run "$torch_env" "$devices" "$workers" --backend torch-tp "${common[@]}" \
    --phase resume --resume "$torch_ref/checkpoints/iter_0000051" --reference "$torch_ref" \
    --no-save-reference-tensors --output "$torch_ref-resumed"
  fixed_run "$meg_env" "$devices" "$workers" --backend megatron --megatron-grad-dtype bf16 \
    "${common[@]}" --phase resume --resume "$meg_ref/checkpoints/iter_0000051" --reference "$meg_ref" \
    --no-save-reference-tensors --output "$meg_ref-resumed"
done
```

The control saves after update 50 and continues to 100. The fresh processes
compare the remaining updates with their respective controls. Each run records
the backward policy on every rank, the loss components and gradient norm at every
update, and logical gradients and optimizer state at the requested milestones.

For an online packed comparison, give both Drafts the same pretokenized Parquet
file and initial weights. SGLang computes the features from the token IDs on
each run. Use short documents that fit together within a 1,024-token row.
The example uses three GPUs for one SGLang Target and a two-rank CP Draft.
For CP1, set `cp=1` and `devices=0,1`.

```bash
sgl_target=/absolute/path/to/sglang-target-venv/bin/python
train=/absolute/path/to/train.parquet
cp=2
devices=0,1,2
online=(--model "$model" --initial "$out/initial/model.safetensors" --train-data "$train"
        --target-python "$sgl_target" --cp "$cp" --steps 100)
for draft in torch megatron; do
  env_dir=$torch_env
  reference=()
  if [[ $draft == megatron ]]; then
    env_dir=$meg_env
    reference=(--reference "$out/online-packed-torch-cp$cp")
  fi
  VIRTUAL_ENV="$env_dir" PATH="$env_dir/bin:$PATH" CUDA_VISIBLE_DEVICES="$devices" \
    "$env_dir/bin/python" -m aspec._environment "$env_dir/bin/python" \
    -m tools.validation.megatron_packed_online "${online[@]}" --draft "$draft" \
    "${reference[@]}" --output "$out/online-packed-$draft-cp$cp"
done
```

The comparison checks the document order, token IDs, masks, received hidden
states and supervision weights for each rank and microbatch. It pools the
objective over both microbatches and all context shards before computing the
per-step relative difference against Torch in `objectives.json`.

On SM100, a live Qwen3-8B SGLang TP1 Target computed features from 16
Alpaca GPT-4 Chinese conversations. Each run consumed 200 packed rows of two
documents, accumulated two rows per update, and saved a checkpoint at update
100. Both Drafts used BF16 gradients and deterministic cuDNN THD backward.

| Draft topology | Torch mean objective | Megatron mean objective | Max absolute per-step rel diff | NVLink reads per run |
| --- | ---: | ---: | ---: | ---: |
| TP1/CP1/DP1 | 2.77931555 | 2.77931555 | 0% | 400 |
| TP1/CP2/DP1 | 2.76814709 | 2.76814709 | 0% | 800 |

All reads were released and each rank recorded 1,000 deterministic attention
backward calls. The [per-step objectives](../benchmarks/megatron-packed-online.csv)
include the signed relative difference against the same-topology Torch run.

For Target comparison, run `tools/validation/megatron_target_aspec_alignment.py`
with `torchrun` in the Megatron environment:

```bash
target_env=/absolute/path/to/megatron-target-venv
VIRTUAL_ENV="$target_env" PATH="$target_env/bin:$PATH" CUDA_VISIBLE_DEVICES=0,1 \
  "$target_env/bin/python" -m aspec._environment "$target_env/bin/torchrun" \
  --standalone --nproc_per_node=2 \
  /absolute/path/to/tools/validation/megatron_target_aspec_alignment.py \
  --model /absolute/path/to/Qwen3-8B \
  --reference-cache /absolute/path/to/aspec-eval-cache.pt \
  --length 4096 --output /tmp/target-tp2.json
```

This compares requested residual captures and final post-norm features with the
existing ASpec/SGLang producer, using the same checkpoint and token IDs. The
cache's producer and value-stage identity are checked. Choose an SGLang cache produced
at the same TP size as the Target for the matched comparison.

To exercise online training and recovery with pretokenized Parquet inputs:

```bash
VIRTUAL_ENV="$meg_env" PATH="$meg_env/bin:$PATH" CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 \
  "$meg_env/bin/python" -m aspec._environment "$meg_env/bin/python" \
  -m tools.validation.megatron_pipeline_acceptance \
  --model /absolute/path/to/Qwen3-8B \
  --train-data /absolute/path/to/train.parquet \
  --eval-data /absolute/path/to/eval.parquet \
  --target-python /absolute/path/to/megatron-target-venv/bin/python \
  --target-tp 2 --draft-tp 2 --dp 2 --accumulation 2 \
  --sequence-length 4096 --anchors 512 --steps 4 \
  --output /tmp/megatron-online-check
```

This uses the ordinary launcher and recipe audits, restarts the driver/workers
from a native checkpoint, checks state/RNG/work restoration, continues training,
and exports an asset for strict loading. Use compiler caches on a filesystem
supporting the toolchain's concurrent cache operations. Functional pipeline
completion and Target feature equivalence remain separate gates.

## Validated configurations

The deterministic online comparison on SM90 used 200 Open PerfectBlend prompts
with Qwen3-8B responses, two samples per optimizer update, a 4096-token context
limit and 512 anchors. DP1 accumulated two microbatches and DP2 accumulated one
per replica. Each pair used the same Target TP size, Draft topology, initial
weights and BF16 gradients.

| Draft topology | SGLang Target + Torch Draft mean objective | Megatron Target + Megatron Draft mean objective | Max absolute per-step rel diff |
| --- | ---: | ---: | ---: |
| TP1/DP1 | 2.83849453 | 2.83849453 | 0% |
| TP2/DP1 | 2.83903998 | 2.83903998 | 0% |
| TP2/DP2 | 2.85058107 | 2.85058107 | 0% |

The objective is 0.1 × CE + 0.9 × L1 + confidence loss, averaged over 100
optimizer updates. Final Draft weights, FP32 masters and Adam moments matched
at every topology. For TP1, set `inference.sglang.extra_args.prefill_max_requests=1`
in the reference profile to match Megatron's single-request prefill. Batching
requests changes the token-row count of the first MLP down projection and can
change BF16 rounding.

The two TP2 reference runs used live SGLang prefill and the validation-only
native Torch DTensor Draft. Each run had one two-GPU Target and either two or
four Draft GPUs. The original observer revision (SHA-256) was
`1c5fea43a3fe6859f802bee6bc2eac2b06d70c606d5425fb18bccba024bcce64`. The run evidence identifies
the installed ASpec source trees by digest: Torch
`e79301553a64bfd6cf12d9e50047e9d2dc82c29ddb8b0976943c2e3d39fcd30f` and Megatron
`26d72e38a3fc8e492a283acb7f6ba608dee3bf4ca74c9e70085a609d19458e8b`. The [per-update objectives](../benchmarks/megatron-online-tp2.csv)
contain both measurements and their absolute relative difference. Each objective
pools the two microbatch numerators and denominators, counting each TP replica once.

Run the following Bash commands from the root of the ASpec checkout. Set the
paths to the Draft environments, Target interpreters, Qwen3-8B checkpoint,
pretokenized Parquet files and a fresh output directory. Both backends use the
same training and evaluation files. Each invocation starts its own managed Target and writes
`objectives.json` in a fresh directory. The Torch validation plan uses Megatron's
TP2 scheduling topology and selects native DTensor operations in the Draft workers.

```bash
torch_env=/absolute/path/to/torch-draft-venv
meg_env=/absolute/path/to/megatron-draft-venv
sgl_target=/absolute/path/to/sglang-target-venv/bin/python
meg_target=/absolute/path/to/megatron-target-venv/bin/python
model=/absolute/path/to/Qwen3-8B
train=/absolute/path/to/train.parquet
eval_data=/absolute/path/to/eval.parquet
out=/absolute/path/to/fresh-online-runs
export CUDA_DEVICE_MAX_CONNECTIONS=1
for dp in 1 2; do
  devices=0,1,2,3
  if [[ $dp == 2 ]]; then devices=0,1,2,3,4,5; fi
  for draft in torch megatron; do
    env_dir=$torch_env
    target_python=$sgl_target
    if [[ $draft == megatron ]]; then env_dir=$meg_env; target_python=$meg_target; fi
    VIRTUAL_ENV="$env_dir" PATH="$env_dir/bin:$PATH" CUDA_VISIBLE_DEVICES="$devices" \
      "$env_dir/bin/python" -m aspec._environment \
      "$env_dir/bin/python" -m tools.validation.megatron_online_alignment \
      --draft "$draft" --dp "$dp" --model "$model" --train-data "$train" --eval-data "$eval_data" \
      --target-python "$target_python" --steps 100 --output "$out/online-$draft-dp$dp"
  done
done
```

The independent fixed-feature checks at TP2/DP1 and TP2/DP2 verified that
restoring after update 50 reproduced the remaining 50 updates and final state
on each backend. Online mixed-backend checks covered SGLang Target with
Megatron Draft, Megatron Target with Torch Draft, feature transport, export
and strict native loading. The 4K fixture exercised CuTeDSL Q/K preparation,
tiled FP32 L1 and fused optimizer clipping. Multi-chunk projection caching,
optional backbone checkpointing, throughput, peak memory and held-out quality
require separate measurements.
