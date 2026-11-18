# Megatron environments and execution boundary

The Megatron adapter targets upstream **Megatron Core 0.20.0**, tag `core_v0.20.0`,
commit `685fb8bd970185e9ac3283fe32a5d6667bf4b268`. ASpec does not import an internal
Megatron fork or modify Core. The current integration is under validation; use the
checks below before treating a topology as accepted.

The initial scope is dense Qwen3 Target prefill and dense, full-GQA DSpark Draft
training, BF16, MBS1, CP1/PP1 on one node. Target and Draft select their execution backends
independently. Shared Target, Megatron Draft THD packing, CP, Target PP, decode and MTP conditioning
are outside this adapter's current boundary and are rejected.

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
Transformer Engine 2.20.2, FA4 b34, CUTLASS DSL 4.7.1 and Quack 0.6.5. The ordinary
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

This describes the new release artifacts, not an already published PyPI release.
Models, internal source trees, CUDA drivers and machine-specific compatibility
libraries are not included in ASpec packages. The host needs a compatible driver
and the native prerequisites in the [installation guide](installation-advanced.md).

Each process prepares native-library paths from its own interpreter. The optional
local supervisor removes automatically selected Draft library paths before starting
Target. A separate Target service requires a service address; managed-local mode
also requires an explicit Target interpreter. Neither role imports the other's
Python dependency graph or transfers Python model objects across the boundary.

## Target and Draft adapters

Megatron Target constructs an upstream GPT model from public layer specs. The
Qwen3 adapter maps HF grouped Q/K/V and gate/up weights strictly, including TP
partitions. Its norm spec retains Qwen3's BF16 normalize/cast/weight order.
Changing that rounding order can change the intermediate residual features even
when the final normalized output is close.

Target runs frozen prefill with requested post-layer residual features and the
final normalized hidden state. A public output processor omits vocabulary-logit
materialization. Capture writes directly into bounded transport storage. The
initial implementation executes one sequence at a time in each engine's TP world;
request batching does not imply a fused multi-sequence prefill kernel.

Megatron Draft partitions attention heads and dense FFN channels using public
Core TP projections, and uses Core DDP gradient buffers with ASpec's FP32-master
AdamW update and schedule. The retained DSpark graph still owns attention/QK
kernels, block compaction, bounded projection caches, tiled/indexed L1, statistics,
selective recomputation and deferred metrics. Vocabulary, Markov and confidence
computations remain replicated across TP. This preserves the existing global
softmax/L1 semantics; vocabulary sharding is not a claimed capability.

One TP leader prefetches source GETs into bounded, preallocated destination rings.
TP broadcasts and collation run on the consumer thread between microbatches, in
the same order as model collectives. Availability and tensor metadata use a CPU
Gloo group; no background reader launches NCCL work. Training and evaluation own
separate input groups. With source prefetch enabled, destination slots must cover
`prefetch_depth + 2` samples, including the active consumer and an in-progress GET.
The destination lease remains attached until the consumer finishes. Source release
occurs once per DP replica. Anchors/keep masks are shared from the TP leader;
commit collection checks matching work identity and reports it once per replica.

Gradients are normalized by the complete optimizer window's supervision weight,
pooled across DP. TP-partial Q/K norm gradients are reduced once, and clipping
counts each replicated parameter once. Checkpoints describe logical TP shards and
restore model, FP32 masters, moments, schedule and both ASpec and Core RNG streams.
Exact resume requires the same Draft backend, TP topology and world size. Cross-backend weights-only
initialization is a separate stage. Native export gathers the logical Draft weights
for the existing strict loader.

## First functional path

Use the existing Draft venv and a separate Megatron Target venv. Select one GPU
for each role and a fresh output directory:

```bash
source /absolute/path/to/draft-venv/bin/activate
CUDA_VISIBLE_DEVICES=0,1 aspec quickstart qwen3-8b \
  --target /absolute/path/to/Qwen3-8B --output outputs/megatron-target-001 \
  --managed-target /absolute/path/to/megatron-target-venv/bin/python \
  --set model.target_model_backend=megatron \
  --set inference.inference_engine_type=megatron
```

This uses Megatron Target with the existing FSDP Draft and exercises training,
fresh-process state recovery, further training, native export and strict loading.
For a Megatron Draft, activate its own Megatron venv and add
`--set training.train_backend=megatron --set mooncake.destination_gpu_ring_slots=3`.
To pair that Draft with SGLang, use the
SGLang Target interpreter and keep the default Target backend/engine. The existing
Quickstart's two-GPU budget covers TP1 for both roles.

For TP2, use the ordinary training configuration with
`training.draft_tensor_parallel_size=2` and two Draft GPUs per DP replica.
Megatron Target independently uses `inference.megatron.tp_size=2` and
`inference.inference_num_gpus_per_engine=2`. Source ring capacity counts engines
and DP replicas, rather than counting every TP rank as a separate sample owner.

These are functional checks on bundled conversations, not held-out quality or
throughput results. Checkpoint state and RNG can restore exactly while a later
backward pass differs because of kernel reduction order; future bitwise replay
is measured separately.

## Validation commands

Run on allocated GPUs with the Megatron venv activated:

```bash
torchrun --standalone --nproc_per_node=2 -m tools.validation.megatron_target_alignment \
  --model /absolute/path/to/Qwen3-8B --sequence-length 4096 --output /tmp/target-tp2.json
torchrun --standalone --nproc_per_node=2 -m tools.validation.megatron_draft_alignment \
  --model /absolute/path/to/Qwen3-8B --tp 2 --output /tmp/draft-tp2-check \
  --attention-backend cudnn_flex_attention --reference-tp-projections
```

Use ASpec's environment bootstrap when invoking the tools directly:
`python -m aspec._environment /absolute/path/to/venv/bin/torchrun ...`.
The Target check compares actual HF features and verifies frozen parameters.
The Draft check uses synthetic fixed features with real frozen model bindings;
it compares loss, gradients, updates and checkpoint replay. It does not establish
online transport, committed-work recovery, throughput or model quality. Complete
online pairing and lifecycle gates remain separate acceptance requirements.

The default Draft reference uses unpartitioned GEMMs. The optional independent
PyTorch TP reference retains BF16 partial-output rounding and groups each
projection's input-gradient sum before other consumers accumulate into that input.
It uses neither Core TP operators nor distributed communication. Raw comparisons
against unpartitioned GEMMs remain separate: BF16 TP rounding can produce small
gradient differences that Adam amplifies into larger update differences. The
tool also checks clipping/Adam with the same gradients and records checkpoint
load fidelity, in-memory replay and checkpoint replay independently. A failed
future-bitwise or unpartitioned-update gate remains a failed gate.
