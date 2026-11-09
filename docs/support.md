# Supported paths and limitations

ASpec is a source release for trusted Linux GPU training environments. The primary performance
path is SGLang + DSpark/DFlash + Qwen3 on multi-GPU systems. Training requires CUDA GPUs;
configuration planning can run on a CPU. See the [installation guide](user-guide/installation.md)
for the supported dependencies.

## Algorithms and structures

| Path | Implemented behavior | Boundary |
| --- | --- | --- |
| DSpark / DFlash | Training, exact recovery, evaluation and export; Dense and selected MoE/attention structures | Objectives, feature bindings and assets remain algorithm-specific |
| EAGLE3 | Recurrent TTT, full/pruned vocabulary, GQA/MHA or materialized MLA; FA4 packing/CP | No EAGLE3 MoE, anchored mode or whole-model compile |
| EAGLE4 | Dense and selected upstream MoE; Draft/candidate stages; packing/CP and native assets | Candidate Top-K forward-KL is explicitly implemented from public semantics, not an unpublished official recipe; no serving integration |
| DFlash2 | Experimental full/causal/SWA training with Flex, cuDNN THD or FA4 packing/CP | Experimental objective; no claim to reproduce an official training recipe |
| MTP-conditioned DSpark / DFlash | Frozen native Qwen3.5 predictor, token/hidden dual-anchor training, exact recovery and strict native assets | Dense full GQA, CP1, MBS1, REPLICATE, one node, non-packing, decay objective; no generic serving consumer |
| Native MTP | Qwen3.5-9B native single predictor, same-Target frozen embedding/head, multi-step training, recovery and export/stitch | Text-only Dense MTP; SDPA CP1 or independently validated FA4 packing/CP; no generic multi-layer/MoE MTP |
| Dense / Qwen MoE / Ling MoE | Structure-specific FFNs, routing state, checkpointing and strict assets | No promise of every algorithm × structure combination |
| MHA/GQA / materialized MLA | Explicit head geometry and tensor bindings | MLA materializes K/V; no absorbed-MLA claim |
| SWA / CSA / QSA / DSA | Explicit window or staged compressed/indexed structures; native assets where required | DSA uses per-query raw-token selection on DSpark/DFlash GQA/MHA; no CSA2 or arbitrary structure combinations; short contexts may be slower |

Use [recipes](user-guide/recipes.md), [algorithm details](user-guide/algorithms.md) and
[structure details](user-guide/structures.md) to choose a concrete configuration. A configuration
passing Plan validation establishes legality, not performance, convergence or downstream quality.

## Execution and assets

- Single-node online training supports required NVLink and explicitly selected Store/transport paths.
  With packing enabled, DSpark/DFlash full and sliding attention preserve document semantics with
  FA4 or native cuDNN flattened-token execution at CP1 and above. Selected-entry CSA/QSA/DSA
  packing/CP remains FA4-only.
- Shared Target capture supports independent Drafts, progress, evaluation, recovery and tracking.
  Both single-node and static homogeneous multi-node sharing support independent packing/CP views.
  Multi-node sharing requires one original document per Target request and node-local feature affinity.
  Draft weights and optimizer state remain independent; online weight synchronization is unsupported.
- Static homogeneous multi-node single-Draft training supports node-local model parallelism,
  explicit transports and required node-local feature affinity. Elastic membership and cross-node
  Target TP are unsupported.
- Offline feature replay is single-node, single-Draft, CP1 with CPU Store/TCP transport.
- SGLang export is accepted only when the actual structure can be faithfully represented. CSA/QSA/DSA,
  materialized MLA and EAGLE4 use native assets where required; strict loading/proposal components
  do not provide a serving scheduler or arbitrary Target-cache compatibility.

## Known limitations

Measured training-loss and verifier-acceptance results are available in the
[Qwen3-8B DSpark comparison](benchmarks/dspark-alignment.md) and
[CP comparison](benchmarks/dspark-cp-alignment.md). Each report lists its hardware,
runtime settings and workloads.

- Automatic recovery from CUDA driver, DMA or RDMA hangs is unsupported.
- In single-Draft static multi-node training, worker loss stops the current attempt. Fence the old
  processes, restore the worker and resume a new attempt from a complete checkpoint; see
  [worker recovery](contributing/validation.md#whole-worker-node-loss-and-recovery).
- Reports use `incomplete_evidence` when a run ends before required pipeline activity is observed.
- SGLang may emit `resource_tracker` warnings during repeated engine startup and shutdown.
- Benchmark results apply to the software versions and settings listed in each report. See
  [benchmark guidance](user-guide/benchmarks.md) for throughput and quality measurements.

### Decode reproducibility

Repeated greedy evaluation of the Qwen3-8B DSpark recipe has produced different output token IDs
under default SGLang inference settings with radix cache enabled. Default-inference repeatability
is therefore not established.

For controlled comparisons, use `--deterministic-inference` on the evaluator. The
[framework comparison](benchmarks/dspark-alignment.md) and separate
[CP quality comparison](benchmarks/dspark-cp-alignment.md) use this option and normal
radix cache: output token IDs match across Target-only decoding, the reported Drafts
and all repeats. That result applies to its stated
configuration, not to the default inference mode. See [evaluation](user-guide/evaluation.md) for
the command and failure reporting.
