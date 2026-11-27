# DSpine Draft model and objective

This is the model/loss prerequisite for [ASpec #37](https://github.com/inclusionAI/ASpec/issues/37).
It adds DSpine adjacent-token injection, persistent transfer codes and candidate
refinement on the retained DFlash block backbone. Trainer, recipe, exact resume,
export and serving are follow-ups; this page describes the tested model/loss scope
only.

## References

- [Tencent/DSpine](https://github.com/Tencent/DSpine) @
  `75cade7ab0307fc2daac8eda95337705226b217e` for the architecture, inference
  implementation and published checkpoint layout.
- [SpecForge DSpine training](https://github.com/sgl-project/SpecForge/pull/965)
  merge `eaa6b03f6b4a65f38954fa0ebc13e59a8f295d02` and the objective corrections in
  [PR #1038](https://github.com/sgl-project/SpecForge/pull/1038) merge
  `1b5c0dde137a28447606ffb6078e01eddfa77980`.
- SpecForge [issue #1170](https://github.com/sgl-project/SpecForge/issues/1170)
  documents a stale-config hazard when replacing a draft config after construction.
  ASpec reads the resolved `DSpineConfig` at use time and does not need a mutable
  online-model alias.

## Selected scope

The first implementation targets one dense full-attention DFlash backbone such as
Qwen3-8B with padded inputs, `draft_context_parallel_size=1` and no packing. Native
MTP conditioning, MoE FFNs, sparse attention, packed documents and CP>1 fail closed.

## Components

- `DSpineConfig(DFlashConfig)` requires causal draft attention and validates the
  transfer rank, message size, top-k width, replacement schedule and alignment
  weights.
- `AdjacentInjection` owns the persistent `transfer_codes` table, per-layer readouts,
  gated predecessor messages and per-layer feature writes. Each block's anchor code
  and the previous position's feature condition the next position's write.
- `DSpineDraftModel.initialize_transfer_space` builds whitened token codes and
  readout directions from the frozen Target output head.
- `DSpineDraftModel.refine` applies the exact last-write refinement used for candidate
  rescoring.
- `DSpineModel` reuses ASpec anchor sampling, block masks, Target features and DFlash
  reductions. It adds backbone CE/L1, candidate refinement CE/L1, transfer-code
  alignment, the alignment warmup and the predecessor-replacement schedule.

The current objective is dense full attention, padded, CP1, non-packing. The
trainer-side accumulation replay and exact-resume contract are not part of this
increment.

## Validation and limits

`tests/cpu/test_dspine.py` covers configuration fail-closed cases, transfer-space
initialization, finite forward/backward, adjacent-injection causality and packed-input
rejection. CPU numerical contracts do not establish Qwen3-8B convergence, GPU
throughput or decoding acceptance.

The published DSpine checkpoint layout uses a `dflash_config` code bridge and a
serving-specific candidate sampler. Mapping those fields, trainer wiring, export,
consumer/acceptance and a bounded SpecForge comparison are follow-up work.
