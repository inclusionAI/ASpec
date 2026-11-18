# Copyright 2026 Ant Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Dependency-light capability boundary for the upstream Megatron integration."""

MEGATRON_VERSION = "0.20.0"
MEGATRON_REVISION = "685fb8bd970185e9ac3283fe32a5d6667bf4b268"


def validate_megatron_execution(args, draft):
    """Reject combinations outside the runnable dense DSpark CP1/PP1 lane."""
    training = getattr(args, "train_backend", "fsdp")
    target = getattr(args, "target_model_backend", "sglang")
    tp = getattr(args, "draft_tensor_parallel_size", 1)
    if type(tp) is not int or tp < 1:
        raise ValueError("training.draft_tensor_parallel_size must be a positive integer")
    if training != "megatron" and tp != 1:
        raise ValueError("Draft TP requires training.train_backend=megatron")
    if training != "megatron" and target != "megatron":
        return
    if getattr(args, "cluster_mode", "single_node") != "single_node":
        raise ValueError("Megatron currently requires single-node execution")
    if getattr(args, "aspec_shared_source", False) or getattr(args, "aspec_shared_reader", None):
        raise ValueError("Megatron shared Target execution is not supported yet")
    if draft["family"] != "dspark":
        raise ValueError("Megatron currently supports the DSpark objective")
    if draft.get("mtp_conditioning"):
        raise ValueError("Megatron currently requires an unconditioned DSpark Draft")
    if getattr(args, "draft_context_parallel_size", 1) != 1:
        raise ValueError("Megatron currently requires Draft CP1")
    if target == "megatron":
        if getattr(args, "sglang_extra_args", None) or getattr(args, "sglang_allow_context_length_override", False):
            raise ValueError("Megatron Target does not apply SGLang backend overrides")
        if getattr(args, "megatron_implementation", "transformer_engine") not in {"local", "transformer_engine"}:
            raise ValueError("Unknown Megatron model implementation")
        if not getattr(args, "target_prefill_memory_optimized", False):
            raise ValueError("Megatron Target requires target_prefill_memory_optimized=true")
        if getattr(args, "sglang_nnodes", 1) != 1 or getattr(args, "sglang_pp_size", 1) != 1:
            raise ValueError("Megatron Target currently requires node-local TP and PP1")
        if getattr(args, "train_with_decode", False):
            raise ValueError("Megatron Target supports prefill-only execution")
        target_tp = getattr(args, "megatron_tp_size", 1)
        if type(target_tp) is not int or target_tp < 1:
            raise ValueError("inference.megatron.tp_size must be a positive integer")
        if target_tp not in {1, 2}:
            raise ValueError("Megatron Target currently validates TP1/TP2")
        if target_tp != getattr(args, "inference_num_gpus_per_engine", 1):
            raise ValueError("inference.megatron.tp_size must equal inference_num_gpus_per_engine")
        slots = (getattr(args, "mooncake_source_gpu_ring_slots", None)
                 or getattr(args, "mooncake_gpu_ring_slots", 2))
        if getattr(args, "inference_batch_size", 1) > slots:
            raise ValueError("Megatron Target source ring must retain one complete inference batch")
    if training == "megatron":
        if getattr(args, "distributed_backend", "nccl") != "nccl":
            raise ValueError("Megatron Draft requires the NCCL distributed backend")
        if getattr(args, "micro_batch_size", 1) != 1:
            raise ValueError("Megatron Draft currently requires micro_batch_size=1")
        prefetch = max(int(getattr(args, "prefetch_depth", 0)), 0)
        destination_slots = (getattr(args, "mooncake_destination_gpu_ring_slots", None)
                             or getattr(args, "mooncake_gpu_ring_slots", 2))
        if prefetch and destination_slots < prefetch + 2:
            raise ValueError("Megatron source prefetch requires destination_gpu_ring_slots >= prefetch_depth + 2")
        if tp not in {1, 2}:
            raise ValueError("Megatron Draft currently validates TP1/TP2")
        if getattr(args, "inference_engine_type", "sgl") == "offline":
            raise ValueError("Offline replay with Megatron Draft is not validated")
        if getattr(args, "fsdp_strategy", "REPLICATE") != "REPLICATE":
            raise ValueError("Megatron Draft currently uses replicated DDP rather than FSDP sharding")
        if getattr(args, "fsdp_cpu_offload", False):
            raise ValueError("Megatron Draft CPU offload is not supported")
        if getattr(args, "world_size", 1) % tp:
            raise ValueError("Draft world size must be divisible by draft_tensor_parallel_size")
        if getattr(args, "dflash_packing", False):
            raise ValueError("Megatron Draft packed execution is not supported yet")
        if getattr(args, "compile_model", False):
            raise ValueError("Megatron Draft currently compiles individual optimized kernels only")
        if draft.get("structure", {}).get("ffn_type", "dense") != "dense":
            raise ValueError("Megatron Draft currently requires a dense FFN")
        attention = draft.get("attention", {})
        if (attention.get("head_layout", {}).get("type", "gqa") not in {"gqa", "mha"}
                or any(window is not None for window in attention.get("context_windows", []))
                or any(attention.get(kind) is not None for kind in ("csa", "qsa", "dsa"))):
            raise ValueError("Megatron Draft currently requires dense full GQA/MHA attention")
