# Copyright 2026 Ant Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Public Megatron Core construction and strict Qwen3 weight mapping.

Uses Megatron Core 0.20's public model/spec/process-group interfaces. Qwen3
mapping follows the grouped QKV and gate/up layouts documented by NVIDIA's
Megatron Bridge Qwen3 adapter.
"""

from __future__ import annotations

import importlib.metadata
import json
from pathlib import Path

import torch
import torch.distributed as dist
from safetensors import safe_open

from aspec.config.megatron import MEGATRON_VERSION


def require_megatron():
    from packaging.version import Version

    try:
        version = importlib.metadata.version("megatron-core")
    except importlib.metadata.PackageNotFoundError as exc:
        raise RuntimeError("Install the pinned ASpec Megatron environment") from exc
    if Version(version).base_version != MEGATRON_VERSION:
        raise RuntimeError(f"Megatron Core {MEGATRON_VERSION} is required; found {version}")
    return version


def initialize_parallelism(tp_size: int, *, seed: int = 42):
    """The caller's default process group contains only one execution role."""
    require_megatron()
    from megatron.core import parallel_state
    from megatron.core.process_groups_config import ProcessGroupCollection

    if not dist.is_initialized():
        raise RuntimeError("Initialize the role-local process group before Megatron")
    if parallel_state.model_parallel_is_initialized():
        raise RuntimeError("A Megatron model-parallel world is already initialized")
    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size=tp_size,
        pipeline_model_parallel_size=1,
        context_parallel_size=1,
        create_gloo_process_groups=False,
    )
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

    model_parallel_cuda_manual_seed(seed)
    return ProcessGroupCollection.use_mpu_process_groups()


def transformer_config(config, tp_size: int):
    from megatron.core.transformer.transformer_config import TransformerConfig

    hidden = int(config.hidden_size)
    heads = int(config.num_attention_heads)
    groups = int(config.num_key_value_heads)
    intermediate = int(config.intermediate_size)
    if any(value % tp_size for value in (heads, groups, intermediate, config.vocab_size)):
        raise ValueError("Qwen3 heads, KV groups, FFN and vocabulary must divide Target TP")
    if config.model_type != "qwen3" or getattr(config, "num_experts", None):
        raise ValueError("Megatron Target currently requires dense Qwen3")
    layer_types = getattr(config, "layer_types", None)
    if layer_types is not None:
        full_attention = all(kind == "full_attention" for kind in layer_types)
    else:
        full_attention = not (
            getattr(config, "use_sliding_window", False) and getattr(config, "sliding_window", None)
            and getattr(config, "max_window_layers", 0) < config.num_hidden_layers
        )
    if not full_attention:
        raise ValueError("Megatron Qwen3 Target requires full attention in every layer")
    if getattr(config, "attention_bias", False):
        raise ValueError("Megatron Qwen3 prefill requires bias-free attention projections")
    if getattr(config, "quantization_config", None) or getattr(config, "hidden_act", "silu") != "silu":
        raise ValueError("Megatron Qwen3 requires unquantized dense SwiGLU weights")
    rope = getattr(config, "rope_parameters", None) or {}
    scaling = getattr(config, "rope_scaling", None) or rope
    if scaling.get("rope_type", scaling.get("type", "default")) != "default":
        raise ValueError("Megatron Qwen3 currently requires the original default RoPE policy")
    return TransformerConfig(
        num_layers=2 * int(config.num_hidden_layers),
        hidden_size=hidden,
        ffn_hidden_size=intermediate,
        num_attention_heads=heads,
        num_query_groups=groups,
        kv_channels=int(getattr(config, "head_dim", None) or hidden // heads),
        normalization="RMSNorm",
        layernorm_epsilon=float(config.rms_norm_eps),
        qk_layernorm=True,
        gated_linear_unit=True,
        activation_func=torch.nn.functional.silu,
        add_bias_linear=False,
        add_qkv_bias=False,
        attention_dropout=0.0,
        hidden_dropout=0.0,
        tensor_model_parallel_size=tp_size,
        pipeline_model_parallel_size=1,
        context_parallel_size=1,
        params_dtype=torch.bfloat16,
        bf16=True,
        perform_initialization=False,
        use_cpu_initialization=False,
        gradient_accumulation_fusion=False,
    )


class SafetensorReader:
    """Read one source parameter at a time, with an explicit model identity."""

    def __init__(self, path):
        from huggingface_hub import snapshot_download

        self.root = Path(path)
        if not self.root.is_dir():
            self.root = Path(snapshot_download(path, allow_patterns=["*.json", "*.safetensors"]))
        index = self.root / "model.safetensors.index.json"
        if index.is_file():
            self.files = json.loads(index.read_text())["weight_map"]
        else:
            with safe_open(self.root / "model.safetensors", framework="pt", device="cpu") as source:
                self.files = {key: "model.safetensors" for key in source.keys()}
        self.read = set()

    def tensor(self, name):
        if name not in self.files:
            raise ValueError(f"Target checkpoint is missing required parameter {name}")
        self.read.add(name)
        with safe_open(self.root / self.files[name], framework="pt", device="cpu") as source:
            return source.get_tensor(name)


def _shard(value, axis, rank, size):
    if value.shape[axis] % size:
        raise ValueError(f"Tensor dimension {axis} cannot be partitioned over TP{size}")
    return value.chunk(size, dim=axis)[rank].contiguous()


def qwen3_parameter(reader, name, config, rank, tp_size):
    """Map public MCore parameter names, preserving native Qwen3 values."""
    if name == "embedding.word_embeddings.weight":
        return _shard(reader.tensor("model.embed_tokens.weight"), 0, rank, tp_size)
    if name == "final_norm.weight":
        return reader.tensor("model.norm.weight")
    pieces = name.split(".")
    if pieces[:2] != ["decoder", "layers"] or len(pieces) < 5:
        raise ValueError(f"Unsupported Megatron Qwen3 parameter: {name}")
    prefix = f"model.layers.{int(pieces[2]) // 2}."
    suffix = ".".join(pieces[3:])
    aliases = {
        "input_layernorm.weight": "input_layernorm.weight",
        "pre_mlp_layernorm.weight": "post_attention_layernorm.weight",
        "self_attention.q_layernorm.weight": "self_attn.q_norm.weight",
        "self_attention.k_layernorm.weight": "self_attn.k_norm.weight",
    }
    if suffix in aliases:
        return reader.tensor(prefix + aliases[suffix])
    if suffix == "self_attention.linear_qkv.weight":
        groups = int(config.num_key_value_heads)
        heads_per_group = int(config.num_attention_heads) // groups
        width = int(getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads)
        query = reader.tensor(prefix + "self_attn.q_proj.weight").view(groups, heads_per_group, width, -1)
        key = reader.tensor(prefix + "self_attn.k_proj.weight").view(groups, 1, width, -1)
        value = reader.tensor(prefix + "self_attn.v_proj.weight").view(groups, 1, width, -1)
        return _shard(torch.cat((query, key, value), dim=1), 0, rank, tp_size).flatten(0, 2)
    if suffix == "self_attention.linear_proj.weight":
        return _shard(reader.tensor(prefix + "self_attn.o_proj.weight"), 1, rank, tp_size)
    if suffix == "mlp.linear_fc1.weight":
        return torch.cat([
            _shard(reader.tensor(prefix + "mlp." + part + ".weight"), 0, rank, tp_size)
            for part in ("gate_proj", "up_proj")
        ], dim=0)
    if suffix == "mlp.linear_fc2.weight":
        return _shard(reader.tensor(prefix + "mlp.down_proj.weight"), 1, rank, tp_size)
    raise ValueError(f"Unsupported Megatron Qwen3 parameter: {name}")


def load_qwen3_target(path, groups, *, implementation="transformer_engine"):
    """Load the frozen feature-only HybridModel, including its final norm."""
    from transformers import AutoConfig

    from aspec.models.megatron_target import build_target

    reader = SafetensorReader(path)
    config = AutoConfig.from_pretrained(reader.root, trust_remote_code=False)
    tp_size = dist.get_world_size(groups.tp)
    rank = dist.get_rank(groups.tp)
    core_config = transformer_config(config, tp_size)
    model = build_target(config, core_config, groups, implementation)
    model.to(dtype=torch.bfloat16)
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            value = qwen3_parameter(reader, name, config, rank, tp_size)
            if tuple(value.shape) != tuple(parameter.shape):
                raise ValueError(f"Qwen3 mapping shape mismatch for {name}: {value.shape} != {parameter.shape}")
            parameter.copy_(value)
    unused = set(reader.files) - reader.read
    # The feature-only Target never constructs the vocabulary projection.
    unused.discard("lm_head.weight")
    if unused:
        raise ValueError(f"Unmapped Target checkpoint parameters: {sorted(unused)}")
    model.eval().requires_grad_(False)
    return model, config
