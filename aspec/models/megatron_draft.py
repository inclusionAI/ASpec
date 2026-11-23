# Copyright 2026 Ant Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Megatron TP projections around the retained optimized DSpark computation.

The vocabulary objective and its bounded/tiled caches remain replicated in this
CP1 implementation. Only attention heads and dense FFN channels are partitioned.
This avoids gathering vocabulary logits or changing global probability semantics.
"""

import torch
from torch.distributed.tensor import DTensor, Shard

from aspec.models.megatron import require_megatron


def draft_transformer_config(config, tp_size, cp_size=1):
    require_megatron()
    from megatron.core.transformer.transformer_config import TransformerConfig

    for name in ("num_attention_heads", "num_key_value_heads", "intermediate_size"):
        if getattr(config, name) % tp_size:
            raise ValueError(f"Draft {name} must be divisible by TP{tp_size}")
    return TransformerConfig(
        num_layers=config.num_hidden_layers, hidden_size=config.hidden_size,
        ffn_hidden_size=config.intermediate_size,
        num_attention_heads=config.num_attention_heads,
        num_query_groups=config.num_key_value_heads,
        kv_channels=int(getattr(config, "head_dim", None)
                        or config.hidden_size // config.num_attention_heads), normalization="RMSNorm",
        layernorm_epsilon=config.rms_norm_eps, gated_linear_unit=True,
        activation_func=torch.nn.functional.silu, add_bias_linear=False,
        params_dtype=torch.bfloat16, bf16=True,
        tensor_model_parallel_size=tp_size, context_parallel_size=cp_size,
        pipeline_model_parallel_size=1, sequence_parallel=False,
        perform_initialization=False, use_cpu_initialization=False,
        gradient_accumulation_fusion=False, attention_dropout=0.0, hidden_dropout=0.0,
    )


def checkpoint_tensor(tensor, parameter, tp_mesh):
    """Expose the logical tensor to DCP without changing compute parameters."""
    axis = getattr(parameter, "_aspec_tp_axis", None)
    if isinstance(tensor, DTensor) or axis is None or tensor.ndim == 0:
        return tensor
    return DTensor.from_local(tensor, device_mesh=tp_mesh, placements=(Shard(axis),),
                              run_check=False)


def partition_dspark(draft, core_config, groups, tp_mesh):
    from megatron.core.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear

    class Column(ColumnParallelLinear):
        def forward(self, value):
            output, bias = super().forward(value)
            if bias is not None:
                raise RuntimeError("DSpark TP projections must be bias-free")
            return output

    class Row(RowParallelLinear):
        def forward(self, value):
            output, bias = super().forward(value)
            if bias is not None:
                raise RuntimeError("DSpark TP projections must be bias-free")
            return output

    rank, size = groups.tp.rank(), groups.tp.size()

    def replace(owner, name, axis):
        original = getattr(owner, name)
        if original.bias is not None:
            raise ValueError("Megatron DSpark requires bias-free projections")
        options = {"gather_output": False} if axis == 0 else {"input_is_parallel": True}
        layer_type = Column if axis == 0 else Row
        replacement = layer_type(
            original.in_features, original.out_features, config=core_config,
            init_method=core_config.init_method, bias=False, skip_bias_add=False,
            tp_group=groups.tp, **options,
        )
        with torch.no_grad():
            replacement.weight.copy_(original.weight.chunk(size, dim=axis)[rank])
        replacement.weight.requires_grad_(original.weight.requires_grad)
        replacement.weight._aspec_tp_axis = axis
        setattr(owner, name, replacement)

    for layer in draft.layers:
        attention = layer.self_attn
        for name in ("q_proj", "k_proj", "v_proj"):
            replace(attention, name, 0)
        replace(attention, "o_proj", 1)
        attention.num_heads //= size
        attention.num_kv_heads //= size
        for name in ("gate_proj", "up_proj"):
            replace(layer.mlp, name, 0)
        replace(layer.mlp, "down_proj", 1)
    draft.cuda()
    for layer in draft.layers:
        for norm in (layer.self_attn.q_norm, layer.self_attn.k_norm):
            norm.weight._aspec_tp_partial_gradient = True
    draft._aspec_tp_mesh = tp_mesh
    draft._aspec_tp_group = groups.tp
    return draft


def model_checkpoint_state(model, tp_mesh):
    parameters = dict(model.named_parameters())
    return {name: checkpoint_tensor(value, parameters[name], tp_mesh)
            if name in parameters else value
            for name, value in model.state_dict().items() if isinstance(value, torch.Tensor)}


def load_model_checkpoint_state(model, state):
    current = {name: value for name, value in model.state_dict().items()
               if isinstance(value, torch.Tensor)}
    if set(current) != set(state):
        raise ValueError("Megatron checkpoint does not cover the complete DSpark state")
    with torch.no_grad():
        for name, value in state.items():
            local = value.to_local() if isinstance(value, DTensor) else value
            if current[name].shape != local.shape:
                raise ValueError(f"Megatron checkpoint shape mismatch: {name}")
            current[name].copy_(local)
