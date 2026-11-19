# Copyright 2026 Ant Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Frozen single-sequence Qwen3 prefill on the upstream Core HybridStack."""

from contextvars import ContextVar
from dataclasses import dataclass
from functools import partial

import torch
from flashinfer.norm import fused_add_rmsnorm, rmsnorm
from flashinfer.rope import apply_rope_with_cos_sin_cache_inplace
from megatron.core.models.backends import get_backend
from megatron.core.models.hybrid.hybrid_block import HybridStack, HybridStackSubmodules
from megatron.core.models.hybrid.hybrid_model import HybridModel
from megatron.core.transformer.attention import SelfAttention, SelfAttentionSubmodules
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.mlp import MLP, MLPSubmodules
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_layer import TransformerLayer, TransformerLayerSubmodules
from sgl_kernel import silu_and_mul
from sgl_kernel.flash_attn import flash_attn_varlen_func


class NormWeight(torch.nn.Module):
    """Checkpoint gamma for a normalization executed by the prefill kernels."""

    def __init__(self, *, config, hidden_size, eps):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.empty(hidden_size, dtype=config.params_dtype))
        self.eps = eps


@dataclass
class _Prefill:
    positions: torch.Tensor
    bounds: torch.Tensor
    layers: frozenset
    capture: object
    residual: torch.Tensor | None = None
    rope_cache: torch.Tensor | None = None

    def normalize(self, branch, norm):
        if self.residual is None:
            self.residual = branch
            return rmsnorm(branch, norm.weight, norm.eps)
        fused_add_rmsnorm(branch, self.residual, norm.weight, norm.eps)
        return branch


# Core passes a single activation between HybridStack positions. The fused norm
# also needs the pending residual. Scope that second stream to one model call.
_prefill = ContextVar("qwen3_target_prefill")


def _normalize_heads(value, norm):
    # Warp reduction order for Qwen3 Q/K gamma, with FP32 lane accumulators.
    x = value.float()
    lanes = x.unflatten(-1, (32, -1))
    sums = lanes[..., 0].square()
    for component in lanes.unbind(-1)[1:]:
        sums = torch.addcmul(sums, component, component)
    for stride in (16, 8, 4, 2, 1):
        sums = sums[..., :stride] + sums[..., stride:]
    inv_rms = torch.rsqrt(sums / x.shape[-1] + norm.eps)
    return (x * inv_rms * norm.weight.float()).to(value.dtype)


class Qwen3AttentionLayer(TransformerLayer):
    def forward(self, hidden_states, rotary_pos_emb, **kwargs):
        state = _prefill.get()
        attention = self.self_attention
        width = self.config.kv_channels
        length = hidden_states.size(0)
        if state.rope_cache is None:
            angles = rotary_pos_emb[:length, 0, 0, :width // 2].contiguous()
            state.rope_cache = torch.cat((angles.cos(), angles.sin()), dim=-1)
        normalized = state.normalize(hidden_states.squeeze(1), self.input_layernorm)
        packed, _ = attention.linear_qkv(normalized.unsqueeze(1))
        queries = self.config.num_attention_heads // self.config.num_query_groups
        packed = packed.reshape(length, -1, queries + 2, width)
        q = _normalize_heads(packed[:, :, :queries].reshape(length, -1, width), attention.q_layernorm)
        k = _normalize_heads(packed[:, :, queries], attention.k_layernorm)
        v = packed[:, :, queries + 1].contiguous()
        apply_rope_with_cos_sin_cache_inplace(
            state.positions, q.flatten(1), k.flatten(1), width, state.rope_cache, True,
        )
        context = flash_attn_varlen_func(q, k, v, state.bounds, state.bounds, length, length, causal=True)
        branch, _ = attention.linear_proj(context.reshape(length, 1, -1))
        return branch, None


class Qwen3MLPLayer(TransformerLayer):
    def forward(self, hidden_states, **kwargs):
        state = _prefill.get()
        normalized = state.normalize(hidden_states.squeeze(1), self.pre_mlp_layernorm)
        gate_up, _ = self.mlp.linear_fc1(normalized.unsqueeze(1))
        branch, _ = self.mlp.linear_fc2(silu_and_mul(gate_up.squeeze(1)).unsqueeze(1))
        logical_layer = (self.layer_number - 1) // 2
        if state.capture is not None and logical_layer in state.layers:
            state.capture(logical_layer, branch.squeeze(1), state.residual)
        return branch, None


class Qwen3Target(HybridModel):
    """ASpec feature endpoint. Core owns embeddings, layer order and TP projections."""

    @torch.inference_mode()
    def prefill(self, ids, layers, capture=None):
        if ids.ndim != 2 or ids.size(0) != 1 or not 0 < ids.size(1) <= self.max_sequence_length:
            raise ValueError("Qwen3 Target expects one nonempty sequence within its context")
        if any(layer < 0 or layer >= self.config.num_layers // 2 for layer in layers):
            raise ValueError("Qwen3 Target capture is outside the logical decoder")
        positions = torch.arange(ids.size(1), device=ids.device)
        bounds = torch.tensor([0, ids.size(1)], device=ids.device, dtype=torch.int32)
        state = _Prefill(positions, bounds, frozenset(layers), capture)
        token = _prefill.set(state)
        try:
            branch = self(input_ids=ids, position_ids=positions.unsqueeze(0), attention_mask=None)
            return state.normalize(branch.squeeze(1), self.final_norm)
        finally:
            _prefill.reset(token)


def build_target(config, core_config, groups, implementation):
    if core_config.kv_channels % 32:
        raise ValueError("Qwen3 prefill requires heads divisible by a warp")
    backend = get_backend(implementation)
    attention = ModuleSpec(
        module=SelfAttention, params={"attn_mask_type": AttnMaskType.causal},
        submodules=SelfAttentionSubmodules(
            linear_qkv=backend.column_parallel_linear(), linear_proj=backend.row_parallel_linear(),
            core_attention=backend.core_attention(), q_layernorm=NormWeight, k_layernorm=NormWeight,
        ),
    )
    mlp = partial(MLP.as_mlp_submodule, submodules=MLPSubmodules(
        linear_fc1=backend.column_parallel_linear(), linear_fc2=backend.row_parallel_linear(),
    ))
    stack = ModuleSpec(module=HybridStack, submodules=HybridStackSubmodules(
        attention_layer=ModuleSpec(module=Qwen3AttentionLayer, submodules=TransformerLayerSubmodules(
            input_layernorm=NormWeight, self_attention=attention)),
        mlp_layer=ModuleSpec(module=Qwen3MLPLayer, submodules=TransformerLayerSubmodules(
            pre_mlp_layernorm=NormWeight, mlp=mlp)),
    ))
    rope = getattr(config, "rope_parameters", None) or {}
    with torch.device("cuda"):
        model = Qwen3Target(
            config=core_config, hybrid_stack_spec=stack,
            hybrid_layer_pattern="*-" * config.num_hidden_layers,
            vocab_size=config.vocab_size, max_sequence_length=config.max_position_embeddings,
            post_process=False, position_embedding_type="rope",
            rotary_base=rope.get("rope_theta", getattr(config, "rope_theta", 1000000.0)),
            pg_collection=groups,
        )
        model.final_norm = NormWeight(config=core_config, hidden_size=config.hidden_size,
                                      eps=core_config.layernorm_epsilon)
    return model
