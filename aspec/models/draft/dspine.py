# Copyright 2026 Ant Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

# Adapted from the public DSpine implementation (Tencent/DSpine, Apache-2.0)
# and the merged SpecForge DSpine training path (PR #965, corrected by PR #1038).
# The adjacent injection, transfer-code space and candidate refinement follow
# those references; ASpec supplies configuration validation and training wiring.
"""DSpine Draft: DFlash backbone with adjacent causal injection and refinement."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from functools import partial
from typing import NamedTuple

import torch
import torch.nn.functional as F
from torch import nn

from aspec.models.draft.dflash import DFlashConfig, DFlashDraftModel


@dataclass(frozen=True)
class DSpineOptions:
    transfer_rank: int = 1024
    message_size: int = 512
    top_k: int = 16
    ce_weight: float = 0.1
    alignment_weight: float = 0.5
    alignment_warmup_steps: int = 146
    replacement_layers: int = 3
    replacement_probability: float = 0.5
    replacement_hold_ratio: float = 1 / 6
    replacement_end_ratio: float = 1 / 3
    candidate_mass_threshold: float = 1e-4

    def validate(self, hidden_size: int, vocab_size: int) -> None:
        for name in (
            "transfer_rank", "message_size", "top_k", "alignment_warmup_steps",
            "replacement_layers",
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool):
                raise ValueError(f"DSpine {name} must be an integer")
        for name, value in asdict(self).items():
            if not math.isfinite(value):
                raise ValueError(f"DSpine {name} must be finite")
        if not 0 < self.transfer_rank <= min(hidden_size, vocab_size - 1):
            raise ValueError(
                "DSpine transfer_rank must be in [1, min(hidden_size, vocab_size - 1)]"
            )
        if self.message_size < 1 or not 1 <= self.top_k <= vocab_size:
            raise ValueError("DSpine requires message_size > 0 and top_k in [1, vocab_size]")
        if not 0 <= self.ce_weight <= 1 or self.alignment_weight < 0:
            raise ValueError(
                "DSpine requires ce_weight in [0, 1] and alignment_weight >= 0"
            )
        if self.alignment_warmup_steps < 0 or self.replacement_layers < 0:
            raise ValueError(
                "DSpine warmup steps and replacement layers must be nonnegative"
            )
        if not 0 <= self.replacement_probability <= 1:
            raise ValueError("DSpine replacement_probability must be in [0, 1]")
        if not 0 <= self.replacement_hold_ratio < self.replacement_end_ratio <= 1:
            raise ValueError(
                "DSpine requires 0 <= replacement_hold_ratio < replacement_end_ratio <= 1"
            )
        if not 0 <= self.candidate_mass_threshold < 1:
            raise ValueError("DSpine candidate_mass_threshold must be in [0, 1)")

    def schedule(self, step: int, total_steps: int | None) -> tuple[float, float]:
        warmup = self.alignment_warmup_steps
        beta = self.alignment_weight * (min(max(step, 0) / warmup, 1) if warmup else 1)
        if not total_steps or total_steps <= 0:
            return beta, 0.0
        progress = max(step, 0) / total_steps
        fraction = (self.replacement_end_ratio - progress) / (
            self.replacement_end_ratio - self.replacement_hold_ratio
        )
        return beta, self.replacement_probability * min(max(fraction, 0), 1)


class DSpineConfig(DFlashConfig):
    """DFlash backbone plus DSpine injection/refinement options."""

    model_type = "dspine"

    def __init__(
        self,
        *,
        block_size: int = 16,
        dspine_config: dict | None = None,
        **kwargs,
    ):
        architectures = kwargs.pop("architectures", ["DSpineDraftModel"])
        if architectures != ["DSpineDraftModel"]:
            raise ValueError("DSpine uses exactly one DSpineDraftModel architecture")
        if kwargs.get("is_causal") is False:
            raise ValueError("DSpine requires causal draft attention")
        if isinstance(block_size, bool) or not isinstance(block_size, int) or block_size < 2:
            raise ValueError("DSpine block_size must be an integer >= 2")
        kwargs["is_causal"] = True
        options = DSpineOptions(**(dspine_config or {}))
        super().__init__(architectures=architectures, block_size=block_size, **kwargs)
        options.validate(self.hidden_size, self.vocab_size)
        if self.ffn_type != "dense":
            raise ValueError("DSpine currently supports dense Draft FFNs only")
        if any(kind != "full_attention" for kind in (self.layer_types or [])):
            raise ValueError("DSpine currently supports full intra-block attention only")
        if getattr(self, "mtp_conditioning", None) is not None:
            raise ValueError("DSpine does not use the native-MTP conditioning contract")
        for stage in ("csa_training_stage", "qsa_training_stage", "dsa_training_stage"):
            if getattr(self, stage, None) is not None:
                raise ValueError("DSpine is not integrated with sparse-attention training")
        self.dspine_config = asdict(options)


class DSpineOutput(NamedTuple):
    hidden_states: torch.Tensor
    pre_injection: torch.Tensor
    layer_features: torch.Tensor


class AdjacentInjection(nn.Module):
    """Adjacent-token transfer codes and gated writes between draft layers."""

    def __init__(self, hidden_size: int, num_layers: int, options: DSpineOptions):
        super().__init__()
        self.rank = options.transfer_rank
        self.replacement_layers = options.replacement_layers
        self.readouts = nn.ModuleList(
            nn.Linear(hidden_size, self.rank, bias=False) for _ in range(num_layers)
        )
        self.writes = nn.ModuleList(
            nn.Linear(options.message_size, hidden_size, bias=False)
            for _ in range(num_layers)
        )
        self.message = nn.Linear(self.rank, options.message_size, bias=False)
        self.receiver_gate = nn.Linear(self.rank, options.message_size, bias=False)
        self.predecessor_gate = nn.Linear(self.rank, options.message_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(options.message_size))
        for write in self.writes:
            nn.init.zeros_(write.weight)
        nn.init.zeros_(self.predecessor_gate.weight)

    def feature(self, hidden: torch.Tensor, layer: int) -> torch.Tensor:
        projected = self.readouts[layer](hidden)
        return (math.sqrt(self.rank) * F.normalize(projected.float(), dim=-1)).to(
            hidden.dtype
        )

    def write(
        self,
        hidden: torch.Tensor,
        receiver: torch.Tensor,
        predecessor: torch.Tensor,
        layer: int,
    ) -> torch.Tensor:
        predecessor = predecessor.to(receiver.dtype)
        gate = torch.sigmoid(
            self.receiver_gate(receiver)
            + self.predecessor_gate(predecessor)
            + self.bias
        )
        delta = self.writes[layer](gate * self.message(predecessor))
        scale = (
            torch.linalg.vector_norm(hidden.float(), dim=-1, keepdim=True)
            / math.sqrt(hidden.shape[-1])
        ).to(hidden.dtype)
        return hidden + scale * delta

    def forward(
        self,
        hidden: torch.Tensor,
        layer: int,
        token_ids: torch.Tensor,
        codes: torch.Tensor,
        replace: torch.Tensor | None = None,
        replacement_layers: int | None = None,
    ) -> torch.Tensor:
        features = self.feature(hidden, layer)
        anchor = math.sqrt(self.rank) * F.embedding(token_ids[..., :1], codes)
        predecessors = torch.cat((anchor, features[..., 1:-1, :]), dim=-2)
        effective_layers = (
            self.replacement_layers if replacement_layers is None else replacement_layers
        )
        if replace is not None and layer < effective_layers:
            correct = math.sqrt(self.rank) * F.embedding(token_ids[..., :-1], codes)
            predecessors = torch.where(replace[..., None, None], correct, predecessors)
        successors = self.write(
            hidden[..., 1:, :], features[..., 1:, :], predecessors, layer
        )
        return torch.cat((hidden[..., :1, :], successors), dim=-2)


class DSpineDraftModel(DFlashDraftModel):
    """DFlash Draft with DSpine adjacent injection and refinement."""

    config_class = DSpineConfig

    def __init__(self, config):
        super().__init__(config)
        self.dspine_config = DSpineOptions(**config.dspine_config)
        self.dspine_config.validate(config.hidden_size, config.vocab_size)
        if config.block_size < 2:
            raise ValueError(
                "DSpine block_size must include an anchor and at least one proposal"
            )
        self.injection = AdjacentInjection(
            config.hidden_size, config.num_hidden_layers, self.dspine_config
        )
        self.register_buffer(
            "transfer_codes",
            torch.zeros(config.vocab_size, self.dspine_config.transfer_rank),
        )
        self.register_buffer("transfer_ready", torch.tensor(False))

    @torch.no_grad()
    def initialize_transfer_space(
        self, head_weight: torch.Tensor, chunk_size: int = 4096
    ) -> None:
        """Initialize whitened token codes and layer readouts from the Target head."""
        if tuple(head_weight.shape) != (
            self.config.vocab_size,
            self.config.hidden_size,
        ):
            raise ValueError(
                "DSpine transfer space requires the Target head's full vocabulary and hidden width"
            )
        if chunk_size < 1:
            raise ValueError("chunk_size must be positive")
        width = head_weight.shape[1]
        covariance = torch.zeros(
            width, width, device=head_weight.device, dtype=torch.float32
        )
        total = covariance.new_zeros(width)
        for rows in head_weight.split(chunk_size):
            normalized = F.normalize(rows.float(), dim=-1)
            total.add_(normalized.sum(dim=0))
            covariance.add_(normalized.T @ normalized)
        mean = total / head_weight.shape[0]
        covariance.sub_(head_weight.shape[0] * torch.outer(mean, mean))
        values, directions = torch.linalg.eigh(covariance)
        values = values[-self.dspine_config.transfer_rank :]
        directions = directions[:, -self.dspine_config.transfer_rank :]
        if not torch.isfinite(values).all() or (values <= 0).any():
            raise ValueError(
                "Target head has insufficient positive principal directions for transfer_rank"
            )
        for start in range(0, head_weight.shape[0], chunk_size):
            rows = head_weight[start : start + chunk_size]
            codes = F.normalize(
                (F.normalize(rows.float(), dim=-1) - mean) @ directions
                / values.sqrt(),
                dim=-1,
            )
            self.transfer_codes[start : start + len(rows)].copy_(codes.to(self.transfer_codes.dtype))
        for readout in self.injection.readouts:
            readout.weight.copy_(directions.T.to(readout.weight.dtype))
        self.transfer_ready.fill_(True)

    def _require_transfer_ready(self) -> None:
        if not bool(self.transfer_ready):
            raise ValueError(
                "Initialize or load the DSpine transfer space before use"
            )

    def forward(
        self,
        context_feature: torch.Tensor,
        draft_position_ids: torch.Tensor,
        context_position_ids: torch.Tensor,
        block_mask=None,
        noise_embedding: torch.Tensor | None = None,
        block_token_ids: torch.Tensor | None = None,
        replace_mask: torch.Tensor | None = None,
        draft_attention_mask=None,
        **kwargs,
    ) -> DSpineOutput:
        self._require_transfer_ready()
        if block_mask is None and draft_attention_mask is None:
            raise ValueError(
                "DSpineDraftModel.forward requires block_mask or draft_attention_mask "
                "to keep each block causal"
            )
        if kwargs.pop("use_cache", False) or kwargs.pop("past_key_values", None) is not None:
            raise ValueError("DSpine reference forward requires explicit context without a draft KV cache")
        if block_token_ids is None:
            raise ValueError("DSpine requires block_token_ids for adjacent injection")
        if noise_embedding is None:
            raise ValueError("DSpine requires precomputed noise embeddings")
        batch, blocks, size = block_token_ids.shape
        if size != self.config.block_size or noise_embedding.shape[:2] != (batch, blocks * size):
            raise ValueError("DSpine token blocks and noise embeddings must match block_size")
        hidden = noise_embedding.to(context_feature.dtype)
        layer_features = []
        pre_injection = None
        for index, layer in enumerate(self.layers):
            layer_forward = (
                partial(self._gradient_checkpointing_func, layer)
                if self.gradient_checkpointing and self.training else layer
            )
            hidden = layer_forward(
                draft_hidden=hidden,
                context_hidden=context_feature,
                draft_position_ids=draft_position_ids,
                context_position_ids=context_position_ids,
                block_mask=block_mask,
                draft_attention_mask=draft_attention_mask,
            )
            pre_injection = hidden.reshape(batch, blocks, size, -1)
            hidden = self.injection(
                pre_injection, index, block_token_ids, self.transfer_codes,
                replace_mask,
                replacement_layers=self.dspine_config.replacement_layers,
            )
            layer_features.append(self.injection.feature(hidden, index))
            hidden = hidden.reshape(batch, blocks * size, -1)
        return DSpineOutput(
            self.final_norm(hidden).reshape(batch, blocks, size, -1),
            pre_injection,
            torch.stack(layer_features),
        )

    def refine(
        self, hidden: torch.Tensor, predecessor_ids: torch.Tensor
    ) -> torch.Tensor:
        self._require_transfer_ready()
        layer = len(self.layers) - 1
        receiver = self.injection.feature(hidden, layer)
        predecessor = math.sqrt(self.injection.rank) * F.embedding(
            predecessor_ids, self.transfer_codes
        )
        return self.final_norm(
            self.injection.write(hidden, receiver, predecessor, layer)
        )
