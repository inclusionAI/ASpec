# Copyright 2026 Ant Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

# Adapted from FasterDecoding/Medusa e2a5d20c048a9b0a4092e6933c34313687422518
# (medusa/model/utils.py, medusa/model/medusa_choices.py, Apache-2.0).
# Tree buffers, candidate construction and greedy acceptance follow the
# reference; batched path verification and explicit evidence accounting are
# ASpec additions.
"""Medusa candidate-tree proposals, verification and acceptance accounting.

The consumer is deliberately not a serving runtime: callers provide a verifier
that owns the Target forward pass, KV/cache policy and sampling. The reference
verifier here batches independent teacher-forced paths, which is exact for
greedy verification but makes no throughput claim.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, Sequence

import torch
from torch import Tensor

DEFAULT_TOPK = 10

# Reference Medusa tree for Vicuna/7B-style heads (medusa/model/medusa_choices.py).
DEFAULT_MEDUSA_CHOICES = [
    [0], [0, 0], [1], [0, 1], [2], [0, 0, 0], [1, 0], [0, 2], [3], [0, 3],
    [4], [0, 4], [2, 0], [0, 5], [0, 0, 1], [5], [0, 6], [6], [0, 7],
    [0, 1, 0], [1, 1], [7], [0, 8], [0, 0, 2], [3, 0], [0, 9], [8], [9],
    [1, 0, 0], [0, 2, 0], [1, 2], [0, 0, 3], [4, 0], [2, 1], [0, 0, 4],
    [0, 0, 5], [0, 0, 0, 0], [0, 1, 1], [0, 0, 6], [0, 3, 0], [5, 0],
    [1, 3], [0, 0, 7], [0, 0, 8], [0, 0, 9], [6, 0], [0, 4, 0], [1, 4],
    [7, 0], [0, 1, 2], [2, 0, 0], [3, 1], [2, 2], [8, 0], [0, 5, 0],
    [1, 5], [1, 0, 1], [0, 2, 1], [9, 0], [0, 6, 0], [0, 0, 0, 1],
    [1, 6], [0, 7, 0],
]


@dataclass(frozen=True)
class TreeBuffers:
    choices: tuple[tuple[int, ...], ...]
    tree_indices: Tensor
    position_ids: Tensor
    retrieve_indices: Tensor
    attention_mask: Tensor


@dataclass(frozen=True)
class TreeProposal:
    bonus_token: Tensor
    candidates: Tensor
    lengths: Tensor
    medusa_logits: Tensor


@dataclass(frozen=True)
class Acceptance:
    path_index: int
    accept_length: int
    accepted_tokens: Tensor


def _validate_choices(medusa_choices: Sequence[Sequence[int]], top_k: int) -> list[tuple[int, ...]]:
    if top_k < 1:
        raise ValueError("top_k must be a positive integer")
    choices = [tuple(int(value) for value in path) for path in medusa_choices]
    if not choices:
        raise ValueError("medusa_choices must not be empty")
    if len(set(choices)) != len(choices):
        raise ValueError("medusa_choices must not contain duplicates")
    available = set(choices)
    for path in choices:
        if not path:
            raise ValueError("medusa_choices paths must be non-empty")
        if any(value < 0 or value >= top_k for value in path):
            raise ValueError(f"medusa_choices values must be in [0, {top_k})")
        for depth in range(1, len(path)):
            if path[:depth] not in available:
                raise ValueError(f"medusa_choices must be prefix-closed: {path}")
    return choices


def generate_tree_buffers(
    medusa_choices: Sequence[Sequence[int]] | None = None,
    *,
    top_k: int = DEFAULT_TOPK,
    device: Any = "cpu",
) -> TreeBuffers:
    """Port of the reference ``generate_medusa_buffers`` with an explicit top-k."""
    choices = _validate_choices(DEFAULT_MEDUSA_CHOICES if medusa_choices is None else medusa_choices, top_k)
    sorted_choices = sorted(choices, key=lambda path: (len(path), path))
    depth_counts: list[int] = []
    previous_depth = 0
    for path in sorted_choices:
        depth = len(path)
        if depth != previous_depth:
            depth_counts.append(0)
        depth_counts[depth - 1] += 1
        previous_depth = depth

    tree_len = len(sorted_choices) + 1
    attention_mask = torch.eye(tree_len, dtype=torch.bool)
    attention_mask[:, 0] = True
    start = 0
    for depth, count in enumerate(depth_counts, start=1):
        for offset in range(count):
            choice = sorted_choices[start + offset]
            if len(choice) > 1:
                ancestors = [
                    sorted_choices.index(choice[:prefix]) + 1
                    for prefix in range(1, len(choice))
                ]
                attention_mask[start + offset + 1, ancestors] = True
        start += count

    tree_indices = torch.zeros(tree_len, dtype=torch.long)
    start = 0
    for depth, count in enumerate(depth_counts):
        for offset in range(count):
            tree_indices[start + offset + 1] = (
                sorted_choices[start + offset][-1] + top_k * depth + 1)
        start += count

    position_ids = torch.zeros(tree_len, dtype=torch.long)
    start = 0
    for depth, count in enumerate(depth_counts):
        position_ids[start + 1:start + count + 1] = depth + 1
        start += count

    retrieve_nest: list[list[int]] = []
    retrieve_paths: set[tuple[int, ...]] = set()
    for index in range(len(sorted_choices)):
        choice = sorted_choices[-index - 1]
        if choice in retrieve_paths:
            continue
        retrieve_indices = []
        for depth in range(len(choice)):
            prefix = choice[:depth + 1]
            retrieve_indices.append(sorted_choices.index(prefix))
            retrieve_paths.add(prefix)
        retrieve_nest.append(retrieve_indices)
    max_length = max(len(path) for path in retrieve_nest)
    padded = [path + [-2] * (max_length - len(path)) for path in retrieve_nest]
    retrieve = torch.tensor(padded, dtype=torch.long) + 1
    retrieve = torch.cat([torch.zeros((retrieve.shape[0], 1), dtype=torch.long), retrieve], dim=1)

    return TreeBuffers(
        choices=tuple(sorted_choices),
        tree_indices=tree_indices.to(device),
        position_ids=position_ids.to(device),
        retrieve_indices=retrieve.to(device),
        attention_mask=attention_mask.unsqueeze(0).unsqueeze(0).to(device),
    )


def _draft_dtype(model: torch.nn.Module) -> torch.dtype:
    parameter = next(model.parameters(), None)
    return parameter.dtype if parameter is not None else torch.float32


class MedusaTreeProposer:
    """Build a Medusa candidate tree from one Target final hidden state."""

    def __init__(
        self,
        draft_model: torch.nn.Module,
        *,
        medusa_choices: Sequence[Sequence[int]] | None = None,
        top_k: int = DEFAULT_TOPK,
        device: Any = None,
    ):
        self.model = draft_model.eval()
        heads = getattr(draft_model, "medusa_num_heads", None)
        if not isinstance(heads, int) or heads < 1:
            raise ValueError("MedusaTreeProposer requires a heads-only Medusa draft")
        choices = DEFAULT_MEDUSA_CHOICES if medusa_choices is None else medusa_choices
        if device is None:
            tensor = next(draft_model.parameters(), None)
            if tensor is None:
                tensor = next(draft_model.buffers(), None)
            device = tensor.device if tensor is not None else "cpu"
        self.buffers = generate_tree_buffers(
            [path for path in choices if len(path) <= heads], top_k=top_k,
            device=device)
        if not self.buffers.choices:
            raise ValueError("medusa_choices has no path within the draft's head depth")
        self.top_k = top_k

    @torch.no_grad()
    def propose(self, last_hidden: Tensor, base_logits: Tensor) -> TreeProposal:
        hidden = last_hidden.reshape(1, 1, -1).to(
            device=self.buffers.tree_indices.device, dtype=_draft_dtype(self.model))
        logits = self.model(hidden)
        if logits.ndim != 4 or logits.shape[0] != self.model.medusa_num_heads:
            raise ValueError("Medusa draft must return [heads, batch, sequence, vocab] logits")
        head_logits = logits[:, 0, 0]
        topk = torch.topk(head_logits, self.top_k, dim=-1).indices.reshape(-1)
        flat_base = base_logits.reshape(-1)
        if flat_base.numel() != logits.shape[-1]:
            raise ValueError("Base logits vocabulary differs from the draft head vocabulary")
        bonus = flat_base.argmax()
        candidates = torch.cat([bonus.reshape(1), topk])
        tree_tokens = candidates[self.buffers.tree_indices]
        extended = torch.cat([tree_tokens, torch.zeros(1, dtype=torch.long,
                                                       device=tree_tokens.device)])
        cart = extended[self.buffers.retrieve_indices]
        lengths = (self.buffers.retrieve_indices[:, 1:] >= 0).sum(dim=1) + 1
        return TreeProposal(
            bonus_token=bonus,
            candidates=cart,
            lengths=lengths,
            medusa_logits=head_logits,
        )


class GreedyVerifier(Protocol):
    """Caller-owned Target forward. Returns post-norm hidden and next-token argmax."""

    def prefill(self, prefix_ids: Tensor) -> tuple[Tensor, Tensor]:
        ...

    def verify(self, prefix_ids: Tensor, candidates: Tensor, lengths: Tensor) -> Tensor:
        ...


def accept_greedy(proposal: TreeProposal, next_tokens: Tensor) -> Acceptance:
    """Reference greedy acceptance over verified path predictions."""
    candidates = proposal.candidates
    lengths = proposal.lengths
    if next_tokens.shape[0] != candidates.shape[0]:
        raise ValueError("Verifier returned a different number of paths")
    best_path = 0
    best_length = -1
    for path in range(candidates.shape[0]):
        real = int(lengths[path])
        if real < 1:
            continue
        accepted = 0
        for depth in range(real - 1):
            if int(next_tokens[path, depth]) != int(candidates[path, depth + 1]):
                break
            accepted += 1
        if accepted > best_length:
            best_length = accepted
            best_path = path
    accepted_tokens = candidates[best_path, :best_length + 1].clone()
    return Acceptance(path_index=best_path, accept_length=best_length,
                      accepted_tokens=accepted_tokens)


class MedusaConsumer:
    """Greedy speculative steps with explicit acceptance accounting."""

    def __init__(
        self,
        draft_model: torch.nn.Module,
        verifier: GreedyVerifier,
        *,
        medusa_choices: Sequence[Sequence[int]] | None = None,
        top_k: int = DEFAULT_TOPK,
    ):
        self.proposer = MedusaTreeProposer(
            draft_model, medusa_choices=medusa_choices, top_k=top_k)
        self.verifier = verifier

    @torch.no_grad()
    def step(self, prefix_ids: Tensor) -> Acceptance:
        hidden, base_logits = self.verifier.prefill(prefix_ids)
        proposal = self.proposer.propose(hidden, base_logits)
        next_tokens = self.verifier.verify(prefix_ids, proposal.candidates, proposal.lengths)
        return accept_greedy(proposal, next_tokens)

    @torch.no_grad()
    def generate(
        self,
        prefix_ids: Tensor,
        *,
        max_new_tokens: int,
        eos_token_id: int | None = None,
    ) -> dict[str, Any]:
        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be positive")
        prefix = prefix_ids.reshape(1, -1).long()
        generated = prefix
        steps = 0
        accepted_tokens = 0
        while accepted_tokens < max_new_tokens:
            acceptance = self.step(generated)
            new = acceptance.accepted_tokens
            remaining = max_new_tokens - accepted_tokens
            if new.numel() > remaining:
                new = new[:remaining]
            generated = torch.cat([generated, new.reshape(1, -1)], dim=1)
            accepted_tokens += new.numel()
            steps += 1
            if eos_token_id is not None and int(eos_token_id) in new.tolist():
                stop = generated[0].tolist().index(int(eos_token_id))
                generated = generated[:, :stop + 1]
                accepted_tokens = generated.shape[1] - prefix.shape[1]
                break
        return {
            "token_ids": generated[0],
            "generated_tokens": accepted_tokens,
            "verifier_calls": steps,
            "acceptance_length": accepted_tokens / steps if steps else None,
        }


class HFCausalVerifier:
    """Reference verifier: batched teacher-forced paths through a Hugging Face causal LM.

    Exact for greedy acceptance. It re-runs the prefix on every step and makes no
    throughput, KV-cache or serving claim.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        *,
        device: Any = "cpu",
        verify_batch_size: int = 16,
    ):
        if verify_batch_size < 1:
            raise ValueError("verify_batch_size must be positive")
        self.model = model.eval()
        self.device = torch.device(device)
        self.verify_batch_size = verify_batch_size
        self.base = getattr(model, "model", None)
        if self.base is None or not hasattr(self.base, "forward"):
            raise ValueError("HFCausalVerifier requires a causal LM exposing model.model")
        self.lm_head = getattr(model, "lm_head", None)
        if self.lm_head is None:
            raise ValueError("HFCausalVerifier requires a causal LM exposing lm_head")

    def _forward(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        outputs = self.base(
            input_ids=input_ids.to(self.device),
            attention_mask=attention_mask.to(self.device),
            use_cache=False,
        )
        return outputs.last_hidden_state

    @torch.no_grad()
    def prefill(self, prefix_ids: Tensor) -> tuple[Tensor, Tensor]:
        prefix = prefix_ids.reshape(1, -1).long()
        hidden = self._forward(prefix, torch.ones_like(prefix))[0, -1]
        logits = self.lm_head(hidden)
        return hidden, logits

    @torch.no_grad()
    def verify(self, prefix_ids: Tensor, candidates: Tensor, lengths: Tensor) -> Tensor:
        prefix = prefix_ids.reshape(-1).long()
        paths = candidates.to(self.device)
        lengths = lengths.to(self.device)
        width = int(lengths.max())
        outputs = torch.empty(
            (paths.shape[0], width - 1), dtype=torch.long, device=self.device)
        for start in range(0, paths.shape[0], self.verify_batch_size):
            stop = min(start + self.verify_batch_size, paths.shape[0])
            rows = paths[start:stop, :width]
            mask = (torch.arange(width, device=self.device)[None, :]
                    < lengths[start:stop, None])
            sequences = torch.cat(
                [prefix[None, :].expand(rows.shape[0], -1),
                 torch.where(mask, rows, torch.zeros_like(rows))], dim=1)
            attention = torch.cat(
                [torch.ones((rows.shape[0], prefix.numel()), dtype=torch.long,
                            device=self.device), mask.long()], dim=1)
            hidden = self._forward(sequences, attention)
            path_hidden = hidden[:, prefix.numel():prefix.numel() + width - 1]
            next_tokens = self.lm_head(path_hidden).argmax(dim=-1)
            outputs[start:stop] = next_tokens
        return outputs

    @torch.no_grad()
    def greedy_generate(
        self,
        prefix_ids: Tensor,
        *,
        max_new_tokens: int,
        eos_token_id: int | None = None,
    ) -> dict[str, Any]:
        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be positive")
        generated = prefix_ids.reshape(1, -1).long()
        steps = 0
        while steps < max_new_tokens:
            _, logits = self.prefill(generated)
            token = logits.argmax().reshape(1, 1)
            generated = torch.cat([generated, token], dim=1)
            steps += 1
            if eos_token_id is not None and int(token) == int(eos_token_id):
                break
        return {
            "token_ids": generated[0],
            "generated_tokens": steps,
            "verifier_calls": steps,
            "acceptance_length": 1.0,
        }
