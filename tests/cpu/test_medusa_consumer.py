# Copyright 2026 Ant Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Medusa candidate-tree consumer: buffers, proposals, verification and acceptance."""

import pytest
import torch
from torch import nn
from transformers import LlamaConfig, LlamaForCausalLM

from aspec.inference.medusa_consumer import (
    HFCausalVerifier,
    MedusaConsumer,
    TreeProposal,
    accept_greedy,
    generate_tree_buffers,
)


def _tiny_model():
    torch.manual_seed(5)
    config = LlamaConfig(
        vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
        num_attention_heads=2, num_key_value_heads=1, max_position_embeddings=128,
        pad_token_id=0,
    )
    return LlamaForCausalLM(config).eval()


class _FixedDraft(nn.Module):
    def __init__(self, heads, vocab):
        super().__init__()
        torch.manual_seed(3)
        self.register_buffer("logits", torch.randn(heads, vocab))
        self.medusa_num_heads = heads

    def forward(self, hidden):
        return self.logits[:, None, None, :].expand(
            -1, hidden.shape[0], hidden.shape[1], -1)


def test_tree_buffers_follow_the_reference_layout():
    buffers = generate_tree_buffers([[0], [1], [0, 0]], top_k=2)
    assert buffers.choices == ((0,), (1,), (0, 0))
    assert buffers.tree_indices.tolist() == [0, 1, 2, 3]
    assert buffers.position_ids.tolist() == [0, 1, 1, 2]
    # The reference emits one retrieval row per maximal path: (0,0) and (1,).
    assert buffers.retrieve_indices.shape == (2, 3)
    assert buffers.retrieve_indices[:, 0].tolist() == [0, 0]
    # Tree position 3 is (0,0); it attends to its ancestor (0,) at position 1.
    assert buffers.attention_mask[0, 0, 3, 1].item() is True
    assert buffers.attention_mask[0, 0, 3, 2].item() is False
    with pytest.raises(ValueError, match="prefix-closed"):
        generate_tree_buffers([[0, 0]], top_k=2)
    with pytest.raises(ValueError, match="in \\[0, 2\\)"):
        generate_tree_buffers([[2]], top_k=2)


def test_greedy_acceptance_chooses_the_longest_path():
    proposal = TreeProposal(
        bonus_token=torch.tensor(10),
        candidates=torch.tensor([[10, 1, 2], [10, 3, 4], [11, 5, 0]]),
        lengths=torch.tensor([3, 3, 2]),
        medusa_logits=torch.zeros(2, 12),
    )
    acceptance = accept_greedy(proposal, torch.tensor([[1, 2], [1, 9], [5, 8]]))
    assert acceptance.path_index == 0
    assert acceptance.accept_length == 2
    assert acceptance.accepted_tokens.tolist() == [10, 1, 2]


def test_hf_verifier_matches_sequential_teacher_forcing():
    model = _tiny_model()
    verifier = HFCausalVerifier(model, verify_batch_size=1)
    prefix = torch.tensor([[1, 2, 3, 4, 5]])
    candidates = torch.tensor([[6, 7, 8], [9, 10, 0]])
    lengths = torch.tensor([3, 2])
    actual = verifier.verify(prefix, candidates, lengths)
    for row, length in enumerate(lengths.tolist()):
        sequence = torch.cat([prefix[0], candidates[row, :length]])
        with torch.no_grad():
            hidden = model.model(input_ids=sequence[None]).last_hidden_state
            logits = model.lm_head(hidden[0])
        expected = [int(logits[prefix.shape[1] + index].argmax())
                    for index in range(length - 1)]
        assert actual[row, :len(expected)].tolist() == expected


def test_greedy_speculation_matches_target_and_is_stable():
    model = _tiny_model()
    verifier = HFCausalVerifier(model)
    draft = _FixedDraft(heads=2, vocab=32)
    consumer = MedusaConsumer(
        draft, verifier, medusa_choices=[[0], [1], [0, 0], [0, 1]], top_k=2)
    prefix = torch.tensor([[1, 2, 3, 4]])
    speculative = consumer.generate(prefix, max_new_tokens=6)
    baseline = verifier.greedy_generate(prefix, max_new_tokens=6)
    assert speculative["token_ids"].tolist() == baseline["token_ids"].tolist()
    assert speculative["verifier_calls"] >= 1
    assert speculative["acceptance_length"] >= 1.0
    repeat = consumer.generate(prefix, max_new_tokens=6)
    assert repeat["token_ids"].tolist() == speculative["token_ids"].tolist()

def test_default_choices_filter_to_the_draft_depth():
    from aspec.commands.evaluate_medusa_acceptance import _choices

    choices = _choices(None, 2)
    assert choices and max(len(path) for path in choices) <= 2
    with pytest.raises(ValueError, match="head depth"):
        _choices(None, 0)


def test_default_tree_matches_the_reference_buffers():
    buffers = generate_tree_buffers(top_k=10)
    assert buffers.tree_indices.shape == (64,)
    assert buffers.tree_indices[:11].tolist() == list(range(11))
    assert buffers.position_ids[:11].tolist() == [0] + [1] * 10
    assert buffers.retrieve_indices.shape == (42, 5)
    assert int(buffers.attention_mask.sum()) == 207


def test_eos_stops_both_paths_at_the_same_token():
    model = _tiny_model()
    verifier = HFCausalVerifier(model)
    draft = _FixedDraft(heads=2, vocab=32)
    consumer = MedusaConsumer(
        draft, verifier, medusa_choices=[[0], [1], [0, 0], [0, 1]], top_k=2)
    prefix = torch.tensor([[1, 2, 3, 4]])
    probe = verifier.greedy_generate(prefix, max_new_tokens=6)
    eos = int(probe["token_ids"][prefix.shape[1]])
    baseline = verifier.greedy_generate(
        prefix, max_new_tokens=6, eos_token_id=eos)
    speculative = consumer.generate(prefix, max_new_tokens=6, eos_token_id=eos)
    assert speculative["token_ids"].tolist() == baseline["token_ids"].tolist()
    assert speculative["token_ids"][-1].item() == eos
