# Copyright 2026 Ant Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""DSpine model/injection/loss contracts on CPU."""

import pytest
import torch

from aspec.models.draft.dspine import (
    AdjacentInjection,
    DSpineConfig,
    DSpineDraftModel,
    DSpineOptions,
)
from aspec.models.dspine import DSpineModel, refinement_loss
from aspec.models.ops.anchors import sample_anchor_positions, validate_injected_anchors
from aspec.models.ops.cudnn_flex_attention import build_dspark_mla_mask


def make_config(**overrides):
    raw = dict(
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        vocab_size=32,
        num_target_layers=1,
        target_hidden_size=16,
        target_num_hidden_layers=4,
        target_layer_ids=[1],
        mask_token_id=31,
        max_position_embeddings=128,
        block_size=4,
        dspine_config={
            "transfer_rank": 8,
            "message_size": 8,
            "top_k": 4,
            "alignment_warmup_steps": 0,
            "replacement_layers": 1,
            "replacement_probability": 0.0,
        },
    )
    raw.update(overrides)
    return DSpineConfig(**raw)


def make_model(config=None):
    config = config or make_config()
    draft = DSpineDraftModel(config)
    draft.freeze_embedding()
    return DSpineModel(
        draft, block_size=config.block_size, num_anchors=2,
        attention_backend="sdpa",
    )


def make_inputs(seed=0, length=12, batch=1):
    generator = torch.Generator().manual_seed(seed)
    return {
        "input_ids": torch.randint(0, 31, (batch, length), generator=generator),
        "hidden_states_list": torch.randn(batch, length, 16, generator=generator),
        "loss_mask": torch.ones(batch, length),
        "lm_head_weight": torch.randn(32, 16, generator=generator),
        "last_hidden_states": torch.randn(batch, length, 16, generator=generator),
    }


def test_config_fails_closed_on_non_causal_and_invalid_rank():
    with pytest.raises(ValueError, match="causal"):
        make_config(is_causal=False)
    with pytest.raises(ValueError, match="transfer_rank"):
        make_config(dspine_config={"transfer_rank": 32})
    with pytest.raises(ValueError, match="block_size"):
        make_config(block_size=1)


def test_forward_initializes_transfer_space_and_backpropagates():
    torch.manual_seed(3)
    model = make_model()
    inputs = make_inputs()
    loss, accuracy, loss_pp, acc_pp, count_pp, components, terms = model(
        **inputs, global_step=0, total_steps=1,
    )

    assert torch.isfinite(loss)
    assert loss_pp.shape == acc_pp.shape == count_pp.shape == (4,)
    assert loss_pp[0] == 0 and acc_pp[0] == 0
    assert torch.equal(count_pp[0], torch.tensor(0.0))
    assert bool(model.draft_model.transfer_ready)
    assert model.draft_model.transfer_codes.shape == (32, 8)
    assert "dspine/backbone_ce" in components
    assert terms[1] > 0

    loss.backward()
    gradients = [p.grad for p in model.parameters() if p.requires_grad]
    assert gradients and all(g is not None and torch.isfinite(g).all() for g in gradients)


def test_transfer_space_codes_are_normalized_and_readouts_use_pca():
    model = make_model()
    head = torch.randn(32, 16)
    model.draft_model.initialize_transfer_space(head)
    norms = torch.linalg.vector_norm(model.draft_model.transfer_codes.float(), dim=-1)
    torch.testing.assert_close(norms, torch.ones_like(norms), atol=1e-4, rtol=1e-4)
    for readout in model.draft_model.injection.readouts:
        assert readout.weight.shape == (8, 16)
        assert torch.isfinite(readout.weight).all()


def test_adjacent_injection_does_not_leak_future_tokens_backwards():
    options = DSpineOptions(
        transfer_rank=8, message_size=8, top_k=4,
        alignment_warmup_steps=0, replacement_layers=1,
        replacement_probability=0.0,
    )
    injection = AdjacentInjection(16, 1, options)
    _randomize_injection(injection)
    generator = torch.Generator().manual_seed(11)
    hidden = torch.randn(1, 1, 4, 16, generator=generator)
    token_ids = torch.randint(0, 32, (1, 1, 4), generator=generator)
    codes = torch.randn(32, 8, generator=generator)

    replace = torch.ones(1, 1, dtype=torch.bool)
    first = injection(hidden, 0, token_ids, codes, replace)
    changed = hidden.clone()
    changed[..., -1, :] += 10.0
    second = injection(changed, 0, token_ids, codes, replace)

    torch.testing.assert_close(first[..., :-1, :], second[..., :-1, :], atol=0, rtol=0)
    assert not torch.equal(first[..., -1, :], second[..., -1, :])

    predecessor_changed = token_ids.clone()
    predecessor_changed[..., 1] = (predecessor_changed[..., 1] + 1) % 32
    third = injection(hidden, 0, predecessor_changed, codes, replace)
    torch.testing.assert_close(first[..., :2, :], third[..., :2, :], atol=0, rtol=0)
    assert not torch.equal(first[..., 2, :], third[..., 2, :])


def _randomize_injection(injection, seed=31):
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for write in injection.writes:
            write.weight.copy_(torch.randn(write.weight.shape, generator=generator) * 0.1)
        injection.predecessor_gate.weight.copy_(
            torch.randn(injection.predecessor_gate.weight.shape, generator=generator) * 0.1
        )


def _reference_injection(injection, hidden, layer, token_ids, codes, replace):
    import math

    import torch.nn.functional as F

    features = injection.feature(hidden, layer)
    anchor = math.sqrt(injection.rank) * F.embedding(token_ids[..., :1], codes)
    predecessors = torch.cat((anchor, features[..., 1:-1, :]), dim=-2)
    if replace is not None and layer < injection.replacement_layers:
        correct = math.sqrt(injection.rank) * F.embedding(token_ids[..., :-1], codes)
        predecessors = torch.where(replace[..., None, None], correct, predecessors)
    gate = torch.sigmoid(
        injection.receiver_gate(features[..., 1:, :])
        + injection.predecessor_gate(predecessors)
        + injection.bias
    )
    delta = injection.writes[layer](gate * injection.message(predecessors))
    scale = (
        torch.linalg.vector_norm(hidden[..., 1:, :].float(), dim=-1, keepdim=True)
        / math.sqrt(hidden.shape[-1])
    ).to(hidden.dtype)
    return torch.cat((hidden[..., :1, :], hidden[..., 1:, :] + scale * delta), dim=-2)


def test_adjacent_injection_matches_independent_equation():
    options = DSpineOptions(
        transfer_rank=8, message_size=8, top_k=4,
        alignment_warmup_steps=0, replacement_layers=1,
        replacement_probability=0.0,
    )
    injection = AdjacentInjection(16, 2, options)
    _randomize_injection(injection)
    generator = torch.Generator().manual_seed(19)
    hidden = torch.randn(1, 2, 4, 16, generator=generator)
    token_ids = torch.randint(0, 32, (1, 2, 4), generator=generator)
    codes = torch.randn(32, 8, generator=generator)
    replace = torch.tensor([[True, False]])

    actual = injection(hidden, 0, token_ids, codes, replace)
    expected = _reference_injection(injection, hidden, 0, token_ids, codes, replace)
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)


def test_refine_matches_independent_equation():
    model = make_model()
    model.draft_model.initialize_transfer_space(torch.randn(32, 16))
    generator = torch.Generator().manual_seed(23)
    hidden = torch.randn(1, 2, 3, 16, generator=generator)
    predecessors = torch.randint(0, 32, (1, 2, 3), generator=generator)
    layer = len(model.draft_model.layers) - 1
    _randomize_injection(model.draft_model.injection)
    actual = model.draft_model.refine(hidden, predecessors)
    import math

    import torch.nn.functional as F

    receiver = model.draft_model.injection.feature(hidden, layer)
    codes = math.sqrt(model.draft_model.injection.rank) * F.embedding(
        predecessors, model.draft_model.transfer_codes
    )
    expected = model.draft_model.final_norm(
        model.draft_model.injection.write(hidden, receiver, codes, layer)
    )
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)


def test_refinement_loss_is_zero_for_equal_candidate_distributions():
    scores = torch.tensor([[[2.0, 1.0, 0.0], [2.0, 1.0, 0.0]]])
    target = torch.softmax(scores, dim=-1)
    candidates = torch.tensor([[[4, 5, 6], [4, 5, 6]]])
    labels = torch.tensor([[4, 4]])
    weights = torch.ones_like(labels, dtype=torch.float32)
    supervised = torch.ones_like(labels, dtype=torch.float32)
    (
        refinement, ref_ce_sum, ref_l1_sum, ce_mass, l1_mass,
        ref_ce_mean, ref_l1_mean,
    ) = refinement_loss(
        scores=scores, candidates=candidates, target_candidates=target,
        teacher_candidate_logits=scores,
        proposal_labels=labels, proposal_weights=weights,
        proposal_supervised=supervised,
        sequence_shape=(1, 1),
        ce_weight=0.1, mass_threshold=1e-4,
    )
    assert ref_l1_sum == 0
    assert ref_l1_mean == 0
    assert ce_mass == l1_mass == weights.sum()
    torch.testing.assert_close(refinement, 0.1 * ref_ce_mean * supervised.sum())


def test_refinement_conditional_target_uses_candidate_softmax_not_mass_floor():
    # Teacher mass over the two candidates is tiny, but the conditional
    # distribution of [0, 0] is exactly [0.5, 0.5].
    teacher_probs = torch.tensor([[[1e-14, 1e-14, 1.0]]])
    teacher_candidate_logits = torch.tensor([[[0.0, 0.0]]])
    scores = torch.zeros(1, 1, 2)
    candidates = torch.tensor([[[0, 1]]])
    labels = torch.tensor([[0]])
    weights = torch.ones_like(labels, dtype=torch.float32)
    supervised = torch.ones_like(labels, dtype=torch.float32)
    result = refinement_loss(
        scores=scores, candidates=candidates,
        target_candidates=teacher_probs.gather(-1, candidates),
        teacher_candidate_logits=teacher_candidate_logits,
        proposal_labels=labels, proposal_weights=weights,
        proposal_supervised=supervised,
        sequence_shape=(1, 1),
        ce_weight=0.0, mass_threshold=0.0,
    )
    assert result[6] == 0


def test_draft_forward_requires_a_block_or_dense_mask():
    model = make_model()
    model.draft_model.initialize_transfer_space(torch.randn(32, 16))
    with pytest.raises(ValueError, match="block_mask or draft_attention_mask"):
        model.draft_model(
            context_feature=torch.randn(1, 12, 16),
            draft_position_ids=torch.zeros(1, 8, dtype=torch.long),
            context_position_ids=torch.zeros(1, 12, dtype=torch.long),
            noise_embedding=torch.randn(1, 8, 16),
            block_token_ids=torch.randint(0, 32, (1, 2, 4)),
        )


def test_block_mask_excludes_left_padding_but_keeps_valid_context():
    model = make_model()
    anchors = torch.tensor([[4], [4]])
    keep = torch.ones(2, 1, dtype=torch.bool)
    attention = torch.tensor([
        [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    ])
    _, dense = model._block_mask(
        anchors, keep, attention.shape[1], 2, attention_mask=attention
    )
    assert dense.shape == (2, 4, 16)
    assert not dense[0, :, 0].any()
    assert not dense[0, :, 1].any()
    assert dense[0, :, 2].any()
    assert dense[1, :, 0].any()
    _, no_mask_dense = model._block_mask(
        anchors, keep, attention.shape[1], 2, attention_mask=None
    )
    assert no_mask_dense[..., 0].any()


def test_anchor_sampling_is_stable_under_left_and_right_padding():
    block_size = 4
    num_anchors = 2
    real_length = 7
    device = torch.device("cpu")
    base_mask = torch.ones(1, real_length)
    generator = torch.Generator(device=device).manual_seed(11)
    base, base_keep = sample_anchor_positions(
        real_length, base_mask, num_anchors, block_size, device,
        generator=generator,
    )

    left_mask = torch.zeros(1, real_length + 5)
    left_mask[:, 5:] = 1
    left_attention = left_mask.clone()
    generator = torch.Generator(device=device).manual_seed(11)
    left, left_keep = sample_anchor_positions(
        real_length + 5, left_mask, num_anchors, block_size, device,
        generator=generator, attention_mask=left_attention,
    )
    assert left_keep.sum() == base_keep.sum() == num_anchors
    assert (left[left_keep] - 5).tolist() == base[base_keep].tolist()

    right_mask = torch.zeros(1, real_length + 5)
    right_mask[:, :real_length] = 1
    right_attention = right_mask.clone()
    generator = torch.Generator(device=device).manual_seed(11)
    right, right_keep = sample_anchor_positions(
        real_length + 5, right_mask, num_anchors, block_size, device,
        generator=generator, attention_mask=right_attention,
    )
    assert right_keep.sum() == base_keep.sum() == num_anchors
    assert right[right_keep].tolist() == base[base_keep].tolist()


def test_loss_terms_use_sequence_count_as_the_additive_denominator():
    model = make_model()
    inputs = make_inputs(batch=2)
    *_, terms = model(**inputs, global_step=0, total_steps=1)
    assert torch.equal(terms[1], torch.tensor(2.0))


def test_short_row_does_not_gain_anchors_from_a_longer_batch_neighbor():
    block_size = 7
    num_anchors = 2
    loss_mask = torch.zeros(2, 12)
    loss_mask[0, :7] = 1
    loss_mask[1, :] = 1
    attention = loss_mask.clone()
    generator = torch.Generator(device="cpu").manual_seed(13)
    anchors, keep = sample_anchor_positions(
        12, loss_mask, num_anchors, block_size, torch.device("cpu"),
        generator=generator, attention_mask=attention,
    )
    assert not keep[0].any()
    assert keep[1].any()


def test_explicit_anchor_validation_intersects_attention_and_loss_masks():
    loss_mask = torch.ones(1, 5)
    attention = torch.tensor([[1, 1, 0, 0, 0]])
    with pytest.raises(ValueError, match="first targets must be supervised"):
        validate_injected_anchors(
            torch.tensor([[1]]), torch.tensor([[True]]),
            loss_mask=loss_mask, num_anchors=1, block_size=3,
            attention_mask=attention,
        )


def _cudnn_visible(mask_func, batch, query, key, query_length):
    flat_query = batch * query_length + query
    endpoints = mask_func[0, :, flat_query]
    visible = key < int(endpoints[0])
    for index in range(1, endpoints.numel(), 2):
        visible |= int(endpoints[index]) <= key < int(endpoints[index + 1])
    return visible


def test_cudnn_mask_starts_context_at_first_valid_token():
    mask = build_dspark_mla_mask(
        anchor_positions=torch.tensor([[6]]),
        block_keep_mask=torch.tensor([[True]]),
        context_length=8,
        block_size=4,
        attention_mask=torch.tensor([[0, 0, 0, 0, 0, 1, 1, 1]]),
    )
    assert _cudnn_visible(mask.mask_func, 0, 0, 5, 4)
    assert not _cudnn_visible(mask.mask_func, 0, 0, 0, 4)


def test_cudnn_mask_broadcasts_two_rows_and_anchors_with_padding():
    mask = build_dspark_mla_mask(
        anchor_positions=torch.tensor([[2, 5], [3, 6]]),
        block_keep_mask=torch.ones(2, 2, dtype=torch.bool),
        context_length=8,
        block_size=2,
        attention_mask=torch.tensor([
            [1, 1, 1, 1, 1, 1, 0, 0],
            [0, 0, 1, 1, 1, 1, 1, 1],
        ]),
    )
    query_length = 4
    assert _cudnn_visible(mask.mask_func, 0, 0, 0, query_length)
    assert not _cudnn_visible(mask.mask_func, 1, 0, 0, query_length)
    assert _cudnn_visible(mask.mask_func, 1, 0, 2, query_length)


def test_adjacent_injection_uses_current_replacement_layer_count():
    options = DSpineOptions(
        transfer_rank=8, message_size=8, top_k=4,
        alignment_warmup_steps=0, replacement_layers=1,
        replacement_probability=0.0,
    )
    injection = AdjacentInjection(16, 1, options)
    _randomize_injection(injection)
    generator = torch.Generator().manual_seed(29)
    hidden = torch.randn(1, 1, 4, 16, generator=generator)
    token_ids = torch.randint(0, 32, (1, 1, 4), generator=generator)
    changed_tokens = token_ids.clone()
    changed_tokens[..., 1] = (changed_tokens[..., 1] + 1) % 32
    codes = torch.randn(32, 8, generator=generator)
    replace = torch.ones(1, 1, dtype=torch.bool)
    active = injection(hidden, 0, token_ids, codes, replace, replacement_layers=1)
    disabled = injection(hidden, 0, token_ids, codes, replace, replacement_layers=0)
    changed_disabled = injection(
        hidden, 0, changed_tokens, codes, replace, replacement_layers=0
    )
    torch.testing.assert_close(disabled, changed_disabled, atol=0, rtol=0)
    assert not torch.equal(active, disabled)


def test_fixed_alignment_coefficient_is_additive_across_microbatches():
    model = make_model()
    inputs = make_inputs(batch=2)
    anchors = torch.tensor([[2, 6], [3, 7]])
    keep = torch.ones_like(anchors, dtype=torch.bool)
    coefficient = torch.tensor(0.7)
    keys = [name for name in inputs if name != "lm_head_weight"]

    def run(rows, batch_anchors, batch_keep):
        data = {
            name: (inputs[name] if rows is None else inputs[name][rows])
            for name in keys
        }
        data["lm_head_weight"] = inputs["lm_head_weight"]
        return model(
            **data, injected_anchors=batch_anchors,
            injected_keep_mask=batch_keep, global_step=0, total_steps=1,
            alignment_ce=coefficient,
        )[-1]

    combined = run(None, anchors, keep)
    first = run(slice(0, 1), anchors[:1], keep[:1])
    second = run(slice(1, 2), anchors[1:], keep[1:])
    torch.testing.assert_close(combined[0], first[0] + second[0], atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(combined[1], first[1] + second[1], atol=0, rtol=0)


def test_gradient_checkpointing_calls_the_parent_layer_hook(monkeypatch):
    model = make_model()
    model.train()
    model.draft_model.gradient_checkpointing_enable()
    calls = []

    def checkpointed(func, *args, **kwargs):
        calls.append(1)
        return func(*args, **kwargs)

    monkeypatch.setattr(model.draft_model, "_gradient_checkpointing_func", checkpointed)
    model(**make_inputs(), global_step=0, total_steps=1)
    assert len(calls) == model.draft_model.num_layers


def test_packed_inputs_are_rejected_before_backbone():
    model = make_model()
    inputs = make_inputs()
    with pytest.raises(NotImplementedError, match="padded, non-packed"):
        model(**inputs, ctx_doc_ids=torch.zeros_like(inputs["input_ids"]))
