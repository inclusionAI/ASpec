# Copyright 2026 Ant Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Absorbed-MLA feasibility checks.

The reference expands ``kv_b_proj`` into per-head K/V and runs an explicit
materialized attention. These checks cover the forward identity, affected
gradients, dense masks and the DSpark MLA geometry; they do not establish a
production speedup.
"""

import math

import pytest
import torch

from aspec.models.draft.dspark import (
    K3DSparkConfig,
    K3DSparkMLAAttention,
    _apply_rope_interleaved,
)
from aspec.models.ops.absorbed_mla import absorbed_mla_attention, split_kv_b_proj


def make_tensors(batch=1, query=4, key=7, heads=2, nope=4, rope=2, value=3, latent=5):
    generator = torch.Generator().manual_seed(23)
    return dict(
        q_nope=torch.randn(batch, query, heads, nope, generator=generator),
        q_rope=torch.randn(batch, query, heads, rope, generator=generator),
        compressed_kv=torch.randn(batch, key, latent, generator=generator),
        k_rope=torch.randn(batch, key, rope, generator=generator),
        kv_b_proj_weight=torch.randn(
            heads * (nope + value), latent, generator=generator
        ),
    )


def reference_materialized(
    q_nope, q_rope, compressed_kv, k_rope, kv_b_proj_weight,
    *, softmax_scale=None, attn_mask=None,
):
    """Explicitly materialize per-head K/V, then run dense attention."""
    batch, query, heads, nope = q_nope.shape
    rope = q_rope.size(-1)
    value = kv_b_proj_weight.shape[0] // heads - nope
    latent = compressed_kv.size(-1)
    view = kv_b_proj_weight.view(heads, nope + value, latent)
    w_uk, w_uv = view[:, :nope, :], view[:, nope:, :]
    k_nope = torch.einsum("btk,hnk->bthn", compressed_kv, w_uk)
    values = torch.einsum("btk,hvk->bthv", compressed_kv, w_uv)
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(nope + rope)
    scores = torch.einsum("bshn,bthn->bhst", q_nope, k_nope)
    scores = scores + torch.einsum("bshr,btr->bhst", q_rope, k_rope)
    scores = scores * softmax_scale
    if attn_mask is not None:
        mask = attn_mask.expand(scores.shape)
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
    probs = torch.softmax(scores.float(), dim=-1)
    if attn_mask is not None:
        probs = probs * mask.any(dim=-1, keepdim=True)
    probs = probs.to(q_nope.dtype)
    output = torch.einsum("bhst,bthv->bshv", probs, values)
    return output.reshape(batch, query, heads * value)


def make_attention_config(**overrides):
    values = dict(
        hidden_size=16, intermediate_size=32, num_hidden_layers=1,
        num_attention_heads=2, num_key_value_heads=2, vocab_size=32,
        q_lora_rank=8, kv_lora_rank=5, qk_nope_head_dim=4,
        qk_rope_head_dim=2, v_head_dim=3, num_target_layers=1,
        target_hidden_size=16, target_num_hidden_layers=4,
        target_layer_ids=[1], mask_token_id=31, max_position_embeddings=64,
        rms_norm_eps=1e-6, rope_theta=10000.0,
    )
    values.update(overrides)
    return K3DSparkConfig(**values)


def project_q_and_latent(attention, draft_hidden, context_hidden,
                         draft_position_ids, context_position_ids):
    """Reproduce the materialized MLA boundary before attention."""
    batch, draft_len, _ = draft_hidden.shape
    if attention.q_lora_rank is not None:
        query = attention.q_b_proj(
            attention.q_a_layernorm(attention.q_a_proj(draft_hidden))
        )
    else:
        query = attention.q_proj(draft_hidden)
    query = query.view(batch, draft_len, attention.num_heads, attention.qk_head_dim)
    # The retained path transposes to [B, H, S, D] before RoPE.
    query = query.transpose(1, 2)
    q_nope, q_rope = torch.split(
        query, [attention.qk_nope_head_dim, attention.qk_rope_head_dim], dim=-1
    )
    kv_input = torch.cat([context_hidden, draft_hidden], dim=1)
    kv_combined = attention.kv_a_proj_with_mqa(kv_input)
    compressed, k_rope = torch.split(
        kv_combined, [attention.kv_lora_rank, attention.qk_rope_head_dim], dim=-1
    )
    compressed = attention.kv_a_layernorm(compressed)
    k_rope = k_rope.unsqueeze(1)
    positions = torch.cat([context_position_ids, draft_position_ids], dim=1)
    cos, sin = attention.rotary_emb(q_rope, seq_len=positions.size(1))
    q_rope = _apply_rope_interleaved(
        q_rope, cos, sin, draft_position_ids, unsqueeze_dim=1
    )
    k_rope = _apply_rope_interleaved(
        k_rope, cos, sin, positions, unsqueeze_dim=1
    )
    # The materialized path keeps [B, H, T, D]; the absorbed reference takes
    # the BSHD-latent boundary, so K rope is the shared [B, T, D] tensor.
    return (
        q_nope.transpose(1, 2),
        q_rope.transpose(1, 2),
        compressed,
        k_rope.squeeze(1),
    )


def test_absorbed_forward_matches_materialized_equations():
    tensors = make_tensors()
    absorbed = absorbed_mla_attention(**tensors)
    expected = reference_materialized(**tensors)
    assert absorbed.shape == (1, 4, 6)
    torch.testing.assert_close(absorbed, expected, rtol=2e-6, atol=2e-6)


@pytest.mark.parametrize("materialized_first", [True, False])
def test_absorbed_gradients_match_materialized(materialized_first):
    tensors = make_tensors(batch=2)
    reference_tensors = {name: value.clone() for name, value in tensors.items()}
    for value in tensors.values():
        value.requires_grad_(True)
    for value in reference_tensors.values():
        value.requires_grad_(True)

    def run(fn, values):
        output = fn(**values)
        loss = output.float().pow(2).sum()
        return torch.autograd.grad(
            loss, [*values.values()], allow_unused=False
        )

    if materialized_first:
        expected = run(reference_materialized, reference_tensors)
        actual = run(absorbed_mla_attention, tensors)
    else:
        actual = run(absorbed_mla_attention, tensors)
        expected = run(reference_materialized, reference_tensors)
    for name, actual_grad, expected_grad in zip(
        tensors, actual, expected, strict=True
    ):
        torch.testing.assert_close(
            actual_grad, expected_grad, rtol=1e-5, atol=2e-5, msg=name
        )


def test_dense_mask_excludes_hidden_keys():
    tensors = make_tensors()
    mask = torch.ones(1, 2, 4, 7, dtype=torch.bool)
    mask[:, :, :, 5:] = False
    expected = reference_materialized(**tensors, attn_mask=mask)
    actual = absorbed_mla_attention(**tensors, attn_mask=mask)
    torch.testing.assert_close(actual, expected, rtol=2e-6, atol=2e-6)

    changed = {name: value.clone() for name, value in tensors.items()}
    changed["compressed_kv"][:, 5:] += 100.0
    changed["k_rope"][:, 5:] -= 100.0
    moved = absorbed_mla_attention(**changed, attn_mask=mask)
    torch.testing.assert_close(moved, actual, rtol=0, atol=0)


def test_fully_masked_rows_have_zero_output_and_gradient():
    tensors = make_tensors()
    for value in tensors.values():
        value.requires_grad_(True)
    mask = torch.ones(1, 2, 4, 7, dtype=torch.bool)
    mask[:, :, :2] = False
    output = absorbed_mla_attention(**tensors, attn_mask=mask)
    assert torch.count_nonzero(output[:, :2]) == 0
    grads = torch.autograd.grad(output.float().pow(2).sum(), list(tensors.values()))
    q_nope_grad = grads[0]
    assert torch.count_nonzero(q_nope_grad[:, :2]) == 0
    expected = reference_materialized(**tensors, attn_mask=mask)
    torch.testing.assert_close(output, expected, rtol=2e-6, atol=2e-6)


def test_mask_broadcasts_over_batch_and_heads():
    tensors = make_tensors(batch=2)
    mask = torch.tril(torch.ones(4, 7, dtype=torch.bool))
    actual = absorbed_mla_attention(**tensors, attn_mask=mask)
    expected = reference_materialized(**tensors, attn_mask=mask)
    torch.testing.assert_close(actual, expected, rtol=2e-6, atol=2e-6)


def test_short_and_single_token_sequences():
    tensors = make_tensors(query=1, key=1)
    torch.testing.assert_close(
        absorbed_mla_attention(**tensors), reference_materialized(**tensors),
        rtol=2e-6, atol=2e-6,
    )
    tensors = make_tensors(query=1, key=3)
    torch.testing.assert_close(
        absorbed_mla_attention(**tensors), reference_materialized(**tensors),
        rtol=2e-6, atol=2e-6,
    )


def test_split_rejects_wrong_weight_shape():
    with pytest.raises(ValueError, match="kv_b_proj.weight"):
        split_kv_b_proj(
            torch.zeros(13, 5), num_heads=2, qk_nope_head_dim=4,
            v_head_dim=3, kv_lora_rank=5,
        )


@pytest.mark.parametrize("mutation,match", [
    (lambda t: t.update(k_rope=t["k_rope"][:, :2]), "k_rope"),
    (lambda t: t.update(k_rope=t["k_rope"].double()), "dtype"),
    (lambda t: t.update(attn_mask=torch.ones(4, 7)), "boolean"),
    (lambda t: t.update(attn_mask=torch.ones(1, 3, 4, 7, dtype=torch.bool)), "broadcast"),
    (lambda t: t.update(softmax_scale=0.0), "softmax_scale"),
    (lambda t: t.update(softmax_scale=float("nan")), "softmax_scale"),
    (lambda t: t.update(softmax_scale=True), "softmax_scale"),
])
def test_invalid_inputs_fail_closed(mutation, match):
    tensors = dict(make_tensors(), softmax_scale=None, attn_mask=None)
    tensors.pop("softmax_scale")
    tensors.pop("attn_mask")
    mutation(tensors)
    with pytest.raises(ValueError, match=match):
        absorbed_mla_attention(**tensors)


@pytest.mark.parametrize("q_lora_rank", [8, None])
def test_matches_the_retained_materialized_dspark_attention(q_lora_rank):
    torch.manual_seed(11)
    attention = K3DSparkMLAAttention(
        make_attention_config(q_lora_rank=q_lora_rank)
    )
    batch, query, context = 1, 3, 6
    draft_hidden = torch.randn(batch, query, 16)
    context_hidden = torch.randn(batch, context, 16)
    draft_positions = torch.arange(context, context + query).unsqueeze(0)
    context_positions = torch.arange(context).unsqueeze(0)

    retained = attention(
        draft_hidden, context_hidden, draft_positions, context_positions,
        block_mask=None,
    )
    q_nope, q_rope, compressed, k_rope = project_q_and_latent(
        attention, draft_hidden, context_hidden, draft_positions, context_positions
    )
    reference = reference_materialized(
        q_nope, q_rope, compressed, k_rope, attention.kv_b_proj.weight,
        softmax_scale=attention.softmax_scale,
    )
    absorbed = absorbed_mla_attention(
        q_nope, q_rope, compressed, k_rope, attention.kv_b_proj.weight,
        softmax_scale=attention.softmax_scale,
    )
    torch.testing.assert_close(
        attention.o_proj(reference), retained, rtol=2e-6, atol=2e-6
    )
    torch.testing.assert_close(
        attention.o_proj(absorbed), retained, rtol=2e-6, atol=2e-6
    )
