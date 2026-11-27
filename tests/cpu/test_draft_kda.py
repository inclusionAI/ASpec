# Copyright 2026 Ant Group Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Focused checks for KDA-based DSpark Draft architectures (Issue #20).

Covers:
  - Config/policy resolution and fail-closed combinations (pure + 3:1 hybrid)
  - Numerical and gradient parity between chunkwise WY and recurrent KDA
  - Optional CUDA FLA vs reference parity (when CUDA + fla are installed)
  - Strict document-boundary isolation for both KDA state and short conv1d
  - Exact equivalence between training forward and stateful incremental proposal
  - Hybrid 3:1 DSparkDraftModel + DSparkModel forward/backward + observations
  - Exact resume topology checks, native --format aspec export, strict reload,
    mixed FP32/BF16 gate precision, and BlockDraftConsumer proposal parity
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed.checkpoint as dcp

from aspec.commands.convert_to_hf import convert_checkpoint
from aspec.config.draft_attention import resolve_draft_attention
from aspec.draft_artifact import BlockDraftConsumer, load_native_draft, verify_native_export
from aspec.models.draft.dspark import DSparkConfig, DSparkDraftModel
from aspec.models.draft.kda import (
    KDAForwardMetadata,
    KDALayer,
    chunk_kda_reference,
    kda_gate_reference,
    recurrent_kda_reference,
)
from aspec.models.dspark import DSparkModel
from aspec.training.checkpoint import _logical_training_topology, _validate_exact_resume_topology

REPO_ROOT = Path(__file__).resolve().parents[2]


def _tiny_kda_config(
    *,
    layer_types: list[str] | None = None,
    kda_backend: str = "reference",
    use_short_conv: bool = True,
    conv_size: int = 4,
    allow_neg_eigval: bool = False,
    kda_chunk_size: int = 8,
    block_size: int | None = None,
) -> DSparkConfig:
    if layer_types is None:
        layer_types = [
            "kimi_delta_attention",
            "kimi_delta_attention",
            "kimi_delta_attention",
            "full_attention",
        ]
    kwargs = dict(
        architectures=["DSparkDraftModel"],
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=len(layer_types),
        layer_types=layer_types,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        kda_num_heads=4,
        kda_head_dim=8,
        kda_expand_v=1.0,
        kda_use_short_conv=use_short_conv,
        kda_conv_size=conv_size,
        kda_conv_bias=False,
        kda_allow_neg_eigval=allow_neg_eigval,
        kda_backend=kda_backend,
        kda_chunk_size=kda_chunk_size,
        vocab_size=64,
        max_position_embeddings=256,
        num_target_layers=2,
        target_hidden_size=32,
        target_num_hidden_layers=8,
        target_layer_ids=[1, 5],
        mask_token_id=63,
        markov_rank=8,
        enable_confidence_head=True,
        confidence_head_with_markov=True,
    )
    if block_size is not None:
        kwargs["block_size"] = block_size
    return DSparkConfig(**kwargs)


def test_shipped_ling3_tiny_kda_config_resolves_3_to_1_hybrid():
    raw = json.loads(
        (REPO_ROOT / "aspec/config/dspark_ling3_tiny_kda_draft_config.json").read_text()
    )
    policy = resolve_draft_attention(raw)
    assert policy["layer_types"] == [
        "kimi_delta_attention",
        "kimi_delta_attention",
        "kimi_delta_attention",
        "full_attention",
    ]
    assert policy["draft_block_attention"] == [
        "causal_kda_state_branch",
        "causal_kda_state_branch",
        "causal_kda_state_branch",
        "bidirectional",
    ]
    assert policy["kda"]["backend"] == "fla"
    assert policy["kda"]["num_heads"] == 12
    assert policy["kda"]["head_dim"] == 128
    assert policy["kda"]["value_head_dim"] == 128
    assert policy["kda"]["use_short_conv"] is True
    assert policy["kda"]["conv_size"] == 4


def test_kda_config_validation_fails_closed():
    # Mixing KDA with DSA must fail closed
    with pytest.raises(ValueError, match="cannot be mixed"):
        resolve_draft_attention({
            "num_hidden_layers": 2,
            "layer_types": ["kimi_delta_attention", "deepseek_sparse_attention"],
        })
    # KDA fields without a KDA layer must fail closed
    with pytest.raises(ValueError, match="kimi_delta_attention"):
        resolve_draft_attention({
            "num_hidden_layers": 2,
            "layer_types": ["full_attention", "full_attention"],
            "kda_backend": "fla",
        })
    # Invalid backend must fail closed
    with pytest.raises(ValueError, match="kda_backend"):
        resolve_draft_attention({
            "num_hidden_layers": 1,
            "layer_types": ["kimi_delta_attention"],
            "kda_backend": "unknown_backend",
        })


def test_chunk_vs_recurrent_kda_numerical_and_gradient_parity():
    torch.manual_seed(42)
    bsz, seq_len, heads, dk, dv = 2, 19, 3, 8, 8
    q = torch.randn(bsz, seq_len, heads, dk, requires_grad=True)
    k = torch.nn.functional.normalize(torch.randn(bsz, seq_len, heads, dk), dim=-1).detach().requires_grad_(True)
    v = torch.randn(bsz, seq_len, heads, dv, requires_grad=True)
    f_raw = torch.randn(bsz, seq_len, heads, dk, requires_grad=True)
    a_log = torch.randn(heads, requires_grad=True)
    dt_bias = torch.randn(heads * dk, requires_grad=True)
    beta = torch.sigmoid(torch.randn(bsz, seq_len, heads)).detach().requires_grad_(True)
    init_state = torch.randn(bsz, heads, dk, dv, requires_grad=True)

    g_rec = kda_gate_reference(f_raw, a_log, dt_bias)
    out_rec, state_rec = recurrent_kda_reference(
        q, k, v, g_rec, beta, initial_state=init_state, output_final_state=True
    )
    loss_rec = (out_rec.pow(2).sum() + state_rec.pow(2).sum())
    grads_rec = torch.autograd.grad(
        loss_rec, (q, k, v, f_raw, a_log, dt_bias, beta, init_state)
    )

    g_chk = kda_gate_reference(f_raw, a_log, dt_bias)
    out_chk, state_chk = chunk_kda_reference(
        q, k, v, g_chk, beta, initial_state=init_state, output_final_state=True, chunk_size=6
    )
    loss_chk = (out_chk.pow(2).sum() + state_chk.pow(2).sum())
    grads_chk = torch.autograd.grad(
        loss_chk, (q, k, v, f_raw, a_log, dt_bias, beta, init_state)
    )

    torch.testing.assert_close(out_chk, out_rec, atol=1e-5, rtol=1e-4)
    torch.testing.assert_close(state_chk, state_rec, atol=1e-5, rtol=1e-4)
    for g_chk, g_rec in zip(grads_chk, grads_rec, strict=True):
        torch.testing.assert_close(g_chk, g_rec, atol=1e-4, rtol=1e-3)


def test_kda_document_boundary_isolation_forward_and_backward():
    """Mutating document 0 must have zero effect on document 1 outputs and gradients."""
    torch.manual_seed(7)
    cfg = _tiny_kda_config(layer_types=["kimi_delta_attention"], use_short_conv=True, conv_size=4)
    layer = KDALayer(cfg, layer_idx=0)

    # Sequence of length 12 split into doc 0 [0..5] and doc 1 [6..11]
    ctx_a = torch.randn(1, 12, 32, requires_grad=True)
    ctx_b = ctx_a.detach().clone()
    ctx_b[:, :6, :] = torch.randn(1, 6, 32) * 5.0  # Perturb doc 0 heavily
    ctx_b.requires_grad_(True)

    draft_a = torch.randn(1, 8, 32, requires_grad=True)  # 2 anchors * block_size=4
    draft_b = draft_a.detach().clone().requires_grad_(True)

    doc_ids = torch.tensor([[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]], dtype=torch.long)
    # Anchor 0 at pos 6 (start of doc 1 -> must have ZERO prefix & conv state)
    # Anchor 1 at pos 9 (inside doc 1 -> sees only doc 1 [6..8])
    meta = KDAForwardMetadata(
        anchor_positions=torch.tensor([[6, 9]], dtype=torch.long),
        block_keep_mask=torch.tensor([[True, True]]),
        block_size=4,
        context_doc_ids=doc_ids,
    )

    out_a, _ = layer(draft_a, ctx_a, meta)
    out_b, _ = layer(draft_b, ctx_b, meta)
    torch.testing.assert_close(out_a, out_b, atol=0.0, rtol=0.0)

    out_a.sum().backward()
    # Gradient w.r.t. doc 0 context tokens must be identically zero
    assert torch.equal(ctx_a.grad[:, :6, :], torch.zeros_like(ctx_a.grad[:, :6, :]))
    assert ctx_a.grad[:, 6:9, :].abs().sum() > 0


def test_kda_training_forward_matches_stateful_incremental_proposal():
    """Training forward at anchor a must match incremental update_prefix_state + propose_block."""
    torch.manual_seed(11)
    cfg = _tiny_kda_config(layer_types=["kimi_delta_attention"], use_short_conv=True, conv_size=4)
    layer = KDALayer(cfg, layer_idx=0).eval()

    ctx = torch.randn(1, 15, 32)
    draft = torch.randn(1, 5, 32)  # block_size=5 at anchor=10
    meta = KDAForwardMetadata(
        anchor_positions=torch.tensor([[10]], dtype=torch.long),
        block_keep_mask=torch.tensor([[True]]),
        block_size=5,
    )

    with torch.no_grad():
        train_out, _ = layer(draft, ctx, meta)

        # Incremental stateful prefix: prefill [0..6), then verify-append [6..10)
        state_1 = layer.update_prefix_state(ctx[:, :6, :], prefix_state=None)
        state_2 = layer.update_prefix_state(ctx[:, 6:10, :], prefix_state=state_1)
        rec_before = state_2.recurrent_state.clone()
        kconv_before = state_2.k_conv_state.clone()
        vconv_before = state_2.v_conv_state.clone()

        prop_out = layer.propose_block(draft, state_2)

        # Persistent prefix state must NOT be mutated by propose_block
        assert torch.equal(state_2.recurrent_state, rec_before)
        assert torch.equal(state_2.k_conv_state, kconv_before)
        assert torch.equal(state_2.v_conv_state, vconv_before)
        torch.testing.assert_close(prop_out, train_out, atol=1e-5, rtol=1e-4)


def test_kda_chunked_vs_segmented_anchor_state_parity_and_gradients():
    """Chunk-boundary + batched intra-chunk extraction must match segmented extraction."""
    torch.manual_seed(29)
    cfg = _tiny_kda_config(layer_types=["kimi_delta_attention"], kda_chunk_size=6)
    layer = KDALayer(cfg, layer_idx=0)
    bsz, ctx_len, heads, dk, dv = 2, 23, 4, 8, 8
    k_ctx = torch.nn.functional.normalize(torch.randn(bsz, ctx_len, heads, dk), dim=-1).detach().requires_grad_(True)
    v_ctx = torch.randn(bsz, ctx_len, heads, dv, requires_grad=True)
    g_ctx = (-torch.nn.functional.softplus(torch.randn(bsz, ctx_len, heads, dk))).detach().requires_grad_(True)
    b_ctx = torch.sigmoid(torch.randn(bsz, ctx_len, heads)).detach().requires_grad_(True)

    # Mix of pos=0, exact chunk boundary (6, 12), mid-chunk (3, 17, 22), and dropped anchor
    anchors = torch.tensor([[0, 3, 6, 12, 17, 22], [1, 6, 9, 12, 18, 20]], dtype=torch.long)
    keep = torch.tensor([[True, True, True, True, True, False], [True, True, False, True, True, True]])

    s_chunked = layer._extract_anchor_prefix_states_chunked(k_ctx, v_ctx, g_ctx, b_ctx, anchors, keep)
    g_chunked = torch.autograd.grad(s_chunked.pow(2).sum(), (k_ctx, v_ctx, g_ctx, b_ctx))

    s_seg = layer._extract_anchor_prefix_states_segmented(k_ctx, v_ctx, g_ctx, b_ctx, anchors, keep)
    g_seg = torch.autograd.grad(s_seg.pow(2).sum(), (k_ctx, v_ctx, g_ctx, b_ctx))

    torch.testing.assert_close(s_chunked, s_seg, atol=1e-5, rtol=1e-4)
    for gc, gs in zip(g_chunked, g_seg, strict=True):
        torch.testing.assert_close(gc, gs, atol=1e-4, rtol=1e-3)


def test_kda_fla_backend_fails_closed_on_cpu():
    cfg = _tiny_kda_config(layer_types=["kimi_delta_attention"], kda_backend="fla")
    layer = KDALayer(cfg, layer_idx=0)
    meta = KDAForwardMetadata(
        anchor_positions=torch.tensor([[3]], dtype=torch.long),
        block_keep_mask=torch.tensor([[True]]),
        block_size=2,
    )
    with pytest.raises(RuntimeError, match="CUDA"):
        layer(torch.randn(1, 2, 32), torch.randn(1, 6, 32), meta)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for FLA parity")
def test_kda_fla_vs_reference_cuda_parity():
    pytest.importorskip("fla.ops.kda")
    torch.manual_seed(19)
    cfg_ref = _tiny_kda_config(layer_types=["kimi_delta_attention"], kda_backend="reference")
    cfg_fla = _tiny_kda_config(layer_types=["kimi_delta_attention"], kda_backend="fla")
    layer_ref = KDALayer(cfg_ref, layer_idx=0).cuda().to(torch.bfloat16)
    layer_ref.keep_gate_precision_fp32()
    layer_fla = KDALayer(cfg_fla, layer_idx=0).cuda().to(torch.bfloat16)
    layer_fla.keep_gate_precision_fp32()
    layer_fla.load_state_dict(layer_ref.state_dict(), strict=True)

    ctx = torch.randn(1, 24, 32, device="cuda", dtype=torch.bfloat16)
    draft = torch.randn(1, 8, 32, device="cuda", dtype=torch.bfloat16)
    meta = KDAForwardMetadata(
        anchor_positions=torch.tensor([[6, 15]], device="cuda"),
        block_keep_mask=torch.tensor([[True, True]], device="cuda"),
        block_size=4,
    )
    out_ref, _ = layer_ref(draft, ctx, meta)
    out_fla, _ = layer_fla(draft, ctx, meta)
    torch.testing.assert_close(out_fla.float(), out_ref.float(), atol=2e-2, rtol=2e-2)


def test_hybrid_3_to_1_dspark_model_forward_backward_and_observations():
    torch.manual_seed(123)
    cfg = _tiny_kda_config(block_size=4)
    draft = DSparkDraftModel(cfg)
    draft.freeze_embedding()
    wrapper = DSparkModel(
        draft,
        block_size=4,
        num_anchors=3,
        attention_backend="sdpa",
        deterministic_anchor_sampling=True,
        anchor_seed=7,
    )

    bsz, seq_len = 1, 20
    input_ids = torch.randint(0, 60, (bsz, seq_len))
    hidden_list = [torch.randn(bsz, seq_len, 32) for _ in range(2)]
    last_hidden = torch.randn(bsz, seq_len, 32)
    loss_mask = torch.ones(bsz, seq_len)
    lm_head = torch.randn(64, 32)

    loss, acc, loss_pp, acc_pp, count_pp, components, (num, den) = wrapper(
        input_ids=input_ids,
        hidden_states_list=hidden_list,
        loss_mask=loss_mask,
        lm_head_weight=lm_head,
        last_hidden_states=last_hidden,
    )
    assert torch.isfinite(loss)
    assert den > 0
    loss.backward()

    # Verify every trainable parameter (all 3 KDA layers, full attn layer, FFN, Markov, Conf) got gradients
    missing_grads = [
        name for name, param in draft.named_parameters()
        if param.requires_grad and param.grad is None
    ]
    assert missing_grads == []

    obs = wrapper.attention_policy_observed
    assert obs["kda"]["backend"] == "reference"
    kda_layer_obs = [layer for layer in obs["layers"] if layer.get("type") == "kimi_delta_attention"]
    assert [layer["layer"] for layer in kda_layer_obs] == [0, 1, 2]


def test_kda_exact_resume_topology_and_native_export_proposal_roundtrip(tmp_path: Path):
    torch.manual_seed(321)
    cfg = _tiny_kda_config(block_size=4)

    def _make_args(model_cfg: DSparkConfig):
        return SimpleNamespace(
            world_size=1,
            draft_algorithm="dspark",
            draft_model_config_obj=model_cfg,
            draft_context_parallel_size=1,
            draft_tensor_parallel_size=1,
            micro_batch_size=1,
            draft_accumulation_steps=1,
            dflash_block_size=4,
            dflash_packing=False,
            dspark_loss_decay_gamma=4.0,
            dspark_ce_loss_alpha=0.1,
            dspark_l1_loss_alpha=0.9,
            dspark_confidence_head_alpha=1.0,
        )

    base_args = _make_args(cfg)
    saved_meta = {
        "world_size": 1,
        "training_topology": _logical_training_topology(base_args, 1),
    }
    assert saved_meta["training_topology"]["draft_attention_policy"]["kda"] is not None

    # 1. Identical KDA topology passes exact resume
    _validate_exact_resume_topology(base_args, saved_meta)

    # 2. Changing kda_chunk_size (execution tile) is allowed on exact resume
    cfg_chunk = _tiny_kda_config(block_size=4, kda_chunk_size=16)
    _validate_exact_resume_topology(_make_args(cfg_chunk), saved_meta)

    # 3. Changing KDA semantics (allow_neg_eigval) must fail closed on exact resume
    cfg_neg = _tiny_kda_config(block_size=4, allow_neg_eigval=True)
    with pytest.raises(RuntimeError, match="Exact resume"):
        _validate_exact_resume_topology(_make_args(cfg_neg), saved_meta)

    # 4. Native export (--format aspec) + strict reload + mixed FP32/BF16 + BlockDraftConsumer parity
    model = DSparkDraftModel(cfg).to(torch.bfloat16)
    model.keep_kda_gate_precision()
    assert model.layers[0].self_attn.kda_layer.A_log.dtype == torch.float32
    assert model.layers[0].self_attn.kda_layer.dt_bias.dtype == torch.float32
    assert model.layers[0].self_attn.kda_layer.q_proj.weight.dtype == torch.bfloat16

    ckpt_dir = tmp_path / "iter_0000001"
    ckpt_dir.mkdir()
    (ckpt_dir / "draft_config.json").write_text(json.dumps(cfg.to_dict()))
    dcp.save(
        {"model_state": {"model": {"draft_model": model.state_dict()}}},
        checkpoint_id=str(ckpt_dir / "model"),
    )

    # SGLang serving export must fail closed
    with pytest.raises(NotImplementedError, match="aspec"):
        convert_checkpoint(
            input_dir=ckpt_dir,
            output_dir=tmp_path / "sglang_out",
            artifact_format="sglang",
        )

    asset_dir = tmp_path / "native_asset"
    convert_checkpoint(
        input_dir=ckpt_dir,
        output_dir=asset_dir,
        artifact_format="aspec",
        verify=True,
    )
    verify_native_export(asset_dir, expected=model.state_dict())
    reloaded = load_native_draft(asset_dir, device="cpu")
    assert reloaded.layers[0].self_attn.kda_layer.A_log.dtype == torch.float32
    assert reloaded.layers[0].self_attn.kda_layer.dt_bias.dtype == torch.float32

    ids = torch.randint(0, 60, (1, 12))
    features = torch.randn(1, 12, 64, dtype=torch.bfloat16)
    head = torch.randn(64, 32, dtype=torch.bfloat16)
    expected = BlockDraftConsumer(model, block_size=4).propose(ids, features, lm_head_weight=head)
    actual = BlockDraftConsumer(reloaded, block_size=4).propose(ids, features, lm_head_weight=head)
    torch.testing.assert_close(actual["token_ids"], expected["token_ids"], atol=0, rtol=0)
    torch.testing.assert_close(
        actual["draft_confidence_logits"], expected["draft_confidence_logits"], atol=0, rtol=0
    )
