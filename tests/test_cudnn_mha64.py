# Copyright 2026 Ant Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""MHA64 DSpark loss and update parity across attention backends."""

import gc
import os

import pytest
import torch

pytestmark = pytest.mark.skipif(
    os.getenv("ASPEC_RUN_CUDNN_FLEX_INTEGRATION") != "1" or not torch.cuda.is_available(),
    reason="requires a CUDA cuDNN/FA4 training integration environment",
)


def _relative_l2(actual, expected):
    difference = reference = 0.0
    for a, b in zip(actual, expected, strict=True):
        a, b = a.cuda().float(), b.cuda().float()
        difference += (a - b).square().sum().item()
        reference += b.square().sum().item()
    return (difference / max(reference, 1e-30)) ** 0.5


def run_mha64_training(context_length, *, hidden_size=128, vocab_size=1024,
                       num_anchors=16, steps=3, gradient_checkpointing=False):
    """Compare identical-state VJPs and independent three-update trajectories."""
    from aspec.models.draft.dspark import DSparkConfig, DSparkDraftModel
    from aspec.models.dspark import DSparkModel
    from aspec.training.optimizer import BF16Optimizer

    class CausalBlockDSpark(DSparkModel):
        # Exercise causal block visibility through the existing mask-policy hook.
        def _block_mask_options(self):
            return {"is_causal": True}

    torch.manual_seed(811)
    config = DSparkConfig(
        hidden_size=hidden_size, intermediate_size=3 * hidden_size,
        num_hidden_layers=5, num_attention_heads=64, num_key_value_heads=64,
        head_dim=64, vocab_size=vocab_size, mask_token_id=vocab_size - 1,
        rms_norm_eps=1e-5, max_position_embeddings=1048576, rope_theta=10000,
        num_target_layers=6, target_hidden_size=hidden_size,
        target_num_hidden_layers=45, sliding_window=2048,
        block_size=8, markov_rank=256,
    )
    initial = DSparkDraftModel(config).bfloat16()
    initial.freeze_embedding()
    # Non-unit Q/K scales exercise learned normalization weights.
    with torch.no_grad():
        scale = torch.linspace(0.7, 1.3, 64)
        for layer in initial.layers:
            layer.self_attn.q_norm.weight.copy_(scale)
            layer.self_attn.k_norm.weight.copy_(scale.flip(0))
    state = {name: value.detach().cpu().clone() for name, value in initial.state_dict().items()}
    del initial
    inputs = torch.randint(vocab_size - 1, (1, context_length), device="cuda")
    features = torch.randn(1, context_length, 6 * hidden_size,
                           device="cuda", dtype=torch.bfloat16)
    teacher = torch.randn(1, context_length, hidden_size, device="cuda", dtype=torch.bfloat16)
    head = torch.randn(vocab_size, hidden_size, device="cuda", dtype=torch.bfloat16)
    head /= hidden_size ** 0.5
    supervision = torch.ones_like(inputs)
    supervision[:, -1] = 0
    records, snapshots, reference_states = {}, {}, []
    for backend in ("fa4", "flex_attention", "cudnn_flex_attention"):
        draft = DSparkDraftModel(config).bfloat16()
        draft.load_state_dict(state)
        if gradient_checkpointing:
            draft.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        draft.freeze_embedding()
        draft.cuda().train()
        model = CausalBlockDSpark(
            draft, block_size=8, num_anchors=num_anchors,
            attention_backend=backend, deterministic_anchor_sampling=True,
        )
        optimizer = BF16Optimizer(
            draft, lr=6e-4, min_lr=6e-5, total_steps=100, warmup_ratio=0,
            weight_decay=0, max_grad_norm=1.0,
        )
        history, states = [], []
        for step in range(steps):
            aligned_gradients = aligned_terms = aligned_loss = None
            if backend == "fa4":
                reference_states.append({name: value.detach().cpu().clone()
                                         for name, value in draft.state_dict().items()})
            elif step:
                # A backend precision comparison must use identical weights.
                # Preserve the independent trajectory around this read-only VJP replay.
                own_state = {name: value.detach().cpu().clone()
                             for name, value in draft.state_dict().items()}
                draft.load_state_dict(reference_states[step])
                aligned = model(inputs, features, supervision, head, teacher, anchor_seed=911 + step)
                aligned[0].backward()
                aligned_gradients = [p.grad.detach().cpu().clone()
                                     for p in draft.parameters() if p.requires_grad]
                aligned_terms = {name: float(numerator.detach() / denominator.detach().clamp_min(1e-8))
                                 for name, (numerator, denominator) in aligned[5].items()}
                aligned_loss = float(aligned[0].detach())
                optimizer.zero_grad()
                draft.load_state_dict(own_state)
                del aligned, own_state
            result = model(inputs, features, supervision, head, teacher, anchor_seed=911 + step)
            loss = result[0]
            loss.backward()
            parameters = [(name, p) for name, p in draft.named_parameters() if p.requires_grad]
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for _, p in parameters)
            gradients = [p.grad.detach().cpu().clone() for _, p in parameters]
            terms = {name: float(numerator.detach() / denominator.detach().clamp_min(1e-8))
                     for name, (numerator, denominator) in result[5].items()}
            if aligned_gradients is None:
                aligned_gradients, aligned_terms, aligned_loss = gradients, terms, float(loss.detach())
            qk_backends = [layer.self_attn.qk_prepare_backend for layer in draft.layers]
            norm = optimizer.step()
            masters = [p.detach().cpu().clone() for p in optimizer.fp32_params]
            assert all(torch.isfinite(p).all() for p in masters)
            assert torch.equal(draft.embed_tokens.weight.cpu(), state["embed_tokens.weight"])
            history.append({"loss": float(loss.detach()), "components": terms,
                            "grad_norm": float(norm), "qk_backends": qk_backends,
                            "aligned_loss": aligned_loss, "aligned_components": aligned_terms})
            states.append((gradients, masters, aligned_gradients))
        records[backend], snapshots[backend] = history, states
        del model, draft, optimizer, result, loss, parameters, gradients, masters
        gc.collect()
        torch.cuda.empty_cache()
    metrics = []
    for step in range(steps):
        comparisons = {}
        for candidate, baseline in (("fa4", "flex_attention"),
                                    ("cudnn_flex_attention", "fa4"),
                                    ("cudnn_flex_attention", "flex_attention")):
            a, b = records[candidate][step], records[baseline][step]
            comparisons[candidate + "_vs_" + baseline] = {
                "loss_absolute": abs(a["loss"] - b["loss"]),
                "loss_relative": abs(a["loss"] - b["loss"]) / max(abs(b["loss"]), 1e-12),
                "gradient_relative_l2": _relative_l2(snapshots[candidate][step][0],
                                                       snapshots[baseline][step][0]),
                "aligned_gradient_relative_l2": _relative_l2(snapshots[candidate][step][2],
                                                               snapshots[baseline][step][2]),
                "master_relative_l2": _relative_l2(snapshots[candidate][step][1],
                                                     snapshots[baseline][step][1]),
                "component_absolute": {name: abs(a["components"][name] - b["components"][name])
                                       for name in a["components"]},
                "aligned_loss_relative": abs(a["aligned_loss"] - b["aligned_loss"])
                                         / max(abs(b["aligned_loss"]), 1e-12),
                "aligned_component_absolute": {
                    name: abs(a["aligned_components"][name] - b["aligned_components"][name])
                    for name in a["aligned_components"]},
            }
        metrics.append(comparisons)
    return {"context_length": context_length,
            "hidden_size": hidden_size, "vocab_size": vocab_size,
            "gradient_checkpointing": gradient_checkpointing,
            "steps": steps, "records": records, "metrics": metrics}


@pytest.mark.parametrize("context_length", [8192, 262144])
@pytest.mark.parametrize("full_geometry", [
    pytest.param(False, id="bounded"),
    pytest.param(True, id="full", marks=pytest.mark.skipif(
        os.getenv("ASPEC_RUN_CUDNN_MHA64_FULL") != "1",
        reason="requires explicit full-width training validation",
    )),
])
def test_mha64_training_loss_and_updates(context_length, full_geometry):
    geometry = (dict(hidden_size=4096, vocab_size=154880, num_anchors=64,
                     gradient_checkpointing=context_length >= 262144)
                if full_geometry else {})
    result = run_mha64_training(context_length, **geometry)
    assert all(set(step["qk_backends"]) == {"aspec_cutedsl"}
               for step in result["records"]["cudnn_flex_attention"])
    for step, comparisons in enumerate(result["metrics"]):
        for pair, metric in comparisons.items():
            assert metric["loss_relative"] < 1e-3, result
            assert metric["aligned_loss_relative"] < 1e-3, result
            assert metric["aligned_gradient_relative_l2"] < 0.01, result
            assert metric["master_relative_l2"] < 1e-3, result
            baseline = result["records"][pair.split("_vs_")[1]][step]
            for actual, expected in (("component_absolute", "components"),
                                     ("aligned_component_absolute", "aligned_components")):
                for name, error in metric[actual].items():
                    assert error <= 1e-4 + 1e-3 * abs(baseline[expected][name]), result


@pytest.mark.parametrize("kv_heads,mode,error", [(16, "native", "Hq=Hkv"),
                                               (64, "internal", "Hq/Hkv=4")])
def test_mha64_rejects_unvalidated_gqa_modes(kv_heads, mode, error):
    from aspec.models.ops.cudnn_flex_attention import CudnnFlexMask, cudnn_flex_attention

    query = torch.zeros(1, 8, 64, 64, device="cuda", dtype=torch.bfloat16)
    key = torch.zeros(1, 8, kv_heads, 64, device="cuda", dtype=torch.bfloat16)
    mask = CudnnFlexMask(torch.full((1, 1, 8), 8, device="cuda", dtype=torch.int32))
    with pytest.raises(ValueError, match=error):
        cudnn_flex_attention(query, key, key, mask=mask, scale=64 ** -0.5, gqa_mode=mode)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_mha64_native_attention_matches_flex(dtype):
    from aspec.models.dflash import _create_dflash_mask_mod
    from aspec.models.ops.cudnn_flex_attention import build_dspark_mla_mask, cudnn_flex_attention
    from aspec.models.ops.flex_attention import (
        compile_friendly_create_block_mask,
        compile_friendly_flex_attention,
    )

    generator = torch.Generator(device="cuda").manual_seed(93)
    context_length, block_size, heads, dim = 71, 8, 4, 64
    anchors = torch.tensor([[17, 63]], device="cuda")
    keep = torch.ones_like(anchors, dtype=torch.bool)
    queries = block_size * anchors.shape[1]
    mask_options = dict(is_causal=True, context_window=32)
    mask_mod = _create_dflash_mask_mod(anchors, keep, context_length, block_size, **mask_options)
    flex_mask = compile_friendly_create_block_mask(
        mask_mod, B=1, H=None, Q_LEN=queries, KV_LEN=context_length + queries,
        device=torch.device("cuda"),
    )
    cudnn_mask = build_dspark_mla_mask(
        anchor_positions=anchors, block_keep_mask=keep, context_length=context_length,
        block_size=block_size, **mask_options,
    )
    inputs = [torch.randn(1, n, heads, dim, device="cuda", dtype=dtype, generator=generator)
              for n in (queries, context_length + queries, context_length + queries)]
    dout = torch.randn(inputs[0].shape, device="cuda", dtype=dtype, generator=generator)
    results = []
    for native in (False, True):
        q, k, v = [x.detach().clone().requires_grad_() for x in inputs]
        if native:
            out = cudnn_flex_attention(q, k, v, mask=cudnn_mask, scale=dim ** -0.5,
                                       gqa_mode="native")
        else:
            out = compile_friendly_flex_attention(
                q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
                block_mask=flex_mask, scale=dim ** -0.5,
            ).transpose(1, 2)
        gradients = torch.autograd.grad(out, (q, k, v), dout)
        assert torch.isfinite(out).all()
        assert all(torch.isfinite(g).all() for g in gradients)
        results.append((out, gradients))
    assert _relative_l2([results[1][0]], [results[0][0]]) < 0.01
    for actual, expected in zip(results[1][1], results[0][1], strict=True):
        assert _relative_l2([actual], [expected]) < 0.01
