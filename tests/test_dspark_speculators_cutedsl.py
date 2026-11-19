# Copyright 2026 Ant Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Red Hat CE/TV and the Markov epilogue, including BF16 rounding."""

import os

import pytest
import torch

pytestmark = pytest.mark.skipif(
    os.getenv("ASPEC_RUN_CUTEDSL_INTEGRATION") != "1"
    or not torch.cuda.is_available()
    or torch.cuda.get_device_capability() not in {(9, 0), (10, 0)},
    reason="requires a Hopper/Blackwell CuTeDSL integration environment",
)


@pytest.mark.parametrize("vocab", [257, 154880])
@pytest.mark.parametrize("indexed", [False, True])
@pytest.mark.parametrize("teacher_kind", ["tie", "independent"])
def test_speculators_cute_components_and_gradient(vocab, indexed, teacher_kind):
    from aspec.kernels.cutedsl.speculators_loss import speculators_cutedsl_loss

    gen = torch.Generator(device="cuda").manual_seed(6)
    rows = 32
    x = torch.randn(1, rows, vocab, generator=gen, device="cuda", dtype=torch.bfloat16)
    # The identical row and a permutation exercise the <= branch independently
    # of random probabilities. Duplicate maxima straddle vocabulary tiles.
    x[0, :, [2, vocab - 1]] = 4
    if teacher_kind == "tie":
        t = x.clone()
        t[:, 1:, :2] = x[:, 1:, [1, 0]]
    else:
        t = torch.randn(x.shape, generator=gen, device=x.device, dtype=x.dtype) * 1.5 + 4
    idx = None
    if indexed:
        idx = torch.arange(rows, device=x.device).roll(1)
        idx[-1] = idx[0]
        x = x.index_select(1, idx)
    teacher = t if idx is None else t.index_select(1, idx)
    x.requires_grad_()
    ce, tv, label, correct = speculators_cutedsl_loss(x, t, idx)
    q = x.detach().double().softmax(-1)
    p = teacher.double().softmax(-1)
    if teacher_kind == "independent":
        student_norm = x.detach().double().logsumexp(-1)
        teacher_norm = teacher.double().logsumexp(-1)
        assert (teacher_norm - student_norm).abs().min() > 0.5
        wrong_teacher = (teacher.double() - student_norm[..., None]).exp()
        expected_tv = 1 - torch.minimum(q, p).sum(-1)
        wrong_tv = 1 - torch.minimum(q, wrong_teacher).sum(-1)
        assert (expected_tv - wrong_tv).abs().min() > 1e-3
    w = torch.linspace(-0.5, 0.5, rows, device=x.device).view(1, -1)
    w[:, 5] = 0
    target = teacher.argmax(-1)
    torch.testing.assert_close(label, target)
    torch.testing.assert_close(correct, target == x.argmax(-1))
    ref_ce = x.double().logsumexp(-1) - x.double().gather(-1, target[..., None]).squeeze(-1)
    torch.testing.assert_close(ce.double(), ref_ce, rtol=0, atol=2e-6)
    torch.testing.assert_close(tv.double(), 1 - torch.minimum(q, p).sum(-1), rtol=0, atol=2e-6)
    # Raw logits resolve exact ties only when the partition sums are identical.
    indicator = ((x.detach() <= teacher) if teacher_kind == "tie" else (q <= p)).double()
    mass = (q * indicator).sum(-1, keepdim=True)
    gt = (w[..., None] * q * (mass - indicator)).bfloat16()
    gc = q.clone()
    gc.scatter_add_(-1, target[..., None], -torch.ones_like(target[..., None], dtype=q.dtype))
    gc = (gc * (-w[..., None])).bfloat16()
    for loss, coeff, grad in ((ce, -w, gc), (tv, w, gt)):
        (g,) = torch.autograd.grad(loss, x, coeff, retain_graph=True)
        torch.testing.assert_close(g, grad, rtol=0.009, atol=2e-7)
    (g,) = torch.autograd.grad((ce, tv), x, (-w, w))
    torch.testing.assert_close(g, gc + gt, rtol=0.012, atol=2e-7)


def test_markov_epilogue_rounds_before_add_and_both_inputs_receive_gradient():
    from aspec.kernels.cutedsl.speculators_loss import markov_residual_projection

    gen = torch.Generator(device="cuda").manual_seed(42)

    def sample(shape):
        return torch.randn(
            shape, generator=gen, dtype=torch.bfloat16, device="cuda"
        ).requires_grad_()

    base, emb, weight = sample((1, 32, 8, 4096)), sample((1, 32, 8, 256)), sample((4096, 256))
    ref = base + torch.nn.functional.linear(emb, weight)
    out = markov_residual_projection(base, emb, weight)
    torch.testing.assert_close(out, ref, rtol=0, atol=0)
    g = torch.randn(ref.shape, generator=gen, dtype=ref.dtype, device=ref.device)
    xs = (base, emb, weight)
    for actual, expected in zip(
        torch.autograd.grad(out, xs, g), torch.autograd.grad(ref, xs, g), strict=True
    ):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("context_length", [256, 262144])
def test_markov_confidence_preserves_separate_embedding_gradient_reductions(context_length):
    import copy

    from aspec.kernels.cutedsl.speculators_loss import speculators_cutedsl_loss
    from aspec.models.draft.dspark import AcceptRatePredictor, VanillaMarkov
    from aspec.models.dspark import DSparkModel

    torch.manual_seed(113)
    draft = torch.nn.Module()
    draft.markov_head = VanillaMarkov(vocab_size=1024, markov_rank=256)
    draft.confidence_head = AcceptRatePredictor(32 + 256)
    draft.confidence_head_with_markov = True
    draft.cuda().bfloat16()
    reference = copy.deepcopy(draft)
    hidden = torch.randn(1, 32, 8, 32, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    expected_hidden = hidden.detach().clone().requires_grad_()
    teacher = torch.randn_like(hidden)
    head = torch.randn(1024, 32, device="cuda", dtype=torch.bfloat16)
    previous = torch.randint(4, (1, 32, 8), device="cuda")
    weights = torch.rand(1, 32, 8, device="cuda") / 256
    context = teacher.new_zeros(1, context_length, 32)
    context[:, :256] = teacher.view(1, 256, 32)
    model = DSparkModel(draft, loss_recipe="speculators_redhat_v1")
    actual = model._prediction_terms(
        hidden,
        previous,
        previous,
        head,
        teacher,
        torch.ones_like(weights),
        weights,
        teacher_context=(context, torch.arange(256, device="cuda").view(1, 32, 8)),
    )
    (0.1 * actual[0] + 0.9 * actual[1] + actual[2]).backward()

    logits = torch.nn.functional.linear(expected_hidden, head)
    logits = reference.markov_head.apply_block_logits(logits, token_ids=previous)
    ce, tv, _, _ = speculators_cutedsl_loss(logits, torch.nn.functional.linear(teacher, head))
    embedding = reference.markov_head.get_prev_embeddings(previous)
    prediction = reference.confidence_head(torch.cat([expected_hidden, embedding], dim=-1))
    target = (1 - tv).detach().to(prediction.dtype)
    bce = torch.nn.functional.binary_cross_entropy_with_logits(
        prediction.float(), target.float(), reduction="none"
    )
    expected = tuple((value * weights).sum() for value in (ce, tv, bce))
    (0.1 * expected[0] + 0.9 * expected[1] + expected[2]).backward()
    assert model.l1_kernel_invocations["cutedsl_markov_projection"] == 1
    for value, wanted in zip(actual[:3], expected, strict=True):
        torch.testing.assert_close(value, wanted, rtol=0, atol=0)
    torch.testing.assert_close(hidden.grad, expected_hidden.grad, rtol=0, atol=0)
    for value, wanted in zip(draft.parameters(), reference.parameters(), strict=True):
        torch.testing.assert_close(value.grad, wanted.grad, rtol=0, atol=0)


def test_pinned_speculators_fused_gpu_oracle():
    """Optional release gate: supply the unmodified b9c51bc src/speculators/losses/fused.py."""
    import hashlib
    import importlib.util
    import sys
    from pathlib import Path

    from aspec.kernels.cutedsl.speculators_loss import speculators_cutedsl_loss

    reference = os.environ.get("ASPEC_SPECULATORS_FUSED_REFERENCE")
    if not reference:
        pytest.skip("ASPEC_SPECULATORS_FUSED_REFERENCE not supplied")
    path = Path(reference).resolve()
    assert hashlib.sha256(path.read_bytes()).hexdigest() == (
        "0465967844a5850facdb3f5c05c22a3f94569a180121c51e867939fa878b6a77"
    )
    spec = importlib.util.spec_from_file_location("_aspec_pinned_speculators_fused", path)
    lib = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = lib
    spec.loader.exec_module(lib)
    for vocab in (257, 154880):
        gen = torch.Generator(device="cuda").manual_seed(79)
        d = torch.randn(1, 32, vocab, generator=gen, device="cuda", dtype=torch.bfloat16)
        t = d.roll(vocab // 2 + 1, dims=-1)
        d.requires_grad_()
        ce, tv, label, correct = speculators_cutedsl_loss(d, t)
        torch.testing.assert_close(label, t.argmax(-1))
        torch.testing.assert_close(correct, d.argmax(-1) == label)
        ref_ce, ref_tv = lib.fused_ce_loss(d, t), lib.fused_tv_loss(d, t)
        torch.testing.assert_close(ce, ref_ce, rtol=0, atol=1e-6)
        torch.testing.assert_close(tv, ref_tv, rtol=0, atol=3e-7)
        # Compare TV alone: CE at the teacher label can drown out the wrong subgradient.
        (actual,) = torch.autograd.grad(tv.sum(), d)
        (expected,) = torch.autograd.grad(ref_tv.sum(), d)
        torch.testing.assert_close(actual, expected, rtol=0.008, atol=1e-9)


@pytest.mark.parametrize("stride, last_offset, unique_rows", [(1, 0, 25), (8, -1, 0), (8, 0, 0)])
def test_teacher_projection_policy_preserves_values_and_gradients(stride, last_offset, unique_rows):
    from aspec.models.dspark import DSparkModel

    gen = torch.Generator(device="cuda").manual_seed(402)
    teacher = torch.randn(1, 256, 32, generator=gen, device="cuda", dtype=torch.bfloat16)
    head = torch.randn(1024, 32, generator=gen, device="cuda", dtype=torch.bfloat16)
    ids = torch.randint(1024, (1, 256), generator=gen, device="cuda")
    active = 4 + stride * torch.arange(18, device="cuda")
    active[-1] += last_offset
    anchors = torch.cat([active[:7], active.new_zeros(1), active[7:]])[None]
    keep = anchors != 0
    positions = anchors[:, :, None] + torch.arange(8, device="cuda")
    # Equal draft and teacher distributions exercise the TV subgradient at a tie.
    hidden = teacher[0, positions].reshape(1, -1, 32).requires_grad_()
    draft = torch.nn.Module()
    draft.markov_head = draft.confidence_head = None
    model = DSparkModel(
        draft,
        block_size=8,
        num_anchors=19,
        confidence_head_alpha=0,
        loss_recipe="speculators_redhat_v1",
    )
    model._draft_backbone = lambda *args, **kwargs: (hidden, anchors, keep, 19)

    def evaluate():
        outputs = model(ids, [], torch.ones_like(ids), head, last_hidden_states=teacher)
        (grad,) = torch.autograd.grad(outputs[0], hidden)
        return (*outputs[:5], grad)

    model._reuse_head_projections = False
    reference = evaluate()
    model._reuse_head_projections = True
    actual = evaluate()
    if torch.cuda.get_device_capability() == (10, 0):
        unique_rows = 0
    assert model.loss_head_cached_context_tokens_observed == unique_rows
    assert model.loss_head_cached_teacher_blocks_observed == (18 if unique_rows else 0)
    assert model.l1_kernel_invocations.get("indexed_cutedsl_ce_tv", 0) == int(unique_rows > 0)
    for observed, expected in zip(actual, reference, strict=True):
        torch.testing.assert_close(observed, expected, rtol=0, atol=0)
