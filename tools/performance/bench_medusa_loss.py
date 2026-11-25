# Copyright 2026 Ant Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Single-GPU Medusa heads-only loss benchmark.

Measures one forward+backward over the vocabulary heads, excluding Target
features, transport and orchestration.  Three cells are supported:

- ``reference``: the current chunked Medusa path (128-token CE chunks, optional
  gradient checkpointing).
- ``restructure``: a script-only attribution cell that gathers the whole head,
  uses one wide vocabulary GEMM and one ``dW`` accumulation.
- ``fused``: the Triton CE/argmax/dlogits epilogue plus one ``dW`` per head.

The script records loss/gradient fingerprints alongside timings so a faster
cell cannot silently change the training math.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from aspec.models.draft.medusa import MedusaConfig, MedusaDraftModel
from aspec.models.medusa import should_save_fused_logits
from aspec.models.ops.medusa_ce import medusa_head_ce, use_fused_medusa_ce
from aspec.training.optimizer import BF16Optimizer


def _percentile(values, fraction):
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def _summary(values):
    return {
        "median_ms": statistics.median(values),
        "mean_ms": statistics.fmean(values),
        "p10_ms": _percentile(values, 0.10),
        "p90_ms": _percentile(values, 0.90),
        "min_ms": min(values),
        "max_ms": max(values),
        "samples": len(values),
    }


def _arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("reference", "restructure", "fused", "both"),
                        default="both")
    parser.add_argument("--heads", type=int, default=5)
    parser.add_argument("--tokens", type=int, default=4094)
    parser.add_argument("--hidden", type=int, default=4096)
    parser.add_argument("--vocab", type=int, default=151936)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--row-chunk", type=int, default=None)
    parser.add_argument("--grad-ckpt", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fused-logits", choices=("auto", "save", "recompute"),
                        default="auto")
    parser.add_argument("--optimizer-step", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="medusa_loss_bench.json")
    args = parser.parse_args()
    if min(args.heads, args.tokens, args.hidden, args.vocab, args.warmup, args.iters) <= 0:
        parser.error("heads/tokens/hidden/vocab/warmup/iters must be positive")
    if args.optimizer_step and args.mode == "both":
        parser.error("--optimizer-step requires a single mode so A/B weights stay identical")
    if args.optimizer_step and args.heads * args.hidden * args.vocab > 200_000_000:
        parser.error("--optimizer-step is limited to scaled dimensions that fit 48 GB")
    return args


def _build_model(args):
    config = MedusaConfig(
        hidden_size=args.hidden,
        vocab_size=args.vocab,
        target_hidden_size=args.hidden,
        target_num_hidden_layers=4,
        medusa_num_heads=args.heads,
        medusa_num_layers=1,
    )
    model = MedusaDraftModel(config).cuda().bfloat16()
    with torch.no_grad():
        for head in model.medusa_head:
            nn.init.normal_(head[0].linear.weight, std=0.02)
    return model


def _make_inputs(args):
    generator = torch.Generator(device="cuda").manual_seed(args.seed)
    hidden = torch.randn(
        args.tokens, args.hidden, device="cuda", dtype=torch.bfloat16, generator=generator
    )
    targets = [
        torch.randint(0, args.vocab, (args.tokens,), device="cuda", generator=generator)
        for _ in range(args.heads)
    ]
    return hidden, targets


def _head_params(model):
    return [parameter for parameter in model.parameters() if parameter.requires_grad]


def _grad_stats(params):
    total = square = 0.0
    elements = 0
    for parameter in params:
        if parameter.grad is None:
            continue
        gradient = parameter.grad.detach().float()
        total += gradient.sum().item()
        square += gradient.pow(2).sum().item()
        elements += gradient.numel()
    return {
        "grad_sum": total,
        "grad_norm": math.sqrt(square),
        "grad_numel": elements,
    }


def _reference_head(head, hidden, targets, grad_ckpt):
    total = hidden.new_zeros((), dtype=torch.float32)
    hits = hidden.new_zeros((), dtype=torch.float32)
    for indices in torch.arange(hidden.shape[0], device=hidden.device).split(128):
        chunk_hidden = hidden[indices]
        chunk_targets = targets[indices]

        def loss_chunk(chunk_hidden, chunk_targets, head=head):
            logits = head(chunk_hidden)
            return (
                F.cross_entropy(logits.float(), chunk_targets, reduction="sum"),
                (logits.argmax(-1) == chunk_targets).float().sum(),
            )

        if grad_ckpt:
            loss, chunk_hits = checkpoint(
                loss_chunk, chunk_hidden, chunk_targets, use_reentrant=False
            )
        else:
            loss, chunk_hits = loss_chunk(chunk_hidden, chunk_targets)
        total = total + loss
        hits = hits + chunk_hits.detach()
    return total, hits


def _restructure_head(head, hidden, targets):
    projected = head[:-1](hidden)
    logits = F.linear(projected, head[-1].weight).float()
    return (
        F.cross_entropy(logits, targets, reduction="sum"),
        (logits.argmax(-1) == targets).float().sum(),
    )


def _fused_head(head, hidden, targets, save_logits, row_chunk):
    projected = head[:-1](hidden)
    nll, hits = medusa_head_ce(
        projected,
        head[-1].weight,
        targets,
        row_chunk=row_chunk,
        save_logits=save_logits,
    )
    return nll.sum(), hits.sum()


def _fused_save_logits(args):
    if args.fused_logits == "save":
        return True
    if args.fused_logits == "recompute":
        return False
    return should_save_fused_logits(
        args.tokens, args.vocab, args.heads, args.grad_ckpt
    )


def _run_mode(model, hidden, targets, mode, args):
    params = _head_params(model)
    fused_save = _fused_save_logits(args)
    optimizer = (
        BF16Optimizer(
            model,
            lr=1e-3,
            weight_decay=0.0,
            max_grad_norm=1.0,
            total_steps=max(1, args.warmup + args.iters + 1),
            warmup_ratio=0.0,
            decay_style="cosine",
        )
        if args.optimizer_step
        else None
    )

    def forward_backward(step_optimizer):
        for parameter in params:
            parameter.grad = None
        total = hidden.new_zeros((), dtype=torch.float32)
        hits = hidden.new_zeros((), dtype=torch.float32)
        for head, target in zip(model.medusa_head, targets, strict=True):
            if mode == "reference":
                loss, head_hits = _reference_head(head, hidden, target, args.grad_ckpt)
            elif mode == "restructure":
                loss, head_hits = _restructure_head(head, hidden, target)
            else:
                loss, head_hits = _fused_head(
                    head, hidden, target, fused_save, args.row_chunk
                )
            total = total + loss
            hits = hits + head_hits.detach()
        total.backward()
        if step_optimizer and optimizer is not None:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        return total.detach(), hits

    for _ in range(args.warmup):
        forward_backward(step_optimizer=False)
    for parameter in params:
        parameter.grad = None
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    peak_base = torch.cuda.memory_allocated()

    events = []
    for _ in range(args.iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        forward_backward(step_optimizer=args.optimizer_step)
        end.record()
        events.append((start, end))
    torch.cuda.synchronize()
    # Capture the measured peak before fingerprint conversions allocate.
    peak = torch.cuda.max_memory_allocated() - peak_base
    times = [start.elapsed_time(end) for start, end in events]

    # Fingerprint outside the timed window, without an optimizer step, so the
    # recorded gradients belong to the measured math.
    for parameter in params:
        parameter.grad = None
    loss, hits = forward_backward(step_optimizer=False)
    torch.cuda.synchronize()
    fingerprint = {
        "loss": float(loss.item()),
        "hits": float(hits.item()),
        **_grad_stats(params),
    }
    return {
        "mode": mode,
        "timing": _summary(times),
        "peak_extra_bytes": peak,
        "fingerprint": fingerprint,
    }


def main():
    args = _arguments()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    torch.cuda.set_device(0)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False

    model = _build_model(args)
    hidden, targets = _make_inputs(args)

    probe_hidden = hidden[: min(64, hidden.shape[0])]
    probe_weight = model.medusa_head[0][-1].weight
    fused_supported = use_fused_medusa_ce(
        probe_hidden, probe_weight, num_rows=probe_hidden.shape[0]
    )
    modes = (
        ["reference", "restructure", "fused"]
        if args.mode == "both"
        else [args.mode]
    )
    if "fused" in modes and not fused_supported:
        modes = [mode for mode in modes if mode != "fused"]
        if not modes:
            raise SystemExit("fused Medusa CE is not supported on this device/runtime")

    results = []
    for mode in modes:
        torch.cuda.empty_cache()
        results.append(_run_mode(model, hidden, targets, mode, args))

    capability = torch.cuda.get_device_capability(0)
    report = {
        "script": "tools/performance/bench_medusa_loss.py",
        "config": {
            "heads": args.heads,
            "tokens": args.tokens,
            "hidden": args.hidden,
            "vocab": args.vocab,
            "warmup": args.warmup,
            "iters": args.iters,
            "row_chunk": args.row_chunk,
            "grad_ckpt": args.grad_ckpt,
            "fused_logits": args.fused_logits,
            "optimizer_step": args.optimizer_step,
            "seed": args.seed,
        },
        "hardware": {
            "device": torch.cuda.get_device_name(0),
            "capability": list(capability),
            "total_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
        },
        "software": {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "triton": getattr(sys.modules.get("triton"), "__version__", None),
            "python": sys.version.split()[0],
        },
        "fused_supported": fused_supported,
        "results": results,
    }
    text = json.dumps(report, indent=2, sort_keys=True)
    print(text)
    with open(args.out, "w", encoding="utf-8") as handle:
        handle.write(text + "\n")


if __name__ == "__main__":
    main()
