#!/usr/bin/env python3
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

"""Feasibility benchmark for absorbed versus materialized MLA attention.

The workload starts at the DSpark MLA attention boundary (BSHD query/latent
tensors, BF16, no dense mask). It compares:

- ``materialized_sdpa``: expand ``kv_b_proj`` and call fused SDPA;
- ``materialized_explicit``: expand ``kv_b_proj`` and materialize logits;
- ``absorbed``: keep the normalized latent and re-associate both projections;
- ``absorbed_fused``: the same absorbed identity with the fused Triton
  score/softmax/latent core (when Triton and CUDA are available).

FlexAttention/cuDNN masks, packed layouts and Direct-CP are outside this
benchmark. Report the boundary together with the result; this does not establish
end-to-end training throughput.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

from aspec.models.ops.absorbed_mla import absorbed_mla_attention
from aspec.models.ops.absorbed_mla_fused import (
    _HAS_TRITON,
    fused_absorbed_mla_attention,
)


@dataclass(frozen=True)
class Workload:
    context_length: int = 4096
    num_anchors: int = 512
    block_size: int = 7
    num_heads: int = 32
    qk_nope_head_dim: int = 128
    qk_rope_head_dim: int = 64
    v_head_dim: int = 128
    kv_lora_rank: int = 512
    batch_size: int = 1

    @property
    def query_length(self) -> int:
        return self.num_anchors * self.block_size

    @property
    def key_length(self) -> int:
        return self.context_length + self.query_length

    @property
    def softmax_scale(self) -> float:
        return 1.0 / math.sqrt(self.qk_nope_head_dim + self.qk_rope_head_dim)


def build_inputs(workload: Workload, device: torch.device, dtype: torch.dtype):
    generator = torch.Generator(device="cpu").manual_seed(7)
    shape = (workload.batch_size, workload.query_length, workload.num_heads)
    tensors = {
        "q_nope": torch.randn(
            *shape, workload.qk_nope_head_dim, generator=generator
        ).to(device=device, dtype=dtype),
        "q_rope": torch.randn(
            *shape, workload.qk_rope_head_dim, generator=generator
        ).to(device=device, dtype=dtype),
        "compressed_kv": torch.randn(
            workload.batch_size, workload.key_length, workload.kv_lora_rank,
            generator=generator,
        ).to(device=device, dtype=dtype),
        "k_rope": torch.randn(
            workload.batch_size, workload.key_length, workload.qk_rope_head_dim,
            generator=generator,
        ).to(device=device, dtype=dtype),
        "kv_b_proj_weight": torch.randn(
            workload.num_heads
            * (workload.qk_nope_head_dim + workload.v_head_dim),
            workload.kv_lora_rank,
            generator=generator,
        ).to(device=device, dtype=dtype),
    }
    for tensor in tensors.values():
        tensor.requires_grad_(True)
    return tensors


def split_projection(weight, workload: Workload):
    view = weight.view(
        workload.num_heads,
        workload.qk_nope_head_dim + workload.v_head_dim,
        workload.kv_lora_rank,
    )
    return view[:, : workload.qk_nope_head_dim], view[:, workload.qk_nope_head_dim :]


def materialized_kv(compressed_kv, weight, workload: Workload):
    w_uk, w_uv = split_projection(weight, workload)
    k_nope = torch.einsum("btk,hnk->bthn", compressed_kv, w_uk)
    values = torch.einsum("btk,hvk->bthv", compressed_kv, w_uv)
    return k_nope, values


def materialized_sdpa(tensors, workload: Workload):
    q_nope = tensors["q_nope"].transpose(1, 2)
    q_rope = tensors["q_rope"].transpose(1, 2)
    k_nope, values = materialized_kv(
        tensors["compressed_kv"], tensors["kv_b_proj_weight"], workload
    )
    key = torch.cat(
        [
            k_nope.transpose(1, 2),
            tensors["k_rope"].unsqueeze(1).expand(-1, workload.num_heads, -1, -1),
        ],
        dim=-1,
    )
    query = torch.cat([q_nope, q_rope], dim=-1)
    output = F.scaled_dot_product_attention(
        query, key, values.transpose(1, 2), is_causal=False, dropout_p=0.0,
        scale=workload.softmax_scale,
    )
    return output.transpose(1, 2).reshape(
        workload.batch_size, workload.query_length,
        workload.num_heads * workload.v_head_dim,
    )


def materialized_explicit(tensors, workload: Workload):
    q_nope = tensors["q_nope"]
    k_nope, values = materialized_kv(
        tensors["compressed_kv"], tensors["kv_b_proj_weight"], workload
    )
    scores = torch.einsum("bshn,bthn->bhst", q_nope, k_nope)
    scores = scores + torch.einsum(
        "bshr,btr->bhst", tensors["q_rope"], tensors["k_rope"]
    )
    scores = scores * workload.softmax_scale
    probs = torch.softmax(scores.float(), dim=-1).to(q_nope.dtype)
    output = torch.einsum("bhst,bthv->bshv", probs, values)
    return output.reshape(
        workload.batch_size, workload.query_length,
        workload.num_heads * workload.v_head_dim,
    )


def absorbed(tensors, workload: Workload):
    return absorbed_mla_attention(
        tensors["q_nope"], tensors["q_rope"], tensors["compressed_kv"],
        tensors["k_rope"], tensors["kv_b_proj_weight"],
        softmax_scale=workload.softmax_scale,
    )


def absorbed_fused(tensors, workload: Workload):
    return fused_absorbed_mla_attention(
        tensors["q_nope"], tensors["q_rope"], tensors["compressed_kv"],
        tensors["k_rope"], tensors["kv_b_proj_weight"],
        softmax_scale=workload.softmax_scale,
    )


def run_step(fn, tensors, workload):
    output = fn(tensors, workload)
    loss = output.float().pow(2).sum()
    return torch.autograd.grad(loss, list(tensors.values()))


def measure_variants(variants, tensors, workload, *, warmup: int, runs: int):
    """Time each variant in rotating order and record per-round raw samples."""
    names = list(variants)
    for round_index in range(warmup):
        order = names[round_index % len(names):] + names[: round_index % len(names)]
        for name in order:
            run_step(variants[name], tensors, workload)
    torch.cuda.synchronize()
    samples = {name: [] for name in names}
    peaks = {name: [] for name in names}
    orders = []
    for round_index in range(runs):
        offset = round_index % len(names)
        order = names[offset:] + names[:offset]
        orders.append(order)
        for name in order:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            run_step(variants[name], tensors, workload)
            end.record()
            end.synchronize()
            samples[name].append(start.elapsed_time(end))
            peaks[name].append(torch.cuda.max_memory_allocated())
    results = {}
    for name in names:
        results[name] = {
            "median_ms": statistics.median(samples[name]),
            "min_ms": min(samples[name]),
            "max_ms": max(samples[name]),
            "samples_ms": samples[name],
            "peak_memory_samples_bytes": peaks[name],
            "peak_memory_bytes": max(peaks[name]),
        }
    return results, orders


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--context-length", type=int, default=4096)
    parser.add_argument("--num-anchors", type=int, default=512)
    parser.add_argument("--block-size", type=int, default=7)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--result", type=Path)
    args = parser.parse_args(argv)
    if not torch.cuda.is_available():
        raise SystemExit("absorbed MLA benchmark requires CUDA")
    if args.warmup < 1 or args.runs < 1:
        raise SystemExit("warmup and runs must be positive")
    workload = Workload(
        context_length=args.context_length,
        num_anchors=args.num_anchors,
        block_size=args.block_size,
        batch_size=args.batch_size,
    )
    dtype = getattr(torch, args.dtype)
    device = torch.device("cuda")
    tensors = build_inputs(workload, device, dtype)
    variants = {
        "materialized_sdpa": materialized_sdpa,
        "materialized_explicit": materialized_explicit,
        "absorbed": absorbed,
    }
    if _HAS_TRITON:
        variants["absorbed_fused"] = absorbed_fused
    results = {
        "workload": asdict(workload) | {
            "query_length": workload.query_length,
            "key_length": workload.key_length,
            "dtype": args.dtype,
        },
        "device": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "boundary": "attention core, dense unmasked, forward+backward, no o_proj",
        "variants": {},
    }
    measured, orders = measure_variants(
        variants, tensors, workload, warmup=args.warmup, runs=args.runs
    )
    results["variant_order"] = orders
    results["variants"] = measured
    for name, value in measured.items():
        print(name, json.dumps(value))
    if args.result:
        args.result.write_text(json.dumps(results, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
