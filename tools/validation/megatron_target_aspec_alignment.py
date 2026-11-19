# Copyright 2026 Ant Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Compare frozen Qwen3 Target features with a same-TP ASpec/SGLang cache.

Run with torchrun in the Megatron environment.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist

from aspec.models.megatron import initialize_parallelism, load_qwen3_target


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--reference-cache", type=Path, required=True)
    parser.add_argument("--sample", type=int, default=0)
    parser.add_argument("--length", type=int, default=4096)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl")
    cached = torch.load(args.reference_cache, map_location="cpu", weights_only=False, mmap=True)
    identity = cached["identity"]
    contract = identity["inputs"]["aspec_feature_contract"]
    if contract["producer"]["backend"] != "sglang":
        raise ValueError("Reference cache must come from the existing ASpec SGLang producer")
    if contract["tensors"]["last_hidden_states"]["value_stage"] != "post_norm":
        raise ValueError("Expected the existing post-norm Target feature ABI")
    sample = cached["samples"][args.sample]
    if sample["input_ids"].shape[0] != 1 or sample["input_ids"].shape[1] < args.length:
        raise ValueError("Select one complete reference sequence")
    layers = contract["aux_hidden_state_layers"]
    groups = initialize_parallelism(dist.get_world_size())
    model, config = load_qwen3_target(args.model, groups)
    captured = {}

    def capture(layer, branch, residual):
        captured[str(layer)] = branch + residual

    versions = [p._version for p in model.parameters()]
    ids = sample["input_ids"][:, :args.length].cuda()
    captured["final"] = model.prefill(ids, layers, capture)
    frozen = (not any(p.requires_grad for p in model.parameters())
              and versions == [p._version for p in model.parameters()])
    verdict = [None]
    if dist.get_rank() == 0:
        expected = {str(layer): sample["hidden_states"][0, :args.length,
                            index * config.hidden_size:(index + 1) * config.hidden_size].cuda()
                    for index, layer in enumerate(layers)}
        expected["final"] = sample["last_hidden_states"][0, :args.length].cuda()
        measurements = {}
        for name, target in expected.items():
            value, target = captured[name].double(), target.double()
            measurements[name] = {
                "mean": value.mean().item(), "reference_mean": target.mean().item(),
                "equal": torch.equal(value, target),
                "relative_l2": ((value - target).norm() / target.norm().clamp_min(1e-12)).item(),
                "max_absolute": (value - target).abs().max().item(),
                "cosine": torch.nn.functional.cosine_similarity(value.flatten(), target.flatten(), dim=0).item()}
        passed = frozen and all(row["relative_l2"] <= 0.02 and row["cosine"] >= 0.999
                                for row in measurements.values())
        result = {"reference_tp": identity["inputs"].get("sglang_tp_size"), "model_class": type(model).__name__,
                  "passed": passed, "reference": "existing_aspec_sglang_feature_cache",
                  "reference_sha256": sha(args.reference_cache), "reference_identity": identity,
                  "sample": args.sample, "length": args.length, "tp": dist.get_world_size(),
                  "target_frozen": frozen, "features": measurements,
                  "tolerances": {"relative_l2_max": 0.02, "cosine_min": 0.999}}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as stream:
            json.dump(result, stream, indent=2)
        print(json.dumps({key: value for key, value in result.items() if key != "reference_identity"}), flush=True)
        verdict[0] = passed
    dist.broadcast_object_list(verdict, src=0)
    dist.destroy_process_group()
    if not verdict[0]:
        raise SystemExit("Megatron features did not align with the existing ASpec Target")


if __name__ == "__main__":
    main()
