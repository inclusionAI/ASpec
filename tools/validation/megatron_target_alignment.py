# Copyright 2026 Ant Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Compare frozen upstream Qwen3 prefill with the unmodified HF model.

Run with torchrun in the isolated Megatron environment. Captures use post-layer
residuals and the final normalized hidden state, matching ASpec's feature ABI.
This checks numerical correctness, not throughput or transfer lifecycle.
"""

import argparse
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist

from aspec.models.megatron import initialize_parallelism, load_qwen3_target


def error_metrics(value, reference):
    value, reference = value.float(), reference.float()
    delta = value - reference
    return {"relative_l2": (delta.norm() / reference.norm().clamp_min(1e-12)).item(),
            "max_absolute": delta.abs().max().item(),
            "cosine": torch.nn.functional.cosine_similarity(
                value.flatten(), reference.flatten(), dim=0).item()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sequence-length", type=int, default=128)
    parser.add_argument("--implementation", choices=("local", "transformer_engine"),
                        default="transformer_engine")
    args = parser.parse_args()
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl")
    groups = initialize_parallelism(dist.get_world_size())
    model, config = load_qwen3_target(args.model, groups, implementation=args.implementation)
    layers = [1, 9, 17, 25, 33]
    captured = {}
    for index in layers:
        def capture(_module, _inputs, output, index=index):
            value = output[0] if isinstance(output, tuple) else output
            captured[str(index)] = value.squeeze(1).detach().clone()

        model.decoder.layers[index].register_forward_hook(capture)

    def finish(*, hidden_states, **_kwargs):
        captured["final"] = hidden_states.squeeze(1).detach().clone()

    # Fixed tokens and positions on all TP ranks, independent of a tokenizer.
    ids = (torch.arange(args.sequence_length, device="cuda") * 137 + 42) % config.vocab_size
    versions = [p._version for p in model.parameters()]
    with torch.inference_mode():
        model(input_ids=ids.unsqueeze(0),
              position_ids=torch.arange(ids.numel(), device="cuda").unsqueeze(0),
              attention_mask=None, output_processor=finish)
    frozen = (not any(p.requires_grad for p in model.parameters())
              and versions == [p._version for p in model.parameters()])
    if dist.get_rank() == 0:
        from transformers import AutoModelForCausalLM

        reference = AutoModelForCausalLM.from_pretrained(
            args.model, dtype=torch.bfloat16, attn_implementation="sdpa",
            trust_remote_code=False).cuda().eval().requires_grad_(False)
        expected = {}
        for index in layers:
            def capture_reference(_module, _inputs, output, index=index):
                value = output[0] if isinstance(output, tuple) else output
                expected[str(index)] = value.squeeze(0).detach().clone()

            reference.model.layers[index].register_forward_hook(capture_reference)
        with torch.inference_mode():
            expected["final"] = reference.model(input_ids=ids.unsqueeze(0)).last_hidden_state.squeeze(0)
        measurements = {key: error_metrics(captured[key], value) for key, value in expected.items()}
        # Declared before observing results; no per-layer fitted thresholds.
        passed = frozen and all(m["relative_l2"] <= 0.02 and m["cosine"] >= 0.999
                                for m in measurements.values())
        result = {"passed": passed, "target_frozen": frozen, "features": measurements,
                  "tensor_parallel": dist.get_world_size(), "sequence_length": ids.numel(),
                  "implementation": args.implementation, "torch": torch.__version__,
                  "cuda": torch.version.cuda, "device_capability": torch.cuda.get_device_capability(),
                  "module": str(Path(__import__("aspec").__file__).resolve()),
                  "tolerances": {"relative_l2_max": 0.02, "cosine_min": 0.999}}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result), flush=True)
    else:
        passed = None
    outcome = [passed]
    dist.broadcast_object_list(outcome, src=0)
    dist.destroy_process_group()
    if not outcome[0]:
        raise SystemExit("Qwen3 feature alignment failed; inspect the recorded measurements")


if __name__ == "__main__":
    main()
