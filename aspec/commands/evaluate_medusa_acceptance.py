# Copyright 2026 Ant Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Measure Medusa greedy acceptance and decode equality with a Hugging Face Target."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch


def _choices(path, heads):
    if path is not None:
        choices = json.loads(Path(path).read_text())
    else:
        from aspec.inference.medusa_consumer import DEFAULT_MEDUSA_CHOICES

        choices = DEFAULT_MEDUSA_CHOICES
    filtered = [list(item) for item in choices if len(item) <= heads]
    if not filtered:
        raise ValueError("No medusa_choices path fits the draft's head depth")
    return filtered


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--draft", type=Path, required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--choices", type=Path, default=None)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--verify-batch-size", type=int, default=16)
    parser.add_argument("--sampling", choices=("greedy", "typical", "nucleus"),
                        default="greedy")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.8, dest="top_p")
    parser.add_argument("--posterior-threshold", type=float, default=0.3,
                        dest="posterior_threshold")
    parser.add_argument("--posterior-alpha", type=float, default=0.09,
                        dest="posterior_alpha")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--enable-thinking", action="store_true")
    return parser.parse_args()


def main():
    args = _parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from aspec.commands.evaluate_draft import load_prompts
    from aspec.draft_artifact import load_native_draft
    from aspec.inference.medusa_consumer import HFCausalVerifier, MedusaConsumer

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = getattr(torch, args.dtype)
    tokenizer = AutoTokenizer.from_pretrained(
        args.target, trust_remote_code=args.trust_remote_code)
    target = AutoModelForCausalLM.from_pretrained(
        args.target, torch_dtype=dtype, trust_remote_code=args.trust_remote_code)
    target.to(device).eval()
    draft = load_native_draft(args.draft, device=device, dtype=dtype)
    choices = _choices(args.choices, draft.medusa_num_heads)
    verifier = HFCausalVerifier(
        target, device=device, verify_batch_size=args.verify_batch_size)
    consumer = MedusaConsumer(
        draft, verifier, medusa_choices=choices, top_k=args.top_k)
    prompts = load_prompts(
        args.prompts, tokenizer, args.limit,
        enable_thinking=True if args.enable_thinking else None)

    greedy = args.sampling == "greedy"
    if greedy:
        temperature = 0.0
    else:
        if args.temperature <= 0:
            raise ValueError("sampling acceptance requires --temperature > 0")
        temperature = args.temperature

    def generator():
        value = torch.Generator(device=torch.device(device))
        value.manual_seed(args.seed)
        return value

    eos = tokenizer.eos_token_id
    results = []
    for ids in prompts:
        prefix = torch.tensor(ids, dtype=torch.long)
        if greedy:
            baseline = verifier.greedy_generate(
                prefix, max_new_tokens=args.max_new_tokens, eos_token_id=eos)
        else:
            baseline = None
        speculative = consumer.generate(
            prefix, max_new_tokens=args.max_new_tokens, eos_token_id=eos,
            sampling=args.sampling, temperature=temperature, top_p=args.top_p,
            posterior_threshold=args.posterior_threshold,
            posterior_alpha=args.posterior_alpha,
            generator=None if args.sampling != "nucleus" else generator())
        repeat = consumer.generate(
            prefix, max_new_tokens=args.max_new_tokens, eos_token_id=eos,
            sampling=args.sampling, temperature=temperature, top_p=args.top_p,
            posterior_threshold=args.posterior_threshold,
            posterior_alpha=args.posterior_alpha,
            generator=None if args.sampling != "nucleus" else generator())
        start = len(ids)
        equal = None
        if baseline is not None:
            equal = (
                speculative["token_ids"][start:].tolist()
                == baseline["token_ids"][start:].tolist())
        results.append({
            "prompt_sha256": hashlib.sha256(
                json.dumps(ids).encode("utf-8")).hexdigest(),
            "prompt_tokens": len(ids),
            "generated_tokens": speculative["generated_tokens"],
            "verifier_calls": speculative["verifier_calls"],
            "acceptance_length": speculative["acceptance_length"],
            "decode_equal": equal,
            "repeat_stable": (
                speculative["token_ids"].tolist() == repeat["token_ids"].tolist()),
        })

    total_tokens = sum(item["generated_tokens"] for item in results)
    total_calls = sum(item["verifier_calls"] for item in results)
    if total_calls < 1:
        raise ValueError("Acceptance measurement produced no verifier calls")
    choices_blob = json.dumps(choices, sort_keys=True, separators=(",", ":"))
    report = {
        "schema_version": "aspec.medusa-acceptance/v1",
        "draft_manifest_sha256": hashlib.sha256(
            (args.draft / "export_manifest.json").read_bytes()).hexdigest(),
        "target": args.target,
        "prompts": str(args.prompts),
        "prompt_count": len(results),
        "max_new_tokens": args.max_new_tokens,
        "top_k": args.top_k,
        "sampling": args.sampling,
        "temperature": temperature,
        "top_p": args.top_p if not greedy else None,
        "posterior_threshold": (args.posterior_threshold if not greedy else None),
        "posterior_alpha": args.posterior_alpha if not greedy else None,
        "seed": args.seed if args.sampling == "nucleus" else None,
        "choices_sha256": hashlib.sha256(choices_blob.encode("utf-8")).hexdigest(),
        "choices_count": len(choices),
        "device": device,
        "device_name": (
            torch.cuda.get_device_name(torch.device(device))
            if device.startswith("cuda") else None),
        "dtype": args.dtype,
        "acceptance_length": total_tokens / total_calls,
        "acceptance_convention": (
            "generated_tokens / verifier_calls; includes the bonus token"),
        "decode_equal": (all(item["decode_equal"] for item in results)
                         if greedy else None),
        "decode_equal_applicable": greedy,
        "repeat_stable": all(item["repeat_stable"] for item in results),
        "results": results,
    }
    try:
        serialized = json.dumps(report, indent=2, sort_keys=True, allow_nan=False)
    except ValueError as exc:
        raise ValueError("Acceptance report contains non-finite metrics") from exc
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(serialized + "\n")
    print(serialized)


if __name__ == "__main__":
    main()
