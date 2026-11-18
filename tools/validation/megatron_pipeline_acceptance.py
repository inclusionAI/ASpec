# Copyright 2026 Ant Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Bounded mixed-backend train/resume/export through the normal ASpec launcher.

Reuses ASpec's existing recipe audits. Unlike the bundled two-GPU Quickstart,
this validation entry keeps explicit TP, DP and gradient-accumulation settings.
Input capacity, future bitwise replay, quality and throughput are separate gates.
"""

import argparse
import json
import os
import sys
from pathlib import Path

from omegaconf import OmegaConf

from aspec.commands.quickstart import build_profile
from aspec.commands.recipes import audit_run, digest, execute, export_asset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--train-data", required=True, type=Path)
    parser.add_argument("--eval-data", type=Path)
    parser.add_argument("--target-python", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--target-backend", choices=("sglang", "megatron"), default="megatron")
    parser.add_argument("--draft-backend", choices=("fsdp", "megatron"), default="megatron")
    parser.add_argument("--target-tp", type=int, choices=(1, 2), default=1)
    parser.add_argument("--draft-tp", type=int, choices=(1, 2), default=1)
    parser.add_argument("--dp", type=int, default=1)
    parser.add_argument("--accumulation", type=int, default=2)
    parser.add_argument("--sequence-length", type=int, default=4096)
    parser.add_argument("--anchors", type=int, default=512)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--attention-backend", choices=("fa4", "cudnn_flex_attention"),
                        default="cudnn_flex_attention")
    parser.add_argument("--target-address", default="http://127.0.0.1:30000")
    args = parser.parse_args()
    if min(args.dp, args.accumulation, args.sequence_length, args.anchors) < 1:
        parser.error("DP, accumulation, sequence length and anchors must be positive")
    if args.steps < 2 or args.steps % 2:
        parser.error("Use an even step count >=2")
    if args.target_backend == "sglang" and args.target_tp != 1:
        parser.error("The retained SGLang Target path requires TP1")
    if args.draft_backend == "fsdp" and args.draft_tp != 1:
        parser.error("Draft TP requires the Megatron backend")
    if not args.train_data.is_file() or not args.target_python.is_absolute() or not args.target_python.is_file():
        parser.error("Provide existing data and an absolute Target interpreter path")
    if args.eval_data is not None and not args.eval_data.is_file():
        parser.error("Evaluation data must exist")
    world = args.draft_tp * args.dp
    budget = world + args.target_tp
    devices = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
    if len(devices) != budget or any(not item.strip() for item in devices) or len(set(devices)) != budget:
        parser.error(f"Select exactly {budget} distinct Draft+Target GPUs")
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    config = build_profile("qwen3-8b", args.model)
    config = OmegaConf.merge(config, OmegaConf.create({
        "dataset": {"train_data_path": str(args.train_data.resolve()), "chat_template": None,
                    "eval_data_path": str(args.eval_data.resolve()) if args.eval_data else None,
                    "eval_interval": args.steps // 2},
        "model": {"target_model_backend": args.target_backend},
        "training": {
            "train_backend": args.draft_backend, "draft_tensor_parallel_size": args.draft_tp,
            "training_num_gpus_per_node": world, "draft_accumulation_steps": args.accumulation,
            "attention_backend": args.attention_backend, "dflash_packing": False,
            "gradient_checkpointing": args.gradient_checkpointing,
            "max_seq_length": args.sequence_length, "dspark_num_anchors": args.anchors,
            "num_epochs": 100, "lr_total_steps": args.steps,
            "save_interval": args.steps // 2, "save_final_checkpoint": True,
        },
        "inference": {
            "inference_engine_type": "megatron" if args.target_backend == "megatron" else "sgl",
            "inference_num_gpus": args.target_tp, "inference_num_gpus_per_engine": args.target_tp,
            "inference_num_gpus_per_node": budget,
            "megatron": {"tp_size": args.target_tp},
            "target_service": {"mode": "managed-local", "address": args.target_address,
                               "python_executable": str(args.target_python)},
        },
        "mooncake": {"max_seq_len": args.sequence_length,
                     "destination_gpu_ring_slots": 3 if args.draft_backend == "megatron" else 1,
                     "source_gpu_ring_slots": max(12, 2 * args.dp)},
    }))
    profile = root / "profile.yaml"
    OmegaConf.save(config, profile)
    result = {"status": "running", "scope": "functional train/resume/native export",
              "source_input": {"path": str(args.train_data.resolve()), "sha256": digest(args.train_data)},
              "world_size": world, "dp": args.dp, "draft_tp": args.draft_tp,
              "target_tp": args.target_tp, "accumulation": args.accumulation}
    result["gradient_checkpointing"] = args.gradient_checkpointing
    if args.eval_data:
        result["eval_input"] = {"path": str(args.eval_data.resolve()), "sha256": digest(args.eval_data)}
    try:
        doctor = [sys.executable, "-m", "aspec.commands.doctor", "--gpu"]
        if args.draft_backend == "megatron":
            doctor += ["--backend", "megatron"]
        execute(doctor, root / "doctor.log")
        execute([sys.executable, "-m", "aspec.train_entry", "--config", profile, "--dry-run"],
                root / "plan.log")
        for label, limit in (("first", args.steps // 2), ("resumed", args.steps)):
            run = root / label
            overrides = [f"training.num_train_steps={limit}", f"cache_dir={root / 'cache'}"]
            if label == "resumed":
                overrides.append(f"training.load_path={root / 'first/checkpoints'}")
            execute([sys.executable, "-m", "aspec.commands._recipe_smoke", "--config", profile,
                     "--output", run, "--gpus", budget, *overrides], root / f"{label}.log")
            audit_run(run, limit, label == "resumed", require_eval=bool(args.eval_data), expected_ranks=world)
        export_asset(root / "resumed/checkpoints" / f"iter_{args.steps + 1:07d}", root / "asset",
                     log_prefix=root / "asset")
        result["status"] = "passed"
    except BaseException as error:
        result.update(status="failed", error=str(error))
        raise
    finally:
        (root / "acceptance.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
