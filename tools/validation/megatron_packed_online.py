# Copyright 2026 Ant Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Compare packed DSpark training with a live SGLang Target on matching CP topologies."""

import argparse
import json
import os
import sys
from pathlib import Path
from statistics import mean
from unittest.mock import patch

from aspec.commands import _recipe_smoke
from aspec.training.trainer_actor import TrainerActor
from tools.validation.megatron_torch_alignment import (
    deterministic_attention_backward,
    tensor_record,
)


class PackedObjectiveActor(TrainerActor):
    def init(self, args, *values, **kwargs):
        import torch

        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        result = super().init(args, *values, **kwargs)
        trainer = self._trainer
        if (
            trainer.execution_backend != args.train_backend
            or trainer.dp_size != 1
            or trainer.cp_size != args.draft_context_parallel_size
        ):
            raise RuntimeError("Unexpected packed Draft topology or backend")
        self.rows = []
        self.backward_calls = 0
        forward = trainer._forward

        def observe(batch, **options):
            outputs = forward(batch, **options)
            if trainer.model.training:
                lengths = batch["packed_document_lengths"]
                features = ["input_ids", "loss_mask", "hidden_states", "last_hidden_states"]
                if trainer.cp_size > 1:
                    features.append("packed_context_positions")
                numerator, denominator = outputs[-1]
                self.rows.append(
                    {
                        "step": trainer.global_step + 1,
                        "documents": lengths,
                        "features": {k: tensor_record(batch[k])["sha256"] for k in features},
                        "anchor_seed": options["anchor_seed"],
                        "numerator": numerator.item(),
                        "denominator": denominator.item(),
                    }
                )
            return outputs

        trainer._forward = observe
        return result

    def train_from_queue(self, step, num_batches):
        with deterministic_attention_backward() as calls:
            result = super().train_from_queue(step, num_batches)
        trainer = self._trainer
        layers = trainer.draft_model.last_attention_observations
        if (
            not calls["backward"]
            or not layers
            or not all(
                layer["qkv_format"] == "thd"
                and layer["backward_observed"]
                and layer["context_parallel_rank"] == trainer.cp_rank
                and layer["context_parallel_size"] == trainer.cp_size
                for layer in layers
            )
        ):
            raise RuntimeError("Missing deterministic THD backward on the expected context rank")
        self.backward_calls += calls["backward"]
        return result

    def save_model(self, step, force_sync=False):
        super().save_model(step, force_sync)
        name = f"recipe-packed-cp{self._trainer.cp_rank}.json"
        record = {"rows": self.rows, "backward_calls": self.backward_calls}
        return {
            "output_dir": self.args.output_dir,
            "files": {name: json.dumps(record, allow_nan=False)},
        }


def collect(root, cp, steps, reference=None):
    expected = [step for step in range(1, steps + 1) for _ in range(2)]
    numerators = [0.0] * steps
    denominators = [0.0] * steps
    first_rows = None
    for rank in range(cp):
        name = f"recipe-packed-cp{rank}.json"
        rows = json.loads((root / name).read_text())["rows"]
        if [r["step"] for r in rows] != expected or not any(len(r["documents"]) > 1 for r in rows):
            raise ValueError("Incomplete optimizer windows or missing multi-document packing")
        row_ids = [(r["documents"], r["features"]["input_ids"]) for r in rows]
        if first_rows is not None and first_rows != row_ids:
            raise ValueError("Context ranks consumed different packed rows")
        first_rows = row_ids
        if reference:
            baseline = json.loads((reference / name).read_text())["rows"]
            for actual, ref in zip(rows, baseline, strict=True):
                if any(
                    actual[k] != ref[k]
                    for k in ("features", "step", "documents", "anchor_seed", "denominator")
                ):
                    raise ValueError(
                        f"Different online input or supervision on CP rank {rank}, step {actual['step']}"
                    )
        for i, row in enumerate(rows):
            numerators[i // 2] += row["numerator"]
            denominators[i // 2] += row["denominator"]
    losses = [n / d for n, d in zip(numerators, denominators, strict=True)]
    result = {"steps": losses, "mean": mean(losses)}
    if reference:
        baseline = json.loads((reference / "objectives.json").read_text())["steps"]
        result["rel_diff"] = [
            (a - b) / max(abs(b), 1e-12) for a, b in zip(losses, baseline, strict=True)
        ]
    return result


def check_online_run(root, cp, steps):
    evidence = json.loads((root / "evidence.json").read_text())
    if evidence["state"] != "completed":
        raise ValueError("Online training did not complete")
    observations = evidence["observations"]
    lifecycle = observations["cleanup"]["work_lifecycle"]
    transport = observations["training"]["feature_contract"]["transport"]
    reads = lifecycle["committed_work"] * cp
    if (
        lifecycle["state"] != "validated"
        or lifecycle["optimizer_steps_committed"] != steps
        or transport["state"] != "validated"
        or transport["protocol"] != "nvlink_intra"
        or transport["consumer_endpoints"] != cp
        or reads == 0
        or transport["destination_reads"] != reads
        or transport["destination_releases"] != reads
        or transport["transfer_failures"] != 0
    ):
        raise ValueError("Incomplete online training or NVLink feature transfer")
    checkpoints = root / "checkpoints"
    iteration = int((checkpoints / "latest_checkpointed_iteration.txt").read_text())
    marker = json.loads((checkpoints / f"iter_{iteration:07d}" / "complete.json").read_text())
    if (
        marker["optimizer_step"] != steps
        or marker["committed_work_count"] != lifecycle["committed_work"]
    ):
        raise ValueError("Final checkpoint does not cover all committed training work")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--draft", choices=("torch", "megatron"), required=True)
    parser.add_argument("--cp", type=int, choices=(1, 2), required=True)
    for name in ("model", "initial", "train-data", "target-python", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--port", type=int, default=30000)
    parser.add_argument("--reference", type=Path)
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("steps must be positive")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    root = args.output.resolve()
    config = Path(__file__).resolve().parents[2] / "configs/recipes/quickstart_qwen3_8b.yaml"
    backend = "fsdp" if args.draft == "torch" else "megatron"
    budget = args.cp + 1
    argv = [
        sys.argv[0],
        "--config",
        str(config),
        "--output",
        str(root),
        "--gpus",
        str(budget),
        f"model.target_model_path={args.model.resolve()}",
        f"model.initial_draft_model_path={args.initial.resolve()}",
        f"dataset.train_data_path={args.train_data.resolve()}",
        "dataset.chat_template=null",
        f"cache_dir={root / 'cache'}",
        f"training.train_backend={backend}",
        "training.dflash_packing=true",
        "training.anchor_sampling_scope=sample",
        "training.deterministic_anchor_sampling=true",
        "training.dspark_num_anchors=32",
        "training.draft_accumulation_steps=2",
        "training.packing_min_samples_per_dispatch=2",
        f"training.training_num_gpus_per_node={args.cp}",
        f"training.draft_context_parallel_size={args.cp}",
        f"training.num_train_steps={args.steps}",
        f"training.lr_total_steps={args.steps}",
        f"training.save_interval={args.steps}",
        "training.save_per_epoch=false",
        f"training.num_epochs={args.steps}",
        "inference.target_service.mode=managed-local",
        f"inference.target_service.address=http://127.0.0.1:{args.port}",
        f"inference.target_service.python_executable={args.target_python.absolute()}",
        "inference.inference_num_gpus=1",
        f"inference.inference_num_gpus_per_node={budget}",
        "inference.max_sample_pool_size=2",
        "inference.inference_fetch_batch=2",
        "inference.inference_buffer_threshold=2",
        f"mooncake.source_gpu_ring_slots={2 * args.cp}",
    ]
    if backend == "megatron":
        argv.append("training.megatron_grad_dtype=bf16")
    with patch.object(sys, "argv", argv):
        _recipe_smoke.main(trainer_class=PackedObjectiveActor)
    check_online_run(root, args.cp, args.steps)
    result = collect(root, args.cp, args.steps, args.reference)
    (root / "objectives.json").write_text(json.dumps(result, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
