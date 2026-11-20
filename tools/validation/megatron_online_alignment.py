# Copyright 2026 Ant Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Measure the deterministic TP2 DSpark objective with a live, same-TP Target."""

import argparse
import json
import sys
from contextlib import ExitStack
from dataclasses import replace
from pathlib import Path
from statistics import mean
from unittest.mock import patch

import torch.distributed as dist

from aspec import train_entry
from aspec.commands import _recipe_smoke
from aspec.ray.train_group import RayTrainGroup
from aspec.training import trainer_actor
from tools.validation.megatron_torch_alignment import deterministic_attention_backward
from tools.validation.torch_tp_dspark import TorchTPDSparkTrainer


class OnlineTorchTPTrainer(TorchTPDSparkTrainer):
    def _setup_device_mesh(self):
        super()._setup_device_mesh()
        for replica in range(self.dp_size):
            ranks = list(range(replica * self.tp_size, (replica + 1) * self.tp_size))
            groups = [dist.new_group(ranks, backend=backend) for backend in ("nccl", "gloo", "nccl", "gloo")]
            if replica == self.dp_rank:
                (self.input_tp_group, self.input_tp_control_group,
                 self.eval_input_tp_group, self.eval_input_tp_control_group) = groups


class ObjectiveActor(trainer_actor.TrainerActor):
    def init(self, args, *rest, **kwargs):
        with ExitStack() as stack:
            if args.train_backend == "torch-tp-validation":
                stack.enter_context(patch.object(trainer_actor, "_trainer_class_for_config",
                                                 return_value=OnlineTorchTPTrainer))
            result = super().init(args, *rest, **kwargs)
        trainer = self._trainer
        if trainer.tp_size != 2 or trainer.execution_backend != args.train_backend:
            raise ValueError("Online comparison requires the selected TP2 trainer")
        if args.train_backend == "torch-tp-validation" and "megatron.core" in sys.modules:
            raise RuntimeError("Torch TP validation worker imported Megatron Core")
        self.objectives = []
        forward = trainer._forward

        def observe(*inputs, **options):
            outputs = forward(*inputs, **options)
            if trainer.model.training and trainer.tp_rank == 0:
                numerator, denominator = outputs[-1]
                self.objectives.append([trainer.global_step + 1, numerator.item(), denominator.item()])
            return outputs

        trainer._forward = observe
        return result

    def train_from_queue(self, step, num_batches):
        with deterministic_attention_backward() as calls:
            result = super().train_from_queue(step, num_batches)
        if not calls["backward"]:
            raise RuntimeError("Missing deterministic cuDNN attention backward on this rank")
        return result

    def save_model(self, step, force_sync=False):
        super().save_model(step, force_sync)
        name = f"recipe-objective-dp{self._trainer.dp_rank}.json"
        files = {name: json.dumps(self.objectives)} if self._trainer.tp_rank == 0 else {}
        return {"output_dir": self.args.output_dir, "files": files}


def torch_validation_plan(plan):
    policy = plan.policy
    if (policy.backends["training"], policy.backends["target_features"]) != ("megatron", "sglang"):
        raise ValueError("Torch validation needs a TP2 scheduling plan and a live SGLang Target")
    if any(policy.parallelism[role]["tensor_parallel"] != 2 for role in ("training", "inference")):
        raise ValueError("Both Target and Draft must use TP2")
    training = {**policy.parallelism["training"], "fsdp_strategy": "REPLICATE"}
    training.pop("core_grad_reduce_in_fp32", None)
    policy = replace(policy, backends={**policy.backends, "training": "torch-tp-validation"},
                     parallelism={**policy.parallelism, "training": training})
    return replace(plan, policy=policy, _runtime_values={**plan._runtime_values,
                                                        "train_backend": "torch-tp-validation"})


def replica_queues(group, queues):
    if len(group._actor_handlers) != 2 * len(queues):
        raise ValueError("Expected one input queue per TP2 replica")
    return [queue for queue in queues for _ in range(2)]


def objectives(root, dp, steps):
    windows = [[] for _ in range(steps)]
    for rank in range(dp):
        rows = json.loads((root / f"recipe-objective-dp{rank}.json").read_text())
        expected = [step for step in range(1, steps + 1) for _ in range(2 // dp)]
        if [row[0] for row in rows] != expected:
            raise ValueError(f"Replica {rank} did not observe the expected optimizer windows")
        for step, numerator, denominator in rows:
            windows[step - 1].append((numerator, denominator))
    return [sum(n for n, _ in window) / max(sum(d for _, d in window), 1e-6) for window in windows]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--draft", choices=("torch", "megatron"), required=True)
    parser.add_argument("--dp", type=int, choices=(1, 2), required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--train-data", type=Path, required=True)
    parser.add_argument("--eval-data", type=Path, required=True)
    parser.add_argument("--target-python", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=100)
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("steps must be positive")
    root = args.output.resolve()
    target = "sglang" if args.draft == "torch" else "megatron"
    config = Path(__file__).resolve().parents[2] / "configs/recipes/quickstart_qwen3_8b.yaml"
    budget = args.dp * 2 + 2
    argv = [sys.argv[0], "--config", str(config), "--gpus", str(budget), "--output", str(root),
            f"model.target_model_path={args.model.resolve()}", f"model.target_model_backend={target}",
            f"dataset.train_data_path={args.train_data.resolve()}", "dataset.chat_template=null",
            f"dataset.eval_data_path={args.eval_data.resolve()}", f"dataset.eval_interval={args.steps}",
            f"cache_dir={root / 'cache'}", "training.train_backend=megatron", "training.megatron_grad_dtype=bf16",
            "training.draft_tensor_parallel_size=2", f"training.training_num_gpus_per_node={args.dp * 2}",
            f"training.draft_accumulation_steps={2 // args.dp}", "training.num_epochs=100",
            "training.max_seq_length=4096", "training.dspark_num_anchors=512", "training.dflash_packing=false",
            "training.gradient_checkpointing=false", f"training.num_train_steps={args.steps}",
            f"training.save_interval={args.steps}", f"training.lr_total_steps={args.steps}",
            "inference.target_service.mode=managed-local", "inference.target_service.address=http://127.0.0.1:30000",
            f"inference.target_service.python_executable={args.target_python.absolute()}",
            f"inference.inference_engine_type={'sgl' if target == 'sglang' else target}",
            f"inference.target_prefill_memory_optimized={'false' if target == 'sglang' else 'true'}",
            "inference.inference_num_gpus=2", "inference.inference_num_gpus_per_engine=2",
            f"inference.inference_num_gpus_per_node={budget}", "inference.sglang.tp_size=2",
            "inference.megatron.tp_size=2", "mooncake.max_seq_len=4096", "mooncake.destination_gpu_ring_slots=3"]
    with ExitStack() as stack:
        stack.enter_context(patch.object(sys, "argv", argv))
        if args.draft == "torch":
            compile_plan = train_entry.compile_execution_plan
            stack.enter_context(patch.object(train_entry, "compile_execution_plan",
                lambda *a, **kw: torch_validation_plan(compile_plan(*a, **kw))))
            stack.enter_context(patch.object(RayTrainGroup, "_expand_queues_for_usp", replica_queues))
        _recipe_smoke.main(trainer_class=ObjectiveActor)
    values = objectives(root, args.dp, args.steps)
    (root / "objectives.json").write_text(json.dumps({"steps": values, "mean": mean(values)}, allow_nan=False))


if __name__ == "__main__":
    main()
