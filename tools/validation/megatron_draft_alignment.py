# Copyright 2026 Ant Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Fixed-feature DSpark TP/DP loss, gradient, update and checkpoint checks.

Uses the full Qwen3 DSpark geometry and real frozen embedding/head bindings.
Synthetic features isolate execution correctness; this is neither a quality
evaluation nor a throughput benchmark. The reference keeps ordinary ASpec
projections and uses an independent FP32-master PyTorch AdamW update.
"""

import argparse
import copy
import json
import os
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
import torch.nn.functional as F
from torch.distributed.tensor import DTensor

from aspec.config import load_config
from aspec.models.draft.auto import AutoDraftModelConfig
from aspec.models.draft.dspark import DSparkDraftModel
from aspec.models.megatron_draft import checkpoint_tensor, model_checkpoint_state
from aspec.planning import compile_execution_plan
from aspec.training.checkpoint import LRSchedulerState, ModelState, OptimizerState
from aspec.training.dspark_trainer import DSparkTrainer
from aspec.training.lr_scheduler import LRSchedulerWithWarmup
from aspec.training.megatron_trainer import MegatronDSparkTrainer
from aspec.utils.distributed import init_direct_cp_groups, init_gloo_group


def fixture(config, *, length, anchors, step, micro, replica):
    generator = torch.Generator().manual_seed(7100 + step * 101 + micro * 7 + replica * 31)
    hidden, vocab = config.hidden_size, config.vocab_size
    ids = torch.randint(vocab, (1, length), generator=generator, dtype=torch.int64)
    features = torch.randn(1, length, config.num_target_layers * hidden,
                           generator=generator, dtype=torch.bfloat16)
    teacher = torch.randn(1, length, hidden, generator=generator, dtype=torch.bfloat16)
    mask = torch.ones(1, length)
    prefix, tail = 17 + 11 * micro, 7 * (micro + 1)
    mask[:, :prefix] = 0
    mask[:, -tail:] = 0
    positions = torch.linspace(prefix + 1, length - tail - 9, anchors).long().unsqueeze(0)
    keep = torch.ones_like(positions, dtype=torch.bool)
    if micro:
        keep[:, -max(1, anchors // 4):] = False
    positions = torch.where(keep, positions, torch.zeros_like(positions))
    return {"input_ids": ids.cuda(), "hidden_states_list": features.cuda(),
            "last_hidden_states": teacher.cuda(), "loss_mask": mask.cuda(),
            "injected_anchors": positions.cuda(),
            "injected_keep_mask": keep.cuda()}


def reference_tp_projections(draft, size):
    """Simulate TP GEMMs/reduction with ordinary single-process PyTorch.

    BF16 row partials round before the TP sum; an unpartitioned GEMM rounds
    once. This independent oracle preserves those boundaries without using
    Core TP operators or distributed communication. The unpartitioned reference
    remains separately available; neither reference changes the native graph.
    """
    class Column(torch.autograd.Function):
        @staticmethod
        def forward(ctx, value, weight):
            ctx.save_for_backward(value, weight)
            return torch.cat([F.linear(value, part.contiguous())
                              for part in weight.chunk(size, dim=0)], dim=-1)

        @staticmethod
        def backward(ctx, gradient):
            value, weight = ctx.saved_tensors
            inputs = value.reshape(-1, value.shape[-1])
            input_gradient, weight_gradients = None, []
            for grad, part in zip(gradient.chunk(size, dim=-1), weight.chunk(size, dim=0), strict=True):
                grad = grad.contiguous().reshape(-1, grad.shape[-1])
                partial = grad @ part.contiguous()
                input_gradient = partial if input_gradient is None else input_gradient + partial
                weight_gradients.append(grad.T @ inputs)
            # Group the BF16 dX sum inside each projection, before gradients
            # from Q/K/V or other consumers meet at their common input.
            return input_gradient.reshape_as(value), torch.cat(weight_gradients, dim=0)

    class Projection(torch.nn.Module):
        def __init__(self, original, axis):
            super().__init__()
            self.weight, self.axis = original.weight, axis

        def forward(self, value):
            weights = self.weight.chunk(size, dim=self.axis)
            if self.axis == 0:
                return Column.apply(value, self.weight)
            partials = [F.linear(part.contiguous(), weight.contiguous())
                        for part, weight in zip(value.chunk(size, dim=-1), weights, strict=True)]
            result = partials[0]
            for partial in partials[1:]:
                result = result + partial
            return result

    for layer in draft.layers:
        for owner, names, axis in (
            (layer.self_attn, ("q_proj", "k_proj", "v_proj"), 0),
            (layer.self_attn, ("o_proj",), 1),
            (layer.mlp, ("gate_proj", "up_proj"), 0),
            (layer.mlp, ("down_proj",), 1),
        ):
            for name in names:
                setattr(owner, name, Projection(getattr(owner, name), axis))


class ReferenceOptimizer:
    def __init__(self, model, args):
        self.parameters = [p for p in model.parameters() if p.requires_grad]
        self.masters = [p.detach().float().clone().requires_grad_(True) for p in self.parameters]
        self.gradients = [torch.zeros_like(master) for master in self.masters]
        self.gradient_by_parameter = dict(zip(map(id, self.parameters), self.gradients, strict=True))
        self.optimizer = torch.optim.AdamW(self.masters, lr=args.learning_rate,
                                           weight_decay=args.weight_decay, fused=True)
        self.scheduler = LRSchedulerWithWarmup(
            self.optimizer, max_lr=args.learning_rate, min_lr=args.min_lr,
            total_steps=args.lr_total_steps,
            warmup_steps=int(args.warmup_ratio * args.lr_total_steps), decay_style=args.lr_decay_style,
        )

    def accumulate(self):
        # Match Core DDP's FP32 accumulation, independently of its hooks and
        # buffers. Accumulating into BF16 leaf .grad changes the reference.
        with torch.no_grad():
            for parameter, gradient in zip(self.parameters, self.gradients, strict=True):
                gradient.add_(parameter.grad)
                parameter.grad = None

    def step(self, max_norm, *, copy_model=True):
        for gradient, master in zip(self.gradients, self.masters, strict=True):
            master.grad = gradient
        norm = torch.nn.utils.clip_grad_norm_(self.masters, max_norm)
        if not torch.isfinite(norm).item():
            raise RuntimeError("Non-finite independent reference gradient")
        self.optimizer.step()
        self.scheduler.step()
        if copy_model:
            with torch.no_grad():
                torch._foreach_copy_(self.parameters, self.masters)
        self.optimizer.zero_grad(set_to_none=True)
        for parameter in self.parameters:
            parameter.grad = None
        torch._foreach_zero_(self.gradients)
        return norm


def logical_value(tensor, parameter, mesh):
    value = checkpoint_tensor(tensor.detach(), parameter, mesh)
    return value.full_tensor() if isinstance(value, DTensor) else value


def vector_error(pairs):
    sums = torch.zeros(2, dtype=torch.float64, device="cuda")
    for value, expected in pairs:
        delta = value.float() - expected.float()
        sums[0] += delta.double().square().sum()
        sums[1] += expected.double().square().sum()
    return (sums[0] / sums[1].clamp_min(1e-30)).sqrt().item()


def cpu_snapshot(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: cpu_snapshot(item) for key, item in value.items()}
    if isinstance(value, list):
        return [cpu_snapshot(item) for item in value]
    return copy.deepcopy(value)


def state_mismatches(current, expected, prefix=""):
    if isinstance(expected, torch.Tensor):
        return [] if torch.equal(current.detach().cpu(), expected) else [prefix]
    if isinstance(expected, dict):
        if set(current) != set(expected):
            return [prefix + ".keys"]
        return [name for key in expected for name in state_mismatches(
            current[key], expected[key], f"{prefix}.{key}")]
    if isinstance(expected, list):
        if len(current) != len(expected):
            return [prefix + ".length"]
        return [name for index, item in enumerate(expected) for name in state_mismatches(
            current[index], item, f"{prefix}.{index}")]
    return [] if current == expected else [prefix]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tp", type=int, default=2)
    parser.add_argument("--sequence-length", type=int, default=128)
    parser.add_argument("--anchors", type=int, default=16)
    parser.add_argument("--attention-backend", choices=("fa4", "cudnn_flex_attention"), default="fa4")
    parser.add_argument("--reference-tp-projections", action="store_true")
    args = parser.parse_args()
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl")
    init_gloo_group()
    init_direct_cp_groups(1)
    world, rank = dist.get_world_size(), dist.get_rank()
    root = Path(__file__).resolve().parents[2]
    os.environ.setdefault("ASPEC_TRAIN_DATA", str(root / "examples/data/sample_conversations.jsonl"))
    os.environ.setdefault("ASPEC_EVAL_DATA", str(root / "examples/data/sample_conversations.jsonl"))
    output = args.output.resolve()
    if rank == 0:
        output.mkdir(parents=True, exist_ok=False)
    dist.barrier()
    config = load_config(str(root / "configs/recipes/qwen3_8b_dspark.yaml"), cli_args=[
        f"model.target_model_path={args.model}", "model.target_model_backend=megatron",
        f"dataset.train_data_path={root / 'examples/data/sample_conversations.jsonl'}",
        f"dataset.eval_data_path={root / 'examples/data/sample_conversations.jsonl'}",
        "training.train_backend=megatron", f"training.draft_tensor_parallel_size={args.tp}",
        f"training.attention_backend={args.attention_backend}",
        f"training.training_num_gpus_per_node={world}", "training.dflash_packing=false",
        "training.draft_accumulation_steps=2", "training.deterministic_anchor_sampling=true",
        f"training.max_seq_length={args.sequence_length}", f"training.dspark_num_anchors={args.anchors}",
        "training.lr_total_steps=4", "training.num_train_steps=4",
        "mooncake.destination_gpu_ring_slots=4",
        "inference.inference_engine_type=megatron", "inference.target_prefill_memory_optimized=true",
        "inference.inference_num_gpus=1", "inference.inference_num_gpus_per_engine=1",
        "inference.megatron.tp_size=1", f"output_dir={output / 'run'}",
    ])
    plan = compile_execution_plan(config)
    runtime = plan.to_runtime_args()
    runtime.rank, runtime.world_size = rank, world
    draft_config = AutoDraftModelConfig.from_file(runtime.draft_model_config)
    trainer = MegatronDSparkTrainer(runtime)
    trainer.init_model(draft_config, args.model)
    initial = model_checkpoint_state(trainer.draft_model, trainer.tp_mesh)
    initial = {name: (value.full_tensor() if isinstance(value, DTensor) else value).cpu()
               for name, value in initial.items()}
    reference = reference_optimizer = update_oracle = None
    if rank == 0:
        with torch.device("cpu"):
            reference_draft = DSparkDraftModel(copy.deepcopy(draft_config)).to(torch.bfloat16)
        reference_draft.load_state_dict(initial, strict=True)
        reference_draft.freeze_embedding()
        if args.reference_tp_projections:
            reference_tp_projections(reference_draft, args.tp)
        reference = DSparkTrainer._build_training_wrapper(trainer, reference_draft.cuda())
        reference_optimizer = ReferenceOptimizer(reference_draft, runtime)
        update_oracle = ReferenceOptimizer(reference_draft, runtime)
    del initial
    accumulation = runtime.draft_accumulation_steps
    report = []

    def train_window(step, compare_reference, *, capture_gradients=False):
        native_num = native_den = None
        reference_num = reference_den = None
        reference_den_by_replica = {}
        for micro in range(accumulation):
            batch = fixture(draft_config, length=args.sequence_length, anchors=args.anchors,
                            step=step, micro=micro, replica=trainer.dp_rank)
            with trainer.model.no_sync() if micro < accumulation - 1 else nullcontext():
                values = trainer.model(**batch, lm_head_weight=trainer.target_lm_head_weight)
                num, den = values[-1]
                (num / accumulation).backward()
            native_num = num.detach().clone() if native_num is None else native_num + num.detach()
            native_den = den.clone() if native_den is None else native_den + den
            if compare_reference and rank == 0:
                for replica in range(trainer.dp_size):
                    ref_batch = fixture(draft_config, length=args.sequence_length, anchors=args.anchors,
                                        step=step, micro=micro, replica=replica)
                    ref_values = reference(**ref_batch, lm_head_weight=trainer.target_lm_head_weight)
                    ref_num, ref_den = ref_values[-1]
                    (ref_num / accumulation).backward()
                    reference_optimizer.accumulate()
                    reference_num = (ref_num.detach().clone() if reference_num is None
                                     else reference_num + ref_num.detach())
                    previous = reference_den_by_replica.get(replica)
                    reference_den_by_replica[replica] = ref_den.clone() if previous is None else previous + ref_den
        denominator = trainer._normalize_accumulated_gradients(native_den, accumulation)
        observed_gradients = (cpu_snapshot([p.main_grad for p in trainer.optimizer.model_params])
                              if capture_gradients else None)
        dist.all_reduce(native_num, group=trainer.dp_group)
        if compare_reference and rank == 0:
            reference_den = torch.stack(list(reference_den_by_replica.values())).sum(0)
            with torch.no_grad():
                for gradient in reference_optimizer.gradients:
                    gradient.mul_(accumulation / reference_den)
        gradient_pairs, before, expected_before = [], [], []
        if compare_reference and trainer.dp_rank == 0:
            ref_parameters = dict(reference.draft_model.named_parameters()) if rank == 0 else {}
            for name, parameter in trainer.draft_model.named_parameters():
                if not parameter.requires_grad:
                    continue
                gradient = logical_value(parameter.main_grad, parameter, trainer.tp_mesh)
                if rank == 0:
                    reference_parameter = ref_parameters[name]
                    gradient_pairs.append((gradient, reference_optimizer.gradient_by_parameter[
                        id(reference_parameter)]))
                    update_oracle.gradient_by_parameter[id(reference_parameter)].copy_(gradient)
            for index, parameter in enumerate(trainer.optimizer.model_params):
                value = logical_value(trainer.optimizer.fp32_params[index], parameter, trainer.tp_mesh)
                if rank == 0:
                    before.append(value.clone())
            if rank == 0:
                expected_before = [p.detach().clone() for p in reference_optimizer.masters]
                oracle_before = [p.detach().clone() for p in update_oracle.masters]
        measurements = {}
        if compare_reference and rank == 0:
            measurements = {
                "loss_relative_error": abs((native_num / denominator / (reference_num / reference_den) - 1).item()),
                "gradient_relative_l2": vector_error(gradient_pairs),
                "denominator": denominator.item(), "reference_denominator": reference_den.item(),
            }
            reference_optimizer.step(runtime.max_grad_norm)
            oracle_norm = update_oracle.step(runtime.max_grad_norm, copy_model=False)
        native_norm = trainer.optimizer.step()
        if compare_reference and trainer.dp_rank == 0:
            updates, same_gradient_updates = [], []
            for index, parameter in enumerate(trainer.optimizer.model_params):
                value = logical_value(trainer.optimizer.fp32_params[index], parameter, trainer.tp_mesh)
                if rank == 0:
                    updates.append((value - before[index],
                                    reference_optimizer.masters[index] - expected_before[index]))
                    same_gradient_updates.append((value - before[index],
                                                  update_oracle.masters[index] - oracle_before[index]))
            if rank == 0:
                measurements["update_relative_l2"] = vector_error(updates)
                measurements["same_gradient_adam_update_relative_l2"] = vector_error(same_gradient_updates)
                measurements["clip_norm_relative_error"] = abs((native_norm / oracle_norm - 1).item())
                names = [name for name, parameter in trainer.draft_model.named_parameters()
                         if parameter.requires_grad]
                contributions = [{"parameter": name,
                                  "squared_error": (actual - expected).double().square().sum().item(),
                                  "relative_l2": vector_error([(actual, expected)])}
                                 for name, (actual, expected) in zip(names, updates, strict=True)]
                measurements["largest_update_errors"] = sorted(
                    contributions, key=lambda item: item["squared_error"], reverse=True)[:5]
                report.append(measurements)
        dist.barrier()
        return observed_gradients

    train_window(0, True)
    checkpoint_root = output / "checkpoint"
    state = {"model_state": ModelState(trainer.model),
             "optim_state": OptimizerState(trainer.model, trainer.optimizer),
             "lr_scheduler_state": LRSchedulerState(trainer.lr_scheduler)}
    dcp.save(state, checkpoint_id=checkpoint_root)
    saved_masters = cpu_snapshot(trainer.optimizer.fp32_params)
    saved_model = cpu_snapshot({name: value for name, value in trainer.draft_model.state_dict().items()
                               if isinstance(value, torch.Tensor)})
    saved_optimizer = cpu_snapshot(trainer.optimizer.state_dict())
    saved_scheduler = cpu_snapshot(trainer.lr_scheduler.state_dict())
    rng = {"torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state(),
           "backend": trainer._checkpoint_backend_rng()}
    expected_gradients = train_window(1, True, capture_gradients=True)
    expected_masters = [p.detach().cpu().clone() for p in trainer.optimizer.fp32_params]
    expected_optimizer = copy.deepcopy(trainer.optimizer.state_dict())
    expected_scheduler = copy.deepcopy(trainer.lr_scheduler.state_dict())
    # A same-state in-memory replay independently measures the arithmetic's
    # determinism, before involving checkpoint deserialization.
    trainer.draft_model.load_state_dict(saved_model, strict=True)
    with torch.no_grad():
        for master, saved in zip(trainer.optimizer.fp32_params, saved_masters, strict=True):
            master.copy_(saved)
    trainer.optimizer.load_state_dict(saved_optimizer)
    trainer.lr_scheduler.load_state_dict(saved_scheduler)
    torch.set_rng_state(rng["torch"])
    torch.cuda.set_rng_state(rng["cuda"])
    trainer._restore_backend_rng(rng["backend"])
    memory_gradients = train_window(1, False, capture_gradients=True)
    memory_exact = all(torch.equal(p.cpu(), expected) for p, expected in zip(
        trainer.optimizer.fp32_params, expected_masters, strict=True))
    memory_gradient_error = vector_error(zip(memory_gradients, expected_gradients, strict=True))
    del memory_gradients
    state["optim_state"] = OptimizerState(trainer.model, trainer.optimizer, load_path=checkpoint_root)
    dcp.load(state, checkpoint_id=checkpoint_root)
    load_errors = {
        "masters": state_mismatches(trainer.optimizer.fp32_params, saved_masters),
        "model": state_mismatches({name: value for name, value in trainer.draft_model.state_dict().items()
                                   if isinstance(value, torch.Tensor)}, saved_model),
        "optimizer": state_mismatches(trainer.optimizer.state_dict(), saved_optimizer),
        "scheduler": state_mismatches(trainer.lr_scheduler.state_dict(), saved_scheduler),
    }
    print(json.dumps({"rank": rank, "checkpoint_load_errors": load_errors}), flush=True)
    load_exact = torch.tensor(int(not any(load_errors.values())), device="cuda")
    dist.all_reduce(load_exact, op=dist.ReduceOp.MIN)
    del saved_model, saved_masters, saved_optimizer, saved_scheduler
    torch.set_rng_state(rng["torch"])
    torch.cuda.set_rng_state(rng["cuda"])
    trainer._restore_backend_rng(rng["backend"])
    replay_gradients = train_window(1, False, capture_gradients=True)
    replay_gradient_error = vector_error(zip(replay_gradients, expected_gradients, strict=True))
    del replay_gradients, expected_gradients
    exact_masters = all(torch.equal(p.cpu(), expected) for p, expected in zip(
        trainer.optimizer.fp32_params, expected_masters, strict=True))
    exact_optimizer = all(torch.equal(value, expected_optimizer["state"][index][field])
                          for index, values in trainer.optimizer.state_dict()["state"].items()
                          for field, value in values.items())
    exact_scheduler = trainer.lr_scheduler.state_dict() == expected_scheduler
    exact = torch.tensor(int(exact_masters and exact_optimizer and exact_scheduler), device="cuda")
    dist.all_reduce(exact, op=dist.ReduceOp.MIN)
    if rank == 0:
        passed = exact.item() == 1 and load_exact.item() == 1 and all(
            row["loss_relative_error"] <= 0.01 and row["gradient_relative_l2"] <= 0.03
            and row["update_relative_l2"] <= 0.03
            and row["denominator"] == row["reference_denominator"] for row in report)
        result = {"passed": passed, "windows": report, "exact_checkpoint_replay": bool(exact.item()),
                  "exact_checkpoint_state": bool(load_exact.item()),
                  "local_load_errors": load_errors,
                  "local_replay_exact": {"masters": exact_masters, "optimizer": exact_optimizer,
                                         "scheduler": exact_scheduler},
                  "in_memory_replay_exact_masters": memory_exact,
                  "in_memory_replay_gradient_relative_l2": memory_gradient_error,
                  "checkpoint_replay_gradient_relative_l2": replay_gradient_error,
                  "tp": args.tp, "dp": trainer.dp_size, "sequence_length": args.sequence_length,
                  "anchors": args.anchors, "plan_id": plan.plan_id,
                  "reference_accumulation_dtype": "float32",
                  "reference_projection_topology": "simulated_tp" if args.reference_tp_projections else "unpartitioned",
                  "attention_backend": args.attention_backend,
                  "dspark_execution": trainer._kernel_observation(),
                  "tolerances": {"loss_relative_error": 0.01, "gradient_relative_l2": 0.03,
                                 "update_relative_l2": 0.03}}
        (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result), flush=True)
    else:
        passed = None
    verdict = [passed]
    dist.broadcast_object_list(verdict, src=0)
    dist.destroy_process_group()
    if not verdict[0]:
        raise SystemExit("DSpark Megatron execution alignment failed")


if __name__ == "__main__":
    main()
