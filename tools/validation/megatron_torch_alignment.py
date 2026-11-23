# Copyright 2026 Ant Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Compare native ASpec Torch and Megatron DSpark updates on fixed Target features.

Run each backend in its own environment. Measure the objective at every update
and compare logical gradients, parameters and optimizer state at selected steps.
A fresh process can restore a checkpoint and compare with a continuous run.
"""

import argparse
import copy
import hashlib
import importlib.util
import inspect
import json
import math
import os
import time
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import torch
import torch.distributed as dist
from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict
from torch.distributed.tensor import DTensor

import aspec


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_record(value):
    value = value.detach().cpu().contiguous()
    return {"shape": list(value.shape), "dtype": str(value.dtype),
            "sha256": hashlib.sha256(value.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()}


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def local_state(trainer):
    def record(value):
        if isinstance(value, DTensor):
            value = value.to_local()
        if isinstance(value, torch.Tensor):
            return tensor_record(value)
        if isinstance(value, dict):
            return {str(k): record(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [record(v) for v in value]
        return value

    return {"model": record(trainer.model.state_dict()),
            "masters": record(trainer.optimizer.fp32_params),
            "optimizer": record(trainer.optimizer.state_dict()),
            "scheduler": record(trainer.lr_scheduler.state_dict()),
            "torch_rng": record(torch.get_rng_state()),
            "cuda_rng": record(torch.cuda.get_rng_state()),
            "backend_rng": (record(trainer._checkpoint_backend_rng())
                            if hasattr(trainer, "_checkpoint_backend_rng") else None),
            "frozen_target_head": tensor_record(trainer.target_lm_head_weight),
            "global_step": trainer.global_step}


def logical_cpu(trainer, value, parameter):
    if value is None:
        return None
    if isinstance(value, DTensor):
        value = value.full_tensor()
    elif getattr(parameter, "_aspec_tp_axis", None) is not None:
        if hasattr(trainer, "logical_tensor"):
            value = trainer.logical_tensor(value, parameter).full_tensor()
        else:
            from aspec.models.megatron_draft import checkpoint_tensor

            value = checkpoint_tensor(value, parameter, trainer.tp_mesh).full_tensor()
    return value.detach().cpu().clone()


def named_trainable(trainer):
    result = [(name, parameter) for name, parameter in trainer.draft_model.named_parameters()
              if parameter.requires_grad]
    assert all(p is expected for (_, p), expected in zip(
        result, trainer.optimizer.model_params, strict=True))
    return result


def logical_optimizer_state(trainer):
    result = {key: {} for key in ("model", "masters", "exp_avg", "exp_avg_sq")}
    for index, (name, parameter) in enumerate(named_trainable(trainer)):
        master = trainer.optimizer.fp32_params[index]
        result["model"][name] = logical_cpu(trainer, parameter, parameter)
        result["masters"][name] = logical_cpu(trainer, master, parameter)
        state = trainer.optimizer.optimizer.state.get(master, {})
        for key in ("exp_avg", "exp_avg_sq"):
            if key in state:
                result[key][name] = logical_cpu(trainer, state[key], parameter)
    return result


def scalar_diff(actual, reference, epsilon=1e-12):
    """Reference-relative signed difference, with an explicit denominator floor."""
    actual, reference = float(actual), float(reference)
    if not math.isfinite(actual) or not math.isfinite(reference):
        raise ValueError("Nonfinite scalar in numerical comparison")
    delta = actual - reference
    return {"actual": actual, "reference": reference, "signed_difference": delta,
            "absolute_difference": abs(delta),
            "signed_rel_diff": delta / max(abs(reference), epsilon),
            "abs_rel_diff": abs(delta) / max(abs(reference), epsilon),
            "denominator_floored": abs(reference) < epsilon, "epsilon": epsilon}


def compare_vectors(actual, expected, *, device=None):
    """Full-vector reductions and explicit, near-zero-aware pointwise rel diffs.

    Norms/means/maxima/threshold counts use every element and FP64 reductions.
    Only quantiles are sampled: systematic coordinates in the logical vector,
    independent of values, with separate per-parameter samples. No training state
    or RNG is changed. The denominator floor is max(1e-12, 1e-6 * reference RMS)
    per parameter; floored coordinates are counted rather than silently removed.
    """
    if set(actual) != set(expected):
        raise ValueError("Compared trainable parameter sets differ")
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    count = sum(v.numel() for v in expected.values() if v is not None)
    stride = max(1, math.ceil(count / 262144))
    totals = torch.zeros(14, dtype=torch.float64, device=device)
    global_samples, per_parameter = [], []
    exceed_counts = torch.zeros(4, dtype=torch.int64, device=device)
    thresholds = (.01, .03, .1, 1.0)
    exact, offset = True, 0

    def ratio(numerator, denominator):
        return numerator / denominator if denominator else (0.0 if numerator == 0 else None)

    def summarize(values):
        n, e2, b2, a2, e1, b1, es, dot, floors, point_sum, maximum, reference_max, point_max, mismatches = values
        return {"numel": int(n), "relative_l2": ratio(math.sqrt(e2), math.sqrt(b2)),
                "relative_l1": ratio(e1, b1), "relative_max": ratio(maximum, reference_max),
                "rmse": math.sqrt(e2 / max(n, 1)), "mean_absolute": e1 / max(n, 1),
                "max_absolute": maximum, "signed_mean_difference": es / max(n, 1),
                "reference_rms": math.sqrt(b2 / max(n, 1)),
                "cosine": ratio(dot, math.sqrt(a2 * b2)),
                "pointwise_rel_diff_mean": point_sum / max(n, 1),
                "pointwise_rel_diff_max": point_max, "denominator_floored_count": int(floors),
                "different_elements": int(mismatches), "zero_reference": b2 == 0}

    def quantiles(values):
        if not values or not sum(v.numel() for v in values):
            return {"sample_size": 0, "p50": None, "p95": None, "p99": None}
        values = torch.cat(values).float()
        return {"sample_size": values.numel(),
                **{label: val for label, val in zip(("p50", "p95", "p99"),
                    torch.quantile(values, torch.tensor([.5, .95, .99], device=device)).tolist(), strict=True)}}

    for name, value in actual.items():
        reference = expected[name]
        if value is None or reference is None:
            if value is not None or reference is not None:
                raise ValueError(f"Missing gradient differs: {name}")
            continue
        if value.shape != reference.shape:
            raise ValueError(f"Compared tensor shapes differ: {name}")
        value, reference = value.to(device), reference.to(device)
        a, b = value.float().flatten(), reference.float().flatten()
        if not (torch.isfinite(a).all() & torch.isfinite(b).all()).item():
            raise ValueError(f"Nonfinite tensor in numerical comparison: {name}")
        ad, bd = a.double(), b.double()
        dd = delta = ad - bd
        e2, b2 = dd.square().sum(), bd.square().sum()
        floor = max(1e-12, 1e-6 * (b2 / max(b.numel(), 1)).sqrt().item())
        pointwise = delta.abs() / bd.abs().clamp_min(floor)
        exceeds = torch.stack([(pointwise > threshold).sum() for threshold in thresholds])
        exceed_counts += exceeds
        row = torch.stack((b.new_tensor(b.numel()), e2, b2, ad.square().sum(), dd.abs().sum(),
                           bd.abs().sum(), dd.sum(), (ad * bd).sum(), (b.abs() < floor).sum(),
                           pointwise.double().sum(), delta.abs().max(), b.abs().max(),
                           pointwise.max(), (a != b).sum())).double()
        totals[:10] += row[:10]
        totals[10:13] = torch.maximum(totals[10:13], row[10:13])
        totals[13] += row[13]
        values = row.tolist()
        exact = exact and value.dtype == reference.dtype and torch.equal(
            value.contiguous().reshape(-1).view(torch.uint8),
            reference.contiguous().reshape(-1).view(torch.uint8),
        )
        row_summary = summarize(values)
        per_parameter.append({"parameter": name, "squared_error": values[1],
                              "denominator_floor": floor, **row_summary,
                              "pointwise_exceed_counts": dict(zip(map(str, thresholds), exceeds.tolist(), strict=True)),
                              "pointwise_rel_diff_quantiles": quantiles([
                                  pointwise[::max(1, math.ceil(pointwise.numel() / 4096))]])})
        global_samples.append(pointwise[(-offset) % stride::stride].clone())
        offset += b.numel()
        del a, b, delta, ad, bd, dd, pointwise
    return {**summarize(totals.tolist()), "bitwise_equal": exact,
            "pointwise_exceed_counts": dict(zip(map(str, thresholds), exceed_counts.tolist(), strict=True)),
            "pointwise_rel_diff_quantiles": quantiles(global_samples),
            "quantile_method": "systematic logical coordinates, global <=262144, per-parameter <=4096",
            "denominator_floor_policy": "max(1e-12, 1e-6 * per-parameter reference RMS)",
            "per_parameter": per_parameter,
            "largest_errors": sorted(per_parameter, key=lambda row: row["squared_error"], reverse=True)[:5]}


def selected_steps(value, maximum):
    if value == "all":
        return set(range(1, maximum + 1))
    result = {int(step) for step in value.split(",") if step}
    if any(step < 1 or step > maximum for step in result):
        raise ValueError("Observation steps must be within the training trajectory")
    return result


def wait_reference(path, timeout):
    deadline = time.monotonic() + timeout
    while not path.is_file():
        if time.monotonic() >= deadline:
            raise TimeoutError(f"Reference observation unavailable: {path}")
        time.sleep(.5)


def reference_state_path(reference, step, final_step, timeout):
    if step == final_step:
        wait_reference(reference / "completed.json", timeout)
        final = reference / "final-logical.pt"
        if final.is_file():
            return final
    else:
        wait_reference(reference / f"step{step}.json", timeout)
    return reference / f"logical-state-step{step}.pt"


def source_identity(trainer, expected):
    from aspec.training import checkpoint

    root = Path(aspec.__file__).resolve().parent
    if root != expected.resolve() / "aspec":
        raise RuntimeError(f"Wrong ASpec checkout: {root}, expected {expected}")
    files = {}
    for method in (type(trainer), trainer._train_core_from_queue, trainer._train_step,
                   trainer._normalize_accumulated_gradients, type(trainer.optimizer),
                   type(trainer.optimizer).step, checkpoint.save, checkpoint.load):
        path = Path(inspect.getfile(method)).resolve()
        if path == Path(__file__).with_name("torch_tp_dspark.py").resolve():
            files["validation/torch_tp_dspark.py"] = file_sha(path)
        else:
            files[str(path.relative_to(root))] = file_sha(path)
    return {"aspec": str(root), "python": os.sys.executable, "prefix": os.sys.prefix,
            "torch": torch.__version__, "trainer": type(trainer).__name__, "files": files,
            "harness_sha256": file_sha(__file__)}


def save_native_checkpoint(trainer, runtime, committed, fixture_sha, output, dataset_size):
    from aspec.training import checkpoint

    rank = dist.get_rank()
    if rank == 0:
        checkpoint.prepare_checkpoint_save(runtime.checkpoint_dir, trainer.global_step)
    dist.barrier()
    trainer.save_model(trainer.global_step, force_sync=True)
    if rank == 0:
        checkpoint.save_checkpoint_work_state(runtime.checkpoint_dir, trainer.global_step, {
            "committed_work_ids": committed, "dataset_fingerprint": "sha256:" + fixture_sha,
            "dataset_size": dataset_size})
    dist.barrier()
    write_json(output / f"checkpoint-state-rank{rank}.json", local_state(trainer))


def fixed_feature_row(samples, indices, length, work_id, collator):
    if collator is None:
        batch = {key: value[:, :length] if isinstance(value, torch.Tensor) else copy.deepcopy(value)
                 for key, value in samples[indices[0]].items()}
        batch["_aspec_work"] = [{"execution_id": work_id}]
        return batch

    docs = []
    for index in indices:
        sample = samples[index]
        if sample.get("attention_mask") is not None and not sample["attention_mask"].all():
            raise ValueError("Packed alignment requires complete, unpadded fixture documents")
        doc = {**sample, "_aspec_work": {"execution_id": f"{work_id}/sample-{index}"}}
        if collator.cp_size > 1:
            from aspec.inference.context import context_shard_bounds

            lo, hi = context_shard_bounds(sample["input_ids"].shape[1], collator.cp_rank,
                                          collator.cp_size, alignment=collator.cp_alignment)
            for key in ("hidden_states", "last_hidden_states"):
                doc[key] = sample[key][:, lo:hi]
        docs.append(doc)
    return collator(docs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("fsdp", "megatron", "torch-tp"), required=True)
    parser.add_argument("--phase", choices=("prepare", "uninterrupted", "checkpoint-control", "split", "resume"), required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--initial", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference", type=Path, action="append", default=[])
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--cp", type=int, choices=(1, 2), default=1)
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--megatron-grad-dtype", choices=("fp32", "bf16"))
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--stop-after", type=int, help="Stop at this update without shortening the LR horizon")
    parser.add_argument("--split-step", type=int, default=2)
    parser.add_argument("--length", type=int, default=4096)
    parser.add_argument("--packed-pairs", action="store_true",
                        help="Pack adjacent pairs of complete fixture documents with per-document anchor budgets")
    parser.add_argument("--anchors", type=int, default=512)
    parser.add_argument("--tensor-steps", default="all", help="Full gradient/update observations: all or comma-separated steps")
    parser.add_argument("--state-steps", default="", help="Model/master/moment observations: comma-separated steps")
    parser.add_argument("--save-reference-tensors", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--reference-wait-seconds", type=float, default=0)
    parser.add_argument("--attention-backend", choices=("fa4", "cudnn_flex_attention"), default="cudnn_flex_attention")
    parser.add_argument("--attention-backward-deterministic", action="store_true",
                        help="Require native deterministic cuDNN-FE attention backward on every rank")
    parser.add_argument("--deterministic", action="store_true",
                        help="Require deterministic PyTorch algorithms and native cuDNN attention backward")
    args = parser.parse_args()
    if args.deterministic:
        args.attention_backward_deterministic = True
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    if args.steps < 1 or (args.phase in {"split", "checkpoint-control"}
                          and not 1 <= args.split_step <= args.steps):
        parser.error("Require positive steps and a split within the training horizon")
    if args.cp > 1 and not args.packed_pairs:
        parser.error("CP2 alignment requires packed fixture documents")
    if args.packed_pairs and args.tp != 1:
        parser.error("Packed fixed-feature alignment currently requires TP1")
    if args.backend == "fsdp" and args.tp != 1:
        parser.error("The Torch baseline uses its existing TP1 execution")
    if args.backend != "megatron" and args.megatron_grad_dtype is not None:
        parser.error("Megatron gradient precision applies only to the Megatron backend")
    if not args.attention_backward_deterministic:
        run(args)
        return
    if args.attention_backend != "cudnn_flex_attention" or args.phase == "prepare":
        parser.error("The deterministic backward check requires a cuDNN training run")

    with deterministic_attention_backward() as calls:
        run(args, attention_calls=calls)


@contextmanager
def deterministic_attention_backward():
    from cudnn.flex_attention import autograd, flex_attn_func

    backward = autograd._flex_attention_backward
    calls = {"forward": 0, "backward": 0}

    def deterministic_forward(*inputs, **options):
        options["deterministic"] = True
        calls["forward"] += 1
        return flex_attn_func(*inputs, **options)

    def checked_backward(*inputs, **options):
        if not options.get("deterministic"):
            raise RuntimeError("cuDNN attention backward did not receive deterministic=True")
        calls["backward"] += 1
        return backward(*inputs, **options)

    with (patch("cudnn.flex_attention.flex_attn_func", deterministic_forward),
          patch.object(autograd, "_flex_attention_backward", checked_backward)):
        yield calls


def run(args, *, attention_calls=None):
    from aspec.config.train_config import load_config
    from aspec.models.draft.auto import AutoDraftModelConfig
    from aspec.planning import compile_execution_plan
    from aspec.training import checkpoint
    from aspec.training.dspark_trainer import DSparkTrainer
    from aspec.utils.distributed import init_direct_cp_groups, init_gloo_group

    if args.backend == "torch-tp":
        class RejectMegatronImports:
            def find_spec(self, fullname, path=None, target=None):
                if fullname == "megatron" or fullname.startswith("megatron."):
                    raise RuntimeError("The independent Torch TP validation cannot import Megatron Core")

        os.sys.meta_path.insert(0, RejectMegatronImports())
    tensor_steps = selected_steps(args.tensor_steps, args.steps)
    state_steps = selected_steps(args.state_steps, args.steps)
    torch.set_num_threads(4)
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl", device_id=torch.device("cuda", torch.cuda.current_device()))
    init_gloo_group()
    init_direct_cp_groups(args.cp)
    rank, world = dist.get_rank(), dist.get_world_size()
    output = args.output.resolve()
    if rank == 0:
        output.mkdir(parents=True, exist_ok=False)
    dist.barrier()
    os.environ.setdefault("ASPEC_TRAIN_DATA", str(args.source / "examples/data/sample_conversations.jsonl"))
    os.environ.setdefault("ASPEC_EVAL_DATA", str(args.source / "examples/data/sample_conversations.jsonl"))
    base_backend = "fsdp" if args.backend == "torch-tp" else args.backend
    overrides = [f"model.target_model_path={args.model}",
                 f"model.draft_model_config={args.source / 'aspec/config/dspark_draft_config.json'}",
                 f"training.draft_context_parallel_size={args.cp}", f"training.train_backend={base_backend}",
                 f"training.dflash_packing={str(args.packed_pairs).lower()}",
                 f"training.training_num_gpus_per_node={world}", "training.draft_accumulation_steps=2",
                 "training.deterministic_anchor_sampling=true", f"training.max_seq_length={args.length}",
                 f"training.dspark_num_anchors={args.anchors}", f"training.num_train_steps={args.steps}",
                 f"training.lr_total_steps={args.steps}", f"training.attention_backend={args.attention_backend}",
                 "inference.target_prefill_memory_optimized=true",
                 "mooncake.source_gpu_ring_slots=12", "mooncake.destination_gpu_ring_slots=3",
                 f"output_dir={output}"]
    if args.packed_pairs:
        overrides.append("training.anchor_sampling_scope=sample")
    if args.initial:
        overrides.append(f"model.initial_draft_model_path={args.initial}")
    if args.resume:
        overrides.append(f"training.load_path={args.resume}")
    if args.backend == "megatron":
        overrides.append(f"training.draft_tensor_parallel_size={args.tp}")
        if args.megatron_grad_dtype is not None:
            overrides.append(f"training.megatron_grad_dtype={args.megatron_grad_dtype}")
    config = load_config(str(args.source / "configs/recipes/qwen3_8b_dspark.yaml"), cli_args=overrides)
    plan = compile_execution_plan(config)
    runtime = plan.to_runtime_args()
    runtime.rank, runtime.world_size = rank, world
    if args.backend == "megatron":
        from aspec.training.megatron_trainer import MegatronDSparkTrainer

        trainer = MegatronDSparkTrainer(runtime)
    elif args.backend == "torch-tp":
        # The maintained public Plan still rejects Torch Draft TP. This
        # explicitly recorded validation extension tests its native DTensor
        # arithmetic without claiming production Plan/backend support.
        runtime.draft_tensor_parallel_size = args.tp
        path = Path(__file__).with_name("torch_tp_dspark.py")
        spec = importlib.util.spec_from_file_location("aspec_torch_tp_validation", path)
        module = importlib.util.module_from_spec(spec)
        os.sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        trainer = module.TorchTPDSparkTrainer(runtime)
        write_json(output / f"torch-tp-validation-rank{rank}.json", {
            "scope": "validation-only extension of existing Torch DSpark trainer; not a production backend",
            "base_plan_backend": "fsdp", "base_plan_draft_tp": 1,
            "observed_draft_tp": trainer.tp_size, "observed_dp": trainer.dp_size,
            "tp_operators": "native PyTorch DTensor linear/redistribute/autograd",
            "adapter_sha256": file_sha(path),
        })
    else:
        trainer = DSparkTrainer(runtime)
    trainer.init_model(AutoDraftModelConfig.from_file(runtime.draft_model_config), args.model)
    identity = source_identity(trainer, args.source)
    if args.backend == "megatron":
        write_json(output / f"gradient-precision-rank{rank}.json", trainer._gradient_precision)
    if args.backend != "megatron" and any(name == "megatron" or name.startswith("megatron.") for name in os.sys.modules):
        raise RuntimeError("The baseline imported Megatron")
    write_json(output / f"origin-rank{rank}.json", identity)
    if args.phase == "prepare":
        if args.backend != "fsdp" or world != 1:
            raise ValueError("Prepare the common initialization with the actual TP1/DP1 Torch trainer")
        from safetensors.torch import save_file

        state = get_model_state_dict(trainer.draft_model, options=StateDictOptions(full_state_dict=True, cpu_offload=True))
        save_file({name: value.contiguous() for name, value in state.items()}, str(output / "model.safetensors"))
        trainer.draft_model.config.save_pretrained(output)
        write_json(output / "prepared.json", {"source": identity, "initial_sha256": file_sha(output / "model.safetensors")})
        dist.destroy_process_group()
        return

    initial_state = local_state(trainer)
    write_json(output / f"initial-state-rank{rank}.json", initial_state)
    if args.phase == "resume":
        if args.resume is None:
            raise ValueError("Resume requires a native checkpoint")
        expected = json.loads((args.resume.parent.parent / f"checkpoint-state-rank{rank}.json").read_text())
        restored = initial_state == expected
        write_json(output / f"restore-rank{rank}.json", {"exact": restored})
        if not restored:
            differences = [key for key in expected if expected[key] != initial_state[key]]
            raise RuntimeError(f"Native checkpoint state did not restore exactly on rank {rank}: {differences}")
    del initial_state
    saved_work = checkpoint.load_checkpoint_work_state(runtime) if args.resume else None
    committed = list(saved_work["committed_work_ids"]) if saved_work else []
    fixture_sha = file_sha(args.teacher)
    if args.resume and (saved_work or {}).get("dataset_fingerprint") != "sha256:" + fixture_sha:
        raise ValueError("Resume teacher fixture differs from checkpoint dataset fingerprint")
    samples = torch.load(args.teacher, map_location="cpu", weights_only=True, mmap=True)["samples"]
    manifest = {"teacher_sha256": fixture_sha, "initial_sha256": file_sha(args.initial) if args.initial else None,
                "steps": args.steps, "length": args.length, "anchors": args.anchors,
                "accumulation": 2, "attention_backend": args.attention_backend,
                "dp_size": trainer.dp_size, "seed": runtime.seed,
                "learning_rate": runtime.learning_rate, "warmup_ratio": runtime.warmup_ratio,
                "objective_weights": [trainer.ce_loss_alpha, trainer.l1_loss_alpha, trainer.confidence_head_alpha]}
    if args.deterministic:
        manifest["deterministic_algorithms"] = torch.are_deterministic_algorithms_enabled()
        manifest["deterministic_cudnn"] = torch.backends.cudnn.deterministic
    collator = None
    docs_per_row = 1
    if args.packed_pairs:
        from aspec.data.utils import DFlashPackingCollator

        if len(samples) < 2 or len(samples) % 2:
            raise ValueError("Packed alignment requires an even number of complete fixture documents")
        docs_per_row = 2
        collator = DFlashPackingCollator(args.length, pad_to_max_length=False,
                                         context_parallel_rank=trainer.cp_rank, context_parallel_size=args.cp,
                                         context_shard_alignment=getattr(runtime, "draft_context_shard_alignment", 1))
        if trainer.cp_size != args.cp or trainer.dp_size * args.cp != world:
            raise RuntimeError("Observed Draft topology does not match the packed comparison")
        if args.cp > 1:
            manifest["cp_size"] = args.cp
        manifest["packed_documents_per_row"] = docs_per_row
    if args.resume:
        prior = json.loads((args.resume.parent.parent / "run-manifest.json").read_text())
        if prior.get("packed_documents_per_row", 1) != docs_per_row:
            raise ValueError("Resume requires the original fixture packing policy")

    if rank == 0:
        write_json(output / "run-manifest.json", manifest)
        write_json(output / "runtime-config.json", json.loads(json.dumps(vars(runtime), default=str)))
        for reference in args.reference:
            wait_reference(reference / "run-manifest.json", args.reference_wait_seconds)
            if manifest != json.loads((reference / "run-manifest.json").read_text()):
                raise RuntimeError("Compared trajectories differ in fixed inputs or training contract")
    trainable = named_trainable(trainer)
    publish = trainer.dp_rank == 0
    writer = publish and getattr(trainer, "tp_rank", 0) == 0 and trainer.cp_rank == 0
    if args.phase != "resume" and publish:
        logical = logical_optimizer_state(trainer)
        if writer:
            initial = {key: {name: tensor_record(value) for name, value in tensors.items()}
                       for key, tensors in logical.items()}
            write_json(output / "initial-logical.json", initial)
            for reference in args.reference:
                wait_reference(reference / "initial-logical.json", args.reference_wait_seconds)
                expected = json.loads((reference / "initial-logical.json").read_text())
                if initial != expected:
                    raise RuntimeError("Compared trainers did not start from identical logical weights/masters")
        del logical
    observation = {"terms": [], "anchors": [], "gradients": {}, "before": {}}
    original_forward = trainer._forward

    def observed_forward(*values, **kwargs):
        result = original_forward(*values, **kwargs)
        observation["terms"].append(torch.stack([part.detach() for part in result[-1]]))
        return result

    trainer._forward = observed_forward
    original_sample = trainer.dflash._sample_anchor_positions

    def observed_sample(*values, **kwargs):
        result = original_sample(*values, **kwargs)
        observation["anchors"].append(tuple(value.detach().clone() for value in result))
        return result

    trainer.dflash._sample_anchor_positions = observed_sample
    original_step = trainer.optimizer.step

    def observed_step(*values, **kwargs):
        if publish and trainer.global_step + 1 in tensor_steps:
            for index, (name, parameter) in enumerate(trainable):
                gradient = (parameter.main_grad if hasattr(parameter, "main_grad") else parameter.grad)
                observation["gradients"][name] = logical_cpu(trainer, gradient, parameter)
                observation["before"][name] = logical_cpu(trainer, trainer.optimizer.fp32_params[index], parameter)
        norm = original_step(*values, **kwargs)
        observation["norm"] = norm.detach().clone()
        return norm

    trainer.optimizer.step = observed_step
    trainer._work_lifecycle_enabled = True
    stop = args.split_step if args.phase == "split" else args.steps
    if args.stop_after is not None:
        if not trainer.global_step < args.stop_after <= stop:
            raise ValueError("stop-after must follow the initial step and stay within the declared horizon")
        stop = args.stop_after
    dataset_size = args.steps * 2 * trainer.dp_size * docs_per_row
    while trainer.global_step < stop:
        step = trainer.global_step + 1
        observation.update(terms=[], anchors=[], gradients={}, before={})
        batches = []
        indices_by_replica = {}
        issued = []
        for replica in range(trainer.dp_size):
            indices_by_replica[str(replica)] = []
            for micro in range(2):
                row = (step - 1) * 2 * trainer.dp_size + micro * trainer.dp_size + replica
                indices = [(row * docs_per_row + doc) % len(samples) for doc in range(docs_per_row)]
                indices_by_replica[str(replica)].extend(indices)
                work_id = f"window-{step}/dp-{replica}/micro-{micro}"
                if args.packed_pairs:
                    issued.extend(f"{work_id}/sample-{index}" for index in indices)
                else:
                    issued.append(work_id)
                if replica == trainer.dp_rank:
                    batches.append(fixed_feature_row(samples, indices, args.length, work_id, collator))
        trainer.data_fetcher = batches
        metrics = trainer._train_core_from_queue(step=step - 1, num_batches=2)
        if hasattr(metrics, "finalize"):
            metrics = metrics.finalize()
        if args.packed_pairs:
            layers = trainer.draft_model.last_attention_observations
            if len(layers) != len(trainer.draft_model.layers) or not all(
                layer.get("qkv_format") == "thd" and layer.get("kernel_backend", "fa4") == args.attention_backend
                and layer.get("context_parallel_size") == args.cp
                and layer.get("context_parallel_rank") == trainer.cp_rank
                and layer.get("backward_observed") for layer in layers
            ):
                raise RuntimeError(f"Missing packed THD forward/backward evidence on rank {rank}: {layers}")
            write_json(output / f"thd-step{step}-rank{rank}.json", layers)
        local_work = trainer.pop_committed_work(step)
        participants = [None] * world
        dist.all_gather_object(participants, local_work)
        if args.cp > 1:
            for first in range(0, world, args.cp):
                if any(work != participants[first] for work in participants[first:first + args.cp]):
                    raise RuntimeError("CP ranks consumed different packed documents")
            participants = participants[::args.cp]
        new_work = [item for replica in participants for item in replica]
        if sorted(new_work) != sorted(issued) or set(new_work).intersection(committed):
            raise RuntimeError("Committed work was duplicated or omitted")
        committed.extend(new_work)
        totals = torch.stack(observation["terms"]).sum(0)
        dist.all_reduce(totals, group=trainer.dp_group)
        anchors = [{"positions": tensor_record(a), "keep": tensor_record(k)} for a, k in observation["anchors"]]
        write_json(output / f"anchors-step{step}-rank{rank}.json", anchors)
        participant_anchors = [None] * world
        replica_key = f"{trainer.dp_rank}/cp{trainer.cp_rank}" if args.cp > 1 else trainer.dp_rank
        dist.all_gather_object(participant_anchors, {"replica": replica_key, "anchors": anchors})
        by_replica = {}
        for participant in participant_anchors:
            replica = str(participant["replica"])
            if replica in by_replica and by_replica[replica] != participant["anchors"]:
                raise RuntimeError("TP participants used different anchors")
            by_replica[replica] = participant["anchors"]
        updates = {}
        if publish and step in tensor_steps:
            for index, (name, parameter) in enumerate(trainable):
                current = logical_cpu(trainer, trainer.optimizer.fp32_params[index], parameter)
                if writer:
                    updates[name] = current - observation["before"][name]
        if writer:
            vectors = {"gradients": observation["gradients"], "updates": updates}
            record = {"step": step, "loss": (totals[0] / totals[1].clamp_min(1e-6)).item(),
                      "numerator": totals[0].item(), "denominator": totals[1].item(),
                      "gradient_norm": observation["norm"].item(),
                      "learning_rate_after_update": trainer.optimizer.get_learning_rate(),
                      "gradient_dtypes": sorted({str(v.dtype) for v in vectors["gradients"].values() if v is not None}),
                      "anchors": anchors, "anchors_by_replica": by_replica,
                      "metrics": metrics, "comparisons": {},
                      "full_tensor_observation": step in tensor_steps,
                      "fixture_sample_indices_by_replica": indices_by_replica}
            if args.save_reference_tensors and step in tensor_steps and args.phase in {"uninterrupted", "checkpoint-control"}:
                torch.save(vectors, output / f"vectors-step{step}.pt")
            for reference in args.reference:
                wait_reference(reference / f"step{step}.json", args.reference_wait_seconds)
                expected_record = json.loads((reference / f"step{step}.json").read_text())
                measurements = {}
                if step in tensor_steps:
                    expected = torch.load(reference / f"vectors-step{step}.pt", weights_only=True, mmap=True)
                    measurements = {key: compare_vectors(value, expected[key]) for key, value in vectors.items()}
                    del expected
                scalar_values = {"objective": record["loss"], "gradient_norm": record["gradient_norm"],
                                 "learning_rate": record["learning_rate_after_update"],
                                 **{key: value for key, value in metrics.items() if key.startswith("train/")
                                    and isinstance(value, (int, float))}}
                expected_scalars = {"objective": expected_record["loss"],
                                    "gradient_norm": expected_record["gradient_norm"],
                                    "learning_rate": expected_record["learning_rate_after_update"],
                                    **expected_record["metrics"]}
                measurements["scalars"] = {key: scalar_diff(value, expected_scalars[key])
                                            for key, value in scalar_values.items()}
                measurements.update(loss_relative_error=measurements["scalars"]["objective"]["abs_rel_diff"],
                                    denominator_exact=record["denominator"] == expected_record["denominator"],
                                    anchors_exact=by_replica == expected_record["anchors_by_replica"],
                                    learning_rate_exact=record["learning_rate_after_update"] == expected_record["learning_rate_after_update"])
                measurements["tolerance_passed"] = (measurements["loss_relative_error"] <= 0.01
                    and measurements["denominator_exact"] and measurements["anchors_exact"]
                    and measurements["learning_rate_exact"]
                    and (step not in tensor_steps or all(
                        measurements[key]["relative_l2"] is not None
                        and measurements[key]["relative_l2"] <= 0.03 for key in vectors)))
                measurements["tolerance_scope"] = "loss/gradient/update" if step in tensor_steps else "loss only; full tensors not observed"
                record["comparisons"][str(reference)] = measurements
        if publish and step in state_steps:
            logical = logical_optimizer_state(trainer)
            if writer:
                if args.save_reference_tensors:
                    torch.save(logical, output / f"logical-state-step{step}.pt")
                state_comparisons = {}
                for reference in args.reference:
                    path = reference_state_path(reference, step, args.steps, args.reference_wait_seconds)
                    expected = torch.load(path, weights_only=True, mmap=True)
                    state_comparisons[str(reference)] = {key: compare_vectors(value, expected[key])
                                                         for key, value in logical.items()}
                    del expected
                write_json(output / f"logical-state-step{step}.json", state_comparisons)
            del logical
        if writer:
            write_json(output / f"step{step}.json", record)
            print(json.dumps({"step": step, "backend": args.backend, "loss": record["loss"],
                              "full_tensors": step in tensor_steps,
                              "scalar_rel_diffs": {ref: {key: values["abs_rel_diff"] for key, values in c["scalars"].items()
                                   if key in {"objective", "train/ce_loss", "train/l1_loss", "train/confidence_loss"}}
                                   for ref, c in record["comparisons"].items()}}), flush=True)
        observation.update(gradients={}, before={}, anchors=[], terms=[])
        if writer:
            del vectors
        del updates, batches
        dist.barrier()
        if args.phase == "checkpoint-control" and step == args.split_step:
            save_native_checkpoint(trainer, runtime, committed, fixture_sha, output, dataset_size)

    if args.phase == "split":
        save_native_checkpoint(trainer, runtime, committed, fixture_sha, output, dataset_size)
    write_json(output / f"final-state-rank{rank}.json", local_state(trainer))
    if trainer.global_step == args.steps and publish and args.steps not in state_steps:
        logical = logical_optimizer_state(trainer)
        if writer:
            if args.save_reference_tensors and args.phase in {"uninterrupted", "checkpoint-control"}:
                torch.save(logical, output / "final-logical.pt")
            comparisons = {}
            for reference in args.reference:
                path = reference_state_path(reference, args.steps, args.steps, args.reference_wait_seconds)
                expected = torch.load(path, weights_only=True, mmap=True)
                comparisons[str(reference)] = {key: compare_vectors(values, expected[key])
                                               for key, values in logical.items()}
            write_json(output / "final-logical-comparisons.json", comparisons)
        del logical
    elif trainer.global_step == args.steps and writer:
        write_json(output / "final-logical-comparisons.json",
                   json.loads((output / f"logical-state-step{args.steps}.json").read_text()))
    if attention_calls is not None:
        if not attention_calls["backward"]:
            raise RuntimeError("No native deterministic cuDNN attention backward calls were observed")
        write_json(output / f"attention-backward-rank{rank}.json", attention_calls)
    if rank == 0:
        if args.backend == "torch-tp":
            write_json(output / "torch-tp-execution.json", {
                "optimizer_fused_clip_scaling": trainer.optimizer._fuse_clip_scaling,
                "qk_preparation_backends": [layer.self_attn.qk_prepare_backend
                                            for layer in trainer.draft_model.layers],
                "l1_backend": trainer.dflash.l1_loss_backend,
                "l1_kernel_invocations": dict(trainer.dflash.l1_kernel_invocations),
                "megatron_core_imported": any(name == "megatron" or name.startswith("megatron.")
                                               for name in os.sys.modules),
            })
        write_json(output / "completed.json", {"phase": args.phase, "steps": trainer.global_step,
                   "backend": args.backend, "world_size": world, "tp": args.tp,
                   "committed_work_ids": committed, "teacher_sha256": fixture_sha,
                   "runtime": identity, "attention_backend": args.attention_backend})
        write_json(output / "observation-policy.json", {
            "tensor_steps": sorted(tensor_steps), "state_steps": sorted(state_steps),
            "save_reference_tensors": args.save_reference_tensors,
            "scalar_rel_diff": "(actual-reference) / max(abs(reference), 1e-12)",
            "tensor_pointwise_rel_diff": "abs(actual-reference) / max(abs(reference), max(1e-12, 1e-6 * per-parameter reference RMS))",
            "trajectory": "independent native updates from identical initial weights; subsequent differences include accumulated drift",
            "fixture_samples": len(samples), "fixture_repeats": True,
            "initial_sha256": file_sha(args.initial) if args.initial else None})
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
