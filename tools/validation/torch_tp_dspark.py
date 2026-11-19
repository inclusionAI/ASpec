# Copyright 2026 Ant Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Validation-only native PyTorch TP for the retained ASpec DSpark trainer.

Independently implements the public DTensor linear placements; no Megatron
operators or optimizer are imported. Plain local Parameters preserve ASpec's
fused master-AdamW/clipping path. DTensor wraps operands inside each projection
so its native autograd owns TP collectives. This is not a production backend:
the public Torch configuration continues to reject Draft TP greater than one.
"""

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor, Replicate, Shard

from aspec.training.dspark_trainer import DSparkTrainer
from aspec.training.fsdp import apply_fsdp2
from aspec.training.optimizer import BF16Optimizer


class TorchTPLinear(torch.nn.Module):
    def __init__(self, original, mesh, axis):
        super().__init__()
        if original.bias is not None:
            raise ValueError("The Torch TP validation path requires bias-free projections")
        self.mesh, self.axis = mesh, axis
        self.weight = torch.nn.Parameter(
            original.weight.chunk(mesh.size(), dim=axis)[mesh.get_local_rank()].clone(),
            requires_grad=original.weight.requires_grad,
        )
        self.weight._aspec_tp_axis = axis

    def forward(self, value):
        placement = Replicate() if self.axis == 0 else Shard(value.ndim - 1)
        inputs = DTensor.from_local(value, self.mesh, (placement,), run_check=False)
        weight = DTensor.from_local(self.weight, self.mesh, (Shard(self.axis),), run_check=False)
        result = F.linear(inputs, weight)
        if self.axis == 1:
            result = result.redistribute(placements=(Replicate(),))
        return result.to_local()


class TorchTPOptimizer(BF16Optimizer):
    def checkpoint_tensor(self, index, value):
        return self.model._validation_logical_tensor(value, self.model_params[index])

    def _clip_gradients(self):
        group = self.model._validation_tp_group
        gradients = [master.grad for parameter, master in zip(
            self.model_params, self.fp32_params, strict=True)
            if master.grad is not None
            and (hasattr(parameter, "_aspec_tp_axis") or group.rank() == 0)]
        # The global vector contains each replicated parameter once, and every
        # distinct shard. Retain the native Torch norm and fused clip-scaling.
        norm_squared = torch.nn.utils.get_total_norm(gradients).square()
        dist.all_reduce(norm_squared, group=group)
        norm = norm_squared.sqrt_()
        coefficient = (self.max_grad_norm / (norm + 1e-6)).clamp(max=1.0)
        if self._fuse_clip_scaling:
            self.optimizer.grad_scale = coefficient.reciprocal()
        else:
            from aspec.models.ops.gradient_scaling import scale_gradients_

            self.optimizer.grad_scale = None
            scale_gradients_([p.grad for p in self.fp32_params if p.grad is not None], coefficient)
        return norm


class TorchTPDSparkTrainer(DSparkTrainer):
    execution_backend = "torch-tp-validation"
    _optimizer_class = TorchTPOptimizer

    def _setup_device_mesh(self):
        size = self.args.draft_tensor_parallel_size
        if size != 2 or dist.get_world_size() % size:
            raise ValueError("The Torch TP validation adapter requires TP2 and complete DP replicas")
        mesh = init_device_mesh("cuda", (dist.get_world_size() // size, size),
                                mesh_dim_names=("dp", "tp"))
        self.tp_mesh, self.dp_mesh = mesh["tp"], mesh["dp"]
        self.tp_group, self.dp_group = self.tp_mesh.get_group(), self.dp_mesh.get_group()
        self.tp_size, self.tp_rank = size, self.tp_mesh.get_local_rank()
        self.dp_size, self.dp_rank = self.dp_mesh.size(), self.dp_mesh.get_local_rank()
        self.mesh = self.grad_sync_mesh = self.dp_mesh
        self.cp_size, self.cp_rank, self.cp_group = 1, 0, None
        self.cache_rank = dist.get_rank()

    def _get_init_weight_context_manager(self):
        return lambda: torch.device("cpu")

    def _loads_initial_weights_here(self):
        return True

    def _build_training_wrapper(self, draft_model):
        wrapper = super()._build_training_wrapper(draft_model)
        original = wrapper._sample_anchor_positions

        def sample(*args, **kwargs):
            anchors, keep = original(*args, **kwargs)
            leader = dist.get_global_rank(self.tp_group, 0)
            dist.broadcast(anchors, src=leader, group=self.tp_group)
            dist.broadcast(keep, src=leader, group=self.tp_group)
            return anchors, keep

        wrapper._sample_anchor_positions = sample
        return wrapper

    def logical_tensor(self, value, parameter):
        axis = getattr(parameter, "_aspec_tp_axis", None)
        if axis is None or value.ndim == 0:
            return value
        return DTensor.from_local(value, self.tp_mesh, (Shard(axis),), run_check=False)

    def _wrap_training_model(self, model, draft_model):
        if (self.args.fsdp_strategy != "REPLICATE" or self.args.draft_context_parallel_size != 1
                or self.args.dflash_packing or self.args.compile_model):
            raise ValueError("Torch TP validation requires native REPLICATE, padded CP1 execution")
        config = draft_model.config
        if any(getattr(config, key) % self.tp_size for key in (
            "num_attention_heads", "num_key_value_heads", "intermediate_size",
        )):
            raise ValueError("Attention heads and FFN width must divide the Torch TP size")
        draft_model.cuda()
        for layer in draft_model.layers:
            for owner, names, axis in (
                (layer.self_attn, ("q_proj", "k_proj", "v_proj"), 0),
                (layer.self_attn, ("o_proj",), 1),
                (layer.mlp, ("gate_proj", "up_proj"), 0),
                (layer.mlp, ("down_proj",), 1),
            ):
                for name in names:
                    setattr(owner, name, TorchTPLinear(getattr(owner, name), self.tp_mesh, axis))
            layer.self_attn.num_heads //= self.tp_size
            layer.self_attn.num_kv_heads //= self.tp_size
        self._partial_norm_parameters = [norm.weight for layer in draft_model.layers
                                         for norm in (layer.self_attn.q_norm, layer.self_attn.k_norm)]
        # Head-shared norm weights receive contributions from local heads only.
        # Sum their partial derivatives in FP32, retaining BF16 leaf gradients.
        self._partial_buffer = torch.empty(sum(p.numel() for p in self._partial_norm_parameters),
                                            device="cuda", dtype=torch.float32)
        self._partial_views = list(self._partial_buffer.split(
            [p.numel() for p in self._partial_norm_parameters]))
        model = apply_fsdp2(model, mesh=self.dp_mesh, args=self.args)
        object.__setattr__(draft_model, "_validation_tp_group", self.tp_group)
        object.__setattr__(draft_model, "_validation_logical_tensor", self.logical_tensor)

        def state():
            parameters = dict(model.named_parameters())
            return {name: self.logical_tensor(value, parameters[name]) if name in parameters else value
                    for name, value in model.state_dict().items() if isinstance(value, torch.Tensor)}

        def load(values):
            current = model.state_dict()
            if set(current) != set(values):
                raise ValueError("The Torch TP checkpoint must cover the full model")
            with torch.no_grad():
                for name, value in values.items():
                    local = value.to_local() if isinstance(value, DTensor) else value
                    if current[name].shape != local.shape:
                        raise ValueError(f"Torch TP checkpoint shape mismatch: {name}")
                    current[name].copy_(local)

        model._aspec_checkpoint_state, model._aspec_load_checkpoint_state = state, load
        return model

    def _normalize_accumulated_gradients(self, local_denominator, accumulation_steps):
        flat = [p.grad.view(-1) for p in self._partial_norm_parameters]
        torch._foreach_copy_(self._partial_views, flat)
        dist.all_reduce(self._partial_buffer, group=self.tp_group)
        torch._foreach_copy_(flat, self._partial_views)
        return super()._normalize_accumulated_gradients(local_denominator, accumulation_steps)

    def pop_committed_work(self, optimizer_step):
        local = super().pop_committed_work(optimizer_step)
        participants = [None] * self.tp_size
        dist.all_gather_object(participants, local, group=self.tp_group)
        if any(work != local for work in participants):
            raise RuntimeError("Torch TP participants committed different work")
        return local if self.tp_rank == 0 else []

    def _checkpoint_backend_rng(self):
        return {"backend": "torch-tp-validation", "tp_size": self.tp_size}

    def _restore_backend_rng(self, state):
        if state != self._checkpoint_backend_rng():
            raise RuntimeError("Torch TP validation resume cannot change backend or TP size")
