# Copyright 2026 Ant Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""DSpark's retained training loop with upstream Megatron TP and DDP.

Inspired by the public Megatron Core 0.20 custom-model DDP interface. DSpark
algorithm, batching, optimizer-step normalization and recovery remain ASpec-owned.
"""

import weakref
from pathlib import Path

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor

from aspec.models.megatron import initialize_parallelism
from aspec.models.megatron_draft import (
    draft_transformer_config,
    load_model_checkpoint_state,
    model_checkpoint_state,
    partition_dspark,
)
from aspec.training.dspark_trainer import DSparkTrainer
from aspec.training.megatron_optimizer import MegatronBF16Optimizer


class MegatronDSparkTrainer(DSparkTrainer):
    execution_backend = "megatron"
    _optimizer_class = MegatronBF16Optimizer

    def _setup_device_mesh(self):
        self.groups = initialize_parallelism(self.args.draft_tensor_parallel_size,
                                            seed=getattr(self.args, "seed", 42))
        self.tp_size, self.tp_rank = self.groups.tp.size(), self.groups.tp.rank()
        self.dp_size, self.dp_rank = self.groups.dp.size(), self.groups.dp.rank()
        self.dp_group = self.groups.dp
        self.cp_size, self.cp_rank, self.cp_group = 1, 0, None
        self.cache_rank = dist.get_rank()
        self.tp_mesh = DeviceMesh.from_group(self.groups.tp, "cuda")
        self.dp_mesh = DeviceMesh.from_group(self.groups.dp, "cuda")
        self.mesh = self.grad_sync_mesh = self.dp_mesh
        # Only source GET is prefetched. Input collectives run on the model's
        # consumer thread, in a consistent order across overlapping TP/DP groups.
        for replica in range(self.dp_size):
            ranks = list(range(replica * self.tp_size, (replica + 1) * self.tp_size))
            group = dist.new_group(ranks, backend="nccl")
            control_group = dist.new_group(ranks, backend="gloo")
            eval_group = dist.new_group(ranks, backend="nccl")
            eval_control_group = dist.new_group(ranks, backend="gloo")
            if replica == self.dp_rank:
                self.input_tp_group = group
                self.input_tp_control_group = control_group
                self.eval_input_tp_group = eval_group
                self.eval_input_tp_control_group = eval_control_group

    def _get_init_weight_context_manager(self):
        return lambda: torch.device("cpu")

    def _loads_initial_weights_here(self):
        return True

    def _build_training_wrapper(self, draft_model):
        wrapper = super()._build_training_wrapper(draft_model)
        sample = wrapper._sample_anchor_positions
        wrapper._aspec_anchor_tp_broadcasts = 0

        def shared_anchors(*args, **kwargs):
            anchors, keep = sample(*args, **kwargs)
            if self.tp_size > 1:
                leader = dist.get_global_rank(self.groups.tp, 0)
                dist.broadcast(anchors, src=leader, group=self.groups.tp)
                dist.broadcast(keep, src=leader, group=self.groups.tp)
                wrapper._aspec_anchor_tp_broadcasts += 1
            return anchors, keep

        wrapper._sample_anchor_positions = shared_anchors
        return wrapper

    def _wrap_training_model(self, model, draft_model):
        from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
        from megatron.core.tensor_parallel import (
            set_defaults_if_not_set_tensor_model_parallel_attributes,
        )

        config = draft_transformer_config(draft_model.config, self.tp_size)
        partition_dspark(draft_model, config, self.groups, self.tp_mesh)
        for parameter in model.parameters():
            set_defaults_if_not_set_tensor_model_parallel_attributes(parameter)
        ddp = DistributedDataParallel(
            config, DistributedDataParallelConfig(
                grad_reduce_in_fp32=True, overlap_grad_reduce=True,
                average_in_collective=True, use_distributed_optimizer=False),
            model, pg_collection=self.groups,
        )
        ddp.broadcast_params()
        self._tp_partial_gradients = [p.main_grad for p in model.parameters()
                                     if getattr(p, "_aspec_tp_partial_gradient", False)]
        self._tp_partial_buffer = torch.empty(
            sum(g.numel() for g in self._tp_partial_gradients), device="cuda", dtype=torch.float32,
        )
        self._tp_partial_views = list(self._tp_partial_buffer.split(
            [g.numel() for g in self._tp_partial_gradients]))
        # A weak reference avoids making a second module parent / state-dict cycle.
        object.__setattr__(draft_model, "_aspec_ddp", weakref.proxy(ddp))
        ddp._aspec_checkpoint_state = lambda: model_checkpoint_state(model, self.tp_mesh)
        ddp._aspec_load_checkpoint_state = lambda state: load_model_checkpoint_state(model, state)
        return ddp

    def _normalize_accumulated_gradients(self, local_denominator, accumulation_steps):
        self.model.finish_grad_sync()
        if self.tp_size > 1:
            flattened = [g.view(-1) for g in self._tp_partial_gradients]
            torch._foreach_copy_(self._tp_partial_views, flattened)
            dist.all_reduce(self._tp_partial_buffer, group=self.groups.tp)
            torch._foreach_copy_(flattened, self._tp_partial_views)
        denominator = local_denominator.detach().clone()
        dist.all_reduce(denominator, group=self.dp_group)
        scale = torch.where(
            denominator > 0,
            self.dp_size * accumulation_steps / denominator.clamp_min(1e-6),
            torch.zeros_like(denominator),
        )
        from aspec.models.ops.gradient_scaling import scale_gradients_

        scale_gradients_([p.main_grad for p in self.optimizer.model_params], scale)
        return denominator

    def _checkpoint_backend_rng(self):
        from megatron.core.tensor_parallel.random import get_cuda_rng_tracker

        return {"backend": "megatron", "cuda_tracker": get_cuda_rng_tracker().get_states()}

    def _restore_backend_rng(self, state):
        from megatron.core.tensor_parallel.random import get_cuda_rng_tracker

        if not isinstance(state, dict) or state.get("backend") != "megatron":
            raise RuntimeError("Exact Megatron resume requires the model-parallel RNG tracker")
        saved = state.get("cuda_tracker")
        if not isinstance(saved, dict) or set(saved) != set(get_cuda_rng_tracker().get_states()):
            raise RuntimeError("Megatron RNG tracker streams differ from the checkpoint")
        if any(not isinstance(value, torch.Tensor) or value.dtype != torch.uint8
               or value.device.type != "cpu" or value.ndim != 1 for value in saved.values()):
            raise RuntimeError("Invalid Megatron RNG tracker state")
        get_cuda_rng_tracker().set_states(saved)

    def pop_committed_work(self, optimizer_step):
        local = super().pop_committed_work(optimizer_step)
        replicas = [None] * self.tp_size
        dist.all_gather_object(replicas, local, group=self.groups.tp)
        if any(items != local for items in replicas):
            raise RuntimeError("TP ranks completed different ASpec work identities")
        return local if self.tp_rank == 0 else []

    def _kernel_observation(self):
        wrapper = getattr(self, "dflash", None)
        if getattr(wrapper, "l1_loss_backend", None) is None:
            return None
        return {
            "l1_backend": wrapper.l1_loss_backend,
            "l1_kernel_invocations": dict(wrapper.l1_kernel_invocations),
            "anchor_tp_broadcasts": wrapper._aspec_anchor_tp_broadcasts,
            "last_batch": {name.removesuffix("_observed"): value
                           for name, value in vars(wrapper).items()
                           if name.endswith("_observed") and type(value) in (int, float)},
            "attention_layers": list(self.draft_model.last_attention_observations),
        }

    def get_feature_contract_status(self):
        status = super().get_feature_contract_status()
        if observation := self._kernel_observation():
            status["dspark_execution"] = observation
        return status

    def save_draft_model_for_serving(self, output_dir):
        error = None
        try:
            if self.dp_rank == 0:
                state = model_checkpoint_state(self.draft_model, self.tp_mesh)
                state = {name: (value.full_tensor() if isinstance(value, DTensor) else value).cpu()
                         for name, value in state.items()}
                if self.tp_rank == 0:
                    root = Path(output_dir)
                    root.mkdir(parents=True, exist_ok=True)
                    self.draft_model.config.save_pretrained(root)
                    torch.save(state, root / "pytorch_model.bin")
        except Exception as caught:
            error = caught
        message = self._synchronize_serving_error(error)
        if message is not None:
            raise RuntimeError("Megatron Draft export failed: " + message)
