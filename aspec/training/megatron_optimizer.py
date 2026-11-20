# Copyright 2026 Ant Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Retain ASpec's BF16 master-AdamW semantics over Megatron TP/DDP buffers."""

import torch
import torch.distributed as dist

from aspec.models.megatron_draft import checkpoint_tensor
from aspec.training.optimizer import BF16Optimizer


class MegatronBF16Optimizer(BF16Optimizer):
    def _model_gradient(self, parameter):
        return parameter.main_grad

    def checkpoint_tensor(self, index, value):
        return checkpoint_tensor(value, self.model_params[index], self.model._aspec_tp_mesh)

    def _clip_gradients(self):
        group = self.model._aspec_tp_group
        # Replicated vocabulary/confidence/norm parameters contribute once;
        # every distinct attention/FFN shard contributes to the global norm.
        gradients = [master.grad for parameter, master in zip(
            self.model_params, self.fp32_params, strict=True)
            if master.grad is not None
            and (getattr(parameter, "_aspec_tp_axis", None) is not None or group.rank() == 0)]
        norm = torch.nn.utils.get_total_norm(gradients).square()
        dist.all_reduce(norm, op=dist.ReduceOp.SUM, group=group)
        norm.sqrt_()
        coefficient = (self.max_grad_norm / (norm + 1e-6)).clamp(max=1.0)
        if self._fuse_clip_scaling:
            self.optimizer.grad_scale = coefficient.reciprocal()
        else:
            from aspec.models.ops.gradient_scaling import scale_gradients_

            self.optimizer.grad_scale = None
            scale_gradients_([p.grad for p in self.fp32_params if p.grad is not None], coefficient)
        return norm

    def step(self, closure=None):
        norm = super().step(closure)
        self.model._aspec_ddp.zero_grad_buffer()
        return norm

    def zero_grad(self, set_to_none=True):
        super().zero_grad(set_to_none=set_to_none)
        self.model._aspec_ddp.zero_grad_buffer()
