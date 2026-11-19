# Copyright 2026 Ant Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Check native DTensor TP gradients against independent split-linear autograd."""

from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.device_mesh import init_device_mesh

from tools.validation.torch_tp_dspark import TorchTPLinear


def _projection_worker(rank, rendezvous):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2,
                            timeout=timedelta(seconds=60))
    try:
        mesh = init_device_mesh("cpu", (2,))
        for dtype in (torch.float32, torch.bfloat16):
            for axis in (0, 1):
                torch.manual_seed(114)
                full = torch.nn.Linear(8, 12, bias=False, dtype=dtype)
                layer = TorchTPLinear(full, mesh, axis)
                reference_weight = full.weight.detach().clone().requires_grad_(True)
                reference_input = torch.randn(2, 3, 8, dtype=dtype, requires_grad=True)
                gradient = torch.randn(2, 3, 12, dtype=dtype)
                pieces = reference_weight.chunk(2, dim=axis)
                if axis == 0:
                    expected = torch.cat([torch.nn.functional.linear(reference_input, w)
                                          for w in pieces], dim=-1)
                    local = reference_input.detach().clone().requires_grad_(True)
                    local_gradient = gradient.chunk(2, dim=-1)[rank]
                else:
                    partials = [torch.nn.functional.linear(x.contiguous(), w.contiguous())
                                for x, w in zip(reference_input.chunk(2, dim=-1), pieces, strict=True)]
                    expected = partials[0] + partials[1]
                    local = reference_input.detach().chunk(2, dim=-1)[rank].clone().requires_grad_(True)
                    local_gradient = gradient
                actual = layer(local)
                expected_output = expected.chunk(2, dim=-1)[rank] if axis == 0 else expected
                torch.testing.assert_close(actual, expected_output, rtol=0, atol=0)
                # Exercise native BF16 leaf accumulation across two microbatches.
                for _ in range(2):
                    (layer(local).float() * local_gradient.float()).sum().backward()
                    (expected.float() * gradient.float()).sum().backward(retain_graph=True)
                torch.testing.assert_close(layer.weight.grad,
                                           reference_weight.grad.chunk(2, dim=axis)[rank], rtol=0, atol=0)
                expected_input_gradient = (reference_input.grad if axis == 0
                                           else reference_input.grad.chunk(2, dim=-1)[rank])
                torch.testing.assert_close(local.grad, expected_input_gradient, rtol=0, atol=0)
                assert layer.weight.grad.dtype == dtype
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_gloo_available(), reason="Requires native Gloo collectives")
def test_dtensor_column_and_row_shards_preserve_forward_and_accumulated_backward(tmp_path):
    mp.spawn(_projection_worker, args=((tmp_path / "rendezvous").as_uri(),), nprocs=2, join=True)
