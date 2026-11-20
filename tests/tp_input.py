# Copyright 2026 Ant Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

from datetime import timedelta
from types import SimpleNamespace

import pytest
import torch


def tp_input_worker(rank, init_method, use_cuda):
    import torch.distributed as dist

    from aspec.training.data_fetcher import MooncakeDataset

    dist.init_process_group("gloo", init_method=init_method, rank=rank, world_size=2,
                            timeout=timedelta(seconds=60))
    try:
        dataset = object.__new__(MooncakeDataset)
        dataset.tensor_parallel_control_group = dist.group.WORLD
        if use_cuda:
            torch.cuda.set_device(rank)
        group = dist.new_group(backend="nccl") if use_cuda else dist.group.WORLD
        dataset.tensor_parallel_group = group
        dataset._tp_source_prefetch = None
        dataset._tensor_parallel_samples = dataset._tensor_parallel_payload_bytes = 0
        dataset._feature_contract_runtime = None
        sample = SimpleNamespace(tensor_shapes={}, tensor_dtypes={}, metadata={})
        dataset._read_tp_source = lambda: (sample, {"ids": torch.tensor([5])} if use_cuda else {})
        _, data = dataset._tensor_parallel_item_on_device()
        if use_cuda:
            assert data["ids"].tolist() == [5]
            assert data["ids"].device.type == "cpu"

        def reject(**kwargs):
            raise ValueError("invalid follower metadata")

        dataset._feature_contract_runtime = SimpleNamespace(validate_metadata=reject)
        with pytest.raises(RuntimeError, match="rank 1.*invalid follower metadata"):
            dataset._tensor_parallel_item_on_device()
    finally:
        dist.destroy_process_group()
