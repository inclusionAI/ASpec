# Copyright 2026 Ant Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

import os

import pytest
import torch

from tests.tp_input import tp_input_worker


@pytest.mark.skipif(
    os.getenv("ASPEC_RUN_GPU_INTEGRATION") != "1" or torch.cuda.device_count() < 2,
    reason="requires ASPEC_RUN_GPU_INTEGRATION=1 and two visible GPUs",
)
def test_tp_preparation_error_reaches_both_ranks_nccl(tmp_path):
    torch.multiprocessing.spawn(tp_input_worker, args=((tmp_path / "rdzv").as_uri(), True), nprocs=2)
