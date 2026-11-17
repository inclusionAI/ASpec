# Copyright 2026 Ant Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Offline geometry contracts for DFlash's cuDNN attention adapter."""

import pytest
import torch

from aspec.models.draft.dflash import DFlashAttention, DFlashConfig


@pytest.mark.parametrize("head_dim,kv_heads,supported", [
    (64, 4, True), (64, 1, False), (96, 4, False), (128, 1, True),
])
def test_cudnn_draft_head_geometry(head_dim, kv_heads, supported):
    config = DFlashConfig(
        hidden_size=256, intermediate_size=512, num_hidden_layers=1,
        num_attention_heads=4, num_key_value_heads=kv_heads,
        head_dim=head_dim, vocab_size=256, num_target_layers=1,
        target_hidden_size=256, target_num_hidden_layers=1,
    )
    with torch.device("meta"):
        attention = DFlashAttention(config)
    if supported:
        attention.set_attention_backend("cudnn_flex_attention")
        assert attention.attention_backend == "cudnn_flex_attention"
    else:
        with pytest.raises(NotImplementedError, match="head_dim"):
            attention.set_attention_backend("cudnn_flex_attention")
