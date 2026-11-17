# Copyright 2026 Ant Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Backend policy must preserve explicit choices in both compilation paths."""

import copy
from collections import Counter

import pytest
import torch

import aspec.models.ops.flex_attention as flex_mod


@pytest.mark.parametrize("compiling", [False, True], ids=["wrapper", "nested-compile"])
@pytest.mark.parametrize("options,expected", [
    ({}, {"BACKEND": "TRITON"}),
    ({"kernel_options": None}, {"BACKEND": "TRITON"}),
    ({"kernel_options": {}}, {"BACKEND": "TRITON"}),
    ({"kernel_options": {"BLOCK_M": 64}}, {"BLOCK_M": 64, "BACKEND": "TRITON"}),
    ({"kernel_options": {"BACKEND": "AUTO"}}, {"BACKEND": "AUTO"}),
    ({"kernel_options": {"BACKEND": "TRITON"}}, {"BACKEND": "TRITON"}),
    ({"kernel_options": {"BACKEND": "TRITON_DECODE"}}, {"BACKEND": "TRITON_DECODE"}),
    ({"kernel_options": {"BACKEND": "FLASH"}}, {"BACKEND": "FLASH"}),
    ({"kernel_options": {"FORCE_USE_FLEX_ATTENTION": True}}, {"FORCE_USE_FLEX_ATTENTION": True}),
    ({"kernel_options": {"FORCE_USE_FLEX_ATTENTION": False}}, {"FORCE_USE_FLEX_ATTENTION": False}),
], ids=["omitted", "none", "empty", "tuning", "auto", "triton", "decode", "flash",
        "legacy-force", "legacy-auto"])
def test_kernel_backend_policy_preserves_caller_choices(monkeypatch, compiling, options, expected):
    original = copy.deepcopy(options)
    inputs = [torch.empty(1) for _ in range(3)]
    result = object()
    calls = []

    def attention(*args, **kwargs):
        calls.append((args, kwargs))
        return result

    monkeypatch.setattr(flex_mod, "is_torchdynamo_compiling", lambda: compiling)
    monkeypatch.setattr(flex_mod, "WrappedFlexAttention", lambda: lambda: attention)
    monkeypatch.setattr(flex_mod, "flex_attention", attention)
    assert flex_mod.compile_friendly_flex_attention(*inputs, enable_gqa=True, **options) is result
    assert len(calls) == 1
    args, kwargs = calls[0]
    assert all(actual is supplied for actual, supplied in zip(args, inputs, strict=True))
    assert kwargs == {"enable_gqa": True, "kernel_options": expected}
    assert options == original
    if options.get("kernel_options") is not None:
        assert kwargs["kernel_options"] is not options["kernel_options"]


def test_new_compilation_collects_tracing_cycles_but_cache_hits_do_not(monkeypatch):
    counters = {"frames": Counter(), "stats": Counter()}
    collected, calls = [], []
    result = object()

    def attention(*args, **kwargs):
        if not calls:
            counters["frames"]["total"] += 1
        calls.append(args)
        return result

    monkeypatch.setattr(flex_mod, "is_torchdynamo_compiling", lambda: False)
    monkeypatch.setattr(flex_mod, "WrappedFlexAttention", lambda: lambda: attention)
    monkeypatch.setattr(flex_mod, "dynamo_counters", counters)
    monkeypatch.setattr(flex_mod.gc, "collect", lambda: collected.append(True))
    for _ in range(3):
        assert flex_mod.compile_friendly_flex_attention(None, None, None) is result
    assert len(calls) == 3
    assert collected == [True]


def test_tracing_cleanup_preserves_compilation_failure(monkeypatch):
    counters = {"frames": Counter(), "stats": Counter()}
    collected = []
    failure = RuntimeError("attention compilation failed")

    def attention(*args, **kwargs):
        counters["stats"]["unique_graphs"] += 1
        raise failure

    monkeypatch.setattr(flex_mod, "is_torchdynamo_compiling", lambda: False)
    monkeypatch.setattr(flex_mod, "WrappedFlexAttention", lambda: lambda: attention)
    monkeypatch.setattr(flex_mod, "dynamo_counters", counters)
    monkeypatch.setattr(flex_mod.gc, "collect", lambda: collected.append(True))
    with pytest.raises(RuntimeError) as caught:
        flex_mod.compile_friendly_flex_attention(None, None, None)
    assert caught.value is failure
    assert collected == [True]
