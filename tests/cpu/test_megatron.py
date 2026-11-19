# Copyright 2026 Ant Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

import subprocess
import sys
import threading
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from aspec.config.megatron import validate_megatron_execution
from aspec.config.train_config import load_config
from aspec.feature_contract import build_target_feature_contract, validate_feature_contract
from aspec.models.megatron import qwen3_parameter
from aspec.planning import compile_execution_plan


def test_baseline_import_does_not_require_megatron():
    result = subprocess.run([sys.executable, "-c", """
import sys
class RejectMegatron:
    def find_spec(self, name, *args):
        if name == 'megatron' or name.startswith('megatron.'):
            raise AssertionError('Baseline imported an optional backend')
sys.meta_path.insert(0, RejectMegatron())
import aspec.inference.factory
import aspec.training.trainer_actor
"""], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def settings(**overrides):
    return Namespace(**{"train_backend": "megatron", "target_model_backend": "megatron",
                        "draft_tensor_parallel_size": 2, "world_size": 4,
                        "megatron_tp_size": 2, "inference_num_gpus_per_engine": 2,
                        "target_prefill_memory_optimized": True, **overrides})


@pytest.mark.parametrize("target,training", [("megatron", "fsdp"), ("sglang", "megatron"),
                                           ("megatron", "megatron")])
def test_independent_target_draft_pairings(target, training):
    validate_megatron_execution(settings(target_model_backend=target, train_backend=training,
                                         draft_tensor_parallel_size=1 if training == "fsdp" else 2),
                                {"family": "dspark"})


@pytest.mark.parametrize("dtype,resolved", [("fp32", "float32"), ("bf16", "bfloat16")])
def test_gradient_precision_configuration_reaches_plan_and_runtime(dtype, resolved):
    recipe = Path(__file__).resolve().parents[2] / "configs/recipes/quickstart_qwen3_8b.yaml"
    config = load_config(str(recipe), cli_args=[
        "training.train_backend=megatron", "training.draft_tensor_parallel_size=2",
        "training.training_num_gpus_per_node=2", "training.prefetch_depth=0",
        f"training.megatron_grad_dtype={dtype}",
    ])
    plan = compile_execution_plan(config)
    assert plan.to_runtime_args().megatron_grad_dtype == dtype
    policy = plan.policy.parallelism["training"]
    assert policy["gradient_accumulation_dtype"] == policy["gradient_reduce_dtype"] == resolved
    assert policy["core_grad_reduce_in_fp32"] == (dtype == "fp32")
    assert policy["tp_partial_gradient_reduce_dtype"] == policy["optimizer_state_dtype"] == "float32"


@pytest.mark.parametrize("backend,dtype", [("megatron", "fp16"), ("megatron", "float32"),
                                          ("megatron", None), ("fsdp", "bf16")])
def test_gradient_precision_rejects_invalid_or_wrong_backend_configuration(backend, dtype):
    with pytest.raises(ValueError, match="megatron_grad_dtype"):
        validate_megatron_execution(settings(train_backend=backend, draft_tensor_parallel_size=1,
                                             megatron_grad_dtype=dtype), {"family": "dspark"})


@pytest.mark.parametrize("saved,current,allowed", [
    (None, "fp32", True), (None, "bf16", False),
    ("fp32", "fp32", True), ("bf16", "bf16", True),
    ("fp32", "bf16", False), ("bf16", "fp32", False),
])
def test_exact_resume_preserves_megatron_gradient_precision(saved, current, allowed):
    from aspec.training.checkpoint import (
        _logical_training_topology,
        _validate_exact_resume_topology,
    )

    args = Namespace(train_backend="megatron", draft_tensor_parallel_size=1,
                     world_size=1, megatron_grad_dtype=current)
    topology = _logical_training_topology(args, 1)
    if saved is None:
        topology.pop("megatron_grad_dtype")
    else:
        topology["megatron_grad_dtype"] = saved
    metadata = {"world_size": 1, "training_topology": topology}
    if allowed:
        _validate_exact_resume_topology(args, metadata)
    else:
        with pytest.raises(RuntimeError, match="gradient precision"):
            _validate_exact_resume_topology(args, metadata)


@pytest.mark.parametrize("overrides,message", [
    ({"world_size": 3}, "divisible"),
    ({"draft_context_parallel_size": 2}, "CP1"),
    ({"dflash_packing": True}, "packed"),
    ({"fsdp_cpu_offload": True}, "offload"),
    ({"distributed_backend": "gloo"}, "NCCL"),
    ({"micro_batch_size": 2}, "micro_batch_size"),
    ({"prefetch_depth": 1, "mooncake_destination_gpu_ring_slots": 1}, "destination_gpu_ring_slots"),
    ({"sglang_pp_size": 2}, "PP1"),
    ({"target_prefill_memory_optimized": False}, "memory_optimized"),
    ({"inference_num_gpus_per_engine": 1}, "must equal"),
    ({"train_backend": "fsdp", "draft_tensor_parallel_size": 2}, "Draft TP"),
])
def test_megatron_fails_closed_before_process_launch(overrides, message):
    with pytest.raises(ValueError, match=message):
        validate_megatron_execution(settings(**overrides), {"family": "dspark"})


def test_producer_backend_is_part_of_the_feature_identity():
    kwargs = dict(drafter_family="dspark", target_hidden_size=16, target_num_hidden_layers=4,
                  aux_hidden_state_layers=[0, 2], store_last_hidden_states=True)
    sglang = build_target_feature_contract(**kwargs)
    megatron = build_target_feature_contract(**kwargs, producer_backend="megatron")
    validate_feature_contract(megatron)
    assert sglang["contract_id"] != megatron["contract_id"]


def test_tp_source_capacity_counts_dp_replicas_and_target_engines():
    recipe = Path(__file__).resolve().parents[2] / "configs/recipes/quickstart_qwen3_8b.yaml"
    overrides = [
        "model.target_model_backend=megatron", "inference.inference_engine_type=megatron",
        "training.train_backend=megatron", "training.draft_tensor_parallel_size=2",
        "training.training_num_gpus_per_node=4", "training.draft_accumulation_steps=1",
        "inference.megatron.tp_size=2", "inference.inference_num_gpus=2",
        "inference.inference_num_gpus_per_engine=2", "inference.inference_batch_size=1",
        "mooncake.source_gpu_ring_slots=2",
        "mooncake.destination_gpu_ring_slots=3",
    ]
    config = load_config(str(recipe), cli_args=overrides)
    plan = compile_execution_plan(config)
    assert plan.to_runtime_args().dp_size == 2
    assert plan.policy.pipeline["global_batch_size"] == 2
    # Two Target GPUs still represent one engine with one source ring.
    with pytest.raises(ValueError, match="cannot retain one complete Draft dispatch"):
        load_config(base_config=config, cli_args=["mooncake.source_gpu_ring_slots=1"])


def test_tp_receive_slot_remains_owned_until_consumer_releases_it(monkeypatch):
    from aspec.training.data_fetcher import MooncakeDataset, _release_p2p_receive_leases

    dataset = object.__new__(MooncakeDataset)
    dataset.tensor_parallel_group = object()
    dataset.tensor_parallel_control_group = object()
    dataset._tp_source_prefetch = None
    dataset._tensor_parallel_samples = dataset._tensor_parallel_payload_bytes = 0
    dataset._next_queue_item = lambda: "sample"
    receive_buffer = torch.zeros(3)
    ring = {"busy": False, "reads": 0, "releases": 0}

    def release():
        assert ring["busy"]
        ring["busy"] = False
        ring["releases"] += 1

    def load(_item):
        assert not ring["busy"], "next prefetch cannot wait on the active consumer"
        ring["busy"] = True
        ring["reads"] += 1
        receive_buffer.fill_(ring["reads"])
        return {"features": receive_buffer,
                "_mooncake_p2p_receive_lease": SimpleNamespace(release=release)}

    dataset._load_from_mooncake = load
    dataset._read_tp_source = lambda: ("sample", load("sample"))
    monkeypatch.setattr(torch.distributed, "get_global_rank", lambda *_: 0)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda *_: 0)
    def broadcast_control(*_, group, **__):
        assert group is dataset.tensor_parallel_control_group

    monkeypatch.setattr(torch.distributed, "broadcast_object_list", broadcast_control)
    monkeypatch.setattr(torch.distributed, "broadcast", lambda *_, **__: None)
    original_to = torch.Tensor.to

    def to(tensor, *args, **kwargs):
        return tensor if kwargs.get("device") == "cuda" else original_to(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "to", to)
    _, first = dataset._tensor_parallel_item_on_device()
    assert first["features"].tolist() == [1, 1, 1]
    assert ring["busy"] and ring["releases"] == 0
    _release_p2p_receive_leases(first)
    _, second = dataset._tensor_parallel_item_on_device()
    assert second["features"].tolist() == [2, 2, 2]
    _release_p2p_receive_leases(second)
    assert ring == {"busy": False, "reads": 2, "releases": 2}


def test_grouped_qkv_tp_mapping_preserves_independent_hf_projections():
    generator = torch.Generator().manual_seed(1729)
    tensors = {f"model.layers.0.self_attn.{name}_proj.weight": torch.randn(rows, 16, generator=generator)
               for name, rows in (("q", 16), ("k", 8), ("v", 8))}
    reader = SimpleNamespace(tensor=lambda key: tensors[key])
    config = SimpleNamespace(num_attention_heads=4, num_key_value_heads=2, hidden_size=16, head_dim=4)
    inputs = torch.randn(3, 16, generator=generator)
    queries, keys, values = [], [], []
    for rank in range(2):
        mapped = qwen3_parameter(reader, "decoder.layers.0.self_attention.linear_qkv.weight",
                                 config, rank, 2)
        projected = F.linear(inputs, mapped).view(3, 1, 4, 4)
        queries.append(projected[:, :, :2].flatten(1))
        keys.append(projected[:, :, 2].flatten(1))
        values.append(projected[:, :, 3].flatten(1))
    for name, shards in (("q", queries), ("k", keys), ("v", values)):
        torch.testing.assert_close(torch.cat(shards, dim=-1),
                                   F.linear(inputs, tensors[f"model.layers.0.self_attn.{name}_proj.weight"]))


def test_source_prefetch_only_reads_in_background_and_drains_unused_leases():
    from aspec.training.tp_source_prefetch import TPSourcePrefetch

    main_thread = threading.get_ident()
    calls, released = [], []
    second_read = threading.Event()
    values = iter((("one", {"id": 1}), ("two", {"id": 2}), (None, {})))

    def read():
        calls.append(threading.get_ident())
        item = next(values)
        if item[0] == "two":
            second_read.set()
        return item

    prefetch = TPSourcePrefetch(read, lambda item: released.append(item[1]), depth=1)
    assert prefetch.next() == ("one", {"id": 1})
    assert second_read.wait(timeout=2)
    prefetch.close()
    assert calls and all(thread != main_thread for thread in calls)
    assert all(item.get("id") in (None, 2) for item in released)
    assert {"id": 2} in released


def test_hybrid_physical_layers_map_to_qwen3_logical_block():
    tensors = {name: torch.rand(8) for name in (
        "model.layers.9.input_layernorm.weight", "model.layers.9.post_attention_layernorm.weight",
        "model.norm.weight",
    )}
    reader = SimpleNamespace(tensor=tensors.__getitem__)
    for core_name, source_name in (
        ("decoder.layers.18.input_layernorm.weight", "model.layers.9.input_layernorm.weight"),
        ("decoder.layers.19.pre_mlp_layernorm.weight", "model.layers.9.post_attention_layernorm.weight"),
        ("final_norm.weight", "model.norm.weight"),
    ):
        assert qwen3_parameter(reader, core_name, None, 0, 1) is tensors[source_name]


def test_megatron_eval_cache_binds_target_worker_and_topology(tmp_path):
    from aspec.controller.eval import eval_cache_identity

    args = SimpleNamespace(
        target_model_path=str(tmp_path), target_model_backend="megatron",
        megatron_tp_size=1, megatron_implementation="transformer_engine",
        megatron_target_cache_identity={"adapter": "candidate", "core": "core", "dependencies": {}},
    )
    first = eval_cache_identity(args, "sha256:dataset", 8)
    assert first["producer_code"] == args.megatron_target_cache_identity
    args.megatron_tp_size = 2
    assert first != eval_cache_identity(args, "sha256:dataset", 8)
