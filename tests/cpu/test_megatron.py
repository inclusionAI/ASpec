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
from aspec.planning import PlanCompileError, compile_execution_plan


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


@pytest.mark.parametrize("alpha", [0.0, 1.0])
def test_megatron_freezes_disabled_confidence_head(monkeypatch, alpha):
    from aspec.training.dspark_trainer import DSparkTrainer
    from aspec.training.megatron_trainer import MegatronDSparkTrainer

    trainer = object.__new__(MegatronDSparkTrainer)
    trainer.confidence_head_alpha = alpha
    draft = torch.nn.Module()
    draft.confidence_head = torch.nn.Linear(2, 1)
    draft.markov_head = torch.nn.Linear(2, 2)

    def build_wrapper(_trainer, model):
        assert all(p.requires_grad == (alpha > 0) for p in model.confidence_head.parameters())
        assert all(p.requires_grad for p in model.markov_head.parameters())
        return SimpleNamespace(_sample_anchor_positions=lambda: None)

    monkeypatch.setattr(DSparkTrainer, "_build_training_wrapper", build_wrapper)
    trainer._build_training_wrapper(draft)


@pytest.mark.parametrize("backend", ["fa4", "cudnn_flex_attention"])
def test_packed_megatron_draft_requires_thd_and_document_anchors(backend):
    args = settings(draft_tensor_parallel_size=1, dflash_packing=True,
                    attention_backend=backend, anchor_sampling_scope="sample",
                    prefetch_depth=1, mooncake_destination_gpu_ring_slots=2)
    validate_megatron_execution(args, {"family": "dspark"})
    args.anchor_sampling_scope = "row"
    with pytest.raises(ValueError, match="per-sample"):
        validate_megatron_execution(args, {"family": "dspark"})
    args.anchor_sampling_scope = "sample"
    args.attention_backend = "flex_attention"
    with pytest.raises(ValueError, match="THD"):
        validate_megatron_execution(args, {"family": "dspark"})


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

    # A second batch can stage two samples while the first leaves one ready.
    # With only three slots, the next complete Draft dispatch cannot be fed.
    config = load_config(base_config=config, cli_args=[
        "inference.inference_batch_size=3", "mooncake.source_gpu_ring_slots=4",
    ])
    assert compile_execution_plan(config).to_runtime_args().dp_size == 2
    with pytest.raises(ValueError, match="capacity=1"):
        load_config(base_config=config, cli_args=["mooncake.source_gpu_ring_slots=3"])
    config.mooncake.source_gpu_ring_slots = 3
    with pytest.raises(PlanCompileError, match="capacity=1"):
        compile_execution_plan(config)


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
    monkeypatch.setattr(torch.distributed, "all_reduce", lambda *_, **__: None)

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


@pytest.mark.parametrize("failure", ["prefill", "validate", "peer", None])
def test_target_batch_defers_publication_until_all_workers_finish(failure):
    from unittest.mock import Mock

    from aspec.inference.engine.megatron_engine import _prefill_batch

    first, second = Mock(), Mock()
    calls = []

    def synchronize():
        calls.append("ready")
        if failure == "peer":
            raise ValueError("peer")

    def generate():
        with _prefill_batch(synchronize) as stage:
            stage(first)
            stage(second)
            if failure == "prefill":
                raise ValueError("prefill")
            if failure == "validate":
                second.validate_publishable.side_effect = ValueError("validate")
            first.publish.side_effect = lambda: calls.append("published")

    if failure:
        with pytest.raises(ValueError, match=failure):
            generate()
        for sink in (first, second):
            sink.publish.assert_not_called()
            sink.abort.assert_called_once()
    else:
        generate()
        assert calls == ["ready", "published"]
        second.publish.assert_called_once()


@pytest.mark.parametrize("released", [False, True])
def test_target_closes_source_only_after_release(released):
    from unittest.mock import Mock

    from aspec.inference.engine.megatron_engine import _close_source

    store = Mock()
    store.config.get_retry_max_wait_seconds = 7
    store.wait_for_p2p_source_releases.return_value = released
    if released:
        _close_source(store)
        store.close.assert_called_once()
    else:
        with pytest.raises(TimeoutError, match="active readers"):
            _close_source(store)
        store.close.assert_not_called()
    store.wait_for_p2p_source_releases.assert_called_once_with(timeout=7)


def test_bitwise_equality_and_final_reference_policy(tmp_path):
    from tools.validation.megatron_torch_alignment import compare_vectors, reference_state_path

    reference = {"weight": torch.tensor([0.0])}
    result = compare_vectors({"weight": torch.tensor([-0.0])}, reference, device="cpu")
    assert result["max_absolute"] == 0 and not result["bitwise_equal"]
    (tmp_path / "completed.json").touch()
    milestone = tmp_path / "logical-state-step4.pt"
    milestone.touch()
    assert reference_state_path(tmp_path, 4, 4, 0) == milestone
    final = tmp_path / "final-logical.pt"
    final.touch()
    assert reference_state_path(tmp_path, 4, 4, 0) == final


@pytest.mark.parametrize("prefetch_factor", [0, 2])
def test_tp_input_rejects_packing_before_setup(prefetch_factor):
    from aspec.training.data_fetcher import MooncakeDataset

    with pytest.raises(ValueError, match="TP input cannot be combined with packing"):
        MooncakeDataset(
            ray_queue=None, mooncake_store=None, device=torch.device("cpu"),
            tensor_parallel_group=object(), tensor_parallel_control_group=object(),
            packing_enabled=True, max_seq_length=16, prefetch_factor=prefetch_factor,
        )


def test_tp_preparation_error_reaches_both_ranks(tmp_path):
    from tests.tp_input import tp_input_worker

    torch.multiprocessing.spawn(tp_input_worker, args=((tmp_path / "rdzv").as_uri(), False), nprocs=2)


def test_rotary_source_change_invalidates_megatron_eval_cache(tmp_path, monkeypatch):
    from unittest.mock import Mock

    from aspec.controller import eval as evaluation
    from aspec.runtime_identity import MEGATRON_RUNTIME_RELATIVE_PATHS, module_runtime_identity

    core = tmp_path / "fixture_core"
    core.mkdir()
    (core / "__init__.py").touch()
    for relative in MEGATRON_RUNTIME_RELATIVE_PATHS:
        path = core / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# original source\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setattr(evaluation.ray, "get", lambda value: value)
    controller, group = Mock(), Mock()
    controller.get_dataset_admission_evidence.remote.return_value = {"eval": {"dataset_fingerprint": "sha256:data"}}
    cached = set()
    group.load_eval_cache.side_effect = lambda path, identity: [int(path in cached)]
    args = SimpleNamespace(target_model_path=str(tmp_path), target_model_backend="megatron",
                           eval_interval=1, checkpoint_dir=None, cache_dir=str(tmp_path / "cache"),
                           dp_size=1, inference_batch_size=1, megatron_tp_size=1,
                           megatron_implementation="transformer_engine")

    def lookup():
        digest = module_runtime_identity(core.name, relative_paths=MEGATRON_RUNTIME_RELATIVE_PATHS)
        assert not digest["missing_files"]
        args.megatron_target_cache_identity = {"core": digest["effective_tree_sha256"],
                                               "dependencies": {"megatron-core": "0.20.0"}}
        return evaluation.setup_eval(controller, group, args, eval_dataset_size=1)

    initial = lookup()
    cached.add(initial.eval_cache_path)
    assert lookup().eval_cache_loaded
    (core / "models/common/embeddings/rotary_pos_embedding.py").write_text("# changed source\n")
    changed = lookup()
    assert not changed.eval_cache_loaded
    assert changed.eval_cache_path != initial.eval_cache_path
