# Copyright 2026 Ant Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Opt-in Target worker checks. Set ASPEC_RUN_GPU_INTEGRATION=1 and ASPEC_TARGET_MODEL."""

import asyncio
import json
import os
import socket
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

from aspec.config.mooncake_config import MooncakeConfig
from aspec.feature_contract import build_target_feature_contract
from aspec.inference.engine.megatron_engine import MegatronEngine

pytestmark = pytest.mark.skipif(
    os.getenv("ASPEC_RUN_GPU_INTEGRATION") != "1" or not os.getenv("ASPEC_TARGET_MODEL"),
    reason="requires a local Qwen3 checkpoint, the Megatron Target environment and NVLink GPUs",
)


def _config():
    return MooncakeConfig(local_hostname="127.0.0.1", data_plane="transfer_engine",
                          protocol="nvlink_intra", require_nvlink=True, enable_gpu_direct=True,
                          allow_host_fallback=False, gpu_buffer_size=2 << 20, gpu_ring_slots=2,
                          get_retry_max_wait_seconds=5)


def _gpu_ids(count):
    import torch

    if torch.cuda.device_count() < count:
        pytest.skip(f"requires {count} visible GPUs")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    return visible.split(",")[:count] if visible else list(range(count))


@contextmanager
def _engine(tp, *, context_offset=0, request_offset=0):
    model = Path(os.environ["ASPEC_TARGET_MODEL"])
    config = json.loads((model / "config.json").read_text())
    limit = config["max_position_embeddings"]
    contract = build_target_feature_contract(
        drafter_family="dspark", target_hidden_size=config["hidden_size"],
        target_num_hidden_layers=config["num_hidden_layers"], aux_hidden_state_layers=[0],
        store_last_hidden_states=True, producer_backend="megatron",
    )
    args = SimpleNamespace(target_model_path=str(model), aspec_feature_contract=contract,
                           aspec_context_contract={"backend_context_length": limit + context_offset},
                           max_seq_length=limit + request_offset, aux_hidden_states_layers=[0],
                           store_last_hidden_states=True, attention_backend="cudnn_flex_attention",
                           megatron_implementation=os.getenv("ASPEC_MEGATRON_IMPLEMENTATION", "transformer_engine"))
    engine = MegatronEngine(args, rank=0, num_gpus_per_engine=tp, gpu_ids=_gpu_ids(tp))
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    try:
        engine.init(_config(), pre_allocated_port=port)
        yield engine, limit
    finally:
        # Failed workers may intentionally retain registrations. These tests own
        # all clients and finish fencing them before leaving this context.
        for process in engine._processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=10)
            if process.is_alive():
                process.kill()
                process.join(timeout=10)
        for connection in engine._connections:
            connection.close()


@pytest.mark.parametrize("tp", [1, 2])
@pytest.mark.parametrize("context_offset,request_offset", [(0, 0), (1, 0), (0, 1)])
def test_worker_readiness_at_model_limit(tp, context_offset, request_offset):
    if context_offset or request_offset:
        with pytest.raises(RuntimeError, match="Planned Target context exceeds the loaded model"):
            with _engine(tp, context_offset=context_offset, request_offset=request_offset):
                pytest.fail("worker advertised readiness above the model limit")
        return
    with _engine(tp) as (engine, limit):
        assert len(engine._status) == tp
        assert all(s["max_sequence_length"] == limit and s["tensor_parallel_size"] == tp for s in engine._status)
        assert engine._call("status")["transport"]["ring"]["in_use"] == 0
        asyncio.run(engine.shutdown())


def _transport_worker(audit, draining, connection, rank, gpu_ids, *args):
    """Exercise the worker RPC and real source ring with small known CUDA payloads."""
    from contextlib import ExitStack
    from unittest.mock import patch

    import torch
    import torch.distributed as dist

    from aspec.inference.engine import megatron_engine
    from aspec.inference.target_prefill import TargetPrefillDirectSink
    from aspec.models import megatron
    from aspec.transfer.mooncake.eagle_store import EagleMooncakeStore

    def record(event, value):
        with audit.open("a") as stream:
            stream.write(json.dumps([event, value]) + "\n")

    load = megatron.load_qwen3_target

    def load_with_known_features(*values, **options):
        model, config = load(*values, **options)

        def prefill(ids, layers, capture):
            hidden = ids.view(-1, 1).expand(-1, config.hidden_size).to(torch.bfloat16).contiguous()
            dist.all_reduce(hidden)
            hidden.div_(dist.get_world_size())
            if capture:
                capture(layers[0], hidden, None)
            if ids[0, 0].item() == 98 + rank:
                raise RuntimeError("injected second prefill failure")
            return hidden

        model.prefill = prefill
        return model, config

    class ObservedSink(TargetPrefillDirectSink):
        def commit_request(self, request_id):
            key = super().commit_request(request_id)
            record("commit", request_id)
            return key

        def publish(self):
            super().publish()
            record("publish", self.requests[0].request_id)

        def abort(self):
            before = self.store.get_p2p_status()["ring"]["in_use"]
            super().abort()
            after = self.store.get_p2p_status()["ring"]["in_use"]
            record("abort", [before, after])
            if after == 0:
                with ExitStack() as cleanup:
                    for i in range(2):
                        slot = self.store.reserve_p2p_output(
                            key=f"reuse-{i}", sequence_length=1, hidden_size=32, num_aux_layers=1,
                            store_last_hidden_states=True, data_id=str(i), request_id=str(i), source_engine="0")
                        cleanup.callback(slot.discard)
                    record("reacquired", self.store.get_p2p_status()["ring"]["in_use"])

    wait = EagleMooncakeStore.wait_for_p2p_source_releases

    def observe_drain(store, timeout):
        draining.set()
        return wait(store, timeout=timeout)

    with (patch.object(megatron, "load_qwen3_target", load_with_known_features),
          patch.object(megatron_engine, "TargetPrefillDirectSink", ObservedSink),
          patch.object(EagleMooncakeStore, "wait_for_p2p_source_releases", observe_drain)):
        megatron_engine._target_worker(connection, rank, gpu_ids, *args)


@pytest.fixture
def observed_transport(tmp_path, monkeypatch):
    from functools import partial

    import torch.multiprocessing as mp

    from aspec.inference.engine import megatron_engine

    audit = tmp_path / "ring.jsonl"
    draining = mp.get_context("spawn").Event()
    # Each spawned child imports the original worker from the module afresh.
    monkeypatch.setattr(megatron_engine, "_target_worker", partial(_transport_worker, audit, draining))
    monkeypatch.setenv("ASPEC_RUN_ID", "target-test")
    monkeypatch.setenv("ASPEC_ATTEMPT_ID", "target-test-attempt")
    return audit, draining


@pytest.mark.parametrize("failing_rank", [0, 1], ids=["publisher", "follower"])
def test_tp2_failed_rpc_returns_no_keys_and_releases_slots(observed_transport, failing_rank):
    import torch

    audit, _ = observed_transport
    with _engine(2) as (engine, _):
        engine.args.megatron_init_timeout = 30  # Bound the RPC after checkpoint loading.
        with pytest.raises(RuntimeError):
            asyncio.run(engine.generate(["A", "B"], input_ids_ref=[torch.tensor([7]), torch.tensor([98 + failing_rank])],
                                        request_ids=["A", "B"]))
        if failing_rank == 1:
            # The facade receives the publishing rank's rollback error first.
            assert engine._connections[1].poll(5)
            assert "injected second prefill failure" in engine._connections[1].recv()["error"]
        assert engine._processes[0].is_alive()
        events = [json.loads(line) for line in audit.read_text().splitlines()]
        committed = [["commit", "A"], ["commit", "B"]] if failing_rank == 1 else [["commit", "A"]]
        assert events == committed + [["abort", [2, 1]], ["abort", [1, 0]], ["reacquired", 2]]


def _read_on_second_gpu(connection):
    import torch

    from aspec.transfer.mooncake.p2p import MooncakeP2PDescriptor, MooncakeP2PTransfer

    torch.cuda.set_device(1)
    transfer = MooncakeP2PTransfer(_config())
    try:
        transfer.setup(torch.device("cuda", 1), role="destination")
        connection.send("ready")
        result = transfer.read(MooncakeP2PDescriptor.from_key(connection.recv()))
        try:
            ids = result.tensors["input_ids"]
            assert ids.tolist() == [7, 11]
            for name in ("hidden_states", "last_hidden_states"):
                features = result.tensors[name]
                torch.testing.assert_close(features, ids[:, None].expand_as(features).to(features.dtype))
        finally:
            result.lease.release()
        connection.send("read")
        assert connection.recv() == "exit"
    finally:
        transfer.close()
        connection.close()


@pytest.mark.parametrize("timeout", [False, True], ids=["active-release", "timeout-fence"])
def test_shutdown_keeps_published_source_readable(observed_transport, timeout):
    from concurrent.futures import ThreadPoolExecutor

    import psutil
    import torch

    from aspec.inference.service import readers_exited, stop_owned_processes

    _gpu_ids(2)
    _, draining = observed_transport
    ctx = torch.multiprocessing.get_context("spawn")
    # The active-release case also exercises a successful TP2 publication.
    with _engine(1 if timeout else 2) as (engine, _):
        client, child = ctx.Pipe()
        reader = ctx.Process(target=_read_on_second_gpu, args=(child,))
        reader.start()
        child.close()
        try:
            assert client.poll(120) and client.recv() == "ready"
            owners = [{"node_ip": "127.0.0.1", "pid": reader.pid, "created": psutil.Process(reader.pid).create_time()}]
            source = engine._processes[0]
            identity = {source.pid: psutil.Process(source.pid).create_time()}
            outputs = asyncio.run(engine.generate("sample", input_ids_ref=[torch.tensor([7, 11])]))
            assert engine._call("status")["transport"]["ring"]["in_use"] == 1
            with ThreadPoolExecutor(max_workers=1) as executor:
                shutdown = executor.submit(asyncio.run, engine.shutdown())
                assert draining.wait(timeout=30)
                if timeout:
                    with pytest.raises(RuntimeError, match="source leases still have active readers"):
                        shutdown.result(timeout=30)
                else:
                    assert not shutdown.done()
                assert source.is_alive() and not readers_exited(owners, "127.0.0.1")
                client.send(outputs[0]["mooncake_key"])
                assert client.poll(30) and client.recv() == "read"
                if not timeout:
                    shutdown.result(timeout=30)
            client.send("exit")
            reader.join(timeout=15)
            assert reader.exitcode == 0 and readers_exited(owners, "127.0.0.1")
            if timeout:
                assert source.is_alive()
                stop_owned_processes(identity)
                source.join(timeout=10)
            assert not source.is_alive()
        finally:
            if reader.is_alive():
                reader.kill()
            reader.join(timeout=10)
            client.close()
