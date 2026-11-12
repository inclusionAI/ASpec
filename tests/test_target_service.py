# Copyright 2026 Ant Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Control protocol, source ownership and independent launch regressions."""
import os
import threading
from argparse import Namespace
from types import SimpleNamespace
from unittest.mock import MagicMock

import psutil
import pytest
import torch

from aspec.inference.service import TargetService, http_server, readers_exited
from aspec.inference.service_protocol import (
    decode,
    encode,
    engine_arguments,
    request,
    validate_address,
)


def test_control_payload_roundtrip_and_feature_tensor_rejection():
    payload = {"input_ids_ref": [torch.tensor([1, 3, 5])],
               "result": {"tensor_dtypes": {"hidden_states": torch.bfloat16}}}
    result = decode(encode(payload))
    assert result["input_ids_ref"][0].tolist() == [1, 3, 5]
    assert result["result"]["tensor_dtypes"]["hidden_states"] == torch.bfloat16
    with pytest.raises(TypeError, match="Only CPU token IDs"):
        encode(torch.ones(3, dtype=torch.bfloat16))
    with pytest.raises(ValueError, match="dtype"):
        decode({"$dtype": "not_a_dtype"})


@pytest.mark.parametrize("address", ["file:///tmp/a", "https://target:30000", "http://target",
                                    "http://user:secret@target:30000", "http://target:30000/run"])
def test_invalid_target_address_is_rejected(address):
    with pytest.raises(ValueError):
        validate_address(address)


def test_no_driver_objects_cross_control_boundary():
    args = Namespace(model="qwen", draft_model_config_obj=object(), _mooncake_master_actor=object())
    assert engine_arguments(args) == {"model": "qwen"}
    args.mystery = object()
    with pytest.raises(TypeError, match="mystery"):
        engine_arguments(args)


def test_complete_training_plan_has_an_explicit_data_only_target_boundary():
    import json

    from aspec.commands.quickstart import build_profile
    from aspec.config.train_config import load_config
    from aspec.planning import compile_execution_plan

    args = compile_execution_plan(load_config(base_config=build_profile("qwen3-8b"))).to_runtime_args()
    payload = engine_arguments(args)
    json.dumps(payload, allow_nan=False)
    assert payload["aspec_feature_contract"]["contract_id"]
    assert payload["target_service"]["mode"] == "external"


@pytest.fixture
def service(monkeypatch):
    import aspec.inference.service as module

    handle = MagicMock()
    handle.get_status.remote.return_value = {"initialized": True}
    handle.shutdown.remote.return_value = None
    actor = MagicMock()
    actor.options.return_value.remote.return_value = handle
    import ray

    monkeypatch.setattr(module, "ray", SimpleNamespace(remote=lambda _: actor,
                                                     get=lambda x, **_: x, kill=MagicMock(),
                                                     exceptions=ray.exceptions))
    instance = TargetService(model="qwen", devices=[0, 1], runtime={})
    instance.test_handle = handle
    return instance


def creation(service, key="engine", gpu=0):
    return {"key": key, "identity": service.identity,
            "arguments": {"target_model_path": "qwen", "aspec_attempt_id": "attempt-1"},
            "rank": 0, "base_gpu_id": gpu, "num_gpus_per_engine": 1,
            "node_rank": 0, "engine_group": 0}


def test_gpu_ownership_and_closed_engine_idempotence(service):
    service.create(creation(service))
    with pytest.raises(RuntimeError, match="already belongs"):
        service.create(creation(service, "other"))
    service.call("engine", "shutdown", [], {})
    service.call("engine", "shutdown", [], {})
    assert service.test_handle.shutdown.remote.call_count == 1
    assert service.idle()
    service.create(creation(service, "next"))


def test_model_identity_and_allowed_gpu_fail_before_creation(service):
    wrong = creation(service)
    wrong["identity"] = {}
    with pytest.raises(ValueError, match="identity"):
        service.create(wrong)
    wrong = creation(service)
    wrong["arguments"]["target_model_path"] = "other-model"
    with pytest.raises(ValueError, match="model"):
        service.create(wrong)
    with pytest.raises(ValueError, match="GPU set"):
        service.create(creation(service, gpu=3))
    assert service.idle()


def test_disconnect_retains_unverified_reader_storage(service, monkeypatch):
    service.create(creation(service))
    service.reap_disconnected(expiry=-1)
    assert service.engines["engine"]["state"] == "retained"
    assert not service.idle()
    service.test_handle.shutdown.remote.assert_not_called()
    monkeypatch.setattr("aspec.inference.service.readers_exited", lambda *_: True)
    service.reap_disconnected(expiry=-1)
    assert service.idle()


def test_failed_source_actor_is_retired_only_after_reader_fence(service, monkeypatch):
    import ray

    service.create(creation(service))
    item = service.engines["engine"]
    item["processes"] = {999999999: 1.0}
    service.test_handle.shutdown.remote.side_effect = ray.exceptions.RayActorError("actor exited")
    monkeypatch.setattr("aspec.inference.service.readers_exited", lambda *_: False)
    stop = MagicMock()
    monkeypatch.setattr("aspec.inference.service.stop_owned_processes", stop)
    service.reap_disconnected(expiry=-1)
    stop.assert_not_called()
    assert not service.idle()
    monkeypatch.setattr("aspec.inference.service.readers_exited", lambda *_: True)
    service.reap_disconnected(expiry=-1)
    stop.assert_called_once_with(item["processes"])
    assert service.idle()
    assert item["terminal_reason"] == "actor_failed_readers_fenced_sources_exited"


def test_shutdown_does_not_race_active_generation(service):
    service.create(creation(service))
    service.engines["engine"]["active"] = 1
    with pytest.raises(RuntimeError, match="active calls"):
        service.call("engine", "shutdown", [], {})
    service.test_handle.shutdown.remote.assert_not_called()


def test_reader_fence_requires_process_identity_and_local_observation():
    owner = {"pid": os.getpid(), "created": psutil.Process().create_time(), "node_ip": "local"}
    assert not readers_exited([owner], "local")
    assert not readers_exited([owner], "another-host")
    assert not readers_exited([], "local")
    assert readers_exited([{**owner, "created": owner["created"] - 100}], "local")


def test_real_http_control_roundtrip_and_unknown_method(service):
    server = http_server(service, "127.0.0.1", 0)
    worker = threading.Thread(target=server.serve_forever)
    worker.start()
    address = f"http://127.0.0.1:{server.server_port}"
    try:
        assert request(address, "/v1/info")["identity"] == service.identity
        request(address, "/v1/engines", creation(service))
        assert request(address, "/v1/call", {"key": "engine", "method": "get_status",
                                             "args": [], "kwargs": {}}) == {"initialized": True}
        with pytest.raises(RuntimeError, match="Unknown Target control method"):
            request(address, "/v1/call", {"key": "engine", "method": "__dict__",
                                          "args": [], "kwargs": {}})
    finally:
        server.shutdown()
        server.server_close()
        worker.join()


def test_managed_local_keeps_resume_configuration(tmp_path):
    from aspec.commands.quickstart import build_profile
    from aspec.config.train_config import load_config

    config = build_profile("qwen3-8b")
    config.inference.target_service = {"mode": "managed-local", "python_executable": "/target/bin/python"}
    config.training.load_path = str(tmp_path / "checkpoint")
    loaded = load_config(base_config=config)
    assert loaded.training.load_path == str(tmp_path / "checkpoint")


def test_managed_supervisor_never_stops_an_existing_service(monkeypatch, tmp_path):
    from aspec.inference import managed

    monkeypatch.setattr(managed, "request", lambda *_a, **_k: {})
    popen = MagicMock()
    monkeypatch.setattr(managed.subprocess, "Popen", popen)
    with pytest.raises(RuntimeError, match="existing Target"):
        with managed.managed_target(Namespace(target_service={"mode": "managed-local",
                                                              "address": "http://127.0.0.1:30000"})):
            pytest.fail("existing service was adopted")
    popen.assert_not_called()
