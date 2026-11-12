# Copyright 2026 Ant Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Isolation regressions; real Ray/venv and GPU checks run on Linux separately."""
import subprocess
import sys

import pytest


def test_factory_import_does_not_import_sglang():
    result = subprocess.run([sys.executable, "-c", """
import sys
class Reject:
    def find_spec(self, fullname, *args):
        if fullname == 'sglang' or fullname.startswith('sglang.'):
            raise AssertionError('Draft imported SGLang')
sys.meta_path.insert(0, Reject())
import aspec.inference.factory
import aspec.inference.engine.target_actor
"""], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_target_bootstrap_rebuilds_environment_before_starting_worker(monkeypatch):
    import os

    from aspec import _target_worker

    for name in ("PYTHONPATH", "PYTHONHOME", "LD_LIBRARY_PATH", "LIBRARY_PATH"):
        monkeypatch.setenv(name, "/draft/foreign")
    monkeypatch.setenv("VIRTUAL_ENV", "/draft")
    monkeypatch.setenv("PATH", "/draft/bin:/usr/bin")
    monkeypatch.setenv("ASPEC_CACHE_HOME", "/tmp/aspec-cache")
    monkeypatch.setattr(sys, "argv", ["bootstrap", "ray-worker.py", "--node-id", "123"])
    calls = []
    monkeypatch.setattr(_target_worker, "execute", calls.append)
    _target_worker.main()
    assert calls == [[sys.executable, "ray-worker.py", "--node-id", "123"]]
    assert os.environ["VIRTUAL_ENV"] == sys.prefix
    assert os.environ["PATH"].startswith(os.path.dirname(sys.executable) + os.pathsep)
    assert os.environ["ASPEC_CACHE_HOME"] == "/tmp/aspec-cache/target"
    assert not any(name in os.environ for name in ("PYTHONPATH", "PYTHONHOME", "LD_LIBRARY_PATH", "LIBRARY_PATH"))


def test_target_rejects_different_code_before_loading_engine(monkeypatch):
    from aspec.inference.engine.target_actor import TargetActor

    with pytest.raises(RuntimeError, match="code differs"):
        TargetActor(expected_runtime={"version": "different", "code_sha256": "different"})


@pytest.mark.parametrize("command", ["evaluate", "prepare-target-data", "patch-sglang"])
def test_target_cli_commands_run_in_the_active_environment(monkeypatch, command):
    from aspec import cli

    def missing(_):
        raise cli.importlib.metadata.PackageNotFoundError("ling-aspec-target")

    monkeypatch.setattr(cli.importlib.metadata, "version", missing)
    monkeypatch.setenv("ASPEC_TARGET_PYTHON", "/target/bin/python")
    monkeypatch.setattr(sys, "argv", ["aspec", command])
    calls = []
    monkeypatch.setattr(cli, "execute", calls.append)
    cli.main()
    assert calls[0][:2] == [sys.executable, "-m"]


def test_target_address_is_recorded_in_plan_without_opposing_interpreter(monkeypatch):
    from aspec.commands.quickstart import build_profile
    from aspec.config.train_config import load_config
    from aspec.planning import compile_execution_plan

    monkeypatch.setenv("ASPEC_TARGET_PYTHON", "/environment-target/bin/python")
    config = build_profile("qwen3-8b")
    config.inference.target_service = {"address": "http://target:30000"}
    plan = compile_execution_plan(load_config(base_config=config))
    assert plan.to_runtime_args().target_service["address"] == "http://target:30000"
    inference = plan.to_dict()["policy"]["resources"]["inference"]
    assert "python_executable" not in inference
    assert inference["target_service"]["mode"] == "external"
