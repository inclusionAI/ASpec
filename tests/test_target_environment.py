# Copyright 2026 Ant Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Isolation regressions; real Ray/venv and GPU checks run on Linux separately."""
import os
import subprocess
import sys
from argparse import Namespace
from unittest.mock import MagicMock, patch

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


@pytest.mark.parametrize("interpreter", [None, "relative/bin/python"])
def test_target_interpreter_must_be_explicit_and_absolute(monkeypatch, interpreter):
    from aspec.inference.factory import _prepare_sgl_engines

    monkeypatch.delenv("ASPEC_TARGET_PYTHON", raising=False)
    args = Namespace(target_python=interpreter, inference_num_gpus=1,
                     inference_num_gpus_per_engine=1)
    with pytest.raises(ValueError, match="absolute"):
        _prepare_sgl_engines(args, None)


def test_target_actor_selects_its_interpreter_without_draft_loader_paths(monkeypatch):
    from aspec.inference.factory import _prepare_sgl_engines

    args = Namespace(target_python="/target env/bin/python", inference_num_gpus=1,
                     inference_num_gpus_per_engine=1, sglang_nnodes=1)
    remote_class = MagicMock()
    with (patch("aspec.inference.factory.ray") as ray,
          patch("aspec.inference.factory.get_aspec_env_vars", return_value={
              "LD_LIBRARY_PATH": "/draft/lib", "PYTHONPATH": "/draft/source",
              "ASPEC_CUDA_COMPAT_DIR": "/driver/compat"})):
        ray.remote.return_value = remote_class
        ray.get.return_value = 25000
        _prepare_sgl_engines(args, (MagicMock(), [0], [0]))
    environment = remote_class.options.call_args.kwargs["runtime_env"]
    assert environment["py_executable"] == "'/target env/bin/python' -m aspec._target_worker"
    assert environment["env_vars"] == {"ASPEC_CUDA_COMPAT_DIR": "/driver/compat"}
    assert "code_sha256" in remote_class.options.return_value.remote.call_args.kwargs["expected_runtime"]


def test_target_bootstrap_rebuilds_environment_before_starting_worker(monkeypatch):
    from aspec import _target_worker

    for name in ("PYTHONPATH", "PYTHONHOME", "LD_LIBRARY_PATH"):
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
    assert not any(name in os.environ for name in ("PYTHONPATH", "PYTHONHOME", "LD_LIBRARY_PATH"))


def test_target_rejects_different_code_before_loading_engine(monkeypatch):
    from aspec.inference.engine.target_actor import TargetActor

    with pytest.raises(RuntimeError, match="code differs"):
        TargetActor(expected_runtime={"version": "different", "code_sha256": "different"})


@pytest.mark.parametrize("command", ["evaluate", "prepare-target-data", "patch-sglang"])
def test_target_cli_commands_use_the_selected_target_interpreter(monkeypatch, command):
    from aspec import cli

    def missing(_):
        raise cli.importlib.metadata.PackageNotFoundError("ling-aspec-target")

    monkeypatch.setattr(cli.importlib.metadata, "version", missing)
    monkeypatch.setenv("ASPEC_TARGET_PYTHON", "/target/bin/python")
    monkeypatch.setattr(sys, "argv", ["aspec", command])
    calls = []
    monkeypatch.setattr(cli, "execute", calls.append)
    cli.main()
    assert calls[0][:4] == ["/target/bin/python", "-m", "aspec._target_worker", "-m"]


@pytest.mark.parametrize("explicit", [None, "/configured-target/bin/python"])
def test_target_interpreter_selection_is_recorded_in_plan(monkeypatch, explicit):
    from aspec.commands.quickstart import build_profile
    from aspec.config.train_config import load_config
    from aspec.planning import compile_execution_plan

    monkeypatch.setenv("ASPEC_TARGET_PYTHON", "/environment-target/bin/python")
    config = build_profile("qwen3-8b")
    config.inference.target_python = explicit
    plan = compile_execution_plan(load_config(base_config=config))
    expected = explicit or "/environment-target/bin/python"
    assert plan.to_runtime_args().target_python == expected
    assert plan.to_dict()["topology"]["inference"]["python_executable"] == expected
