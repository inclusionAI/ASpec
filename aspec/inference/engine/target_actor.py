# Copyright 2026 Ant Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Lazy Target boundary: the submitting Draft process never imports SGLang.

Inspired by SpecForge 629092ad's separate capture-server interpreter. Adapted
to ASpec's retained Ray actor contract so placement, storage and cancellation
continue to use the existing SglEngine rather than a second serving pipeline.
"""

import importlib.metadata
import sys
from pathlib import Path


class TargetActor:
    def __init__(self, *args, **kwargs):
        from aspec.commands.doctor import inspect_environment
        from aspec.inference.target_runtime import code_identity

        expected = kwargs.pop("expected_runtime")
        self.runtime_identity = code_identity()
        if self.runtime_identity != expected:
            raise RuntimeError(f"Draft/Target ASpec code differs: {expected} != {self.runtime_identity}")

        result = inspect_environment(target=True)
        if not result["ok"]:
            raise RuntimeError("Target environment is not ready: " + "; ".join(result["errors"]))
        from aspec.inference.engine.sgl_engine import SglEngine

        self.engine = SglEngine(*args, **kwargs)

    def init(self, **kwargs):
        return self.engine.init(**kwargs)

    async def generate(self, *args, **kwargs):
        return await self.engine.generate(*args, **kwargs)

    def generate_with_decode(self, *args, **kwargs):
        return self.engine.generate_with_decode(*args, **kwargs)

    def update_weights_from_disk(self, *args, **kwargs):
        return self.engine.update_weights_from_disk(*args, **kwargs)

    def get_node_ip(self):
        return self.engine.get_node_ip()

    def find_free_port(self, *args, **kwargs):
        return self.engine.find_free_port(*args, **kwargs)

    def get_status(self):
        import psutil
        import ray
        import sglang
        import torch

        import aspec
        from aspec.runtime_identity import SGLANG_RUNTIME_RELATIVE_PATHS, module_runtime_identity

        backend = module_runtime_identity("sglang", relative_paths=SGLANG_RUNTIME_RELATIVE_PATHS)
        backend_identity = {
            "path": backend["path"], "git_revision": None, "git_dirty": None,
            "effective_tree_sha256": backend["effective_tree_sha256"],
            "source_file_count": backend["file_count"], "missing_files": backend["missing_files"],
        }
        result = self.engine.get_status()
        result["created"] = psutil.Process().create_time()
        result["runtime"] = {
            **self.runtime_identity,
            "python": sys.executable, "prefix": sys.prefix,
            "aspec": str(Path(aspec.__file__).resolve()),
            "ray": str(Path(ray.__file__).resolve()),
            "sglang": str(Path(sglang.__file__).resolve()),
            "backend_identity": backend_identity,
            "sglang_version": importlib.metadata.version("sglang"),
            "torch": torch.__version__, "cuda": torch.version.cuda,
            "cutlass": importlib.metadata.version("nvidia-cutlass-dsl"),
            "quack": importlib.metadata.version("quack-kernels"),
        }
        return result

    def get_mooncake_config(self):
        return self.engine.get_mooncake_config()

    def get_feature_contract_status(self):
        return self.engine.get_feature_contract_status()

    async def prepare_feature_contract_status(self):
        return await self.engine.prepare_feature_contract_status()

    def health_check(self, **kwargs):
        return self.engine.health_check(**kwargs)

    async def shutdown(self):
        return await self.engine.shutdown()
