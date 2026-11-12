# Copyright 2026 Ant Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Draft-owned Ray facade for an independently launched Target service."""
import asyncio
import threading
import uuid

import ray

from aspec.inference.service_protocol import PROTOCOL, engine_arguments, request
from aspec.inference.target_runtime import code_identity


class TargetServiceClient:
    def __init__(self, *, address, args, rank, base_gpu_id, num_gpus_per_engine,
                 node_rank=0, engine_group=0):
        self.address = address
        self.key = uuid.uuid4().hex
        self.timeout = float(getattr(args, "sglang_init_timeout", 1200)) + 60
        self._closed = threading.Event()
        identity = code_identity()
        info = request(address, "/v1/info", timeout=30)
        if info["protocol"] != PROTOCOL or info["identity"] != identity:
            raise RuntimeError("Target service has different ASpec code or protocol")
        request(address, "/v1/engines", {
            "key": self.key, "identity": identity, "arguments": engine_arguments(args),
            "rank": rank, "base_gpu_id": base_gpu_id,
            "num_gpus_per_engine": num_gpus_per_engine, "node_rank": node_rank,
            "engine_group": engine_group,
        }, timeout=self.timeout)
        self._heartbeat = threading.Thread(target=self._keepalive, daemon=True)
        self._heartbeat.start()

    def _keepalive(self):
        while not self._closed.wait(5):
            try:
                request(self.address, "/v1/heartbeat", {"key": self.key}, timeout=5)
            except Exception:
                # Calls fail visibly; a heartbeat never retries a generation request.
                pass

    def _call(self, method, *args, **kwargs):
        return request(self.address, "/v1/call", {
            "key": self.key, "method": method, "args": args, "kwargs": kwargs,
        }, timeout=self.timeout)

    def init(self, **kwargs):
        return self._call("init", **kwargs)

    async def generate(self, *args, **kwargs):
        # ObjectRefs are resolved inside Draft's Ray job, never serialized to Target.
        if isinstance(kwargs.get("input_ids_ref"), ray.ObjectRef):
            kwargs["input_ids_ref"] = await kwargs["input_ids_ref"]
        return await asyncio.to_thread(self._call, "generate", *args, **kwargs)

    def generate_with_decode(self, *args, **kwargs):
        return self._call("generate_with_decode", *args, **kwargs)

    def update_weights_from_disk(self, *args, **kwargs):
        return self._call("update_weights_from_disk", *args, **kwargs)

    def get_node_ip(self):
        return self._call("get_node_ip")

    def find_free_port(self, *args, **kwargs):
        return self._call("find_free_port", *args, **kwargs)

    def get_status(self):
        status = self._call("get_status")
        status["target_service"] = {"address": self.address, "protocol": PROTOCOL}
        return status

    def get_mooncake_config(self):
        from aspec.config.mooncake_config import MooncakeConfig

        value = self._call("get_mooncake_config")
        return MooncakeConfig(**value) if value else None

    def get_feature_contract_status(self):
        return self._call("get_feature_contract_status")

    async def prepare_feature_contract_status(self):
        return await asyncio.to_thread(self._call, "prepare_feature_contract_status")

    def health_check(self, **kwargs):
        return self._call("health_check", **kwargs)

    async def shutdown(self):
        result = await asyncio.to_thread(self._call, "shutdown")
        self._closed.set()
        return result
