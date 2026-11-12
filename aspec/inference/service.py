# Copyright 2026 Ant Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Standalone Target control service around the retained SglEngine.

Inspired by SpecForge f8cc3dc external capture. JSON control messages replace
cross-environment Ray calls; ASpec's GPU leases, feature contracts and tensor
transport remain in use. No SpecForge server code is copied.
"""
import argparse
import json
import logging
import os
import signal
import tempfile
import threading
import time
import uuid
from argparse import Namespace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

import ray

from aspec.inference.service_protocol import (
    MAX_BODY,
    METHODS,
    PROTOCOL,
    decode,
    encode,
    validate_address,
)
from aspec.inference.target_runtime import code_identity, inventory
from aspec.utils.env import get_aspec_env_vars

logger = logging.getLogger(__name__)


def readers_exited(owners, local_ip):
    """Only observe local processes; unknown/remote owners retain source storage."""
    import psutil

    if not owners:
        return False
    for owner in owners:
        if (owner.get("node_ip") != local_ip or not isinstance(owner.get("pid"), int)
                or not isinstance(owner.get("created"), (int, float))):
            return False
        try:
            process = psutil.Process(owner["pid"])
            if (process.create_time() == owner["created"]
                    and process.status() not in (psutil.STATUS_ZOMBIE, psutil.STATUS_DEAD)):
                return False
        except psutil.NoSuchProcess:
            pass
    return True


class TargetService:
    def __init__(self, *, model=None, devices=None, runtime=None):
        self.runtime = runtime if runtime is not None else inventory()
        self.identity = code_identity()
        self.model = model
        self.devices = None if devices is None else frozenset(devices)
        self.server_id = uuid.uuid4().hex
        self.lock = threading.RLock()
        self.engines = {}
        self.claims = {}
        self.draining = False

    def info(self):
        with self.lock:
            return {"protocol": PROTOCOL, "identity": self.identity, "runtime": self.runtime,
                    "pid": os.getpid(),
                    "server_id": self.server_id, "model": self.model,
                    "devices": sorted(self.devices) if self.devices is not None else None,
                    "draining": self.draining,
                    "engines": {key: {k: item[k] for k in ("state", "session", "gpus")}
                                for key, item in self.engines.items()}}

    def create(self, value):
        from aspec.inference.engine.target_actor import TargetActor

        if value["identity"] != self.identity:
            raise ValueError("Draft/Target ASpec code identity differs")
        args = Namespace(**decode(value["arguments"]))
        if self.model and args.target_model_path != self.model:
            raise ValueError("Target model differs from the service configuration")
        session = getattr(args, "aspec_attempt_id", None)
        run_id = getattr(args, "aspec_run_id", None)
        if not isinstance(session, str) or not session or not isinstance(run_id, str) or not run_id:
            raise ValueError("Target requires explicit run and attempt identities")
        base, count = value["base_gpu_id"], value["num_gpus_per_engine"]
        if type(base) is not int or type(count) is not int or base < 0 or count <= 0:
            raise ValueError("Invalid Target GPU allocation")
        gpus = list(range(base, base + count))
        key = value["key"]
        with self.lock:
            if self.draining:
                raise RuntimeError("Target service is draining")
            if key in self.engines:
                raise ValueError("Target engine identity already exists; creation is not replayed")
            if self.devices is not None and not set(gpus).issubset(self.devices):
                raise ValueError("Target allocation is outside the service GPU set")
            if any(gpu in self.claims for gpu in gpus):
                raise RuntimeError("Target GPU already belongs to another engine/session")
            for old_key in list(self.engines):
                if len(self.engines) < 128:
                    break
                if self.engines[old_key]["state"] == "closed":
                    del self.engines[old_key]
            # Draft's placement group reserves these physical GPUs. This private
            # Ray instance only orchestrates Target processes; it owns no second
            # GPU allocator and cannot schedule an overlapping training worker.
            environment = get_aspec_env_vars(args)
            environment.update(session_environment(args))
            actor = ray.remote(TargetActor).options(
                num_cpus=0.2, num_gpus=0, runtime_env={"env_vars": environment},
            ).remote(expected_runtime=self.identity, args=args, rank=value["rank"],
                     base_gpu_id=base, num_gpus_per_engine=count,
                     node_rank=value["node_rank"], engine_group=value["engine_group"])
            self.engines[key] = {"actor": actor, "state": "created", "session": session,
                                 "gpus": gpus, "heartbeat": time.monotonic(),
                                 "owners": getattr(args, "aspec_read_owners", None),
                                 "status": {}, "features": {}, "active": 1, "processes": {}}
            self.claims.update({gpu: key for gpu in gpus})
        try:
            status = ray.get(actor.get_status.remote(), timeout=120)
            with self.lock:
                self.engines[key]["status"] = status
                if status.get("pid") and status.get("created"):
                    self.engines[key]["processes"][status["pid"]] = status["created"]
        except BaseException:
            # SglEngine's constructor loads no model or source buffers. Failure
            # here precedes init/publication, so this creation can be rolled back.
            ray.kill(actor, no_restart=True)
            with self.lock:
                self.engines[key]["state"] = "closed"
                for gpu in gpus:
                    self.claims.pop(gpu)
            raise
        finally:
            with self.lock:
                self.engines[key]["active"] -= 1
        return {"key": key}

    def heartbeat(self, key):
        with self.lock:
            item = self.engines[key]
            if item["state"] in ("retained", "closing"):
                raise RuntimeError("Target session has stopped admitting work")
            item["heartbeat"] = time.monotonic()
        return None

    def call(self, key, method, args, kwargs):
        if method not in METHODS:
            raise ValueError("Unknown Target control method")
        with self.lock:
            item = self.engines[key]
            if item["state"] == "closed":
                if method == "shutdown":
                    return None
                if method == "get_status":
                    return {**item["status"], "initialized": False}
                if method in ("get_feature_contract_status", "prepare_feature_contract_status"):
                    return item["features"]
                raise RuntimeError("Target engine is closed")
            if item["state"] in ("retained", "closing") and method != "shutdown":
                raise RuntimeError("Target engine is retained pending safe cleanup")
            if method == "shutdown":
                if item["active"]:
                    raise RuntimeError("Target still has active calls; drain before shutdown")
                item["state"] = "closing"
            item["heartbeat"] = time.monotonic()
            item["active"] += 1
        try:
            if method == "init" and kwargs.get("mooncake_config") is not None:
                from aspec.config.mooncake_config import MooncakeConfig

                kwargs["mooncake_config"] = MooncakeConfig(**kwargs["mooncake_config"])
            result = ray.get(getattr(item["actor"], method).remote(*args, **kwargs), timeout=1800)
            with self.lock:
                if method == "get_status":
                    item["status"] = result
                if method in ("get_feature_contract_status", "prepare_feature_contract_status"):
                    item["features"] = result
                if method == "init":
                    item["state"] = "ready"
                if method == "shutdown":
                    ray.kill(item["actor"], no_restart=True)
                    item["state"] = "closed"
                    for gpu in item["gpus"]:
                        self.claims.pop(gpu)
                    logger.info("Target session %s engine %s closed", item["session"], key)
            return result
        except BaseException:
            if method == "shutdown":
                with self.lock:
                    item["state"] = "retained"
            raise
        finally:
            with self.lock:
                item["active"] -= 1

    def reap_disconnected(self, *, expiry=30):
        """Fail closed on lost control; reclaim only after observed reader exit."""
        from aspec.utils.misc import get_current_node_ip

        with self.lock:
            # Record descendants while their parent is still alive. A backend
            # scheduler can kill its actor on initialization failure; later
            # cleanup must identify only this service's own remaining children.
            for item in self.engines.values():
                if item["state"] != "closed":
                    observe_process_tree(item["processes"])
            candidates = [(key, item) for key, item in self.engines.items()
                          if item["state"] not in ("closed", "closing")
                          and item["active"] == 0
                          and time.monotonic() - item["heartbeat"] > expiry]
            for _, item in candidates:
                item["state"] = "retained"
        for key, item in candidates:
            if readers_exited(item["owners"], get_current_node_ip()):
                logger.warning("Fenced disconnected session %s; cancelling Target %s", item["session"], key)
                try:
                    self.call(key, "shutdown", [], {})
                except ray.exceptions.RayActorError:
                    # The read fence is established above. Observe exact source
                    # process identities before retiring a dead actor's lease.
                    if not item["processes"]:
                        raise
                    stop_owned_processes(item["processes"])
                    with self.lock:
                        item["state"] = "closed"
                        item["terminal_reason"] = "actor_failed_readers_fenced_sources_exited"
                        for gpu in item["gpus"]:
                            self.claims.pop(gpu, None)
                    logger.warning("Retired failed Target %s after observed source exit", key)

    def idle(self):
        with self.lock:
            return not self.claims


def observe_process_tree(identities):
    import psutil

    for pid, created in list(identities.items()):
        try:
            process = psutil.Process(pid)
            if process.create_time() == created:
                for child in process.children(recursive=True):
                    try:
                        identities[child.pid] = child.create_time()
                    except psutil.NoSuchProcess:
                        pass
        except psutil.NoSuchProcess:
            pass


def session_environment(args):
    """Native scheduler children inherit this session, never the service's job."""
    contracts = getattr(args, "aspec_shared_consumer_contracts", None) or {}
    return {
        "ASPEC_RUN_ID": args.aspec_run_id,
        "ASPEC_ATTEMPT_ID": args.aspec_attempt_id,
        "ASPEC_SHARED_SOURCE": str(int(bool(getattr(args, "aspec_shared_source", False)))),
        "ASPEC_SHARED_FEATURE_VIEWS": str(int(len(set(contracts.values())) > 1)),
    }


def stop_owned_processes(identities):
    """Called only after the read fence; never discover or stop unrelated PIDs."""
    import psutil

    processes = []
    for pid, created in identities.items():
        try:
            process = psutil.Process(pid)
            if process.create_time() == created:
                processes.append(process)
                process.terminate()
        except psutil.NoSuchProcess:
            pass
    _, alive = psutil.wait_procs(processes, timeout=5)
    for process in alive:
        try:
            if process.create_time() == identities[process.pid]:
                process.kill()
        except psutil.NoSuchProcess:
            pass
    _, alive = psutil.wait_procs(alive, timeout=5)
    if alive:
        raise RuntimeError("Owned Target source process exit remains unverified")


def http_server(service, host, port):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def respond(self, status, result):
            data = json.dumps({"protocol": PROTOCOL, "result": encode(result)}, allow_nan=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            try:
                self.wfile.write(data)
            except (BrokenPipeError, ConnectionResetError):
                logger.debug("Target caller disconnected before reading its response")

        def do_GET(self):
            if self.path != "/v1/info":
                self.respond(404, "Unknown Target endpoint")
                return
            self.respond(200, service.info())

        def do_POST(self):
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= MAX_BODY:
                    raise ValueError("Invalid Target request size")
                value = decode(json.loads(self.rfile.read(length)))
                if self.path == "/v1/engines":
                    result = service.create(value)
                elif self.path == "/v1/heartbeat":
                    result = service.heartbeat(value["key"])
                elif self.path == "/v1/call":
                    result = service.call(value["key"], value["method"], value["args"], value["kwargs"])
                else:
                    raise ValueError("Unknown Target endpoint")
                self.respond(200, result)
            except Exception as error:
                logger.exception("Target control request failed")
                self.respond(409, f"{type(error).__name__}: {error}")

    server = ThreadingHTTPServer((host, port), Handler)
    server.daemon_threads = True
    return server


def main():
    from omegaconf import OmegaConf

    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Run Target in this environment; Draft connects by address.")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--address", help="Listening http://host:port; defaults to the config or localhost:30000")
    parser.add_argument("--gpus", help="Allowed physical GPU IDs, comma separated")
    args = parser.parse_args()
    config = OmegaConf.load(args.config) if args.config else OmegaConf.create({})
    address = validate_address(args.address or OmegaConf.select(
        config, "inference.target_service.address", default="http://127.0.0.1:30000"))
    model = OmegaConf.select(config, "model.target_model_path")
    devices = [int(gpu) for gpu in args.gpus.split(",")] if args.gpus else None
    runtime = inventory()  # Check this environment and patch before starting processes.
    service = TargetService(model=model, devices=devices, runtime=runtime)
    parsed = urlsplit(address)
    server = http_server(service, parsed.hostname, parsed.port)
    stop = threading.Event()

    def stopping(*_):
        service.draining = True
        stop.set()

    previous = {sig: signal.signal(sig, stopping) for sig in (signal.SIGINT, signal.SIGTERM)}
    os.environ["RAY_USAGE_STATS_ENABLED"] = "0"
    temporary = tempfile.mkdtemp(prefix="aspec-t-", dir="/tmp")
    try:
        ray.init(address="local", num_gpus=0, num_cpus=min(8, os.cpu_count() or 1),
                 include_dashboard=False, object_store_memory=256 * 1024 * 1024,
                 _temp_dir=temporary)
        # Ray may replace the process SIGTERM handler during initialization.
        for sig in previous:
            signal.signal(sig, stopping)
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        print(json.dumps({"state": "ready", "address": address, "protocol": PROTOCOL,
                          "python": runtime["python"], "pid": os.getpid()}), flush=True)
        while not stop.wait(1) or not service.idle():
            service.reap_disconnected()
        server.shutdown()
        worker.join()
    finally:
        # Active sources must survive until readers are fenced, including during
        # signal handling. Never let Ray's implicit driver exit reclaim them.
        while not service.idle():
            try:
                service.draining = True
                service.reap_disconnected()
                time.sleep(1)
            except BaseException:
                logger.exception("Target cleanup is incomplete; retaining the service")
        server.server_close()
        ray.shutdown()
        for sig, handler in previous.items():
            signal.signal(sig, handler)


if __name__ == "__main__":
    main()
