# Copyright 2026 Ant Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Frozen upstream Megatron Target workers behind ASpec's Target service.

Each engine owns a role-local TP world. Only the publishing rank owns source
storage; TP activations are replicated because sequence parallelism is disabled.
Control pipes carry tokens and descriptors, never feature tensors.
"""

from __future__ import annotations

import asyncio
import dataclasses
import os
import sys
import threading
import traceback
import uuid
from contextlib import ExitStack, contextmanager
from datetime import timedelta

import torch

from aspec.feature_contract import FeatureContractRuntime
from aspec.inference.target_prefill import TargetPrefillDirectSink, TargetPrefillRequest
from aspec.ray.ray_actor import RayActor
from aspec.utils.misc import get_current_node_ip


@contextmanager
def _prefill_batch(synchronize):
    """Hold committed source slots until every TP worker finishes the batch."""
    sinks = []
    with ExitStack() as cleanup:
        def stage(sink):
            cleanup.callback(sink.abort)
            sinks.append(sink)

        yield stage
        for sink in sinks:
            sink.validate_publishable()
        if synchronize is not None:
            synchronize()
        for sink in sinks:
            sink.publish()
        cleanup.pop_all()


def _close_source(store):
    timeout = max(float(store.config.get_retry_max_wait_seconds), 5.0)
    if not store.wait_for_p2p_source_releases(timeout=timeout):
        raise TimeoutError("Megatron Target source leases still have active readers")
    store.close()


def _target_worker(connection, local_rank, gpu_ids, args, mooncake_config, port, engine_rank):
    import torch.distributed as dist

    from aspec.models.megatron import initialize_parallelism, load_qwen3_target

    store = None
    try:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(gpu) for gpu in gpu_ids)
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            "nccl", init_method=f"tcp://127.0.0.1:{port}",
            rank=local_rank, world_size=len(gpu_ids),
            timeout=timedelta(seconds=getattr(args, "megatron_init_timeout", 600)),
        )
        groups = initialize_parallelism(len(gpu_ids))
        model, config = load_qwen3_target(
            args.target_model_path, groups,
            implementation=getattr(args, "megatron_implementation", "transformer_engine"),
        )
        planned_context = args.aspec_context_contract["backend_context_length"]
        if max(planned_context, args.max_seq_length) > model.max_sequence_length:
            raise ValueError("Planned Target context exceeds the loaded model's maximum sequence length")
        captures = list(args.aux_hidden_states_layers)
        runtime = FeatureContractRuntime(
            args.aspec_feature_contract, role="producer", participant=f"megatron_engine_{engine_rank}",
        )
        runtime.validate_producer_binding(
            hidden_size=config.hidden_size, num_hidden_layers=config.num_hidden_layers,
            aux_hidden_state_layers=captures, store_last_hidden_states=args.store_last_hidden_states,
        )
        if local_rank == 0:
            from aspec.transfer.mooncake.eagle_store import EagleMooncakeStore

            source_config = dataclasses.replace(mooncake_config, global_segment_size=0, async_put_pool_size=0)
            store = EagleMooncakeStore(source_config)
            store.setup(device=local_rank, p2p_role="source")
        status = {
            "pid": os.getpid(), "hidden_size": config.hidden_size,
            "python": sys.executable, "prefix": sys.prefix,
            "aspec": __import__("aspec").__file__,
            "megatron": __import__("megatron.core", fromlist=["__file__"]).__file__,
            "torch": torch.__version__, "cuda": torch.version.cuda,
            "tensor_parallel_size": len(gpu_ids), "local_rank": local_rank,
            "device_capability": list(torch.cuda.get_device_capability()),
            "model_implementation": getattr(args, "megatron_implementation", "transformer_engine"),
            "model_class": type(model).__name__, "max_sequence_length": model.max_sequence_length,
            "hybrid_layer_pattern": model.hybrid_layer_pattern,
            "target_frozen": not any(p.requires_grad for p in model.parameters()),
        }
        connection.send({"ok": True, "value": status})
        while True:
            method, payload = connection.recv()
            if method == "shutdown":
                if store is not None:
                    _close_source(store)
                dist.destroy_process_group()
                connection.send({"ok": True, "value": None})
                return
            if method == "status":
                features = runtime.status()
                if store is not None:
                    features["transport"] = store.get_p2p_status()
                connection.send({"ok": True, "value": features})
                continue
            if method != "generate":
                raise ValueError(f"Unknown Megatron Target operation: {method}")
            outputs = []
            with _prefill_batch(dist.barrier if len(gpu_ids) > 1 else None) as stage:
                for sample in payload:
                    ids = torch.tensor(sample["tokens"], dtype=torch.long, device="cuda")
                    length = ids.numel()
                    if not 0 < length <= args.max_seq_length:
                        raise ValueError("Target input length is outside the planned context")
                    request = TargetPrefillRequest(sample["request_id"], sample["data_id"], length)
                    sink = None
                    if local_rank == 0:
                        sink = TargetPrefillDirectSink(
                            store=store, requests=[request], capture_layers=captures,
                            hidden_size=config.hidden_size, store_last_hidden_states=args.store_last_hidden_states,
                            source_engine=f"megatron_{engine_rank}", context_parallel_size=1,
                        )
                        stage(sink)
                        sink.begin(input_ids=ids, request_ids=[request.request_id],
                                   sequence_lengths=[length], prefix_lengths=[0])
                    final = model.prefill(ids.unsqueeze(0), captures, sink.capture if sink is not None else None)
                    if local_rank == 0:
                        sink.finish(final)
                        key = sink.commit_request(request.request_id)
                        shapes = {"hidden_states": (length, len(captures) * config.hidden_size),
                                  "input_ids": (length,)}
                        dtypes = {"hidden_states": torch.bfloat16, "input_ids": torch.int64}
                        if args.store_last_hidden_states:
                            shapes["last_hidden_states"] = (length, config.hidden_size)
                            dtypes["last_hidden_states"] = torch.bfloat16
                        metadata = {
                            "request_id": request.request_id, "engine_rank": engine_rank,
                            "aspec_feature_contract_id": runtime.contract_id,
                            "source_node_ip": get_current_node_ip(),
                            "feature_affinity": getattr(args, "mooncake_feature_affinity", "any"),
                            "target_feature_output": "direct_to_transport",
                            "transport_data_plane": "transfer_engine",
                            "transport_protocol": mooncake_config.protocol,
                            "feature_layout": "full",
                            "draft_context_parallel_size": 1, "draft_feature_layout": "full",
                            "draft_attention_backend": args.attention_backend,
                        }
                        from aspec.config.draft_execution import is_direct_cp_backend

                        if is_direct_cp_backend(args.attention_backend):
                            metadata.update(direct_cp_sharded=True, direct_cp_size=1)
                        runtime.validate_metadata(tensor_shapes=shapes, tensor_dtypes=dtypes, metadata=metadata)
                        outputs.append({"mooncake_key": key, "tensor_shapes": shapes,
                                        "tensor_dtypes": dtypes, "metadata": metadata})
            connection.send({"ok": True, "value": outputs if local_rank == 0 else None})
    except BaseException as error:
        try:
            connection.send({"ok": False, "error": repr(error), "traceback": traceback.format_exc()})
        finally:
            # Published storage remains owned until the service fences readers
            # and observes this process exiting, including native-call failures.
            connection.close()
            if store is not None:
                # Keep all published CUDA registrations alive on failure. The
                # Target service terminates this world only after its read fence.
                threading.Event().wait()


class MegatronEngine(RayActor):
    """Ray control facade; model and source storage live in owned child workers."""

    def __init__(self, args, rank, base_gpu_id=None, num_gpus_per_engine=1,
                 node_rank=0, engine_group=0, gpu_ids=None):
        self.args, self.rank = args, rank
        self.base_gpu_id = base_gpu_id
        self.gpu_ids = list(gpu_ids or range(base_gpu_id, base_gpu_id + num_gpus_per_engine))
        self.num_gpus_per_engine = num_gpus_per_engine
        self._connections, self._processes = [], []
        self._status = []
        self._lock = threading.Lock()
        self._mooncake_config = None
        self._features = None
        self._failed = False

    def _receive(self):
        try:
            return self._receive_workers()
        except BaseException:
            self._failed = True
            raise

    def _receive_workers(self):
        results = []
        for connection, process in zip(self._connections, self._processes, strict=True):
            if not connection.poll(getattr(self.args, "megatron_init_timeout", 600)):
                raise TimeoutError(f"Megatron Target worker {process.pid} did not complete")
            result = connection.recv()
            if not result["ok"]:
                raise RuntimeError(f"Megatron Target worker failed: {result['error']}\n{result['traceback']}")
            results.append(result["value"])
        return results

    def _call(self, method, payload=None):
        with self._lock:
            if not self.health_check():
                raise RuntimeError("Megatron Target worker world is unavailable")
            try:
                for connection in self._connections:
                    connection.send((method, payload))
                return self._receive()[0]
            except BaseException:
                self._failed = True
                raise

    def init(self, mooncake_config=None, pre_allocated_port=None):
        if self._processes:
            raise RuntimeError("Megatron Target engine cannot be initialized twice")
        if mooncake_config is None or mooncake_config.data_plane != "transfer_engine":
            raise ValueError("Megatron Target requires the Mooncake transfer_engine data plane")
        if pre_allocated_port is None:
            raise ValueError("Megatron Target requires a reserved initialization port")
        self._mooncake_config = mooncake_config
        context = torch.multiprocessing.get_context("spawn")
        for rank in range(self.num_gpus_per_engine):
            parent, child = context.Pipe()
            process = context.Process(target=_target_worker, args=(
                child, rank, self.gpu_ids, self.args, mooncake_config, pre_allocated_port, self.rank,
            ))
            process.start()
            child.close()
            self._connections.append(parent)
            self._processes.append(process)
        self._status = self._receive()

    async def generate(self, data_id, input_ids_ref=None, packed_loss_mask_list=None,
                       formatted_prompts=None, request_ids=None, return_last_hidden_states=False,
                       return_logits=True, multimodal_inputs=None):
        if multimodal_inputs:
            raise ValueError("Megatron Target currently supports text-only Qwen3")
        if (input_ids_ref is None) == (formatted_prompts is None):
            raise ValueError("Provide exactly one of input_ids_ref or formatted_prompts")
        if formatted_prompts is not None:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(self.args.target_model_path, trust_remote_code=False)
            tokens = tokenizer(formatted_prompts, add_special_tokens=False)["input_ids"]
        else:
            tokens = [value.reshape(-1).tolist() for value in input_ids_ref]
        if len(tokens) > self._mooncake_config.ring_slots_for_role("source"):
            raise ValueError("Megatron Target batch exceeds the number of source ring slots")
        ids = ([f"{data_id}_{index}" for index in range(len(tokens))]
               if isinstance(data_id, str) else list(data_id))
        if len(ids) != len(tokens):
            raise ValueError("Target data identities and token sequences differ in length")
        if request_ids is None:
            request_ids = [uuid.uuid4().hex for _ in tokens]
        elif isinstance(request_ids, str):
            request_ids = [f"{request_ids}_{index}" for index in range(len(tokens))]
        if len(request_ids) != len(tokens):
            raise ValueError("Target request identities and token sequences differ in length")
        payload = [dict(tokens=value, data_id=data, request_id=request)
                   for value, data, request in zip(tokens, ids, request_ids, strict=True)]
        return await asyncio.to_thread(self._call, "generate", payload)

    def get_status(self):
        return {
            "rank": self.rank, "pid": os.getpid(), "node_ip": self.get_node_ip(),
            "initialized": bool(self._status) and self.health_check(),
            "base_gpu_id": self.base_gpu_id, "engine_gpu_ids": self.gpu_ids,
            "num_gpus_per_engine": self.num_gpus_per_engine,
            "hidden_size": self._status[0]["hidden_size"] if self._status else None,
            "target_prefill_memory_optimized": True, "workers": self._status,
            "token_capacity": {"context": self.args.aspec_context_contract,
                               "context_length": self.args.aspec_context_contract["backend_context_length"],
                               "profiled_max_total_tokens": None,
                               "required_max_total_tokens": None,
                               "max_prefill_tokens": self.args.max_seq_length,
                               "requested_max_running_requests": 1,
                               "allocation_policy": "sequential_full_prefill_without_kv_pool"},
        }

    def health_check(self, timeout=5.0):
        return (not self._failed and bool(self._processes)
                and all(process.is_alive() for process in self._processes))

    def get_mooncake_config(self):
        return self._mooncake_config

    def get_feature_contract_status(self):
        if self._processes and self.health_check():
            self._features = self._call("status")
        return self._features or {"role": "producer", "state": "pending_observation"}

    async def prepare_feature_contract_status(self):
        return await asyncio.to_thread(self.get_feature_contract_status)

    async def shutdown(self):
        if self._processes:
            await asyncio.to_thread(self._call, "shutdown")
            for process in self._processes:
                process.join(timeout=30)
                if process.is_alive():
                    raise RuntimeError("Megatron Target worker did not exit after source drain")
            for connection in self._connections:
                connection.close()
            self._processes.clear()
            self._connections.clear()
