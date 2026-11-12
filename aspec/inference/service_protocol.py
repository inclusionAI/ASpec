# Copyright 2026 Ant Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Versioned JSON control plane; feature tensors stay in Mooncake.

Inspired by SpecForge f8cc3dc's external capture adapter. ASpec retains its
SglEngine, feature contracts and transport rather than copying that adapter.
"""
import json
from dataclasses import asdict, is_dataclass
from urllib.error import HTTPError
from urllib.parse import urlsplit
from urllib.request import ProxyHandler, Request, build_opener

PROTOCOL = "aspec.target-service/v1"
MAX_BODY = 64 * 1024 * 1024
METHODS = frozenset({
    "init", "generate", "generate_with_decode", "update_weights_from_disk",
    "get_node_ip", "find_free_port", "get_status", "get_mooncake_config",
    "get_feature_contract_status", "prepare_feature_contract_status", "health_check", "shutdown",
})


def validate_address(value):
    parsed = urlsplit(value)
    if (parsed.scheme != "http" or not parsed.hostname or parsed.username or parsed.password
            or parsed.path not in ("", "/") or parsed.query or parsed.fragment):
        raise ValueError("Target address must be an http://host:port origin")
    if parsed.port is None:
        raise ValueError("Target address must include a port")
    return value.rstrip("/")


def encode(value):
    """Encode only explicit data types, never Python objects or feature tensors."""
    from omegaconf import OmegaConf

    if OmegaConf.is_config(value):
        value = OmegaConf.to_container(value, resolve=True)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): encode(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [encode(v) for v in value]
    if is_dataclass(value):
        return encode(asdict(value))
    # Importing torch is confined to actual control payload conversion.
    import torch

    if isinstance(value, torch.dtype):
        return {"$dtype": str(value).removeprefix("torch.")}
    if isinstance(value, torch.Tensor):
        if value.device.type != "cpu" or value.dtype not in (torch.int64, torch.int32):
            raise TypeError("Only CPU token IDs may cross the Target control plane")
        return {"$tokens": value.tolist(), "dtype": str(value.dtype).removeprefix("torch.")}
    raise TypeError(f"Unsupported Target control value: {type(value).__name__}")


def decode(value):
    if isinstance(value, list):
        return [decode(v) for v in value]
    if isinstance(value, dict):
        if set(value) == {"$dtype"}:
            import torch

            dtype = getattr(torch, value["$dtype"], None)
            if not isinstance(dtype, torch.dtype):
                raise ValueError("Invalid wire dtype")
            return dtype
        if set(value) == {"$tokens", "dtype"}:
            import torch

            if value["dtype"] not in ("int64", "int32"):
                raise ValueError("Invalid token dtype")
            return torch.tensor(value["$tokens"], dtype=getattr(torch, value["dtype"]))
        return {k: decode(v) for k, v in value.items()}
    return value


def request(address, path, payload=None, *, timeout=1200):
    address = validate_address(address)
    body = None if payload is None else json.dumps(encode(payload), allow_nan=False).encode()
    if body is not None and len(body) > MAX_BODY:
        raise ValueError("Target control request exceeds the size limit")
    req = Request(address + path, data=body, headers={"Content-Type": "application/json"})
    # Cluster-internal control endpoints must not be sent through HTTP proxies.
    opener = build_opener(ProxyHandler({}))
    try:
        with opener.open(req, timeout=timeout) as response:
            raw = response.read(MAX_BODY + 1)
    except HTTPError as error:
        detail = error.read(8192).decode(errors="replace")
        raise RuntimeError(f"Target {address}{path}: {detail}") from error
    if len(raw) > MAX_BODY:
        raise ValueError("Target control response exceeds the size limit")
    result = json.loads(raw)
    if result.get("protocol") != PROTOCOL:
        raise RuntimeError("Target service protocol differs")
    return decode(result["result"])


def engine_arguments(args):
    # Driver-only Python model/config objects are deliberately not sent to Target.
    result = {}
    for key, value in vars(args).items():
        if key.startswith("_") or key == "draft_model_config_obj":
            continue
        try:
            result[key] = encode(value)
        except TypeError:
            # Unknown objects cannot become an implicit cross-environment API.
            raise TypeError(f"Non-data runtime argument at Target boundary: {key}") from None
    return result
