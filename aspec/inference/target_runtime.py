# Copyright 2026 Ant Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Identity shared across the independently installed Draft and Target roles."""
import hashlib
import importlib.metadata
from pathlib import Path


def code_identity():
    package = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for path in sorted(package.rglob("*")):
        if path.suffix in {".py", ".patch"}:
            digest.update(path.relative_to(package).as_posix().encode())
            digest.update(b"\0")
            digest.update(path.read_bytes())
    try:
        version = importlib.metadata.version("ling-aspec-runtime")
    except importlib.metadata.PackageNotFoundError:
        version = None
    return {"version": version, "code_sha256": digest.hexdigest()}
