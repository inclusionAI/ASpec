# Copyright 2026 Ant Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Bootstrap a Ray worker with its own venv libraries before native imports."""
import os
import sys

from aspec._environment import execute


def main():
    os.environ["VIRTUAL_ENV"] = sys.prefix
    # The Ray head belongs to Draft. Never load its CUDA libraries in Target.
    os.environ.pop("LD_LIBRARY_PATH", None)
    os.environ.pop("PYTHONPATH", None)
    os.environ.pop("PYTHONHOME", None)
    execute([sys.executable, *sys.argv[1:]])


if __name__ == "__main__":
    main()
