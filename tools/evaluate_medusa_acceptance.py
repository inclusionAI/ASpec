# Copyright 2026 Ant Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Compatibility entry point for the installed evaluate_medusa_acceptance command."""

import sys
from importlib import import_module

_command = import_module("aspec.commands.evaluate_medusa_acceptance")
if __name__ == "__main__":
    _command.main()
else:
    sys.modules[__name__] = _command
