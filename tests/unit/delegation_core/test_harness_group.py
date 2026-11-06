# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102
"""The harness's packages are the optional ``harness`` dependency group: the rest of the example runs without them."""

import os
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
EXAMPLE = "examples.frontend_backend_live"
HARNESS = EXAMPLE + ".delegation_core.backend.harness"
# In a fresh interpreter without the group's packages: imports every module of the example but the harness's, then
# builds the harness's source and prints why it refused.
WITHOUT_THE_GROUP = f"""
import importlib, pkgutil, sys
for package in ("acp", "mcp"):
    sys.modules[package] = None  # as if the harness group were not installed
import {EXAMPLE} as example
for module in pkgutil.walk_packages(example.__path__, example.__name__ + "."):
    if module.name != "{HARNESS}" and not module.name.startswith("{HARNESS}."):
        importlib.import_module(module.name)
from {HARNESS} import HarnessSource
try:
    HarnessSource.from_settings({{}}, endpoints=None, tools=None, vocabulary=None)
except ValueError as error:
    print(error)
"""


class HarnessPackagesTests(unittest.TestCase):
    def test_without_the_group_the_example_loads_and_only_the_harness_refuses_naming_the_install(self):
        path = os.pathsep.join([str(ROOT / "src"), str(ROOT)])
        done = subprocess.run(
            [sys.executable, "-c", WITHOUT_THE_GROUP],
            cwd=ROOT,
            env={**os.environ, "PYTHONPATH": path},
            capture_output=True,
            text=True,
            check=True,
        )
        self.assertEqual(
            done.stdout.splitlines()[-1],
            "backend.adapter harness needs the packages of the harness dependency group (missing: acp, mcp): "
            "install them with `uv sync --group harness`",
        )


if __name__ == "__main__":
    unittest.main()
