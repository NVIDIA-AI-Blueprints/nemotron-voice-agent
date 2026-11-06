# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102
"""The delegation core stays a piece of its own: importing it loads no Pipecat, no gateway and no frontend module."""

import json
import os
import subprocess
import sys
import unittest
from pathlib import Path

from tests.unit.core_helpers import HARNESS_GROUP, NO_HARNESS_GROUP

ROOT = Path(__file__).resolve().parents[3]
CORE = "examples.frontend_backend_live.delegation_core"
# Imports every module of the core in a fresh interpreter, then prints the names of the modules it loaded.
LOAD_THE_CORE = f"""
import importlib, json, pkgutil, sys
import {CORE} as core
for module in pkgutil.walk_packages(core.__path__, core.__name__ + "."):
    importlib.import_module(module.name)
print(json.dumps(sorted(sys.modules)))
"""


def outside_the_core(name: str) -> bool:
    """Whether ``name`` is Pipecat, the gateway, or a module of the example outside its core."""
    in_core = name == CORE or name.startswith(CORE + ".")
    return name.split(".")[0] in ("pipecat", "gateways") or (
        name.startswith("examples.frontend_backend_live.") and not in_core
    )


class CoreBoundaryTests(unittest.TestCase):
    @unittest.skipUnless(HARNESS_GROUP, NO_HARNESS_GROUP)
    def test_importing_the_core_loads_no_pipecat_no_gateway_and_no_frontend_module(self):
        path = os.pathsep.join([str(ROOT / "src"), str(ROOT)])
        done = subprocess.run(
            [sys.executable, "-c", LOAD_THE_CORE],
            cwd=ROOT,
            env={**os.environ, "PYTHONPATH": path},
            capture_output=True,
            text=True,
            check=True,
        )
        loaded = json.loads(done.stdout.splitlines()[-1])
        self.assertIn(CORE + ".app", loaded)  # every module of the core was imported
        self.assertEqual([name for name in loaded if outside_the_core(name)], [])


if __name__ == "__main__":
    unittest.main()
