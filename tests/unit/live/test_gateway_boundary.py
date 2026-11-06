# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102
"""The Live gateway stays a piece of its own: importing it loads no Pipecat, no other gateway and no example's code."""

import json
import os
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
GATEWAY = "gateways.live"
SHARED = "examples.shared"  # the contracts that the gateway and the examples share
# Imports every module of the gateway in a fresh interpreter, then prints the names of the modules it loaded.
LOAD_THE_GATEWAY = f"""
import importlib, json, pkgutil, sys
import {GATEWAY} as gateway
for module in pkgutil.walk_packages(gateway.__path__, gateway.__name__ + "."):
    importlib.import_module(module.name)
print(json.dumps(sorted(sys.modules)))
"""


def outside_the_gateway(name: str) -> bool:
    """Whether ``name`` is Pipecat, another gateway, or an example's own module; the shared contracts are not."""
    top = name.split(".")[0]
    if top == "pipecat":
        return True
    if top == "gateways":
        return name not in ("gateways", GATEWAY) and not name.startswith(GATEWAY + ".")
    return top == "examples" and name not in ("examples", SHARED) and not name.startswith(SHARED + ".")


class GatewayBoundaryTests(unittest.TestCase):
    def test_importing_the_gateway_loads_no_pipecat_no_other_gateway_and_no_example(self):
        path = os.pathsep.join([str(ROOT / "src"), str(ROOT)])
        done = subprocess.run(
            [sys.executable, "-c", LOAD_THE_GATEWAY],
            cwd=ROOT,
            env={**os.environ, "PYTHONPATH": path},
            capture_output=True,
            text=True,
            check=True,
        )
        loaded = json.loads(done.stdout.splitlines()[-1])
        self.assertIn(GATEWAY + ".gateway.websocket", loaded)  # every module of the gateway was imported
        self.assertEqual([name for name in loaded if outside_the_gateway(name)], [])


if __name__ == "__main__":
    unittest.main()
