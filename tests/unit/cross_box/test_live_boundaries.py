# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102

"""What each part of the Live example loads in a fresh interpreter: only its composition roots import the core."""

import json
import os
import subprocess
import sys
import unittest
from pathlib import Path

from tests.unit.core_helpers import HARNESS_GROUP, NO_HARNESS_GROUP

ROOT = Path(__file__).resolve().parents[3]
EXAMPLE = "examples.frontend_backend_live"
CORE, ADAPTER = f"{EXAMPLE}.delegation_core", f"{EXAMPLE}.adapter"
HARNESS = f"{CORE}.backend.harness"
ROOTS = (f"{EXAMPLE}.factory", f"{EXAMPLE}.pipeline")
GATEWAY = "gateways.live"
SHARED = "examples.shared"
LOAD = """
import importlib, json, pkgutil, sys
names, packages = map(json.loads, sys.argv[1:])
for name in names:
    importlib.import_module(name)
for package in map(importlib.import_module, packages):
    for module in pkgutil.walk_packages(package.__path__, package.__name__ + "."):
        importlib.import_module(module.name)
print(json.dumps(sorted(sys.modules)))
"""
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


def within(name: str, package: str) -> bool:
    """Return whether ``name`` is ``package`` or one of its modules."""
    return name == package or name.startswith(package + ".")


def modules() -> list[str]:
    """Return every module of the example, named from its file, so no package walk imports a subpackage."""
    src = ROOT / "src"
    names = []
    for path in sorted((src / "examples" / "frontend_backend_live").rglob("*.py")):
        parts = path.relative_to(src).with_suffix("").parts
        names.append(".".join(parts[:-1] if parts[-1] == "__init__" else parts))
    return names


def outside_the_core(name: str) -> bool:
    """Return whether ``name`` is Pipecat, the gateway, or a module of the example outside its core."""
    return name.split(".")[0] in ("pipecat", "gateways") or (name.startswith(EXAMPLE + ".") and not within(name, CORE))


def outside_the_gateway(name: str) -> bool:
    """Return whether ``name`` is Pipecat, another gateway, or an example module other than the shared contracts."""
    top = name.split(".")[0]
    if top == "pipecat":
        return True
    if top == "gateways":
        return name not in ("gateways", GATEWAY) and not name.startswith(GATEWAY + ".")
    return top == "examples" and name not in ("examples", SHARED) and not name.startswith(SHARED + ".")


class BoundaryTests(unittest.TestCase):
    def fresh(self, script: str, *args: str) -> str:
        """Return the last line ``script`` prints in a fresh interpreter on this checkout's modules."""
        path = os.pathsep.join([str(ROOT / "src"), str(ROOT)])
        done = subprocess.run(
            [sys.executable, "-c", script, *args],
            cwd=ROOT,
            env={**os.environ, "PYTHONPATH": path},
            capture_output=True,
            text=True,
            timeout=120,
        )
        self.assertEqual(done.returncode, 0, done.stderr)
        return done.stdout.splitlines()[-1]

    def loaded_by(self, names=(), packages=()) -> list[str]:
        """Return the modules a fresh interpreter loads to import ``names`` and every module of ``packages``."""
        return json.loads(self.fresh(LOAD, json.dumps(list(names)), json.dumps(list(packages))))

    def test_the_cascade_loads_no_gateway_core_or_adapter_module(self):
        cascade = [name for name in modules() if not (within(name, CORE) or within(name, ADAPTER) or name in ROOTS)]
        self.assertIn(f"{EXAMPLE}.tool_handlers", cascade)
        loaded = self.loaded_by(cascade)
        crossing = [name for name in loaded if within(name, "gateways") or within(name, CORE) or within(name, ADAPTER)]
        self.assertEqual(crossing, [])

    def test_the_adapter_loads_no_core_module(self):
        adapter = [name for name in modules() if within(name, ADAPTER)]
        self.assertIn(f"{ADAPTER}.adapter", adapter)
        self.assertEqual([name for name in self.loaded_by(adapter) if within(name, CORE)], [])

    @unittest.skipUnless(HARNESS_GROUP, NO_HARNESS_GROUP)
    def test_the_core_loads_no_pipecat_gateway_or_frontend_module(self):
        loaded = self.loaded_by(packages=[CORE])
        self.assertIn(CORE + ".app", loaded)
        self.assertEqual([name for name in loaded if outside_the_core(name)], [])

    def test_the_gateway_loads_no_pipecat_other_gateway_or_example(self):
        loaded = self.loaded_by(packages=[GATEWAY])
        self.assertIn(GATEWAY + ".gateway.websocket", loaded)
        self.assertEqual([name for name in loaded if outside_the_gateway(name)], [])

    def test_without_the_harness_group_only_the_harness_refuses(self):
        self.assertEqual(
            self.fresh(WITHOUT_THE_GROUP),
            "backend.adapter harness needs the packages of the harness dependency group (missing: acp, mcp): "
            "install them with `uv sync --group harness`",
        )
