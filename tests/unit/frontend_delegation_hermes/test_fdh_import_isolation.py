# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D100, D101, D102, D103, D107

"""C5: the shared core never imports an adapter, no adapter imports another, the gateway never loads Pipecat.

Adapters (the server heads) are discovered by globbing ``adapters/*``, so adding one needs no edit here. The
static check walks every import of every module; the runtime check imports each entry
point in a fresh interpreter and inspects ``sys.modules``.
"""

from __future__ import annotations

import ast
import json
import subprocess
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC = REPO_ROOT / "src"
PACKAGE = SRC / "examples" / "frontend_delegation_hermes"
PREFIX = "examples.frontend_delegation_hermes"
ADAPTERS = sorted(path.name for path in (PACKAGE / "adapters").iterdir() if (path / "__init__.py").is_file())

# Modules from outside the example that the example may import (Section 3 of the plan).
SHARED_ALLOWED = {"utils", "examples.shared.nemotron_speech_text_filter"}
ADAPTER_ALLOWED = {
    # The shared Pipecat STT, and for browser UI sessions the repository's UI transport and registry.
    "pipecat": {"examples.shared.nvidia_force_eou_stt", "examples.shared.pipeline_utils", "examples_registry"},
}
PIPECAT_RUNTIME = ("pipecat.pipeline", "pipecat.processors", "pipecat.transports", "pipecat.services")
# Hermes is imported only by the worker, in its own interpreter (never by a voice adapter).
HERMES_TOP_LEVEL = {"run_agent", "tools", "toolsets", "model_tools", "hermes_cli"}


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names.add(node.module)
    return names


def _repo_module(name: str) -> bool:
    top = name.split(".")[0]
    return top in {"examples", "utils", "realtime", "prototypes"} or (SRC / f"{top}.py").is_file()


def _files(sub: str) -> list[Path]:
    return [path for path in sorted((PACKAGE / sub).rglob("*.py")) if "__pycache__" not in path.parts]


class StaticLayeringTests(unittest.TestCase):
    def test_adapters_are_discovered(self) -> None:
        self.assertIn("pipecat", ADAPTERS)

    def test_core_imports_no_adapter_and_only_allowed_repo_modules(self) -> None:
        for path in _files("core"):
            for name in _imports(path):
                with self.subTest(file=str(path.relative_to(PACKAGE)), module=name):
                    self.assertFalse(name.startswith(f"{PREFIX}.adapters"), "the core imports a adapter")
                    self.assertFalse(name.startswith(PIPECAT_RUNTIME), "the core imports Pipecat's runtime")
                    if _repo_module(name) and not name.startswith(PREFIX):
                        self.assertIn(name, SHARED_ALLOWED)

    def test_no_adapter_imports_another_adapter(self) -> None:
        for adapter in ADAPTERS:
            others = [f"{PREFIX}.adapters.{other}" for other in ADAPTERS if other != adapter]
            for path in _files(f"adapters/{adapter}"):
                for name in _imports(path):
                    with self.subTest(adapter=adapter, file=path.name, module=name):
                        self.assertFalse(name.startswith(tuple(others)), f"adapters/{adapter} imports another adapter")
                        if _repo_module(name) and not name.startswith(PREFIX):
                            self.assertIn(name, SHARED_ALLOWED | ADAPTER_ALLOWED.get(adapter, set()))

    def test_hermes_is_imported_only_by_the_worker(self) -> None:
        for path in PACKAGE.rglob("*.py"):
            if "__pycache__" in path.parts or path.parent.name == "worker":
                continue
            for name in _imports(path):
                with self.subTest(file=str(path.relative_to(PACKAGE)), module=name):
                    self.assertNotIn(name.split(".")[0], HERMES_TOP_LEVEL - {"tools"})


def _loaded_after(module: str) -> list[str]:
    code = f"import json, sys; import {module}; print(json.dumps(sorted(sys.modules)))"
    result = subprocess.run(  # noqa: S603 - fixed interpreter and module
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        env={"PYTHONPATH": str(SRC)},
        check=True,
    )
    return json.loads(result.stdout.strip().splitlines()[-1])


class RuntimeIsolationTests(unittest.TestCase):
    def test_gateway_loads_no_adapter_and_no_pipecat(self) -> None:
        # The gateway runs in the Hermes image (Python 3.14), which has no Pipecat.
        loaded = _loaded_after(f"{PREFIX}.core.sidecar.gateway_server")
        self.assertEqual([m for m in loaded if m.startswith(f"{PREFIX}.adapters")], [])
        self.assertEqual([m for m in loaded if m.startswith("pipecat")], [])

    def test_each_adapter_loads_no_other_adapter(self) -> None:
        for adapter in ADAPTERS:
            with self.subTest(adapter=adapter):
                loaded = _loaded_after(f"{PREFIX}.adapters.{adapter}")
                others = tuple(f"{PREFIX}.adapters.{other}" for other in ADAPTERS if other != adapter)
                self.assertEqual([m for m in loaded if others and m.startswith(others)], [])

    def test_worker_imports_without_utils_or_catalog(self) -> None:
        loaded = _loaded_after(f"{PREFIX}.core.worker.worker_main")
        self.assertNotIn("utils", loaded)
        self.assertNotIn(f"{PREFIX}.core.catalog", loaded)
        self.assertEqual([m for m in loaded if m.startswith("pipecat")], [])


if __name__ == "__main__":
    unittest.main()
