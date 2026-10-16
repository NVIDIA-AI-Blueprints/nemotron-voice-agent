# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""Every other example and the shared server load without the Frontend/Backend Agent package.

Plan ``misc/frontend-backend-verdict/fba-removal-plan.md``, phase 2: in a fresh interpreter where
any import of ``examples.frontend_backend_agent`` fails, the server modules, every ``realtime``
module and every registered example's bot module still import.
"""

# ruff: noqa: D102

from __future__ import annotations

import os
import subprocess
import sys
import unittest
from pathlib import Path

import yaml

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_BLOCKED = "examples.frontend_backend_agent"


def _modules() -> list[str]:
    modules = ["server", "realtime_server", "examples_registry", "utils"]
    modules += sorted(
        f"realtime.{path.stem}" for path in (_PROJECT_ROOT / "src" / "realtime").glob("*.py") if path.stem != "__init__"
    )
    registry = yaml.safe_load((_PROJECT_ROOT / "examples_registry.yaml").read_text(encoding="utf-8"))
    bots = {str(example.get("bot", "")).split(":", 1)[0] for example in registry["examples"].values()}
    modules += sorted(bot for bot in bots if bot and not bot.startswith(_BLOCKED))
    return modules


class ExamplesWithoutFbaTests(unittest.TestCase):
    """Import checks with the Frontend/Backend Agent package blocked."""

    def test_modules_import_with_the_fba_package_blocked(self) -> None:
        source = f"""
import importlib
import sys
from importlib.abc import MetaPathFinder


class Block(MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name == {_BLOCKED!r} or name.startswith({_BLOCKED!r} + "."):
            raise ImportError(f"blocked: {{name}}")
        return None


sys.meta_path.insert(0, Block())
failed = []
for module in {_modules()!r}:
    try:
        importlib.import_module(module)
    except Exception as exc:  # noqa: BLE001
        failed.append(f"{{module}}: {{type(exc).__name__}}: {{exc}}")
if failed:
    raise SystemExit("\\n".join(failed))
"""
        env = os.environ.copy()
        src_path = str(_PROJECT_ROOT / "src")
        env["PYTHONPATH"] = src_path + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        env["EXAMPLE_SELECTION"] = "all"
        env["TRANSPORT_SELECTION"] = "all"
        completed = subprocess.run(
            [sys.executable, "-c", source],
            cwd=_PROJECT_ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stdout[-2000:] + completed.stderr[-4000:])


if __name__ == "__main__":
    unittest.main()
