# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""One throwaway ``HERMES_HOME`` per worker process (plan section 4.2).

Each home holds a rendered ``SOUL.md`` and a ``config.yaml`` with the settings every
worker needs (see ``host.py`` for why each is required). Homes are runtime state: they
live under ``workers.home_root`` and are never committed. :class:`HermesWorkerSetup` is the
Hermes spec's ``worker_setup``: the gateway calls it around every worker process.
"""

from __future__ import annotations

import contextlib
import shutil
from pathlib import Path
from typing import Any

import yaml

from examples.frontend_delegation_agent.core.backends.base import WorkerPrep

MARKER = ".fda-home"

#: ``config.yaml`` content every worker needs; ``model.context_length`` is added per config.
REQUIRED_HERMES_CONFIG: dict[str, Any] = {
    "tools": {"tool_search": {"enabled": "off"}},
    "agent": {
        "coding_context": "off",
        "task_completion_guidance": False,
        "parallel_tool_call_guidance": False,
        "tool_use_enforcement": False,
        "execution_guidance": False,
    },
}


def hermes_config(context_length: int) -> dict[str, Any]:
    """The worker's ``config.yaml`` mapping."""
    return {"model": {"context_length": int(context_length)}, **REQUIRED_HERMES_CONFIG}


def create_home(root: str | Path, name: str, *, soul: str, context_length: int) -> Path:
    """Create (or reset) ``root/name`` with ``SOUL.md`` and ``config.yaml``."""
    path = Path(root) / name
    if path.exists():
        shutil.rmtree(path, ignore_errors=True)
    path.mkdir(parents=True)
    (path / MARKER).write_text("fda worker home\n", encoding="utf-8")
    (path / "SOUL.md").write_text(soul.strip() + "\n", encoding="utf-8")
    (path / "config.yaml").write_text(yaml.safe_dump(hermes_config(context_length), sort_keys=False), encoding="utf-8")
    return path


def remove_home(path: str | Path | None) -> None:
    """Delete a home created by :func:`create_home` (only directories carrying the marker)."""
    if path is None:
        return
    target = Path(path)
    if (target / MARKER).exists():
        shutil.rmtree(target, ignore_errors=True)


def clear_stale(root: str | Path) -> int:
    """Remove homes left behind by a crashed gateway; returns how many were removed."""
    base = Path(root)
    if not base.is_dir():
        return 0
    removed = 0
    for child in base.iterdir():
        if child.is_dir() and (child / MARKER).exists():
            with contextlib.suppress(OSError):
                shutil.rmtree(child)
                removed += 1
    return removed


class HermesWorkerSetup:
    """``HERMES_HOME``, ``HERMES_YOLO_MODE`` and the Hermes checkout on ``PYTHONPATH`` for one worker."""

    def prepare(self, config: Any, name: str, *, soul: str) -> WorkerPrep:
        """Create the worker's home; Hermes freezes both variables at import time."""
        w = config.workers
        home = create_home(w.home_root, name, soul=soul, context_length=config.backend_settings.context_length)
        return WorkerPrep(
            home=home,
            env={"HERMES_HOME": str(home), "HERMES_YOLO_MODE": "1"},
            pythonpath=(w.hermes_repo,) if w.hermes_repo else (),
        )

    def remove(self, prep: WorkerPrep) -> None:
        """Delete the home (only directories carrying the marker)."""
        remove_home(prep.home)

    def clear_stale(self, config: Any) -> int:
        """Remove homes left behind by a crashed gateway."""
        return clear_stale(config.workers.home_root)
