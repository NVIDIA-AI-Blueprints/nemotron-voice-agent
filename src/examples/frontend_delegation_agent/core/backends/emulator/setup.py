# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A scratch home per emulator worker process, with the same lifecycle as a Hermes home (stdlib only).

Emulator process workers stand in for Hermes in the gateway's process tests, so they get a
throwaway directory under ``workers.home_root`` that the gateway creates and removes the
same way. It carries the Hermes home marker, so either backend's stale-home cleanup removes it.
"""

from __future__ import annotations

import contextlib
import shutil
from pathlib import Path
from typing import Any

from examples.frontend_delegation_agent.core.backends.base import WorkerPrep

#: The marker of a gateway-created worker home (the same file as ``hermes/home.py``).
MARKER = ".fda-home"


class EmulatorWorkerSetup:
    """Creates and removes ``workers.home_root/<name>``; no environment variables."""

    def prepare(self, config: Any, name: str, *, soul: str) -> WorkerPrep:
        """Create (or reset) the worker's scratch home."""
        del soul
        path = Path(config.workers.home_root) / name
        shutil.rmtree(path, ignore_errors=True)
        path.mkdir(parents=True)
        (path / MARKER).write_text("fda worker home\n", encoding="utf-8")
        return WorkerPrep(home=path)

    def remove(self, prep: WorkerPrep) -> None:
        """Delete the home (only a directory carrying the marker)."""
        if prep.home is not None and (prep.home / MARKER).exists():
            shutil.rmtree(prep.home, ignore_errors=True)

    def clear_stale(self, config: Any) -> int:
        """Remove homes left behind by a crashed gateway."""
        base = Path(config.workers.home_root)
        removed = 0
        for child in base.iterdir() if base.is_dir() else ():
            if child.is_dir() and (child / MARKER).exists():
                with contextlib.suppress(OSError):
                    shutil.rmtree(child)
                    removed += 1
        return removed
