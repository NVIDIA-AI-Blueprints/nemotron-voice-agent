# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""Refuse to start when the installed Pipecat is not the version this source tree is written for.

The Compose services mount ``./src`` into the container, so the code is always current, while the
Python packages come from the image (``docker/Dockerfile`` runs ``uv sync --frozen``). An image built
before a Pipecat upgrade then runs new code on an old Pipecat, which fails in subtle ways (for
example, events dropped while a pipeline starts) instead of failing at startup. This check makes
that mismatch a clear startup error.

``REQUIRED_PIPECAT_VERSION`` must equal the ``pipecat-ai`` pin in ``pyproject.toml`` and
``uv.lock``; ``tests/unit/test_dependency_check.py`` enforces that.
"""

from __future__ import annotations

import importlib.metadata

#: The ``pipecat-ai`` version pinned in pyproject.toml and uv.lock.
REQUIRED_PIPECAT_VERSION = "1.11.0"


def pipecat_mismatch(installed: str | None = None) -> str | None:
    """A message when the installed ``pipecat-ai`` is not :data:`REQUIRED_PIPECAT_VERSION`, else ``None``."""
    if installed is None:
        try:
            installed = importlib.metadata.version("pipecat-ai")
        except importlib.metadata.PackageNotFoundError:
            installed = "not installed"
    if installed == REQUIRED_PIPECAT_VERSION:
        return None
    return (
        f"pipecat-ai {installed} is installed, but this source tree requires {REQUIRED_PIPECAT_VERSION}. "
        "The container image is older than the code: rebuild it (for example, add --build to your "
        "`docker compose ... up -d` command, or run `uv sync` for a host-native run)."
    )


def require_matching_pipecat() -> None:
    """Exit with a clear message when the installed Pipecat does not match the source tree."""
    message = pipecat_mismatch()
    if message is not None:
        raise SystemExit(message)
