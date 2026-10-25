# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Session pieces the adapter shares: session admission, the ``/health`` body, the filler query flag.

The repository's shared server (``src/server.py``) owns the HTTP app, TLS, the ``/v1/realtime``
route, its bearer check and the ``realtime`` subprotocol; these helpers keep what FDA's own
server did per session (``server_busy`` at ``server.max_sessions``, the health summary) unchanged.
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

# ``?x_nvidia_filler=1`` on the WebSocket URL: send unspoken filler as ``x_nvidia.filler``
# events. Off unless asked for, so a tau2 session sees the standard event stream only.
SHOW_FILLER_PARAM = "x_nvidia_filler"


def show_silent_filler(query: Mapping[str, str]) -> bool:
    """Whether the client asked for ``x_nvidia.filler`` events."""
    return query.get(SHOW_FILLER_PARAM, "").lower() in ("1", "true")


@dataclass(slots=True)
class SessionSlots:
    """Live session count against ``server.max_sessions``."""

    max_sessions: int
    sessions: int = 0
    total_sessions: int = 0

    def refusal(self, ready: bool) -> str:
        """Why a new session is refused (``""`` when it may start)."""
        if not ready:
            return "server is starting"
        if self.sessions >= self.max_sessions:
            return "server at capacity (server.max_sessions)"
        return ""

    def acquire(self) -> None:
        """Count a session that starts."""
        self.sessions += 1
        self.total_sessions += 1

    def release(self) -> None:
        """Count a session that ended."""
        self.sessions -= 1


def health_body(
    *,
    ready: bool,
    slots: SessionSlots,
    startup_error: str = "",
    extra: Callable[[], dict[str, Any]] | None = None,
) -> tuple[dict[str, Any], int]:
    """The ``/health`` JSON and status code (503 until ready)."""
    body: dict[str, Any] = {
        "status": "ok" if ready else "starting",
        "sessions": slots.sessions,
        "max_sessions": slots.max_sessions,
        "total_sessions": slots.total_sessions,
    }
    if startup_error:
        body["error"] = startup_error
    if extra is not None:
        with contextlib.suppress(Exception):  # a failing extra must not break /health
            body.update(extra())
    return body, 200 if ready else 503
