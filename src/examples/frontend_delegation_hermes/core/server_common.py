# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Server pieces shared by every head: auth, capacity, ``/health``, TLS, the browser page.

Each head owns its FastAPI app and transport; these helpers keep the parts every
OpenAI Realtime client sees (bearer check, ``server_busy``, ``?model=`` tag,
``/health`` body, the ``realtime`` subprotocol) identical across adapters.
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from examples.frontend_delegation_hermes.core.voice_config import VoiceConfig

# Browser microphone client served at "/" (it connects back to server.path on the same origin).
WEB_PAGE = Path(__file__).resolve().parents[1] / "web" / "index.html"

# ``?x_nvidia_filler=1`` on the WebSocket URL: send unspoken filler as ``x_nvidia.filler``
# events. Off unless asked for, so a tau2 session sees the standard event stream only.
SHOW_FILLER_PARAM = "x_nvidia_filler"


def bearer_ok(headers: Mapping[str, str], config: VoiceConfig) -> bool:
    """Whether the ``Authorization`` header satisfies ``server.require_bearer``."""
    if not config.server.require_bearer:
        return True
    header = headers.get("authorization", "")
    token = header[7:].strip() if header.lower().startswith("bearer ") else ""
    return bool(config.server.bearer_token) and token == config.server.bearer_token


def realtime_subprotocol(headers: Mapping[str, str]) -> str | None:
    """``realtime`` when the client offered it (the OpenAI SDKs do), else no subprotocol."""
    offered = [value.strip() for value in headers.get("sec-websocket-protocol", "").split(",")]
    return "realtime" if "realtime" in offered else None


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


def tls_files(cert: str = "", key: str = "") -> dict[str, str]:
    """Uvicorn's certificate and key: the given pair, or the stock app's self-signed one in ``.certs/``."""
    if bool(cert) != bool(key):
        raise SystemExit("--tls-cert and --tls-key must be given together")
    if cert:
        return {"ssl_certfile": cert, "ssl_keyfile": key}
    from utils import ensure_self_signed_cert  # noqa: PLC0415 - src/utils.py, as used by src/server.py

    certfile, keyfile = ensure_self_signed_cert(Path.cwd() / ".certs")
    return {"ssl_certfile": certfile, "ssl_keyfile": keyfile}
