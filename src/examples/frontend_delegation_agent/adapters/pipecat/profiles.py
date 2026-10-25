# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``FDA_PROFILE``: the two deployment profiles and the files each one runs.

=========================  ==================================  ======================================
                           ``default`` (browser UI)            ``eval`` (OpenAI Realtime, benchmarks)
=========================  ==================================  ======================================
Delegation profile         ``profiles/pipecat_demo.yaml``      ``profiles/pipecat_realtime_eval.yaml``
Pipecat settings           ``adapters/pipecat.yaml``           ``adapters/pipecat.eval.yaml``
Gateway config             ``gateway.yaml``                    ``gateway.eval.yaml``
=========================  ==================================  ======================================

The same rule as the Frontend/Backend Agent example: OpenAI Realtime sessions always run the
``eval`` row, whatever ``FDA_PROFILE`` says, and a deployment with ``FDA_PROFILE=eval`` refuses
browser sessions. ``FDA_PROFILE`` unset means ``default``; any other value is an error.

The gateway config is not read here: the Hermes runtime (``fda-hermes``) loads it from its own
``FDA_GATEWAY_CONFIG``, which the eval Compose overlay sets with ``FDA_PROFILE``.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

CONFIG_DIR = Path(__file__).resolve().parents[2] / "config"
PROFILE_ENV = "FDA_PROFILE"
DEFAULT_PROFILE = "default"
#: The profile every OpenAI Realtime session runs.
REALTIME_PROFILE = "eval"


class ProfileError(RuntimeError):
    """``FDA_PROFILE`` or the server's WebSocket keepalive does not fit a profile."""


@dataclass(frozen=True, slots=True)
class Profile:
    """The files one profile runs (paths relative to ``config/``)."""

    name: str
    delegation: str
    pipecat_settings: str
    gateway: str

    @property
    def delegation_path(self) -> Path:
        """The delegation profile file."""
        return CONFIG_DIR / self.delegation


PROFILES: dict[str, Profile] = {
    DEFAULT_PROFILE: Profile(
        name=DEFAULT_PROFILE,
        delegation="profiles/pipecat_demo.yaml",
        pipecat_settings="adapters/pipecat.yaml",
        gateway="gateway.yaml",
    ),
    REALTIME_PROFILE: Profile(
        name=REALTIME_PROFILE,
        delegation="profiles/pipecat_realtime_eval.yaml",
        pipecat_settings="adapters/pipecat.eval.yaml",
        gateway="gateway.eval.yaml",
    ),
}


def profile_name(environ: Mapping[str, str] | None = None) -> str:
    """The deployment's profile (``FDA_PROFILE``; unset means ``default``)."""
    env = os.environ if environ is None else environ
    name = (env.get(PROFILE_ENV) or "").strip() or DEFAULT_PROFILE
    if name not in PROFILES:
        raise ProfileError(f"{PROFILE_ENV}={name!r} is not one of {sorted(PROFILES)}")
    return name


def keepalive_mismatch(ws_ping_interval_s: float, environ: Mapping[str, str] | None = None) -> str:
    """A message when the server pings WebSocket clients but the profile turns pings off, else ``""``.

    FDA's profiles set ``server.ws_ping_interval_s: 0``: a τ² client sends no pings and pauses for
    seconds, so the server must not ping it. ``src/server.py`` passes ``UVICORN_WS_PING_INTERVAL`` to
    uvicorn; unset, uvicorn pings every 20 s.
    """
    if ws_ping_interval_s:
        return ""
    env = os.environ if environ is None else environ
    raw = (env.get("UVICORN_WS_PING_INTERVAL") or "").strip()
    try:
        interval = float(raw) if raw else 20.0
    except ValueError:
        interval = 20.0
    if interval == 0:
        return ""
    return (
        f"the server pings WebSocket clients every {interval:g} s, but this profile turns pings off "
        "(server.ws_ping_interval_s: 0); set UVICORN_WS_PING_INTERVAL=0"
    )
