# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Operator settings for the live endpoints."""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field


def positive_setting(name: str, default: int) -> int:
    """Return the whole number of at least 1 that environment variable ``name`` holds, or ``default`` if it is empty."""
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        raise ValueError(f"{name} must be a whole number of at least 1, not {raw!r}") from None
    if value < 1:
        raise ValueError(f"{name} must be at least 1, not {value}")
    return value


@dataclass(frozen=True)
class LiveSettings:
    """Operator settings for the live endpoints: the serving example's values, the token and the worker count."""

    # Kept out of the repr, so a logged traceback that shows the settings never shows the token.
    api_token: str = field(default="", repr=False)
    max_sessions: int = 8
    max_session_seconds: int = 1800
    # A limit of this server's own, the three kinds together: the Live API limits only each append's size.
    max_appends_per_minute: int = 64
    ice_urls: tuple[str, ...] = field(default_factory=tuple)
    # Live sessions live in one server process, so the routes are served only with one uvicorn worker.
    workers: int = 1
    # The backstop of one delegate round, for a core that does not end the round within its own limits.
    delegate_round_seconds: float = 90.0
    # The delegate route's limits: the rounds at once, and the bytes of one round's body, the client's whole history.
    max_delegate_rounds: int = 2
    delegate_body_bytes: int = 512 * 1024
    # With no token, the names the routes answer besides IP literals and localhost, against DNS rebinding.
    allowed_hosts: tuple[str, ...] = field(default_factory=tuple)

    @classmethod
    def from_env(cls, values: Mapping[str, int | Sequence[str]] | None = None) -> LiveSettings:
        """Build the settings from the example's ``values``, and the token and the worker count from the environment."""
        values = dict(values or {})
        return cls(
            **{
                **values,
                "ice_urls": listed(values.get("ice_urls", ())),
                "allowed_hosts": tuple(name.lower() for name in listed(values.get("allowed_hosts", ()))),
            },
            api_token=os.getenv("LIVE_API_TOKEN", "").strip(),
            workers=positive_setting("UVICORN_WORKERS", cls.workers),
        )


def listed(values: Sequence[str]) -> tuple[str, ...]:
    """Return the strings of ``values``, stripped, without blanks."""
    return tuple(value.strip() for value in values if value.strip())
