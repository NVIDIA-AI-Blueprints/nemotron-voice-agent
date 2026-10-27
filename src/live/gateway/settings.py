# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Operator settings for the live endpoints."""

from __future__ import annotations

import os
from dataclasses import dataclass, field


def positive_setting(name: str, default: int) -> int:
    """The whole number the environment variable ``name`` holds, at least 1; an unset or empty one is ``default``.

    Raises:
        ValueError: The value is not a whole number, or is below 1: the server does not start.
    """
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
    """Operator settings for the live endpoints, read from the environment."""

    api_token: str = ""
    max_sessions: int = 8
    max_session_seconds: int = 1800
    ice_urls: tuple[str, ...] = field(default_factory=tuple)
    # Live sessions live in one server process, so the routes are served only with one uvicorn worker.
    workers: int = 1
    # The backstop of one console round (POST /v1/live/delegate): the core ends a round that runs past its own limit
    # (the backend LLM's ``first_event_timeout_seconds``, 45 s by default), so this is twice that.
    console_round_seconds: float = 90.0
    # The console route's limits: rounds at once, and the size of one round's body (its whole history).
    max_console_rounds: int = 2
    console_body_bytes: int = 512 * 1024
    # With no token, the names besides IP literals and ``localhost`` that the console route answers to: a request for
    # any other ``Host`` is refused, which stops DNS rebinding on plain HTTP.
    allowed_hosts: tuple[str, ...] = field(default_factory=tuple)

    @classmethod
    def from_env(cls) -> LiveSettings:
        """Read the live settings from the environment; an unset or empty value is the default.

        The variables: ``LIVE_API_TOKEN``, ``LIVE_MAX_SESSIONS``, ``LIVE_MAX_SESSION_SECONDS``, ``LIVE_ICE_URLS``,
        ``LIVE_MAX_CONSOLE_ROUNDS``, ``LIVE_ALLOWED_HOSTS`` (comma-separated) and ``UVICORN_WORKERS``.

        Raises:
            ValueError: A session or round limit or the worker count is not a whole number of at least 1.
        """
        return cls(
            api_token=os.getenv("LIVE_API_TOKEN", ""),
            max_sessions=positive_setting("LIVE_MAX_SESSIONS", cls.max_sessions),
            max_session_seconds=positive_setting("LIVE_MAX_SESSION_SECONDS", cls.max_session_seconds),
            ice_urls=listed("LIVE_ICE_URLS"),
            workers=positive_setting("UVICORN_WORKERS", cls.workers),
            max_console_rounds=positive_setting("LIVE_MAX_CONSOLE_ROUNDS", cls.max_console_rounds),
            allowed_hosts=tuple(name.lower() for name in listed("LIVE_ALLOWED_HOSTS")),
        )


def listed(name: str) -> tuple[str, ...]:
    """The comma-separated values the environment variable ``name`` holds, without blanks."""
    return tuple(value.strip() for value in os.getenv(name, "").split(",") if value.strip())
