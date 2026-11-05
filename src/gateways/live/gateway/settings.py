# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Operator settings for the live endpoints."""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
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
    """Operator settings for the live endpoints: the serving example's values, the token and the worker count."""

    # Kept out of the repr, so a logged traceback that shows the settings never shows the token.
    api_token: str = field(default="", repr=False)
    max_sessions: int = 8
    max_session_seconds: int = 1800
    # The context appends one session accepts in any minute, the three kinds together: a limit of this server's own.
    max_appends_per_minute: int = 64
    ice_urls: tuple[str, ...] = field(default_factory=tuple)
    # Live sessions live in one server process, so the routes are served only with one uvicorn worker.
    workers: int = 1
    # The backstop of one delegate round (POST /v1/live/delegate): the core ends a round that runs past its own limit
    # (the backend LLM's ``first_event_timeout_seconds``, 45 s by default), so this is twice that.
    delegate_round_seconds: float = 90.0
    # The delegate route's limits: rounds at once, and the size of one round's body (its whole history).
    max_delegate_rounds: int = 2
    delegate_body_bytes: int = 512 * 1024
    # With no token, the names besides IP literals and ``localhost`` that the live routes answer to: a request for any
    # other ``Host`` is refused, which stops DNS rebinding on plain HTTP.
    allowed_hosts: tuple[str, ...] = field(default_factory=tuple)

    @classmethod
    def from_env(cls, values: Mapping[str, int | Sequence[str]] | None = None) -> LiveSettings:
        """Build the settings from ``values`` and the environment; a key that ``values`` lacks keeps its default.

        ``values``, which the root passes in from the serving example, holds the limits as whole numbers and
        ``ice_urls`` and ``allowed_hosts`` as lists of strings; each string is stripped, and a blank one is dropped.
        The environment gives ``LIVE_API_TOKEN`` and ``UVICORN_WORKERS``.

        Raises:
            ValueError: ``UVICORN_WORKERS`` is not a whole number of at least 1.
        """
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
    """The strings of ``values``, stripped, without blanks."""
    return tuple(value.strip() for value in values if value.strip())
