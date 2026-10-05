# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Runtime context helpers for the Frontend/Backend Agent."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from datetime import date, datetime


def runtime_today() -> date:
    """Return today's date, with an explicit override for deterministic evals."""
    override = os.getenv("FRONTEND_BACKEND_AGENT_TODAY", "").strip()
    if not override:
        return date.today()
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", override):
        raise ValueError("FRONTEND_BACKEND_AGENT_TODAY must use YYYY-MM-DD format")
    try:
        return date.fromisoformat(override)
    except ValueError as exc:
        raise ValueError("FRONTEND_BACKEND_AGENT_TODAY must use YYYY-MM-DD format") from exc


@dataclass(frozen=True, slots=True)
class SessionClock:
    """The current date (and optionally time) a session's own instructions state."""

    date: date
    time: str | None = None
    timezone: str | None = None


#: A sentence must say it states the present before its date is taken as today.
_PRESENT_CUE_RE = re.compile(r"\b(?:current|currently|today|today's|now|present)\b", re.IGNORECASE)
_SESSION_DATETIME_RE = re.compile(
    r"(?<!\d)(?P<date>\d{4}-\d{2}-\d{2})"
    r"(?:[T ](?P<time>\d{2}:\d{2}(?::\d{2})?))?"
    r"(?:\s*(?P<tz>Z|UTC[+-]\d{1,2}(?::\d{2})?|[A-Z]{2,5})\b)?"
)
_SENTENCE_SPLIT_RE = re.compile(r"\n+|(?<=[.!?])\s+")


def session_clock_from_instructions(instructions: object) -> SessionClock | None:
    """Return the date a client's instructions give as the present, if any.

    Reads a copy of the text; the instructions themselves are never changed.
    The first sentence that both mentions the present ("current", "today",
    "now") and carries an ISO date wins.
    """
    if not isinstance(instructions, str) or not instructions:
        return None
    for sentence in _SENTENCE_SPLIT_RE.split(instructions):
        if not _PRESENT_CUE_RE.search(sentence):
            continue
        for match in _SESSION_DATETIME_RE.finditer(sentence):
            try:
                day = date.fromisoformat(match.group("date"))
            except ValueError:
                continue
            return SessionClock(date=day, time=match.group("time"), timezone=match.group("tz"))
    return None


def session_today(clock: SessionClock | None) -> date:
    """Return today: the explicit override, then the session's stated date, then the clock."""
    if os.getenv("FRONTEND_BACKEND_AGENT_TODAY", "").strip():
        return runtime_today()
    return clock.date if clock is not None else runtime_today()


def session_runtime_fields(clock: SessionClock | None) -> dict[str, str]:
    """Return the Thinker's ``runtime_context`` fields under the same precedence as :func:`session_today`."""
    if os.getenv("FRONTEND_BACKEND_AGENT_TODAY", "").strip():
        return {"date": runtime_today().isoformat(), "date_source": "override"}
    if clock is not None:
        fields = {"date": clock.date.isoformat(), "date_source": "session"}
        if clock.time is not None:
            fields["local_datetime"] = f"{clock.date.isoformat()}T{clock.time}"
        if clock.timezone is not None:
            fields["timezone"] = clock.timezone
        return fields
    now = datetime.now().astimezone()
    return {
        "local_datetime": now.isoformat(timespec="seconds"),
        "date": now.date().isoformat(),
        "timezone": str(now.tzinfo),
    }
