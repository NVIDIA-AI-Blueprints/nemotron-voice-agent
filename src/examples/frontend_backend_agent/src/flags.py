# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""Feature switches of the Frontend/Backend Agent, read per session.

Every switch is a kill switch: the shipped default is the behaviour this
example is evaluated with, and the same default applies to every domain. A
value is read when a session starts, so a deployment change applies to the
next session.
"""

from __future__ import annotations

from utils import parse_env_bool

#: Environment variable -> shipped default.
FLAG_DEFAULTS: dict[str, bool] = {
    "FRONTEND_BACKEND_FRONTEND_VERDICT": True,
    "FRONTEND_BACKEND_BACKEND_HISTORY": True,
    "FRONTEND_BACKEND_LATE_ANSWERS": True,
    "FRONTEND_BACKEND_PENDING_QUESTION": True,
    "FRONTEND_BACKEND_NORMALIZATION": True,
    "FRONTEND_BACKEND_PHONE_FORMAT": True,
    "FRONTEND_BACKEND_DIRECT_WRITE": True,
    "FRONTEND_BACKEND_DONE_GUARD": True,
    "FRONTEND_BACKEND_REALTIME_TOOL_ROUNDS": True,
}


def flag(name: str) -> bool:
    """Return one switch's effective value."""
    return parse_env_bool(name, FLAG_DEFAULTS[name])


#: Switches that act only through the backend history (its ledger).
_NEEDS_BACKEND_HISTORY = (
    "FRONTEND_BACKEND_LATE_ANSWERS",
    "FRONTEND_BACKEND_DIRECT_WRITE",
    "FRONTEND_BACKEND_DONE_GUARD",
)


def effective_flags() -> dict[str, bool]:
    """Return every switch's effective value, for session metadata and logs.

    A switch that needs the backend history reports off while history is off.
    """
    values = {name: flag(name) for name in FLAG_DEFAULTS}
    if not values["FRONTEND_BACKEND_BACKEND_HISTORY"]:
        values.update(dict.fromkeys(_NEEDS_BACKEND_HISTORY, False))
    return values
