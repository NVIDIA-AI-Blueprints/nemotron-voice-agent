# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Server event construction: every event carries a fresh ``event_id``; rejections become ``error`` events."""

from __future__ import annotations

from typing import Any

from live.protocol.errors import ProtocolError, uid


def event(kind: str, **fields: Any) -> dict:
    """Build a server event with a fresh ``event_id``."""
    return {"type": kind, "event_id": uid("event"), **fields}


def error_event(exc: ProtocolError, client_event_id: str | None = None) -> dict:
    """Build the ``error`` event for ``exc``, echoing the client event it answers."""
    return event(
        "error",
        error={
            "type": "invalid_request_error",
            "code": exc.code,
            "message": str(exc),
            "param": exc.param,
            "client_event_id": client_event_id,
        },
    )
