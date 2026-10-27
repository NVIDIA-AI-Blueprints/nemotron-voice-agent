# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Server event construction: every event carries a fresh ``event_id``; rejections become ``error`` events.

An optional field the server has no value for is omitted, not sent as ``null``. :func:`error_event`
is the one builder of ``error`` events: its ``type`` follows its code, so a server fault is never labelled as the
client's mistake.
"""

from __future__ import annotations

from typing import Any

from live.protocol.errors import ProtocolError, uid

# The codes of failures on the server's side; every other code is an invalid client command.
SERVER_CODES = frozenset({"server_error", "pipeline_error"})


def event(kind: str, **fields: Any) -> dict:
    """Build a server event with a fresh ``event_id``; a field whose value is ``None`` is left out."""
    return {
        "type": kind,
        "event_id": uid("event"),
        **{name: value for name, value in fields.items() if value is not None},
    }


def error_event(exc: ProtocolError, client_event_id: str | None = None) -> dict:
    """Build the ``error`` event for ``exc``, echoing the client event it answers."""
    error = {
        "type": "server_error" if exc.code in SERVER_CODES else "invalid_request_error",
        "code": exc.code,
        "message": str(exc),
        "param": exc.param,
        "client_event_id": client_event_id,
    }
    return event(
        "error",
        error={name: value for name, value in error.items() if value is not None},
        client_event_id=client_event_id,
    )
