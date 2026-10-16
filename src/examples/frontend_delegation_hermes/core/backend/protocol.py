# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""JSON messages between the voice server, the backend gateway and a Hermes worker (plan section 9).

Stdlib only: this module is the contract shared by all three processes (Python 3.12
voice server and gateway, Python 3.14 Hermes worker). A message is a JSON object with
``v`` (protocol version), ``type`` and the type's fields. Every direction has its own
table of types and required fields; :func:`decode` rejects anything else.
"""

from __future__ import annotations

import json
from typing import Any, Final

PROTOCOL_VERSION: Final = 1

#: Direction names.
VOICE_TO_GATEWAY: Final = "voice->gateway"
GATEWAY_TO_VOICE: Final = "gateway->voice"
GATEWAY_TO_WORKER: Final = "gateway->worker"
WORKER_TO_GATEWAY: Final = "worker->gateway"

#: Required fields per message type and direction (optional fields are allowed and passed through).
SPECS: Final[dict[str, dict[str, tuple[str, ...]]]] = {
    VOICE_TO_GATEWAY: {
        "session.open": ("session_id", "settings"),
        "session.configure": ("tools", "instructions"),
        "history.append": ("entries",),
        "delegate": ("turn_id", "request", "run_input"),
        "tool.result": ("call_id", "output"),
        "session.close": (),
    },
    GATEWAY_TO_VOICE: {
        "session.ready": ("worker_pid",),
        "session.configured": (),
        "history.ack": ("committed_upto",),
        "state": ("state", "epoch"),
        "action": ("turn_id", "kind"),
        "tool.call": ("call_id", "epoch", "name", "arguments"),
        "tool.cancel": ("call_id", "reason"),
        "activity": ("epoch", "event", "name"),
        "answer": ("answer_id", "epoch", "kind", "text"),
        "backend_run_done": ("run_id", "epoch", "status", "history"),
        "status": ("turn_id", "summary"),
        "worker": ("event",),
        "error": ("code", "message", "fatal"),
    },
    GATEWAY_TO_WORKER: {
        "configure": ("tools", "instructions", "hermes"),
        "run": ("epoch", "task_id", "user_message", "conversation_history"),
        "steer": ("epoch", "text"),
        "redirect": ("epoch", "text"),
        "status": ("epoch",),
        "tool.result": ("call_id", "output"),
        "interrupt": ("hard",),
        "close": (),
    },
    WORKER_TO_GATEWAY: {
        "ready": ("pid",),
        "configured": (),
        "steer_result": ("epoch", "accepted", "via"),
        "status_result": ("epoch", "summary"),
        "tool.call": ("call_id", "epoch", "name", "arguments"),
        "tool.cancel": ("call_id", "reason"),
        "activity": ("epoch", "event", "name"),
        "run_outcome": ("epoch", "status"),
        "closed": (),
        "error": ("phase", "message"),
    },
}

#: Backend states (plan section 5.1).
NO_SESSION: Final = "NO_SESSION"
IDLE: Final = "IDLE"
WORKING: Final = "WORKING"
CLOSING: Final = "CLOSING"
CLOSED: Final = "CLOSED"

#: ``delegate.request`` values (RC1).
REQUEST_TASK: Final = "task"
REQUEST_STATUS: Final = "status"
REQUESTS: Final = (REQUEST_TASK, REQUEST_STATUS)

#: ``run_outcome.status`` / ``backend_run_done.status`` values (plan section 5.3).
RUN_STATUSES: Final = ("ok", "failed", "interrupted", "error", "deadline", "worker_died", "closed")

#: Transcript entry origins and playback outcomes (plan section 7).
ORIGINS: Final = ("user", "frontend", "controller", "derived")
OUTCOMES: Final = ("heard", "partial", "not_heard")


class ProtocolError(ValueError):
    """A message that does not match the contract."""


def message(direction: str, type_: str, **fields: Any) -> dict[str, Any]:
    """Build and validate one message as a dict."""
    data = {"v": PROTOCOL_VERSION, "type": type_, **fields}
    validate(direction, data)
    return data


def encode(direction: str, type_: str, **fields: Any) -> str:
    """Build, validate and serialize one message (no trailing newline)."""
    return dumps(message(direction, type_, **fields))


def dumps(data: dict[str, Any]) -> str:
    """Serialize an already-built message compactly."""
    return json.dumps(data, separators=(",", ":"), ensure_ascii=False, default=str)


def decode(direction: str, raw: str | bytes) -> dict[str, Any]:
    """Parse and validate one message; raises :class:`ProtocolError`."""
    try:
        data = json.loads(raw)
    except (TypeError, ValueError) as exc:
        raise ProtocolError(f"not JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise ProtocolError("a message must be a JSON object")
    validate(direction, data)
    return data


def validate(direction: str, data: dict[str, Any]) -> None:
    """Check version, type and required fields for ``direction``."""
    specs = SPECS.get(direction)
    if specs is None:
        raise ProtocolError(f"unknown direction {direction!r}")
    if data.get("v") != PROTOCOL_VERSION:
        raise ProtocolError(f"protocol version {data.get('v')!r} is not supported (expected {PROTOCOL_VERSION})")
    type_ = data.get("type")
    if type_ not in specs:
        raise ProtocolError(f"unknown message type {type_!r} for {direction}")
    missing = [name for name in specs[type_] if name not in data]
    if missing:
        raise ProtocolError(f"{type_}: missing field(s) {missing}")
