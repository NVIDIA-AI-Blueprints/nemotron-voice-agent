# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""Canonical OpenAI Realtime protocol primitives.

This module intentionally contains no pre-GA event aliases. Callers get one
canonical event for every state transition and a structured protocol error for
every rejected operation.
"""

from __future__ import annotations

import json
import math
import uuid
from dataclasses import dataclass
from typing import Any

MAX_CLIENT_EVENT_ID_LENGTH = 512
# OpenAI Realtime permits one input_audio_buffer.append payload up to 15 MiB.
MAX_AUDIO_APPEND_BYTES = 15 * 1024 * 1024
# A wire event carrying the maximum decoded audio needs its Base64 expansion,
# plus bounded JSON envelope space for the type and optional event ID.
MAX_REALTIME_EVENT_BYTES = 4 * math.ceil(MAX_AUDIO_APPEND_BYTES / 3) + 64 * 1024


def _reject_non_finite_json(value: str) -> None:
    raise ValueError(f"Non-finite JSON number {value!r} is not valid")


def strict_json_loads(value: str | bytes | bytearray) -> Any:
    """Decode standards-compliant JSON, rejecting Python's NaN/Infinity extensions."""
    return json.loads(value, parse_constant=_reject_non_finite_json)


def validate_client_event_id(value: Any) -> str | None:
    """Validate an optional OpenAI Realtime client-event correlation ID."""
    if value is None:
        return None
    if not isinstance(value, str):
        raise invalid_type("event_id must be a string", param="event_id")
    if len(value) > MAX_CLIENT_EVENT_ID_LENGTH:
        raise invalid_value(
            f"event_id must contain at most {MAX_CLIENT_EVENT_ID_LENGTH} characters",
            param="event_id",
        )
    return value


def new_realtime_id(prefix: str) -> str:
    """Create a stable, opaque identifier for one Realtime resource."""
    if not prefix or not prefix.replace("_", "").isalnum():
        raise ValueError("Realtime ID prefix must contain only letters, numbers, or underscores")
    return f"{prefix}_{uuid.uuid4().hex}"


def build_server_event(event_type: str, **payload: Any) -> dict[str, Any]:
    """Build one canonical server event with a unique server event ID."""
    if not isinstance(event_type, str) or not event_type:
        raise ValueError("event_type must be a non-empty string")
    event: dict[str, Any] = {
        "event_id": new_realtime_id("event"),
        "type": event_type,
    }
    event.update(payload)
    return event


@dataclass(slots=True)
class RealtimeProtocolError(ValueError):
    """A client-visible Realtime protocol failure.

    ``event_id`` is the optional ID of the client event that failed.  The
    generated server event always receives its own independent ID.
    """

    message: str
    code: str
    param: str | None = None
    event_id: str | None = None
    error_type: str = "invalid_request_error"

    def __post_init__(self) -> None:
        """Initialize the inherited exception message."""
        ValueError.__init__(self, self.message)

    def to_event(self) -> dict[str, Any]:
        """Return the canonical ``error`` server event for this failure."""
        error: dict[str, Any] = {
            "type": self.error_type,
            "code": self.code,
            "message": self.message,
        }
        if self.param is not None:
            error["param"] = self.param
        if self.event_id is not None:
            error["event_id"] = self.event_id
        return build_server_event("error", error=error)

    def with_event_id(self, event_id: str | None) -> RealtimeProtocolError:
        """Copy the error with the client event ID that caused it."""
        return RealtimeProtocolError(
            message=self.message,
            code=self.code,
            param=self.param,
            event_id=event_id,
            error_type=self.error_type,
        )


def invalid_type(message: str, *, param: str) -> RealtimeProtocolError:
    """Build a strict type-validation failure."""
    return RealtimeProtocolError(message=message, code="invalid_type", param=param)


def invalid_value(message: str, *, param: str) -> RealtimeProtocolError:
    """Build a strict value-validation failure."""
    return RealtimeProtocolError(message=message, code="invalid_value", param=param)


def unsupported_capability(message: str, *, param: str) -> RealtimeProtocolError:
    """Build an explicit server-capability failure."""
    return RealtimeProtocolError(message=message, code="unsupported_capability", param=param)


def immutable_field(message: str, *, param: str) -> RealtimeProtocolError:
    """Build an immutable-session-field failure."""
    return RealtimeProtocolError(message=message, code="immutable_field", param=param)
