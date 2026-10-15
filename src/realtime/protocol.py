# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""Client-visible Realtime protocol errors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from realtime.events import SERVER_ERROR, server_event


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
        """Return the ``error`` server event for this failure."""
        error: dict[str, Any] = {
            "type": self.error_type,
            "code": self.code,
            "message": self.message,
        }
        if self.param is not None:
            error["param"] = self.param
        if self.event_id is not None:
            error["event_id"] = self.event_id
        return server_event(SERVER_ERROR, error=error)

    def with_event_id(self, event_id: str | None) -> RealtimeProtocolError:
        """Copy the error with the client event ID that caused it."""
        return RealtimeProtocolError(
            message=self.message,
            code=self.code,
            param=self.param,
            event_id=event_id,
            error_type=self.error_type,
        )
