# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""Provider-neutral types for temporary response-only backends."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

RESULT_ENVELOPE_SCHEMA = "voiceclaw.result.v1"
REALTIME_PROJECTION_SCHEMA = "voiceclaw.projection.v1"
RUNTIME_PROJECTION_SCHEMA = "voiceclaw.runtime.v4"


class ResponseOnlyRequestState(StrEnum):
    """VoiceClaw-local lifecycle for one non-durable backend request."""

    IDLE = "idle"
    LOCALLY_QUEUED = "locally_queued"
    DISPATCHING = "dispatching"
    WAITING_FOR_RESPONSE = "waiting_for_response"
    SUCCEEDED = "succeeded"
    FAILED = "failed"

    @property
    def active(self) -> bool:
        """Return whether the request can still produce a terminal response."""
        return self in {
            self.LOCALLY_QUEUED,
            self.DISPATCHING,
            self.WAITING_FOR_RESPONSE,
        }

    @property
    def terminal(self) -> bool:
        """Return whether the local request lifecycle has ended."""
        return self in {self.SUCCEEDED, self.FAILED}


class ResponseOnlyTerminalOutcome(StrEnum):
    """Terminal evidence projected for a non-durable request."""

    NONE = "none"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class ResponseOnlyResultState(StrEnum):
    """Availability of a response-only result in the local projection."""

    NONE = "none"
    AVAILABLE = "available"


class ResponseOnlyResultEventKind(StrEnum):
    """Events produced while a response-only result is assembled."""

    DISPLAY_DELTA = "display_delta"
    COMPLETED = "completed"
    DISCARDED = "discarded"


class ResponseOnlyUpdateKind(StrEnum):
    """Stable projection categories emitted by the response-only runtime."""

    BACKEND_TARGET = "backend_target"
    BACKEND_TURN = "backend_turn"
    RESULT_DISPLAY = "result_display"
    DELIVERY_QUEUE = "delivery_queue"


class ResponseOnlySpeechSource(StrEnum):
    """Provenance of spoken-presentation material in a response-only result."""

    NONE = "none"
    BACKEND_AUTHORED = "backend_authored"


@dataclass(frozen=True, slots=True)
class ResponseOnlyResultEnvelope:
    """Backend result with independent presentation-material and display channels."""

    display: str
    speech: str | None = None
    schema: str = RESULT_ENVELOPE_SCHEMA

    def __post_init__(self) -> None:
        """Reject ambiguous, empty, NUL-bearing, or non-UTF-8 channel content."""
        if self.schema != RESULT_ENVELOPE_SCHEMA:
            raise ValueError("unsupported result envelope schema")
        if not isinstance(self.display, str) or not self.display.strip() or "\x00" in self.display:
            raise ValueError("display must be non-empty Markdown without NUL")
        if self.speech is not None and (
            not isinstance(self.speech, str) or not self.speech.strip() or "\x00" in self.speech
        ):
            raise ValueError("speech must be null or non-empty text without NUL")
        try:
            self.display.encode("utf-8")
            if self.speech is not None:
                self.speech.encode("utf-8")
        except UnicodeEncodeError as exc:
            raise ValueError("result envelope text must be valid UTF-8") from exc


__all__ = [
    "REALTIME_PROJECTION_SCHEMA",
    "RESULT_ENVELOPE_SCHEMA",
    "RUNTIME_PROJECTION_SCHEMA",
    "ResponseOnlyRequestState",
    "ResponseOnlyResultEventKind",
    "ResponseOnlyResultEnvelope",
    "ResponseOnlyResultState",
    "ResponseOnlySpeechSource",
    "ResponseOnlyTerminalOutcome",
    "ResponseOnlyUpdateKind",
]
