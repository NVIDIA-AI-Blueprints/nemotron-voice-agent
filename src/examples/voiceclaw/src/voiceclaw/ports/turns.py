# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""Ephemeral committed-turn boundary for response-only backends.

This deliberately is not the durable :class:`AgentInteractionPort`. A backend
implementing this port admits one committed user turn and returns one terminal
display response; it makes no claim about Work, replay, receipts, or recovery.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Protocol

from voiceclaw.domain.models import BackendCapabilities, CapabilityEvidence, CapabilitySource
from voiceclaw.domain.response_only import ResponseOnlyResultEventKind

# Public committed goals are bounded independently from any adapter-specific
# envelope used to carry them. Adapters must preserve every accepted byte and
# apply their own, separately named wire-payload bound after encoding.
MAX_COMMITTED_TURN_GOAL_BYTES = 48 * 1024


def _required(value: str, name: str) -> None:
    if not isinstance(value, str) or not value.strip() or "\x00" in value:
        raise ValueError(f"{name} must be a non-empty string without NUL")


class CommittedTurnError(RuntimeError):
    """Safe, provider-neutral failure returned by a committed-turn adapter."""

    def __init__(self, code: str) -> None:
        """Retain only a bounded machine-readable failure code."""
        if (
            not isinstance(code, str)
            or not code
            or len(code) > 128
            or not code[0].isalpha()
            or any(character not in "abcdefghijklmnopqrstuvwxyz0123456789_" for character in code)
        ):
            code = "committed_turn_error"
        self.code = code
        super().__init__(code)


@dataclass(frozen=True, slots=True)
class CommittedTurnRequest:
    """One immutable text turn committed by a realtime frontend."""

    runtime_conversation_id: str
    commit_id: str
    text: str

    def __post_init__(self) -> None:
        """Reject missing correlation identities or empty input."""
        _required(self.runtime_conversation_id, "runtime_conversation_id")
        _required(self.commit_id, "commit_id")
        _required(self.text, "text")


@dataclass(frozen=True, slots=True)
class CommittedTurnResult:
    """One terminal response with display and optional spoken-presentation material."""

    backend_session_id: str
    turn_id: str
    response_id: str
    display_text: str
    speak_text: str | None = None

    def __post_init__(self) -> None:
        """Require the response correlation identities returned by a backend."""
        _required(self.backend_session_id, "backend_session_id")
        _required(self.turn_id, "turn_id")
        _required(self.response_id, "response_id")
        if not isinstance(self.display_text, str):
            raise ValueError("display_text must be a string")
        if self.speak_text is not None:
            _required(self.speak_text, "speak_text")


@dataclass(frozen=True, slots=True)
class CommittedTurnDisplayDelta:
    """Provisional Markdown decoded from a still-uncommitted backend result."""

    backend_session_id: str
    turn_id: str
    response_id: str
    sequence: int
    delta: str
    kind: ResponseOnlyResultEventKind = field(
        default=ResponseOnlyResultEventKind.DISPLAY_DELTA,
        init=False,
    )

    def __post_init__(self) -> None:
        """Reject empty or ambiguous display fragments without trimming Markdown."""
        _required(self.backend_session_id, "backend_session_id")
        _required(self.turn_id, "turn_id")
        _required(self.response_id, "response_id")
        if isinstance(self.sequence, bool) or not isinstance(self.sequence, int) or self.sequence < 0:
            raise ValueError("sequence must be a non-negative integer")
        if not isinstance(self.delta, str) or not self.delta or "\x00" in self.delta:
            raise ValueError("delta must be a non-empty string without NUL")
        try:
            self.delta.encode("utf-8")
        except UnicodeEncodeError as error:
            raise ValueError("delta must be valid UTF-8") from error


@dataclass(frozen=True, slots=True)
class CommittedTurnCompleted:
    """One strictly validated terminal result."""

    result: CommittedTurnResult
    kind: ResponseOnlyResultEventKind = field(
        default=ResponseOnlyResultEventKind.COMPLETED,
        init=False,
    )

    def __post_init__(self) -> None:
        """Require the normalized result model at the terminal boundary."""
        if not isinstance(self.result, CommittedTurnResult):
            raise TypeError("result must be CommittedTurnResult")


CommittedTurnEvent = CommittedTurnDisplayDelta | CommittedTurnCompleted


@dataclass(frozen=True, slots=True)
class CommittedTurnBackend:
    """Negotiated facts for one response-only backend deployment.

    These fields describe only the compatibility surface that VoiceClaw can
    actually use.  They deliberately do not imply durable Work, replay,
    cancellation, steering, or delivery receipts.
    """

    label: str
    target_ref: str
    mode: str
    capabilities: BackendCapabilities
    capability_source: CapabilitySource
    capability_source_id: str
    capability_evidence: CapabilityEvidence = field(init=False)

    def __post_init__(self) -> None:
        """Reject unbounded or ambiguous target descriptions."""
        _required(self.label, "label")
        _required(self.target_ref, "target_ref")
        _required(self.mode, "mode")
        if not isinstance(self.capabilities, BackendCapabilities):
            raise TypeError("capabilities must be BackendCapabilities")
        if self.capabilities.target_label != self.label:
            raise ValueError("label must match the typed capability target_label")
        if self.capabilities.backend_kind != self.mode:
            raise ValueError("mode must match the typed capability backend_kind")
        try:
            source = CapabilitySource(self.capability_source)
        except (TypeError, ValueError) as error:
            raise ValueError("capability_source is invalid") from error
        object.__setattr__(self, "capability_source", source)
        object.__setattr__(
            self,
            "capability_evidence",
            CapabilityEvidence.capture(
                self.capabilities,
                source=source,
                source_id=self.capability_source_id,
            ),
        )


class EphemeralCommittedTurnPort(Protocol):
    """Admit one non-durable committed turn and return its terminal response."""

    async def inspect(self) -> CommittedTurnBackend:
        """Verify server-side reachability and return bounded capabilities."""
        ...

    async def commit_turn(self, request: CommittedTurnRequest) -> CommittedTurnResult:
        """Execute one response-only turn."""
        ...

    def stream_turn(self, request: CommittedTurnRequest) -> AsyncIterator[CommittedTurnEvent]:
        """Stream provisional display fragments followed by one terminal result."""
        ...


__all__ = [
    "CommittedTurnBackend",
    "CommittedTurnCompleted",
    "CommittedTurnDisplayDelta",
    "CommittedTurnError",
    "CommittedTurnEvent",
    "CommittedTurnRequest",
    "CommittedTurnResult",
    "EphemeralCommittedTurnPort",
    "MAX_COMMITTED_TURN_GOAL_BYTES",
]
