# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""Facade-facing Interaction Manager contract independent of Realtime/Pipecat."""

from __future__ import annotations

from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType
from typing import Any, Protocol

from voiceclaw.domain.capabilities import SemanticTool
from voiceclaw.domain.models import CapabilitySource, Durability, EventDelivery, FrontendActivity
from voiceclaw.domain.response_only import (
    ResponseOnlyRequestState,
    ResponseOnlyResultEventKind,
    ResponseOnlyUpdateKind,
)


class TurnDirectiveKind(StrEnum):
    """Provider-neutral disposition for one finalized user turn."""

    AUTO = "auto"
    DIRECT = "direct"
    TOOL = "tool"
    REJECT = "reject"


class FrontendResponsePurpose(StrEnum):
    """Server-owned reason for asking the realtime frontend to speak."""

    DELEGATION_ACK = "delegation_ack"
    RESULT_DELIVERY = "result_delivery"
    FAILURE_DELIVERY = "failure_delivery"


class FrontendConversationDeliveryState(StrEnum):
    """VoiceClaw-local evidence for one public conversation turn."""

    COMMITTED = "committed"
    DELIVERED = "delivered"
    HEARD = "heard"
    INTERRUPTED = "interrupted"
    FAILED = "failed"


class FrontendPlaybackReceiptState(StrEnum):
    """Client-reported terminal playback disposition."""

    HEARD = "heard"
    INTERRUPTED = "interrupted"
    FAILED = "failed"


def _runtime_identifier(value: str | None, name: str, *, required: bool = True) -> str | None:
    """Validate one bounded, provider-neutral correlation identity."""
    if value is None and not required:
        return None
    if (
        not isinstance(value, str)
        or not value
        or len(value) > 512
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
    ):
        raise ValueError(f"{name} is invalid")
    return value


@dataclass(frozen=True, slots=True)
class FrontendConversationTurn:
    """One public user or assistant turn eligible for bounded context."""

    turn_id: str
    role: str
    text: str
    delivery_state: FrontendConversationDeliveryState
    presentation_id: str | None = None
    local_request_id: str | None = None
    text_truncated: bool = False

    def __post_init__(self) -> None:
        """Validate public text and local-only delivery correlation."""
        object.__setattr__(self, "turn_id", _runtime_identifier(self.turn_id, "turn_id"))
        if self.role not in {"user", "assistant"}:
            raise ValueError("conversation turn role must be user or assistant")
        if not isinstance(self.text, str) or not self.text.strip() or "\x00" in self.text:
            raise ValueError("conversation turn text must be non-empty text without NUL")
        try:
            delivery_state = FrontendConversationDeliveryState(self.delivery_state)
        except (TypeError, ValueError) as error:
            raise ValueError("conversation turn delivery_state is invalid") from error
        if self.role == "user" and delivery_state is not FrontendConversationDeliveryState.COMMITTED:
            raise ValueError("user conversation turns must be committed")
        if self.role == "assistant" and delivery_state is FrontendConversationDeliveryState.COMMITTED:
            raise ValueError("assistant conversation turns cannot be committed user input")
        object.__setattr__(self, "delivery_state", delivery_state)
        object.__setattr__(
            self,
            "presentation_id",
            _runtime_identifier(self.presentation_id, "presentation_id", required=False),
        )
        object.__setattr__(
            self,
            "local_request_id",
            _runtime_identifier(self.local_request_id, "local_request_id", required=False),
        )
        if not isinstance(self.text_truncated, bool):
            raise TypeError("conversation turn text_truncated must be a boolean")


@dataclass(frozen=True, slots=True)
class FrontendPlaybackReceipt:
    """Bounded client evidence for one assistant turn's playback boundary."""

    turn_id: str
    state: FrontendPlaybackReceiptState
    presentation_id: str
    local_request_id: str | None = None
    heard_through_ms: int | None = None
    audio_end_ms: int | None = None

    def __post_init__(self) -> None:
        """Validate identity and monotonic millisecond boundaries."""
        object.__setattr__(self, "turn_id", _runtime_identifier(self.turn_id, "turn_id"))
        try:
            state = FrontendPlaybackReceiptState(self.state)
        except (TypeError, ValueError) as error:
            raise ValueError("playback receipt state is invalid") from error
        object.__setattr__(self, "state", state)
        object.__setattr__(
            self,
            "presentation_id",
            _runtime_identifier(self.presentation_id, "presentation_id"),
        )
        object.__setattr__(
            self,
            "local_request_id",
            _runtime_identifier(self.local_request_id, "local_request_id", required=False),
        )
        for name in ("heard_through_ms", "audio_end_ms"):
            value = getattr(self, name)
            if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 0):
                raise ValueError(f"{name} must be a non-negative integer or null")
        if (
            self.heard_through_ms is not None
            and self.audio_end_ms is not None
            and self.heard_through_ms > self.audio_end_ms
        ):
            raise ValueError("heard_through_ms cannot exceed audio_end_ms")


@dataclass(frozen=True, slots=True)
class TurnDirective:
    """Interaction Manager decision consumed by a protocol facade."""

    kind: TurnDirectiveKind
    logical_tool: str | None
    reason_code: str

    @classmethod
    def auto(cls, *, reason_code: str) -> TurnDirective:
        """Allow the configured frontend to select an advertised operation."""
        return cls(kind=TurnDirectiveKind.AUTO, logical_tool=None, reason_code=reason_code)

    @classmethod
    def direct(cls, *, reason_code: str) -> TurnDirective:
        """Keep this turn in the low-latency frontend."""
        return cls(kind=TurnDirectiveKind.DIRECT, logical_tool=None, reason_code=reason_code)

    @classmethod
    def tool(cls, logical_tool: str, *, reason_code: str) -> TurnDirective:
        """Require one capability-gated Interaction Manager tool."""
        return cls(kind=TurnDirectiveKind.TOOL, logical_tool=logical_tool, reason_code=reason_code)

    @classmethod
    def reject(cls, *, reason_code: str) -> TurnDirective:
        """Reject a turn that cannot safely enter either execution path."""
        return cls(kind=TurnDirectiveKind.REJECT, logical_tool=None, reason_code=reason_code)


@dataclass(frozen=True, slots=True)
class SessionSnapshot:
    """Application-owned state returned when a voice session opens."""

    session_id: str
    conversation_id: str
    backend_profile: str
    recoverable_inflight: bool
    projection: str
    backend_label: str = "Backend unavailable"
    backend_mode: str = "disabled"
    gateway_reachable: bool = False
    target_ref: str = "not-attached"
    capabilities: tuple[str, ...] = ()
    frontend_tools: tuple[SemanticTool, ...] = ()
    durability: Durability = Durability.NONE
    event_delivery: EventDelivery = EventDelivery.RESPONSE_ONLY
    max_parallel_work: int = 1
    capability_source: CapabilitySource | None = None
    capability_source_id: str | None = None
    capability_revision: str | None = None
    capability_hash: str | None = None
    model_contract_schema: str | None = None
    model_contract_profile: str | None = None
    model_contract_hash: str | None = None
    interaction_profile_schema: str | None = None
    interaction_profile_name: str | None = None
    interaction_profile_hash: str | None = None


@dataclass(frozen=True, slots=True)
class FrontendResponse:
    """One typed speech handoff to the realtime frontend.

    ``payload_text`` is quoted data whose meaning is fixed by ``purpose``:
    a bounded goal for acknowledgement, backend-authorized presentation material
    for a result, or a safe public reason for failure. The realtime frontend uses
    it as the factual basis for a context-aware spoken response; it is not a
    literal utterance or a free-form set of prompt variables.
    ``local_request_id`` is transport/UI correlation only and is never a
    backend-issued Work ID.
    """

    purpose: FrontendResponsePurpose
    payload_text: str
    local_request_id: str | None = None

    def __post_init__(self) -> None:
        """Validate the typed payload crossing the application port."""
        try:
            purpose = FrontendResponsePurpose(self.purpose)
        except ValueError as error:
            raise ValueError("frontend response purpose is invalid") from error
        if not isinstance(self.payload_text, str) or not self.payload_text.strip() or "\x00" in self.payload_text:
            raise ValueError("frontend response payload_text must be non-empty text without NUL")
        local_request_id = self.local_request_id
        if local_request_id is not None and (
            not isinstance(local_request_id, str)
            or not local_request_id
            or len(local_request_id) > 512
            or any(ord(character) < 32 or ord(character) == 127 for character in local_request_id)
        ):
            raise ValueError("local_request_id is invalid")
        object.__setattr__(self, "purpose", purpose)
        object.__setattr__(self, "local_request_id", local_request_id)


@dataclass(frozen=True, slots=True)
class InteractionUpdate:
    """One protocol-neutral projection with optional tool and speech effects.

    ``request_summary`` is bounded model-authored display context for the
    admitted request, separate from correlation identities. ``frontend_tools``
    is an authoritative replacement snapshot for subsequent frontend response
    boundaries. ``None`` leaves the current snapshot unchanged; an empty tuple
    explicitly removes every protected Work tool.
    """

    kind: ResponseOnlyUpdateKind
    phase: ResponseOnlyRequestState | ResponseOnlyResultEventKind
    title: str
    text: str
    correlation: Mapping[str, str] = field(default_factory=dict)
    tool_output: Mapping[str, str] | None = None
    frontend_response: FrontendResponse | None = None
    frontend_tools: tuple[SemanticTool, ...] | None = None
    request_summary: str | None = None

    def __post_init__(self) -> None:
        """Detach caller-owned mappings before handing the update to an adapter."""
        try:
            kind = ResponseOnlyUpdateKind(self.kind)
            phase = (
                ResponseOnlyResultEventKind(self.phase)
                if kind is ResponseOnlyUpdateKind.RESULT_DISPLAY
                else ResponseOnlyRequestState(self.phase)
            )
            object.__setattr__(self, "kind", kind)
            object.__setattr__(self, "phase", phase)
        except ValueError as error:
            raise ValueError("interaction update kind or phase is invalid") from error
        object.__setattr__(self, "correlation", MappingProxyType(dict(self.correlation)))
        if self.request_summary is not None and (
            not isinstance(self.request_summary, str)
            or not self.request_summary.strip()
            or len(self.request_summary) > 512
            or "\x00" in self.request_summary
        ):
            raise ValueError("request_summary is invalid")
        if self.tool_output is not None:
            output = dict(self.tool_output)
            if not all(isinstance(key, str) and isinstance(value, str) for key, value in output.items()):
                raise ValueError("tool output must contain text fields")
            object.__setattr__(self, "tool_output", MappingProxyType(output))
        if self.frontend_response is not None and not isinstance(self.frontend_response, FrontendResponse):
            raise TypeError("frontend_response must be FrontendResponse")
        if self.frontend_tools is not None:
            tools = tuple(self.frontend_tools)
            if not all(isinstance(tool, SemanticTool) for tool in tools):
                raise TypeError("frontend_tools must contain SemanticTool values")
            names = tuple(tool.name for tool in tools)
            if len(names) != len(set(names)):
                raise ValueError("frontend_tools must contain unique logical names")
            object.__setattr__(self, "frontend_tools", tools)

    @property
    def terminal(self) -> bool:
        """Return whether this update carries the tool's terminal outcome."""
        return isinstance(self.phase, ResponseOnlyRequestState) and self.phase.terminal


class RealtimeSessionRuntimePort(Protocol):
    """Interaction Manager surface consumed by any realtime protocol adapter."""

    async def open_session(self, session_id: str, conversation_id: str) -> SessionSnapshot:
        """Record a local voice-session mapping and return its initial projection."""
        ...

    def projection(self, session_id: str) -> str:
        """Return a bounded immutable projection for the next response boundary."""
        ...

    def record_conversation_turn(self, session_id: str, turn: FrontendConversationTurn) -> None:
        """Record one public turn without treating it as backend Work evidence."""
        ...

    def record_playback_receipt(self, session_id: str, receipt: FrontendPlaybackReceipt) -> None:
        """Apply one client playback receipt to an already delivered assistant turn."""
        ...

    def route_finalized_turn(self, session_id: str, text: str) -> TurnDirective:
        """Choose direct conversation or one advertised logical operation."""
        ...

    async def update_activity(self, session_id: str, activity: FrontendActivity) -> None:
        """Update independent input/model/output activity axes."""
        ...

    def execute_tool(
        self,
        *,
        session_id: str,
        commit_id: str,
        call_id: str,
        tool_name: str,
        arguments: Mapping[str, Any],
        finalized_user_text: str | None,
    ) -> AsyncIterator[InteractionUpdate]:
        """Execute one capability-gated Work tool with server-owned turn context.

        The iterator owns cleanup for every locally admitted request: normal
        exhaustion, failure, cancellation, or ``aclose()`` must leave no
        request in a nonterminal local phase.
        """
        ...

    async def close_session(self, session_id: str, reason: str) -> None:
        """Mark the frontend disconnected without cancelling backend Work."""
        ...


__all__ = [
    "FrontendConversationDeliveryState",
    "FrontendConversationTurn",
    "FrontendPlaybackReceipt",
    "FrontendPlaybackReceiptState",
    "FrontendResponse",
    "FrontendResponsePurpose",
    "InteractionUpdate",
    "RealtimeSessionRuntimePort",
    "SessionSnapshot",
    "TurnDirective",
    "TurnDirectiveKind",
]
