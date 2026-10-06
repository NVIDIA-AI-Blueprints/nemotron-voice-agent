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
    ResponseOnlyContextContinuity,
    ResponseOnlyRequestState,
    ResponseOnlyResultEventKind,
    ResponseOnlyTargetAvailability,
    ResponseOnlyUpdateKind,
)

MIN_FRONTEND_CONTEXT_CHARACTERS = 256
MAX_FRONTEND_INSTRUCTION_CHARACTERS = 64_000
FRONTEND_INSTRUCTION_RESERVE_CHARACTERS = 8_192
MAX_FRONTEND_CONTEXT_CHARACTERS = MAX_FRONTEND_INSTRUCTION_CHARACTERS - FRONTEND_INSTRUCTION_RESERVE_CHARACTERS


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


class FrontendContextPurpose(StrEnum):
    """Application-owned view requested for one frontend model boundary."""

    CONVERSATION = "conversation"
    DELEGATION_ACK = "delegation_ack"
    RESULT_DELIVERY = "result_delivery"
    FAILURE_DELIVERY = "failure_delivery"


class FrontendConversationDeliveryState(StrEnum):
    """VoiceClaw-local evidence for one public conversation turn."""

    COMMITTED = "committed"
    DELIVERED = "delivered"
    HEARD = "heard"
    INTERRUPTED = "interrupted"
    CANCELLED = "cancelled"
    FAILED = "failed"
    SKIPPED = "skipped"


class FrontendPlaybackReceiptState(StrEnum):
    """Client-reported terminal playback disposition."""

    HEARD = "heard"
    INTERRUPTED = "interrupted"
    FAILED = "failed"


class FrontendSpeechDeliveryOutcomeState(StrEnum):
    """Server-observed terminal outcome when no playback receipt can exist."""

    SKIPPED = "skipped"
    FAILED = "failed"
    CANCELLED = "cancelled"


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
class FrontendSpeechDeliveryOutcome:
    """Typed terminal speech outcome independent of browser playback evidence."""

    local_request_id: str
    presentation_id: str
    state: FrontendSpeechDeliveryOutcomeState
    reason_code: str

    def __post_init__(self) -> None:
        """Validate local correlation and a bounded machine-readable reason."""
        object.__setattr__(
            self,
            "local_request_id",
            _runtime_identifier(self.local_request_id, "local_request_id"),
        )
        object.__setattr__(
            self,
            "presentation_id",
            _runtime_identifier(self.presentation_id, "presentation_id"),
        )
        try:
            state = FrontendSpeechDeliveryOutcomeState(self.state)
        except (TypeError, ValueError) as error:
            raise ValueError("speech delivery outcome state is invalid") from error
        object.__setattr__(self, "state", state)
        if (
            not isinstance(self.reason_code, str)
            or not self.reason_code
            or len(self.reason_code) > 128
            or not all(character.islower() or character.isdigit() or character == "_" for character in self.reason_code)
        ):
            raise ValueError("speech delivery outcome reason_code is invalid")


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
    target_binding_verified: bool = False
    target_ref: str = "not-attached"
    target_availability: ResponseOnlyTargetAvailability | None = None
    capabilities: tuple[str, ...] = ()
    frontend_tools: tuple[SemanticTool, ...] = ()
    durability: Durability = Durability.NONE
    event_delivery: EventDelivery = EventDelivery.RESPONSE_ONLY
    max_parallel_work: int = 1
    context_continuity: ResponseOnlyContextContinuity = ResponseOnlyContextContinuity.UNQUALIFIED
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
class BackendTargetSnapshot:
    """Atomic replacement state for the configured backend target."""

    backend_label: str
    backend_mode: str
    target_binding_verified: bool
    target_ref: str
    target_availability: ResponseOnlyTargetAvailability | None = None
    capabilities: tuple[str, ...] = ()
    frontend_tools: tuple[SemanticTool, ...] = ()
    durability: Durability = Durability.NONE
    event_delivery: EventDelivery = EventDelivery.RESPONSE_ONLY
    max_parallel_work: int = 1
    context_continuity: ResponseOnlyContextContinuity = ResponseOnlyContextContinuity.NONE
    capability_source: CapabilitySource | None = None
    capability_source_id: str | None = None
    capability_revision: str | None = None
    capability_hash: str | None = None

    def __post_init__(self) -> None:
        """Normalize immutable collections and reject contradictory state."""
        object.__setattr__(self, "capabilities", tuple(self.capabilities))
        tools = tuple(self.frontend_tools)
        if not all(isinstance(tool, SemanticTool) for tool in tools):
            raise TypeError("frontend_tools must contain SemanticTool values")
        object.__setattr__(self, "frontend_tools", tools)
        if self.target_availability is not None:
            try:
                availability = ResponseOnlyTargetAvailability(self.target_availability)
            except (TypeError, ValueError) as error:
                raise ValueError("target_availability is invalid") from error
            object.__setattr__(self, "target_availability", availability)
        if not self.target_binding_verified and self.target_availability is not None:
            raise ValueError("an unavailable backend target cannot advertise target availability")
        if not self.target_binding_verified and (self.capabilities or tools):
            raise ValueError("an unavailable backend target cannot advertise capabilities or tools")


@dataclass(frozen=True, slots=True)
class FrontendResponse:
    """One typed speech handoff to the realtime frontend.

    ``payload_text`` is quoted data whose meaning is fixed by ``purpose``:
    backend-authorized presentation material for a result or a safe public reason
    for failure. A delegation acknowledgement carries no duplicate payload; its
    frozen objective and admission evidence come from ``FrontendContextRequest``.
    ``local_request_id`` is transport/UI correlation only and is never a
    backend-issued Work ID.
    """

    purpose: FrontendResponsePurpose
    payload_text: str | None
    local_request_id: str | None = None

    def __post_init__(self) -> None:
        """Validate the typed payload crossing the application port."""
        try:
            purpose = FrontendResponsePurpose(self.purpose)
        except (TypeError, ValueError) as error:
            raise ValueError("frontend response purpose is invalid") from error
        if purpose is FrontendResponsePurpose.DELEGATION_ACK:
            if self.payload_text is not None:
                raise ValueError("delegation acknowledgement must not carry payload_text")
        elif not isinstance(self.payload_text, str) or not self.payload_text.strip() or "\x00" in self.payload_text:
            raise ValueError("result and failure responses require non-empty payload_text without NUL")
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
class FrontendContextRequest:
    """Select one bounded, purpose-specific frontend model projection."""

    purpose: FrontendContextPurpose
    local_request_id: str | None = None

    def __post_init__(self) -> None:
        """Validate correlation without assigning backend Work identity."""
        try:
            purpose = FrontendContextPurpose(self.purpose)
        except ValueError as error:
            raise ValueError("frontend context purpose is invalid") from error
        local_request_id = self.local_request_id
        if local_request_id is not None and (
            not isinstance(local_request_id, str)
            or not local_request_id
            or len(local_request_id) > 512
            or any(ord(character) < 32 or ord(character) == 127 for character in local_request_id)
        ):
            raise ValueError("local_request_id is invalid")
        if purpose is FrontendContextPurpose.CONVERSATION and local_request_id is not None:
            raise ValueError("conversation context must not target a local request")
        if purpose is FrontendContextPurpose.DELEGATION_ACK and local_request_id is None:
            raise ValueError(f"{purpose.value} context requires local_request_id")
        object.__setattr__(self, "purpose", purpose)
        object.__setattr__(self, "local_request_id", local_request_id)


@dataclass(frozen=True, slots=True)
class InteractionUpdate:
    """One protocol-neutral projection with optional tool and speech effects.

    ``request_summary`` is bounded model-authored display context for the
    admitted request, separate from correlation identities. ``frontend_tools``
    is an authoritative replacement snapshot for subsequent frontend response
    boundaries. ``None`` leaves the current snapshot unchanged; an empty tuple
    explicitly removes every protected Work tool. Production target refreshes
    also carry ``backend_target`` so binding identity, capabilities, and tools
    change atomically.
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
    backend_target: BackendTargetSnapshot | None = None

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
        if self.backend_target is not None:
            if not isinstance(self.backend_target, BackendTargetSnapshot):
                raise TypeError("backend_target must be BackendTargetSnapshot")
            if self.frontend_tools is not None and self.frontend_tools != self.backend_target.frontend_tools:
                raise ValueError("frontend_tools must match backend_target.frontend_tools")

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
        """Return a bounded operational projection for UI and diagnostics."""
        ...

    def frontend_context(
        self,
        session_id: str,
        request: FrontendContextRequest,
        *,
        maximum_characters: int,
    ) -> str:
        """Return a bounded purpose-specific projection for the frontend model."""
        ...

    def record_conversation_turn(self, session_id: str, turn: FrontendConversationTurn) -> None:
        """Record one public turn without treating it as backend Work evidence."""
        ...

    def record_playback_receipt(self, session_id: str, receipt: FrontendPlaybackReceipt) -> None:
        """Apply one client playback receipt to an already delivered assistant turn."""
        ...

    def record_speech_delivery_outcome(
        self,
        session_id: str,
        outcome: FrontendSpeechDeliveryOutcome,
    ) -> None:
        """Record a terminal no-playback outcome without inventing a receipt."""
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
    "BackendTargetSnapshot",
    "FRONTEND_INSTRUCTION_RESERVE_CHARACTERS",
    "FrontendContextPurpose",
    "FrontendContextRequest",
    "FrontendConversationDeliveryState",
    "FrontendConversationTurn",
    "FrontendPlaybackReceipt",
    "FrontendPlaybackReceiptState",
    "FrontendSpeechDeliveryOutcome",
    "FrontendSpeechDeliveryOutcomeState",
    "FrontendResponse",
    "FrontendResponsePurpose",
    "InteractionUpdate",
    "MAX_FRONTEND_CONTEXT_CHARACTERS",
    "MAX_FRONTEND_INSTRUCTION_CHARACTERS",
    "MIN_FRONTEND_CONTEXT_CHARACTERS",
    "RealtimeSessionRuntimePort",
    "SessionSnapshot",
    "TurnDirective",
    "TurnDirectiveKind",
]
