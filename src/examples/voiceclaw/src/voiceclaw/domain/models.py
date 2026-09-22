# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""Canonical VoiceClaw domain models.

These types intentionally contain no Pipecat, LiveKit, OpenAI Realtime,
NemoClaw, harness, or provider-specific objects.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import IntEnum, StrEnum
from types import MappingProxyType
from typing import Any

_CAPABILITY_IDENTIFIER = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9._:-]{0,126}[A-Za-z0-9])?$")


def utc_now() -> datetime:
    """Return an aware UTC timestamp."""
    return datetime.now(UTC)


def _required(value: str, name: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{name} must not be empty")
    return normalized


class BackendOperation(StrEnum):
    """Canonical operations a backend may authorize."""

    SUBMIT = "work.submit"
    STATUS = "work.status"
    CANCEL = "work.cancel"
    STEER = "work.steer"
    ANSWER_QUERY = "work.answer_query"
    RESPOND_PERMISSION = "work.respond_permission"
    RESUME = "work.resume"
    REPLAY = "work.replay"


class InputActivity(StrEnum):
    """Current user-input activity."""

    IDLE = "idle"
    LISTENING = "listening"
    TRANSCRIBING = "transcribing"


class ModelActivity(StrEnum):
    """Current frontend-model activity."""

    IDLE = "idle"
    GENERATING = "generating"
    WAITING_FOR_TOOL = "waiting_for_tool"


class OutputActivity(StrEnum):
    """Current local audio-output activity."""

    UNKNOWN = "unknown"
    IDLE = "idle"
    SPEAKING = "speaking"
    INTERRUPTED = "interrupted"


@dataclass(frozen=True, slots=True)
class FrontendActivity:
    """Independent activity axes used to admit queued speech safely."""

    connected: bool = False
    input: InputActivity = InputActivity.IDLE
    model: ModelActivity = ModelActivity.IDLE
    output: OutputActivity = OutputActivity.IDLE

    @property
    def speech_floor_available(self) -> bool:
        """Return whether a queued presentation may claim the speech floor."""
        return (
            self.connected
            and self.input is InputActivity.IDLE
            and self.model is ModelActivity.IDLE
            and self.output is OutputActivity.IDLE
        )


class Durability(StrEnum):
    """Durability level claimed by an attached backend."""

    NONE = "none"
    SESSION = "session"
    BACKEND = "backend"


class EventDelivery(StrEnum):
    """Event delivery semantics claimed by an attached backend."""

    RESPONSE_ONLY = "response_only"
    LIVE = "live"
    ORDERED_REPLAY = "ordered_replay"


class CapabilitySource(StrEnum):
    """Trust origin for one locally captured backend capability claim."""

    BACKEND_NEGOTIATED = "backend_negotiated"
    OPERATOR_CONFIGURED = "operator_configured"
    COMPATIBILITY_PROJECTION = "compatibility_projection"


class BackendEventKind(StrEnum):
    """Canonical event kinds produced by every backend adapter."""

    WORK_STATE_CHANGED = "work.state.v1"
    AGENT_QUERY_CHANGED = "agent_query.state.v1"
    RESULT_AVAILABLE = "work.result.v1"


class WorkState(StrEnum):
    """Normalized read-only projection of backend-owned Work state."""

    UNKNOWN = "unknown"
    SUBMITTED = "submitted"
    ACCEPTED = "accepted"
    QUEUED = "queued"
    RUNNING = "running"
    WAITING_INPUT = "waiting_input"
    WAITING_PERMISSION = "waiting_permission"
    CANCELLING = "cancelling"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"
    EXPIRED = "expired"

    @property
    def terminal(self) -> bool:
        """Return whether this state is immutable and terminal."""
        return self in {self.SUCCEEDED, self.FAILED, self.CANCELLED, self.EXPIRED}


class AgentQueryKind(StrEnum):
    """Backend-owned kind of input requested from the user."""

    INFORMATION = "information"
    PERMISSION = "permission"


class AgentQueryState(StrEnum):
    """Read-only lifecycle of a backend-owned AgentQuery."""

    PENDING = "pending"
    RESPONSE_RECORDED = "response_recorded"
    RESPONSE_FORWARDED = "response_forwarded"
    RESOLVED = "resolved"
    CANCELLED = "cancelled"
    EXPIRED = "expired"

    @property
    def terminal(self) -> bool:
        """Return whether the backend has closed this query identity."""
        return self in {self.RESOLVED, self.CANCELLED, self.EXPIRED}


class ResultState(StrEnum):
    """Lifecycle of a normalized result inside VoiceClaw."""

    RECEIVED = "received"
    NORMALIZED = "normalized"
    AVAILABLE = "available"
    ACKNOWLEDGED = "acknowledged"
    EXPIRED = "expired"


class DisplayState(StrEnum):
    """Independent display-delivery lifecycle."""

    NOT_REQUESTED = "not_requested"
    READY = "ready"
    DELIVERED = "delivered"
    FAILED = "failed"


class SpeechState(StrEnum):
    """Independent speech-delivery lifecycle."""

    NOT_REQUESTED = "not_requested"
    QUEUED = "queued"
    CLAIMED = "claimed"
    SPEAKING = "speaking"
    HEARD = "heard"
    INTERRUPTED = "interrupted"
    DEFERRED = "deferred"
    EXPIRED = "expired"
    FAILED = "failed"


class SpeechRoute(StrEnum):
    """How approved spoken-presentation material reaches the user."""

    FRONTEND_MODEL = "frontend_model"


class PresentationPriority(IntEnum):
    """Stable ordering for speakable work updates; lower values win."""

    CURRENT_TURN = 0
    BLOCKING_INPUT = 10
    SAFETY_OR_FAILURE = 20
    CURRENT_RESULT = 30
    RELEVANT_RESULT = 40
    BACKGROUND_RESULT = 50
    PROGRESS = 60


class CommandState(StrEnum):
    """Local command/outbox lifecycle until backend acceptance is known."""

    STAGED = "staged"
    DISPATCHING = "dispatching"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    INCONCLUSIVE = "inconclusive"
    RECONCILING = "reconciling"


@dataclass(frozen=True, slots=True)
class BackendCapabilities:
    """Backend identity plus the operations authorized for this attachment.

    The adapter reports only proven backend facts. An operator-selected
    interaction profile may narrow those facts into a model-visible tool
    surface, but it cannot add a capability the backend did not prove.
    """

    backend_kind: str
    target_label: str
    revision: str = "unknown"
    operations: frozenset[BackendOperation] = field(default_factory=frozenset)
    durability: Durability = Durability.NONE
    event_delivery: EventDelivery = EventDelivery.RESPONSE_ONLY
    agent_targets: tuple[str, ...] = ()
    sessionful: bool = False
    supports_parallel_work: bool = False
    max_parallel_work: int = 1

    def __post_init__(self) -> None:
        """Validate identity, revision, operations, and parallelism claims."""
        backend_kind = _required(self.backend_kind, "backend_kind")
        target_label = _required(self.target_label, "target_label")
        revision = _required(self.revision, "revision")
        if _CAPABILITY_IDENTIFIER.fullmatch(backend_kind) is None:
            raise ValueError("backend_kind must be a bounded identifier")
        if _CAPABILITY_IDENTIFIER.fullmatch(revision) is None:
            raise ValueError("revision must be a bounded identifier")
        if len(target_label) > 128 or any(ord(character) < 32 or ord(character) == 127 for character in target_label):
            raise ValueError("target_label is invalid")
        object.__setattr__(self, "backend_kind", backend_kind)
        object.__setattr__(self, "target_label", target_label)
        object.__setattr__(self, "revision", revision)
        try:
            operations = frozenset(BackendOperation(operation) for operation in self.operations)
            durability = Durability(self.durability)
            event_delivery = EventDelivery(self.event_delivery)
        except (TypeError, ValueError) as error:
            raise ValueError("backend capabilities contain an unsupported enum value") from error
        object.__setattr__(self, "operations", operations)
        object.__setattr__(self, "durability", durability)
        object.__setattr__(self, "event_delivery", event_delivery)
        targets = tuple(_required(target, "agent_target") for target in self.agent_targets)
        if len(targets) != len(set(targets)) or len(targets) > 64:
            raise ValueError("agent_targets must contain at most 64 unique identifiers")
        if any(_CAPABILITY_IDENTIFIER.fullmatch(target) is None for target in targets):
            raise ValueError("agent_targets must contain bounded identifiers")
        object.__setattr__(self, "agent_targets", targets)
        if not isinstance(self.sessionful, bool) or not isinstance(self.supports_parallel_work, bool):
            raise ValueError("backend capability flags must be booleans")
        if isinstance(self.max_parallel_work, bool) or not isinstance(self.max_parallel_work, int):
            raise ValueError("max_parallel_work must be an integer")
        if self.max_parallel_work < 1:
            raise ValueError("max_parallel_work must be positive")
        if not self.supports_parallel_work and self.max_parallel_work != 1:
            raise ValueError("non-parallel backends must set max_parallel_work to 1")
        if self.supports_parallel_work and self.max_parallel_work < 2:
            raise ValueError("parallel backends must allow at least two Work items")
        if durability is not Durability.NONE and not self.sessionful:
            raise ValueError("durable backends must be sessionful")
        if event_delivery is EventDelivery.ORDERED_REPLAY and (durability is Durability.NONE or not self.sessionful):
            raise ValueError("ordered replay requires a durable sessionful backend")
        if (
            operations
            & {
                BackendOperation.STEER,
                BackendOperation.ANSWER_QUERY,
                BackendOperation.RESPOND_PERMISSION,
            }
            and not self.sessionful
        ):
            raise ValueError("stateful operations require a sessionful backend")
        if operations & {BackendOperation.RESUME, BackendOperation.REPLAY} and (
            durability is Durability.NONE or not self.sessionful
        ):
            raise ValueError("resume and replay require a durable sessionful backend")
        if BackendOperation.REPLAY in operations and event_delivery is not EventDelivery.ORDERED_REPLAY:
            raise ValueError("work.replay requires ordered replay delivery")

    def supports(self, operation: BackendOperation) -> bool:
        """Return whether the attachment authorizes an operation."""
        return operation in self.operations


@dataclass(frozen=True, slots=True)
class CapabilityEvidence:
    """Local evidence for the exact typed capability claim VoiceClaw used.

    The digest is a locally computed integrity fingerprint, not a provider
    signature or an authorization decision by itself.
    """

    source: CapabilitySource
    source_id: str
    revision: str
    digest: str

    @classmethod
    def capture(
        cls,
        capabilities: BackendCapabilities,
        *,
        source: CapabilitySource,
        source_id: str,
    ) -> CapabilityEvidence:
        """Capture provenance and a deterministic SHA-256 claim digest."""
        if not isinstance(capabilities, BackendCapabilities):
            raise TypeError("capabilities must be BackendCapabilities")
        try:
            normalized_source = CapabilitySource(source)
        except (TypeError, ValueError) as error:
            raise ValueError("capability source is invalid") from error
        payload = {
            "schema": "voiceclaw.backend-capabilities.v1",
            "agent_targets": list(capabilities.agent_targets),
            "backend_kind": capabilities.backend_kind,
            "durability": capabilities.durability.value,
            "event_delivery": capabilities.event_delivery.value,
            "max_parallel_work": capabilities.max_parallel_work,
            "operations": sorted(operation.value for operation in capabilities.operations),
            "revision": capabilities.revision,
            "sessionful": capabilities.sessionful,
            "supports_parallel_work": capabilities.supports_parallel_work,
            "target_label": capabilities.target_label,
        }
        canonical = json.dumps(
            payload,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        return cls(
            source=normalized_source,
            source_id=source_id,
            revision=capabilities.revision,
            digest=f"sha256:{hashlib.sha256(canonical).hexdigest()}",
        )

    def __post_init__(self) -> None:
        """Reject malformed evidence values if constructed directly."""
        try:
            source = CapabilitySource(self.source)
        except (TypeError, ValueError) as error:
            raise ValueError("capability source is invalid") from error
        source_id = _required(self.source_id, "capability evidence source_id")
        revision = _required(self.revision, "capability evidence revision")
        if len(source_id) > 128 or "\x00" in source_id:
            raise ValueError("capability evidence source_id is invalid")
        digest = _required(self.digest, "capability evidence digest")
        if not digest.startswith("sha256:") or len(digest) != 71:
            raise ValueError("capability evidence digest must be a SHA-256 digest")
        hexadecimal = digest.removeprefix("sha256:")
        if any(character not in "0123456789abcdef" for character in hexadecimal):
            raise ValueError("capability evidence digest must use lowercase hexadecimal")
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "source_id", source_id)
        object.__setattr__(self, "revision", revision)
        object.__setattr__(self, "digest", digest)


@dataclass(frozen=True, slots=True)
class DisplayPayload:
    """Structured UI material that bypasses the frontend model and TTS."""

    kind: str
    title: str
    body: str = ""
    data: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate display identity and detach structured data."""
        object.__setattr__(self, "kind", _required(self.kind, "display.kind"))
        object.__setattr__(self, "title", _required(self.title, "display.title"))
        object.__setattr__(self, "data", MappingProxyType(dict(self.data)))


@dataclass(frozen=True, slots=True)
class WorkResult:
    """Normalized backend result before independent display/speech delivery."""

    presentation_id: str
    session_id: str
    attachment_id: str
    work_id: str
    result_id: str
    sequence: int
    display: DisplayPayload
    speech_text: str | None = None
    speech_route: SpeechRoute = SpeechRoute.FRONTEND_MODEL
    priority: PresentationPriority = PresentationPriority.CURRENT_RESULT
    received_at: datetime = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        """Validate result correlation and normalize optional speech."""
        for name in ("presentation_id", "session_id", "attachment_id", "work_id", "result_id"):
            object.__setattr__(self, name, _required(getattr(self, name), name))
        if self.sequence < 1:
            raise ValueError("result sequence must be positive")
        object.__setattr__(self, "speech_route", SpeechRoute(self.speech_route))
        if self.speech_text is not None:
            speech_text = self.speech_text.strip()
            object.__setattr__(self, "speech_text", speech_text or None)

    @property
    def immutable_identity(self) -> tuple[str, str, str, int]:
        """Return the backend result identity that survives reattachment.

        ``attachment_id`` identifies the route on which this copy arrived and
        may change on reconnect.  The backend event sequence remains stable
        within the durable session and is part of replay identity.
        """
        return (self.session_id, self.work_id, self.result_id, self.sequence)


@dataclass(frozen=True, slots=True)
class PresentationRecord:
    """Persistable, independently tracked display and speech lifecycle."""

    presentation_id: str
    session_id: str
    attachment_id: str
    work_id: str
    result_id: str
    sequence: int
    display: DisplayPayload
    speech_text: str | None
    speech_route: SpeechRoute
    priority: PresentationPriority
    result_state: ResultState = ResultState.AVAILABLE
    display_state: DisplayState = DisplayState.READY
    speech_state: SpeechState = SpeechState.QUEUED
    created_at: datetime = field(default_factory=utc_now)
    updated_at: datetime = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        """Reject persisted or adapter-supplied speech routes outside this product contract."""
        object.__setattr__(self, "speech_route", SpeechRoute(self.speech_route))

    @classmethod
    def from_result(cls, result: WorkResult) -> PresentationRecord:
        """Create the initial presentation state for a normalized result."""
        return cls(
            presentation_id=result.presentation_id,
            session_id=result.session_id,
            attachment_id=result.attachment_id,
            work_id=result.work_id,
            result_id=result.result_id,
            sequence=result.sequence,
            display=result.display,
            speech_text=result.speech_text,
            speech_route=result.speech_route,
            priority=result.priority,
            speech_state=SpeechState.QUEUED if result.speech_text else SpeechState.NOT_REQUESTED,
            created_at=result.received_at,
            updated_at=result.received_at,
        )

    @property
    def immutable_identity(self) -> tuple[str, str, str, int]:
        """Return the backend result identity that survives reattachment.

        The stored attachment remains routing provenance for the copy that was
        first admitted; it is not canonical result identity.
        """
        return (self.session_id, self.work_id, self.result_id, self.sequence)

    def has_same_immutable_evidence(self, other: PresentationRecord) -> bool:
        """Compare backend identity/content without local routing or delivery state."""
        return (
            self.immutable_identity == other.immutable_identity
            and self.display.kind == other.display.kind
            and self.display.title == other.display.title
            and self.display.body == other.display.body
            and dict(self.display.data) == dict(other.display.data)
            and self.speech_text == other.speech_text
            and self.speech_route == other.speech_route
            and self.priority == other.priority
        )


@dataclass(frozen=True, slots=True)
class SessionBinding:
    """Recoverable mapping between a client conversation and backend attachment."""

    session_id: str
    conversation_id: str
    backend_profile: str
    attachment_id: str | None = None
    backend_session_id: str | None = None
    last_applied_sequence: int = 0
    last_presented_sequence: int = 0
    created_at: datetime = field(default_factory=utc_now)
    updated_at: datetime = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        """Validate the recoverable local-to-backend session mapping."""
        for name in ("session_id", "conversation_id", "backend_profile"):
            object.__setattr__(self, name, _required(getattr(self, name), name))
        if self.attachment_id is not None:
            object.__setattr__(self, "attachment_id", _required(self.attachment_id, "attachment_id"))
        if self.backend_session_id is not None:
            object.__setattr__(
                self,
                "backend_session_id",
                _required(self.backend_session_id, "backend_session_id"),
            )
        if self.last_applied_sequence < 0:
            raise ValueError("last_applied_sequence must be non-negative")
        if self.last_presented_sequence < 0:
            raise ValueError("last_presented_sequence must be non-negative")
        if self.last_presented_sequence > self.last_applied_sequence:
            raise ValueError("last_presented_sequence cannot exceed last_applied_sequence")


@dataclass(frozen=True, slots=True)
class WorkProjection:
    """Read-only local projection keyed by a backend-issued Work identifier."""

    work_id: str
    session_id: str
    attachment_id: str
    state: WorkState
    sequence: int
    summary: str = ""
    raw_state: str | None = None
    agent_target: str | None = None
    updated_at: datetime = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        """Validate Work identity and backend sequence."""
        object.__setattr__(self, "work_id", _required(self.work_id, "work_id"))
        object.__setattr__(self, "session_id", _required(self.session_id, "session_id"))
        object.__setattr__(self, "attachment_id", _required(self.attachment_id, "attachment_id"))
        if self.agent_target is not None:
            target = _required(self.agent_target, "agent_target")
            if _CAPABILITY_IDENTIFIER.fullmatch(target) is None:
                raise ValueError("agent_target must be a bounded identifier")
            object.__setattr__(self, "agent_target", target)
        if self.sequence < 1:
            raise ValueError("backend event sequence must be positive")


@dataclass(frozen=True, slots=True)
class AgentQueryProjection:
    """Read-only session projection of one backend-authored AgentQuery."""

    query_id: str
    work_id: str
    session_id: str
    attachment_id: str
    kind: AgentQueryKind
    state: AgentQueryState
    blocking: bool
    sequence: int
    prompt: str = ""
    raw_state: str | None = None
    updated_at: datetime = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        """Validate query correlation and backend sequence."""
        for name in ("query_id", "work_id", "session_id", "attachment_id"):
            object.__setattr__(self, name, _required(getattr(self, name), name))
        object.__setattr__(self, "kind", AgentQueryKind(self.kind))
        object.__setattr__(self, "state", AgentQueryState(self.state))
        if not isinstance(self.blocking, bool):
            raise TypeError("AgentQuery blocking must be a boolean")
        if not isinstance(self.prompt, str) or "\x00" in self.prompt:
            raise ValueError("AgentQuery prompt must be text without NUL")
        if len(self.prompt.encode("utf-8")) > 64 * 1024:
            raise ValueError("AgentQuery prompt is too large")
        if self.sequence < 1:
            raise ValueError("backend event sequence must be positive")


@dataclass(frozen=True, slots=True)
class SessionControlSnapshot:
    """Atomic persisted facts used for one tool-projection decision.

    The snapshot contains backend-authored projections plus VoiceClaw's local
    query-response claims. It does not decide which tools are visible.
    """

    session_id: str
    applied_sequence: int
    control_revision: int = 0
    works: tuple[WorkProjection, ...] = ()
    pending_queries: tuple[AgentQueryProjection, ...] = ()
    claimed_query_ids: frozenset[str] = field(default_factory=frozenset)
    reserved_work_ids: tuple[str, ...] = ()
    anonymous_capacity_reservations: int = 0
    cancel_claimed_work_ids: frozenset[str] = field(default_factory=frozenset)
    command_recovery_pending: bool = False

    def __post_init__(self) -> None:
        """Validate that every fact belongs to one captured session prefix."""
        session_id = _required(self.session_id, "session_id")
        object.__setattr__(self, "session_id", session_id)
        if isinstance(self.control_revision, bool) or self.control_revision < 0:
            raise ValueError("control_revision must be a non-negative integer")
        if self.applied_sequence < 0:
            raise ValueError("applied_sequence must be non-negative")
        works = tuple(self.works)
        queries = tuple(self.pending_queries)
        if any(work.session_id != session_id for work in works):
            raise ValueError("Work projection belongs to another session")
        if any(query.session_id != session_id for query in queries):
            raise ValueError("AgentQuery projection belongs to another session")
        if any(work.sequence > self.applied_sequence for work in works):
            raise ValueError("Work projection is ahead of the captured applied cursor")
        if any(query.sequence > self.applied_sequence for query in queries):
            raise ValueError("AgentQuery projection is ahead of the captured applied cursor")
        if any(query.state is not AgentQueryState.PENDING for query in queries):
            raise ValueError("session control snapshot may contain only pending AgentQueries")
        work_ids = tuple(work.work_id for work in works)
        query_ids = tuple(query.query_id for query in queries)
        if len(work_ids) != len(set(work_ids)):
            raise ValueError("session control snapshot Work IDs must be unique")
        if len(query_ids) != len(set(query_ids)):
            raise ValueError("session control snapshot AgentQuery IDs must be unique")
        claims = frozenset(_required(query_id, "claimed_query_id") for query_id in self.claimed_query_ids)
        reserved = tuple(_required(work_id, "reserved_work_id") for work_id in self.reserved_work_ids)
        cancel_claims = frozenset(
            _required(work_id, "cancel_claimed_work_id") for work_id in self.cancel_claimed_work_ids
        )
        if len(reserved) != len(set(reserved)):
            raise ValueError("reserved Work IDs must be unique")
        if isinstance(self.anonymous_capacity_reservations, bool) or self.anonymous_capacity_reservations < 0:
            raise ValueError("anonymous_capacity_reservations must be a non-negative integer")
        if not isinstance(self.command_recovery_pending, bool):
            raise TypeError("command_recovery_pending must be a boolean")
        object.__setattr__(self, "works", works)
        object.__setattr__(self, "pending_queries", queries)
        object.__setattr__(self, "claimed_query_ids", claims)
        object.__setattr__(self, "reserved_work_ids", reserved)
        object.__setattr__(self, "cancel_claimed_work_ids", cancel_claims)


@dataclass(frozen=True, slots=True)
class WorkStateChanged:
    """Canonical backend-owned Work-state update."""

    state: WorkState
    summary: str = ""
    raw_state: str | None = None
    agent_target: str | None = None

    def __post_init__(self) -> None:
        """Normalize state and validate optional backend-authored target identity."""
        object.__setattr__(self, "state", WorkState(self.state))
        if self.agent_target is not None:
            target = _required(self.agent_target, "agent_target")
            if _CAPABILITY_IDENTIFIER.fullmatch(target) is None:
                raise ValueError("agent_target must be a bounded identifier")
            object.__setattr__(self, "agent_target", target)


@dataclass(frozen=True, slots=True)
class AgentQueryChanged:
    """Canonical backend-authored AgentQuery state update."""

    query_id: str
    kind: AgentQueryKind
    state: AgentQueryState
    blocking: bool
    prompt: str = ""
    raw_state: str | None = None

    def __post_init__(self) -> None:
        """Validate the backend-authored query payload."""
        object.__setattr__(self, "query_id", _required(self.query_id, "query_id"))
        object.__setattr__(self, "kind", AgentQueryKind(self.kind))
        object.__setattr__(self, "state", AgentQueryState(self.state))
        if not isinstance(self.blocking, bool):
            raise TypeError("AgentQuery blocking must be a boolean")
        if not isinstance(self.prompt, str) or "\x00" in self.prompt:
            raise ValueError("AgentQuery prompt must be text without NUL")
        if len(self.prompt.encode("utf-8")) > 64 * 1024:
            raise ValueError("AgentQuery prompt is too large")


@dataclass(frozen=True, slots=True)
class ResultAvailable:
    """Canonical rich-display and optional speech projection from a backend."""

    result_id: str
    presentation_id: str
    display: DisplayPayload
    speech_text: str | None = None
    speech_route: SpeechRoute = SpeechRoute.FRONTEND_MODEL
    priority: PresentationPriority = PresentationPriority.CURRENT_RESULT

    def __post_init__(self) -> None:
        """Validate result identities and normalize optional speech."""
        object.__setattr__(self, "result_id", _required(self.result_id, "result_id"))
        object.__setattr__(
            self,
            "presentation_id",
            _required(self.presentation_id, "presentation_id"),
        )
        object.__setattr__(self, "speech_route", SpeechRoute(self.speech_route))
        if self.speech_text is not None:
            speech_text = self.speech_text.strip()
            object.__setattr__(self, "speech_text", speech_text or None)


@dataclass(frozen=True, slots=True)
class BackendEvent:
    """Ordered event emitted by any AgentInteractionPort implementation."""

    event_id: str
    session_id: str
    attachment_id: str
    sequence: int
    kind: BackendEventKind
    work_id: str
    payload: WorkStateChanged | AgentQueryChanged | ResultAvailable
    occurred_at: datetime = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        """Validate event correlation and detach its payload."""
        for name in ("event_id", "session_id", "attachment_id", "work_id"):
            object.__setattr__(self, name, _required(getattr(self, name), name))
        try:
            kind = BackendEventKind(self.kind)
        except ValueError as error:
            raise ValueError(f"unsupported canonical backend event kind: {self.kind}") from error
        object.__setattr__(self, "kind", kind)
        if self.sequence < 1:
            raise ValueError("backend event sequence must be positive")
        expected_payload = {
            BackendEventKind.WORK_STATE_CHANGED: WorkStateChanged,
            BackendEventKind.AGENT_QUERY_CHANGED: AgentQueryChanged,
            BackendEventKind.RESULT_AVAILABLE: ResultAvailable,
        }[kind]
        if not isinstance(self.payload, expected_payload):
            raise TypeError(f"{kind.value} requires {expected_payload.__name__}")


@dataclass(frozen=True, slots=True)
class CommandRecord:
    """Local command outbox record used to avoid blind redispatch.

    This record is durable VoiceClaw evidence, not an authoritative Work
    ledger.  The backend supplies ``work_id`` only after command admission.
    """

    command_id: str
    session_id: str
    attachment_id: str
    commit_id: str
    operation: BackendOperation
    capability_revision: str
    payload: Mapping[str, Any]
    backend_session_id: str | None = None
    state: CommandState = CommandState.STAGED
    work_id: str | None = None
    reason_code: str | None = None
    created_at: datetime = field(default_factory=utc_now)
    updated_at: datetime = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        """Validate command correlation and detach its payload."""
        for name in (
            "command_id",
            "session_id",
            "attachment_id",
            "commit_id",
            "capability_revision",
        ):
            object.__setattr__(self, name, _required(getattr(self, name), name))
        if self.work_id is not None:
            object.__setattr__(self, "work_id", _required(self.work_id, "work_id"))
        if self.backend_session_id is not None:
            object.__setattr__(
                self,
                "backend_session_id",
                _required(self.backend_session_id, "backend_session_id"),
            )
        if self.reason_code is not None:
            if not isinstance(self.reason_code, str):
                raise TypeError("reason_code must be a string")
            reason_code = _required(self.reason_code, "reason_code")
            if "\x00" in reason_code or len(reason_code.encode("utf-8")) > 128:
                raise ValueError("reason_code must be at most 128 UTF-8 bytes and contain no NUL")
            object.__setattr__(self, "reason_code", reason_code)
        object.__setattr__(self, "payload", MappingProxyType(dict(self.payload)))
