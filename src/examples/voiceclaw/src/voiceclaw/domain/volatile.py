# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""Process-local request and result lifecycle for response-only integrations."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from enum import StrEnum
from threading import RLock
from uuid import uuid4

from voiceclaw.domain.response_only import ResponseOnlyResultEnvelope

_MAX_IDENTIFIER_CHARACTERS = 512
_MAX_SUMMARY_CHARACTERS = 512
_MAX_OPERATION_CHARACTERS = 128
_MAX_REASON_CHARACTERS = 128


class VolatileLedgerError(RuntimeError):
    """Base failure raised by the process-local ledger."""


class VolatileCapacityError(VolatileLedgerError):
    """The bounded ledger has no quiescent record it can evict safely."""


class VolatileRequestNotFoundError(VolatileLedgerError):
    """The named request does not belong to the named live session."""


class VolatileTransitionError(VolatileLedgerError):
    """A requested lifecycle transition is not valid from current evidence."""


class VolatileStaleClaimError(VolatileTransitionError):
    """A speech mutation references an expired or superseded claim."""


class VolatileWorkState(StrEnum):
    """VoiceClaw-owned lifecycle for one non-durable request."""

    LOCALLY_QUEUED = "locally_queued"
    DISPATCHING = "dispatching"
    WAITING_FOR_RESPONSE = "waiting_for_response"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"
    OUTCOME_UNKNOWN = "outcome_unknown"

    @property
    def active(self) -> bool:
        """Return whether the request may still reach a new backend outcome."""
        return self in {
            self.LOCALLY_QUEUED,
            self.DISPATCHING,
            self.WAITING_FOR_RESPONSE,
        }

    @property
    def terminal(self) -> bool:
        """Return whether the request outcome is locally final."""
        return not self.active


class VolatileCancelDisposition(StrEnum):
    """Result of an application-owned pre-effects cancellation attempt."""

    CANCELLED = "cancelled"
    TOO_LATE = "too_late"
    ALREADY_TERMINAL = "already_terminal"


class VolatileDisplayState(StrEnum):
    """Independent UI delivery state for a normalized result."""

    READY = "ready"
    DELIVERED = "delivered"
    FAILED = "failed"

    @property
    def terminal(self) -> bool:
        """Return whether display delivery no longer needs work."""
        return self in {self.DELIVERED, self.FAILED}


class VolatileSpeechState(StrEnum):
    """Process-local ownership and playback state for result speech."""

    NOT_REQUESTED = "not_requested"
    READY = "ready"
    CLAIMED = "claimed"
    SPEAKING = "speaking"
    AWAITING_PLAYBACK_RECEIPT = "awaiting_playback_receipt"
    HEARD = "heard"
    INTERRUPTED = "interrupted"
    SKIPPED = "skipped"
    CANCELLED = "cancelled"
    FAILED = "failed"

    @property
    def terminal(self) -> bool:
        """Return whether no later speech action is permitted."""
        return self in {
            self.NOT_REQUESTED,
            self.HEARD,
            self.SKIPPED,
            self.CANCELLED,
            self.FAILED,
        }


class VolatileSpeechClaimOwner(StrEnum):
    """Application path with exclusive permission to present result speech."""

    AUTOMATIC = "automatic"
    STATUS = "status"
    RECAP = "recap"


class VolatilePlaybackState(StrEnum):
    """Browser-observed disposition for generated result audio."""

    HEARD = "heard"
    INTERRUPTED = "interrupted"
    FAILED = "failed"


class VolatileDeliveryOutcome(StrEnum):
    """Server-observed terminal outcome when playback evidence is unavailable."""

    SKIPPED = "skipped"
    CANCELLED = "cancelled"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class VolatileResultSnapshot:
    """Immutable result and delivery evidence returned to application code."""

    result_id: str
    delivery_id: str
    presentation_id: str
    envelope: ResponseOnlyResultEnvelope
    display_state: VolatileDisplayState
    speech_state: VolatileSpeechState
    claim_revision: int
    claim_owner: VolatileSpeechClaimOwner | None
    turn_id: str | None
    playback_state: VolatilePlaybackState | None
    delivery_outcome: VolatileDeliveryOutcome | None
    heard_through_ms: int | None
    audio_end_ms: int | None


@dataclass(frozen=True, slots=True)
class VolatileWorkSnapshot:
    """Immutable view of one request scoped to one live VoiceClaw session."""

    session_id: str
    local_request_id: str
    local_work_id: str
    attempt_id: str
    operation: str
    summary: str
    state: VolatileWorkState
    effects_started: bool
    failure_code: str | None
    result: VolatileResultSnapshot | None
    revision: int


@dataclass(frozen=True, slots=True)
class VolatileSessionSnapshot:
    """Bounded process-local view that never crosses session ownership."""

    session_id: str
    revision: int
    requests: tuple[VolatileWorkSnapshot, ...]


@dataclass(frozen=True, slots=True)
class VolatileSpeechClaim:
    """Exact, single-owner permission to enqueue one result speech attempt."""

    session_id: str
    local_request_id: str
    result_id: str
    presentation_id: str
    revision: int
    owner: VolatileSpeechClaimOwner
    speech_text: str
    source_state: VolatileSpeechState


@dataclass(slots=True)
class _ResultRecord:
    result_id: str
    delivery_id: str
    presentation_id: str
    envelope: ResponseOnlyResultEnvelope
    display_state: VolatileDisplayState
    speech_state: VolatileSpeechState
    claim_revision: int = 0
    claim_owner: VolatileSpeechClaimOwner | None = None
    claim_source_state: VolatileSpeechState | None = None
    turn_id: str | None = None
    playback_state: VolatilePlaybackState | None = None
    delivery_outcome: VolatileDeliveryOutcome | None = None
    heard_through_ms: int | None = None
    audio_end_ms: int | None = None


@dataclass(slots=True)
class _WorkRecord:
    local_request_id: str
    local_work_id: str
    attempt_id: str
    operation: str
    summary: str
    state: VolatileWorkState
    effects_started: bool
    failure_code: str | None
    result: _ResultRecord | None
    revision: int


@dataclass(slots=True)
class _SessionRecord:
    session_id: str
    revision: int
    requests: OrderedDict[str, _WorkRecord]


def _identifier(value: str, name: str, *, maximum: int = _MAX_IDENTIFIER_CHARACTERS) -> str:
    """Return a bounded non-control identifier."""
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    normalized = value.strip()
    if (
        not normalized
        or len(normalized) > maximum
        or any(ord(character) < 32 or ord(character) == 127 for character in normalized)
    ):
        raise ValueError(f"{name} is invalid")
    return normalized


def _summary(value: str) -> str:
    """Return bounded user-visible request context without changing its words."""
    if not isinstance(value, str):
        raise TypeError("summary must be a string")
    normalized = " ".join(value.split()).replace("\x00", "")
    if not normalized or len(normalized) > _MAX_SUMMARY_CHARACTERS:
        raise ValueError("summary is invalid")
    return normalized


def _reason(value: str, name: str) -> str:
    """Validate one stable machine-readable reason."""
    normalized = _identifier(value, name, maximum=_MAX_REASON_CHARACTERS)
    if not all(character.islower() or character.isdigit() or character == "_" for character in normalized):
        raise ValueError(f"{name} must use lowercase letters, digits, and underscores")
    return normalized


class VolatileWorkLedger:
    """Thread-safe, bounded ledger for one process lifetime.

    The ledger owns no backend Work IDs and makes no recovery promise. Every
    lookup requires a live VoiceClaw session ID, so equal local request IDs in
    different sessions cannot alias each other.
    """

    def __init__(self, *, max_sessions: int = 128, max_requests_per_session: int = 32) -> None:
        """Configure hard bounds; active or undelivered records are never evicted."""
        for name, value in {
            "max_sessions": max_sessions,
            "max_requests_per_session": max_requests_per_session,
        }.items():
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        self._max_sessions = max_sessions
        self._max_requests_per_session = max_requests_per_session
        self._sessions: OrderedDict[str, _SessionRecord] = OrderedDict()
        self._lock = RLock()

    def admit(
        self,
        session_id: str,
        local_request_id: str,
        *,
        summary: str,
        operation: str = "work.delegate",
        local_work_id: str | None = None,
        attempt_id: str | None = None,
    ) -> VolatileWorkSnapshot:
        """Atomically admit one locally queued request or replay an identical admission."""
        normalized_session_id = _identifier(session_id, "session_id")
        normalized_request_id = _identifier(local_request_id, "local_request_id")
        normalized_summary = _summary(summary)
        normalized_operation = _identifier(operation, "operation", maximum=_MAX_OPERATION_CHARACTERS)
        normalized_work_id = _identifier(
            local_work_id or f"work_vc_{uuid4().hex}",
            "local_work_id",
        )
        normalized_attempt_id = _identifier(
            attempt_id or f"attempt_vc_{uuid4().hex}",
            "attempt_id",
        )
        with self._lock:
            session = self._sessions.get(normalized_session_id)
            if session is None:
                self._ensure_session_capacity()
                session = _SessionRecord(
                    session_id=normalized_session_id,
                    revision=0,
                    requests=OrderedDict(),
                )
                self._sessions[normalized_session_id] = session
            existing = session.requests.get(normalized_request_id)
            if existing is not None:
                if (
                    existing.summary == normalized_summary
                    and existing.operation == normalized_operation
                    and (local_work_id is None or existing.local_work_id == normalized_work_id)
                    and (attempt_id is None or existing.attempt_id == normalized_attempt_id)
                ):
                    return self._work_snapshot(session, existing)
                raise VolatileTransitionError("local request identity was reused with different admission evidence")
            self._ensure_request_capacity(session)
            revision = self._next_revision(session)
            request = _WorkRecord(
                local_request_id=normalized_request_id,
                local_work_id=normalized_work_id,
                attempt_id=normalized_attempt_id,
                operation=normalized_operation,
                summary=normalized_summary,
                state=VolatileWorkState.LOCALLY_QUEUED,
                effects_started=False,
                failure_code=None,
                result=None,
                revision=revision,
            )
            session.requests[normalized_request_id] = request
            return self._work_snapshot(session, request)

    def get(self, session_id: str, local_request_id: str) -> VolatileWorkSnapshot:
        """Return one request only from its owning live session."""
        with self._lock:
            session, request = self._request(session_id, local_request_id)
            return self._work_snapshot(session, request)

    def latest(self, session_id: str) -> VolatileWorkSnapshot:
        """Return the most recently admitted request for one session."""
        normalized_session_id = _identifier(session_id, "session_id")
        with self._lock:
            session = self._sessions.get(normalized_session_id)
            if session is None or not session.requests:
                raise VolatileRequestNotFoundError("session has no volatile requests")
            request = next(reversed(session.requests.values()))
            return self._work_snapshot(session, request)

    def session(self, session_id: str) -> VolatileSessionSnapshot:
        """Return the bounded ordered request projection for one session."""
        normalized_session_id = _identifier(session_id, "session_id")
        with self._lock:
            session = self._sessions.get(normalized_session_id)
            if session is None:
                raise VolatileRequestNotFoundError("volatile session does not exist")
            return VolatileSessionSnapshot(
                session_id=session.session_id,
                revision=session.revision,
                requests=tuple(self._work_snapshot(session, request) for request in session.requests.values()),
            )

    def begin_dispatch(self, session_id: str, local_request_id: str) -> VolatileWorkSnapshot:
        """Enter dispatch without claiming that backend effects have started."""
        with self._lock:
            session, request = self._request(session_id, local_request_id)
            if request.state is not VolatileWorkState.LOCALLY_QUEUED or request.effects_started:
                raise VolatileTransitionError("only a locally queued request may begin dispatch")
            request.state = VolatileWorkState.DISPATCHING
            self._touch(session, request)
            return self._work_snapshot(session, request)

    def begin_invocation(self, session_id: str, local_request_id: str) -> VolatileWorkSnapshot:
        """Cross the effects fence immediately before entering the backend port."""
        with self._lock:
            session, request = self._request(session_id, local_request_id)
            if request.state is VolatileWorkState.WAITING_FOR_RESPONSE and request.effects_started:
                return self._work_snapshot(session, request)
            if request.state is not VolatileWorkState.DISPATCHING or request.effects_started:
                raise VolatileTransitionError("only a pre-effect dispatched request may begin invocation")
            request.effects_started = True
            request.state = VolatileWorkState.WAITING_FOR_RESPONSE
            self._touch(session, request)
            return self._work_snapshot(session, request)

    def cancel_before_effects(self, session_id: str, local_request_id: str) -> VolatileCancelDisposition:
        """Cancel only if no backend side effect can yet have occurred."""
        with self._lock:
            session, request = self._request(session_id, local_request_id)
            if request.state.terminal:
                return VolatileCancelDisposition.ALREADY_TERMINAL
            if request.effects_started:
                return VolatileCancelDisposition.TOO_LATE
            request.state = VolatileWorkState.CANCELLED
            request.failure_code = None
            self._touch(session, request)
            return VolatileCancelDisposition.CANCELLED

    def fail(
        self,
        session_id: str,
        local_request_id: str,
        *,
        failure_code: str,
        outcome_unknown: bool = False,
    ) -> VolatileWorkSnapshot:
        """Close one active request without fabricating a backend result."""
        normalized_code = _reason(failure_code, "failure_code")
        with self._lock:
            session, request = self._request(session_id, local_request_id)
            target = VolatileWorkState.OUTCOME_UNKNOWN if outcome_unknown else VolatileWorkState.FAILED
            if request.state is target and request.failure_code == normalized_code:
                return self._work_snapshot(session, request)
            if not request.state.active:
                raise VolatileTransitionError("only an active request may fail")
            if outcome_unknown and not request.effects_started:
                raise VolatileTransitionError("an outcome cannot be unknown before backend effects begin")
            request.state = target
            request.failure_code = normalized_code
            self._touch(session, request)
            return self._work_snapshot(session, request)

    def complete_with_result(
        self,
        session_id: str,
        local_request_id: str,
        *,
        result_id: str,
        presentation_id: str,
        envelope: ResponseOnlyResultEnvelope,
        delivery_id: str | None = None,
    ) -> VolatileWorkSnapshot:
        """Atomically commit success and its independent display/speech delivery state."""
        normalized_result_id = _identifier(result_id, "result_id")
        normalized_presentation_id = _identifier(presentation_id, "presentation_id")
        normalized_delivery_id = _identifier(
            delivery_id or f"delivery_vc_{uuid4().hex}",
            "delivery_id",
        )
        if not isinstance(envelope, ResponseOnlyResultEnvelope):
            raise TypeError("envelope must be ResponseOnlyResultEnvelope")
        with self._lock:
            session, request = self._request(session_id, local_request_id)
            if request.state is VolatileWorkState.SUCCEEDED and request.result is not None:
                result = request.result
                if (
                    result.result_id == normalized_result_id
                    and (delivery_id is None or result.delivery_id == normalized_delivery_id)
                    and result.presentation_id == normalized_presentation_id
                    and result.envelope == envelope
                ):
                    return self._work_snapshot(session, request)
                raise VolatileTransitionError("completed request received conflicting result evidence")
            if (
                request.state
                not in {
                    VolatileWorkState.DISPATCHING,
                    VolatileWorkState.WAITING_FOR_RESPONSE,
                }
                or not request.effects_started
            ):
                raise VolatileTransitionError("only a dispatched request may complete")
            request.result = _ResultRecord(
                result_id=normalized_result_id,
                delivery_id=normalized_delivery_id,
                presentation_id=normalized_presentation_id,
                envelope=envelope,
                display_state=VolatileDisplayState.READY,
                speech_state=(
                    VolatileSpeechState.READY if envelope.speech is not None else VolatileSpeechState.NOT_REQUESTED
                ),
            )
            request.state = VolatileWorkState.SUCCEEDED
            request.failure_code = None
            self._touch(session, request)
            return self._work_snapshot(session, request)

    def mark_display_delivered(self, session_id: str, local_request_id: str) -> VolatileWorkSnapshot:
        """Record successful UI delivery independently of speech."""
        return self._mark_display(session_id, local_request_id, VolatileDisplayState.DELIVERED)

    def mark_display_failed(self, session_id: str, local_request_id: str) -> VolatileWorkSnapshot:
        """Record terminal UI delivery failure independently of speech."""
        return self._mark_display(session_id, local_request_id, VolatileDisplayState.FAILED)

    def claim_speech(
        self,
        session_id: str,
        local_request_id: str,
        *,
        owner: VolatileSpeechClaimOwner,
    ) -> VolatileSpeechClaim | None:
        """Atomically grant one presentation path exclusive result speech ownership."""
        normalized_owner = VolatileSpeechClaimOwner(owner)
        with self._lock:
            session, request = self._request(session_id, local_request_id)
            result = self._result(request)
            source = result.speech_state
            allowed = (
                source is VolatileSpeechState.READY
                and normalized_owner in {VolatileSpeechClaimOwner.AUTOMATIC, VolatileSpeechClaimOwner.STATUS}
            ) or (source is VolatileSpeechState.INTERRUPTED and normalized_owner is VolatileSpeechClaimOwner.RECAP)
            if not allowed:
                return None
            if result.envelope.speech is None:
                raise VolatileTransitionError("result has no speech material")
            result.claim_revision += 1
            result.claim_owner = normalized_owner
            result.claim_source_state = source
            result.speech_state = VolatileSpeechState.CLAIMED
            result.playback_state = None
            result.delivery_outcome = None
            self._touch(session, request)
            return VolatileSpeechClaim(
                session_id=session.session_id,
                local_request_id=request.local_request_id,
                result_id=result.result_id,
                presentation_id=result.presentation_id,
                revision=result.claim_revision,
                owner=normalized_owner,
                speech_text=result.envelope.speech,
                source_state=source,
            )

    def release_speech_claim(self, claim: VolatileSpeechClaim) -> VolatileWorkSnapshot:
        """Release an unsent exact claim without reviving stale ownership."""
        if not isinstance(claim, VolatileSpeechClaim):
            raise TypeError("claim must be VolatileSpeechClaim")
        with self._lock:
            session, request, result = self._claimed_result(claim)
            if result.speech_state is not VolatileSpeechState.CLAIMED or result.claim_source_state is None:
                raise VolatileTransitionError("only an unsent speech claim may be released")
            result.speech_state = result.claim_source_state
            result.claim_owner = None
            result.claim_source_state = None
            self._touch(session, request)
            return self._work_snapshot(session, request)

    def mark_speech_started(self, claim: VolatileSpeechClaim) -> VolatileWorkSnapshot:
        """Record that the claimed response crossed the frontend send boundary."""
        with self._lock:
            session, request, result = self._claimed_result(claim)
            if result.speech_state is VolatileSpeechState.SPEAKING:
                return self._work_snapshot(session, request)
            if result.speech_state is not VolatileSpeechState.CLAIMED:
                raise VolatileTransitionError("only claimed speech may start")
            result.speech_state = VolatileSpeechState.SPEAKING
            result.claim_source_state = None
            self._touch(session, request)
            return self._work_snapshot(session, request)

    def mark_speech_delivered(
        self,
        claim: VolatileSpeechClaim,
        *,
        turn_id: str,
    ) -> VolatileWorkSnapshot:
        """Record transcript delivery while awaiting exact browser playback evidence."""
        normalized_turn_id = _identifier(turn_id, "turn_id")
        with self._lock:
            session, request, result = self._claimed_result(claim)
            if result.speech_state is VolatileSpeechState.AWAITING_PLAYBACK_RECEIPT:
                if result.turn_id == normalized_turn_id:
                    return self._work_snapshot(session, request)
                raise VolatileTransitionError("speech delivery already references another turn")
            if result.speech_state not in {VolatileSpeechState.CLAIMED, VolatileSpeechState.SPEAKING}:
                raise VolatileTransitionError("speech must be claimed or speaking before delivery")
            result.speech_state = VolatileSpeechState.AWAITING_PLAYBACK_RECEIPT
            result.claim_source_state = None
            result.turn_id = normalized_turn_id
            self._touch(session, request)
            return self._work_snapshot(session, request)

    def record_playback(
        self,
        claim: VolatileSpeechClaim,
        *,
        state: VolatilePlaybackState,
        heard_through_ms: int,
        audio_end_ms: int,
    ) -> VolatileWorkSnapshot:
        """Apply one exact terminal playback receipt to its claimed result."""
        normalized_state = VolatilePlaybackState(state)
        self._validate_playback_bounds(
            state=normalized_state,
            heard_through_ms=heard_through_ms,
            audio_end_ms=audio_end_ms,
        )
        with self._lock:
            session, request = self._request(claim.session_id, claim.local_request_id)
            result = self._result(request)
            if result.claim_revision != claim.revision or result.result_id != claim.result_id:
                raise VolatileStaleClaimError("speech claim is stale")
            if result.playback_state is not None:
                if (
                    result.playback_state is normalized_state
                    and result.heard_through_ms == heard_through_ms
                    and result.audio_end_ms == audio_end_ms
                ):
                    return self._work_snapshot(session, request)
                raise VolatileTransitionError("result already has different playback evidence")
            if result.speech_state is not VolatileSpeechState.AWAITING_PLAYBACK_RECEIPT:
                raise VolatileTransitionError("playback evidence requires delivered speech")
            result.playback_state = normalized_state
            result.heard_through_ms = heard_through_ms
            result.audio_end_ms = audio_end_ms
            result.speech_state = {
                VolatilePlaybackState.HEARD: VolatileSpeechState.HEARD,
                VolatilePlaybackState.INTERRUPTED: VolatileSpeechState.INTERRUPTED,
                VolatilePlaybackState.FAILED: VolatileSpeechState.FAILED,
            }[normalized_state]
            result.claim_owner = None
            result.claim_source_state = None
            self._touch(session, request)
            return self._work_snapshot(session, request)

    def record_delivery_outcome(
        self,
        claim: VolatileSpeechClaim,
        *,
        outcome: VolatileDeliveryOutcome,
    ) -> VolatileWorkSnapshot:
        """Close result speech when no browser playback receipt can exist."""
        normalized_outcome = VolatileDeliveryOutcome(outcome)
        with self._lock:
            session, request = self._request(claim.session_id, claim.local_request_id)
            result = self._result(request)
            if result.claim_revision != claim.revision or result.result_id != claim.result_id:
                raise VolatileStaleClaimError("speech claim is stale")
            if result.delivery_outcome is not None:
                if result.delivery_outcome is normalized_outcome:
                    return self._work_snapshot(session, request)
                raise VolatileTransitionError("result already has another delivery outcome")
            if result.speech_state not in {
                VolatileSpeechState.CLAIMED,
                VolatileSpeechState.SPEAKING,
                VolatileSpeechState.AWAITING_PLAYBACK_RECEIPT,
            }:
                raise VolatileTransitionError("result speech is not owned by this delivery")
            result.delivery_outcome = normalized_outcome
            result.speech_state = {
                VolatileDeliveryOutcome.SKIPPED: VolatileSpeechState.SKIPPED,
                VolatileDeliveryOutcome.CANCELLED: VolatileSpeechState.CANCELLED,
                VolatileDeliveryOutcome.FAILED: VolatileSpeechState.FAILED,
            }[normalized_outcome]
            result.claim_owner = None
            result.claim_source_state = None
            self._touch(session, request)
            return self._work_snapshot(session, request)

    def discard_session(self, session_id: str, *, force: bool = False) -> bool:
        """Discard one volatile session, requiring quiescence unless explicitly fenced."""
        normalized_session_id = _identifier(session_id, "session_id")
        with self._lock:
            session = self._sessions.get(normalized_session_id)
            if session is None:
                return False
            if not force and not self._session_evictable(session):
                raise VolatileTransitionError("session still contains active or undelivered requests")
            del self._sessions[normalized_session_id]
            return True

    def _mark_display(
        self,
        session_id: str,
        local_request_id: str,
        target: VolatileDisplayState,
    ) -> VolatileWorkSnapshot:
        with self._lock:
            session, request = self._request(session_id, local_request_id)
            result = self._result(request)
            if result.display_state is target:
                return self._work_snapshot(session, request)
            if result.display_state is not VolatileDisplayState.READY:
                raise VolatileTransitionError("display delivery is already terminal")
            result.display_state = target
            self._touch(session, request)
            return self._work_snapshot(session, request)

    def _ensure_session_capacity(self) -> None:
        if len(self._sessions) < self._max_sessions:
            return
        for session_id, session in self._sessions.items():
            if self._session_evictable(session):
                del self._sessions[session_id]
                return
        raise VolatileCapacityError("all volatile session slots contain active or undelivered requests")

    def _ensure_request_capacity(self, session: _SessionRecord) -> None:
        if len(session.requests) < self._max_requests_per_session:
            return
        for request_id, request in session.requests.items():
            if self._request_evictable(request):
                del session.requests[request_id]
                return
        raise VolatileCapacityError("all volatile request slots contain active or undelivered requests")

    @classmethod
    def _session_evictable(cls, session: _SessionRecord) -> bool:
        return all(cls._request_evictable(request) for request in session.requests.values())

    @staticmethod
    def _request_evictable(request: _WorkRecord) -> bool:
        if not request.state.terminal:
            return False
        result = request.result
        if result is None:
            return True
        return result.display_state.terminal and result.speech_state.terminal

    def _request(self, session_id: str, local_request_id: str) -> tuple[_SessionRecord, _WorkRecord]:
        normalized_session_id = _identifier(session_id, "session_id")
        normalized_request_id = _identifier(local_request_id, "local_request_id")
        session = self._sessions.get(normalized_session_id)
        if session is None:
            raise VolatileRequestNotFoundError("volatile session does not exist")
        request = session.requests.get(normalized_request_id)
        if request is None:
            raise VolatileRequestNotFoundError("volatile request does not exist in this session")
        return session, request

    @staticmethod
    def _result(request: _WorkRecord) -> _ResultRecord:
        if request.result is None:
            raise VolatileTransitionError("request has no result")
        return request.result

    def _claimed_result(
        self,
        claim: VolatileSpeechClaim,
    ) -> tuple[_SessionRecord, _WorkRecord, _ResultRecord]:
        if not isinstance(claim, VolatileSpeechClaim):
            raise TypeError("claim must be VolatileSpeechClaim")
        session, request = self._request(claim.session_id, claim.local_request_id)
        result = self._result(request)
        if (
            result.result_id != claim.result_id
            or result.presentation_id != claim.presentation_id
            or result.claim_revision != claim.revision
            or result.claim_owner is not claim.owner
        ):
            raise VolatileStaleClaimError("speech claim is stale")
        return session, request, result

    @staticmethod
    def _validate_playback_bounds(
        *,
        state: VolatilePlaybackState,
        heard_through_ms: int,
        audio_end_ms: int,
    ) -> None:
        for name, value in {
            "heard_through_ms": heard_through_ms,
            "audio_end_ms": audio_end_ms,
        }.items():
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if heard_through_ms > audio_end_ms:
            raise ValueError("heard_through_ms cannot exceed audio_end_ms")
        if state is VolatilePlaybackState.HEARD and heard_through_ms != audio_end_ms:
            raise ValueError("heard playback requires the complete audio boundary")
        if state is VolatilePlaybackState.INTERRUPTED and heard_through_ms >= audio_end_ms:
            raise ValueError("interrupted playback must stop before the complete audio boundary")

    @staticmethod
    def _next_revision(session: _SessionRecord) -> int:
        session.revision += 1
        return session.revision

    def _touch(self, session: _SessionRecord, request: _WorkRecord) -> None:
        request.revision = self._next_revision(session)

    @classmethod
    def _work_snapshot(cls, session: _SessionRecord, request: _WorkRecord) -> VolatileWorkSnapshot:
        return VolatileWorkSnapshot(
            session_id=session.session_id,
            local_request_id=request.local_request_id,
            local_work_id=request.local_work_id,
            attempt_id=request.attempt_id,
            operation=request.operation,
            summary=request.summary,
            state=request.state,
            effects_started=request.effects_started,
            failure_code=request.failure_code,
            result=None if request.result is None else cls._result_snapshot(request.result),
            revision=request.revision,
        )

    @staticmethod
    def _result_snapshot(result: _ResultRecord) -> VolatileResultSnapshot:
        return VolatileResultSnapshot(
            result_id=result.result_id,
            delivery_id=result.delivery_id,
            presentation_id=result.presentation_id,
            envelope=result.envelope,
            display_state=result.display_state,
            speech_state=result.speech_state,
            claim_revision=result.claim_revision,
            claim_owner=result.claim_owner,
            turn_id=result.turn_id,
            playback_state=result.playback_state,
            delivery_outcome=result.delivery_outcome,
            heard_through_ms=result.heard_through_ms,
            audio_end_ms=result.audio_end_ms,
        )


__all__ = [
    "VolatileCancelDisposition",
    "VolatileCapacityError",
    "VolatileDeliveryOutcome",
    "VolatileDisplayState",
    "VolatileLedgerError",
    "VolatilePlaybackState",
    "VolatileRequestNotFoundError",
    "VolatileResultSnapshot",
    "VolatileSessionSnapshot",
    "VolatileSpeechClaim",
    "VolatileSpeechClaimOwner",
    "VolatileSpeechState",
    "VolatileStaleClaimError",
    "VolatileTransitionError",
    "VolatileWorkLedger",
    "VolatileWorkSnapshot",
    "VolatileWorkState",
]
