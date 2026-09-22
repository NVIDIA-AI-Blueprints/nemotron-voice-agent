# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""Per-session display and speech presentation scheduling."""

from __future__ import annotations

import heapq
from collections import deque
from dataclasses import dataclass, field, replace

from voiceclaw.domain.models import (
    DisplayState,
    FrontendActivity,
    PresentationRecord,
    ResultState,
    SpeechState,
    WorkResult,
    utc_now,
)


@dataclass(frozen=True, slots=True)
class OfferOutcome:
    """Result of idempotently offering a normalized result."""

    record: PresentationRecord
    created: bool


@dataclass(slots=True)
class _SessionQueue:
    activity: FrontendActivity = field(default_factory=FrontendActivity)
    records: dict[str, PresentationRecord] = field(default_factory=dict)
    result_keys: dict[tuple[str, str, str, int], str] = field(default_factory=dict)
    display_ready: deque[str] = field(default_factory=deque)
    display_enqueued: set[str] = field(default_factory=set)
    speech_ready: list[tuple[int, int, int, str]] = field(default_factory=list)
    speech_enqueued: set[str] = field(default_factory=set)
    active_speech_id: str | None = None
    enqueue_counter: int = 0


class PresentationScheduler:
    """Keep UI delivery independent from safely admitted speech.

    The scheduler is intentionally in-memory. Its records are persisted through
    ``StateStore`` and can be restored after a client reconnect or process
    restart. It never changes backend Work state and never routes one session's
    result to another.
    """

    def __init__(self) -> None:
        """Create an empty collection of isolated session queues."""
        self._sessions: dict[str, _SessionQueue] = {}

    def set_activity(self, session_id: str, activity: FrontendActivity) -> None:
        """Update the independent input/model/output activity axes."""
        self._session(session_id).activity = activity

    def activity(self, session_id: str) -> FrontendActivity:
        """Return the latest activity snapshot for a session."""
        return self._session(session_id).activity

    def offer(self, result: WorkResult) -> OfferOutcome:
        """Offer one result idempotently to display and speech channels."""
        queue = self._session(result.session_id)
        candidate = PresentationRecord.from_result(result)
        result_key = result.immutable_identity
        existing_id = queue.result_keys.get(result_key)
        if existing_id is not None:
            existing = queue.records[existing_id]
            self._require_same_evidence(existing, candidate)
            return OfferOutcome(record=existing, created=False)
        if result.presentation_id in queue.records:
            existing = queue.records[result.presentation_id]
            self._require_same_evidence(existing, candidate)
            return OfferOutcome(record=existing, created=False)

        record = candidate
        queue.records[record.presentation_id] = record
        queue.result_keys[result_key] = record.presentation_id
        self._enqueue_display(queue, record)
        if record.speech_state is SpeechState.QUEUED:
            self._enqueue_speech(queue, record)
        return OfferOutcome(record=record, created=True)

    def restore(
        self,
        record: PresentationRecord,
        *,
        reclaim_delivery_leases: bool = False,
    ) -> PresentationRecord:
        """Restore persisted state and optionally reclaim abandoned local leases.

        ``reclaim_delivery_leases`` is for a new client attachment. It requeues
        display-ready records that an earlier attachment claimed and converts
        claimed or speaking speech into deferred speech. Backend Work remains
        untouched.
        """
        queue = self._session(record.session_id)
        result_key = record.immutable_identity
        existing_by_id = queue.records.get(record.presentation_id)
        existing_id = queue.result_keys.get(result_key)
        if existing_by_id is not None:
            self._require_same_evidence(existing_by_id, record)
        if existing_id is not None:
            existing_by_key = queue.records[existing_id]
            self._require_same_evidence(existing_by_key, record)
        current = existing_by_id or (queue.records[existing_id] if existing_id is not None else record)

        if reclaim_delivery_leases and current.speech_state in {
            SpeechState.CLAIMED,
            SpeechState.SPEAKING,
        }:
            current = replace(current, speech_state=SpeechState.DEFERRED, updated_at=utc_now())
            if queue.active_speech_id == current.presentation_id:
                queue.active_speech_id = None

        queue.records[current.presentation_id] = current
        queue.result_keys[result_key] = current.presentation_id
        if current.speech_state in {SpeechState.CLAIMED, SpeechState.SPEAKING}:
            if queue.active_speech_id not in {None, current.presentation_id}:
                raise ValueError("session already has a different active speech lease")
            queue.active_speech_id = current.presentation_id
        if current.display_state is DisplayState.READY:
            self._enqueue_display(queue, current)
        if current.speech_state in {SpeechState.QUEUED, SpeechState.DEFERRED}:
            self._enqueue_speech(queue, current)
        return current

    def claim_next_display(self, session_id: str) -> PresentationRecord | None:
        """Claim the next UI payload without consulting voice/model activity."""
        queue = self._session(session_id)
        while queue.display_ready:
            presentation_id = queue.display_ready.popleft()
            queue.display_enqueued.discard(presentation_id)
            record = queue.records[presentation_id]
            if record.display_state is DisplayState.READY:
                return record
        return None

    def mark_displayed(self, session_id: str, presentation_id: str) -> PresentationRecord:
        """Record a client display receipt independently from speech."""
        return self._replace(
            session_id,
            presentation_id,
            display_state=DisplayState.DELIVERED,
        )

    def mark_display_failed(self, session_id: str, presentation_id: str) -> PresentationRecord:
        """Record a terminal display failure without changing speech."""
        return self._replace(
            session_id,
            presentation_id,
            display_state=DisplayState.FAILED,
        )

    def claim_next_speech(self, session_id: str) -> PresentationRecord | None:
        """Atomically claim one queued speech item when the floor is safe."""
        queue = self._session(session_id)
        if queue.active_speech_id is not None or not queue.activity.speech_floor_available:
            return None

        while queue.speech_ready:
            _, _, _, presentation_id = heapq.heappop(queue.speech_ready)
            queue.speech_enqueued.discard(presentation_id)
            record = queue.records[presentation_id]
            if record.speech_state not in {SpeechState.QUEUED, SpeechState.DEFERRED}:
                continue
            claimed = self._replace(
                session_id,
                presentation_id,
                speech_state=SpeechState.CLAIMED,
            )
            queue.active_speech_id = presentation_id
            return claimed
        return None

    def mark_speech_started(self, session_id: str, presentation_id: str) -> PresentationRecord:
        """Record that the frontend runtime began rendering speech."""
        queue = self._session(session_id)
        if queue.active_speech_id != presentation_id:
            raise ValueError("presentation does not own the active speech lease")
        return self._replace(session_id, presentation_id, speech_state=SpeechState.SPEAKING)

    def mark_speech_completed(self, session_id: str, presentation_id: str) -> PresentationRecord:
        """Record heard-through completion and release the speech lease."""
        queue = self._session(session_id)
        if queue.active_speech_id != presentation_id:
            raise ValueError("presentation does not own the active speech lease")
        record = self._replace(session_id, presentation_id, speech_state=SpeechState.HEARD)
        queue.active_speech_id = None
        return record

    def mark_speech_interrupted(self, session_id: str, presentation_id: str) -> PresentationRecord:
        """Release interrupted speech without automatically replaying it."""
        queue = self._session(session_id)
        if queue.active_speech_id != presentation_id:
            raise ValueError("presentation does not own the active speech lease")
        record = self._replace(session_id, presentation_id, speech_state=SpeechState.INTERRUPTED)
        queue.active_speech_id = None
        return record

    def mark_speech_failed(self, session_id: str, presentation_id: str) -> PresentationRecord:
        """Record a rendering failure and release the active speech lease."""
        queue = self._session(session_id)
        if queue.active_speech_id != presentation_id:
            raise ValueError("presentation does not own the active speech lease")
        record = self._replace(session_id, presentation_id, speech_state=SpeechState.FAILED)
        queue.active_speech_id = None
        return record

    def get(self, session_id: str, presentation_id: str) -> PresentationRecord:
        """Return one presentation record."""
        return self._session(session_id).records[presentation_id]

    def records(self, session_id: str) -> tuple[PresentationRecord, ...]:
        """Return a stable snapshot ordered by backend sequence."""
        return tuple(sorted(self._session(session_id).records.values(), key=lambda record: record.sequence))

    def pending_speech_count(self, session_id: str) -> int:
        """Return the number of unclaimed speakable records."""
        return sum(
            record.speech_state in {SpeechState.QUEUED, SpeechState.DEFERRED}
            for record in self._session(session_id).records.values()
        )

    def active_speech_id(self, session_id: str) -> str | None:
        """Return the presentation currently holding this session's speech lease."""
        return self._session(session_id).active_speech_id

    def _session(self, session_id: str) -> _SessionQueue:
        normalized = session_id.strip()
        if not normalized:
            raise ValueError("session_id must not be empty")
        return self._sessions.setdefault(normalized, _SessionQueue())

    @staticmethod
    def _require_same_evidence(existing: PresentationRecord, candidate: PresentationRecord) -> None:
        if not existing.has_same_immutable_evidence(candidate):
            raise ValueError("presentation identity conflicts with existing result evidence")

    @staticmethod
    def _enqueue_display(queue: _SessionQueue, record: PresentationRecord) -> None:
        if record.presentation_id not in queue.display_enqueued:
            queue.display_ready.append(record.presentation_id)
            queue.display_enqueued.add(record.presentation_id)

    @staticmethod
    def _enqueue_speech(queue: _SessionQueue, record: PresentationRecord) -> None:
        if record.presentation_id in queue.speech_enqueued:
            return
        queue.enqueue_counter += 1
        heapq.heappush(
            queue.speech_ready,
            (int(record.priority), record.sequence, queue.enqueue_counter, record.presentation_id),
        )
        queue.speech_enqueued.add(record.presentation_id)

    def _replace(self, session_id: str, presentation_id: str, **changes: object) -> PresentationRecord:
        queue = self._session(session_id)
        current = queue.records[presentation_id]
        updated = replace(current, updated_at=utc_now(), **changes)
        if (
            updated.result_state is ResultState.AVAILABLE
            and updated.display_state is DisplayState.DELIVERED
            and updated.speech_state in {SpeechState.NOT_REQUESTED, SpeechState.HEARD}
        ):
            updated = replace(updated, result_state=ResultState.ACKNOWLEDGED)
        queue.records[presentation_id] = updated
        return updated
