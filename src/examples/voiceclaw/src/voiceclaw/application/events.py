# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""Ordered backend-event admission before projection and presentation."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from voiceclaw.application.delivery import DeliveryCoordinator
from voiceclaw.domain.models import (
    AgentQueryChanged,
    AgentQueryProjection,
    BackendEvent,
    BackendEventKind,
    ResultAvailable,
    WorkProjection,
    WorkResult,
    WorkStateChanged,
)
from voiceclaw.ports.state import StateStore


class EventDisposition(StrEnum):
    """How a backend event relates to the contiguous local cursor."""

    APPLY = "apply"
    DUPLICATE = "duplicate"
    GAP = "gap"


class UnknownSessionError(LookupError):
    """Raised when an event cannot be bound to a known voice session."""


class MisroutedEventError(ValueError):
    """Raised when an event attachment does not own the target session."""


@dataclass(frozen=True, slots=True)
class EventDecision:
    """Admission decision without prematurely advancing durable state."""

    disposition: EventDisposition
    current_sequence: int
    expected_sequence: int


@dataclass(frozen=True, slots=True)
class EventApplication:
    """Outcome of canonical event admission and durable projection."""

    decision: EventDecision
    committed_sequence: int | None = None
    work_projection_changed: bool = False
    agent_query_projection_changed: bool = False
    presentation_id: str | None = None


class OrderedEventInbox:
    """Deduplicate replay and stop projection at the first sequence gap."""

    def __init__(self, state_store: StateStore) -> None:
        """Bind the inbox to the durable session/cursor store."""
        self._state_store = state_store

    def classify(self, event: BackendEvent) -> EventDecision:
        """Classify an event without applying it or advancing the cursor."""
        binding = self._state_store.get_session(event.session_id)
        if binding is None:
            raise UnknownSessionError(event.session_id)
        if binding.attachment_id != event.attachment_id:
            raise MisroutedEventError(f"attachment {event.attachment_id!r} does not own session {event.session_id!r}")

        current = binding.last_applied_sequence
        expected = current + 1
        if event.sequence <= current:
            disposition = EventDisposition.DUPLICATE
        elif event.sequence == expected:
            disposition = EventDisposition.APPLY
        else:
            disposition = EventDisposition.GAP
        return EventDecision(
            disposition=disposition,
            current_sequence=current,
            expected_sequence=expected,
        )

    def commit(self, event: BackendEvent) -> int:
        """Advance the cursor only after the caller persisted its projection."""
        decision = self.classify(event)
        if decision.disposition is EventDisposition.DUPLICATE:
            return decision.current_sequence
        if decision.disposition is EventDisposition.GAP:
            raise ValueError(f"cannot commit sequence {event.sequence}; expected {decision.expected_sequence}")
        return self._state_store.advance_cursor(event.session_id, event.sequence)


class BackendEventCoordinator:
    """Apply canonical backend events before triggering external presentation."""

    def __init__(
        self,
        *,
        inbox: OrderedEventInbox,
        state_store: StateStore,
        delivery: DeliveryCoordinator,
    ) -> None:
        """Bind ordered admission, durable projection, and delivery scheduling."""
        self._inbox = inbox
        self._state_store = state_store
        self._delivery = delivery

    async def ingest(self, event: BackendEvent) -> EventApplication:
        """Persist one contiguous canonical event and then drive presentation."""
        decision = self._inbox.classify(event)
        if decision.disposition is not EventDisposition.APPLY:
            return EventApplication(decision=decision)

        work_changed = False
        query_changed = False
        presentation_id: str | None = None
        if event.kind is BackendEventKind.WORK_STATE_CHANGED:
            payload = event.payload
            if not isinstance(payload, WorkStateChanged):
                raise TypeError("work.state.v1 event payload was not normalized")
            work_changed = self._state_store.save_work_projection(
                WorkProjection(
                    work_id=event.work_id,
                    session_id=event.session_id,
                    attachment_id=event.attachment_id,
                    state=payload.state,
                    sequence=event.sequence,
                    summary=payload.summary,
                    raw_state=payload.raw_state,
                    agent_target=payload.agent_target,
                    updated_at=event.occurred_at,
                )
            )
        elif event.kind is BackendEventKind.AGENT_QUERY_CHANGED:
            payload = event.payload
            if not isinstance(payload, AgentQueryChanged):
                raise TypeError("agent_query.state.v1 event payload was not normalized")
            query_changed = self._state_store.save_agent_query_projection(
                AgentQueryProjection(
                    query_id=payload.query_id,
                    work_id=event.work_id,
                    session_id=event.session_id,
                    attachment_id=event.attachment_id,
                    kind=payload.kind,
                    state=payload.state,
                    blocking=payload.blocking,
                    sequence=event.sequence,
                    prompt=payload.prompt,
                    raw_state=payload.raw_state,
                    updated_at=event.occurred_at,
                )
            )
        elif event.kind is BackendEventKind.RESULT_AVAILABLE:
            payload = event.payload
            if not isinstance(payload, ResultAvailable):
                raise TypeError("work.result.v1 event payload was not normalized")
            presentation_id = self._delivery.stage_result(
                WorkResult(
                    presentation_id=payload.presentation_id,
                    session_id=event.session_id,
                    attachment_id=event.attachment_id,
                    work_id=event.work_id,
                    result_id=payload.result_id,
                    sequence=event.sequence,
                    display=payload.display,
                    speech_text=payload.speech_text,
                    speech_route=payload.speech_route,
                    priority=payload.priority,
                    received_at=event.occurred_at,
                )
            ).presentation_id

        committed = self._inbox.commit(event)
        if presentation_id is not None:
            await self._delivery.deliver_pending(event.session_id)
        return EventApplication(
            decision=decision,
            committed_sequence=committed,
            work_projection_changed=work_changed,
            agent_query_projection_changed=query_changed,
            presentation_id=presentation_id,
        )
