# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""Persistence port for local recovery evidence, not backend Work authority."""

from __future__ import annotations

from typing import Protocol

from voiceclaw.domain.models import (
    AgentQueryProjection,
    CommandRecord,
    PresentationRecord,
    SessionBinding,
    SessionControlSnapshot,
    WorkProjection,
)


class StaleSessionControlError(RuntimeError):
    """Raised when admission was based on an obsolete control snapshot."""


class StateStore(Protocol):
    """Local state required for reconnect, dedupe, and delivery recovery."""

    def save_session(self, binding: SessionBinding) -> None:
        """Insert or update a local session/backend attachment mapping."""
        ...

    def get_session(self, session_id: str) -> SessionBinding | None:
        """Load a session mapping."""
        ...

    def discard_ephemeral_session(self, session_id: str) -> None:
        """Delete a mapping only when it has no durable backend or Work evidence."""
        ...

    def advance_cursor(self, session_id: str, sequence: int) -> int:
        """Advance and return the monotonically increasing applied-event cursor."""
        ...

    def advance_presented_cursor(self, session_id: str, sequence: int) -> int:
        """Persist a prefix already proven by the presentation-receipt layer."""
        ...

    def save_work_projection(self, projection: WorkProjection) -> bool:
        """Persist a newer backend-owned Work projection."""
        ...

    def get_work_projection(self, attachment_id: str, work_id: str) -> WorkProjection | None:
        """Load one Work projection."""
        ...

    def get_session_work_projection(self, session_id: str, work_id: str) -> WorkProjection | None:
        """Load the newest projection for backend Work bound to one voice session."""
        ...

    def get_session_control_snapshot(self, session_id: str) -> SessionControlSnapshot:
        """Atomically load the applied Work/query prefix and local query claims."""
        ...

    def save_agent_query_projection(self, projection: AgentQueryProjection) -> bool:
        """Persist a newer backend-owned AgentQuery projection."""
        ...

    def get_session_agent_query_projection(
        self,
        session_id: str,
        query_id: str,
    ) -> AgentQueryProjection | None:
        """Load one session-scoped backend AgentQuery projection."""
        ...

    def pending_agent_queries(self, session_id: str) -> tuple[AgentQueryProjection, ...]:
        """Load the newest pending AgentQuery projections for one voice session."""
        ...

    def save_presentation(self, record: PresentationRecord) -> bool:
        """Persist a presentation idempotently."""
        ...

    def update_presentation(self, record: PresentationRecord) -> None:
        """Persist independent display and speech lifecycle changes."""
        ...

    def get_presentation(self, presentation_id: str) -> PresentationRecord | None:
        """Load one presentation."""
        ...

    def get_result_presentation(
        self,
        session_id: str,
        work_id: str,
        result_id: str,
        sequence: int,
    ) -> PresentationRecord | None:
        """Load the presentation for one attachment-independent backend result."""
        ...

    def pending_presentations(self, session_id: str) -> tuple[PresentationRecord, ...]:
        """Load pending records, including speech with an abandoned local lease."""
        ...

    def save_command(
        self,
        command: CommandRecord,
        *,
        expected_control_revision: int | None = None,
    ) -> bool:
        """Persist an idempotent outbox record and atomically claim a query.

        Query-response commands reserve their backend-session/query identity
        until admission is definitively rejected.  ``False`` therefore means
        either the command ID or that query claim already exists.

        When ``expected_control_revision`` is supplied, persistence must fail
        with :class:`StaleSessionControlError` unless the captured session
        control snapshot is still current in the same write transaction.
        """
        ...

    def update_command(self, command: CommandRecord) -> None:
        """Persist an admission outcome without asserting Work completion."""
        ...

    def get_command(self, command_id: str) -> CommandRecord | None:
        """Load one command/outbox record by its idempotency identity."""
        ...

    def get_agent_query_command_claim(
        self,
        backend_session_id: str,
        query_id: str,
    ) -> CommandRecord | None:
        """Load the non-rejected command that locally claims one backend query."""
        ...

    def unsettled_commands(self, session_id: str) -> tuple[CommandRecord, ...]:
        """Load commands whose acceptance outcome still needs reconciliation."""
        ...
