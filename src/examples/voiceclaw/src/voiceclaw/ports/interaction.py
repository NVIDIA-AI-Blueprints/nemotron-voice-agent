# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""General backend interaction port; not limited to agent harnesses."""

from __future__ import annotations

from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType
from typing import Any, Protocol, runtime_checkable

from voiceclaw.domain.models import (
    BackendCapabilities,
    BackendEvent,
    BackendOperation,
    CapabilityEvidence,
    CapabilitySource,
)


@dataclass(frozen=True, slots=True)
class AttachRequest:
    """Request attachment to an operator-selected backend profile."""

    session_id: str
    conversation_id: str
    backend_profile: str
    resume_backend_session_id: str | None = None
    after_sequence: int | None = None


@dataclass(frozen=True, slots=True)
class BackendAttachment:
    """Opaque attachment plus its current capability revision."""

    attachment_id: str
    backend_session_id: str | None
    capabilities: BackendCapabilities
    capability_source: CapabilitySource = CapabilitySource.BACKEND_NEGOTIATED
    capability_source_id: str | None = None
    capability_evidence: CapabilityEvidence = field(init=False)

    def __post_init__(self) -> None:
        """Capture provenance for the exact typed claim on this attachment."""
        attachment_id = self.attachment_id.strip()
        if not attachment_id or "\x00" in attachment_id or len(attachment_id.encode("utf-8")) > 512:
            raise ValueError("attachment_id is invalid")
        object.__setattr__(self, "attachment_id", attachment_id)
        if self.backend_session_id is not None:
            backend_session_id = self.backend_session_id.strip()
            if not backend_session_id or "\x00" in backend_session_id or len(backend_session_id.encode("utf-8")) > 512:
                raise ValueError("backend_session_id is invalid")
            object.__setattr__(self, "backend_session_id", backend_session_id)
        if not isinstance(self.capabilities, BackendCapabilities):
            raise TypeError("capabilities must be BackendCapabilities")
        try:
            source = CapabilitySource(self.capability_source)
        except (TypeError, ValueError) as error:
            raise ValueError("capability_source is invalid") from error
        source_id = self.capability_source_id or self.capabilities.backend_kind
        object.__setattr__(self, "capability_source", source)
        object.__setattr__(self, "capability_source_id", source_id)
        object.__setattr__(
            self,
            "capability_evidence",
            CapabilityEvidence.capture(
                self.capabilities,
                source=source,
                source_id=source_id,
            ),
        )


@dataclass(frozen=True, slots=True)
class WorkCommand:
    """Server-bound command with VoiceClaw-owned idempotency identity."""

    command_id: str
    session_id: str
    attachment_id: str
    commit_id: str
    operation: BackendOperation
    capability_revision: str
    payload: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Detach command arguments from the caller's mutable mapping."""
        object.__setattr__(self, "payload", MappingProxyType(dict(self.payload)))


@dataclass(frozen=True, slots=True)
class ReconcileRequest:
    """Resolve an original command through the current authorized attachment.

    ``command`` is the immutable command originally dispatched.  The current
    attachment fields authorize only this reconciliation attempt; they never
    rewrite or redispatch the original command body.
    """

    command: WorkCommand
    current_attachment_id: str
    current_backend_session_id: str | None
    current_capability_revision: str


class BackendAdmission(StrEnum):
    """Narrow command-admission outcome returned by any backend adapter."""

    ACCEPTED = "accepted"
    REJECTED = "rejected"
    INCONCLUSIVE = "inconclusive"


@dataclass(frozen=True, slots=True)
class BackendCommandReceipt:
    """Backend admission result with an optional backend-issued Work ID."""

    command_id: str
    admission: BackendAdmission
    work_id: str | None = None
    reason_code: str | None = None


@dataclass(frozen=True, slots=True)
class DetachRequest:
    """Detach a local voice session while preserving presentation evidence."""

    attachment_id: str
    last_presented_sequence: int
    reason: str


@runtime_checkable
class AgentInteractionPort(Protocol):
    """Backend-neutral attachment, command, event, and replay boundary."""

    async def attach(self, request: AttachRequest) -> BackendAttachment:
        """Attach or resume a backend interaction."""
        ...

    async def execute(self, command: WorkCommand) -> BackendCommandReceipt:
        """Admit one idempotent semantic command."""
        ...

    async def reconcile(self, request: ReconcileRequest) -> BackendCommandReceipt:
        """Resolve an inconclusive command without blindly redispatching it."""
        ...

    def events(self, attachment_id: str, *, after_sequence: int | None) -> AsyncIterator[BackendEvent]:
        """Stream live or replayed ordered backend events."""
        ...

    async def detach(self, request: DetachRequest) -> None:
        """Detach this client presentation without cancelling backend Work."""
        ...
