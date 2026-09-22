# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""Authenticated selected-agent readiness boundary.

Readiness is intentionally separate from backend discovery and ordinary work
execution.  A backend may be reachable while the deployment-scoped credential
is unable to access the selected agent; only this port is allowed to attest the
latter condition.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Protocol, runtime_checkable


class SelectedAgentReadinessCode(StrEnum):
    """Finite log-safe outcomes for a non-generative access check."""

    UNAVAILABLE = "selected_agent_unavailable"
    ACCESS_DENIED = "selected_agent_access_denied"
    CREDENTIAL_EXPIRED = "selected_agent_credential_expired"
    CREDENTIAL_REVOKED = "selected_agent_credential_revoked"
    WRONG_AGENT = "selected_agent_wrong_agent"
    REPLACED_AGENT = "selected_agent_replaced_agent"
    ENDPOINT_UNAVAILABLE = "selected_agent_endpoint_unavailable"
    PROTOCOL_ERROR = "selected_agent_protocol_error"
    READINESS_UNSUPPORTED = "selected_agent_readiness_unsupported"


class SelectedAgentReadinessError(RuntimeError):
    """A safe, machine-readable selected-agent readiness failure."""

    def __init__(
        self,
        code: SelectedAgentReadinessCode | str = SelectedAgentReadinessCode.UNAVAILABLE,
    ) -> None:
        """Retain one allowlisted reason code, never a provider response."""
        try:
            resolved = SelectedAgentReadinessCode(code)
        except (TypeError, ValueError):
            resolved = SelectedAgentReadinessCode.UNAVAILABLE
        self.code = resolved.value
        super().__init__(resolved.value)


@runtime_checkable
class SelectedAgentReadinessPort(Protocol):
    """Verify scoped access to the configured agent without sending a prompt."""

    async def check_selected_agent(self) -> None:
        """Return only after authenticated selected-agent access is verified."""
        ...


__all__ = ["SelectedAgentReadinessCode", "SelectedAgentReadinessError", "SelectedAgentReadinessPort"]
