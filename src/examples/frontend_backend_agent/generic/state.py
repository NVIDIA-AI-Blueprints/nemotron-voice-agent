# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""Session-local orchestration state for the generic domain."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

from examples.frontend_backend_agent.src.protocol import ThinkerLifecycleEvent


@dataclass(slots=True)
class GenericThinkerSessionState:
    """Only lifecycle state; factual tool results are never reused as fresh data."""

    active_task: asyncio.Task | None = None
    active_call_id: str | None = None
    events: list[ThinkerLifecycleEvent] = field(default_factory=list)

    def add_event(self, event: ThinkerLifecycleEvent) -> None:
        """Keep a bounded diagnostic history for this session."""
        self.events.append(event)
        if len(self.events) > 100:
            del self.events[:-100]

    def planner_state(self) -> dict[str, object]:
        """Expose minimal non-factual state to the planner."""
        return {"active_call_id": self.active_call_id}
