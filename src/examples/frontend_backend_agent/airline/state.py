# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Per-session state for the independent Frontend/Backend Agent example."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

from examples.frontend_backend_agent.src.protocol import ThinkerLifecycleEvent

MAX_LIFECYCLE_EVENTS = 200


@dataclass(slots=True)
class BookingDraft:
    """Current booking draft created after a user selects a searched flight."""

    flight: dict[str, Any]
    seat_pref: str | None = None
    meal_pref: str | None = None
    passenger_name: str | None = None


@dataclass(slots=True)
class ThinkerSessionState:
    """Mutable state scoped to one frontend/backend voice session."""

    lifecycle_events: list[ThinkerLifecycleEvent] = field(default_factory=list)
    active_call_id: str | None = None
    active_task: asyncio.Task | None = field(default=None, repr=False)
    search_context: dict[str, Any] = field(default_factory=dict)
    search_results: list[dict[str, Any]] = field(default_factory=list)
    booking_draft: BookingDraft | None = None
    waiting_for_preferences: bool = False
    waiting_for_confirmation: bool = False

    def reset_booking(self) -> None:
        """Clear booking-scoped state while preserving completed PNR records."""
        self.booking_draft = None
        self.waiting_for_preferences = False
        self.waiting_for_confirmation = False

    def reset_search_and_booking(self) -> None:
        """Clear search and booking state for a new flight-search workflow."""
        self.search_context.clear()
        self.search_results.clear()
        self.reset_booking()

    def add_event(self, event: ThinkerLifecycleEvent) -> None:
        """Append an internal lifecycle/history event."""
        self.lifecycle_events.append(event)
        excess = len(self.lifecycle_events) - MAX_LIFECYCLE_EVENTS
        if excess > 0:
            del self.lifecycle_events[:excess]

    def lifecycle_as_dicts(self) -> list[dict[str, Any]]:
        """Return serializable lifecycle history."""
        return [event.as_dict() for event in self.lifecycle_events]
