# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pydantic request / response models for the booking HTTP surface.

Kept intentionally loose on response bodies — most endpoints return a
pass-through dict from :class:`examples.frontend_backend_agent.airline.database.api.BookingAPI`.  Only
the mutation request bodies get strict models so invalid payloads
fail at the boundary instead of deep in the DB layer.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class Ancillaries(BaseModel):
    """Seat, baggage, and meal details attached to a PNR."""

    seat: str | None = None
    bag_count: int = 0
    meal: str | None = None


class Booking(BaseModel):
    """Joined booking record returned by the booking service."""

    pnr: str
    passenger: str
    flight_number: str
    origin: str
    destination: str
    departure: str
    cabin: str
    fare_basis: str
    elite_tier: str
    status: str  # FLIGHT operational status (scheduled / delayed / ...).
    booking_status: str = "active"  # PNR lifecycle (active / cancelled).
    delay_minutes: int
    ancillaries: Ancillaries


class Flight(BaseModel):
    """Scheduled flight row exposed by route and status endpoints."""

    flight_number: str
    origin: str
    destination: str
    departure: str
    arrival: str
    cabin: str
    status: str = "scheduled"
    delay_minutes: int = 0


class BookingCreateRequest(BaseModel):
    """Request body for creating a new PNR."""

    passenger: str = Field(..., min_length=1, description="Passenger name for the new PNR.")
    origin: str = Field(..., min_length=3, max_length=3, description="Origin IATA (uppercase).")
    destination: str = Field(..., min_length=3, max_length=3, description="Destination IATA (uppercase).")
    flight_number: str = Field(..., min_length=2, description="Target flight (2L + digits, uppercase).")
    departure: str | None = Field(default=None, description="ISO 8601 departure timestamp for selected flight.")
    seat: str | None = Field(default=None, description="Seat preference ('14A', 'aisle', 'window') or null.")
    meal: str | None = Field(default=None, description="Meal preference (free-form, canonicalised server-side).")
    cabin: str | None = Field(default=None, description="Cabin override; defaults to the flight's scheduled cabin.")


class RebookRequest(BaseModel):
    """Request body for moving an existing PNR to another flight."""

    new_flight_number: str = Field(..., min_length=2, description="Target flight (2L + digits, uppercase).")
    departure: str | None = Field(
        default=None,
        description=(
            "ISO 8601 departure timestamp (e.g. 2026-05-02T15:00:00). "
            "Required to disambiguate when the flight number recurs across "
            "dates / routes; falls back to earliest scheduled when absent."
        ),
    )
    seat: str | None = Field(
        default=None,
        description=(
            "Optional seat update. 'keep' / 'same' / the existing seat is a no-op; "
            "otherwise takes effect atomically with the flight swap."
        ),
    )
    meal: str | None = Field(
        default=None,
        description=(
            "Optional meal update. Free-form ('vegetarian' / 'non_vegetarian' / etc.) "
            "is canonicalised to a meal code; 'keep' is a no-op."
        ),
    )


class CancelRequest(BaseModel):
    """Request body for cancelling a PNR."""

    kind: str = Field(..., description="Short classifier: airline_refund / voluntary_refundable / etc.")
    policy_ref: str = Field(..., description="Policy identifier to cite in the audit log.")


class StandbyRequest(BaseModel):
    """Request body for adding a PNR to a standby list."""

    flight_number: str = Field(..., min_length=2)
    departure: str | None = Field(
        default=None,
        description="ISO 8601 departure timestamp; disambiguates when the flight number recurs.",
    )


class ActivityEvent(BaseModel):
    """Audit-log event associated with a PNR."""

    id: int
    pnr: str
    action: str
    session_id: str | None = None
    from_flight: str | None = None
    from_origin: str | None = None
    from_destination: str | None = None
    from_departure: str | None = None
    to_flight: str | None = None
    to_origin: str | None = None
    to_destination: str | None = None
    to_departure: str | None = None
    outcome: str | None = None
    confirmation_code: str | None = None
    policy_ref: str | None = None
    amounts: dict[str, Any] | None = None
    notes: str | None = None
    created_at: str


class AncillariesDiff(BaseModel):
    """Preview of ancillary carry-over for a proposed flight change."""

    pnr: str
    new_flight_number: str
    seat: dict[str, Any]
    bags: dict[str, Any]
    meal: dict[str, Any]
    losses: list[str]


class MutationResponse(BaseModel):
    """Common response for booking mutations that mint confirmation codes."""

    confirmation_code: str
    pnr: str


class RebookResponse(MutationResponse):
    """Response payload for a completed rebook operation."""

    booking: Booking
    from_flight: str
    to_flight: str
