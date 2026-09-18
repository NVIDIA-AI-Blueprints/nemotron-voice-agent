# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Slot parsing helpers for Thinker booking workflows."""

from __future__ import annotations

from typing import Any

from examples.frontend_backend_agent.airline.airports import iata_code


def canonical_pnr(value: str) -> str:
    """Return canonical alphanumeric PNR text as planned by the Thinker LLM."""
    compact = "".join(ch for ch in value.upper() if ch.isalnum())
    return compact or value.strip().upper()


def flight_identity(flight: dict[str, Any]) -> tuple[str, str, str, str]:
    """Return stable identity fields for a searched flight option."""
    return (
        str(flight.get("flight_id") or ""),
        str(flight.get("origin_airport") or ""),
        str(flight.get("dest_airport") or ""),
        str(flight.get("departure_time") or ""),
    )


def slot(slots: dict[str, Any], key: str) -> str | None:
    """Return a normalized string slot value."""
    value = slots.get(key)
    if value is None:
        return None
    if isinstance(value, bool):
        return str(value).lower()
    text = str(value).strip()
    return text or None


def bool_slot(slots: dict[str, Any], key: str) -> bool | None:
    """Return a parsed boolean slot value when present."""
    value = slots.get(key)
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"true", "yes", "confirm", "confirmed"}:
        return True
    if text in {"false", "no", "cancel", "cancelled"}:
        return False
    return None


def has_booking_preferences(slots: dict[str, Any]) -> bool:
    """Return whether the turn supplied seat or meal preferences."""
    return slot(slots, "seat_pref") is not None or slot(slots, "meal_pref") is not None


def summary_passenger_name(value: str | None) -> str | None:
    """Return a passenger name for confirmation summaries, excluding the MVP default."""
    passenger = (value or "").strip()
    if not passenger or passenger.lower() == "guest":
        return None
    return passenger


def extract_route(slots: dict[str, Any]) -> tuple[str | None, str | None]:
    """Extract origin and destination airport codes from planned route slots."""
    origin = slot_airport(slots, "origin_city") or slot_airport(slots, "origin_airport")
    destination = slot_airport(slots, "dest_city") or slot_airport(slots, "dest_airport")
    return origin, destination


def slot_airport(slots: dict[str, Any], key: str) -> str | None:
    """Return an IATA airport code from a city or airport slot."""
    value = slot(slots, key)
    return iata_code(value) if value else None


def extract_selected_flight(
    selection: str | None,
    search_results: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """Resolve an ordinal, index, or flight ID to a searched flight option."""
    if selection is None:
        return None
    lowered = selection.lower()
    ordinal = {"first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5}
    if lowered in ordinal and ordinal[lowered] <= len(search_results):
        return search_results[ordinal[lowered] - 1]
    if lowered.isdigit():
        index = int(lowered) - 1
        if 0 <= index < len(search_results):
            return search_results[index]
    for flight in search_results:
        if str(flight["flight_id"]).lower() == lowered:
            return flight
    return None
