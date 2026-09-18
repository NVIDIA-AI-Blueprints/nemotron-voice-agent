# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Booking-server row transformation helpers for the Frontend/Backend Agent example."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, date, datetime, timedelta
from typing import Any

from examples.frontend_backend_agent.airline.airports import airport_display_name


def server_flight_to_option(row: dict[str, Any], *, fallback_date: str) -> dict[str, Any]:
    """Convert a booking-server flight row into a Thinker flight option."""
    raw_departure = row.get("departure")
    raw_arrival = row.get("arrival")
    departure, arrival = materialize_timestamps(
        "" if raw_departure is None else str(raw_departure),
        "" if raw_arrival is None else str(raw_arrival),
        fallback_date,
    )
    origin = str(row.get("origin") or "").upper()
    destination = str(row.get("destination") or "").upper()
    return {
        "flight_id": str(row.get("flight_number") or "").upper(),
        "carrier": "Booking Server",
        "origin_city": airport_display_name(origin),
        "dest_city": airport_display_name(destination),
        "origin_airport": origin,
        "dest_airport": destination,
        "date": departure[:10] or fallback_date,
        "departure_time": departure,
        "arrival_time": arrival,
        "duration_minutes": duration_minutes(departure, arrival),
        "price_usd": None,
        "cabin": row.get("cabin"),
    }


def materialize_timestamps(departure: str, arrival: str, travel_date: str) -> tuple[str, str]:
    """Move seed timestamps onto ``travel_date`` while preserving duration."""
    if not departure or "T" not in departure:
        return departure, arrival
    try:
        departure_dt = datetime.fromisoformat(departure)
        arrival_dt = datetime.fromisoformat(arrival or departure)
        target_date = date.fromisoformat(travel_date)
        materialized_departure = departure_dt.replace(
            year=target_date.year,
            month=target_date.month,
            day=target_date.day,
        )
        duration = arrival_dt - departure_dt
        if duration.total_seconds() < 0:
            duration += timedelta(days=1)
        return materialized_departure.isoformat(), (materialized_departure + duration).isoformat()
    except ValueError:
        materialized_departure = f"{travel_date}{departure[10:]}"
        materialized_arrival = f"{travel_date}{arrival[10:]}" if arrival and "T" in arrival else arrival
        return materialized_departure, materialized_arrival


def duration_minutes(departure: str | int | float | None, arrival: str | int | float | None) -> int:
    """Return non-negative whole minutes between departure and arrival."""
    departure_dt = parse_timestamp(departure)
    arrival_dt = parse_timestamp(arrival)
    if departure_dt is None or arrival_dt is None:
        return 0
    delta = arrival_dt - departure_dt
    if delta.total_seconds() < 0:
        delta += timedelta(days=1)
    return max(0, int(delta.total_seconds() / 60))


def parse_timestamp(value: str | int | float | None) -> datetime | None:
    """Parse ISO or epoch timestamps into UTC datetimes."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(value, UTC)
    text = value.strip()
    if not text:
        return None
    try:
        if text.isdigit():
            return datetime.fromtimestamp(int(text), UTC)
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except (OSError, ValueError):
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def unique_flights(flights: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Collapse repeated daily seed rows into one option per flight/time."""
    unique: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str, str]] = set()
    for flight in flights:
        key = (
            str(flight.get("flight_id") or ""),
            str(flight.get("origin_airport") or ""),
            str(flight.get("dest_airport") or ""),
            str(flight.get("departure_time") or "")[11:],
        )
        if key in seen:
            continue
        seen.add(key)
        unique.append(flight)
    return unique


def server_booking_to_record(
    data: dict[str, Any],
    *,
    flight: dict[str, Any],
    passenger_name: str | None = None,
    seat_pref: str | None,
    meal_pref: str | None,
) -> dict[str, Any]:
    """Convert a booking-server PNR creation response into a Thinker record."""
    pnr = str(data.get("pnr") or "").upper()
    response_passenger = str(data.get("passenger") or "").strip()
    provided_passenger = str(passenger_name or "").strip()
    return {
        "pnr": pnr,
        "confirmation_code": data.get("confirmation_code"),
        "passenger_name": response_passenger or provided_passenger or "Guest",
        "status": "confirmed",
        "flight_id": str(data.get("flight_number") or flight["flight_id"]).upper(),
        "carrier": flight.get("carrier", ""),
        "origin_city": flight["origin_city"],
        "dest_city": flight["dest_city"],
        "date": str(data.get("departure") or flight["departure_time"])[:10],
        "departure_time": data.get("departure") or flight["departure_time"],
        "seat_pref": seat_pref,
        "meal_pref": meal_pref,
        "price_usd": data.get("price") or flight.get("price_usd"),
        "currency": data.get("currency", "USD"),
    }


def server_pnr_to_record(data: dict[str, Any]) -> dict[str, Any]:
    """Convert a booking-server PNR lookup response into a Thinker record."""
    origin = str(data.get("origin") or "").upper()
    destination = str(data.get("destination") or "").upper()
    ancillaries = data.get("ancillaries") if isinstance(data.get("ancillaries"), dict) else {}
    return {
        "pnr": str(data.get("pnr") or "").upper(),
        "passenger_name": data.get("passenger") or "Guest",
        "status": data.get("booking_status") or data.get("status") or "unknown",
        "flight_id": str(data.get("flight_number") or "").upper(),
        "carrier": "Booking Server",
        "origin_city": airport_display_name(origin),
        "dest_city": airport_display_name(destination),
        "date": str(data.get("departure") or "")[:10],
        "departure_time": data.get("departure"),
        "seat_pref": ancillaries.get("seat"),
        "meal_pref": ancillaries.get("meal"),
    }


def sort_flights(flights: Sequence[dict[str, Any]], sorting: str | None) -> list[dict[str, Any]]:
    """Sort flight options according to planner/user preference."""
    if sorting == "departure_time":
        return sorted(flights, key=lambda item: str(item.get("departure_time") or ""))
    if sorting == "duration":
        return sorted(flights, key=lambda item: int(item.get("duration_minutes") or 0))
    return sorted(flights, key=lambda item: int(item.get("price_usd") or 0))


# Backwards-compatible aliases for tests and any local callers using the old
# private names during the module split.
_server_flight_to_option = server_flight_to_option
_materialize_timestamps = materialize_timestamps
_duration_minutes = duration_minutes
_parse_timestamp = parse_timestamp
_unique_flights = unique_flights
_server_booking_to_record = server_booking_to_record
_server_pnr_to_record = server_pnr_to_record
_sort_flights = sort_flights
