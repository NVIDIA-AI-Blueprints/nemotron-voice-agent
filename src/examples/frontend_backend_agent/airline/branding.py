# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""User-facing airline branding and display helpers."""

from __future__ import annotations

from typing import Any

from examples.frontend_backend_agent.airline.airports import spoken_time

USER_FACING_AIRLINE = "G Force Airlines"
USER_FACING_FLIGHT_PREFIX = "G Force Airline's"
INTERNAL_BACKEND_CARRIER = "Booking Server"


def format_flight_option(flight: dict[str, Any]) -> str:
    """Return a compact user-facing label for one search result."""
    price = flight.get("price_usd")
    price_text = f" for ${price}" if price is not None else ""
    departure = spoken_time(str(flight.get("departure_time") or ""))
    return f"{flight_label(flight)} at {departure}{price_text}"


def flight_label(flight: dict[str, Any]) -> str:
    """Return the branded flight label spoken to the user."""
    return f"{user_facing_carrier(flight)} {flight['flight_id']}"


def user_facing_carrier(flight: dict[str, Any]) -> str:
    """Map backend carrier names to the public airline brand."""
    carrier = str(flight.get("carrier") or "").strip()
    if not carrier or carrier == INTERNAL_BACKEND_CARRIER:
        return USER_FACING_FLIGHT_PREFIX
    return carrier


def user_facing_flight(flight: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of a flight with user-facing carrier fields."""
    display = dict(flight)
    display["carrier"] = user_facing_carrier(flight)
    display["airline_brand"] = USER_FACING_AIRLINE
    return display


def user_facing_record(record: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of a booking record with user-facing carrier fields."""
    display = dict(record)
    display["carrier"] = user_facing_carrier(record)
    display["airline_brand"] = USER_FACING_AIRLINE
    return display
