# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Thinker flight-search tool implementation."""

from __future__ import annotations

from typing import Any

from loguru import logger

from examples.frontend_backend_agent.airline.airports import airport_display_name
from examples.frontend_backend_agent.airline.backend import BookingBackend
from examples.frontend_backend_agent.airline.branding import format_flight_option, user_facing_flight
from examples.frontend_backend_agent.airline.slot_parsing import extract_route, slot
from examples.frontend_backend_agent.airline.state import ThinkerSessionState
from examples.frontend_backend_agent.src.protocol import response_hint, tool_result


async def flight_search(
    *,
    state: ThinkerSessionState,
    backend: BookingBackend,
    slots: dict[str, Any],
) -> dict[str, Any]:
    """Search flights, update session search state, and return a Thinker payload."""
    origin, destination = extract_route(slots)
    travel_date = slot(slots, "date")
    sorting = slot(slots, "sorting")
    params_resolved: dict[str, Any] = {}
    params_needed: list[str] = []
    if origin:
        params_resolved["origin_city"] = airport_display_name(origin)
    else:
        params_needed.append("origin_city")
    if destination:
        params_resolved["dest_city"] = airport_display_name(destination)
    else:
        params_needed.append("dest_city")
    if travel_date:
        params_resolved["date"] = travel_date
    else:
        params_needed.append("date")

    if params_needed:
        return response_hint(
            reason="params_missing",
            action="req_params",
            params_needed=params_needed,
            params_resolved=params_resolved,
            response_text="Where are you flying from, where to, and when?",
            context="flight_search",
        )

    try:
        flights = await backend.search_flights(
            origin=origin,
            destination=destination,
            travel_date=travel_date,
            sorting=sorting,
        )
    except Exception as exc:
        logger.warning(f"flight_search backend failed: {exc}")
        flights = []
    if not flights:
        state.reset_search_and_booking()
        return response_hint(
            reason="tool_error",
            action="req_params",
            params_needed=["origin_city", "dest_city", "date"],
            params_resolved=params_resolved,
            error="No flights found for the requested route.",
            response_text="I could not find flights for that route. Would you like to try different cities?",
            context="flight_search",
        )

    state.search_context = {
        "origin_city": airport_display_name(origin),
        "dest_city": airport_display_name(destination),
        "origin_airport": origin,
        "dest_airport": destination,
        "date": travel_date,
        "sorting": sorting or "price",
    }
    state.search_results = flights
    state.reset_booking()
    options = ", ".join(format_flight_option(flight) for flight in flights[:5])
    response_text = f"I found {len(flights)} flights: {options}. Which flight would you like to book?"
    return tool_result(
        tool="flight_search",
        status="success",
        data={
            "flights": [user_facing_flight(flight) for flight in flights],
            "search_context": state.search_context,
        },
        response_text=response_text,
        context="flight_search",
    )
