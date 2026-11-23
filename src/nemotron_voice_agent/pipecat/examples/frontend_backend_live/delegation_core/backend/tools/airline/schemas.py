# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Function schemas for the airline desk tools the backend may call (Responses function-tool format).

The desk is Demo Air, a made-up airline that flies between twelve U.S. cities, on dates up to 60 days ahead.

The ``date`` parameters of ``flights_search`` and ``get_flight_status`` end without today's date: the tool source adds
it when it lists the tools (``AirlineTools.tools``), so the model can work out "tomorrow" from its own tool list.
"""

from typing import Any

CABINS = ["economy", "premium_economy", "business", "first"]
# The tools whose ``date`` description gets today's date appended, by tool name.
DATED_TOOLS = ("flights_search", "get_flight_status")

_CONFIRM_RULE = (
    "Changes only after a read-back: call without confirmed to get the read-back (status needs_confirmation, nothing "
    "changes), tell the caller, and call again with the same arguments and confirmed=true only after they clearly say "
    "yes."
)
_CABIN = {
    "type": "string",
    "enum": CABINS,
    "description": "Cabin: economy, premium_economy, business or first.",
}
_BOOKING_ID = {
    "type": "string",
    "description": (
        "Six-character booking code such as K7Q2ZP. Optional: without it the tool uses the booking just made or "
        "changed, or the caller's only active booking, and otherwise asks which."
    ),
}
_CONFIRMED = {
    "type": "boolean",
    "description": "True only after the caller said yes to the read-back of this exact change. Omit it to read back.",
}


def _tool(name: str, description: str, properties: dict[str, Any], required: list[str]) -> dict[str, Any]:
    return {
        "type": "function",
        "name": name,
        "description": description,
        "parameters": {
            "type": "object",
            "properties": properties,
            "required": required,
            "additionalProperties": False,
        },
        "strict": False,
    }


AIRLINE_TOOLS = [
    _tool(
        "flights_search",
        (
            "Search up to five nonstop Demo Air flights between two cities on one date, each with a demo fare. Pass "
            "origin and destination as the caller said them (a city, an airport name or an IATA code); the tool "
            "resolves them. Demo Air flies between only twelve U.S. cities: for any other place the result is "
            "not_served and names the closest served city, so offer that and never invent a flight. time_of_day, "
            "depart_after and depart_before narrow the day's flights before the best five are kept; when nothing "
            "matches, the result says so and lists no other flights. Each flight has an option number that "
            "new_booking and modify_existing_booking accept. Flights come from a sample timetable, not real "
            "departures; dates run up to 60 days ahead."
        ),
        {
            "origin": {"type": "string", "description": "Departure city, airport name or IATA code."},
            "destination": {"type": "string", "description": "Arrival city, airport name or IATA code."},
            "date": {"type": "string", "description": "Travel date as YYYY-MM-DD."},
            "cabin": {**_CABIN, "description": "Cabin to price. Defaults to economy."},
            "sort": {
                "type": "string",
                "enum": ["price", "duration", "departure_time"],
                "description": "Order of the results. Defaults to departure_time.",
            },
            "time_of_day": {
                "type": "string",
                "description": (
                    "Part of the day on the searched date: morning 05:00-11:59, afternoon 12:00-17:59, evening "
                    "18:00-23:59, overnight (red-eye) 21:00-23:59."
                ),
            },
            "depart_after": {"type": "string", "description": "Earliest departure time, 24-hour HH:MM."},
            "depart_before": {
                "type": "string",
                "description": "Departures before this time, 24-hour HH:MM; a flight at exactly this time is left out.",
            },
        },
        ["origin", "destination", "date"],
    ),
    _tool(
        "where_we_fly",
        (
            "The cities Demo Air serves, with their airport codes, and the airline's routes. Use it when the caller "
            "asks where Demo Air flies or which cities are covered. Every pair of the cities has nonstop flights."
        ),
        {},
        [],
    ),
    _tool(
        "get_flight_status",
        (
            "Status of one Demo Air flight on one day: scheduled, boarding, delayed, en route, landed or cancelled, "
            "with scheduled, estimated and actual times, terminal and gate, and the reason for a delay or a "
            "cancellation. Uses a flight from the caller's bookings or the latest search when it knows one. Sample "
            "data, not a real departure."
        ),
        {
            "flight_number": {
                "type": "string",
                "description": "Demo Air flight number, such as DA 214 or just 214.",
            },
            "date": {
                "type": "string",
                "description": "Flight date as YYYY-MM-DD. Defaults to the flight's date in the trips, else today.",
            },
        },
        ["flight_number"],
    ),
    _tool(
        "get_ontime_trend",
        (
            "How often a flight was on time over about the last 90 days: on_time_pct, late_pct, very_late_pct, "
            "avg_delay_min when it is late, and flights_counted. Sample data."
        ),
        {"flight_number": {"type": "string", "description": "Demo Air flight number, such as DA 214 or just 214."}},
        ["flight_number"],
    ),
    _tool(
        "sign_in",
        (
            "Sign in to a demo account by user ID for this session, or create a demo account in the caller's own name "
            "and sign in to it. No password. The sample accounts are jane_doe_001 (two trips, one of them delayed) and "
            "marcus_lee_002 (a trip today whose flight was cancelled). Give either user_id or new_account_name."
        ),
        {
            "user_id": {"type": "string", "description": "User ID such as jane_doe_001."},
            "new_account_name": {
                "type": "string",
                "description": "The caller's full name, to create a new demo account.",
            },
        },
        [],
    ),
    _tool(
        "get_existing_bookings",
        (
            "The signed-in caller's trips with booking codes, routes, dates, cabins, seats and meals, including "
            "cancelled ones. Pass user_id to sign in first."
        ),
        {"user_id": {"type": "string", "description": "User ID, to sign in before listing."}},
        [],
    ),
    _tool(
        "new_booking",
        (
            "Book one flight from the latest flights_search as a demo booking for the signed-in caller. Needs a "
            "search first. " + _CONFIRM_RULE
        ),
        {
            "flight": {
                "type": "string",
                "description": "Which flight of the latest search: its option number, flight number or departure time.",
            },
            "passengers": {
                "type": "string",
                "description": "Passenger names, comma separated. Defaults to the account holder.",
            },
            "cabin": {**_CABIN, "description": "Cabin to book. Defaults to the searched cabin."},
            "confirmed": _CONFIRMED,
        },
        ["flight"],
    ),
    _tool(
        "modify_existing_booking",
        (
            "Move a booking to a flight from the latest flights_search and state the fare difference. Needs a "
            "search first. A seat on the old flight is released; the meal stays. " + _CONFIRM_RULE
        ),
        {
            "flight": {
                "type": "string",
                "description": "Which flight of the latest search: its option number, flight number or departure time.",
            },
            "booking_id": _BOOKING_ID,
            "confirmed": _CONFIRMED,
        },
        ["flight"],
    ),
    _tool(
        "cancel_booking",
        "Cancel a booking and state its refund rule. " + _CONFIRM_RULE,
        {"booking_id": _BOOKING_ID, "confirmed": _CONFIRMED},
        [],
    ),
    _tool(
        "select_seat_and_meal",
        ("Choose a seat and/or a special meal on a booking. Give seat, meal or both. " + _CONFIRM_RULE),
        {
            "booking_id": _BOOKING_ID,
            "seat": {
                "type": "string",
                "description": "A seat such as 18A, or window, aisle or exit row.",
            },
            "meal": {
                "type": "string",
                "description": "A meal in words (vegetarian, vegan, kosher, gluten-free) or a code such as VGML.",
            },
            "flight_number": {
                "type": "string",
                "description": "One flight of a multi-flight trip. Defaults to every flight of the booking.",
            },
            "confirmed": _CONFIRMED,
        },
        [],
    ),
    _tool(
        "upgrade_class",
        (
            "Move a booking to another cabin and state the demo price difference. Seats are released because they "
            "belong to a cabin. " + _CONFIRM_RULE
        ),
        {
            "cabin": {**_CABIN, "description": "The cabin to move to."},
            "booking_id": _BOOKING_ID,
            "confirmed": _CONFIRMED,
        },
        ["cabin"],
    ),
]
