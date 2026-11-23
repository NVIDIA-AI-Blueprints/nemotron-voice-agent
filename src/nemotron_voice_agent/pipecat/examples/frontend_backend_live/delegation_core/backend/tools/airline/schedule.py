# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Demo Air's timetable, and the sample statuses and on-time records of its flights.

The airline flies a nonstop route between every pair of its twelve airports, every day. Each departure has a fixed
flight number (the same on every day, as a real flight's is), a fixed local departure time and a block time, all made
from the airports by a fixed rule, so the same route and day always give the same flights:

* the pair of airports sets the block of numbers: pairs are ordered by airport code, and the n-th pair gets the numbers
  from ``100 + 10 * n``;
* a flight toward the east has an even number and a flight toward the west an odd one, and the departures of a day
  are numbered in the order of the clock: Denver to Boston is 210, 212, 214 and 216 from the morning on, and Boston
  to Denver is 211, 213, 215 and 217;
* how many departures a route has (three to five) follows from the size of its two airports.

Every flight, status and trend is marked ``source: "sample"``: the airline is made up, and none of it is a real
departure.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from functools import cache
from typing import Any

from .airports import Airport, all_airports, get_airport, great_circle_km, local_datetime
from .carriers import CODE, NAME

SAMPLE = "sample"
TREND_WINDOW_DAYS = 90
# The timetable runs this many days ahead of today.
WINDOW_DAYS = 60
# Boarding starts this long before a flight leaves.
BOARDING_MINUTES = 40
FIRST_NUMBER = 100
PAIR_STEP = 10
# Local departure times of a route's flights, in minutes after midnight, by how many flights it has.
_BANKS = {
    3: (410, 760, 1100),
    4: (375, 595, 850, 1125),
    5: (360, 530, 735, 940, 1160),
}
_DELAY_REASONS = (
    "a late-arriving aircraft",
    "air traffic control",
    "weather along the route",
    "a crew scheduling change",
)
_CANCEL_REASON = "an aircraft maintenance issue"


def _rng(*parts: str) -> random.Random:
    seed = int.from_bytes(hashlib.sha256("|".join(parts).encode("utf-8")).digest()[:8], "big")
    return random.Random(seed)


def block_minutes(distance_km: float) -> int:
    """Return a typical gate-to-gate time for a nonstop flight of ``distance_km``."""
    return round(35 + distance_km / 830 * 60)


@dataclass(frozen=True, slots=True)
class Departure:
    """One flight of the timetable, which leaves every day."""

    number: str
    origin: str
    destination: str
    # Local departure time at the origin, in minutes after midnight.
    minute: int
    # Gate-to-gate time in minutes.
    block: int
    aircraft: str

    @property
    def eastbound(self) -> bool:
        """Whether the flight goes toward the east: its number is even."""
        return int(self.number.split(" ")[1]) % 2 == 0


def frequency(origin: Airport, destination: Airport) -> int:
    """Return how many flights a day the airline has on a route in each direction (three to five)."""
    return origin.size + destination.size - 1


def _aircraft(pair: int, distance_km: float) -> str:
    choices = (
        ("Airbus A220-300", "Embraer 175")
        if distance_km < 1100
        else ("Airbus A320neo", "Boeing 737-800", "Airbus A321neo")
        if distance_km < 3000
        else ("Airbus A321neo", "Boeing 787-9")
    )
    return choices[pair % len(choices)]


def _block(distance_km: float, eastbound: bool) -> int:
    """Return the block time rounded to five minutes: a flight toward the east is a little shorter, with the wind."""
    return 5 * round(block_minutes(distance_km) * (0.96 if eastbound else 1.04) / 5)


@cache
def timetable() -> dict[str, Departure]:
    """Return every flight of the airline by its number ("DA 214")."""
    airports = sorted(all_airports().values(), key=lambda airport: airport.iata)
    table: dict[str, Departure] = {}
    pair = 0
    for position, first in enumerate(airports):
        for second in airports[position + 1 :]:
            base = FIRST_NUMBER + PAIR_STEP * pair
            for origin, destination in ((first, second), (second, first)):
                eastbound = destination.lon > origin.lon
                digest = hashlib.sha256(f"{origin.iata}>{destination.iata}".encode()).digest()
                shift = (int.from_bytes(digest[:2], "big") % 9 - 4) * 5
                distance = great_circle_km(origin, destination)
                aircraft = _aircraft(pair, distance)
                block = _block(distance, eastbound)
                for index, minute in enumerate(_BANKS[frequency(origin, destination)]):
                    number = f"{CODE} {base + 2 * index + (0 if eastbound else 1)}"
                    table[number] = Departure(number, origin.iata, destination.iata, minute + shift, block, aircraft)
            pair += 1
    return table


@cache
def _by_route() -> dict[tuple[str, str], tuple[Departure, ...]]:
    routes: dict[tuple[str, str], list[Departure]] = {}
    for departure in timetable().values():
        routes.setdefault((departure.origin, departure.destination), []).append(departure)
    return {key: tuple(sorted(items, key=lambda item: item.minute)) for key, items in routes.items()}


def routes() -> list[tuple[str, str]]:
    """Return every route as an (origin, destination) pair of airport codes, in order."""
    return sorted(_by_route())


def _terminal(airport: Airport) -> str:
    """Return the terminal that the airline uses at an airport: one letter, the same for every flight and day."""
    return "ABCDE"[int.from_bytes(hashlib.sha256(f"terminal|{airport.iata}".encode()).digest()[:1], "big") % 5]


def _gate(airport: Airport, flight_number: str, day: date) -> str:
    number = _rng("gate", airport.iata, flight_number, day.isoformat()).randint(1, 38)
    return f"{_terminal(airport)}{number}"


def _endpoint(airport: Airport) -> dict[str, Any]:
    return {**airport.ref(), "terminal": _terminal(airport)}


def _leaves(departure: Departure, origin: Airport, day: date) -> tuple[datetime, datetime]:
    """Return the local departure at the origin and the local arrival at the destination of a flight on a day."""
    leave = local_datetime(origin, day, time(departure.minute // 60, departure.minute % 60))
    destination = get_airport(departure.destination)
    arrive = leave + timedelta(minutes=departure.block)
    return leave, arrive.astimezone(destination.zone) if destination else arrive


def sample_flights(origin: Airport, destination: Airport, day: date) -> list[dict[str, Any]]:
    """Return the airline's nonstop flights for a route and day, earliest first (fares not included)."""
    flights = []
    for departure in _by_route().get((origin.iata, destination.iata), ()):
        leave, arrive = _leaves(departure, origin, day)
        flights.append(
            {
                "airline": {"code": CODE, "name": NAME},
                "flight_number": departure.number,
                "origin": _endpoint(origin),
                "destination": _endpoint(destination),
                "depart_local": leave.isoformat(),
                "arrive_local": arrive.isoformat(),
                "duration_min": departure.block,
                "aircraft": departure.aircraft,
                "status": "scheduled",
                "source": SAMPLE,
            }
        )
    return flights


def scheduled(flight_number: str, day: date) -> dict[str, str] | None:
    """Return a timetable flight's route and local times on a day, or None for a number the airline does not fly."""
    departure = timetable().get(flight_number)
    origin = get_airport(departure.origin) if departure else None
    if departure is None or origin is None:
        return None
    leave, arrive = _leaves(departure, origin, day)
    return {
        "origin": departure.origin,
        "destination": departure.destination,
        "depart_local": leave.isoformat(),
        "arrive_local": arrive.isoformat(),
    }


def move_to_day(schedule: dict[str, Any], day: date) -> dict[str, Any]:
    """Return a known schedule on ``day``: the same local departure time and flight time, on another date."""
    depart = datetime.fromisoformat(schedule["depart_local"])
    minutes = round((datetime.fromisoformat(schedule["arrive_local"]) - depart).total_seconds() / 60)
    destination = get_airport(schedule["destination"])
    moved = depart.replace(year=day.year, month=day.month, day=day.day)
    arrive = moved + timedelta(minutes=minutes)
    same_day = day == depart.date()
    return {
        **schedule,
        "depart_local": moved.isoformat(),
        "arrive_local": (arrive.astimezone(destination.zone) if destination else arrive).isoformat(),
        "delay_min": schedule.get("delay_min") if same_day else None,
        "cancelled": bool(schedule.get("cancelled")) and same_day,
    }


def sample_delay_minutes(flight_number: str, day: date) -> int:
    """Return a stable sample delay: most flights are on time, about one in eight runs late."""
    rng = _rng("delay", flight_number.replace(" ", "").upper(), day.isoformat())
    return rng.choice((20, 35, 50)) if rng.random() < 0.125 else 0


def delay_reason(flight_number: str, day: date) -> str:
    """Return a stable sample reason for a delay."""
    return _rng("reason", flight_number.replace(" ", "").upper(), day.isoformat()).choice(_DELAY_REASONS)


def _movement(
    airport: str | None, scheduled_at: datetime | None, delay: int, done: bool, flight_number: str, day: date
) -> dict[str, Any]:
    shifted = scheduled_at + timedelta(minutes=delay) if scheduled_at else None
    place = get_airport(airport)
    return {
        "airport": airport,
        "scheduled": scheduled_at.isoformat() if scheduled_at else None,
        "estimated": shifted.isoformat() if shifted and delay and not done else None,
        "actual": shifted.isoformat() if shifted and done else None,
        "terminal": _terminal(place) if place and scheduled_at else None,
        "gate": _gate(place, flight_number, day) if place and scheduled_at else None,
    }


def sample_status(
    flight_number: str,
    day: date | None,
    *,
    schedule: dict[str, Any] | None,
    now: datetime,
) -> dict[str, Any]:
    """Return a flight's status from its known schedule (route, local times, optional ``delay_min`` and ``cancelled``).

    The state is ``unknown`` when there is no schedule, and ``cancelled`` for a cancelled flight. Otherwise a delay of
    15 minutes or more before departure is ``delayed``; boarding starts 40 minutes before the flight leaves (its
    delayed time), and once it has left it is ``en_route``, then ``landed``. ``reason`` says why a flight is late or
    cancelled.
    """
    try:
        depart = datetime.fromisoformat(schedule["depart_local"]) if schedule else None
        arrive = datetime.fromisoformat(schedule["arrive_local"]) if schedule else None
    except (KeyError, ValueError):
        depart = arrive = None
    origin, destination = (schedule or {}).get("origin"), (schedule or {}).get("destination")
    if depart is None or arrive is None:
        blank = {"airport": None, "scheduled": None, "estimated": None, "actual": None, "terminal": None, "gate": None}
        return {
            "flight_number": flight_number,
            "date": day.isoformat() if day else None,
            "state": "unknown",
            "departure": {**blank, "airport": origin},
            "arrival": {**blank, "airport": destination},
            "delay_min": None,
            "reason": None,
            "source": SAMPLE,
            "trend": None,
        }
    if (schedule or {}).get("cancelled"):
        movement = {"estimated": None, "actual": None, "gate": None}
        return {
            "flight_number": flight_number,
            "date": depart.date().isoformat(),
            "state": "cancelled",
            "departure": {**_movement(origin, depart, 0, False, flight_number, depart.date()), **movement},
            "arrival": {**_movement(destination, arrive, 0, False, flight_number, arrive.date()), **movement},
            "delay_min": None,
            "reason": _CANCEL_REASON,
            "source": SAMPLE,
            "trend": None,
        }
    given = (schedule or {}).get("delay_min")
    delay = sample_delay_minutes(flight_number, depart.date()) if given is None else max(0, given)
    leaves = depart + timedelta(minutes=delay)
    departed = now >= leaves
    landed = now >= arrive + timedelta(minutes=delay)
    boarding = not departed and now >= leaves - timedelta(minutes=BOARDING_MINUTES)
    state = (
        "landed"
        if landed
        else "en_route"
        if departed
        else "boarding"
        if boarding
        else "delayed"
        if delay >= 15
        else "scheduled"
    )
    return {
        "flight_number": flight_number,
        "date": depart.date().isoformat(),
        "state": state,
        "departure": _movement(origin, depart, delay, departed, flight_number, depart.date()),
        "arrival": _movement(destination, arrive, delay, landed, flight_number, arrive.date()),
        "delay_min": delay,
        "reason": delay_reason(flight_number, depart.date()) if delay >= 15 else None,
        "source": SAMPLE,
        "trend": None,
    }


def sample_trend(flight_number: str) -> dict[str, Any]:
    """Return a stable sample on-time record for about the last 90 days; the three shares add up to 100."""
    rng = _rng("trend", flight_number.replace(" ", "").upper())
    on_time = rng.randint(58, 93)
    very_late = rng.randint(1, max(1, (100 - on_time) // 3))
    return {
        "window_days": TREND_WINDOW_DAYS,
        "on_time_pct": on_time,
        "late_pct": 100 - on_time - very_late,
        "very_late_pct": very_late,
        "avg_delay_min": rng.randint(9, 38),
        "flights_counted": rng.randint(58, TREND_WINDOW_DAYS),
    }
