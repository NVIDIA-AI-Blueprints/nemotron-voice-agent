# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The Demo Air desk's eleven tools, as a server tool source: flights, status, sign-in and demo bookings.

Read tools: ``where_we_fly``, ``flights_search``, ``get_flight_status``, ``get_ontime_trend``, ``sign_in``,
``get_existing_bookings``.
Change tools: ``new_booking``, ``modify_existing_booking``, ``cancel_booking``, ``select_seat_and_meal``,
``upgrade_class``.

Every change tool reads the change back first and stores it as the pending change. It runs only when a later call
names the same change with ``confirmed: true``; the model cannot skip the read-back. The server cannot hear the
caller, so the model's instructions must still say that ``confirmed`` follows a clear yes.

Every result is compact JSON the model can speak from. Its ``status`` is one of ``ok``, ``confirmed``,
``needs_confirmation``, ``needs_sign_in``, ``needs_search``, ``needs_booking_id``, ``needs_clarification``,
``not_served``, ``no_results``, ``no_data``, ``not_found`` or ``error``. A result that is not ``ok`` or
``confirmed`` carries a ``reason`` and a ``question`` for the caller or an ``error``. The first result of a search, a
status, an on-time record, a trip list or a change carries ``data_notice`` once per session. All data is sample data
and nothing is charged.
"""

from __future__ import annotations

import json
import random
import re
from collections.abc import Callable, Mapping
from copy import deepcopy
from datetime import UTC, date, datetime, timedelta
from typing import Any, ClassVar

from loguru import logger

from .airports import Airport, Place, all_airports, get_airport, nearest_served, resolve_airport
from .carriers import HOME_ZONE, airline_name, is_demo_air, normalize_flight_number, spoken_flight
from .carriers import NAME as AIRLINE_NAME
from .fares import CABIN_LABELS, FARE_LABEL, cabin_rank, demo_fare, fare_payload, normalize_cabin
from .filters import FlightFilters, normalize_time_arguments
from .schedule import (
    SAMPLE,
    TREND_WINDOW_DAYS,
    WINDOW_DAYS,
    move_to_day,
    sample_flights,
    sample_status,
    sample_trend,
    scheduled,
)
from .schemas import AIRLINE_TOOLS, DATED_TOOLS
from .seats import (
    CABIN_ROWS,
    EXIT_ROWS,
    NO_MEAL,
    find_seat,
    meal_label,
    parse_meal,
    parse_seat,
    seat_free,
    seat_in_cabin,
    seat_kind,
    seat_type,
)
from .speech import (
    and_list,
    spoken_amount,
    spoken_code,
    spoken_date,
    spoken_day,
    spoken_duration,
    spoken_time,
    spoken_user_id,
)
from .state import DATA_NOTICE, DeskState, FlightSearch, PendingChange
from .store import DEFAULT_USER_ID, DEFAULT_USER_SPOKEN, Booking, DeskStore, Leg, booking_key, user_key

# The schemas are the one list of tools; each name is a method of AirlineTools.
TOOL_NAMES = frozenset(tool["name"] for tool in AIRLINE_TOOLS)
MAX_RESULTS = 5
MAX_PASSENGERS = 6
MAX_DAYS_AHEAD = WINDOW_DAYS
# A place that is no farther than this from a served city is answered with that city; a farther one is not.
NEAR_MILES = 450
_ORDINALS = {"first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5, "last": -1}
_WRONG_ID_ASK = "Could you say it again, one character at a time?"


class Reply(Exception):
    """A finished result raised from deep in a tool, so a helper can answer for the whole call."""

    def __init__(self, body: dict[str, Any]):
        """Hold the result the call returns."""
        super().__init__(body.get("error") or body.get("question") or body.get("status"))
        self.body = body


def _ask(reason: str, question: str, *, status: str = "needs_clarification", **extra: Any) -> Reply:
    """Return the reply that asks the caller for what is missing."""
    return Reply({"status": status, "reason": reason, "question": question, **extra})


def _fail(reason: str, error: str, **extra: Any) -> Reply:
    """Return the reply that tells the model its call cannot run."""
    return Reply({"status": "error", "reason": reason, "error": error, **extra})


def _is_yes(value: Any) -> bool:
    """Whether ``confirmed`` is true: the boolean, or the word "true"."""
    return value is True or (isinstance(value, str) and value.strip().lower() == "true")


def _clock_sort_key(flight: dict[str, Any]) -> str:
    return flight["depart_local"]


class AirlineTools:
    """One session's airline desk, in memory: a server tool source made fresh for each session."""

    label: ClassVar[str] = "airline"

    def __init__(self, *, clock: Callable[[], datetime] | None = None, seed: int | None = None):
        """Start the desk with the sample account and no search.

        ``clock`` gives the current time (an aware ``datetime``; default now, UTC) and ``seed`` makes the booking codes
        repeatable.
        """
        self._clock = clock or (lambda: datetime.now(UTC))
        self.store = DeskStore(now=self.now, rng=random.Random(seed))
        self.state = DeskState()
        self.by_call: dict[str, str] = {}

    @classmethod
    def from_settings(cls, block: Mapping[str, Any]) -> AirlineTools:
        """Return a fresh desk from its ``backend.tools.airline`` block, which must be empty."""
        if not isinstance(block, Mapping) or block:
            raise ValueError(f"backend.tools.{cls.label} takes no settings; got {block!r}")
        return cls()

    # ---------------------------------------------------------------------------------- the clock

    def now(self) -> datetime:
        """Return the current time, aware: a clock without a time zone is read as UTC."""
        moment = self._clock()
        return moment if moment.tzinfo else moment.replace(tzinfo=UTC)

    @property
    def today(self) -> date:
        """The current day: the date in the airline's home zone (``carriers.HOME_ZONE``)."""
        return self.now().astimezone(HOME_ZONE).date()

    # ---------------------------------------------------------------------------- the tool source

    def tools(self) -> list[dict]:
        """Return a copy of the function tools in the Responses shape, with today's date in the date descriptions."""
        tools = deepcopy(AIRLINE_TOOLS)
        today = self.today
        for tool in tools:
            if tool["name"] in DATED_TOOLS:
                date_parameter = tool["parameters"]["properties"]["date"]
                date_parameter["description"] += f" Today is {today.isoformat()} ({today:%A})."
        return tools

    async def execute(self, call_id: str, name: str, arguments: str) -> str:
        """Run one tool call and return its result as JSON; a repeated ``call_id`` returns the first result."""
        # A repeated call_id must not change the desk a second time.
        if call_id in self.by_call:
            return self.by_call[call_id]
        try:
            if name not in TOOL_NAMES:
                raise _fail("unknown_tool", f"Unknown tool: {name}")
            result = getattr(self, name)(self._arguments(arguments))
        except Reply as reply:
            result = reply.body
        except (ValueError, TypeError, KeyError, AttributeError) as exc:
            result = {"status": "error", "reason": "bad_arguments", "error": str(exc)}
        except Exception as exc:  # a bug in a tool must end as an error result, not end the delegation
            logger.exception(f"The airline tool {name} failed")
            result = {"status": "error", "reason": "tool_failed", "error": f"{type(exc).__name__}: {exc}"}
        self.by_call[call_id] = json.dumps(result, ensure_ascii=False)
        return self.by_call[call_id]

    @staticmethod
    def _arguments(arguments: str) -> dict[str, Any]:
        try:
            given = json.loads(arguments or "{}")
        except ValueError:
            raise _fail("bad_arguments", "The arguments are not JSON") from None
        if not isinstance(given, dict):
            raise _fail("bad_arguments", "The arguments must be a JSON object")
        return given

    # ------------------------------------------------------------------------------------- helpers

    def _notice(self) -> dict[str, str]:
        """Return ``data_notice`` the first time it is asked for in this session, and nothing after."""
        if self.state.notice_given:
            return {}
        self.state.notice_given = True
        return {"data_notice": DATA_NOTICE}

    def _place(self, text: object, field: str) -> Airport:
        """Return the served airport that ``text`` names, or raise the reply that settles it."""
        match = resolve_airport(str(text or ""))
        if match.airport is not None:
            return match.airport
        if match.unserved is not None:
            raise self._not_served(match.unserved, field)
        said = str(text or "").strip()
        if not said:
            raise _ask(
                "missing_place", f"Where would you like to fly {'from' if field == 'origin' else 'to'}?", field=field
            )
        raise _ask(
            "unknown_airport",
            f"I couldn't find {said} among the cities Demo Air flies to. Which city is that? "
            "I can also tell you where we fly.",
            field=field,
        )

    @staticmethod
    def _served_list() -> list[dict[str, str]]:
        return [
            {"code": item.iata, "city": item.city, "state": item.state, "airport": item.name}
            for item in sorted(all_airports().values(), key=lambda airport: airport.city)
        ]

    def _not_served(self, place: Place, field: str) -> Reply:
        """Return the reply for a place that Demo Air does not serve: the closest city it does, when one is near."""
        nearest, miles = nearest_served(place)
        body: dict[str, Any] = {
            "status": "not_served",
            "reason": "not_served",
            "field": field,
            "place": place.label,
            "nearest": {**nearest.ref(), "state": nearest.state},
            "distance_miles": miles,
        }
        if miles <= NEAR_MILES:
            body["message"] = (
                f"Demo Air doesn't fly to {place.label}. The closest city we serve is {nearest.city}, about {miles} "
                "miles away."
            )
            body["question"] = f"Would you like to fly {'from' if field == 'origin' else 'to'} {nearest.city} instead?"
        else:
            cities = and_list([item.city for item in sorted(all_airports().values(), key=lambda a: a.city)])
            body["message"] = (
                f"Demo Air doesn't fly to {place.label}, or anywhere near it. We fly between twelve U.S. cities: "
                f"{cities}."
            )
            body["question"] = "Would you like to travel between two of those cities?"
            body["served"] = self._served_list()
        return Reply(body)

    def _day(self, raw: object, *, ahead: bool = True, what: str = "the flight") -> date:
        """Return the date ``raw`` names, or raise the reply for a missing, unreadable, past or too distant one."""
        try:
            day = date.fromisoformat(str(raw or "").strip()[:10])
        except ValueError:
            raise _ask("missing_date", f"What date is {what}? Use a date as YYYY-MM-DD.", field="date") from None
        today = self.today
        last = today + timedelta(days=MAX_DAYS_AHEAD)
        if ahead and day < today:
            raise _fail(
                "past_date",
                f"{spoken_day(day, today)} has already passed. Use a future date; today is {today.isoformat()}.",
            )
        if day > last:
            raise _fail(
                "too_far_ahead",
                f"Demo Air's timetable runs {MAX_DAYS_AHEAD} days ahead. Use a date on or before {last.isoformat()}.",
            )
        return day

    @staticmethod
    def _cabin(raw: object, default: str) -> str:
        """Return the cabin ``raw`` names, ``default`` for none, or raise the question for an unknown one."""
        cabin = normalize_cabin(raw, default=None)
        if cabin is not None:
            return cabin
        if str(raw or "").strip():
            raise _ask(
                "unknown_cabin",
                "Which cabin would you like: economy, premium economy, business or first?",
                field="cabin",
            )
        return default

    def _priced(self, flight: dict[str, Any], cabin: str) -> dict[str, Any]:
        """Return ``flight`` with its demo fare in ``cabin``."""
        origin, destination = get_airport(flight["origin"]["code"]), get_airport(flight["destination"]["code"])
        amount = 0.0
        if origin and destination:
            amount = demo_fare(
                origin=origin,
                destination=destination,
                departure=datetime.fromisoformat(flight["depart_local"]),
                cabin=cabin,
                flight_number=flight["flight_number"],
                today=self.today,
            )
        return {**flight, "fare": fare_payload(amount, cabin)}

    def _flight_view(self, option: int, flight: dict[str, Any]) -> dict[str, Any]:
        """Return one searched flight with the words the voice says."""
        depart, arrive = flight["depart_local"], flight.get("arrive_local")
        fare = flight["fare"]
        return {
            "option": option,
            "flight": flight["flight_number"],
            "flight_spoken": self._flight_phrase(flight),
            "airline": flight["airline"]["name"],
            "airline_code": flight["airline"]["code"],
            "origin": flight["origin"]["code"],
            "destination": flight["destination"]["code"],
            "depart_local": depart,
            "depart_spoken": spoken_time(depart),
            "arrive_local": arrive,
            "arrive_spoken": spoken_time(arrive),
            "arrives_next_day": bool(arrive and arrive[:10] > depart[:10]),
            "duration_min": flight.get("duration_min"),
            "duration_spoken": spoken_duration(flight.get("duration_min")),
            "aircraft": flight.get("aircraft"),
            "cabin": fare["cabin"],
            "price_usd": fare["amount"],
            "price_spoken": spoken_amount(fare["amount"]),
            "fare_label": FARE_LABEL,
        }

    def _flight_phrase(self, flight: Mapping[str, Any]) -> str:
        """Return "Demo Air flight two fourteen"."""
        return spoken_flight(str(flight["flight_number"]), (flight.get("airline") or {}).get("name"))

    def _route_phrase(self, flight: Mapping[str, Any]) -> str:
        return f"{flight['origin']['city']} to {flight['destination']['city']}"

    def _trip_phrase(self, booking: Booking) -> str:
        """Return "Boston to Denver on Thursday, November 5th", or the round trip's "and back"."""
        legs = [leg.view() for leg in booking.legs]
        first, last = legs[0], legs[-1]
        today = self.today
        note = ", cancelled by Demo Air" if any(leg.cancelled for leg in booking.legs) else ""
        if len(legs) == 2 and first["origin"]["code"] == last["destination"]["code"]:
            leaving = spoken_day(first["depart_local"], today)
            return f"{first['origin']['city']} to {first['destination']['city']} and back, leaving {leaving}{note}"
        return (
            f"{first['origin']['city']} to {last['destination']['city']} on "
            f"{spoken_day(first['depart_local'], today)}{note}"
        )

    def _booking_view(self, booking: Booking) -> dict[str, Any]:
        """Return a booking with the words the voice says."""
        legs = []
        for leg in booking.legs:
            view = leg.view()
            legs.append(
                {
                    "flight": leg.flight_number,
                    "airline": leg.airline_name,
                    "origin": leg.origin,
                    "destination": leg.destination,
                    "route_spoken": self._route_phrase(view),
                    "depart_local": leg.depart_local,
                    "arrive_local": leg.arrive_local,
                    "date_spoken": spoken_date(leg.depart_local, self.today),
                    "depart_spoken": spoken_time(leg.depart_local),
                    "seat": leg.seat,
                    "meal": leg.meal,
                    "meal_spoken": meal_label(leg.meal) if leg.meal else None,
                    **({"cancelled": True} if leg.cancelled else {}),
                    **({"delay_min": leg.delay_min} if leg.delay_min else {}),
                }
            )
        return {
            "booking_id": booking.booking_id,
            "booking_id_spoken": spoken_code(booking.booking_id),
            "status": booking.status,
            "cabin": booking.cabin,
            "cabin_spoken": CABIN_LABELS[booking.cabin],
            "passengers": list(booking.passengers),
            "trip_spoken": self._trip_phrase(booking),
            "flights": legs,
            "total_usd": booking.total,
            "total_spoken": spoken_amount(booking.total),
            **({"needs_rebooking": True} if any(leg.cancelled for leg in booking.legs) else {}),
        }

    def _need_sign_in(self, question: str | None = None, **extra: Any) -> Reply:
        """Return the reply that asks the caller to sign in, naming the sample account."""
        return _ask(
            "sign_in_required",
            question
            or (
                "Please sign in first: tell me your user ID, or say your name and I'll create a demo account for you. "
                f"You can also use the sample account {DEFAULT_USER_SPOKEN}."
            ),
            status="needs_sign_in",
            sample_account={"user_id": DEFAULT_USER_ID, "spoken": DEFAULT_USER_SPOKEN},
            **extra,
        )

    def _signed_in(self) -> str:
        """Return the signed-in user id, or raise the reply that asks the caller to sign in."""
        if not self.state.account_id:
            raise self._need_sign_in()
        return self.state.account_id

    def _read_back(
        self, tool: str, key: tuple[str, ...], change: str, summary: str, details: dict[str, Any], confirmed: bool
    ) -> None:
        """Return when this exact change was read back and the caller said yes; else raise the read-back.

        The read-back is stored as the pending change. A call with ``confirmed`` that matches no pending change of the
        same tool is read back again, and its result says that ``confirmed`` was not taken.
        """
        pending = self.state.pending
        if confirmed and pending is not None and pending.tool == tool and pending.key == key:
            self.state.pending = None
            return
        self.state.pending = PendingChange(tool=tool, key=key)
        body: dict[str, Any] = {
            "status": "needs_confirmation",
            "pending": True,
            "change": change,
            "confirmation_summary": f"{summary} Shall I go ahead?",
            **details,
            "next_step": (
                "Read the summary to the caller. Call this tool again with the same arguments and confirmed=true "
                "only after they say yes."
            ),
            **self._notice(),
        }
        if confirmed:
            body["note"] = "confirmed was not taken: this exact change had not been read back yet, so nothing changed."
        raise Reply(body)

    def _committed(self, change: str, booking: Booking, summary: str, **extra: Any) -> dict[str, Any]:
        """Return the result of a change that ran, and mark the booking as the one just changed."""
        self.state.last_change = {"kind": change, "booking_id": booking.booking_id}
        self.store.touch(booking)
        return {
            "status": "confirmed",
            "change": change,
            "booking": self._booking_view(booking),
            "summary": summary,
            **extra,
            **self._notice(),
        }

    def _resolve_booking(self, raw_id: object) -> Booking:
        """Return the booking the caller means, or raise the question; a known code signs in as its owner."""
        state, store = self.state, self.store
        if raw_id:
            code = booking_key(raw_id)
            booking = store.booking(code)
            if booking is None:
                raise _ask(
                    "unknown_booking",
                    f"I couldn't find that booking code. {_WRONG_ID_ASK}",
                    status="not_found",
                    field="booking_id",
                )
            if state.account_id and booking.user_id != state.account_id:
                raise _fail(
                    "other_account",
                    "That booking is on a different account. Sign in with that account's user ID to change it.",
                )
            state.account_id = booking.user_id
            return booking
        if not state.account_id:
            raise self._need_sign_in(
                f"Which booking? Tell me your user ID or the booking code. The sample account is {DEFAULT_USER_SPOKEN}."
            )
        active = store.bookings_of(state.account_id, active_only=True)
        recent = (state.last_change or {}).get("booking_id")
        # Right after a booking or a change, "a window seat" or "upgrade it" means that booking.
        for booking in active:
            if booking.booking_id == recent:
                return booking
        if len(active) == 1:
            return active[0]
        if not active:
            raise Reply(
                {
                    "status": "no_data",
                    "reason": "no_bookings",
                    "message": "The caller has no active bookings right now.",
                }
            )
        options = "; or ".join(f"{self._trip_phrase(item)}, code {item.booking_id}" for item in active[:3])
        raise _ask(
            "which_booking",
            f"Which booking: {options}?",
            status="needs_booking_id",
            field="booking_id",
            bookings=[self._booking_view(item) for item in active[:5]],
        )

    @staticmethod
    def _active(booking: Booking) -> Booking:
        if not booking.active:
            raise _fail("booking_cancelled", f"Booking {booking.booking_id} is already cancelled.")
        return booking

    def _select_flight(self, raw: object) -> int:
        """Return the index in the latest search of the flight ``raw`` names, or raise the reply that settles it."""
        search = self.state.search
        if search is None or not search.results:
            raise _ask(
                "search_first",
                "Let's find a flight first. Where are you flying from and to, and on what date?",
                status="needs_search",
                field="origin,destination,date",
            )
        index = _chosen_index(search.results, raw)
        if index is None:
            raise _ask(
                "which_flight",
                f"Which flight would you like: the first, the second, or another of the {len(search.results)} "
                "on the screen?",
                field="flight",
                options=[self._flight_view(i + 1, item) for i, item in enumerate(search.results)],
            )
        return index

    # ----------------------------------------------------------------------------------- read tools

    def flights_search(self, args: dict[str, Any]) -> dict[str, Any]:
        """Search up to five nonstop flights for a route and day, each with a demo fare.

        ``time_of_day``, ``depart_after`` and ``depart_before`` narrow the day's flights first; the rest are sorted and
        the first five kept. A place that Demo Air does not serve is answered with the closest city that it does.
        """
        origin, destination = (
            self._place(args.get("origin"), "origin"),
            self._place(args.get("destination"), "destination"),
        )
        if origin.iata == destination.iata:
            raise _ask("same_airport", f"Where would you like to fly to from {origin.city}?", field="destination")
        day = self._day(args.get("date"), what=f"the flight from {origin.city} to {destination.city}")
        times = dict(args)
        normalize_time_arguments(times)
        try:
            filters = FlightFilters.from_arguments(times)
        except ValueError:
            raise _ask(
                "invalid_time",
                "I didn't catch that departure time. What time of day would you like to leave?",
                field="time_of_day",
            ) from None
        cabin = self._cabin(args.get("cabin"), "economy")
        sort = str(args.get("sort") or "departure_time").lower()
        now = self.now()
        cancelled = self._cancelled_flights()
        available = [
            self._priced(flight, cabin)
            for flight in sample_flights(origin, destination, day)
            if datetime.fromisoformat(flight["depart_local"]) > now
            and (flight["flight_number"], flight["depart_local"]) not in cancelled
        ]
        flights = list(available)
        if available and filters.active:
            # The filters narrow the day's flights before they are sorted and cut to the first five.
            flights = [flight for flight in available if filters.matches(flight)]
        elif not available:
            filters = FlightFilters()
        if sort == "price":
            flights.sort(key=lambda flight: (flight["fare"]["amount"], flight["depart_local"]))
        elif sort == "duration":
            flights.sort(key=lambda flight: (flight.get("duration_min") or 10_000, flight["depart_local"]))
        else:
            sort = "departure_time"
            flights.sort(key=_clock_sort_key)
        flights = [{"id": f"f-{index + 1}", **flight} for index, flight in enumerate(flights[:MAX_RESULTS])]
        self._remember_search(origin, destination, day, cabin, sort, flights, filters)
        route = {"origin": origin.ref(), "destination": destination.ref()}
        head = {
            **route,
            "route_spoken": f"{origin.city} to {destination.city}",
            "date": day.isoformat(),
            "date_spoken": spoken_date(day, self.today),
            "weekday": f"{day:%A}",
            "cabin": cabin,
            "data_source": SAMPLE,
        }
        if filters.active:
            head["filters"] = filters.as_data()
        if not flights:
            return {
                "status": "no_results",
                **head,
                **self._nothing_matches(filters, available, origin, destination, day),
                **self._notice(),
            }
        return {
            "status": "ok",
            **head,
            "sort": sort,
            "count": len(flights),
            **({"unfiltered_count": len(available)} if filters.active else {}),
            "flights": [self._flight_view(index + 1, flight) for index, flight in enumerate(flights)],
            "next_step": (
                "Offer two flights at most; the rest are on the screen. A flight can be booked by its option number."
            ),
            **self._notice(),
        }

    def _cancelled_flights(self) -> set[tuple[str, str]]:
        """Return the flights that Demo Air cancelled, by number and departure: a search never offers one."""
        return {
            (leg.flight_number, leg.depart_local)
            for booking in self.store.bookings_by_id.values()
            for leg in booking.legs
            if leg.cancelled
        }

    def _remember_search(
        self,
        origin: Airport,
        destination: Airport,
        day: date,
        cabin: str,
        sort: str,
        flights: list[dict[str, Any]],
        filters: FlightFilters,
    ) -> None:
        """Keep the search as the latest one. A different search drops a read-back that moves a booking to a flight.

        Running the same search again (the same route, day, cabin and filters) keeps it, so a following "yes" still
        confirms what was read back.
        """
        state, words = self.state, filters.description()
        previous = state.search
        same = previous is not None and (
            previous.origin.get("code"),
            previous.destination.get("code"),
            previous.date,
            previous.cabin,
            previous.filters,
        ) == (origin.iata, destination.iata, day.isoformat(), cabin, words)
        if state.pending is not None and state.pending.tool in ("new_booking", "modify_existing_booking") and not same:
            state.pending = None
        state.search = FlightSearch(
            origin=origin.ref(),
            destination=destination.ref(),
            date=day.isoformat(),
            cabin=cabin,
            sort=sort,
            results=flights,
            filters=words,
        )

    def _nothing_matches(
        self, filters: FlightFilters, available: list[dict[str, Any]], origin: Airport, destination: Airport, day: date
    ) -> dict[str, Any]:
        """Say that no flight fits, and when the day's flights do leave. Never other flights."""
        when = spoken_day(day, self.today)
        route = f"from {origin.city} to {destination.city} on {when}"
        if not available:
            return {
                "reason": "no_flights_left",
                "message": f"No Demo Air flights are left {route}. Try another day.",
            }
        text = f"I don't see {filters.kind_phrase()}nonstop flights {filters.window_phrase()}{route}."
        pool = sorted(available, key=_clock_sort_key)
        first, last = spoken_time(pool[0]["depart_local"]), spoken_time(pool[-1]["depart_local"])
        if first == last:
            text += f" The only nonstop flight that day leaves at {first}."
        else:
            text += f" The nonstop flights that day leave between {first} and {last}."
        return {
            "reason": "filters",
            "message": text,
            "unfiltered_count": len(available),
            "departures_that_day": [spoken_time(f["depart_local"]) for f in pool],
        }

    def _known_schedule(self, number: str) -> dict[str, Any] | None:
        """Return the schedule of ``number`` from the caller's bookings, the latest search or the sample trips."""
        books = [*self.store.bookings_of(self.state.account_id), *self.store.bookings_by_id.values()]
        for booking in books:
            for leg in booking.legs:
                if leg.flight_number == number:
                    return {
                        "origin": leg.origin,
                        "destination": leg.destination,
                        "depart_local": leg.depart_local,
                        "arrive_local": leg.arrive_local or leg.depart_local,
                        "delay_min": leg.delay_min,
                        "cancelled": leg.cancelled,
                    }
        for flight in self.state.search.results if self.state.search else []:
            if flight["flight_number"] == number:
                return {
                    "origin": flight["origin"]["code"],
                    "destination": flight["destination"]["code"],
                    "depart_local": flight["depart_local"],
                    "arrive_local": flight["arrive_local"],
                }
        return None

    def get_flight_status(self, args: dict[str, Any]) -> dict[str, Any]:
        """Return the status of one flight: scheduled, delayed, en route or landed, with times, gate and terminal."""
        number = normalize_flight_number(args.get("flight_number"))
        if not number:
            raise _ask("missing_flight", "Which flight number, for example DA 214?", field="flight_number")
        if not is_demo_air(number):
            return {
                "status": "no_data",
                "flight": number,
                "data_source": SAMPLE,
                "reason": "not_demo_air",
                "message": f"I can only look up Demo Air flights, whose numbers start with DA, not {number}.",
                **self._notice(),
            }
        known = self._known_schedule(number)
        asked = args.get("date")
        if asked in (None, ""):
            day = date.fromisoformat(known["depart_local"][:10]) if known else self.today
        else:
            day = self._day(asked, ahead=False, what=f"flight {number}")
        schedule = known if known and day.isoformat() == known["depart_local"][:10] else None
        if schedule is None:
            schedule = move_to_day(known, day) if known else scheduled(number, day)
        status = sample_status(number, day, schedule=schedule, now=self.now())
        name = spoken_flight(number, with_code=True)
        departure, arrival = status["departure"], status["arrival"]
        origin, destination = get_airport(departure["airport"]), get_airport(arrival["airport"])
        base = {
            "flight": number,
            "airline": airline_name(number.split(" ")[0]),
            "data_source": SAMPLE,
        }
        if status["state"] == "unknown":
            return {
                "status": "no_data",
                **base,
                "reason": "unknown_flight",
                "message": f"I don't have a Demo Air flight {number} on {spoken_date(day, self.today)}.",
                **self._notice(),
            }
        return {
            "status": "ok",
            **base,
            "state": status["state"],
            "delay_min": status["delay_min"],
            **({"cause": status["reason"]} if status.get("reason") else {}),
            "route_spoken": f"{origin.city} to {destination.city}" if origin and destination else None,
            "date": status["date"],
            "date_spoken": spoken_date(status["date"], self.today),
            "departure": _movement_view(departure),
            "arrival": _movement_view(arrival),
            "summary": _status_text(status, name, self.today),
            **self._notice(),
        }

    def where_we_fly(self, args: dict[str, Any]) -> dict[str, Any]:
        """Return the twelve cities that Demo Air serves, with the airline's scope."""
        return {
            "status": "ok",
            "airline": AIRLINE_NAME,
            "city_count": len(all_airports()),
            "cities": self._served_list(),
            "routes": (
                "Every pair of these cities has nonstop flights, three to five times a day in each direction, "
                f"on dates up to {MAX_DAYS_AHEAD} days ahead."
            ),
            "next_step": "Say the cities in a short list; they are on the screen.",
            **self._notice(),
        }

    def get_ontime_trend(self, args: dict[str, Any]) -> dict[str, Any]:
        """Return how often a flight was on time over about 90 days, and its typical delay."""
        number = normalize_flight_number(args.get("flight_number"))
        if not number:
            raise _ask("missing_flight", "Which flight should I check, for example DA 214?", field="flight_number")
        if not is_demo_air(number):
            raise _ask(
                "not_demo_air",
                f"I can only check Demo Air flights, whose numbers start with DA. Which Demo Air flight is it, "
                f"rather than {number}?",
                field="flight_number",
            )
        trend = sample_trend(number)
        name = spoken_flight(number, with_code=True)
        return {
            "status": "ok",
            "flight": number,
            "airline": airline_name(number.split(" ")[0]),
            **trend,
            "summary": (
                f"Over the last {TREND_WINDOW_DAYS} days, {name} was on time {trend['on_time_pct']} percent of the "
                "time, "
                f"across {trend['flights_counted']} flights. When it's late, the typical delay is about "
                f"{trend['avg_delay_min']} minutes."
            ),
            "data_source": SAMPLE,
            **self._notice(),
        }

    def sign_in(self, args: dict[str, Any]) -> dict[str, Any]:
        """Open a demo account for this session by user id, or create one in the caller's name (no password)."""
        raw_user, new_name = str(args.get("user_id") or "").strip(), str(args.get("new_account_name") or "").strip()
        created = False
        if raw_user:
            user = self.store.find_user(raw_user)
            if user is None:
                raise self._need_sign_in(
                    "I couldn't find an account with that user ID. I can create a demo account in the caller's name, "
                    "or use "
                    f"the sample account {DEFAULT_USER_SPOKEN}.",
                    unknown_user=raw_user,
                )
        elif new_name:
            user = self.store.find_user(new_name)
            if user is None:
                user = self.store.create_user(new_name)
                created = user is not None
            if user is None:
                raise _ask(
                    "invalid_name",
                    "I couldn't open an account with that name. Please tell me the first and last name, or use the "
                    "sample "
                    f"account {DEFAULT_USER_SPOKEN}.",
                    field="new_account_name",
                )
        else:
            raise self._need_sign_in()
        if self.state.account_id != user.user_id:
            self.state.clear_pending()
        self.state.account_id = user.user_id
        trips = self.store.bookings_of(user.user_id, active_only=True)
        result: dict[str, Any] = {
            "status": "ok",
            "signed_in": True,
            "created": created,
            "account": {**user.view(), "user_id_spoken": spoken_user_id(user.user_id)},
            "trip_count": len(trips),
            "trips": [self._booking_view(trip) for trip in trips],
        }
        heads_up = self._delay_heads_up(trips)
        if heads_up:
            result["heads_up"] = heads_up
        if created:
            result["summary"] = (
                f"Done, {user.name}. A demo account is created and signed in. The user ID is "
                f"{spoken_user_id(user.user_id)}. There are no trips yet."
            )
        return result

    def _delay_heads_up(self, trips: list[Booking]) -> str:
        """Return a sentence about a flight within two days that is cancelled, or delayed by 15 minutes or more."""
        now = self.now()
        soon = [
            (booking, leg)
            for booking in trips
            for leg in booking.legs
            if timedelta(0) < datetime.fromisoformat(leg.depart_local) - now < timedelta(days=2)
        ]
        for _, leg in soon:
            if leg.cancelled:
                view = leg.view()
                return (
                    f"{self._flight_phrase(view)} from {self._route_phrase(view)} on "
                    f"{spoken_day(leg.depart_local, self.today)} has been cancelled."
                )
        for _, leg in soon:
            if (leg.delay_min or 0) >= 15:
                view = leg.view()
                return (
                    f"{self._flight_phrase(view)} from {self._route_phrase(view)} shows a delay of about "
                    f"{leg.delay_min} minutes."
                )
        return ""

    def get_existing_bookings(self, args: dict[str, Any]) -> dict[str, Any]:
        """List the signed-in caller's trips, signing in first when a user id is given."""
        raw_user = str(args.get("user_id") or "").strip()
        if raw_user and (self.state.account_id is None or user_key(raw_user) != user_key(self.state.account_id)):
            self.sign_in({"user_id": raw_user})
        user = self.store.accounts[self._signed_in()]
        bookings = self.store.bookings_of(user.user_id)
        active = [item for item in bookings if item.active]
        return {
            "status": "ok",
            "account": {**user.view(), "user_id_spoken": spoken_user_id(user.user_id)},
            "active_count": len(active),
            "cancelled_count": len(bookings) - len(active),
            "bookings": [self._booking_view(item) for item in bookings],
            "next_step": (
                "The trips are on the screen: name three at most by trip_spoken. Give booking codes only when asked."
            ),
            **self._notice(),
        }

    # ------------------------------------------------------------------------------- change tools

    def new_booking(self, args: dict[str, Any]) -> dict[str, Any]:
        """Book one flight from the latest search as a demo booking, after a read-back and a yes."""
        user = self.store.accounts[self._signed_in()]
        index = self._select_flight(args.get("flight"))
        search = self.state.search
        cabin = self._cabin(args.get("cabin"), search.cabin)
        flight = search.results[index]
        if cabin != flight["fare"]["cabin"]:
            flight = self._priced(flight, cabin)
        passengers = _passengers(args.get("passengers"), user.name)
        each = flight["fare"]["amount"]
        total = each * len(passengers)
        price = (
            f"The fare is {spoken_amount(each)}"
            if len(passengers) == 1
            else f"The fare is {spoken_amount(each)} each, {spoken_amount(total)} in total"
        )
        summary = (
            f"Here is the booking: {self._flight_phrase(flight)} from {self._route_phrase(flight)} on "
            f"{spoken_day(flight['depart_local'], self.today)} at {spoken_time(flight['depart_local'])}, "
            f"{CABIN_LABELS[cabin]}, for {', '.join(passengers)}. {price}. Nothing is booked yet."
        )
        key = (
            "new_booking",
            user.user_id,
            flight["flight_number"],
            flight["depart_local"],
            cabin,
            "|".join(passengers),
        )
        details = {"flight": self._flight_view(index + 1, flight), "passengers": passengers, "total_usd": total}
        self._read_back("new_booking", key, "booked", summary, details, _is_yes(args.get("confirmed")))
        booking = self.store.create_booking(
            user_id=user.user_id, cabin=cabin, passengers=passengers, legs=[Leg.from_flight(flight, each)]
        )
        return self._committed(
            "booked",
            booking,
            f"You're booked. The booking code is {spoken_code(booking.booking_id)}: {self._flight_phrase(flight)}, "
            f"{self._route_phrase(flight)}, {spoken_day(flight['depart_local'], self.today)} at "
            f"{spoken_time(flight['depart_local'])}.",
            next_step="Offer a seat or a meal.",
        )

    def modify_existing_booking(self, args: dict[str, Any]) -> dict[str, Any]:
        """Move a booking to a flight from the latest search, after a read-back and a yes."""
        booking = self._active(self._resolve_booking(args.get("booking_id")))
        index = self._select_flight(args.get("flight"))
        search = self.state.search
        new = self._priced(search.results[index], booking.cabin)
        legs = booking.legs
        exact = [
            position
            for position, leg in enumerate(legs)
            if leg.origin == new["origin"]["code"] and leg.destination == new["destination"]["code"]
        ]
        whole_trip = (
            not exact and legs[0].origin == new["origin"]["code"] and legs[-1].destination == new["destination"]["code"]
        )
        if exact:
            replaced = [exact[0]]
        elif whole_trip or len(legs) == 1:
            replaced = list(range(len(legs)))
        else:
            raise _ask(
                "route_mismatch",
                f"That flight, {self._route_phrase(new)}, doesn't match any part of booking {booking.booking_id}, "
                f"{self._trip_phrase(booking)}. A change keeps the booking's route: search that route for another "
                "flight, or book this one as a new trip.",
                field="flight",
            )
        passengers = len(booking.passengers)
        old_fare = sum(self.store.leg_fare_today(legs[position], booking.cabin) for position in replaced)
        difference = (new["fare"]["amount"] - old_fare) * passengers
        free = any(legs[position].cancelled for position in replaced)
        if free:
            difference = 0.0
        fare_line = (
            "Demo Air cancelled the old flight, so there is no charge for the change"
            if free
            else _difference_phrase(difference, passengers)
        )
        old = legs[replaced[0]].view()
        # A one-flight trip may move to another route; the read-back then names both, so a yes never changes one unsaid.
        old_route = f"{old['origin']['city']} to {legs[replaced[-1]].view()['destination']['city']}"
        new_route = self._route_phrase(new)
        old_at, new_at = (f", {old_route},", f", {new_route},") if old_route != new_route else ("", "")
        summary = (
            f"Here is the change: move booking {booking.booking_id} from {self._flight_phrase(old)}{old_at} on "
            f"{spoken_day(old['depart_local'], self.today)} at {spoken_time(old['depart_local'])} to "
            f"{self._flight_phrase(new)}{new_at} on {spoken_day(new['depart_local'], self.today)} at "
            f"{spoken_time(new['depart_local'])}. {fare_line}. "
            "The seat is released; the meal stays. Nothing has changed yet."
        )
        key = (
            "modify_existing_booking",
            booking.booking_id,
            ",".join(map(str, replaced)),
            new["flight_number"],
            new["depart_local"],
        )
        details = {
            "booking_id": booking.booking_id,
            "new_flight": self._flight_view(index + 1, new),
            "fare_difference_usd": difference,
            **({"no_charge_reason": "Demo Air cancelled the old flight"} if free else {}),
        }
        self._read_back("modify_existing_booking", key, "modified", summary, details, _is_yes(args.get("confirmed")))
        # A flight that Demo Air cancelled is rebooked at no charge: the booking keeps the fare that was paid.
        paid = sum(legs[position].fare for position in replaced) if free else new["fare"]["amount"]
        row = Leg.from_flight(new, paid)
        if len(replaced) == 1 and len(legs) > 1:
            row.meal = legs[replaced[0]].meal
            legs[replaced[0]] = row
        else:
            row.meal = legs[0].meal
            booking.legs = [row]
        return self._committed(
            "modified",
            booking,
            f"Done. Booking {booking.booking_id} is now on {self._flight_phrase(new)}, "
            f"{spoken_day(new['depart_local'], self.today)} at {spoken_time(new['depart_local'])}.",
            fare_difference_usd=difference,
            next_step="Offer a seat.",
        )

    def cancel_booking(self, args: dict[str, Any]) -> dict[str, Any]:
        """Cancel a booking and state the refund rule for its fare, after a read-back and a yes."""
        booking = self._active(self._resolve_booking(args.get("booking_id")))
        refund = _refund_rule(booking)
        summary = (
            f"Here is the cancellation: booking {booking.booking_id}, {self._trip_phrase(booking)}. {refund}. "
            "Nothing is cancelled yet."
        )
        details = {"booking_id": booking.booking_id, "refund_rule": refund}
        self._read_back(
            "cancel_booking",
            ("cancel_booking", booking.booking_id),
            "cancelled",
            summary,
            details,
            _is_yes(args.get("confirmed")),
        )
        booking.status = "cancelled"
        return self._committed(
            "cancelled", booking, f"Booking {booking.booking_id} is cancelled. {refund}.", refund_rule=refund
        )

    def select_seat_and_meal(self, args: dict[str, Any]) -> dict[str, Any]:
        """Pick a seat (number, window, aisle or exit row) and/or a special meal, after a read-back and a yes."""
        wanted_seat, wanted_meal = args.get("seat"), args.get("meal")
        if not wanted_seat and not wanted_meal:
            raise _ask(
                "seat_or_meal", "Would you like a window or an aisle seat, and any special meal?", field="seat,meal"
            )
        booking = self._active(self._resolve_booking(args.get("booking_id")))
        positions = list(range(len(booking.legs)))
        number = normalize_flight_number(args.get("flight_number"))
        if number:
            positions = [p for p in positions if booking.legs[p].flight_number == number] or positions
        meal = None
        if wanted_meal:
            meal = parse_meal(wanted_meal)
            if meal is None:
                raise _ask(
                    "unknown_meal",
                    "Which meal would you like? I can arrange vegetarian, vegan, Asian vegetarian, Jain, halal, "
                    "kosher, "
                    "gluten-free, diabetic, or a child meal.",
                    field="meal",
                )
        seats: dict[int, str] = {}
        if wanted_seat:
            for position in positions:
                seats[position] = self._choose_seat(booking, position, wanted_seat)
        parts = []
        for position in positions:
            pieces = []
            if position in seats:
                kind = seat_type(seats[position][-1])
                exit_note = ", an exit row" if int(re.match(r"\d+", seats[position]).group()) in EXIT_ROWS else ""
                pieces.append(f"seat {seats[position]}, {'an' if kind[0] in 'aeiou' else 'a'} {kind} seat{exit_note}")
            if meal:
                pieces.append(f"a {meal_label(None if meal == NO_MEAL else meal)}")
            parts.append(f"{' and '.join(pieces)} on {self._flight_phrase(booking.legs[position].view())}")
        summary = (
            f"Here is the change for booking {booking.booking_id}: {'; and '.join(parts)}. Nothing has changed yet."
        )
        key = (
            "select_seat_and_meal",
            booking.booking_id,
            ",".join(f"{leg}:{seat}" for leg, seat in sorted(seats.items())),
            meal or "",
        )
        details = {
            "booking_id": booking.booking_id,
            "seats": {booking.legs[p].flight_number: seat for p, seat in seats.items()},
            "meal": meal,
        }
        self._read_back("select_seat_and_meal", key, "seat_meal", summary, details, _is_yes(args.get("confirmed")))
        for position in positions:
            leg = booking.legs[position]
            leg.seat = seats.get(position, leg.seat)
            if meal is not None:
                leg.meal = None if meal == NO_MEAL else meal
        return self._committed(
            "seat_meal", booking, f"All set for booking {booking.booking_id}: {'; and '.join(parts)}.", **details
        )

    def _choose_seat(self, booking: Booking, position: int, wanted: object) -> str:
        """Return the seat to assign on one leg, or raise the reply that says why none fits."""
        leg, cabin = booking.legs[position], booking.cabin
        flight_key = f"{leg.flight_number}|{leg.depart_local[:10]}"
        held = self.store.held_seats(leg.flight_number, leg.depart_local) - {leg.seat or ""}
        explicit = parse_seat(wanted)
        if explicit:
            row, letter = explicit
            if not seat_in_cabin(row, cabin):
                rows = CABIN_ROWS[cabin]
                raise _ask(
                    "seat_unavailable",
                    f"Row {row} isn't in {CABIN_LABELS[cabin]}; the cabin has rows {rows.start} to {rows.stop - 1}.",
                    field="seat",
                )
            if seat_free(flight_key, row, letter, held) or leg.seat == f"{row}{letter}":
                return f"{row}{letter}"
            alternative = find_seat(flight_key, cabin, seat_type(letter), held)
            problem = f"{row}{letter} is taken. " + (
                f"The nearest free {seat_type(letter)} seat is {alternative}."
                if alternative
                else f"There are no free {seat_type(letter)} seats in the cabin."
            )
            raise _ask("seat_unavailable", problem, field="seat", alternative=alternative)
        kind = seat_kind(wanted)
        if kind is None:
            raise _ask(
                "seat_unclear",
                "Would you like a window, an aisle, or an exit row seat, or a seat number such as 18A?",
                field="seat",
            )
        if kind == "exit" and cabin != "economy":
            kind = "any"
        seat = find_seat(flight_key, cabin, kind, held)
        if seat is None:
            raise _ask(
                "seat_unavailable", f"There are no free {kind} seats left in the cabin on that flight.", field="seat"
            )
        return seat

    def upgrade_class(self, args: dict[str, Any]) -> dict[str, Any]:
        """Move a booking to another cabin and state the demo price difference, after a read-back and a yes."""
        cabin = normalize_cabin(args.get("cabin"), default=None)
        if cabin is None:
            raise _ask("which_cabin", "Which cabin would you like: premium economy, business, or first?", field="cabin")
        booking = self._active(self._resolve_booking(args.get("booking_id")))
        current = booking.cabin
        if cabin == current:
            raise _fail("same_cabin", f"Booking {booking.booking_id} is already in {CABIN_LABELS[cabin]}.")
        passengers = len(booking.passengers)
        new_fares = [self.store.leg_fare_today(leg, cabin) for leg in booking.legs]
        old_fares = [self.store.leg_fare_today(leg, current) for leg in booking.legs]
        difference = (sum(new_fares) - sum(old_fares)) * passengers
        verb = "upgrade" if cabin_rank(cabin) > cabin_rank(current) else "move"
        summary = (
            f"Here is the change: {verb} booking {booking.booking_id} from {CABIN_LABELS[current]} to "
            f"{CABIN_LABELS[cabin]}. {_difference_phrase(difference, passengers)}. "
            "Seats are released so new ones can be "
            f"picked in {CABIN_LABELS[cabin]}. Nothing has changed yet."
        )
        details = {"booking_id": booking.booking_id, "cabin": cabin, "fare_difference_usd": difference}
        self._read_back(
            "upgrade_class",
            ("upgrade_class", booking.booking_id, cabin),
            "upgraded",
            summary,
            details,
            _is_yes(args.get("confirmed")),
        )
        for leg, new, old in zip(booking.legs, new_fares, old_fares, strict=True):
            leg.fare += new - old
            leg.seat = None
        booking.cabin = cabin
        return self._committed(
            "upgraded",
            booking,
            f"Done. Booking {booking.booking_id} is now in {CABIN_LABELS[cabin]}.",
            fare_difference_usd=difference,
            next_step="Offer to pick a seat in the new cabin.",
        )


# ------------------------------------------------------------------------------------------ helpers


def _movement_view(point: Mapping[str, Any]) -> dict[str, Any]:
    """Return one end of a status with its spoken times."""
    return {
        "airport": point.get("airport"),
        "scheduled": point.get("scheduled"),
        "scheduled_spoken": spoken_time(point.get("scheduled")),
        "estimated": point.get("estimated"),
        "estimated_spoken": spoken_time(point.get("estimated")),
        "actual": point.get("actual"),
        "actual_spoken": spoken_time(point.get("actual")),
        "terminal": point.get("terminal"),
        "gate": point.get("gate"),
    }


def _status_text(status: Mapping[str, Any], flight_name: str, today: date) -> str:
    """Return the status as a sentence."""
    departure, arrival = status["departure"], status["arrival"]
    origin, destination = get_airport(departure.get("airport")), get_airport(arrival.get("airport"))
    route = f" from {origin.city} to {destination.city}" if origin and destination else ""
    state, scheduled = status["state"], departure.get("scheduled")
    when = f" on {spoken_day(scheduled, today)}" if scheduled else ""
    cause = f" because of {status['reason']}" if status.get("reason") else ""
    if state == "delayed":
        expected = departure.get("estimated") or scheduled
        return (
            f"{flight_name}{route} is delayed by about {status.get('delay_min') or 0} minutes{cause}: it's now "
            f"expected to leave at {spoken_time(expected)} instead of {spoken_time(scheduled)}{when}."
        )
    if state == "boarding":
        gate = f" at gate {departure['gate']}" if departure.get("gate") else ""
        expected = departure.get("estimated") or scheduled
        return f"{flight_name}{route} is boarding now{gate}, and leaves at {spoken_time(expected)}{when}."
    if state in ("departed", "en_route"):
        lands = arrival.get("estimated") or arrival.get("scheduled")
        landing = f" and is expected to land at {spoken_time(lands)}" if lands else ""
        return f"{flight_name}{route} departed at {spoken_time(departure.get('actual') or scheduled)}{landing}."
    if state == "landed":
        return f"{flight_name}{route} has landed, at {spoken_time(arrival.get('actual') or arrival.get('scheduled'))}."
    if state == "cancelled":
        return f"{flight_name}{route} is cancelled{when}{cause}."
    gate = f", gate {departure['gate']}" if departure.get("gate") else ""
    terminal = f" from terminal {departure['terminal']}" if departure.get("terminal") else ""
    return f"{flight_name}{route} is on schedule to leave at {spoken_time(scheduled)}{when}{terminal}{gate}."


def _chosen_index(results: list[dict[str, Any]], raw: object) -> int | None:
    """Return the index of the searched flight that ``raw`` names: option number, ordinal, flight number or time."""
    text = str(raw if raw is not None else "").strip().lower()
    if not text or not results:
        return None
    if text in _ORDINALS:
        position = _ORDINALS[text]
        return len(results) - 1 if position == -1 else (position - 1 if position <= len(results) else None)
    for word, position in _ORDINALS.items():
        # "First class" names a cabin, not the first flight.
        if re.search(rf"\b{word}\b(?!\s+class)", text) and position != -1 and position <= len(results):
            return position - 1
    match = re.fullmatch(r"(?:option\s*|f-?|#)?(\d)", text)
    if match and 1 <= int(match.group(1)) <= len(results):
        return int(match.group(1)) - 1
    number = normalize_flight_number(text)
    if number:
        for index, flight in enumerate(results):
            if flight["flight_number"] == number:
                return index
    digits = re.fullmatch(r"\D*(\d{3,4})", text)
    if digits:
        hits = [i for i, flight in enumerate(results) if flight["flight_number"].endswith(f" {int(digits.group(1))}")]
        if len(hits) == 1:
            return hits[0]
    if "cheapest" in text:
        return min(range(len(results)), key=lambda index: results[index]["fare"]["amount"])
    if "earliest" in text:
        return min(range(len(results)), key=lambda index: results[index]["depart_local"])
    clock = re.search(r"\b(\d{1,2})(?::(\d{2}))?\s*(a\.?m\.?|p\.?m\.?)?(?!\d)", text)
    # A bare number is an option number, so a time needs its minutes or am or pm.
    if clock and (clock.group(2) or clock.group(3)):
        hour, minute = int(clock.group(1)), int(clock.group(2) or 0)
        suffix = (clock.group(3) or "").replace(".", "")
        if suffix == "pm" and hour < 12:
            hour += 12
        if suffix == "am" and hour == 12:
            hour = 0
        hits = [
            index
            for index, flight in enumerate(results)
            if ((stamp := datetime.fromisoformat(flight["depart_local"])).minute == minute or clock.group(2) is None)
            and (stamp.hour == hour or (not suffix and stamp.hour == hour + 12))
        ]
        if len(hits) == 1:
            return hits[0]
    return None


def _passengers(raw: object, account_name: str) -> list[str]:
    """Return the passenger names: the account holder when none are given, "me" meaning the account holder."""
    names = [str(item).strip() for item in raw] if isinstance(raw, list) else re.split(r",|\band\b|;", str(raw or ""))
    cleaned = []
    for name in (part.strip() for part in names):
        if not name:
            continue
        if name.lower() in {"me", "myself", "i", "self"}:
            name = account_name
        cleaned.append(" ".join(word.capitalize() if word.islower() else word for word in name.split()))
    return (list(dict.fromkeys(cleaned)) or [account_name])[:MAX_PASSENGERS]


def _difference_phrase(difference: float, passengers: int) -> str:
    who = "" if passengers == 1 else f" for {passengers} passengers"
    if abs(difference) < 1:
        return "There is no fare difference"
    if difference > 0:
        return f"The fare difference is {spoken_amount(difference)} more{who}"
    return f"The fare is {spoken_amount(-difference)} less{who}"


def _refund_rule(booking: Booking) -> str:
    """Return the refund rule of a booking's fare, as a sentence without its final period."""
    amount = spoken_amount(booking.total)
    if any(leg.cancelled for leg in booking.legs):
        return f"Demo Air cancelled the flight, so the full {amount} would be refunded"
    if booking.created_here:
        return f"It was booked within the last 24 hours, so the full {amount} would be refunded"
    if booking.cabin in ("business", "first"):
        return f"This {CABIN_LABELS[booking.cabin]} fare is fully refundable: {amount} back"
    if booking.cabin == "premium_economy":
        back = spoken_amount(max(0.0, booking.total - 100))
        return f"The premium economy fare is refundable minus a $100 fee: {back} back"
    return f"The economy fare is non-refundable, so {amount} would go to a travel credit"


__all__ = ["TOOL_NAMES", "AirlineTools"]
