# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The desk's in-memory accounts and bookings: seeded with the sample accounts, made fresh for each session.

Nothing is written to disk and nothing is shared between sessions. The sample accounts' trips are flights of the
timetable on dates relative to the session's day, so they are always upcoming.
"""

from __future__ import annotations

import random
import re
import unicodedata
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from typing import Any

from .airports import Airport, get_airport
from .carriers import DIGIT_WORDS, HOME_ZONE
from .fares import CURRENCY, demo_fare
from .schedule import sample_flights

DEFAULT_USER_ID = "jane_doe_001"
DEFAULT_USER_SPOKEN = "jane doe zero zero one"
# A sample flight "next" is the first one that leaves at least this long after the session starts.
NEXT_FLIGHT_LEAD = timedelta(minutes=90)
MAX_CREATED_ACCOUNTS = 20
_BOOKING_ALPHABET = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"
_USER_KEY_RE = re.compile(r"[^a-z0-9]+")
# A name a caller can open an account under: letters (any script), spaces, hyphens, apostrophes and dots.
_ACCOUNT_NAME_RE = re.compile(r"^[^\W\d_]+(?:[ '\-.][^\W\d_]+)*\.?$")
_MAX_ACCOUNT_NAME = 60


@dataclass(frozen=True, slots=True)
class SeedLeg:
    """One flight of a sample trip: a flight of the timetable on a route, on a day, picked by its place in the day.

    ``day`` is days from today. ``slot`` is the flight's place among the route's departures of that day, earliest
    first (0 is the first), or None for the next flight that leaves at least 90 minutes from now, today or else
    tomorrow. ``cancelled`` is a flight that the airline cancelled.
    """

    origin: str
    destination: str
    day: int
    slot: int | None
    seat: str | None = None
    meal: str | None = None
    delay_min: int | None = None
    cancelled: bool = False


@dataclass(frozen=True, slots=True)
class SeedTrip:
    """One sample booking."""

    booking_id: str
    cabin: str
    legs: tuple[SeedLeg, ...]


@dataclass(frozen=True, slots=True)
class SeedAccount:
    """One sample account."""

    user_id: str
    name: str
    tier: str
    hint: str
    trips: tuple[SeedTrip, ...]


# The accounts every session starts with. A caller can also open an account in their own name.
SAMPLE_ACCOUNTS = (
    SeedAccount(
        user_id=DEFAULT_USER_ID,
        name="Jane Doe",
        tier="Gold",
        hint="2 trips, one delayed",
        trips=(
            SeedTrip("K7Q2ZP", "economy", (SeedLeg("BOS", "DEN", 1, 1, "18A", "VGML", 45),)),
            SeedTrip("M4X8RT", "business", (SeedLeg("JFK", "LAX", 24, 3, "5A"),)),
        ),
    ),
    SeedAccount(
        user_id="marcus_lee_002",
        name="Marcus Lee",
        tier="Silver",
        hint="1 trip, flight cancelled",
        trips=(SeedTrip("R9T4WN", "economy", (SeedLeg("ORD", "SEA", 0, None, "12C", cancelled=True),)),),
    ),
)


def user_key(text: object) -> str:
    """Return a user id reduced to letters and digits, with spoken digits as numerals ("one zero" -> "10")."""
    words = re.split(r"[\s_\-.]+", str(text or "").lower().strip())
    return _USER_KEY_RE.sub("", "".join(DIGIT_WORDS.get(word, word) for word in words))


def booking_key(text: object) -> str:
    """Return a booking code as letters and digits in capitals ("k 7 q-2 z p" -> "K7Q2ZP")."""
    words = re.split(r"[\s_\-.,]+", str(text or "").strip())
    joined = "".join(DIGIT_WORDS.get(word.lower(), word) for word in words if word)
    return re.sub(r"[^A-Za-z0-9]", "", joined).upper()


@dataclass(slots=True)
class Leg:
    """One flight of a booking, with its seat, meal and per-passenger fare."""

    flight_number: str
    airline_code: str
    airline_name: str
    origin: str
    destination: str
    depart_local: str
    arrive_local: str | None
    fare: float
    duration_min: int | None = None
    aircraft: str | None = None
    seat: str | None = None
    meal: str | None = None
    delay_min: int | None = None
    cancelled: bool = False

    @classmethod
    def from_flight(cls, flight: dict[str, Any], fare: float) -> Leg:
        """Return the leg that books a searched flight at ``fare`` per passenger."""
        return cls(
            flight_number=flight["flight_number"],
            airline_code=flight["airline"]["code"],
            airline_name=flight["airline"]["name"],
            origin=flight["origin"]["code"],
            destination=flight["destination"]["code"],
            depart_local=flight["depart_local"],
            arrive_local=flight.get("arrive_local"),
            fare=fare,
            duration_min=flight.get("duration_min"),
            aircraft=flight.get("aircraft"),
        )

    def view(self) -> dict[str, Any]:
        """Return the leg as the desk shows it."""
        origin, destination = get_airport(self.origin), get_airport(self.destination)
        return {
            "flight_number": self.flight_number,
            "airline": {"code": self.airline_code, "name": self.airline_name},
            "origin": {"code": self.origin, "city": origin.city if origin else self.origin},
            "destination": {"code": self.destination, "city": destination.city if destination else self.destination},
            "depart_local": self.depart_local,
            "arrive_local": self.arrive_local,
            "seat": self.seat,
            "meal": self.meal,
        }


@dataclass(slots=True)
class Booking:
    """One booking: its passengers, cabin and flights. ``total`` is the demo fare of every leg for every passenger."""

    booking_id: str
    user_id: str
    cabin: str
    passengers: list[str]
    legs: list[Leg]
    status: str = "confirmed"
    created_here: bool = False
    updated_at: str = ""

    @property
    def total(self) -> float:
        """The demo total: every leg's fare, for every passenger."""
        return sum(leg.fare for leg in self.legs) * max(1, len(self.passengers))

    @property
    def active(self) -> bool:
        """Whether the booking is not cancelled."""
        return self.status == "confirmed"

    def view(self) -> dict[str, Any]:
        """Return the booking as the desk shows it."""
        return {
            "booking_id": self.booking_id,
            "status": self.status,
            "cabin": self.cabin,
            "passengers": [{"name": name} for name in self.passengers],
            "flights": [leg.view() for leg in self.legs],
            "total": {"amount": float(self.total), "currency": CURRENCY},
            "updated_at": self.updated_at,
        }


@dataclass(frozen=True, slots=True)
class Account:
    """One demo account."""

    user_id: str
    name: str
    tier: str
    hint: str = ""

    def view(self) -> dict[str, str]:
        """Return the account as the desk shows it."""
        return {"user_id": self.user_id, "name": self.name, "tier": self.tier}


class DeskStore:
    """One session's accounts and bookings, in memory."""

    def __init__(
        self,
        *,
        now: Callable[[], datetime] = lambda: datetime.now(UTC),
        accounts: Iterable[SeedAccount] = SAMPLE_ACCOUNTS,
        rng: random.Random | None = None,
    ) -> None:
        """Seed ``accounts`` (the sample account by default) with trips dated from the day of ``now``.

        ``rng`` draws the booking codes; a seeded one makes them repeatable.
        """
        self._now, self._rng = now, rng or random.Random()
        self.accounts: dict[str, Account] = {}
        self.bookings_by_id: dict[str, Booking] = {}
        self._seeded = 0
        for seed in accounts:
            self._seed(seed)
        self._seeded = len(self.accounts)

    @property
    def today(self) -> date:
        """The day the store's dates are relative to: the day of its clock in the airline's home zone."""
        return self._now().astimezone(HOME_ZONE).date()

    def stamp(self) -> str:
        """Return the current time as the booking's update stamp."""
        return self._now().astimezone(UTC).isoformat(timespec="seconds")

    # ------------------------------------------------------------------------------------------- seed

    def _seed(self, seed: SeedAccount) -> None:
        self.accounts[seed.user_id] = Account(seed.user_id, seed.name, seed.tier, seed.hint)
        for trip in seed.trips:
            legs = [self._seed_leg(item, trip.cabin) for item in trip.legs]
            self.bookings_by_id[trip.booking_id] = Booking(
                trip.booking_id, seed.user_id, trip.cabin, [seed.name], legs, updated_at=self.stamp()
            )

    def _seed_flight(self, origin: Airport, destination: Airport, seed: SeedLeg) -> dict[str, Any]:
        """Return the timetable flight that a sample leg names."""
        if seed.slot is not None:
            return sample_flights(origin, destination, self.today + timedelta(days=seed.day))[seed.slot]
        earliest = self._now() + NEXT_FLIGHT_LEAD
        for ahead in range(seed.day, seed.day + 2):
            for flight in sample_flights(origin, destination, self.today + timedelta(days=ahead)):
                if datetime.fromisoformat(flight["depart_local"]) >= earliest:
                    return flight
        raise ValueError(f"no timetable flight for sample trip {seed.origin}-{seed.destination}")

    def _seed_leg(self, seed: SeedLeg, cabin: str) -> Leg:
        origin, destination = get_airport(seed.origin), get_airport(seed.destination)
        if origin is None or destination is None:
            raise ValueError(f"unknown airport in sample trip {seed.origin}-{seed.destination}")
        flight = self._seed_flight(origin, destination, seed)
        fare = demo_fare(
            origin=origin,
            destination=destination,
            departure=datetime.fromisoformat(flight["depart_local"]),
            cabin=cabin,
            flight_number=flight["flight_number"],
            today=self.today - timedelta(days=30),
        )
        leg = Leg.from_flight(flight, fare)
        leg.seat, leg.meal = seed.seat, seed.meal
        leg.delay_min, leg.cancelled = seed.delay_min or 0, seed.cancelled
        return leg

    # ------------------------------------------------------------------------------------------ users

    def sample_accounts(self) -> list[dict[str, str]]:
        """Return every account, for the panel's sign-in hints."""
        return [
            {"user_id": item.user_id, "name": item.name, "hint": item.hint}
            for item in sorted(self.accounts.values(), key=lambda account: account.user_id)
        ]

    def find_user(self, text: object) -> Account | None:
        """Return the account whose id (or unique full name) matches ``text`` however it was said."""
        wanted = user_key(text)
        if not wanted:
            return None
        for account in self.accounts.values():
            if user_key(account.user_id) == wanted:
                return account
        by_name = [account for account in self.accounts.values() if user_key(account.name) == wanted]
        return by_name[0] if len(by_name) == 1 else None

    def create_user(self, name: str) -> Account | None:
        """Open an account in the caller's ``name``; None for a name that is not one or past the session's limit.

        The user id is the name's ASCII letters joined by underscores plus a three-digit number
        ("alex_morgan_003", after the sample accounts 001 and 002). The account has no trips. A name already on an
        account opens that account.
        """
        name = " ".join(str(name or "").split()).strip(" .")
        if not (2 <= len(name) <= _MAX_ACCOUNT_NAME) or not _ACCOUNT_NAME_RE.fullmatch(name):
            return None
        existing = self.find_user(name)
        if existing is not None:
            return existing
        if len(self.accounts) >= self._seeded + MAX_CREATED_ACCOUNTS:
            return None
        ascii_name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii")
        words = [re.sub(r"[^a-z]", "", word) for word in ascii_name.casefold().split()]
        stem = "_".join(word for word in words if word)[:40] or "guest"
        number = len(self.accounts) + 1
        while f"{stem}_{number:03d}" in self.accounts:
            number += 1
        account = Account(f"{stem}_{number:03d}", name.title() if name.islower() else name, "Member", "New account")
        self.accounts[account.user_id] = account
        return account

    # ------------------------------------------------------------------------------------- bookings

    def booking(self, booking_id: str) -> Booking | None:
        """Return one booking by its code, or None."""
        return self.bookings_by_id.get(booking_id)

    def bookings_of(self, user_id: str | None, *, active_only: bool = False) -> list[Booking]:
        """Return a user's bookings, earliest departure first."""
        found = [
            item
            for item in self.bookings_by_id.values()
            if item.user_id == user_id and (item.active or not active_only)
        ]
        return sorted(found, key=lambda item: (item.legs[0].depart_local, item.booking_id))

    def held_seats(self, flight_number: str, depart_local: str) -> set[str]:
        """Return the seats this session's active bookings hold on one flight."""
        return {
            leg.seat
            for item in self.bookings_by_id.values()
            if item.active
            for leg in item.legs
            if leg.seat and leg.flight_number == flight_number and leg.depart_local == depart_local
        }

    def new_booking_id(self) -> str:
        """Return an unused six-character booking code with at least one letter and one digit."""
        while True:
            code = "".join(self._rng.choice(_BOOKING_ALPHABET) for _ in range(6))
            if code.isalpha() or code.isdigit() or code in self.bookings_by_id:
                continue
            return code

    def create_booking(self, *, user_id: str, cabin: str, passengers: list[str], legs: list[Leg]) -> Booking:
        """Store a new confirmed booking."""
        booking = Booking(
            self.new_booking_id(), user_id, cabin, list(passengers), legs, created_here=True, updated_at=self.stamp()
        )
        self.bookings_by_id[booking.booking_id] = booking
        return booking

    def touch(self, booking: Booking) -> None:
        """Stamp a booking as just changed."""
        booking.updated_at = self.stamp()

    def leg_fare_today(self, leg: Leg, cabin: str) -> float:
        """Return today's demo fare for one booked leg in ``cabin``."""
        origin, destination = get_airport(leg.origin), get_airport(leg.destination)
        if origin is None or destination is None:
            return 0.0
        return demo_fare(
            origin=origin,
            destination=destination,
            departure=datetime.fromisoformat(leg.depart_local),
            cabin=cabin,
            flight_number=leg.flight_number,
            today=self.today,
        )
