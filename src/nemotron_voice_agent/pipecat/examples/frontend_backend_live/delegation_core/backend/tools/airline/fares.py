# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Demo fares: a deterministic price model, so the same flight, cabin and day always cost the same.

    fare = base(distance) x cabin x time of day x days ahead x a small per-flight spread

Every fare is labelled "demo fare"; nothing is ever charged.
"""

from __future__ import annotations

import hashlib
from datetime import date, datetime

from .airports import Airport, great_circle_km

CABINS = ("economy", "premium_economy", "business", "first")
CABIN_LABELS = {
    "economy": "economy",
    "premium_economy": "premium economy",
    "business": "business",
    "first": "first class",
}
FARE_LABEL = "demo fare"
CURRENCY = "USD"

_CABIN_MULTIPLIER = {"economy": 1.0, "premium_economy": 1.5, "business": 2.8, "first": 4.2}
# Long flights charge relatively more for a flat bed.
_LONG_HAUL_PREMIUM = {"economy": 1.0, "premium_economy": 1.1, "business": 1.35, "first": 1.4}
_CABIN_ALIASES = {
    "economy": "economy",
    "coach": "economy",
    "main": "economy",
    "main cabin": "economy",
    "standard": "economy",
    "premium": "premium_economy",
    "premium economy": "premium_economy",
    "economy plus": "premium_economy",
    "business": "business",
    "business class": "business",
    "biz": "business",
    "first": "first",
    "first class": "first",
}


def normalize_cabin(value: object, default: str | None = "economy") -> str | None:
    """Return a cabin key for cabin words, or ``default`` when there are none or they are unknown."""
    text = str(value or "").strip().lower().replace("-", " ").replace("_", " ")
    return _CABIN_ALIASES.get(text, default) if text else default


def cabin_rank(cabin: str) -> int:
    """Return the cabin's order from economy (0) to first (3)."""
    return CABINS.index(cabin) if cabin in CABINS else 0


def _base_economy(distance_km: float) -> float:
    """Return a one-way economy base fare in USD that tapers per kilometre with distance."""
    if distance_km <= 800:
        return 49 + 0.12 * distance_km
    if distance_km <= 3000:
        return 145 + 0.075 * (distance_km - 800)
    return 310 + 0.05 * (distance_km - 3000)


def _time_of_day_factor(hour: int) -> float:
    if 6 <= hour < 9 or 16 <= hour < 20:
        return 1.12
    return 0.86 if hour >= 22 or hour < 5 else 1.0


def _days_ahead_factor(days: int) -> float:
    for limit, factor in ((3, 1.35), (7, 1.2), (14, 1.08), (30, 1.0)):
        if days < limit:
            return factor
    return 0.94


def _spread(key: str) -> float:
    """Return a stable factor between 0.93 and 1.07 for one flight."""
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    return 0.93 + (int.from_bytes(digest[:2], "big") / 65535) * 0.14


def demo_fare(
    *, origin: Airport, destination: Airport, departure: datetime, cabin: str, flight_number: str, today: date
) -> float:
    """Return a one-way demo fare per passenger in USD, rounded to whole dollars."""
    cabin = normalize_cabin(cabin) or "economy"
    distance = great_circle_km(origin, destination)
    fare = _base_economy(distance) * _CABIN_MULTIPLIER[cabin]
    if distance > 4000:
        fare *= _LONG_HAUL_PREMIUM[cabin]
    fare *= _time_of_day_factor(departure.hour)
    fare *= _days_ahead_factor(max(0, (departure.date() - today).days))
    fare *= _spread(f"{flight_number}|{departure.date().isoformat()}")
    return float(round(fare))


def fare_payload(amount: float, cabin: str) -> dict[str, object]:
    """Return the ``fare`` object of a flight."""
    return {"amount": float(amount), "currency": CURRENCY, "cabin": cabin, "label": FARE_LABEL}
