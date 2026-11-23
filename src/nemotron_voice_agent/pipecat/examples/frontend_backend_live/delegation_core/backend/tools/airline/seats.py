# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A simple seat map (window, aisle, exit row) and the standard special-meal codes."""

from __future__ import annotations

import hashlib
import re

SEAT_LETTERS = "ABCDEF"
WINDOW, AISLE = frozenset("AF"), frozenset("CD")
CABIN_ROWS = {
    "first": range(1, 3),
    "business": range(3, 10),
    "premium_economy": range(10, 15),
    "economy": range(15, 46),
}
EXIT_ROWS = frozenset({15, 28})
NO_MEAL = "NONE"

MEAL_NAMES = {
    "VGML": "vegan meal",
    "VLML": "vegetarian meal",
    "AVML": "Asian vegetarian meal",
    "VJML": "Jain meal",
    "HNML": "Hindu meal",
    "MOML": "halal meal",
    "KSML": "kosher meal",
    "GFML": "gluten-free meal",
    "DBML": "diabetic meal",
    "LSML": "low-salt meal",
    "LFML": "low-fat meal",
    "CHML": "child meal",
    "BLML": "bland meal",
    "SFML": "seafood meal",
    "FPML": "fruit platter",
}
# Words first, longer phrases before the shorter ones they hold; ``None`` is the standard meal.
_MEAL_WORDS: tuple[tuple[str, str | None], ...] = (
    ("non veg", None),
    ("nonveg", None),
    ("asian vegetarian", "AVML"),
    ("hindu vegetarian", "AVML"),
    ("indian vegetarian", "AVML"),
    ("jain", "VJML"),
    ("vegan", "VGML"),
    ("vegetarian", "VLML"),
    ("veg", "VLML"),
    ("hindu", "HNML"),
    ("halal", "MOML"),
    ("muslim", "MOML"),
    ("kosher", "KSML"),
    ("gluten", "GFML"),
    ("celiac", "GFML"),
    ("diabetic", "DBML"),
    ("low salt", "LSML"),
    ("low sodium", "LSML"),
    ("low fat", "LFML"),
    ("child", "CHML"),
    ("kid", "CHML"),
    ("bland", "BLML"),
    ("seafood", "SFML"),
    ("fish", "SFML"),
    ("fruit", "FPML"),
    ("standard", None),
    ("regular", None),
    ("normal", None),
    ("no special", None),
    ("none", None),
)
_SEAT_RE = re.compile(r"^\s*(?:seat\s*)?(\d{1,2})\s*-?\s*([A-Fa-f])\s*$")


def parse_meal(value: object) -> str | None:
    """Return a meal code for a code or words; ``"NONE"`` means the standard meal; None if unknown."""
    text = str(value or "").strip()
    if not text:
        return None
    if text.upper() in MEAL_NAMES:
        return text.upper()
    lowered = text.lower().replace("-", " ")
    for words, code in _MEAL_WORDS:
        if words in lowered:
            return code or NO_MEAL
    return None


def meal_label(code: str | None) -> str:
    """Return how a meal code is said."""
    return MEAL_NAMES.get(code or "", "standard meal")


def parse_seat(value: object) -> tuple[int, str] | None:
    """Return ``(row, letter)`` for an explicit seat such as "14A" or "14 a", else None."""
    match = _SEAT_RE.match(str(value or ""))
    return (int(match.group(1)), match.group(2).upper()) if match else None


def seat_kind(value: object) -> str | None:
    """Return "window", "aisle", "exit" or "any" for a seat preference in words, else None."""
    lowered = str(value or "").lower()
    if "exit" in lowered or "legroom" in lowered or "leg room" in lowered:
        return "exit"
    if "window" in lowered:
        return "window"
    if "aisle" in lowered:
        return "aisle"
    return "any" if "any" in lowered or "front" in lowered else None


def seat_type(letter: str) -> str:
    """Return window, aisle or middle for a seat letter."""
    return "window" if letter in WINDOW else "aisle" if letter in AISLE else "middle"


def seat_in_cabin(row: int, cabin: str) -> bool:
    """Return whether ``row`` belongs to ``cabin``."""
    return row in CABIN_ROWS.get(cabin, CABIN_ROWS["economy"])


def _occupied(flight_key: str, row: int, letter: str) -> bool:
    digest = hashlib.sha256(f"{flight_key}|{row}{letter}".encode()).digest()
    return digest[0] < 150  # about 59% of seats are taken


def seat_free(flight_key: str, row: int, letter: str, held: set[str]) -> bool:
    """Return whether a seat is free on a flight; ``held`` are seats this session's bookings already hold."""
    return f"{row}{letter}" not in held and not _occupied(flight_key, row, letter)


def find_seat(flight_key: str, cabin: str, kind: str, held: set[str]) -> str | None:
    """Return the frontmost free seat of ``kind`` in ``cabin``, or None."""
    rows = CABIN_ROWS.get(cabin, CABIN_ROWS["economy"])
    if kind == "exit":
        rows = [row for row in rows if row in EXIT_ROWS]
    letters = {"window": "AF", "aisle": "CD", "exit": SEAT_LETTERS, "any": "ACDFBE"}.get(kind, SEAT_LETTERS)
    for row in rows:
        for letter in letters:
            if seat_free(flight_key, row, letter, held):
                return f"{row}{letter}"
    return None
