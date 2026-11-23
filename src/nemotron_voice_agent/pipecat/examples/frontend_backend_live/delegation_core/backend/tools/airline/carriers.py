# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The desk's one airline, Demo Air, and how its flight numbers are read and said.

Demo Air is a fictional carrier made up for this demo: it is not a real company, and its code ``DA`` is used only
here. Every flight of the desk is a Demo Air flight, numbered 100 to 755.
"""

from __future__ import annotations

import re
from datetime import UTC, tzinfo
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

CODE = "DA"
NAME = "Demo Air"
# How the airline is said in a sentence: "Demo Air flight two fourteen".
SHORT_NAME = "Demo Air"

# The desk's day is the date in Central time, the zone of its Chicago hub, so "today" is the U.S. day and not the UTC
# date, which runs ahead every evening. Without time zone data it is the UTC date.
try:
    HOME_ZONE: tzinfo = ZoneInfo("America/Chicago")
except ZoneInfoNotFoundError:
    HOME_ZONE = UTC

# Spoken digits, as a caller or the speech recogniser says them ("one oh seven" is 107).
DIGIT_WORDS = {
    "zero": "0",
    "oh": "0",
    "o": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
}
_ONES = ("zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine")
_TEENS = (
    "ten",
    "eleven",
    "twelve",
    "thirteen",
    "fourteen",
    "fifteen",
    "sixteen",
    "seventeen",
    "eighteen",
    "nineteen",
)
_TENS = ("", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety")

# A code and a number written together or apart ("DA 214", "DA214", "da-214", "6 E 107"), with an optional letter.
_NUMBER_RE = re.compile(r"^\s*([A-Z0-9]{2})\s*-?\s*(\d{1,4})\s*([A-Z]?)\s*$")
_SPACED_RE = re.compile(r"^([A-Z0-9])\s+([A-Z0-9])\s*-?\s*(\d{1,4})$")
_BARE_RE = re.compile(r"^(\d{2,4})$")
_NAME_RE = re.compile(r"^DEMO(?:\s*AIR(?:LINES?)?)?\s*(?:FLIGHT)?\s*")
_FILLER_RE = re.compile(r"\b(?:FLIGHT|NUMBER|NO)\b")


def airline_name(code: str | None) -> str | None:
    """Return the airline's name for its code, or None for any other code."""
    return NAME if (code or "").strip().upper() == CODE else None


def _parse_number(text: str) -> str | None:
    """Return "DA 214" for code and digits written together or apart, else None."""
    spaced = _SPACED_RE.match(text)
    if spaced:
        text = f"{spaced.group(1)}{spaced.group(2)} {spaced.group(3)}"
    match = _NUMBER_RE.match(text)
    if not match or not any(char.isalpha() for char in match.group(1)):
        return None
    return f"{match.group(1)} {int(match.group(2))}{match.group(3)}"


def normalize_flight_number(value: object) -> str | None:
    """Return a flight number as "DA 214", or None when ``value`` is not one.

    Accepts "DA 214", "da214", "DA-214", "Demo Air 214", "Demo Air flight 214", "flight 214" and "214", which
    are all Demo Air flights. A number with another airline's code ("UA 1534") is returned as written, so the desk can
    say it is not a Demo Air flight.
    """
    text = " ".join(_FILLER_RE.sub(" ", str(value or "").strip().upper()).split())
    bare = _BARE_RE.match(_NAME_RE.sub("", text))
    if bare:
        return f"{CODE} {int(bare.group(1))}"
    return _parse_number(text)


def is_demo_air(flight_number: str) -> bool:
    """Whether a normalized flight number ("DA 214") is one of the airline's."""
    return flight_number.split(" ")[0] == CODE


def spoken_number(digits: str) -> str:
    """Return a three-digit flight number as an announcer says it: 214 "two fourteen", 305 "three oh five"."""
    if not (digits.isdigit() and len(digits) == 3):
        return digits
    hundreds, rest = divmod(int(digits), 100)
    return f"{_ONES[hundreds]} {_hundred_rest(rest)}"


def _teen_or_tens(value: int) -> str:
    if value < 20:
        return _TEENS[value - 10]
    tens, ones = divmod(value, 10)
    return _TENS[tens] if ones == 0 else f"{_TENS[tens]}-{_ONES[ones]}"


def _hundred_rest(rest: int) -> str:
    """Return the last two digits of a three-digit number: 0 "hundred", 5 "oh five", 14 "fourteen", 20 "twenty"."""
    if rest == 0:
        return "hundred"
    if rest < 10:
        return f"oh {_ONES[rest]}"
    return _teen_or_tens(rest)


def spoken_flight(flight_number: str, airline: str | None = None, *, with_code: bool = False) -> str:
    """Return a flight as people say it: "Demo Air flight two fourteen".

    A number of another airline keeps its code (``with_code``) or its name (``airline``), so the caller hears what was
    looked up.
    """
    code, _, digits = flight_number.partition(" ")
    if not digits:
        return flight_number
    if airline_name(code):
        return f"{SHORT_NAME} flight {spoken_number(digits)}"
    return f"{code} {digits}" if with_code else f"{airline or code} flight {digits}"
