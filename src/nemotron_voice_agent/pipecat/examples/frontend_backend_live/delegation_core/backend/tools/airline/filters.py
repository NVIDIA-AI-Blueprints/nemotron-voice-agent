# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Filters for ``flights_search``: part of the day and a departure-time window.

The backend model passes ``time_of_day``, ``depart_after``, ``depart_before``; this module reads them
the same way every time, leaves what it cannot read for ``flights_search`` to ask about again, and decides which
flights match. Times are the flight's own local departure time on the searched date. A window is half open, so
"before 12:00" leaves out a noon departure, as the part-of-day windows do.

A search covers one date, so a flight in the small hours of that date belongs to the night before. "Overnight" (a
red-eye) is therefore a late departure on the searched date, "before midnight" is the end of that day, and a window
whose start is later than its end ("after 10 pm, before 2 am") runs from its start to the end of the day.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from .speech import spoken_time

# The minute after the last minute of the day: "before midnight" as a before-bound.
END_OF_DAY = 24 * 60

# Part of the day -> (first minute, minute after the last), counted from local midnight.
TIME_OF_DAY = {
    "morning": (5 * 60, 12 * 60),
    "afternoon": (12 * 60, 18 * 60),
    "evening": (18 * 60, END_OF_DAY),
    "overnight": (21 * 60, END_OF_DAY),
}
_PART_WORDS = {
    "morning": "morning",
    "mornings": "morning",
    "afternoon": "afternoon",
    "afternoons": "afternoon",
    "evening": "evening",
    "evenings": "evening",
    "night": "evening",
    "tonight": "evening",
    "overnight": "overnight",
    "redeye": "overnight",
}
_PART_WORDS_RE = re.compile(r"\b(?:mornings?|afternoons?|evenings?|tonight|night|overnight|redeye|red[\s-]+eye)\b")
_NO_TIME_PREFERENCE = frozenset({"any", "anytime", "any time", "all day", "all", "none", "no preference"})
# Before-bounds are looked for first: "no later than" and "not after" hold the words "later than" and "after".
_BEFORE_RE = re.compile(
    r"\b(?:no later than|not after|at or before|by|until|till|(?<!not )before|(?<!no )earlier than)\b"
)
_AFTER_RE = re.compile(r"\b(?:no earlier than|not before|at or after|after|past|later than|starting)\b")
_CLOCK_NOISE_RE = re.compile(
    r"\b(?:depart(?:ing|ure|s)?|leav(?:e|es|ing)|flights?|after|before|from|at|by|around|about|past|until|till|"
    r"starting|later|earlier|than|no|not|or|in|the|a|an|hours?|o'?clock|local)\b"
)
_MERIDIEM_RE = re.compile(r"\b([ap])\.\s?m\b\.?")
_CLOCK_RE = re.compile(r"^(\d{1,2})(?:[:.](\d{2}))?\s*(?:([ap])\.?m\.?)?$")
_FOUR_DIGITS_RE = re.compile(r"^(\d{2})(\d{2})$")
_BARE_HOUR_RE = re.compile(r"^(\d{1,2})(?:[:.](\d{2}))?$")


def _clean_clock(text: object) -> str:
    """Return a time phrase lowered, with "p.m." as "pm" and the words around the time removed."""
    lowered = _MERIDIEM_RE.sub(r"\1m", " ".join(str(text or "").lower().split()))
    return " ".join(_CLOCK_NOISE_RE.sub(" ", lowered).split())


def _minutes(hour: int, minute: int) -> int | None:
    return hour * 60 + minute if 0 <= hour <= 23 and 0 <= minute <= 59 else None


def parse_clock(text: object) -> int | None:
    """Return a time as minutes after midnight, or None when it is not a clear time.

    Reads "15:00", "09:30", "3 pm", "3:30 p.m.", "7.30 pm", "after 3 pm", "noon", "midnight", "12 midnight" and "1500".
    A time without am or pm must be in 24-hour form: an hour from 13 to 23 ("15:00", "15"), or two hour digits
    ("09:30"), or noon written "12:00". "3" or "3:00" could be morning or afternoon, so they are refused.
    """
    cleaned = _clean_clock(text)
    if cleaned in {"noon", "midday", "12 noon"}:
        return 12 * 60
    if cleaned in {"midnight", "12 midnight"}:
        return 0
    four = _FOUR_DIGITS_RE.match(cleaned)
    if four:
        return _minutes(int(four.group(1)), int(four.group(2)))
    match = _CLOCK_RE.match(cleaned)
    if not match:
        return None
    hour_text, meridiem = match.group(1), match.group(3)
    hour, minute = int(hour_text), int(match.group(2) or 0)
    if meridiem:
        return _minutes(hour % 12 + (12 if meridiem == "p" else 0), minute) if 1 <= hour <= 12 else None
    if hour >= 13 or (match.group(2) is not None and (hour == 0 or len(hour_text) == 2)):
        return _minutes(hour, minute)
    return None


def format_clock(minutes: int) -> str:
    """Return minutes after midnight as 24-hour "HH:MM"; the end of the day is "24:00"."""
    return f"{minutes // 60:02d}:{minutes % 60:02d}"


def _spoken_clock(minutes: int) -> str:
    return "midnight" if minutes >= END_OF_DAY else spoken_time(f"2000-01-01T{format_clock(minutes)}")


def _part_of_day(text: str) -> str | None:
    words = re.findall(r"[a-z]+", text.lower())
    parts = {_PART_WORDS[word] for word in words if word in _PART_WORDS}
    if "red" in words and "eye" in words:
        parts.add("overnight")
    return next(iter(parts)) if len(parts) == 1 else None


def _bound_name(text: str) -> str:
    """Return "depart_before" or "depart_after" for the bound a phrase names, else ""."""
    if _BEFORE_RE.search(text):
        return "depart_before"
    return "depart_after" if _AFTER_RE.search(text) else ""


def _as_bound(name: str, minutes: int) -> int:
    """Return ``minutes`` as a bound: midnight as a before-bound is the end of the day."""
    return END_OF_DAY if name == "depart_before" and minutes == 0 else minutes


def _clock_with_part(text: str, part: str | None) -> int | None:
    """Read the time in a phrase that may also name a part of the day ("evening after 7 pm", "after 7 tonight").

    A bare hour from 1 to 11 takes its am or pm from the part of the day: "evening after 7" is 19:00.
    """
    rest = _PART_WORDS_RE.sub(" ", text)
    minutes = parse_clock(rest)
    if minutes is not None or part is None:
        return minutes
    bare = _BARE_HOUR_RE.match(_clean_clock(rest))
    if not bare or not 1 <= int(bare.group(1)) <= 11:
        return None
    return parse_clock(f"{_clean_clock(rest)} {'am' if part == 'morning' else 'pm'}")


def _bound(arguments: Mapping[str, Any], name: str) -> int | None:
    """Read a 24-hour "HH:MM" bound; "00:00" and "24:00" as a before-bound are the end of the day."""
    value = arguments.get(name)
    if value in (None, ""):
        return None
    text = str(value).strip()
    if name == "depart_before" and text == "24:00":
        return END_OF_DAY
    match = re.fullmatch(r"(\d{2}):(\d{2})", text)
    if not match or int(match.group(1)) > 23 or int(match.group(2)) > 59:
        raise ValueError(f"invalid {name}")
    return _as_bound(name, int(match.group(1)) * 60 + int(match.group(2)))


def _tighten(arguments: dict[str, Any], name: str, minutes: int) -> None:
    """Set the bound ``name``, keeping the narrower of it and one the caller of the tool already sent."""
    try:
        current = _bound(arguments, name)
    except ValueError:
        return  # the value cannot be read; flights_search asks about it
    if current is not None:
        minutes = max(current, minutes) if name == "depart_after" else min(current, minutes)
    arguments[name] = format_clock(minutes)


def normalize_time_arguments(arguments: dict[str, Any]) -> None:
    """Rewrite the spellings of the three time filters in place.

    "morning flights" becomes ``time_of_day`` "morning", "after 3 pm" becomes ``depart_after`` "15:00", "before noon"
    becomes ``depart_before`` "12:00", and "before midnight" the end of the day ("24:00"). A ``time_of_day`` that is
    really a time bound ("after 3 pm") is moved to the matching bound; one that names a part of the day and a bound
    keeps both, so "evening after 7 pm" is 19:00 to the end of the day. A value that stays unreadable is left as it
    is, and ``flights_search`` asks for the time again.
    """
    for name in ("depart_after", "depart_before"):
        raw = arguments.get(name)
        minutes = parse_clock(raw) if isinstance(raw, str) else None
        if minutes is not None:
            arguments[name] = format_clock(_as_bound(name, minutes))
    value = arguments.get("time_of_day")
    if not isinstance(value, str):
        return
    lowered = " ".join(value.lower().split())
    part, bound = _part_of_day(lowered), _bound_name(lowered)
    if lowered in _NO_TIME_PREFERENCE:
        del arguments["time_of_day"]
    elif bound:
        minutes = _clock_with_part(lowered, part)
        if minutes is None:
            return  # "evening after seven": the time cannot be read, so it is asked again, not dropped
        if part:
            arguments["time_of_day"] = part
        else:
            del arguments["time_of_day"]
        _tighten(arguments, bound, _as_bound(bound, minutes))
    elif part:
        arguments["time_of_day"] = part


@dataclass(frozen=True, slots=True)
class FlightFilters:
    """What a caller asked to narrow a search to; an empty filter keeps every flight."""

    time_of_day: str | None = None
    after: int | None = None
    before: int | None = None

    @classmethod
    def from_arguments(cls, arguments: Mapping[str, Any]) -> FlightFilters:
        """Read the time filters of a ``flights_search`` call; ``ValueError`` names one that is not valid.

        ``depart_after`` and ``depart_before`` must be 24-hour "HH:MM" (``depart_before`` may be "24:00") and
        ``time_of_day`` one of morning, afternoon, evening or overnight.
        """
        part = arguments.get("time_of_day")
        if part not in (None, "") and str(part).strip().lower() not in TIME_OF_DAY:
            raise ValueError("invalid time_of_day")
        return cls(
            time_of_day=str(part).strip().lower() if part not in (None, "") else None,
            after=_bound(arguments, "depart_after"),
            before=_bound(arguments, "depart_before"),
        )

    @property
    def active(self) -> bool:
        """Whether any filter is set."""
        return bool(self.time_of_day or self.after is not None or self.before is not None)

    def _window(self) -> tuple[int | None, int | None]:
        """Return the departure window applied: a start later than the end runs to the end of the day."""
        if self.after is not None and self.before is not None and self.after > self.before:
            return self.after, None
        return self.after, self.before

    def matches(self, flight: Mapping[str, Any]) -> bool:
        """Whether a flight leaves in the time filters' window."""
        stamp = datetime.fromisoformat(flight["depart_local"])
        minute = stamp.hour * 60 + stamp.minute
        if self.time_of_day:
            start, end = TIME_OF_DAY[self.time_of_day]
            if not start <= minute < end:
                return False
        after, before = self._window()
        return (after is None or minute >= after) and (before is None or minute < before)

    def kind_phrase(self) -> str:
        """Return the words that go before "nonstop flights", such as "afternoon "."""
        return f"{self.time_of_day} " if self.time_of_day else ""

    def window_phrase(self) -> str:
        """Return the departure window as words that follow "flights", such as "leaving after 3:00 PM "."""
        after, before = self._window()
        if after is not None and before == END_OF_DAY:
            before = None  # "between 7:00 PM and midnight" is "after 7:00 PM" on a one-day search
        if after is not None and before is not None:
            return f"leaving between {_spoken_clock(after)} and {_spoken_clock(before)} "
        if after is not None:
            return f"leaving after {_spoken_clock(after)} "
        if before is not None:
            return f"leaving before {_spoken_clock(before)} "
        return ""

    def description(self) -> str:
        """Return the filters in a few words, such as "afternoon leaving after 3:00 PM"."""
        return f"{self.kind_phrase()}{self.window_phrase()}".strip()

    def as_data(self) -> dict[str, Any]:
        """Return the filters for the tool result; only the ones that are set."""
        data: dict[str, Any] = {}
        if self.time_of_day:
            data["time_of_day"] = self.time_of_day
        if self.after is not None:
            data["depart_after"] = format_clock(self.after)
        if self.before is not None:
            data["depart_before"] = format_clock(self.before)
        return data
