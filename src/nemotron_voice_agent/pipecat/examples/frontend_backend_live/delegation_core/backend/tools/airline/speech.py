# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Spoken forms of dates, times, prices and codes, so a result carries words the voice can say as written."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date, datetime

_DIGIT_NAMES = ("zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine")


def ordinal(number: int) -> str:
    """Return ``number`` with its English ordinal suffix, such as 5th."""
    suffix = "th" if 10 <= number % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(number % 10, "th")
    return f"{number}{suffix}"


def as_date(value: date | str) -> date:
    """Return a date, or the date of an ISO date or date-time string."""
    return date.fromisoformat(value[:10]) if isinstance(value, str) else value


def spoken_date(value: date | str, today: date | None = None) -> str:
    """Return "November 5th", with the year when it is not the year of ``today``."""
    day = as_date(value)
    text = f"{day:%B} {ordinal(day.day)}"
    return text if today is None or day.year == today.year else f"{text}, {day.year}"


def spoken_day(value: date | str, today: date | None = None) -> str:
    """Return "Thursday, November 5th", with the year when it is not the year of ``today``."""
    day = as_date(value)
    return f"{day:%A}, {spoken_date(day, today)}"


def spoken_time(iso_timestamp: str | None) -> str:
    """Return a 12-hour spoken time such as "7:15 AM" for an ISO timestamp, or "" when there is none."""
    if not iso_timestamp:
        return ""
    try:
        stamp = datetime.fromisoformat(iso_timestamp)
    except ValueError:
        return iso_timestamp
    return f"{int(stamp.strftime('%I'))}:{stamp:%M} {stamp:%p}"


def spoken_amount(amount: float) -> str:
    """Return a price as people say it, such as "$214"."""
    return f"${amount:,.0f}"


def spoken_duration(minutes: int | None) -> str:
    """Return "5 hours 5 minutes", "45 minutes" or "2 hours" for a flight time, or "" for none."""
    if not minutes or minutes <= 0:
        return ""
    hours, rest = divmod(int(minutes), 60)
    parts = [f"{hours} hour{'s' if hours != 1 else ''}"] if hours else []
    if rest:
        parts.append(f"{rest} minute{'s' if rest != 1 else ''}")
    return " ".join(parts)


def spoken_code(code: str) -> str:
    """Return a booking code one character at a time: "K7Q2ZP" is "K 7 Q 2 Z P"."""
    return " ".join(code)


def spoken_user_id(user_id: str) -> str:
    """Return a user id as it is said, digit by digit: "jane_doe_001" is "jane doe zero zero one"."""
    words = []
    for part in user_id.split("_"):
        words.append(" ".join(_DIGIT_NAMES[int(char)] for char in part) if part.isdigit() else part)
    return " ".join(words)


def and_list(items: Sequence[str]) -> str:
    """Join ``items`` as "a, b and c"."""
    return "".join(items) if len(items) < 2 else f"{', '.join(items[:-1])} and {items[-1]}"
