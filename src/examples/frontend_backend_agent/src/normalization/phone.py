# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""Ten-digit phone number forms, for retrying a read lookup in the other common spelling.

Strict on purpose: a value with any character other than digits, spaces, dashes,
dots and parentheses, or with other than exactly ten digits (so no ``+1`` country
code), has no alternative form.
"""

from __future__ import annotations

import re

_PHONE_CHARS = re.compile(r"[0-9\s().-]+")
_DASHED = re.compile(r"[0-9]{3}-[0-9]{3}-[0-9]{4}")
_PHONE_DIGITS = 10


def ten_digits(value: object) -> str | None:
    """The bare digits of a ten-digit phone number, or ``None``."""
    if not isinstance(value, str) or not _PHONE_CHARS.fullmatch(value):
        return None
    digits = "".join(char for char in value if char.isdigit())
    return digits if len(digits) == _PHONE_DIGITS else None


def other_phone_form(value: str) -> str | None:
    """``"XXX-XXX-XXXX"`` -> bare digits; any other ten-digit form -> ``"XXX-XXX-XXXX"``; else ``None``."""
    digits = ten_digits(value)
    if digits is None:
        return None
    if _DASHED.fullmatch(value.strip()):
        return digits
    return f"{digits[:3]}-{digits[3:6]}-{digits[6:]}"
