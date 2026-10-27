# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic guards beneath the prompts on what the assistant says: lead-ins, answers and plain speech."""

import re

# Outcomes a delegate lead-in must not claim. The substrings err toward over-flagging: a false positive costs one bland
# sentence, a false negative tells the caller something happened that did not.
_OUTCOME_CLAIMS = (
    "is confirmed",
    "has been confirmed",
    "'s confirmed",
    "is booked",
    "has been booked",
    "i've booked",
    "i have booked",
    "is cancelled",
    "is canceled",
    "has been cancelled",
    "has been canceled",
    "is reserved",
    "has been reserved",
    "i've reserved",
    "i have reserved",
    "is all set",
    "you're all set",
    "all set",
    "is done",
    "has been done",
)


SAFE_LEAD_IN = "One moment, let me take care of that."


def asserts_completion(text: str) -> bool:
    """True if `text` claims an outcome that cannot be verified yet (see `_OUTCOME_CLAIMS`)."""
    low = text.lower()
    return any(marker in low for marker in _OUTCOME_CLAIMS)


def safe_lead_in(speech: str) -> str:
    """A delegate lead-in, replaced with a neutral one if it asserts an unverified outcome."""
    if asserts_completion(speech):
        return SAFE_LEAD_IN
    kept = []
    for sentence in re.findall(r"[^.!?]+[.!?]?(?:[\"\”\’])?", speech):
        if "?" in sentence:
            break
        kept.append(sentence)
    return "".join(kept).strip() or (SAFE_LEAD_IN if speech.strip() else "")


def plain_speech(text: str) -> str:
    """The answer's words without markup: lines become sentences, bullets and emphasis go; numbers and ids stay.

    A list marker goes only when a space follows it, so "12.50 is your total" and "-5 degrees" keep their numbers, and
    underscores stay, so an id such as mia_li_3668 is said as the backend wrote it.
    """
    lines = []
    for line in text.splitlines():
        line = re.sub(r"^\s*(?:[-*\u2022]+|\d+[.)]|#+)\s+", "", line)
        line = re.sub(r"[*`]+", "", line).strip()
        if line:
            lines.append(line if line[-1] in ".!?:" else line + ".")
    return re.sub(r"\s+", " ", " ".join(lines).replace(":.", ":")).strip()
