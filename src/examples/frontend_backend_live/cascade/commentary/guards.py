# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic guards on what the assistant says: lead-ins, answers and plain speech.

Prompts ask the model not to claim an outcome before the backend returns one, but that is a
probabilistic instruction. These checks are the structural floor beneath it.
"""

import re

# A delegate lead-in should only say work is starting, never report its result. The routing prompt says so,
# but a prompt is a probabilistic instruction, not a guarantee, so this is the fallback for whatever it misses.
# Phrases are lowercase substrings that deliberately err toward over-flagging: a false positive costs one bland
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
    """The answer's words without markup: lines become sentences, bullets and emphasis go."""
    lines = []
    for line in text.splitlines():
        line = re.sub(r"^\s*(?:[-*\u2022]+|\d+[.)]|#+)\s*", "", line)
        line = re.sub(r"[*_`]+", "", line).strip()
        if line:
            lines.append(line if line[-1] in ".!?:" else line + ".")
    return re.sub(r"\s+", " ", " ".join(lines).replace(":.", ":")).strip()
