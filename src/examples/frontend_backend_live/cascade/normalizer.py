# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Turn normalization: rewrite the newest caller turn so spoken details reach the talker in written form.

Speech recognition writes what it heard: "m i a underscore l i three six six eight", "zero bags", "Lee, that's
L I". The normalizer reads the same conversation the talker reads (earlier turns and backend results) and rewrites
only the newest caller turn: spelled IDs joined, numbers as digits, a spelling that corrects a word applied, and a
value that clearly matches one already in the conversation written as it appears there. The rewritten turn
replaces the transcript in the conversation, so the talker and the backend both see it.

It runs only on turns that carry details (digits, number words, spelled letters, spoken symbols) unless configured
for every turn. A failure, a timeout or an implausible rewrite keeps the turn as heard.
"""

from __future__ import annotations

import asyncio
import re
import time

from loguru import logger

from examples.frontend_backend_live.config_manager.schema import NormalizerConfig
from examples.frontend_backend_live.models.base import Frontend

_NUMBER_WORDS = (
    r"zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen"
    r"|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|thousand"
    r"|first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|\w+teenth|\w+tieth"
)
_DETAIL = re.compile(
    rf"""\d
    | \b({_NUMBER_WORDS})\b
    | \b[a-z]\b[\s,.\-]+\b[a-z]\b             # two or more spelled letters
    | \b(underscore|hyphen|dash|dot|at\s+sign|spell(ed|ing)?|as\s+in)\b
    """,
    re.I | re.X,
)
# A rewrite this much longer than the turn added content instead of normalizing it.
MAX_GROWTH = 2.0
MAX_GROWTH_CHARS = 40

NORMALIZER_INSTRUCTIONS = """You clean up speech-recognition transcripts for a phone assistant. You do not talk
to the caller. You rewrite only the caller's newest turn so that the details in it are written the way a computer
system stores them, using the conversation so far as context.

Rules:
1. Join spelled letters and digits into one token, keeping the caller's order. Spoken symbols become characters:
   "underscore" -> "_", "dash" or "hyphen" -> "-", and in an email "at" -> "@" and "dot" -> ".".
   "m i a underscore l i three six six eight" -> "mia_li_3668"; "H A T zero four five" -> "HAT045".
2. Numbers become digits: "zero bags" -> "0 bags", "twenty three" -> "23", "May twentieth" -> "May 20".
   "oh" inside a spoken number is zero.
3. When the caller spells a word, the spelling wins over how the word was transcribed: "Lee, that's L I" -> "Li".
4. When a value is clearly the same as one already in the conversation (an ID, a name, a code or an airport from
   earlier turns or from reference data) apart from an obvious transcription error, write it exactly as it appears
   there. Only when the match is clear; never pick between two candidates.
5. When the caller corrects a value in this turn, keep the correction exactly as stated.
6. Keep everything else word for word. Do not answer, add, drop, summarize or explain, and never invent a value
   the caller did not say. If nothing needs normalizing, return the turn unchanged.

Put the rewritten turn in "speech" with "action": "speak"."""

NEWEST_TURN = "Caller's newest turn (rewrite only this):\n"


def has_details(text: str) -> bool:
    """Whether a turn carries details worth normalizing (digits, number words, spelled letters, spoken symbols)."""
    return bool(_DETAIL.search(text))


def plausible(original: str, rewritten: str) -> bool:
    """Whether a rewrite looks like a normalization of ``original`` rather than new content."""
    return bool(rewritten.strip()) and len(rewritten) <= MAX_GROWTH * len(original) + MAX_GROWTH_CHARS


class TurnNormalizer:
    """Rewrites the newest caller turn with the frontend model; see the module docstring."""

    def __init__(self, frontend: Frontend, config: NormalizerConfig):
        """Bind the model that normalizes (the talker's) and the settings."""
        self.frontend, self.config = frontend, config

    def wanted(self, text: str) -> bool:
        """Whether this turn should be normalized."""
        return self.config.every_turn or has_details(text)

    async def normalize(self, history: list[dict], text: str) -> tuple[str, dict]:
        """Return the normalized turn and diagnostics; the turn as heard when anything goes wrong.

        Args:
            history: The talker's voice history, ending with the caller's newest turn.
            text: The caller's newest turn as transcribed.
        """
        context = list(history)
        if context and context[-1]["role"] == "user" and context[-1]["content"] == text:
            context.pop()
        context.append({"role": "user", "content": NEWEST_TURN + text})
        started = time.monotonic()
        try:
            async with asyncio.timeout(self.config.timeout_seconds):
                decision = await self.frontend.complete(
                    NORMALIZER_INSTRUCTIONS, context, actions=("speak",), max_output_tokens=300
                )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning(f"Normalizer failed ({type(exc).__name__}: {exc}); keeping the turn as heard")
            return text, {"outcome": "failed", "error_type": type(exc).__name__, "ms": _ms(started)}
        rewritten = decision.speech.strip() if decision.action == "speak" else ""
        if not plausible(text, rewritten):
            return text, {"outcome": "rejected", "rewrite": rewritten, "ms": _ms(started)}
        if rewritten == text.strip():
            return text, {"outcome": "unchanged", "ms": _ms(started)}
        return rewritten, {"outcome": "changed", "ms": _ms(started)}


def _ms(started: float) -> int:
    return round((time.monotonic() - started) * 1000)
