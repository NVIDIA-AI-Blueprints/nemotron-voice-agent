# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Turn normalization: rewrite the newest caller turn so spoken details reach the talker in written form.

Speech recognition writes what it heard: "m i a underscore l i three six six eight", "zero bags", "Lee, that's
L I". The normalizer reads the same conversation the talker reads (earlier turns and backend results) and rewrites
only the newest caller turn: spelled IDs joined, numbers as digits, a spelling that corrects a word applied, and a
value that clearly matches one already in the conversation written as it appears there. The rewritten turn
replaces the transcript in the conversation, so the talker and the backend both see it.

It runs only on turns that carry details (digits, number words, spelled letters, spoken symbols) unless configured
for every turn. A failure, a timeout or an implausible rewrite keeps the turn as heard. Its instructions, and the label
in front of the turn it rewrites, are the prompt set's ``normalizer_prompt``.
"""

from __future__ import annotations

import asyncio
import re
import time

from loguru import logger

from examples.frontend_backend_live.config_manager.schema import NormalizerConfig
from examples.frontend_backend_live.models.base import CallSpec, Frontend
from examples.frontend_backend_live.prompts import PromptSet

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


def has_details(text: str) -> bool:
    """Whether a turn carries details worth normalizing (digits, number words, spelled letters, spoken symbols)."""
    return bool(_DETAIL.search(text))


def plausible(original: str, rewritten: str) -> bool:
    """Whether a rewrite looks like a normalization of ``original`` rather than new content."""
    return bool(rewritten.strip()) and len(rewritten) <= MAX_GROWTH * len(original) + MAX_GROWTH_CHARS


class TurnNormalizer:
    """Rewrites the newest caller turn with the frontend model; see the module docstring."""

    def __init__(self, frontend: Frontend, prompts: PromptSet, config: NormalizerConfig):
        """Bind the model that normalizes (the talker's), the prompt set its texts come from, and the settings."""
        self.frontend, self.prompts, self.config = frontend, prompts, config
        self.spec = CallSpec("normalizer", ("speak",), prompts.call_hints.normalizer)

    def wanted(self, text: str) -> bool:
        """Whether this turn should be normalized."""
        return self.config.every_turn or has_details(text)

    async def normalize(self, history: list[dict], text: str) -> tuple[str, dict]:
        """Return the normalized turn and diagnostics; the turn as heard when anything goes wrong.

        Args:
            history: The talker's voice history, ending with the caller's newest turn.
            text: The caller's newest turn as transcribed.
        """
        prompt = self.prompts.normalizer_prompt
        context = list(history)
        if context and context[-1]["role"] == "user" and context[-1]["content"] == text:
            context.pop()
        context.append({"role": "user", "content": prompt.newest_turn + text})
        started = time.monotonic()
        try:
            async with asyncio.timeout(self.config.timeout_seconds):
                decision = await self.frontend.complete(prompt.instructions, context, self.spec)
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
