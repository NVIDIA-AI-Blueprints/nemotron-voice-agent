# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Turn normalization: rewrite the newest caller turn so spoken details reach the talker in written form.

Speech recognition writes what it heard: an ID spelled letter by letter, numbers as words, a name then its spelling.
The normalizer reads the same conversation the talker reads (earlier turns and backend results) and rewrites
only the newest caller turn: spelled IDs joined, numbers as digits, a spelling that corrects a word applied, and a
value that clearly matches one already in the conversation written as it appears there. The rewritten turn
replaces the transcript in the conversation, so the talker and the backend both see it.

When enabled, it reads the newest caller turn whenever the caller's turns are decided, and its prompt decides what
needs rewriting. A failure, a timeout, an answer with no rewrite, or one that repeats the turn keeps the turn as heard.
Its instructions, and the label in front of the turn it rewrites, are the prompt set's ``normalizer_prompt``.
"""

from __future__ import annotations

import asyncio
import time

from loguru import logger

from examples.frontend_backend_live.config_manager.schema import NormalizerConfig
from examples.frontend_backend_live.models.base import CallSpec, Frontend
from examples.frontend_backend_live.prompts import PromptSet


class TurnNormalizer:
    """Rewrites the newest caller turn with the frontend model; see the module docstring."""

    def __init__(self, frontend: Frontend, prompts: PromptSet, config: NormalizerConfig):
        """Bind the model that normalizes (the talker's), the prompt set its texts come from, and the settings."""
        self.frontend, self.prompts, self.config = frontend, prompts, config
        self.spec = CallSpec("normalizer", ("speak",), prompts.call_hints.normalizer)

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
        if not rewritten:
            return text, {"outcome": "empty", "ms": _ms(started)}
        if rewritten == text.strip():
            return text, {"outcome": "unchanged", "ms": _ms(started)}
        return rewritten, {"outcome": "changed", "ms": _ms(started)}


def _ms(started: float) -> int:
    return round((time.monotonic() - started) * 1000)
