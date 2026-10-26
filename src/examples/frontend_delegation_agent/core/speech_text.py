# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Speech text shared by every head: sentence splitting and TTS text normalization.

TTS normalization reuses the official example's Nemotron Speech filter. Only the
synthesizer input is normalized; transcript deltas keep the agent's original text,
so the scored transcript matches the conversation history.
"""

from __future__ import annotations

import re

from examples.shared.nemotron_speech_text_filter import NemotronSpeechTextFilter


class NemotronTextNormalizer:
    """Strips characters reserved by the NVIDIA TTS preprocessor (``*``, ``{}``, SSML tag openers)."""

    def __init__(self) -> None:
        """Build the shared filter."""
        self._filter = NemotronSpeechTextFilter()

    async def normalize(self, text: str) -> str:
        """Return synthesizer-safe text."""
        return await self._filter.filter(text)


_SENTENCE_END = re.compile(r"(?<=[.!?])[\"')\]]*\s+|\n+")


def split_sentences(text: str, *, enabled: bool = True) -> list[str]:
    """Split ``text`` into sentences (whitespace-normalized); one item when splitting is disabled."""
    normalized = " ".join(text.split())
    if not normalized:
        return []
    if not enabled:
        return [normalized]
    parts = [" ".join(part.split()) for part in _SENTENCE_END.split(text)]
    return [part for part in parts if part]
