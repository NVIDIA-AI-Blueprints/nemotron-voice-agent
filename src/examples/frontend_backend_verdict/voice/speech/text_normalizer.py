# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TTS text normalization, reusing the official example's Nemotron Speech filter.

Only the synthesizer input is normalized; transcript deltas keep the agent's
original text, so the scored transcript matches the conversation history.
"""

from __future__ import annotations

from examples.shared.nemotron_speech_text_filter import NemotronSpeechTextFilter


class NemotronTextNormalizer:
    """Strips characters reserved by the NVIDIA TTS preprocessor (``*``, ``{}``, SSML tag openers)."""

    def __init__(self) -> None:
        """Build the shared filter."""
        self._filter = NemotronSpeechTextFilter()

    async def normalize(self, text: str) -> str:
        """Return synthesizer-safe text."""
        return await self._filter.filter(text)
