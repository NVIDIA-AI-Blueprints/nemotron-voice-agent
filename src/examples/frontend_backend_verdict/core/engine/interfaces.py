# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The speech-output interface the turn engine uses (type-only protocols).

``TurnManager`` annotates its output object with these protocols instead of importing an
adapter's implementation, so ``core/`` depends on no adapter. The members are exactly the
ones the turn manager uses; the Pipecat adapter's speech output satisfies them.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

from examples.frontend_backend_verdict.core.audio.formats import AudioFormat
from examples.frontend_backend_verdict.core.engine.playback import ResponseProgress
from examples.frontend_backend_verdict.core.engine.response_writer import MessageItemWriter


class PreparedSpeech(Protocol):
    """One item's text, split into sentences and possibly already synthesizing."""

    text: str
    kind: str

    async def aclose(self) -> None:
        """Stop any synthesis still running."""
        ...


class OutputPath(Protocol):
    """Turns an item's text into transcript and audio deltas on its item writer."""

    def prepare(
        self, text: str, *, kind: str, modality: str, voice: str | None, out_format: AudioFormat
    ) -> PreparedSpeech:
        """Split ``text`` and, for audio, start synthesizing its first sentence now."""
        ...

    async def speak(
        self,
        prepared: PreparedSpeech,
        item: MessageItemWriter,
        progress: ResponseProgress,
        *,
        out_format: AudioFormat,
        on_first_audio: Callable[[], None] | None = None,
    ) -> None:
        """Send one item's transcript and audio; cancellation leaves the item open for the caller to close."""
        ...
