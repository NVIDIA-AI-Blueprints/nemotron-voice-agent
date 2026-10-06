# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic speech stand-ins for the tau3 gates, smoke runs and tests.

``StubRecognizer`` returns ``"utterance N"`` per utterance; ``ToneSynthesizer``
makes a tone whose length is proportional to the text. With the energy VAD they
let the real engine (audio clock, segmenter, turn manager, barge-in) run end to
end with no GPU and no network (plan section 18.0, Gate B).
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable

from examples.frontend_backend_verdict.voice.audio.pcm import tone


class _StubStream:
    def __init__(self, transcript: str, on_interim: Callable[[str], None] | None) -> None:
        self._transcript = transcript
        self._on_interim = on_interim
        self.pushed = 0
        self.cancelled = False

    async def push(self, pcm: bytes) -> None:
        self.pushed += len(pcm)

    async def finish(self) -> str:
        if self._on_interim is not None:
            self._on_interim(self._transcript)
        return self._transcript

    async def cancel(self) -> None:
        self.cancelled = True


class StubRecognizer:
    """Returns ``"<prefix> N"`` for the N-th utterance, or scripted transcripts in order."""

    def __init__(self, transcripts: list[str] | None = None, *, prefix: str = "utterance") -> None:
        """Use ``transcripts`` in order, then fall back to numbered utterances."""
        self._scripted = list(transcripts or [])
        self._prefix = prefix
        self.count = 0
        self.streams: list[_StubStream] = []

    def open(self, *, sample_rate: int, on_interim: Callable[[str], None] | None = None) -> _StubStream:
        """Start a stream for one utterance."""
        self.count += 1
        text = self._scripted.pop(0) if self._scripted else f"{self._prefix} {self.count}"
        stream = _StubStream(text, on_interim)
        self.streams.append(stream)
        return stream

    async def warmup(self) -> None:
        """Nothing to probe."""

    async def aclose(self) -> None:
        """Nothing to release."""


class ToneSynthesizer:
    """Tone audio of ``ms_per_char`` milliseconds per character, in ``chunk_ms`` chunks."""

    def __init__(
        self,
        *,
        sample_rate: int = 16000,
        ms_per_char: float = 20.0,
        chunk_ms: int = 100,
        delay_s: float = 0.0,
        gate: asyncio.Event | None = None,
    ) -> None:
        """Configure the stand-in; ``gate`` (tests) blocks synthesis until set."""
        self._rate = sample_rate
        self._ms_per_char = ms_per_char
        self._chunk_ms = chunk_ms
        self._delay_s = delay_s
        self.gate = gate
        self.requests: list[str] = []
        self.cancelled: list[str] = []

    @property
    def sample_rate(self) -> int:
        """Output sample rate."""
        return self._rate

    async def synthesize(self, text: str, *, voice: str | None = None) -> AsyncIterator[bytes]:
        """Yield tone chunks for ``text``."""
        self.requests.append(text)
        total_ms = max(self._chunk_ms, len(text) * self._ms_per_char)
        try:
            if self.gate is not None:
                await self.gate.wait()
            if self._delay_s:
                await asyncio.sleep(self._delay_s)
            sent = 0.0
            while sent < total_ms:
                step = min(self._chunk_ms, total_ms - sent)
                yield tone(step, self._rate)
                sent += step
                await asyncio.sleep(0)
        except (asyncio.CancelledError, GeneratorExit):
            self.cancelled.append(text)
            raise

    async def warmup(self) -> None:
        """Nothing to probe."""

    async def aclose(self) -> None:
        """Nothing to release."""
