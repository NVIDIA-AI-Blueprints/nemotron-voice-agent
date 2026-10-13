# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Ports for the speech components: VAD, streaming ASR, TTS, text normalization.

The engine depends only on these protocols. Riva/Silero adapters implement them
for production; deterministic stand-ins implement them for tests and the tau3
gates, so the whole engine runs offline.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@runtime_checkable
class VoiceActivityDetector(Protocol):
    """Frame-level speech probability at the engine rate."""

    @property
    def frame_samples(self) -> int:
        """Samples per analysis frame."""

    def speech_probability(self, frame: bytes) -> float:
        """Probability in [0, 1] that ``frame`` (PCM16, :attr:`frame_samples` long) contains speech."""

    def reset(self) -> None:
        """Forget model state (new session or after a format change)."""


@runtime_checkable
class RecognizerStream(Protocol):
    """One streaming recognition request, scoped to one utterance."""

    async def push(self, pcm: bytes) -> None:
        """Send PCM16 audio at the engine rate."""

    async def finish(self) -> str:
        """Close the audio stream and return the final transcript."""

    async def cancel(self) -> None:
        """Abandon the utterance and release the stream."""


@runtime_checkable
class StreamingRecognizer(Protocol):
    """Opens utterance-scoped recognition streams."""

    def open(self, *, sample_rate: int, on_interim: Callable[[str], None] | None = None) -> RecognizerStream:
        """Start a stream for one utterance."""

    async def warmup(self) -> None:
        """Verify the endpoint (raises :class:`SpeechServiceError` on failure)."""

    async def aclose(self) -> None:
        """Release shared resources."""


@runtime_checkable
class Synthesizer(Protocol):
    """Text to PCM16 audio."""

    @property
    def sample_rate(self) -> int:
        """Output sample rate of :meth:`synthesize`."""

    def synthesize(self, text: str, *, voice: str | None = None) -> AsyncIterator[bytes]:
        """Yield PCM16 chunks for ``text``; closing the iterator cancels synthesis."""

    async def warmup(self) -> None:
        """Verify the endpoint (raises :class:`SpeechServiceError` on failure)."""

    async def aclose(self) -> None:
        """Release shared resources."""


@runtime_checkable
class TextNormalizer(Protocol):
    """Makes agent text safe for the synthesizer (the transcript keeps the original)."""

    async def normalize(self, text: str) -> str:
        """Return synthesizer-safe text."""


class IdentityNormalizer:
    """No-op normalizer."""

    async def normalize(self, text: str) -> str:
        """Return ``text`` unchanged."""
        return text


VadFactory = Callable[[int], VoiceActivityDetector]


@dataclass(frozen=True, slots=True)
class SpeechServices:
    """Everything speech-related a session needs, shared across sessions."""

    recognizer: StreamingRecognizer
    synthesizer: Synthesizer
    vad_factory: VadFactory
    normalizer: TextNormalizer

    async def warmup(self) -> None:
        """Probe both endpoints."""
        await self.recognizer.warmup()
        await self.synthesizer.warmup()

    async def aclose(self) -> None:
        """Release both endpoints."""
        await self.recognizer.aclose()
        await self.synthesizer.aclose()
