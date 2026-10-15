# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""Immutable records and narrow ports shared by the offline runtime."""

from dataclasses import dataclass, field
from typing import Any, Protocol

RATE = 16000
NANOSECONDS = 1_000_000_000


def samples_to_ns(samples: int, rate: int = RATE) -> int:
    """Convert a cumulative sample position; avoid rounding each packet."""
    return samples * NANOSECONDS // rate


def ns_to_samples(nanoseconds: int, rate: int = RATE) -> int:
    """Round a presentation position forward so audio cannot precede readiness."""
    return (nanoseconds * rate + NANOSECONDS - 1) // NANOSECONDS


@dataclass(frozen=True)
class AudioSource:
    """Canonical source PCM, including silence, owned by a finite session."""

    pcm: bytes
    sample_rate: int = RATE

    def __post_init__(self):
        """Reject formats whose sample positions would be ambiguous."""
        if self.sample_rate != RATE or len(self.pcm) % 2:
            raise ValueError("Offline source must be 16 kHz mono PCM16")

    @property
    def sample_count(self) -> int:
        """Number of frames on the source timeline."""
        return len(self.pcm) // 2


@dataclass(frozen=True)
class SpeechInterval:
    """Speech boundaries plus the VAD confirmation position used for interruption."""

    start_sample: int
    end_sample: int
    confirmed_start_sample: int


@dataclass(frozen=True)
class UserTurn:
    """Audio-based turn proposal; transcription readiness is a separate condition."""

    turn_id: str
    start_sample: int
    end_sample: int
    endpoint_sample: int
    detection_compute_ns: int = 0


@dataclass(frozen=True)
class DetectionTrace:
    """Ordered source decisions independent of upload packetization."""

    speech: tuple[SpeechInterval, ...]
    turns: tuple[UserTurn, ...]
    events: tuple[dict[str, Any], ...] = ()


@dataclass(frozen=True)
class RecognitionSegment:
    """One finalized recognition portion with source ownership and availability."""

    text: str
    start_sample: int
    end_sample: int
    available_sample: int
    words: tuple[tuple[str, int, int], ...] = ()


@dataclass(frozen=True)
class RecognitionTrace:
    """An explicitly drained continuous ASR request and its timing policy."""

    segments: tuple[RecognitionSegment, ...]
    metrics: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class GeneratedAudio:
    """PCM with measured availability relative to response request submission."""

    pcm: bytes
    ready_offset_ns: int


@dataclass(frozen=True)
class SpokenWord:
    """Word interval relative to cumulative synthesized audio, when supplied."""

    text: str
    start_sample: int
    end_sample: int


@dataclass(frozen=True)
class ResponseTrace:
    """Provisional provider result; generation never commits conversation history."""

    text: str
    audio: tuple[GeneratedAudio, ...]
    compute_ns: int
    words: tuple[SpokenWord, ...] = ()
    metrics: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AudioSpan:
    """Committed bot PCM at an absolute presentation sample offset."""

    start_sample: int
    pcm: bytes
    response_id: str

    @property
    def end_sample(self) -> int:
        """Exclusive presentation end."""
        return self.start_sample + len(self.pcm) // 2


@dataclass(frozen=True)
class SessionResult:
    """Committed session output and an auditable state/compute manifest."""

    spans: tuple[AudioSpan, ...]
    rendered_samples: int
    events: tuple[dict[str, Any], ...]
    messages: tuple[dict[str, Any], ...]
    metrics: dict[str, Any] = field(default_factory=dict)


class Detector(Protocol):
    """Source-only turn analysis port."""

    async def detect(self, source: AudioSource) -> DetectionTrace:
        """Return chronological proposals and speech intervals."""
        ...


class Recognizer(Protocol):
    """One continuous stream with explicit EOF draining."""

    async def recognize(self, source: AudioSource) -> RecognitionTrace:
        """Recognize the entire source without restarting at turn boundaries."""
        ...


class Responder(Protocol):
    """Generate against a private history snapshot."""

    async def respond(self, messages: list[dict[str, Any]], turn_id: str) -> ResponseTrace:
        """Return provisional text/audio; do not mutate the supplied history."""
        ...
