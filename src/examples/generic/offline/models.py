# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Source records, append-only upload input, and offline runtime ports."""

import asyncio
import threading
from dataclasses import dataclass, field
from typing import Any, Protocol

RATE = 16000
NANOSECONDS = 1_000_000_000


@dataclass(frozen=True)
class SourceChunk:
    """A stable PCM range shared by independent source-analysis producers."""

    sequence: int
    start_sample: int
    end_sample: int
    pcm: bytes


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

    def chunks(self, samples: int = 512):
        """Yield identical chunk IDs and ranges without depending on upload pacing."""
        if samples <= 0:
            raise ValueError("Source chunk length must be positive")
        for sequence, start in enumerate(range(0, self.sample_count, samples)):
            end = min(start + samples, self.sample_count)
            yield SourceChunk(sequence, start, end, self.pcm[start * 2 : end * 2])

    async def async_chunks(self, samples: int = 512):
        """Expose the same canonical ranges to an asynchronous analyzer."""
        for chunk in self.chunks(samples):
            yield chunk

    async def wait_sealed(self):
        """A finite immutable source already has a validated EOF."""
        return self


class SourceAbortedError(RuntimeError):
    """An unfinished upload was cancelled or rejected, rather than completed."""


class StreamingAudioSource:
    """Share validated contiguous PCM prefixes without inventing source EOF.

    The upload coordinator validates packet order and the final digest. Only
    its successful ``seal`` makes a final partial window or EOF visible. Each
    reader owns a cursor, so ASR and VAD consume identical canonical windows
    independently of upload packet boundaries and one another's progress.
    """

    sample_rate = RATE

    def __init__(self, declared_samples: int, *, max_samples: int = RATE * 3600):
        """Bound retained PCM by a validated finite source declaration."""
        if type(max_samples) is not int or max_samples <= 0:
            raise ValueError("Maximum source extent must be a positive integer")
        if type(declared_samples) is not int or not 0 <= declared_samples <= max_samples:
            raise ValueError("Declared source extent is outside the configured limit")
        self._declared_samples = declared_samples
        self._pcm = bytearray()
        self._source: AudioSource | None = None
        self._aborted = False
        self._condition = threading.Condition()
        self._waiters: set[asyncio.Future] = set()

    @property
    def sample_count(self) -> int:
        """Return the immutable declared extent, independently of received PCM."""
        return self._declared_samples

    @property
    def received_samples(self) -> int:
        """Return only received PCM; the declaration is not a progress counter."""
        with self._condition:
            return len(self._pcm) // 2

    @property
    def sealed(self) -> bool:
        """Return whether the coordinator has accepted the explicit source EOF."""
        with self._condition:
            return self._source is not None

    @property
    def pcm(self) -> bytes:
        """Expose a complete immutable recording only after accepted EOF."""
        with self._condition:
            self._check_aborted()
            if self._source is None:
                raise RuntimeError("Streaming PCM is not sealed")
            return self._source.pcm

    @staticmethod
    def _wake(waiter):
        if not waiter.done():
            waiter.set_result(None)

    def _notify(self):
        self._condition.notify_all()
        # A buffered packet burst may append again before the coroutine gets a
        # turn. Queue one wakeup per waiter; it rechecks the current prefix and
        # registers a fresh waiter under the same condition if still blocked.
        waiters = tuple(self._waiters)
        self._waiters.clear()
        for waiter in waiters:
            loop = waiter.get_loop()
            if not loop.is_closed():
                loop.call_soon_threadsafe(self._wake, waiter)

    def _check_aborted(self):
        if self._aborted:
            raise SourceAbortedError("Streaming source was aborted before EOF")

    def append(self, pcm: bytes) -> None:
        """Accept a validated packet's PCM once, retaining a bounded prefix."""
        if not isinstance(pcm, bytes) or not pcm or len(pcm) % 2:
            raise ValueError("Source append requires nonempty aligned PCM16 bytes")
        with self._condition:
            self._check_aborted()
            if self._source is not None:
                raise RuntimeError("Streaming source is already sealed")
            if len(self._pcm) + len(pcm) > self.sample_count * 2:
                raise ValueError("Source append exceeds its declared extent")
            self._pcm.extend(pcm)
            self._notify()

    def seal(self) -> AudioSource:
        """Expose EOF only after exact received extent and coordinator validation."""
        with self._condition:
            self._check_aborted()
            if self._source is None:
                if len(self._pcm) != self.sample_count * 2:
                    raise ValueError("Source cannot seal before its declared extent")
                self._source = AudioSource(bytes(self._pcm))
                self._pcm = self._source.pcm
                self._notify()
            return self._source

    def abort(self) -> None:
        """Wake every unfinished reader with cancellation, never a normal EOF."""
        with self._condition:
            if self._source is None:
                self._aborted = True
                self._notify()

    def _read(self, start: int, samples: int):
        self._check_aborted()
        available = len(self._pcm) // 2
        if available - start >= samples or self._source is not None and available > start:
            end = min(start + samples, available)
            return SourceChunk(start // samples, start, end, bytes(self._pcm[start * 2 : end * 2])), False
        return None, self._source is not None

    def chunks(self, samples: int = 512):
        """Block a provider thread until the next full window or validated EOF."""
        if type(samples) is not int or samples <= 0:
            raise ValueError("Source chunk length must be a positive integer")
        start = 0
        while True:
            with self._condition:
                while True:
                    chunk, ended = self._read(start, samples)
                    if chunk is not None or ended:
                        break
                    self._condition.wait()
            if chunk is None:
                return
            yield chunk
            start = chunk.end_sample

    async def _wait(self, ready):
        while True:
            with self._condition:
                self._check_aborted()
                result = ready()
                if result is not None:
                    return result
                waiter = asyncio.get_running_loop().create_future()
                self._waiters.add(waiter)
            try:
                await waiter
            finally:
                with self._condition:
                    self._waiters.discard(waiter)

    async def async_chunks(self, samples: int = 512):
        """Await canonical windows without blocking the owning event loop."""
        if type(samples) is not int or samples <= 0:
            raise ValueError("Source chunk length must be a positive integer")
        start = 0
        while True:

            def ready(start=start):
                chunk, ended = self._read(start, samples)
                return (chunk, ended) if chunk is not None or ended else None

            chunk, _ = await self._wait(ready)
            if chunk is None:
                return
            yield chunk
            start = chunk.end_sample

    async def wait_sealed(self) -> AudioSource:
        """Await accepted EOF without converting upload starvation to silence."""
        return await self._wait(lambda: self._source)


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
