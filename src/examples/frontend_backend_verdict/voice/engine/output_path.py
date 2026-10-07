# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Agent text out: sentences -> TTS -> resample -> encode -> deltas.

Each sentence is one synthesis request. Sentence *k+1* is synthesized while
sentence *k* is being sent (one-sentence lookahead), and the first sentence of
an item starts synthesizing as soon as the item is *prepared*, so an answer
queued behind a filler is already synthesizing while the filler is sent.

For every sentence the transcript delta is written just before the sentence's
first audio delta, so a client scoring transcript against played audio (tau2)
never sees text ahead of the audio by more than one sentence.
"""

from __future__ import annotations

import asyncio
import contextlib
import re
import time
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field

from examples.frontend_backend_verdict.voice.audio.formats import AudioFormat
from examples.frontend_backend_verdict.voice.audio.pcm import b64encode_audio
from examples.frontend_backend_verdict.voice.audio.resample import StreamResampler
from examples.frontend_backend_verdict.voice.engine.playback import ResponseProgress
from examples.frontend_backend_verdict.voice.speech.ports import Synthesizer, TextNormalizer
from examples.frontend_backend_verdict.voice.wire.response_writer import MessageItemWriter

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


class _Chunker:
    """Cuts an encoded byte stream into fixed-duration chunks."""

    def __init__(self, chunk_bytes: int) -> None:
        self._chunk_bytes = max(1, chunk_bytes)
        self._buffer = bytearray()

    def push(self, data: bytes) -> list[bytes]:
        self._buffer.extend(data)
        chunks: list[bytes] = []
        while len(self._buffer) >= self._chunk_bytes:
            chunks.append(bytes(self._buffer[: self._chunk_bytes]))
            del self._buffer[: self._chunk_bytes]
        return chunks

    def flush(self) -> list[bytes]:
        if not self._buffer:
            return []
        rest = bytes(self._buffer)
        self._buffer.clear()
        return [rest]


@dataclass(slots=True)
class SynthesizedSentence:
    """One sentence's text and its encoded audio chunks (filled by the producer)."""

    index: int
    text: str
    chunks: asyncio.Queue[bytes | BaseException | None] = field(default_factory=asyncio.Queue)
    reached: asyncio.Event = field(default_factory=asyncio.Event)

    async def audio(self) -> AsyncIterator[bytes]:
        """Encoded client-format chunks, in order."""
        while True:
            item = await self.chunks.get()
            if item is None:
                return
            if isinstance(item, BaseException):
                raise item
            yield item


class SentenceStream:
    """Synthesis of one item's sentences with a one-sentence lookahead."""

    def __init__(
        self,
        sentences: list[str],
        *,
        synthesizer: Synthesizer,
        normalizer: TextNormalizer,
        voice: str | None,
        out_format: AudioFormat,
        chunk_ms: int,
    ) -> None:
        """Prepare; call :meth:`start` to begin synthesizing the first sentence."""
        self.sentences = [SynthesizedSentence(index, text) for index, text in enumerate(sentences)]
        self._synthesizer = synthesizer
        self._normalizer = normalizer
        self._voice = voice
        self._format = out_format
        self._chunk_bytes = max(1, out_format.rate * chunk_ms // 1000) * out_format.bytes_per_sample
        self._task: asyncio.Task[None] | None = None

    def start(self) -> None:
        """Start the producer task (idempotent)."""
        if self._task is None:
            self._task = asyncio.create_task(self._produce(), name="tts-producer")

    async def aclose(self) -> None:
        """Cancel synthesis (closing the synthesizer iterator cancels its RPC)."""
        if self._task is not None and not self._task.done():
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._task

    async def _produce(self) -> None:
        for sentence in self.sentences:
            if sentence.index > 0:
                await self.sentences[sentence.index - 1].reached.wait()
            try:
                await self._synthesize(sentence)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - surfaced to the consumer, which fails the response
                sentence.chunks.put_nowait(exc)
                return

    async def _synthesize(self, sentence: SynthesizedSentence) -> None:
        text = await self._normalizer.normalize(sentence.text)
        if not any(char.isalnum() for char in text):
            sentence.chunks.put_nowait(None)
            return
        resampler = StreamResampler(self._synthesizer.sample_rate, self._format.rate)
        chunker = _Chunker(self._chunk_bytes)
        stream = self._synthesizer.synthesize(text, voice=self._voice)
        async with contextlib.aclosing(stream):  # type: ignore[type-var]
            async for pcm in stream:
                for chunk in chunker.push(self._format.encode(resampler.process(pcm))):
                    sentence.chunks.put_nowait(chunk)
        for chunk in chunker.push(self._format.encode(resampler.flush())) + chunker.flush():
            sentence.chunks.put_nowait(chunk)
        sentence.chunks.put_nowait(None)


@dataclass(slots=True)
class PreparedSpeech:
    """An item's text, split and (for audio) already synthesizing."""

    text: str
    kind: str
    sentences: list[str]
    stream: SentenceStream | None

    async def aclose(self) -> None:
        """Stop any synthesis still running."""
        if self.stream is not None:
            await self.stream.aclose()


class OutputPath:
    """Turns agent text into transcript and audio deltas for one item at a time."""

    def __init__(
        self,
        *,
        synthesizer: Synthesizer,
        normalizer: TextNormalizer,
        sentence_split: bool,
        chunk_ms: int,
        pace_output: bool,
        pace_lead_ms: int,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        """Bind the shared synthesizer and the per-session output settings."""
        self._synthesizer = synthesizer
        self._normalizer = normalizer
        self._sentence_split = sentence_split
        self._chunk_ms = chunk_ms
        self._pace = pace_output
        self._pace_lead_ms = pace_lead_ms
        self._monotonic = monotonic

    def prepare(
        self, text: str, *, kind: str, modality: str, voice: str | None, out_format: AudioFormat
    ) -> PreparedSpeech:
        """Split ``text`` and, for audio, start synthesizing its first sentence now."""
        sentences = split_sentences(text, enabled=self._sentence_split)
        stream = None
        if modality == "audio" and sentences:
            stream = SentenceStream(
                sentences,
                synthesizer=self._synthesizer,
                normalizer=self._normalizer,
                voice=voice,
                out_format=out_format,
                chunk_ms=self._chunk_ms,
            )
            stream.start()
        return PreparedSpeech(text=text, kind=kind, sentences=sentences, stream=stream)

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
        progress.begin_item(item.item_id, kind=prepared.kind, full_text=prepared.text)
        item.start()
        if prepared.stream is None:
            for index, sentence in enumerate(prepared.sentences):
                progress.begin_sentence(sentence)
                item.transcript_delta(sentence if index == 0 else f" {sentence}")
                progress.end_sentence()
            return
        started = self._monotonic()
        sent_ms = 0.0
        first = True
        bytes_per_ms = out_format.rate * out_format.bytes_per_sample / 1000.0
        for sentence in prepared.stream.sentences:
            sentence.reached.set()
            progress.begin_sentence(sentence.text)
            delta = sentence.text if sentence.index == 0 else f" {sentence.text}"
            wrote_transcript = False
            async for chunk in sentence.audio():
                if not wrote_transcript:
                    item.transcript_delta(delta)
                    wrote_transcript = True
                if self._pace:
                    ahead = sent_ms - (self._monotonic() - started) * 1000.0
                    if ahead > self._pace_lead_ms:
                        await asyncio.sleep((ahead - self._pace_lead_ms) / 1000.0)
                item.audio_delta(b64encode_audio(chunk))
                chunk_ms = len(chunk) / bytes_per_ms
                sent_ms += chunk_ms
                progress.add_audio(chunk_ms)
                if first:
                    first = False
                    if on_first_audio is not None:
                        on_first_audio()
                await asyncio.sleep(0)
            if not wrote_transcript:
                item.transcript_delta(delta)
            progress.end_sentence()
