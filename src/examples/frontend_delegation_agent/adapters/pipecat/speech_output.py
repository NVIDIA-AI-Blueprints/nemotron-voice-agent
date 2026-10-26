# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The Pipecat head's ``SpeechOutput``: agent text goes through the pipeline's own TTS service.

:meth:`PipecatSpeechOutput.speak` pushes one ``TTSSpeakFrame`` per sentence (the shared
sentence splitter) between a :class:`SpeechStartFrame` and an :class:`EndOfSpeechFrame`.
The :class:`OutputTagger`, placed after the TTS service, keeps the contract every head
gives the shared turn manager (as the former standalone Realtime head's ``OutputPath`` did):

* each sentence's transcript delta goes out right before that sentence's first audio;
* ``ResponseProgress`` gets ``begin_item``, ``begin_sentence``/``end_sentence`` and
  ``add_audio`` per chunk (the *sent* audio);
* ``speak`` returns when the end marker passes the tagger (the TTS service orders data
  frames after the audio before them), or at once on an interruption.

A cancelled ``speak`` (the turn manager stopping its own output) drops the rest of that
item's audio and flushes what is queued with a Pipecat interruption.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from loguru import logger
from pipecat.frames.frames import (
    ErrorFrame,
    Frame,
    InterruptionFrame,
    OutputAudioRawFrame,
    TTSAudioRawFrame,
    TTSSpeakFrame,
    TTSStoppedFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from examples.frontend_delegation_agent.adapters.pipecat.frames import (
    EndOfSpeechFrame,
    ItemAudioFrame,
    SpeechStartFrame,
)
from examples.frontend_delegation_agent.core.audio.formats import AudioFormat
from examples.frontend_delegation_agent.core.engine.playback import ResponseProgress
from examples.frontend_delegation_agent.core.speech_text import split_sentences
from examples.frontend_delegation_agent.core.wire.response_writer import MessageItemWriter

# Fallback when a sentence never produces its end frames (a TTS failure).
STALL_TIMEOUT_S = 3.0


@dataclass(eq=False)
class SpeechJob:
    """One item being spoken."""

    item: MessageItemWriter
    progress: ResponseProgress
    sentences: list[str]
    on_first_audio: Callable[[], None] | None
    done: asyncio.Future[str]
    index: int = 0
    sentence_open: bool = False
    first_audio: bool = True
    cancelled: bool = False
    last_progress: float = field(default_factory=lambda: asyncio.get_running_loop().time())

    def resolve(self, outcome: str) -> None:
        """Wake ``speak`` (``completed``, ``interrupted``, ``cancelled`` or ``stalled``)."""
        if not self.done.done():
            self.done.set_result(outcome)


@dataclass(slots=True)
class PipecatPreparedSpeech:
    """One item's text, split into sentences (synthesis starts in ``speak``)."""

    text: str
    kind: str
    modality: str
    sentences: list[str]

    async def aclose(self) -> None:
        """Nothing runs before ``speak``."""


class OutputTagger(FrameProcessor):
    """After the TTS service: transcript before audio, ``ResponseProgress`` feeding, item markers."""

    def __init__(self, **kwargs: Any) -> None:
        """No job is active until a :class:`SpeechStartFrame` arrives."""
        super().__init__(**kwargs)
        self.job: SpeechJob | None = None
        self.capture: list[Frame] | None = None
        self.interrupted = asyncio.Event()

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        """Tag the active job's frames; pass everything else through."""
        await super().process_frame(frame, direction)
        if isinstance(frame, InterruptionFrame):
            job, self.job = self.job, None
            if job is not None:
                job.resolve("interrupted")
            self.interrupted.set()
            await self.push_frame(frame, direction)
            return
        if direction != FrameDirection.DOWNSTREAM:
            await self.push_frame(frame, direction)
            return
        if isinstance(frame, SpeechStartFrame):
            self.interrupted.clear()
            self.job = frame.job
            item = frame.job.item
            await self.push_frame(
                ItemAudioFrame(
                    message=None,
                    response_id=item.response_id,
                    item_id=item.item_id,
                    output_index=item.output_index,
                    begin=True,
                )
            )
            return
        if isinstance(frame, EndOfSpeechFrame):
            await self._finish(frame.job)
            return
        job = self.job
        if job is None:
            if not isinstance(frame, OutputAudioRawFrame):  # stray audio has no item to belong to
                await self.push_frame(frame, direction)
            return
        if isinstance(frame, TTSAudioRawFrame):
            if job.cancelled:
                return
            await self._audio(job, frame)
        elif isinstance(frame, TTSStoppedFrame):
            if job.sentence_open and not job.cancelled:
                job.progress.end_sentence()
                job.sentence_open = False
                job.index += 1
            await self.push_frame(frame, direction)
        elif isinstance(frame, ErrorFrame):
            logger.warning(f"TTS error during item {job.item.item_id}: {frame.error}")
            await self.push_frame(frame, direction)
        else:
            await self.push_frame(frame, direction)

    async def _audio(self, job: SpeechJob, frame: TTSAudioRawFrame) -> None:
        job.last_progress = asyncio.get_running_loop().time()
        if not job.sentence_open and job.index < len(job.sentences):
            job.sentence_open = True
            sentence = job.sentences[job.index]
            job.progress.begin_sentence(sentence)
            await self._transcript(job, sentence if job.index == 0 else f" {sentence}")
        bytes_per_ms = frame.sample_rate * 2 * max(1, frame.num_channels) / 1000.0
        chunk_ms = len(frame.audio) / bytes_per_ms if bytes_per_ms else 0.0
        job.progress.add_audio(chunk_ms)
        if job.first_audio:
            job.first_audio = False
            if job.on_first_audio is not None:
                job.on_first_audio()
        await self.push_frame(frame)

    async def _transcript(self, job: SpeechJob, delta: str) -> None:
        """Write one transcript delta in-band, before the audio that follows it."""
        captured: list[Frame] = []
        self.capture = captured
        try:
            job.item.transcript_delta(delta)
        finally:
            self.capture = None
        for event_frame in captured:
            await self.push_frame(event_frame)

    async def _finish(self, job: SpeechJob) -> None:
        if job.cancelled or job is not self.job:
            job.resolve("cancelled" if job.cancelled else "interrupted")
            return
        # Sentences that produced no audio still get their transcript (as the former server did).
        if job.sentence_open:
            job.progress.end_sentence()
            job.sentence_open = False
            job.index += 1
        while job.index < len(job.sentences):
            sentence = job.sentences[job.index]
            job.progress.begin_sentence(sentence)
            await self._transcript(job, sentence if job.index == 0 else f" {sentence}")
            job.progress.end_sentence()
            job.index += 1
        item = job.item
        await self.push_frame(
            ItemAudioFrame(
                message=None,
                response_id=item.response_id,
                item_id=item.item_id,
                output_index=item.output_index,
                begin=False,
            )
        )
        self.job = None
        job.resolve("completed")


class PipecatSpeechOutput:
    """``SpeechOutput`` over the pipeline's TTS service (see the module docstring)."""

    def __init__(
        self,
        *,
        push: Callable[[Frame], Any],
        interrupt: Callable[[], Any],
        tagger: OutputTagger,
        sentence_split: bool,
    ) -> None:
        """Bind how frames enter the pipeline (upstream of the TTS) and how output is flushed."""
        self._push = push
        self._interrupt = interrupt
        self._tagger = tagger
        self._sentence_split = sentence_split
        self._warned_voice = False

    def prepare(
        self, text: str, *, kind: str, modality: str, voice: str | None, out_format: AudioFormat
    ) -> PipecatPreparedSpeech:
        """Split ``text``; the pipeline's TTS keeps its catalog voice (per-response voices are not switched)."""
        if voice and not self._warned_voice:
            self._warned_voice = True
            logger.warning(f"the Pipecat head keeps the catalog TTS voice; requested voice {voice!r} is ignored")
        return PipecatPreparedSpeech(
            text=text,
            kind=kind,
            modality=modality,
            sentences=split_sentences(text, enabled=self._sentence_split),
        )

    async def speak(
        self,
        prepared: PipecatPreparedSpeech,
        item: MessageItemWriter,
        progress: ResponseProgress,
        *,
        out_format: AudioFormat,
        on_first_audio: Callable[[], None] | None = None,
    ) -> None:
        """Send one item's transcript and audio; cancellation leaves the item open for the caller to close."""
        progress.begin_item(item.item_id, kind=prepared.kind, full_text=prepared.text)
        item.start()
        if prepared.modality != "audio" or not prepared.sentences:
            for index, sentence in enumerate(prepared.sentences):
                progress.begin_sentence(sentence)
                item.transcript_delta(sentence if index == 0 else f" {sentence}")
                progress.end_sentence()
            return
        job = SpeechJob(
            item=item,
            progress=progress,
            sentences=list(prepared.sentences),
            on_first_audio=on_first_audio,
            done=asyncio.get_running_loop().create_future(),
        )
        try:
            await self._push(SpeechStartFrame(job=job))
            for sentence in job.sentences:
                await self._push(TTSSpeakFrame(text=sentence, append_to_context=False))
            await self._push(EndOfSpeechFrame(job=job))
            await self._wait(job)
        except asyncio.CancelledError:
            job.cancelled = True
            job.resolve("cancelled")
            if self._tagger.job is job:
                with contextlib.suppress(Exception):
                    await self._interrupt()
            raise

    async def _wait(self, job: SpeechJob) -> None:
        """Wait for the end marker; give up on an item whose TTS stalls (no audio for a while)."""
        loop = asyncio.get_running_loop()
        while not job.done.done():
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(asyncio.shield(job.done), timeout=STALL_TIMEOUT_S)
            if not job.done.done() and loop.time() - job.last_progress > 10 * STALL_TIMEOUT_S:
                logger.warning(f"item {job.item.item_id}: no TTS progress, giving up")
                job.resolve("stalled")
