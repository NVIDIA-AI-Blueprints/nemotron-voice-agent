# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The adapter's speech output: the turn engine's text goes through the pipeline's own TTS service.

Ported from FDH ``adapters/pipecat/speech_output.py`` (``a0ef50c``). It implements the
``OutputPath`` protocol of ``core/engine/interfaces.py``.

:meth:`PipecatSpeechOutput.speak` pushes one ``TTSSpeakFrame`` per sentence (v1's sentence
splitter) between a :class:`SpeechStartFrame` and an :class:`EndOfSpeechFrame`. The
:class:`OutputTagger`, placed after the TTS service, keeps v1's output contract:

* each sentence's transcript delta goes out right before that sentence's first audio;
* ``ResponseProgress`` gets ``begin_item``, ``begin_sentence``/``end_sentence`` and
  ``add_audio`` per synthesized chunk (B5);
* ``speak`` returns when the end marker passes the tagger.

Two rules keep the turn engine's response handling as in v1 (plan §2):

* **Interruption (B9).** Pipecat cuts the audio first; an interrupted ``speak`` does not
  return as finished but waits until the turn manager has handled the interruption, which
  cancels it (the item then closes ``incomplete``). If no cancellation comes within
  :data:`INTERRUPT_GRACE_S`, it returns.
* **Failure.** A TTS error or a stall raises :class:`SpeechOutputError`, so the turn manager
  fails the response with ``response_failed`` as v1 does.

A cancelled ``speak`` (the turn manager stopping its own output) drops the rest of that
item's audio and flushes what is queued with a Pipecat interruption.

The response id of an item comes from the ``response.output_item.added`` event its writer
emitted (:meth:`OutputTagger.note_event`), because FBA's ``MessageItemWriter`` exposes none.
"""

from __future__ import annotations

import asyncio
import contextlib
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from loguru import logger
from pipecat.frames.frames import (
    Frame,
    InterruptionFrame,
    OutputAudioRawFrame,
    TTSAudioRawFrame,
    TTSSpeakFrame,
    TTSStoppedFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from examples.frontend_backend_agent.adapters.pipecat.frames import EndOfSpeechFrame, ItemAudioFrame, SpeechStartFrame
from examples.frontend_backend_agent.core.audio.formats import AudioFormat
from examples.frontend_backend_agent.core.engine.playback import ResponseProgress
from examples.frontend_backend_agent.core.engine.response_writer import MessageItemWriter
from examples.frontend_backend_agent.core.errors import SpeechServiceError

#: How often ``speak`` checks a job for TTS progress.
STALL_CHECK_S = 3.0
#: A job with no TTS progress for this long has stalled (a TTS failure without an error frame).
STALL_TIMEOUT_S = 30.0
#: How long an interrupted ``speak`` waits for the turn manager's cancellation (B9).
INTERRUPT_GRACE_S = 2.0


class SpeechOutputError(SpeechServiceError):
    """The TTS service failed or stalled while speaking an item."""


#: v1's sentence boundary (``voice/engine/output_path.py`` at tag ``fbv-v1-baseline``), unchanged (B3).
_SENTENCE_END = re.compile(r"(?<=[.!?])[\"')\]]*\s+|\n+")


def split_sentences(text: str, *, enabled: bool = True) -> list[str]:
    """Split ``text`` into sentences (whitespace-normalized); one item when splitting is disabled (v1's splitter)."""
    normalized = " ".join(text.split())
    if not normalized:
        return []
    if not enabled:
        return [normalized]
    parts = [" ".join(part.split()) for part in _SENTENCE_END.split(text)]
    return [part for part in parts if part]


@dataclass(eq=False)
class SpeechJob:
    """One item being spoken."""

    item: MessageItemWriter
    response_id: str
    progress: ResponseProgress
    sentences: list[str]
    on_first_audio: Callable[[], None] | None
    done: asyncio.Future[str]
    index: int = 0
    sentence_open: bool = False
    first_audio: bool = True
    cancelled: bool = False
    failed: bool = False  # the TTS service reported an error: nothing more of this item is sent
    error: str = ""
    last_progress: float = field(default_factory=lambda: asyncio.get_running_loop().time())

    def resolve(self, outcome: str) -> None:
        """Wake ``speak`` (``completed``, ``interrupted``, ``cancelled``, ``failed`` or ``stalled``)."""
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
        #: The job whose frames were pushed last; it may still be queued before the tagger.
        self.pushed: SpeechJob | None = None
        self.capture: list[Frame] | None = None
        self.interrupted = asyncio.Event()
        self._response_ids: dict[str, str] = {}

    def note_event(self, event: dict[str, Any]) -> None:
        """Remember each item's response id from ``response.output_item.added``."""
        if event.get("type") == "response.output_item.added":
            item = event.get("item") or {}
            item_id = item.get("id")
            if isinstance(item_id, str):
                self._response_ids[item_id] = str(event.get("response_id") or "")

    def response_id_for(self, item_id: str) -> str:
        """The response id the item's ``output_item.added`` named (``""`` if unknown)."""
        return self._response_ids.get(item_id, "")

    def fail_current(self, reason: str) -> None:
        """The TTS service reported an error: fail the job being spoken.

        The error is a system frame and can arrive before the job's :class:`SpeechStartFrame`
        has reached the tagger (a failure on the first sentence), so the job pushed last is
        failed when no job is active yet.
        """
        job = self.job
        if job is None and self.pushed is not None and not self.pushed.done.done():
            job = self.pushed
        if job is not None and not job.cancelled and not job.failed:
            job.failed = True  # the first error is the cause; later ones (no audio) follow from it
            job.error = reason
            job.resolve("failed")

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
                    response_id=frame.job.response_id,
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
            if job.cancelled or job.failed:
                return
            await self._audio(job, frame)
        elif isinstance(frame, TTSStoppedFrame):
            if job.sentence_open and not job.cancelled and not job.failed:
                job.progress.end_sentence()
                job.sentence_open = False
                job.index += 1
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
        if job.failed:
            # As in v1, a failed item keeps only what was spoken before the error.
            if job is self.job:
                self.job = None
            return
        if job.cancelled or job is not self.job:
            job.resolve("cancelled" if job.cancelled else "interrupted")
            return
        # Sentences that produced no audio still get their transcript (as v1's output path does).
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
                response_id=job.response_id,
                item_id=item.item_id,
                output_index=item.output_index,
                begin=False,
            )
        )
        self.job = None
        job.resolve("completed")


class PipecatSpeechOutput:
    """The ``OutputPath`` over the pipeline's TTS service (see the module docstring)."""

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
        """Split ``text``; the pipeline's TTS keeps its catalog voice (B8)."""
        if voice and not self._warned_voice:
            self._warned_voice = True
            logger.warning(f"the Pipecat adapter keeps the catalog TTS voice; requested voice {voice!r} is ignored")
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
            response_id=self._tagger.response_id_for(item.item_id),
            progress=progress,
            sentences=list(prepared.sentences),
            on_first_audio=on_first_audio,
            done=asyncio.get_running_loop().create_future(),
        )
        self._tagger.pushed = job
        try:
            await self._push(SpeechStartFrame(job=job))
            for sentence in job.sentences:
                await self._push(TTSSpeakFrame(text=sentence, append_to_context=False))
            await self._push(EndOfSpeechFrame(job=job))
            outcome = await self._wait(job)
            if outcome == "failed":  # the service's own message, as v1 reported its TTS errors
                raise SpeechOutputError(job.error or f"TTS failed for item {item.item_id}")
            if outcome == "stalled":
                raise SpeechOutputError(f"TTS stalled for item {item.item_id}: no audio progress")
            if outcome == "interrupted":
                # B9: Pipecat cut the audio; the turn manager cancels this task when it handles the
                # interruption. Return only if it decides to keep the item.
                await asyncio.sleep(INTERRUPT_GRACE_S)
                logger.warning(f"item {item.item_id}: interrupted output was not cancelled by the turn manager")
        except asyncio.CancelledError:
            job.cancelled = True
            job.resolve("cancelled")
            if self._tagger.job is job:
                with contextlib.suppress(Exception):
                    await self._interrupt()
            raise

    async def _wait(self, job: SpeechJob) -> str:
        """Wait for the job's outcome; a job with no TTS progress for :data:`STALL_TIMEOUT_S` has stalled."""
        loop = asyncio.get_running_loop()
        while not job.done.done():
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(asyncio.shield(job.done), timeout=STALL_CHECK_S)
            if not job.done.done() and loop.time() - job.last_progress > STALL_TIMEOUT_S:
                logger.warning(f"item {job.item.item_id}: no TTS progress for {STALL_TIMEOUT_S:g}s")
                job.resolve("stalled")
        return job.done.result()
