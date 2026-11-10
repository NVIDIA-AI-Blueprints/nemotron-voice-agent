# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pipecat processors that adapt a voice pipeline to the live session protocol."""

from __future__ import annotations

import asyncio
import time
from fractions import Fraction
from typing import TYPE_CHECKING

from loguru import logger
from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    ErrorFrame,
    InputAudioRawFrame,
    InterruptionFrame,
    OutputAudioRawFrame,
    StartFrame,
    TranscriptionFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
    TTSTextFrame,
    UserStartedSpeakingFrame,
)
from pipecat.observers.base_observer import BaseObserver, FramePushed
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

if TYPE_CHECKING:
    from examples.frontend_backend_live.adapter.serializer import LiveFrameSerializer
    from gateways.live.ports import SessionPort

BYTES_PER_SAMPLE = 2
SILENCE_PACKET_SECONDS = 0.1


def audio_seconds(audio: bytes, sample_rate: int, channels: int = 1) -> Fraction:
    """Return the exact duration of PCM16 ``audio`` in seconds, so that durations add up without rounding."""
    return Fraction(len(audio), BYTES_PER_SAMPLE * max(sample_rate, 1) * max(channels, 1))


class LiveInputGate(FrameProcessor):
    """Counts caller audio against the input media clock and drops it while muted."""

    def __init__(self, session: SessionPort, **kwargs):
        """Bind the session whose input clock this gate advances."""
        super().__init__(**kwargs)
        self.session = session
        self.muted = False
        self._seconds = Fraction(0)

    @property
    def input_ms(self) -> int:
        """Milliseconds of caller audio seen so far, muted audio included."""
        return int(self._seconds * 1000)

    async def process_frame(self, frame, direction: FrameDirection) -> None:
        """Advance the clock for every audio frame; pass audio on only when not muted."""
        await super().process_frame(frame, direction)
        if isinstance(frame, InputAudioRawFrame):
            self._seconds += audio_seconds(frame.audio, frame.sample_rate, frame.num_channels)
            await self.session.advance_input_clock(self.input_ms)
            if self.muted:
                return
        await self.push_frame(frame, direction)


class LiveEventObserver(BaseObserver):
    """Publishes the caller's transcript deltas and pipeline errors as live events."""

    def __init__(self, session: SessionPort, gate: LiveInputGate, stt, **kwargs):
        """Watch frames pushed by ``stt`` and publish them on ``session``."""
        # Each processor an error frame passes pushes it on: report it at its first push only.
        super().__init__(observe_every_push=False, **kwargs)
        self.session, self.gate, self.stt = session, gate, stt
        self._speech_started_ms = 0
        self._tasks: set[asyncio.Task] = set()

    def _emit(self, kind: str, **fields) -> None:
        self._send(self.session.emit(kind, **fields))

    def _report(self, message: str, fatal: bool) -> None:
        self._send(self.session.report_error(message, fatal=fatal))

    def _send(self, sending) -> None:
        task = asyncio.create_task(sending)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def on_push_frame(self, data: FramePushed) -> None:
        """Translate the frames that carry the caller's transcripts and the pipeline's errors."""
        frame = data.frame
        if isinstance(frame, UserStartedSpeakingFrame) and data.source is not self.stt:
            self._speech_started_ms = self.gate.input_ms
        elif isinstance(frame, TranscriptionFrame) and data.source is self.stt and frame.text.strip():
            end_ms = self.gate.input_ms
            self._emit(
                "session.input_transcript.delta",
                delta=frame.text if frame.text.endswith(" ") else frame.text + " ",
                start_ms=min(self._speech_started_ms, end_ms),
                end_ms=end_ms,
            )
        elif isinstance(frame, ErrorFrame) and not frame.fatal:
            # A processor the error left unusable ends the pipeline, and so the session.
            self._report(str(frame.error)[:500], fatal=frame.processor is not None and not frame.processor.is_usable)


class LiveOutputTranscript(FrameProcessor):
    """The assistant's transcript: each sentence that played, as ``session.output_transcript.delta``."""

    def __init__(self, session: SessionPort, **kwargs):
        """Send on ``session``."""
        super().__init__(**kwargs)
        self.session = session
        self._sentence_start_ms: int | None = None

    async def process_frame(self, frame, direction: FrameDirection) -> None:
        """Send each sentence's text that follows its audio; pass every frame on."""
        await super().process_frame(frame, direction)
        if direction is FrameDirection.DOWNSTREAM:
            if isinstance(frame, OutputAudioRawFrame) and frame.audio and self._sentence_start_ms is None:
                self._sentence_start_ms = self.now_ms
            elif isinstance(frame, TTSTextFrame) and frame.text.strip():
                end_ms = self.now_ms
                start_ms = end_ms if self._sentence_start_ms is None else self._sentence_start_ms
                self._sentence_start_ms = None
                await self.played(frame.text, start_ms, end_ms)
            elif isinstance(frame, (InterruptionFrame, TTSStoppedFrame)):
                # Audio after a line's last sentence, the resampler's tail, belongs to no sentence of the next line.
                self._sentence_start_ms = None
        await self.push_frame(frame, direction)

    @property
    def now_ms(self) -> int:
        """The session's input clock."""
        return self.session.input_ms

    async def played(self, text: str, start_ms: int, end_ms: int) -> None:
        """Send the sentence, ending in a space so the deltas join."""
        await self.session.emit(
            "session.output_transcript.delta",
            delta=text if text.endswith(" ") else text + " ",
            start_ms=start_ms,
            end_ms=end_ms,
        )


class LiveWebSocketSilence(FrameProcessor):
    """Sends a primary WebSocket's client silence between lines: the continuous stream such a client expects."""

    def __init__(self, session: SessionPort, serializer: LiveFrameSerializer, **kwargs):
        """Send through ``serializer`` to ``session``'s primary connection."""
        # Direct mode: a line's start arrives before the output sends its first audio, and its end after its last.
        super().__init__(enable_direct_mode=True, **kwargs)
        self.session, self.serializer = session, serializer
        self._silence = bytes(int(serializer.rate * SILENCE_PACKET_SECONDS) * BYTES_PER_SAMPLE)
        self._between_lines = True
        self._clock: asyncio.Task | None = None

    async def process_frame(self, frame, direction: FrameDirection) -> None:
        """Note where each line starts and ends, and pass every frame on."""
        await super().process_frame(frame, direction)
        if isinstance(frame, StartFrame):
            self._clock = self.create_task(self._tick(), name="silence")
        elif isinstance(frame, (EndFrame, CancelFrame)):
            await self._stop()
        elif direction is FrameDirection.DOWNSTREAM and isinstance(frame, TTSStartedFrame):
            self._between_lines = False
        elif direction is FrameDirection.DOWNSTREAM and isinstance(frame, (TTSStoppedFrame, InterruptionFrame)):
            self._between_lines = True
        await self.push_frame(frame, direction)

    async def _tick(self) -> None:
        """Send silence every 100 ms while the session is active and no line plays; start afresh after a stall."""
        try:
            next_at = time.monotonic()
            while True:
                next_at += SILENCE_PACKET_SECONDS
                await asyncio.sleep(max(next_at - time.monotonic(), 0))
                now = time.monotonic()
                if now - next_at > SILENCE_PACKET_SECONDS:
                    next_at = now
                if self._between_lines and self.session.is_active():
                    # Through the serializer, never as a frame: what reads the call after the output sees only replies.
                    await self.serializer.send(self._silence, self.serializer.rate)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("WebSocket audio output failed")
            await self.session.report_error("Output audio transport failed; start a new session.", fatal=True)
            self.session.request_close("connection_lost")

    async def _stop(self) -> None:
        if self._clock:
            await self.cancel_task(self._clock)
            self._clock = None
