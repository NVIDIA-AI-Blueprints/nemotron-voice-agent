# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pipecat processors that adapt a voice pipeline to the live session protocol.

Each works through the adapter's view of the session (:class:`gateways.live.ports.SessionPort`), never the session
itself.

* :class:`LiveInputGate` counts caller audio against the session's input clock and honors mute.
* :class:`LiveEventObserver` turns pipeline frames into input transcript events and failure reports.
* :class:`LiveOutputTranscript` sends the sentences that played as output transcript events: placed after the output,
  it reads each sentence's text once its audio has gone out.
* :class:`LiveWebSocketSilence` sends a primary WebSocket's client silence between lines, after Pipecat's output
  transport, which sends the reply audio on either transport. What the caller heard is read after the output
  (``examples.frontend_backend_live.media.silent_lines``).
"""

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


def audio_ms(audio: bytes, sample_rate: int, channels: int = 1) -> int:
    """Return the duration of PCM16 ``audio`` in milliseconds."""
    return int(len(audio) * 1000 // (BYTES_PER_SAMPLE * max(sample_rate, 1) * max(channels, 1)))


def audio_seconds(audio: bytes, sample_rate: int, channels: int = 1) -> Fraction:
    """Return the exact duration of PCM16 ``audio`` in seconds, so that durations add up without rounding."""
    return Fraction(len(audio), BYTES_PER_SAMPLE * max(sample_rate, 1) * max(channels, 1))


class LiveInputGate(FrameProcessor):
    """Counts caller audio against the input media clock and drops it while muted.

    The clock counts exactly the caller audio that reaches it, muted audio too, and turns it into milliseconds only when
    it is read, so it never falls behind real time.
    """

    def __init__(self, session: SessionPort, **kwargs):
        """Bind the session whose input clock this gate advances."""
        super().__init__(**kwargs)
        self.session = session
        self.muted = False
        self._seconds = Fraction(0)

    @property
    def input_ms(self) -> int:
        """Milliseconds of caller audio seen so far (muted audio still advances the clock)."""
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
    """Publishes the caller's transcript deltas and pipeline errors as live events.

    The assistant's transcript is :class:`LiveOutputTranscript`, after the output, for the audio sent to the caller.
    The observer sees each frame at its first push only (``observe_every_push=False``), so an error frame, which each
    processor it passes pushes on, is reported once.
    """

    def __init__(self, session: SessionPort, gate: LiveInputGate, stt, **kwargs):
        """Watch frames pushed by ``stt`` and publish them on ``session``."""
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
            # A processor this error left unusable ends the pipeline (ProcessorUnusablePolicy.END), and the session.
            self._report(str(frame.error)[:500], fatal=frame.processor is not None and not frame.processor.is_usable)


class LiveOutputTranscript(FrameProcessor):
    """The assistant's transcript: each sentence that played, as ``session.output_transcript.delta``.

    Placed after the output, behind what stops a sentence nobody heard
    (``examples.frontend_backend_live.media.silent_lines.SilentLines``), it reads each sentence's text once its audio
    has gone out and times it on the session's input clock, from its first audio to its text, so the client's
    transcript holds played sentences only. A line's end starts the clock afresh: audio after a line's last sentence
    (the resampler's tail) belongs to no sentence of the next line.
    """

    def __init__(self, session: SessionPort, **kwargs):
        """Send on ``session``."""
        super().__init__(**kwargs)
        self.session = session
        self._sentence_start_ms: int | None = None  # the input clock when the current sentence's audio began

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
    """Sends a primary WebSocket's client silence between lines: the continuous stream such a client expects.

    It sits right after the output transport and runs in direct mode, taking each frame in the task that pushed it: a
    line's start reaches it before the output sends the line's first audio, and the line's end once the output has
    sent its last audio. While the session is active, from each line's end (``TTSStoppedFrame``) or cut until the next
    line starts (``TTSStartedFrame``), it sends 100 ms of silence every 100 ms through the serializer, never as a frame:
    what reads the call after the output (the lines the caller heard, the output transcript) sees only the reply's own
    audio. While a line's audio has run dry it sends nothing, so no silence lands inside a reply. After a stall it
    starts again from now instead of catching up.
    """

    def __init__(self, session: SessionPort, serializer: LiveFrameSerializer, **kwargs):
        """Send through ``serializer`` to ``session``'s primary connection."""
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
        """Every 100 ms, send silence while the session is active and no line is playing."""
        try:
            next_at = time.monotonic()
            while True:
                next_at += SILENCE_PACKET_SECONDS
                await asyncio.sleep(max(next_at - time.monotonic(), 0))
                now = time.monotonic()
                if now - next_at > SILENCE_PACKET_SECONDS:  # woke more than a tick late, a stall: start again from now
                    next_at = now
                if self._between_lines and self.session.is_active():
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
