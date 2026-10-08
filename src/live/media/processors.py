# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pipecat processors that adapt a voice pipeline to the live session protocol.

* :class:`LiveInputGate` counts caller audio against the session's input clock and honors mute.
* :class:`LiveEventObserver` turns pipeline frames into transcript, error and usage events.
* :class:`LiveWebSocketAudioOutput` sends reply audio as ``session.output_audio.delta`` events on a steady clock,
  including silence, as a primary WebSocket client expects. WebRTC sessions use Pipecat's output transport.
"""

from __future__ import annotations

import asyncio
import base64
import re
import time

from loguru import logger
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
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

from live.session import LiveProtocolSession

# Pipecat's warning for one TTS context that produced no audio (typically after a barge-in cut the reply short).
# Nothing is lost for the caller, so it is logged, not sent as an ``error`` event; a run of them is a different,
# fatal message and is still reported.
SILENT_TTS_CONTEXT = re.compile(r"^TTS context \S+ completed with no audio$")
BYTES_PER_SAMPLE = 2
OUTPUT_RATE = 24000
OUTPUT_PACKET_SECONDS = 0.1


def audio_ms(audio: bytes, sample_rate: int, channels: int = 1) -> int:
    """Return the duration of PCM16 ``audio`` in milliseconds."""
    return int(len(audio) * 1000 // (BYTES_PER_SAMPLE * max(sample_rate, 1) * max(channels, 1)))


class LiveInputGate(FrameProcessor):
    """Counts caller audio against the input media clock and drops it while the session is muted."""

    def __init__(self, session: LiveProtocolSession, **kwargs):
        """Bind the session whose input clock this gate advances."""
        super().__init__(**kwargs)
        self.session = session
        self.muted = False
        self._input_ms = 0.0

    @property
    def input_ms(self) -> int:
        """Milliseconds of caller audio seen so far (muted audio still advances the clock)."""
        return int(self._input_ms)

    async def process_frame(self, frame, direction: FrameDirection) -> None:
        """Advance the clock for every audio frame; pass audio through only when not muted."""
        await super().process_frame(frame, direction)
        if isinstance(frame, InputAudioRawFrame):
            self._input_ms += audio_ms(frame.audio, frame.sample_rate, frame.num_channels)
            await self.session.advance_input_clock(self.input_ms)
            if self.muted:
                return
        await self.push_frame(frame, direction)


class LiveEventObserver(BaseObserver):
    """Publishes transcript deltas and pipeline errors as live events."""

    def __init__(self, session: LiveProtocolSession, gate: LiveInputGate, stt, tts, **kwargs):
        """Watch frames pushed by ``stt`` and ``tts`` and publish them on ``session``."""
        super().__init__(**kwargs)
        self.session, self.gate, self.stt, self.tts = session, gate, stt, tts
        self._speech_started_ms = 0
        self._output_ms = 0
        self._tasks: set[asyncio.Task] = set()

    def _emit(self, kind: str, **fields) -> None:
        task = asyncio.create_task(self.session.emit(kind, **fields))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def on_push_frame(self, data: FramePushed) -> None:
        """Translate the frames that carry transcripts, errors and speech timing."""
        frame = data.frame
        # A frontend that is both the speech-to-text and the speech out (a realtime model) pushes this frame itself.
        if isinstance(frame, UserStartedSpeakingFrame) and (data.source is not self.stt or self.stt is self.tts):
            self._speech_started_ms = self.gate.input_ms
        elif isinstance(frame, TranscriptionFrame) and data.source is self.stt and frame.text.strip():
            end_ms = self.gate.input_ms
            self._emit(
                "session.input_transcript.delta",
                delta=frame.text if frame.text.endswith(" ") else frame.text + " ",
                start_ms=min(self._speech_started_ms, end_ms),
                end_ms=end_ms,
            )
        elif isinstance(frame, OutputAudioRawFrame) and data.source is self.tts:
            self._output_ms += audio_ms(frame.audio, frame.sample_rate, frame.num_channels)
        elif isinstance(frame, TTSTextFrame) and data.source is self.tts and frame.text.strip():
            # The word is spoken after the audio synthesized so far; its length approximates its span.
            start_ms = self._output_ms
            self._emit(
                "session.output_transcript.delta",
                delta=frame.text if frame.text.endswith(" ") else frame.text + " ",
                start_ms=start_ms,
                end_ms=start_ms + max(60 * len(frame.text.strip()), 120),
            )
        elif isinstance(frame, ErrorFrame) and not frame.fatal:
            if SILENT_TTS_CONTEXT.match(str(frame.error)):
                logger.info(f"Ignoring a silent TTS context: {frame.error}")
                return
            self._emit(
                "error",
                error={
                    "type": "server_error",
                    "code": "pipeline_error",
                    "message": str(frame.error)[:500],
                    "param": None,
                    "client_event_id": None,
                },
            )


class LiveWebSocketAudioOutput(FrameProcessor):
    """Sends reply audio on a primary WebSocket as a steady, clocked stream of audio events.

    The primary WebSocket carries a continuous stream including silence, in packets of about 100 ms. Reply audio
    is queued, a barge-in drops what has not been sent, and the processor reports when the bot starts and stops
    speaking so turn-taking works without Pipecat's output transport.
    """

    def __init__(self, session: LiveProtocolSession, **kwargs):
        """Bind the session whose primary connection receives the audio."""
        super().__init__(**kwargs)
        self.session = session
        self._buffer = bytearray()
        self._worker: asyncio.Task | None = None
        self._speaking = False
        self._tts_active = False
        self._packet_bytes = int(OUTPUT_RATE * OUTPUT_PACKET_SECONDS) * BYTES_PER_SAMPLE

    async def process_frame(self, frame, direction: FrameDirection) -> None:
        """Queue audio, track speaking state, and pass non-audio frames downstream."""
        await super().process_frame(frame, direction)
        if isinstance(frame, StartFrame):
            self._worker = asyncio.create_task(self._clock(), name="live-ws-output")
        elif isinstance(frame, OutputAudioRawFrame):
            self._buffer.extend(frame.audio)
            return
        elif isinstance(frame, TTSStartedFrame):
            self._tts_active = True
        elif isinstance(frame, TTSStoppedFrame):
            self._tts_active = False
        elif isinstance(frame, InterruptionFrame):
            self._buffer.clear()
            await self._set_speaking(False)
        elif isinstance(frame, (EndFrame, CancelFrame)):
            await self._stop()
        await self.push_frame(frame, direction)

    async def _set_speaking(self, speaking: bool) -> None:
        if speaking == self._speaking:
            return
        self._speaking = speaking
        await self.broadcast_frame(BotStartedSpeakingFrame if speaking else BotStoppedSpeakingFrame)

    async def _clock(self) -> None:
        """Emit one packet every 100 ms: queued reply audio, or silence."""
        next_at = time.monotonic()
        try:
            while True:
                next_at += OUTPUT_PACKET_SECONDS
                await asyncio.sleep(max(next_at - time.monotonic(), 0))
                if self._buffer:
                    await self._set_speaking(True)
                    packet = bytes(self._buffer[: self._packet_bytes]).ljust(self._packet_bytes, b"\0")
                    del self._buffer[: self._packet_bytes]
                else:
                    packet = bytes(self._packet_bytes)
                    if self._speaking and not self._tts_active:
                        await self._set_speaking(False)
                wire = self.session.codec.encode(packet)
                if wire:
                    await self.session.publish(
                        {"type": "session.output_audio.delta", "delta": base64.b64encode(wire).decode()},
                        audience="primary",
                    )
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("WebSocket audio output failed")
            await self.session.fail("Output audio transport failed; start a new session.")
            self.session.spawn(self.session.close("connection_lost"), "output-failure-close")

    async def _stop(self) -> None:
        if self._worker:
            self._worker.cancel()
            await asyncio.gather(self._worker, return_exceptions=True)
            self._worker = None
