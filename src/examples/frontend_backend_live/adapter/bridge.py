# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pipecat processors that adapt a voice pipeline to the live session protocol.

Each works through the adapter's view of the session (:class:`gateways.live.ports.SessionPort`), never the session
itself.

* :class:`LiveInputGate` counts caller audio against the session's input clock, honors mute, and hands the frontend
  the caller's audio at the rate it hears.
* :class:`LiveEventObserver` turns pipeline frames into input transcript events and failure reports.
* :class:`LiveOutputTranscript` sends the sentences that played as output transcript events: placed after the output,
  it reads each sentence's text once its audio has gone out.
* :class:`LiveWebSocketAudioOutput` sends reply audio on a primary WebSocket in real time, keeping every other frame at
  its place in the audio; the session sends the audio as ``session.output_audio.delta`` events. WebRTC sessions use
  Pipecat's output transport. What the caller heard is read after either output
  (``examples.frontend_backend_live.media.silent_lines``).
"""

from __future__ import annotations

import asyncio
import time
from collections import deque
from fractions import Fraction
from typing import TYPE_CHECKING

from loguru import logger
from pipecat.audio.utils import create_stream_resampler
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    CancelFrame,
    EndFrame,
    ErrorFrame,
    Frame,
    InputAudioRawFrame,
    InterruptionFrame,
    OutputAudioRawFrame,
    StartFrame,
    SystemFrame,
    TranscriptionFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
    TTSTextFrame,
    UserStartedSpeakingFrame,
)
from pipecat.observers.base_observer import BaseObserver, FramePushed
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from examples.shared.pipeline_utils import PIPELINE_AUDIO_IN_SAMPLE_RATE

# The PCM16 the adapter exchanges with the gateway, caller and reply audio.
from gateways.live.ports import EXCHANGE_RATE

if TYPE_CHECKING:
    from gateways.live.ports import SessionPort

BYTES_PER_SAMPLE = 2
OUTPUT_PACKET_SECONDS = 0.1
# The frames that start and end a line: a WebSocket packet never carries audio of two lines.
LINE_BOUNDARIES = (TTSStartedFrame, TTSStoppedFrame)


def audio_ms(audio: bytes, sample_rate: int, channels: int = 1) -> int:
    """Return the duration of PCM16 ``audio`` in milliseconds."""
    return int(len(audio) * 1000 // (BYTES_PER_SAMPLE * max(sample_rate, 1) * max(channels, 1)))


def audio_seconds(audio: bytes, sample_rate: int, channels: int = 1) -> Fraction:
    """Return the exact duration of PCM16 ``audio`` in seconds, so that durations add up without rounding."""
    return Fraction(len(audio), BYTES_PER_SAMPLE * max(sample_rate, 1) * max(channels, 1))


class LiveInputGate(FrameProcessor):
    """Counts caller audio against the input media clock, drops it while muted, and passes it on at the input rate.

    The clock counts exactly the audio the caller sent, before any resampling, and turns it into milliseconds only when
    it is read, so it never falls behind real time. Audio at another rate than the frontend hears (a primary
    WebSocket's exchange rate) is resampled to ``audio_in_rate``; audio already at that rate (WebRTC) passes unchanged.
    """

    def __init__(self, session: SessionPort, audio_in_rate: int = PIPELINE_AUDIO_IN_SAMPLE_RATE, **kwargs):
        """Bind the session whose input clock this gate advances, and the rate the frontend hears the caller at."""
        super().__init__(**kwargs)
        self.session = session
        self.audio_in_rate = audio_in_rate
        self.muted = False
        self._seconds = Fraction(0)
        self._resampler = create_stream_resampler()

    @property
    def input_ms(self) -> int:
        """Milliseconds of caller audio seen so far (muted audio still advances the clock)."""
        return int(self._seconds * 1000)

    async def process_frame(self, frame, direction: FrameDirection) -> None:
        """Advance the clock for every audio frame; pass audio on at the input rate only when not muted."""
        await super().process_frame(frame, direction)
        if isinstance(frame, InputAudioRawFrame):
            self._seconds += audio_seconds(frame.audio, frame.sample_rate, frame.num_channels)
            await self.session.advance_input_clock(self.input_ms)
            if self.muted:
                return
            frame = await self._at_input_rate(frame)
            if frame is None:
                return
        await self.push_frame(frame, direction)

    async def _at_input_rate(self, frame: InputAudioRawFrame) -> InputAudioRawFrame | None:
        """``frame`` at the input rate; ``None`` while the resampler still fills its filter."""
        if frame.sample_rate == self.audio_in_rate:
            return frame
        audio = await self._resampler.resample(frame.audio, frame.sample_rate, self.audio_in_rate)
        if not audio:
            return None
        return InputAudioRawFrame(audio=audio, sample_rate=self.audio_in_rate, num_channels=frame.num_channels)


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


class LiveWebSocketAudioOutput(FrameProcessor):
    """Sends reply audio on a primary WebSocket in real time, with every other frame kept at its place in the audio.

    Reply audio is resampled to the exchange rate and queued, in order, with the frames that come between it: a
    sentence's text after the sentence's audio, a line's end after its last audio. A frame with no audio queued ahead
    of it passes at once, and system frames always do. Once the session is active, a clock sends one packet of up to
    100 ms of queued audio per tick, then pushes downstream the audio it sent and the frames it passed, in queue order:
    a processor after this output sees each frame once the caller has been sent everything before it.

    A packet never carries audio of two lines and is never padded. Between lines the clock sends 100 ms of silence per
    tick, the continuous stream a primary WebSocket client expects; while a line is still being synthesized and its
    audio has run dry, it sends nothing, so no silence lands inside a reply. After a stall the clock starts again from
    now instead of catching up. A barge-in drops everything queued but the frames no interruption may drop. The
    processor reports when the bot starts and stops speaking, so turn-taking works without Pipecat's output transport.
    """

    def __init__(self, session: SessionPort, **kwargs):
        """Bind the session whose primary connection receives the audio."""
        super().__init__(**kwargs)
        self.session = session
        self._queue: deque[bytearray | Frame] = deque()  # audio runs and the frames between them, in order
        self._releasing = asyncio.Lock()  # one task at a time hands frames on, so none overtakes another
        self._worker: asyncio.Task | None = None
        self._speaking = False
        self._tts_active = False  # a line is being synthesized: its end has not reached this output yet
        self._cuts = 0  # how many cuts so far: a packet whose send a cut overtook releases nothing
        self._resampler = create_stream_resampler(clear_after_secs=None)  # line ends flush it, cuts reset it
        self._packet_bytes = int(EXCHANGE_RATE * OUTPUT_PACKET_SECONDS) * BYTES_PER_SAMPLE

    async def process_frame(self, frame, direction: FrameDirection) -> None:
        """Queue audio and the frames between it; pass system and upstream frames at once; drop it all on a cut."""
        await super().process_frame(frame, direction)
        if direction is FrameDirection.UPSTREAM:
            await self.push_frame(frame, direction)
        elif isinstance(frame, StartFrame):
            self._worker = asyncio.create_task(self._clock(), name="live-ws-output")
            await self.push_frame(frame, direction)
        elif isinstance(frame, OutputAudioRawFrame):
            self._queue_audio(await self._resampler.resample(frame.audio, frame.sample_rate, EXCHANGE_RATE))
        elif isinstance(frame, InterruptionFrame):
            await self._cut()
            await self.push_frame(frame, direction)
        elif isinstance(frame, (EndFrame, CancelFrame)):
            await self._stop()
            await self.push_frame(frame, direction)
        elif isinstance(frame, SystemFrame):
            await self.push_frame(frame, direction)
        else:
            if isinstance(frame, TTSStartedFrame):
                self._tts_active = True
            elif isinstance(frame, TTSStoppedFrame):
                self._tts_active = False
                self._queue_audio(await self._resampler.flush())  # the line's last audio, held in the filter
            await self._queue_frame(frame)

    def _queue_audio(self, audio: bytes) -> None:
        if not audio:
            return
        if self._queue and isinstance(self._queue[-1], bytearray):
            self._queue[-1].extend(audio)
        else:
            self._queue.append(bytearray(audio))

    async def _queue_frame(self, frame: Frame) -> None:
        async with self._releasing:
            if self._queue:
                self._queue.append(frame)
            else:
                await self.push_frame(frame)

    async def _cut(self) -> None:
        # What an interruption may not drop, such as a result on its way to the conversation, stays in its place.
        self._queue = deque(item for item in self._queue if isinstance(item, Frame) and not item.interruptible)
        self._cuts += 1
        self._tts_active = False  # the cut ended any line in progress; the next one starts with its own frame
        await self._resampler.reset()
        await self._set_speaking(False)

    async def _set_speaking(self, speaking: bool) -> None:
        if speaking == self._speaking:
            return
        self._speaking = speaking
        await self.broadcast_frame(BotStartedSpeakingFrame if speaking else BotStoppedSpeakingFrame)

    async def _clock(self) -> None:
        """Once the session is active, run one tick every 100 ms; never catch up after a stall."""
        try:
            while not self.session.is_active():
                await asyncio.sleep(OUTPUT_PACKET_SECONDS)
            next_at = time.monotonic()
            while True:
                next_at += OUTPUT_PACKET_SECONDS
                await asyncio.sleep(max(next_at - time.monotonic(), 0))
                now = time.monotonic()
                if now - next_at > OUTPUT_PACKET_SECONDS:  # woke more than a tick late, a stall: start again from now
                    next_at = now
                await self._tick()
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("WebSocket audio output failed")
            await self.session.report_error("Output audio transport failed; start a new session.", fatal=True)
            self.session.request_close("connection_lost")

    async def _tick(self) -> None:
        """Send one packet: queued audio, silence between lines, or nothing while a line's audio has run dry."""
        async with self._releasing:
            await self._release_leading_frames()
            if not self._queue:
                if not self._tts_active:
                    await self._set_speaking(False)
                    await self.session.send_output_audio(bytes(self._packet_bytes), EXCHANGE_RATE)
                return
            cuts = self._cuts
            packet, carried = self._take_packet()
            await self._set_speaking(True)
            await self.session.send_output_audio(packet, EXCHANGE_RATE)
            for item in carried:
                if cuts != self._cuts:  # a cut came meanwhile: what the packet carried is dropped with the rest
                    return
                await self.push_frame(item)
            await self._release_leading_frames()

    def _take_packet(self) -> tuple[bytes, list[Frame]]:
        """Up to one packet of audio from the head of the queue, and what it carries, in queue order.

        What it carries is its audio pieces and the frames between them. It stops early at a line boundary.
        """
        packet = bytearray()
        carried: list[Frame] = []
        while self._queue and len(packet) < self._packet_bytes:
            head = self._queue[0]
            if isinstance(head, bytearray):
                piece = bytes(head[: self._packet_bytes - len(packet)])
                del head[: len(piece)]
                if not head:
                    self._queue.popleft()
                packet += piece
                carried.append(OutputAudioRawFrame(audio=piece, sample_rate=EXCHANGE_RATE, num_channels=1))
            elif isinstance(head, LINE_BOUNDARIES):
                break
            else:
                carried.append(self._queue.popleft())
        return bytes(packet), carried

    async def _release_leading_frames(self) -> None:
        """Push the frames at the head of the queue: everything queued before them has been sent."""
        while self._queue and not isinstance(self._queue[0], bytearray):
            await self.push_frame(self._queue.popleft())

    async def _stop(self) -> None:
        if self._worker:
            self._worker.cancel()
            await asyncio.wait({self._worker})  # not gather: see gateways.live.events.fanout.Subscription.close
            self._worker = None
