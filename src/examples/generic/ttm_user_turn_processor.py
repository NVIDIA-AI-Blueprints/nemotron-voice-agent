# SPDX-FileCopyrightText: Copyright (c) 2024–2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""Pipecat processor for TTM turn events with a Silero EOU fallback."""

from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import soxr
from loguru import logger
from pipecat.audio.vad.vad_analyzer import VADParams, VADState
from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    Frame,
    InputAudioRawFrame,
    StartFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from websockets.asyncio.client import connect as websocket_connect

_MODEL_SAMPLE_RATE = 16_000
_FALLBACK_VAD_STOP_SECS = 0.2


@dataclass(frozen=True)
class _DrainCommand:
    future: asyncio.Future[None]


class TTMUserTurnProcessor(FrameProcessor):
    """Stream audio to TTM and own all emitted Pipecat user-turn frames.

    Native TTM SOU and EOU events are authoritative. When configured, a private
    Silero analyzer closes an active SOU after sustained silence if TTM's EOU
    is delayed or missing. Silero never emits independent VAD frames.
    """

    def __init__(
        self,
        *,
        url: str | None = None,
        headers: dict[str, str] | None = None,
        enable_interruptions: bool = True,
        open_timeout: float = 3.0,
        close_timeout: float = 3.0,
        max_pending_chunks: int = 250,
        silence_fallback_secs: float = 0.0,
        **kwargs,
    ) -> None:
        """Initialize the TTM transport and optional silence fallback."""
        super().__init__(**kwargs)
        if not math.isfinite(open_timeout) or open_timeout <= 0:
            raise ValueError("open_timeout must be greater than zero")
        if not math.isfinite(close_timeout) or close_timeout <= 0:
            raise ValueError("close_timeout must be greater than zero")
        if isinstance(max_pending_chunks, bool) or max_pending_chunks <= 0:
            raise ValueError("max_pending_chunks must be greater than zero")
        if (
            isinstance(silence_fallback_secs, bool)
            or not math.isfinite(silence_fallback_secs)
            or silence_fallback_secs < 0
        ):
            raise ValueError("silence_fallback_secs must be non-negative")

        self._url = url or "ws://127.0.0.1:7860/v1/audio/turn-events"
        self._headers = dict(headers or {})
        self._enable_interruptions = enable_interruptions
        self._open_timeout = float(open_timeout)
        self._close_timeout = float(close_timeout)
        self._max_pending_chunks = int(max_pending_chunks)
        self._silence_fallback_secs = float(silence_fallback_secs)

        self._vad_analyzer = None
        if self._silence_fallback_secs > 0:
            from pipecat.audio.vad.silero import SileroVADAnalyzer

            self._vad_analyzer = SileroVADAnalyzer(
                params=VADParams(stop_secs=_FALLBACK_VAD_STOP_SECS)
            )
        self._vad_state = VADState.QUIET
        self._silence_fallback_task: asyncio.Task[None] | None = None

        self._websocket = None
        self._send_queue: asyncio.Queue[bytes | _DrainCommand] | None = None
        self._send_task: asyncio.Task[None] | None = None
        self._receive_task: asyncio.Task[None] | None = None
        self._resampler: soxr.ResampleStream | None = None
        self._resampler_input_rate: int | None = None
        self._sequence = 0
        self._speaking = False
        self._active_segment_id: str | None = None
        self._stopping = False
        self._failed = False

    @property
    def speaking(self) -> bool:
        """Return whether TTM currently has an active user turn."""
        return self._speaking

    async def cleanup(self):
        """Close TTM and release Silero resources."""
        await self._disconnect()
        if self._vad_analyzer is not None:
            await self._vad_analyzer.cleanup()
        await super().cleanup()

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Forward frames and stream downstream input audio to TTM."""
        await super().process_frame(frame, direction)

        if isinstance(frame, StartFrame):
            await self.push_frame(frame, direction)
            if self._vad_analyzer is not None:
                self._vad_analyzer.set_sample_rate(_MODEL_SAMPLE_RATE)
                self._vad_state = VADState.QUIET
            await self._connect()
        elif isinstance(frame, EndFrame):
            await self._finish_audio()
            self._stopping = True
            await self._disconnect()
            await self.push_frame(frame, direction)
        elif isinstance(frame, CancelFrame):
            self._stopping = True
            await self._disconnect()
            await self.push_frame(frame, direction)
        elif isinstance(frame, InputAudioRawFrame) and direction == FrameDirection.DOWNSTREAM:
            await self.push_frame(frame, direction)
            await self._queue_audio_frame(frame)
        else:
            await self.push_frame(frame, direction)

    async def _connect(self) -> None:
        await self._disconnect()
        self._stopping = False
        self._failed = False
        self._sequence = 0
        self._speaking = False
        self._active_segment_id = None
        try:
            self._websocket = await websocket_connect(
                self._url,
                additional_headers=self._headers,
                open_timeout=self._open_timeout,
                close_timeout=self._close_timeout,
            )
        except Exception as error:
            await self._transport_failed(error)
            return
        self._send_queue = asyncio.Queue(maxsize=self._max_pending_chunks)
        self._send_task = self.create_task(self._send_audio(), name="send_ttm_audio")
        self._receive_task = self.create_task(
            self._receive_events(),
            name="receive_ttm_turn_events",
        )

    async def _disconnect(self) -> None:
        self._stopping = True
        await self._cancel_silence_fallback()
        websocket = self._websocket
        self._websocket = None
        if websocket is not None:
            with contextlib.suppress(Exception):
                await websocket.close()
        if self._receive_task is not None:
            task = self._receive_task
            self._receive_task = None
            if task is not asyncio.current_task():
                await self.cancel_task(task, timeout=self._close_timeout)
        if self._send_task is not None:
            task = self._send_task
            self._send_task = None
            if task is not asyncio.current_task():
                await self.cancel_task(task, timeout=self._close_timeout)
        self._fail_pending(RuntimeError("TTM user-turn transport closed"))
        self._send_queue = None
        self._resampler = None
        self._resampler_input_rate = None
        self._speaking = False
        self._active_segment_id = None

    async def _queue_audio_frame(self, frame: InputAudioRawFrame) -> None:
        if self._failed or self._websocket is None:
            return
        try:
            audio = self._mono_pcm(frame)
            if frame.sample_rate == _MODEL_SAMPLE_RATE:
                await self._queue_and_analyze_pcm(self._flush_resampler())
                await self._queue_and_analyze_pcm(audio)
                return

            if self._resampler_input_rate != frame.sample_rate:
                await self._queue_and_analyze_pcm(self._flush_resampler())
                self._resampler = soxr.ResampleStream(
                    frame.sample_rate,
                    _MODEL_SAMPLE_RATE,
                    num_channels=1,
                    dtype="int16",
                    quality="VHQ",
                )
                self._resampler_input_rate = frame.sample_rate
            assert self._resampler is not None
            samples = np.frombuffer(audio, dtype="<i2")
            output = self._resampler.resample_chunk(samples, last=False)
            await self._queue_and_analyze_pcm(
                output.astype("<i2", copy=False).tobytes()
            )
        except Exception as error:
            await self._transport_failed(error)

    @staticmethod
    def _mono_pcm(frame: InputAudioRawFrame) -> bytes:
        if frame.sample_rate <= 0:
            raise ValueError("audio sample rate must be greater than zero")
        if frame.num_channels <= 0:
            raise ValueError("audio channel count must be greater than zero")
        frame_width = frame.num_channels * 2
        if len(frame.audio) % frame_width:
            raise ValueError("audio must contain complete PCM S16LE frames")
        if frame.num_channels == 1:
            return bytes(frame.audio)

        samples = np.frombuffer(frame.audio, dtype="<i2").reshape(-1, frame.num_channels)
        mono = np.rint(samples.astype(np.float64).mean(axis=1))
        return np.clip(mono, -32768, 32767).astype("<i2").tobytes()

    def _flush_resampler(self) -> bytes:
        if self._resampler is None:
            return b""
        output = self._resampler.resample_chunk(np.empty(0, dtype=np.int16), last=True)
        self._resampler = None
        self._resampler_input_rate = None
        return output.astype("<i2", copy=False).tobytes()

    async def _finish_audio(self) -> None:
        if self._failed or self._send_queue is None:
            return
        try:
            tail = self._flush_resampler()
            if tail:
                await asyncio.wait_for(
                    self._send_queue.put(tail),
                    timeout=self._close_timeout,
                )
            future = asyncio.get_running_loop().create_future()
            await asyncio.wait_for(
                self._send_queue.put(_DrainCommand(future)),
                timeout=self._close_timeout,
            )
            await asyncio.wait_for(future, timeout=self._close_timeout)
        except Exception as error:
            await self._transport_failed(error)

    def _queue_pcm(self, audio: bytes) -> None:
        if not audio or self._send_queue is None:
            return
        try:
            self._send_queue.put_nowait(audio)
        except asyncio.QueueFull as error:
            raise RuntimeError("TTM audio queue is full") from error

    async def _queue_and_analyze_pcm(self, audio: bytes) -> None:
        if not audio:
            return
        self._queue_pcm(audio)
        if self._vad_analyzer is None:
            return
        previous_state = self._vad_state
        self._vad_state = await self._vad_analyzer.analyze_audio(audio)
        if self._vad_state == VADState.SPEAKING:
            await self._cancel_silence_fallback()
        elif self._vad_state == VADState.QUIET and previous_state != VADState.QUIET:
            await self._start_silence_fallback()

    async def _send_audio(self) -> None:
        try:
            assert self._send_queue is not None
            while True:
                command = await self._send_queue.get()
                try:
                    if isinstance(command, _DrainCommand):
                        if not command.future.done():
                            command.future.set_result(None)
                        continue
                    if self._websocket is None:
                        raise ConnectionError("TTM turn-event connection is not open")
                    await self._websocket.send(
                        json.dumps(
                            {
                                "type": "audio.chunk",
                                "sequence": self._sequence,
                                "data": base64.b64encode(command).decode("ascii"),
                            }
                        )
                    )
                    self._sequence += 1
                finally:
                    self._send_queue.task_done()
        except asyncio.CancelledError:
            raise
        except Exception as error:
            await self._transport_failed(error)

    async def _receive_events(self) -> None:
        try:
            assert self._websocket is not None
            async for raw in self._websocket:
                if not isinstance(raw, str):
                    raise ValueError("TTM turn event must be a JSON text message")
                await self._handle_message(json.loads(raw))
            if not self._stopping:
                raise ConnectionError("TTM turn-event connection closed")
        except asyncio.CancelledError:
            raise
        except Exception as error:
            await self._transport_failed(error)

    async def _handle_message(self, message: Any) -> None:
        if isinstance(message, dict) and message.get("type") == "error":
            raise RuntimeError(
                f"{message.get('code', 'error')}: {message.get('message', '')}"
            )
        event, segment_id = self._parse_event(message)
        if event == "SOU":
            if self._speaking:
                logger.warning("Ignoring duplicate TTM SOU event")
                return
            self._speaking = True
            self._active_segment_id = segment_id
            await self.broadcast_frame(UserStartedSpeakingFrame)
            if self._enable_interruptions:
                await self.broadcast_interruption()
            if self._vad_analyzer is not None and self._vad_state == VADState.QUIET:
                await self._start_silence_fallback()
            return

        if not self._speaking:
            logger.warning("Ignoring TTM EOU event without an active SOU")
            return
        if segment_id != self._active_segment_id:
            raise ValueError("TTM EOU segment_id does not match the active SOU")
        await self._emit_user_stopped("TTM EOU")

    async def _emit_user_stopped(self, reason: str) -> None:
        self._speaking = False
        self._active_segment_id = None
        await self._cancel_silence_fallback()
        logger.info("Ending user turn via {}", reason)
        await self.broadcast_frame(UserStoppedSpeakingFrame)

    async def _start_silence_fallback(self) -> None:
        if not self._speaking or self._silence_fallback_secs <= 0:
            return
        await self._cancel_silence_fallback()
        self._silence_fallback_task = self.create_task(
            self._silence_fallback(),
            name="ttm_silence_fallback",
        )

    async def _cancel_silence_fallback(self) -> None:
        task = self._silence_fallback_task
        self._silence_fallback_task = None
        if task is not None and task is not asyncio.current_task():
            await self.cancel_task(task)

    async def _silence_fallback(self) -> None:
        try:
            await asyncio.sleep(self._silence_fallback_secs)
            if self._speaking and self._vad_state == VADState.QUIET:
                logger.warning(
                    "TTM EOU missing after {:.1f}s of silence; using Silero fallback",
                    self._silence_fallback_secs,
                )
                await self._emit_user_stopped("Silero silence fallback")
        finally:
            if self._silence_fallback_task is asyncio.current_task():
                self._silence_fallback_task = None

    @staticmethod
    def _parse_event(message: Any) -> tuple[str, str]:
        if not isinstance(message, dict):
            raise ValueError("TTM turn event is not a JSON object")
        expected = {"type", "event", "segment_id", "audio_offset_ms", "confidence"}
        if set(message) != expected or message["type"] != "turn.event":
            raise ValueError("TTM turn event has unexpected fields")
        event = message["event"]
        if event not in {"SOU", "EOU"}:
            raise ValueError("TTM turn event must be SOU or EOU")
        segment_id = message["segment_id"]
        if not isinstance(segment_id, str) or not segment_id:
            raise ValueError("TTM turn event segment_id must be a non-empty string")
        audio_offset_ms = message["audio_offset_ms"]
        if (
            isinstance(audio_offset_ms, bool)
            or not isinstance(audio_offset_ms, int)
            or audio_offset_ms < 0
        ):
            raise ValueError("TTM turn event audio_offset_ms must be non-negative")
        confidence = message["confidence"]
        if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
            raise ValueError("TTM turn event confidence must be numeric")
        if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
            raise ValueError("TTM turn event confidence is outside [0, 1]")
        return event, segment_id

    async def _transport_failed(self, error: Exception) -> None:
        if self._stopping or self._failed:
            return
        self._failed = True
        await self._cancel_silence_fallback()
        self._speaking = False
        self._active_segment_id = None
        self._resampler = None
        self._resampler_input_rate = None
        self._fail_pending(error)
        logger.error("TTM user-turn transport failed: {}", error)
        await self.push_error(
            "TTM user-turn transport failed",
            exception=error,
            fatal=True,
        )
        websocket = self._websocket
        self._websocket = None
        if websocket is not None:
            with contextlib.suppress(Exception):
                await websocket.close()

    def _fail_pending(self, error: Exception) -> None:
        if self._send_queue is None:
            return
        while True:
            try:
                command = self._send_queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            if isinstance(command, _DrainCommand) and not command.future.done():
                command.future.set_exception(error)
            self._send_queue.task_done()
