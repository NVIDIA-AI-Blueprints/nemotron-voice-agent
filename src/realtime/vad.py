# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""Realtime-only VAD boundaries captured on the input-audio edge."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADAnalyzer, VADState
from pipecat.audio.vad.vad_controller import VADController
from pipecat.frames.frames import (
    CancelFrame,
    ControlFrame,
    EndFrame,
    Frame,
    InputAudioRawFrame,
    StartFrame,
    UninterruptibleFrame,
    VADParamsUpdateFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from realtime.audio import MAX_PENDING_INPUT_SECONDS

_PIPELINE_SAMPLE_RATE = 16_000
_MAX_VAD_FRAME_SAMPLES = 512
_SILERO_STATE_RESET_INTERVAL_SECS = 5.0
REALTIME_VAD_START_CONFIRMATION_SECONDS = 0.2


class RealtimeSileroVADAnalyzer(SileroVADAnalyzer):
    """Reset Silero recurrent state only while the detector is stably quiet."""

    def voice_confidence(self, buffer: bytes) -> float:
        """Preserve Pipecat inference while keeping timed resets out of speech."""
        now = time.time()
        previous_reset_time = self._last_reset_time
        self._last_reset_time = now
        confidence = super().voice_confidence(buffer)
        self._last_reset_time = previous_reset_time

        confidence_value = float(np.asarray(confidence).reshape(-1)[0])
        reset_due = now - previous_reset_time >= _SILERO_STATE_RESET_INTERVAL_SECS
        stably_quiet = self._vad_state == VADState.QUIET and confidence_value < self.params.confidence
        if reset_due and stably_quiet:
            self._model.reset_states()
            self._last_reset_time = now
        return confidence

    def _run_analyzer(self, buffer: bytes) -> VADState:
        """Start each confirmed speech turn with independent recurrent state."""
        previous_state = self._vad_state
        state = super()._run_analyzer(buffer)
        if state == VADState.QUIET and previous_state in {VADState.SPEAKING, VADState.STOPPING}:
            self._model.reset_states()
            self._last_reset_time = time.time()
        return state


def _validate_audio_boundary(sample: int, sample_rate: int) -> None:
    if isinstance(sample, bool) or not isinstance(sample, int) or sample < 0:
        raise ValueError("audio boundary sample must be a non-negative integer")
    if isinstance(sample_rate, bool) or not isinstance(sample_rate, int) or sample_rate <= 0:
        raise ValueError("sample_rate must be a positive integer")


@dataclass
class RealtimeVADUserStartedSpeakingFrame(VADUserStartedSpeakingFrame):
    """VAD start with its absolute pipeline-audio boundary."""

    audio_start_sample: int = 0
    sample_rate: int = _PIPELINE_SAMPLE_RATE

    def __post_init__(self) -> None:
        """Initialize the Pipecat frame and validate its absolute boundary."""
        super().__post_init__()
        _validate_audio_boundary(self.audio_start_sample, self.sample_rate)


@dataclass
class RealtimeVADUserStoppedSpeakingFrame(VADUserStoppedSpeakingFrame):
    """VAD stop with its absolute pipeline-audio boundary."""

    audio_stop_sample: int = 0
    sample_rate: int = _PIPELINE_SAMPLE_RATE

    def __post_init__(self) -> None:
        """Initialize the Pipecat frame and validate its absolute boundary."""
        super().__post_init__()
        _validate_audio_boundary(self.audio_stop_sample, self.sample_rate)


@dataclass
class RealtimeVADConfigurationFrame(ControlFrame, UninterruptibleFrame):
    """Declare the exact input history needed before a VAD start boundary."""

    prefix_padding_samples: int
    start_confirmation_samples: int
    sample_rate: int = _PIPELINE_SAMPLE_RATE

    def __post_init__(self) -> None:
        """Initialize the frame and validate its bounded sample history."""
        super().__post_init__()
        _validate_audio_boundary(self.prefix_padding_samples, self.sample_rate)
        _validate_audio_boundary(self.start_confirmation_samples, self.sample_rate)
        maximum_component_samples = self.sample_rate * MAX_PENDING_INPUT_SECONDS
        if self.prefix_padding_samples > maximum_component_samples:
            raise ValueError("Realtime VAD prefix exceeds the pipeline audio buffer limit")
        if self.start_confirmation_samples > maximum_component_samples:
            raise ValueError("Realtime VAD start confirmation exceeds the pipeline audio buffer limit")

    @property
    def start_lookback_samples(self) -> int:
        """Return the exact prefix plus detector-confirmation history."""
        return self.prefix_padding_samples + self.start_confirmation_samples


class RealtimeVADInputProcessor(FrameProcessor):
    """Detect VAD boundaries before audio reaches the Realtime ASR service.

    The processor packetizes arbitrary OpenAI audio appends into analysis-aligned
    chunks of at most 512 samples. A start boundary is pushed
    before the window that confirms speech; a stop boundary is pushed after
    the window that confirms silence. Absolute sample coordinates make the
    boundary independent of downstream queue skew.
    """

    def __init__(
        self,
        vad_analyzer: VADAnalyzer,
        *,
        vad_prefix_padding_secs: float = 0.0,
        **kwargs,
    ) -> None:
        """Create a Realtime input-edge VAD processor."""
        if (
            isinstance(vad_prefix_padding_secs, bool)
            or not isinstance(vad_prefix_padding_secs, int | float)
            or not math.isfinite(float(vad_prefix_padding_secs))
            or vad_prefix_padding_secs < 0
        ):
            raise ValueError("vad_prefix_padding_secs must be a finite non-negative number")
        if vad_prefix_padding_secs > MAX_PENDING_INPUT_SECONDS:
            raise ValueError(f"vad_prefix_padding_secs must be at most {MAX_PENDING_INPUT_SECONDS} seconds")
        super().__init__(**kwargs)
        self._vad_analyzer = vad_analyzer
        self._vad_prefix_padding_secs = float(vad_prefix_padding_secs)
        self._vad_controller = VADController(vad_analyzer, audio_idle_timeout=0.0)
        self._vad_controller.add_event_handler("on_speech_started", self._on_speech_started)
        self._vad_controller.add_event_handler("on_speech_stopped", self._on_speech_stopped)
        self._vad_controller.add_event_handler("on_push_frame", self._on_controller_push_frame)
        self._vad_controller.add_event_handler("on_broadcast_frame", self._on_controller_broadcast_frame)
        self._processed_samples = 0
        self._audio_buffer_floor_sample = 0
        self._analysis_end_sample: int | None = None
        self._pending_start: RealtimeVADUserStartedSpeakingFrame | None = None
        self._pending_stop: RealtimeVADUserStoppedSpeakingFrame | None = None
        self._vad_window_samples = 0
        self._vad_window_fill = 0
        self._controller_ready = False
        self._controller_cleaned = False

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        """Analyze downstream PCM before forwarding it and its boundaries."""
        if (
            isinstance(frame, StartFrame)
            and direction == FrameDirection.DOWNSTREAM
            and frame.audio_in_sample_rate != _PIPELINE_SAMPLE_RATE
        ):
            raise ValueError("Realtime VAD input must be mono 16 kHz audio")
        await super().process_frame(frame, direction)

        if isinstance(frame, StartFrame) and direction == FrameDirection.DOWNSTREAM:
            self._processed_samples = 0
            self._audio_buffer_floor_sample = 0
            self._vad_window_fill = 0
            self._pending_start = None
            self._pending_stop = None
            await self.push_frame(frame, direction)
            if not self._controller_ready:
                await self._vad_controller.setup(self.task_manager)
                self._controller_ready = True
            await self._vad_controller.process_frame(frame)
            self._vad_window_samples = int(self._vad_analyzer.num_frames_required())
            if self._vad_window_samples <= 0:
                raise ValueError("Realtime VAD analyzer must require at least one audio sample")
            await self.push_frame(self._configuration_frame(), direction)
            return

        if isinstance(frame, (EndFrame, CancelFrame)):
            await self.push_frame(frame, direction)
            await self._cleanup_controller()
            return

        if isinstance(frame, VADParamsUpdateFrame):
            await self._vad_controller.process_frame(frame)
            await self.push_frame(frame, direction)
            if self._controller_ready and self._vad_window_samples > 0:
                await self.push_frame(self._configuration_frame(), FrameDirection.DOWNSTREAM)
            return

        if direction != FrameDirection.DOWNSTREAM or not isinstance(frame, InputAudioRawFrame):
            await self.push_frame(frame, direction)
            return

        self._validate_audio(frame)
        if not self._controller_ready or self._vad_window_samples <= 0:
            raise RuntimeError("Realtime VAD received audio before StartFrame")
        offset = 0
        while offset < len(frame.audio):
            remaining_window_samples = self._vad_window_samples - self._vad_window_fill
            remaining_audio_samples = (len(frame.audio) - offset) // 2
            chunk_samples = min(
                _MAX_VAD_FRAME_SAMPLES,
                remaining_window_samples,
                remaining_audio_samples,
            )
            chunk_bytes = chunk_samples * 2
            audio = frame.audio[offset : offset + chunk_bytes]
            chunk = self._audio_chunk(frame, audio=audio, offset_bytes=offset)
            self._analysis_end_sample = self._processed_samples + chunk.num_frames
            self._pending_start = None
            self._pending_stop = None
            await self._vad_controller.process_frame(chunk)

            if self._pending_start is not None:
                await self.push_frame(self._pending_start, direction)
            await self.push_frame(chunk, direction)
            self._processed_samples = self._analysis_end_sample
            self._vad_window_fill = (self._vad_window_fill + chunk.num_frames) % self._vad_window_samples
            if self._pending_stop is not None:
                await self.push_frame(self._pending_stop, direction)
            self._analysis_end_sample = None
            offset += chunk_bytes

    async def cleanup(self) -> None:
        """Release the VAD analyzer and processor tasks."""
        await self._cleanup_controller()
        await super().cleanup()

    def _validate_audio(self, frame: InputAudioRawFrame) -> None:
        if frame.sample_rate != _PIPELINE_SAMPLE_RATE or frame.num_channels != 1:
            raise ValueError("Realtime VAD input must be mono 16 kHz audio")
        if len(frame.audio) % 2:
            raise ValueError("Realtime VAD input must contain complete signed 16-bit samples")

    @staticmethod
    def _audio_chunk(
        frame: InputAudioRawFrame,
        *,
        audio: bytes,
        offset_bytes: int,
    ) -> InputAudioRawFrame:
        chunk = InputAudioRawFrame(
            audio=audio,
            sample_rate=frame.sample_rate,
            num_channels=frame.num_channels,
        )
        chunk.metadata.update(frame.metadata)
        chunk.transport_source = frame.transport_source
        chunk.transport_destination = frame.transport_destination
        if frame.pts is not None:
            offset_samples = offset_bytes // (2 * frame.num_channels)
            chunk.pts = frame.pts + round(offset_samples * 1_000_000_000 / frame.sample_rate)
        return chunk

    def _vad_confirmation_samples(self, seconds: float) -> int:
        frames = max(1, round(seconds * _PIPELINE_SAMPLE_RATE / self._vad_window_samples))
        return frames * self._vad_window_samples

    def _configuration_frame(self) -> RealtimeVADConfigurationFrame:
        """Describe the exact detector lookbehind required by speech starts."""
        start_confirmation_samples = self._vad_confirmation_samples(float(self._vad_analyzer.params.start_secs))
        prefix_padding_samples = round(self._vad_prefix_padding_secs * _PIPELINE_SAMPLE_RATE)
        return RealtimeVADConfigurationFrame(
            prefix_padding_samples=prefix_padding_samples,
            start_confirmation_samples=start_confirmation_samples,
            sample_rate=_PIPELINE_SAMPLE_RATE,
        )

    async def _on_speech_started(self, _controller: VADController) -> None:
        if self._analysis_end_sample is None:
            return
        start_secs = max(0.0, float(self._vad_analyzer.params.start_secs))
        confirmation_samples = self._vad_confirmation_samples(start_secs)
        prefix_samples = round(self._vad_prefix_padding_secs * _PIPELINE_SAMPLE_RATE)
        audio_start_sample = max(
            self._audio_buffer_floor_sample,
            self._analysis_end_sample - confirmation_samples - prefix_samples,
        )
        self._pending_start = RealtimeVADUserStartedSpeakingFrame(
            start_secs=start_secs,
            audio_start_sample=audio_start_sample,
            sample_rate=_PIPELINE_SAMPLE_RATE,
        )

    async def _on_speech_stopped(self, _controller: VADController) -> None:
        if self._analysis_end_sample is None:
            return
        self._pending_stop = RealtimeVADUserStoppedSpeakingFrame(
            stop_secs=max(0.0, float(self._vad_analyzer.params.stop_secs)),
            audio_stop_sample=self._analysis_end_sample,
            sample_rate=_PIPELINE_SAMPLE_RATE,
        )
        self._audio_buffer_floor_sample = self._analysis_end_sample

    async def _on_controller_push_frame(
        self,
        _controller: VADController,
        frame: Frame,
        direction: FrameDirection = FrameDirection.DOWNSTREAM,
    ) -> None:
        await self.push_frame(frame, direction)

    async def _on_controller_broadcast_frame(
        self,
        _controller: VADController,
        frame_cls: type[Frame],
        **kwargs,
    ) -> None:
        await self.broadcast_frame(frame_cls, **kwargs)

    async def _cleanup_controller(self) -> None:
        if self._controller_cleaned:
            return
        self._controller_cleaned = True
        await self._vad_controller.cleanup()
