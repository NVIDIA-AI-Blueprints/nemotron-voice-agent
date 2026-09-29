# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""Realtime input-edge VAD boundary tests."""

from __future__ import annotations

import unittest
from typing import cast
from unittest.mock import AsyncMock, MagicMock, patch

from pipecat.audio.vad.vad_analyzer import VADAnalyzer, VADParams, VADState
from pipecat.frames.frames import (
    InputAudioRawFrame,
    SpeechControlParamsFrame,
    StartFrame,
    VADParamsUpdateFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from realtime.vad import (
    RealtimeVADConfigurationFrame,
    RealtimeVADInputProcessor,
    RealtimeVADUserStartedSpeakingFrame,
    RealtimeVADUserStoppedSpeakingFrame,
)


class _ScriptedVADAnalyzer:
    def __init__(self, states: list[VADState], *, params: VADParams | None = None) -> None:
        self.params = params or VADParams(start_secs=0.0, stop_secs=0.0)
        self.states = iter(states)
        self.sample_rate = 0
        self.cleaned = False

    def set_sample_rate(self, sample_rate: int) -> None:
        self.sample_rate = sample_rate

    def set_params(self, params: VADParams) -> None:
        self.params = params

    def num_frames_required(self) -> int:
        return 512

    async def analyze_audio(self, _audio: bytes) -> VADState:
        return next(self.states)

    async def cleanup(self) -> None:
        self.cleaned = True


class _BufferedScriptedVADAnalyzer(_ScriptedVADAnalyzer):
    def __init__(self, states: list[VADState]) -> None:
        super().__init__(states)
        self._audio = b""
        self._state = VADState.QUIET

    async def analyze_audio(self, audio: bytes) -> VADState:
        self._audio += audio
        if len(self._audio) < 1_024:
            return self._state
        if len(self._audio) != 1_024:
            raise AssertionError("test analyzer received more than one VAD window")
        self._audio = b""
        self._state = next(self.states)
        return self._state


class RealtimeVADInputProcessorTests(unittest.IsolatedAsyncioTestCase):
    """Verify exact Realtime VAD packetization and boundary ordering."""

    async def _start_processor(self, processor: RealtimeVADInputProcessor) -> None:
        processor._task_manager = MagicMock()
        with (
            patch.object(FrameProcessor, "process_frame", AsyncMock()),
            patch.object(processor, "push_frame", AsyncMock()),
        ):
            await processor.process_frame(
                StartFrame(audio_in_sample_rate=16_000),
                FrameDirection.DOWNSTREAM,
            )

    async def test_boundaries_and_audio_are_byte_exact_and_forward_ordered(self) -> None:
        """Boundaries bracket the confirming audio without changing any bytes."""
        analyzer = _ScriptedVADAnalyzer([VADState.QUIET, VADState.SPEAKING, VADState.QUIET])
        processor = RealtimeVADInputProcessor(
            cast(VADAnalyzer, analyzer),
            vad_prefix_padding_secs=0.01,
        )
        await self._start_processor(processor)
        source = bytes(index % 251 for index in range(3 * 512 * 2))
        frame = InputAudioRawFrame(audio=source, sample_rate=16_000, num_channels=1)
        frame.metadata["source"] = "wire"
        frame.transport_source = "websocket"
        frame.transport_destination = "pipeline"
        frame.pts = 1_000_000_000
        pushed: list[tuple[FrameDirection, object]] = []

        async def record(output, direction=FrameDirection.DOWNSTREAM) -> None:
            pushed.append((direction, output))

        with patch.object(processor, "push_frame", side_effect=record):
            await processor.process_frame(frame, FrameDirection.DOWNSTREAM)

        outputs = [output for _, output in pushed]
        self.assertEqual(
            [type(output) for output in outputs],
            [
                InputAudioRawFrame,
                RealtimeVADUserStartedSpeakingFrame,
                InputAudioRawFrame,
                InputAudioRawFrame,
                RealtimeVADUserStoppedSpeakingFrame,
            ],
        )
        audio = [output for output in outputs if isinstance(output, InputAudioRawFrame)]
        self.assertEqual([len(output.audio) for output in audio], [1_024, 1_024, 1_024])
        self.assertEqual(b"".join(output.audio for output in audio), source)
        self.assertEqual([output.pts for output in audio], [1_000_000_000, 1_032_000_000, 1_064_000_000])
        self.assertTrue(all(output.metadata["source"] == "wire" for output in audio))
        self.assertTrue(all(output.transport_source == "websocket" for output in audio))
        self.assertTrue(all(output.transport_destination == "pipeline" for output in audio))

        started = cast(RealtimeVADUserStartedSpeakingFrame, outputs[1])
        stopped = cast(RealtimeVADUserStoppedSpeakingFrame, outputs[-1])
        self.assertEqual((started.audio_start_sample, started.sample_rate, started.start_secs), (352, 16_000, 0.0))
        self.assertEqual((stopped.audio_stop_sample, stopped.sample_rate, stopped.stop_secs), (1_536, 16_000, 0.0))

    async def test_vad_windows_remain_sample_aligned_across_appends(self) -> None:
        """Complete VAD windows exactly even when appends end mid-window."""
        analyzer = _BufferedScriptedVADAnalyzer([VADState.SPEAKING])
        processor = RealtimeVADInputProcessor(cast(VADAnalyzer, analyzer))
        await self._start_processor(processor)
        first = InputAudioRawFrame(audio=b"\x01\x00" * 100, sample_rate=16_000, num_channels=1)
        second = InputAudioRawFrame(audio=b"\x02\x00" * 600, sample_rate=16_000, num_channels=1)
        pushed: list[object] = []

        async def record(output, _direction=FrameDirection.DOWNSTREAM) -> None:
            pushed.append(output)

        with patch.object(processor, "push_frame", side_effect=record):
            await processor.process_frame(first, FrameDirection.DOWNSTREAM)
            await processor.process_frame(second, FrameDirection.DOWNSTREAM)

        audio = [output for output in pushed if isinstance(output, InputAudioRawFrame)]
        started = [output for output in pushed if isinstance(output, RealtimeVADUserStartedSpeakingFrame)]
        self.assertEqual([output.num_frames for output in audio], [100, 412, 188])
        self.assertEqual(b"".join(output.audio for output in audio), first.audio + second.audio)
        self.assertEqual(len(started), 1)
        self.assertEqual(started[0].audio_start_sample, 0)

    async def test_next_turn_prefix_does_not_cross_prior_auto_commit(self) -> None:
        """Clamp prefix audio to the current OpenAI input buffer."""
        analyzer = _ScriptedVADAnalyzer(
            [VADState.SPEAKING, VADState.QUIET, VADState.SPEAKING],
            params=VADParams(start_secs=0.0, stop_secs=0.0),
        )
        processor = RealtimeVADInputProcessor(
            cast(VADAnalyzer, analyzer),
            vad_prefix_padding_secs=0.1,
        )
        await self._start_processor(processor)
        pushed: list[object] = []

        async def record(output, _direction=FrameDirection.DOWNSTREAM) -> None:
            pushed.append(output)

        with patch.object(processor, "push_frame", side_effect=record):
            await processor.process_frame(
                InputAudioRawFrame(audio=b"\x00\x00" * (3 * 512), sample_rate=16_000, num_channels=1),
                FrameDirection.DOWNSTREAM,
            )

        starts = [frame for frame in pushed if isinstance(frame, RealtimeVADUserStartedSpeakingFrame)]
        stops = [frame for frame in pushed if isinstance(frame, RealtimeVADUserStoppedSpeakingFrame)]
        self.assertEqual([frame.audio_start_sample for frame in starts], [0, 1_024])
        self.assertEqual([frame.audio_stop_sample for frame in stops], [1_024])

    async def test_start_and_param_metadata_follow_start_frame(self) -> None:
        """Initialize VAD after StartFrame and propagate runtime VAD updates."""
        analyzer = _ScriptedVADAnalyzer([])
        processor = RealtimeVADInputProcessor(cast(VADAnalyzer, analyzer))
        processor._task_manager = MagicMock()
        pushed: list[tuple[FrameDirection, object]] = []

        async def record(output, direction=FrameDirection.DOWNSTREAM) -> None:
            pushed.append((direction, output))

        with (
            patch.object(FrameProcessor, "process_frame", AsyncMock()),
            patch.object(processor, "push_frame", side_effect=record),
        ):
            await processor.process_frame(StartFrame(audio_in_sample_rate=16_000), FrameDirection.DOWNSTREAM)
            updated = VADParams(confidence=0.8, start_secs=0.0, stop_secs=0.3, min_volume=0.5)
            update = VADParamsUpdateFrame(params=updated)
            await processor.process_frame(update, FrameDirection.UPSTREAM)

        self.assertIsInstance(pushed[0][1], StartFrame)
        self.assertEqual(pushed[0][0], FrameDirection.DOWNSTREAM)
        initial_metadata = pushed[1:3]
        self.assertTrue(all(isinstance(frame, SpeechControlParamsFrame) for _, frame in initial_metadata))
        self.assertEqual(
            [direction for direction, _ in initial_metadata],
            [FrameDirection.DOWNSTREAM, FrameDirection.UPSTREAM],
        )
        self.assertEqual(analyzer.sample_rate, 16_000)
        self.assertIs(analyzer.params, updated)
        self.assertIs(pushed[-2][1], update)
        self.assertEqual(pushed[-2][0], FrameDirection.UPSTREAM)
        refreshed = cast(RealtimeVADConfigurationFrame, pushed[-1][1])
        self.assertIsInstance(refreshed, RealtimeVADConfigurationFrame)
        self.assertEqual(refreshed.start_lookback_samples, 512)
        self.assertEqual(pushed[-1][0], FrameDirection.DOWNSTREAM)

        await processor.cleanup()
        self.assertTrue(analyzer.cleaned)

    def test_maximum_prefix_keeps_detector_confirmation_separate(self) -> None:
        """Advertise the exact lookback without reducing the public prefix maximum."""
        analyzer = _ScriptedVADAnalyzer([], params=VADParams(start_secs=0.2, stop_secs=0.5))
        processor = RealtimeVADInputProcessor(
            cast(VADAnalyzer, analyzer),
            vad_prefix_padding_secs=60,
        )
        processor._vad_window_samples = 512

        config = processor._configuration_frame()

        self.assertEqual(config.prefix_padding_samples, 960_000)
        self.assertEqual(config.start_confirmation_samples, 3_072)
        self.assertEqual(config.start_lookback_samples, 963_072)

    async def test_rejects_non_pipeline_pcm_and_invalid_prefix(self) -> None:
        """Reject unsupported audio rather than silently reinterpret its samples."""
        analyzer = cast(VADAnalyzer, _ScriptedVADAnalyzer([]))
        with self.assertRaises(ValueError):
            RealtimeVADInputProcessor(analyzer, vad_prefix_padding_secs=float("nan"))

        processor = RealtimeVADInputProcessor(analyzer)
        with self.assertRaises(ValueError):
            await processor.process_frame(
                StartFrame(audio_in_sample_rate=24_000),
                FrameDirection.DOWNSTREAM,
            )
        invalid = (
            InputAudioRawFrame(audio=b"\x00\x00", sample_rate=8_000, num_channels=1),
            InputAudioRawFrame(audio=b"\x00\x00\x00\x00", sample_rate=16_000, num_channels=2),
            InputAudioRawFrame(audio=b"\x00", sample_rate=16_000, num_channels=1),
        )
        for frame in invalid:
            with self.subTest(frame=frame), self.assertRaises(ValueError):
                await processor.process_frame(frame, FrameDirection.DOWNSTREAM)


if __name__ == "__main__":
    unittest.main()
