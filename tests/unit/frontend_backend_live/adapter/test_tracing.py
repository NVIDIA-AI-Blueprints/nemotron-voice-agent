# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D103

"""Observability of a live session's pipeline, the Pipecat half."""

import asyncio
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from loguru import logger
from pipecat.frames.frames import EndFrame, InputAudioRawFrame, UserStartedSpeakingFrame, UserStoppedSpeakingFrame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from examples.frontend_backend_live.adapter import pipeline_host
from examples.frontend_backend_live.adapter.bridge import LiveInputGate
from examples.frontend_backend_live.adapter.pipeline_host import PipelineHost
from tests.unit.frontend_backend_live.adapter.adapter_helpers import RecordingPort, tone


class Passthrough(FrameProcessor):
    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        await self.push_frame(frame, direction)


class TracingTests(unittest.IsolatedAsyncioTestCase):
    async def worker_arguments(self, setting):
        """Run a host's pipeline with ``ENABLE_TRACING`` set to ``setting``; return the worker's arguments."""
        seen = {}
        real = pipeline_host.PipelineWorker

        def worker(*args, **kwargs):
            seen.update(kwargs)
            return real(*args, **kwargs)

        port = RecordingPort(session_id="live_traced")
        host = SimpleNamespace(session=port, gate=None, task=None, runner_task=None)
        with patch.object(pipeline_host, "PipelineWorker", worker), patch.dict(os.environ):
            os.environ.pop("ENABLE_TRACING", None)
            if setting is not None:
                os.environ["ENABLE_TRACING"] = setting
            await PipelineHost.run_pipeline(host, [Passthrough()], stt=object(), tts=object())
        await host.task.queue_frame(EndFrame())
        async with asyncio.timeout(5):
            await host.runner_task
        return seen

    async def test_the_pipeline_is_traced_under_the_sessions_id_when_tracing_is_on(self):
        seen = await self.worker_arguments("true")
        self.assertEqual((seen["enable_tracing"], seen["conversation_id"]), (True, "live_traced"))

    async def test_tracing_is_off_unless_it_is_turned_on(self):
        for setting in (None, "", "false"):
            with self.subTest(setting=setting):
                self.assertFalse((await self.worker_arguments(setting))["enable_tracing"])

    async def test_no_server_idle_close(self):
        self.assertIsNone((await self.worker_arguments(None))["idle_timeout_secs"])  # GPT-Live leaves it to the app


class CallerLevelTests(unittest.IsolatedAsyncioTestCase):
    async def test_each_caller_turn_logs_one_line_with_its_length_and_level(self):
        lines: list[str] = []
        sink = logger.add(lambda message: lines.append(message.record["message"]), level="INFO")
        self.addCleanup(logger.remove, sink)
        port = RecordingPort(session_id="live_level")
        gate = LiveInputGate(port)

        async def forward(frame, direction=FrameDirection.DOWNSTREAM):
            pass

        gate.push_frame = forward
        half_scale = InputAudioRawFrame(audio=tone(20, rate=16000, level=16384), sample_rate=16000, num_channels=1)
        await gate.process_frame(half_scale, FrameDirection.DOWNSTREAM)  # before the turn: not counted
        await gate.process_frame(UserStartedSpeakingFrame(), FrameDirection.UPSTREAM)  # the aggregator's broadcast
        for _ in range(50):  # one second of audio at half of full scale
            await gate.process_frame(half_scale, FrameDirection.DOWNSTREAM)
        await gate.process_frame(UserStoppedSpeakingFrame(), FrameDirection.UPSTREAM)
        turns = [line for line in lines if line.startswith("Caller turn:")]
        self.assertEqual(turns, ["Caller turn: session=live_level speech_ms=1000 peak_dbfs=-6.0 rms_dbfs=-6.0"])

    async def test_a_turn_with_no_audio_logs_silence(self):
        lines: list[str] = []
        sink = logger.add(lambda message: lines.append(message.record["message"]), level="INFO")
        self.addCleanup(logger.remove, sink)
        gate = LiveInputGate(RecordingPort())

        async def forward(frame, direction=FrameDirection.DOWNSTREAM):
            pass

        gate.push_frame = forward
        await gate.process_frame(UserStartedSpeakingFrame(), FrameDirection.UPSTREAM)
        await gate.process_frame(UserStoppedSpeakingFrame(), FrameDirection.UPSTREAM)
        self.assertIn("peak_dbfs=-inf rms_dbfs=-inf", [line for line in lines if line.startswith("Caller turn:")][0])


if __name__ == "__main__":
    unittest.main()
