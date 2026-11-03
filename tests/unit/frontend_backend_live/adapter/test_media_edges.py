# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D103

"""The media edges of a live session's pipeline: rates, the input clock, the WebSocket clock."""

import asyncio
import os
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from pipecat.frames.frames import (
    EndFrame,
    InputAudioRawFrame,
    OutputAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from examples.frontend_backend_live.adapter.adapter import LIVE_AUDIO_IN_RATE
from examples.frontend_backend_live.adapter.bridge import LiveInputGate, LiveWebSocketAudioOutput
from examples.frontend_backend_live.adapter.pipeline_host import MEDIA_EDGES, PipelineHost, WebRtcEdges, WebSocketEdges
from examples.shared.pipeline_utils import PIPELINE_AUDIO_IN_SAMPLE_RATE, PIPELINE_AUDIO_OUT_SAMPLE_RATE
from live.ports import EXCHANGE_RATE
from tests.unit.frontend_backend_live.adapter.adapter_helpers import (
    PACKET_BYTES,
    Probe,
    RecordingPort,
    run_frames,
    tone,
)


class RateProbe(FrameProcessor):
    """Records the rates the pipeline hands its processors when it sets them up."""

    rates: tuple[int, int] | None = None

    async def setup(self, setup):
        await super().setup(setup)
        RateProbe.rates = (setup.audio_in_sample_rate, setup.audio_out_sample_rate)

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        await self.push_frame(frame, direction)


class PipelineRateTests(unittest.IsolatedAsyncioTestCase):
    async def test_the_pipeline_runs_at_nvas_shared_rates_so_magpie_gets_a_rate_it_makes(self):
        port = RecordingPort()
        host = SimpleNamespace(session=port, gate=None, task=None, runner_task=None)
        await PipelineHost.run_pipeline(host, [RateProbe()], stt=object(), tts=object())
        await host.task.queue_frame(EndFrame())
        async with asyncio.timeout(5):
            await host.runner_task
        self.assertEqual(RateProbe.rates, (PIPELINE_AUDIO_IN_SAMPLE_RATE, PIPELINE_AUDIO_OUT_SAMPLE_RATE))
        self.assertEqual(PIPELINE_AUDIO_OUT_SAMPLE_RATE, 22050)  # Magpie's highest rate
        self.assertEqual(LIVE_AUDIO_IN_RATE, PIPELINE_AUDIO_IN_SAMPLE_RATE)


class ReplyAudioRateTests(unittest.IsolatedAsyncioTestCase):
    async def test_a_22050_hz_reply_reaches_the_wire_at_the_exchange_rate_whole(self):
        port = RecordingPort()
        line = [TTSStartedFrame()]
        reply = tone(200, rate=22050)
        line += [
            OutputAudioRawFrame(audio=reply[i : i + 882], sample_rate=22050, num_channels=1)
            for i in range(0, len(reply), 882)  # 20 ms chunks, as a TTS sends them
        ]
        line.append(TTSStoppedFrame())
        await run_frames([LiveWebSocketAudioOutput(port)], line, gap=0.0, settle=0.5)
        self.assertEqual({rate for _, _, rate in port.packets()}, {EXCHANGE_RATE})
        # 200 ms at 24 kHz: the line's last samples, held in the resampler's filter, are flushed at its end.
        self.assertEqual(sum(len(pcm) for _, pcm, _ in port.voiced()), EXCHANGE_RATE // 5 * 2)

    def test_webrtc_reply_audio_goes_out_at_the_exchange_rate_in_the_configured_chunks(self):
        captured = {}

        class FakeWebRTC:
            def __init__(self, webrtc_connection, params):
                captured["params"] = params

            def input(self):
                return "webrtc-in"

            def output(self):
                return "webrtc-out"

        target = "pipecat.transports.smallwebrtc.transport.SmallWebRTCTransport"
        for setting, chunks in ((None, 5), ("8", 8)):
            with self.subTest(setting=setting), patch(target, FakeWebRTC), patch.dict(os.environ):
                os.environ.pop("AUDIO_OUT_10MS_CHUNKS", None)
                if setting:
                    os.environ["AUDIO_OUT_10MS_CHUNKS"] = setting
                host = PipelineHost()
                host.session = RecordingPort()
                processors = host.audio_pipeline(
                    ["frontend"], transport="webrtc", connection=object(), audio_in_rate=LIVE_AUDIO_IN_RATE
                )
                params = captured["params"]
                self.assertEqual(
                    (params.audio_in_sample_rate, params.audio_out_sample_rate, params.audio_out_10ms_chunks),
                    (LIVE_AUDIO_IN_RATE, EXCHANGE_RATE, chunks),
                )
                self.assertEqual(processors, ["webrtc-in", host.gate, "frontend", "webrtc-out", host.playout])

    def test_each_transport_is_a_class_of_the_edges_table(self):
        self.assertEqual(MEDIA_EDGES, {"webrtc": WebRtcEdges, "websocket": WebSocketEdges})
        host = PipelineHost()
        host.session = RecordingPort()
        processors = host.audio_pipeline(
            ["frontend"], transport="websocket", connection=None, audio_in_rate=LIVE_AUDIO_IN_RATE
        )
        self.assertEqual(processors[:2], [host.gate, "frontend"])
        self.assertIsInstance(processors[2], LiveWebSocketAudioOutput)
        self.assertIs(processors[3], host.playout)


class CallerAudioTests(unittest.IsolatedAsyncioTestCase):
    async def test_websocket_caller_audio_reaches_the_frontend_at_16_khz(self):
        port, probe = RecordingPort(), Probe()
        chunk = InputAudioRawFrame(audio=bytes(4800), sample_rate=EXCHANGE_RATE, num_channels=1)  # 100 ms at 24 kHz
        await run_frames([LiveInputGate(port), probe], [chunk] * 5, gap=0.0)
        heard = probe.of(InputAudioRawFrame)
        self.assertEqual({(frame.sample_rate, frame.num_channels) for frame in heard}, {(LIVE_AUDIO_IN_RATE, 1)})
        # About 500 ms at 16 kHz: the stream resampler holds a few milliseconds in its filter.
        self.assertAlmostEqual(sum(len(frame.audio) for frame in heard) / 32, 500, delta=30)

    async def test_audio_already_at_the_input_rate_passes_unchanged(self):
        port, probe = RecordingPort(), Probe()
        chunk = InputAudioRawFrame(audio=bytes(3200), sample_rate=LIVE_AUDIO_IN_RATE, num_channels=1)
        await run_frames([LiveInputGate(port), probe], [chunk], gap=0.0)
        self.assertEqual(probe.of(InputAudioRawFrame), [chunk])

    @staticmethod
    def gate(port):
        gate, passed = LiveInputGate(port), []

        async def record(frame, direction=FrameDirection.DOWNSTREAM):
            passed.append(frame)

        gate.push_frame = record
        return gate, passed

    async def test_the_input_clock_counts_samples_so_it_never_falls_behind(self):
        port = RecordingPort()
        gate, _ = self.gate(port)
        frame = InputAudioRawFrame(audio=bytes(200), sample_rate=16000, num_channels=1)  # 100 samples: 6.25 ms
        for _ in range(160):
            await gate.process_frame(frame, FrameDirection.DOWNSTREAM)
        # Counting samples, not rounded milliseconds per frame (6 ms each would give 960 ms).
        self.assertEqual((gate.input_ms, port.input_ms), (1000, 1000))

    async def test_the_clock_counts_the_audio_the_caller_sent_before_resampling(self):
        port = RecordingPort()
        gate, passed = self.gate(port)
        frame = InputAudioRawFrame(audio=bytes(4800), sample_rate=EXCHANGE_RATE, num_channels=1)
        for _ in range(30):
            await gate.process_frame(frame, FrameDirection.DOWNSTREAM)
        self.assertEqual(gate.input_ms, 3000)
        self.assertLess(sum(len(frame.audio) for frame in passed), 30 * 3200)  # the filter still holds a little


class WebSocketClockTests(unittest.IsolatedAsyncioTestCase):
    async def test_no_packet_goes_out_before_the_session_is_active(self):
        port = RecordingPort(active=False)
        output = LiveWebSocketAudioOutput(port)
        clock = asyncio.create_task(output._clock())
        self.addAsyncCleanup(self.cancel, clock)
        await asyncio.sleep(0.35)
        self.assertEqual(port.packets(), [])
        port.active = True
        await asyncio.sleep(0.35)
        self.assertGreaterEqual(len(port.packets()), 2)
        self.assertTrue(all(len(pcm) == PACKET_BYTES for _, pcm, _ in port.packets()))

    async def test_a_short_reply_goes_out_whole_with_no_padding(self):
        port = RecordingPort()
        frames = [TTSStartedFrame(), OutputAudioRawFrame(audio=tone(150), sample_rate=EXCHANGE_RATE, num_channels=1)]
        await run_frames([LiveWebSocketAudioOutput(port)], [*frames, TTSStoppedFrame()], gap=0.0, settle=0.45)
        self.assertEqual([len(pcm) for _, pcm, _ in port.voiced()], [PACKET_BYTES, PACKET_BYTES // 2])
        sounds = [bool(any(pcm)) for _, pcm, _ in port.packets()]
        first = sounds.index(True)
        self.assertEqual(sounds[first : first + 2], [True, True])  # back to back, no silence between them

    async def test_no_silence_goes_out_inside_a_reply_whose_audio_has_run_dry(self):
        port = RecordingPort()
        output = LiveWebSocketAudioOutput(port)
        started = [TTSStartedFrame(), OutputAudioRawFrame(audio=tone(100), sample_rate=EXCHANGE_RATE, num_channels=1)]
        task = asyncio.create_task(run_frames([output], [*started], gap=0.0, settle=0.6))
        await asyncio.sleep(0.55)
        sounds = [bool(any(pcm)) for _, pcm, _ in port.packets()]
        await task
        # The line's 100 ms went out last: nothing since, while its synthesis goes on (before it, maybe one silence).
        self.assertEqual(sounds[-1:], [True])
        self.assertEqual(sounds.count(True), 1)

    async def test_silence_flows_again_once_the_line_has_ended(self):
        port = RecordingPort()
        frames = [TTSStartedFrame(), OutputAudioRawFrame(audio=tone(100), sample_rate=EXCHANGE_RATE, num_channels=1)]
        await run_frames([LiveWebSocketAudioOutput(port)], [*frames, TTSStoppedFrame()], gap=0.0, settle=0.5)
        sounds = [bool(any(pcm)) for _, pcm, _ in port.packets()]
        after = sounds[sounds.index(True) + 1 :]
        self.assertGreaterEqual(len(after), 2)
        self.assertNotIn(True, after)

    async def test_after_a_stall_the_clock_starts_again_instead_of_catching_up(self):
        port = RecordingPort()
        output = LiveWebSocketAudioOutput(port)
        clock = asyncio.create_task(output._clock())
        self.addAsyncCleanup(self.cancel, clock)
        await asyncio.sleep(0.15)
        time.sleep(0.6)  # the event loop stalls for six ticks
        before = len(port.packets())
        await asyncio.sleep(0.25)
        self.assertLessEqual(len(port.packets()) - before, 4)  # a catch-up would send about eight

    @staticmethod
    async def cancel(task):
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


if __name__ == "__main__":
    unittest.main()
