# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D103, D107

"""The Live edges of a session's pipeline: rates, the input clock, the WebSocket clock, cuts, transcript, logs."""

import asyncio
import os
import time
import unittest
from unittest.mock import patch

from pipecat.frames.frames import (
    InputAudioRawFrame,
    InterruptionFrame,
    LLMMessagesAppendFrame,
    OutputAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
    TTSTextFrame,
)
from pipecat.processors.frame_processor import FrameProcessor

from examples.frontend_backend_live.adapter.adapter import LIVE_AUDIO_IN_RATE
from examples.frontend_backend_live.adapter.bridge import LiveInputGate, LiveOutputTranscript, LiveWebSocketAudioOutput
from examples.frontend_backend_live.adapter.pipeline_host import PipelineHost
from examples.frontend_backend_live.media.preroll import SpeechPreroll
from gateways.live.ports import EXCHANGE_RATE
from tests.unit.frontend_backend_live.adapter.adapter_helpers import (
    FLUSH,
    PACKET_BYTES,
    Probe,
    RecordingPort,
    run_frames,
    tone,
)


def reply(ms: int, rate: int = EXCHANGE_RATE) -> OutputAudioRawFrame:
    return OutputAudioRawFrame(audio=tone(ms, rate=rate), sample_rate=rate, num_channels=1)


def edges(transport="websocket", session_id="live_test", **kwargs):
    host = PipelineHost()
    host.session = RecordingPort(session_id=session_id)
    return host.edges(transport=transport, audio_in_rate=LIVE_AUDIO_IN_RATE, stt=None, **kwargs)


class Clock(FrameProcessor):
    """Moves the session's input clock 100 ms on with each frame, and notes when each frame passed."""

    def __init__(self, port):
        super().__init__()
        self.port, self.times = port, []

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        self.port.input_ms += 100
        self.times.append((type(frame), self.port.input_ms))
        await self.push_frame(frame, direction)


class AudioTests(unittest.IsolatedAsyncioTestCase):
    async def test_a_22050_hz_reply_reaches_the_wire_at_the_exchange_rate_whole(self):
        port = RecordingPort()
        audio = tone(200, rate=22050)
        chunks = [
            OutputAudioRawFrame(audio=audio[i : i + 882], sample_rate=22050, num_channels=1)
            for i in range(0, len(audio), 882)
        ]
        await run_frames(
            [LiveWebSocketAudioOutput(port)], [TTSStartedFrame(), *chunks, TTSStoppedFrame()], gap=0.0, settle=0.5
        )
        self.assertEqual({rate for _, _, rate in port.packets()}, {EXCHANGE_RATE})
        # 200 ms at 24 kHz: the line's last samples, held in the resampler's filter, are flushed at its end.
        self.assertEqual(sum(len(pcm) for _, pcm, _ in port.packets() if any(pcm)), EXCHANGE_RATE // 5 * 2)

    def test_webrtc_reply_audio_goes_out_at_the_exchange_rate_in_the_configured_chunks(self):
        captured = {}

        class FakeWebRTC:
            def __init__(self, webrtc_connection, params):
                captured["params"] = params

            def input(self):
                return "webrtc"

            output = input

        target = "pipecat.transports.smallwebrtc.transport.SmallWebRTCTransport"
        for setting, chunks in ((None, 5), ("8", 8)):
            with self.subTest(setting=setting), patch(target, FakeWebRTC), patch.dict(os.environ):
                os.environ.pop("AUDIO_OUT_10MS_CHUNKS", None)
                os.environ.update({"AUDIO_OUT_10MS_CHUNKS": setting} if setting else {})
                edges("webrtc", connection=object())
                params = captured["params"]
                rates = (params.audio_in_sample_rate, params.audio_out_sample_rate, params.audio_out_10ms_chunks)
                self.assertEqual(rates, (LIVE_AUDIO_IN_RATE, EXCHANGE_RATE, chunks))

    async def test_caller_audio_reaches_the_frontend_at_16_khz_and_the_clock_counts_what_the_caller_sent(self):
        # Samples, not rounded milliseconds per frame, so the clock never falls behind (6 ms each would give 960 ms).
        for rate, size, count, ms in ((EXCHANGE_RATE, 4800, 5, 500), (LIVE_AUDIO_IN_RATE, 200, 160, 1000)):
            with self.subTest(rate=rate):
                port, probe = RecordingPort(), Probe()
                chunk = InputAudioRawFrame(audio=bytes(size), sample_rate=rate, num_channels=1)
                await run_frames([LiveInputGate(port), probe], [chunk] * count, gap=0.0)
                heard = probe.of(InputAudioRawFrame)
                self.assertEqual(
                    {(frame.sample_rate, frame.num_channels) for frame in heard}, {(LIVE_AUDIO_IN_RATE, 1)}
                )
                # The stream resampler holds a few milliseconds in its filter; audio at the input rate passes unchanged.
                self.assertAlmostEqual(sum(len(frame.audio) for frame in heard) / 32, ms, delta=30)
                self.assertEqual(port.input_ms, ms)

    def test_the_pipeline_is_traced_under_the_sessions_id_only_when_tracing_is_on_and_never_closed_when_idle(self):
        for setting, traced in ((None, False), ("", False), ("false", False), ("true", True)):
            with self.subTest(setting=setting), patch.dict(os.environ):
                os.environ.pop("ENABLE_TRACING", None)
                os.environ.update({"ENABLE_TRACING": setting} if setting is not None else {})
                worker = edges(session_id="live_traced", connection=None).worker
                self.assertEqual(worker["enable_tracing"], traced)
                if traced:
                    self.assertEqual(worker["conversation_id"], "live_traced")
                self.assertIsNone(worker["idle_timeout_secs"])  # the Live API leaves it to the app


class WebSocketClockTests(unittest.IsolatedAsyncioTestCase):
    async def test_no_packet_goes_out_before_the_session_is_active(self):
        port = RecordingPort(active=False)
        running = asyncio.create_task(run_frames([LiveWebSocketAudioOutput(port)], [], settle=0.7))
        await asyncio.sleep(0.4)
        self.assertEqual(port.packets(), [])
        port.active = True
        await running
        self.assertGreaterEqual(len(port.packets()), 2)
        self.assertTrue(all(len(pcm) == PACKET_BYTES for _, pcm, _ in port.packets()))

    async def test_a_short_reply_goes_out_whole_back_to_back_with_no_padding_then_silence_flows_again(self):
        port = RecordingPort()
        await run_frames(
            [LiveWebSocketAudioOutput(port)], [TTSStartedFrame(), reply(150), TTSStoppedFrame()], gap=0.0, settle=0.6
        )
        sounds = port.sounds()
        self.assertEqual([len(pcm) for _, pcm, _ in port.packets() if any(pcm)], [PACKET_BYTES, PACKET_BYTES // 2])
        first = sounds.index(True)
        self.assertEqual(sounds[first : first + 2], [True, True])
        self.assertGreaterEqual(len(sounds[first + 2 :]), 2)
        self.assertNotIn(True, sounds[first + 2 :])

    async def test_no_silence_goes_out_inside_a_reply_whose_audio_has_run_dry(self):
        port = RecordingPort()
        await run_frames([LiveWebSocketAudioOutput(port)], [TTSStartedFrame(), reply(100)], gap=0.0, settle=0.6)
        # The line's 100 ms went out last: nothing since, while its synthesis goes on (before it, maybe one silence).
        self.assertEqual((port.sounds()[-1:], port.sounds().count(True)), ([True], 1))

    async def test_after_a_stall_the_clock_starts_again_instead_of_catching_up(self):
        port = RecordingPort()
        running = asyncio.create_task(run_frames([LiveWebSocketAudioOutput(port)], [], settle=1.2))
        while not port.packets():
            await asyncio.sleep(0.01)
        time.sleep(0.6)  # the event loop stalls for six ticks
        before = len(port.packets())
        await asyncio.sleep(0.25)
        self.assertLessEqual(len(port.packets()) - before, 4)  # a catch-up would send about eight
        await running


class LineTests(unittest.IsolatedAsyncioTestCase):
    async def test_a_cut_keeps_what_no_interruption_may_drop_on_the_websocket_output_and_in_the_pre_roll(self):
        kept = LLMMessagesAppendFrame(messages=[{"role": "developer", "content": "result"}], run_llm=True)
        kept.interruptible = False  # a result on its way to the conversation
        dropped = LLMMessagesAppendFrame(messages=[], run_llm=True)
        line = [TTSStartedFrame(), *[reply(100) for _ in range(20)], kept, dropped, InterruptionFrame()]
        output, probe = LiveWebSocketAudioOutput(RecordingPort()), Probe()
        await run_frames([output, probe], line, gap=0.005, settle=0.4)  # two seconds of audio queued at the output
        self.assertEqual(probe.of(LLMMessagesAppendFrame), [kept])
        probe = Probe()
        await run_frames([SpeechPreroll(10_000), probe], [*line[:3], kept, FLUSH, InterruptionFrame()])
        self.assertEqual(
            [type(frame) for frame in probe.of(InterruptionFrame, LLMMessagesAppendFrame)],
            [InterruptionFrame, LLMMessagesAppendFrame],
        )
        self.assertFalse(probe.of(TTSStartedFrame))  # the held line itself goes

    async def test_each_sentence_that_played_goes_to_the_client_timed_from_its_own_lines_first_audio(self):
        # The WebSocket output queues the resampler's tail of a line after the line's last sentence.
        port = RecordingPort()
        port.input_ms, clock = 1000, Clock(port)

        def sentence(text):
            return TTSTextFrame(text=text, text_type="sentence")

        first = [TTSStartedFrame(), reply(100), sentence("One."), reply(100), TTSStoppedFrame()]
        second = [TTSStartedFrame(), reply(100), sentence("Two."), TTSStoppedFrame()]
        await run_frames([clock, LiveOutputTranscript(port)], [*first, *second])
        audio = [ms for kind, ms in clock.times if kind is OutputAudioRawFrame]
        deltas = port.transcript()
        self.assertEqual([delta["delta"] for delta in deltas], ["One. ", "Two. "])
        self.assertEqual([delta["start_ms"] for delta in deltas], [audio[0], audio[2]])
        self.assertTrue(all(delta["start_ms"] < delta["end_ms"] for delta in deltas))
