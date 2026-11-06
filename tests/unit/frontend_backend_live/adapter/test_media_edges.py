# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D103, D107

"""The Live edges of a session's pipeline: rates, the input clock, the WebSocket's silence, cuts, transcript, logs."""

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
from examples.frontend_backend_live.adapter.bridge import LiveInputGate, LiveOutputTranscript
from examples.frontend_backend_live.adapter.pipeline_host import WEBRTC_AUDIO_OUT_RATE, PipelineHost
from examples.frontend_backend_live.adapter.serializer import LiveFrameSerializer
from examples.frontend_backend_live.media.preroll import SpeechPreroll
from examples.shared import audio_recorder
from gateways.live.protocol import AudioFormat
from tests.unit.frontend_backend_live.adapter.adapter_helpers import (
    FLUSH,
    PACKET_BYTES,
    WIRE_RATE,
    FakeSocket,
    Probe,
    RecordingPort,
    run_frames,
    tone,
)


def reply(ms: int, rate: int = WIRE_RATE) -> OutputAudioRawFrame:
    return OutputAudioRawFrame(audio=tone(ms, rate=rate), sample_rate=rate, num_channels=1)


def edges(transport="websocket", session_id="live_test", port=None, **kwargs):
    host = PipelineHost()
    host.session = port or RecordingPort(session_id=session_id)
    # The shared audio recorder's switches at their defaults, whatever the environment sets: the edges have no recorder.
    with (
        patch.object(audio_recorder, "ENABLE_ASR_DUMP", False),
        patch.object(audio_recorder, "ENABLE_TTS_DUMP", False),
    ):
        return host.edges(transport=transport, audio_in_rate=LIVE_AUDIO_IN_RATE, stt=None, **kwargs)


def websocket_output(port):
    """A primary WebSocket's output edge for ``port``: Pipecat's output transport, then the silence between lines."""
    return list(edges(port=port, connection=None).outputs)


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
    async def test_a_22050_hz_reply_reaches_the_client_at_the_wire_rate_whole(self):
        for audio_format in (AudioFormat(), AudioFormat(rate=16000)):
            with self.subTest(rate=audio_format.rate):
                port = RecordingPort(socket=FakeSocket(audio_format))
                audio = tone(200, rate=22050)
                chunks = [
                    OutputAudioRawFrame(audio=audio[i : i + 882], sample_rate=22050, num_channels=1)
                    for i in range(0, len(audio), 882)
                ]
                frames = [TTSStartedFrame(), *chunks, TTSStoppedFrame()]
                await run_frames(websocket_output(port), frames, gap=0.0, settle=0.5)
                packet = audio_format.rate // 10 * 2  # 100 ms at the wire rate
                self.assertEqual({len(audio) for audio in port.packets()}, {packet})
                # 200 ms: the line's last samples, held in the resampler's filter, are flushed at its end.
                self.assertEqual(sum(len(audio) for audio in port.packets() if any(audio)), 2 * packet)

    def test_webrtc_reply_audio_goes_out_at_24_khz_in_the_configured_chunks(self):
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
                self.assertEqual(rates, (LIVE_AUDIO_IN_RATE, WEBRTC_AUDIO_OUT_RATE, chunks))

    def test_a_websockets_transport_reads_the_sessions_socket_and_writes_at_its_wire_rate(self):
        captured = {}

        class FakeWebSocketTransport:
            def __init__(self, websocket, params):
                captured.update(websocket=websocket, params=params)

            def input(self):
                return "websocket input"

            def output(self):
                return "websocket output"

        target = "pipecat.transports.websocket.fastapi.FastAPIWebsocketTransport"
        socket = FakeSocket(AudioFormat(type="audio/pcmu"))
        for setting, chunks in ((None, 10), ("8", 8)):
            with self.subTest(setting=setting), patch(target, FakeWebSocketTransport), patch.dict(os.environ):
                os.environ.pop("AUDIO_OUT_10MS_CHUNKS", None)
                os.environ.update({"AUDIO_OUT_10MS_CHUNKS": setting} if setting else {})
                built = edges(port=RecordingPort(socket=socket), connection=None)
                params = captured["params"]
                self.assertIs(captured["websocket"], socket)
                self.assertIsInstance(params.serializer, LiveFrameSerializer)
                rates = (params.audio_in_sample_rate, params.audio_out_sample_rate, params.audio_out_10ms_chunks)
                self.assertEqual(rates, (LIVE_AUDIO_IN_RATE, 8000, chunks))
                # The gateway checked the origin; the session's own silence fills the time between lines.
                self.assertEqual((params.allowed_origins, params.audio_out_end_silence_secs), ([], 0))
                self.assertEqual(built.inputs[0], "websocket input")
                self.assertEqual(built.outputs[0], "websocket output")

    async def test_the_clock_counts_the_caller_audio_that_reaches_it_and_passes_it_unchanged(self):
        # Samples, not rounded milliseconds per frame, so the clock never falls behind (6 ms each would give 960 ms).
        port, probe = RecordingPort(), Probe()
        chunk = InputAudioRawFrame(audio=bytes(200), sample_rate=LIVE_AUDIO_IN_RATE, num_channels=1)
        await run_frames([LiveInputGate(port), probe], [chunk] * 160, gap=0.0)
        self.assertEqual([frame.audio for frame in probe.of(InputAudioRawFrame)], [bytes(200)] * 160)
        self.assertEqual(port.input_ms, 1000)

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

    def test_the_pipeline_runs_with_pipecats_metrics_and_usage_metrics_on(self):
        params = edges(connection=None).worker["params"]
        self.assertEqual((params.enable_metrics, params.enable_usage_metrics), (True, True))


class SilenceTests(unittest.IsolatedAsyncioTestCase):
    async def test_no_packet_goes_out_before_the_session_is_active(self):
        port = RecordingPort(active=False)
        running = asyncio.create_task(run_frames(websocket_output(port), [], settle=0.7))
        try:
            await asyncio.sleep(0.4)
            self.assertEqual(port.packets(), [])
            port.active = True
        finally:
            await running
        self.assertGreaterEqual(len(port.packets()), 2)
        self.assertEqual(set(port.packets()), {bytes(PACKET_BYTES)})  # 100 ms of silence each

    async def test_a_short_reply_goes_out_whole_back_to_back_then_silence_flows_again(self):
        port = RecordingPort()
        frames = [TTSStartedFrame(), reply(150), TTSStoppedFrame()]
        await run_frames(websocket_output(port), frames, gap=0.0, settle=0.6)
        sounds = port.sounds()
        first = sounds.index(True)
        self.assertEqual(sounds[first : first + 2], [True, True])
        voiced = b"".join(port.packets()[first : first + 2])
        self.assertEqual(voiced, tone(150) + bytes(len(tone(50))))  # the line's last packet is filled with silence
        self.assertGreaterEqual(len(sounds[first + 2 :]), 2)
        self.assertNotIn(True, sounds[first + 2 :])
        self.assertEqual({len(audio) for audio in port.packets()}, {PACKET_BYTES})

    async def test_no_silence_goes_out_inside_a_reply_whose_audio_has_run_dry(self):
        port = RecordingPort()
        await run_frames(websocket_output(port), [TTSStartedFrame(), reply(100)], gap=0.0, settle=0.6)
        # The line's 100 ms went out last: nothing since, while its synthesis goes on (before it, maybe one silence).
        self.assertEqual((port.sounds()[-1:], port.sounds().count(True)), ([True], 1))

    async def test_a_line_with_no_audio_and_a_cut_each_bring_the_silence_back(self):
        for ending in ([TTSStoppedFrame()], [reply(100), InterruptionFrame()]):
            with self.subTest(ending=[type(frame).__name__ for frame in ending]):
                port = RecordingPort()
                await run_frames(websocket_output(port), [TTSStartedFrame(), *ending], gap=0.05, settle=0.4)
                self.assertEqual(port.sounds()[-3:], [False] * 3)

    async def test_the_silence_never_reaches_what_reads_the_call_after_the_output(self):
        port, probe = RecordingPort(), Probe()
        await run_frames([*websocket_output(port), probe], [TTSStartedFrame(), reply(100), TTSStoppedFrame()])
        await asyncio.sleep(0.3)
        self.assertGreater(port.sounds().count(False), 0)
        self.assertEqual([frame.audio for frame in probe.of(OutputAudioRawFrame)], [tone(100)])  # the reply's own

    async def test_a_g711_session_gets_its_silence_in_g711(self):
        for kind, code in (("audio/pcmu", b"\xff"), ("audio/pcma", b"\xd5")):
            with self.subTest(kind=kind):
                port = RecordingPort(socket=FakeSocket(AudioFormat(type=kind)))
                await run_frames(websocket_output(port), [], settle=0.35)
                self.assertGreaterEqual(len(port.packets()), 2)
                self.assertEqual(set(port.packets()), {code * 800})  # 100 ms at 8 kHz, one byte a sample

    async def test_after_a_stall_the_clock_starts_again_instead_of_catching_up(self):
        port = RecordingPort()
        running = asyncio.create_task(run_frames(websocket_output(port), [], settle=1.2))
        try:
            while not port.packets():
                await asyncio.sleep(0.01)
            time.sleep(0.6)  # the event loop stalls for six ticks
            before = len(port.packets())
            await asyncio.sleep(0.25)
            self.assertLessEqual(len(port.packets()) - before, 4)  # a catch-up would send about eight
        finally:
            await running


class LineTests(unittest.IsolatedAsyncioTestCase):
    async def test_a_cut_keeps_what_no_interruption_may_drop_on_the_websocket_output_and_in_the_pre_roll(self):
        kept = LLMMessagesAppendFrame(messages=[{"role": "developer", "content": "result"}], run_llm=True)
        kept.interruptible = False  # a result on its way to the conversation
        dropped = LLMMessagesAppendFrame(messages=[], run_llm=True)
        line = [TTSStartedFrame(), *[reply(100) for _ in range(20)], kept, dropped, InterruptionFrame()]
        probe = Probe()
        await run_frames([*websocket_output(RecordingPort()), probe], line, gap=0.005, settle=0.4)  # 2 s queued
        self.assertEqual(probe.of(LLMMessagesAppendFrame), [kept])
        probe = Probe()
        await run_frames([SpeechPreroll(10_000), probe], [*line[:3], kept, FLUSH, InterruptionFrame()])
        self.assertEqual(
            [type(frame) for frame in probe.of(InterruptionFrame, LLMMessagesAppendFrame)],
            [InterruptionFrame, LLMMessagesAppendFrame],
        )
        self.assertFalse(probe.of(TTSStartedFrame))  # the held line itself goes

    async def test_each_sentence_that_played_goes_to_the_client_timed_from_its_own_lines_first_audio(self):
        # The output queues the resampler's tail of a line after the line's last sentence.
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
