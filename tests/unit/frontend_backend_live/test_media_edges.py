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
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineWorker
from pipecat.processors.audio.audio_buffer_processor import AudioBufferProcessor
from pipecat.processors.filters.identity_filter import IdentityFilter
from pipecat.processors.frame_processor import FrameProcessor

from examples.frontend_backend_live.adapter.adapter import LIVE_AUDIO_IN_RATE
from examples.frontend_backend_live.adapter.bridge import LiveInputGate, LiveOutputTranscript
from examples.frontend_backend_live.adapter.pipeline_host import WEBRTC_AUDIO_OUT_RATE, PipelineHost
from examples.frontend_backend_live.adapter.serializer import LiveFrameSerializer
from examples.frontend_backend_live.media.preroll import SpeechPreroll
from examples.shared import audio_recorder
from gateways.live.protocol import AudioFormat
from tests.unit.frontend_helpers import (
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


def edges(transport="websocket", port=None, **kwargs):
    host = PipelineHost()
    host.session = port or RecordingPort(session_id="live_traced")
    with patch.object(audio_recorder, "ENABLE_ASR_DUMP", False), patch.object(audio_recorder, "ENABLE_TTS_DUMP", False):
        return host.edges(transport=transport, audio_in_rate=LIVE_AUDIO_IN_RATE, stt=None, **kwargs)


def websocket_output(port):
    """Return a primary WebSocket's output edge for ``port``: Pipecat's output transport, then the silence."""
    return list(edges(port=port, connection=None).outputs)


def environ(name, value):
    """Set ``name`` to ``value`` in the environment, or unset it when ``value`` is ``None``; return the patch."""
    changed = patch.dict(os.environ)
    changed.start()
    os.environ.pop(name, None)
    os.environ.update({name: value} if value is not None else {})
    return changed


class Clock(FrameProcessor):
    """Processor that moves the session's input clock 100 ms on with each frame and notes each frame's time."""

    def __init__(self, port):
        super().__init__()
        self.port, self.times = port, []

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        self.port.input_ms += 100
        self.times.append((type(frame), self.port.input_ms))
        await self.push_frame(frame, direction)


class TextHost(PipelineHost):
    """A PipelineHost adapter with no speech services: the host still gives it a gate and a transcript."""

    async def start(self, session, delegation):
        self.session = session
        edges = self.edges(transport="websocket", connection=None, audio_in_rate=LIVE_AUDIO_IN_RATE, stt=None)
        self.processors = edges.around([IdentityFilter()], [])
        await self.run_pipeline(PipelineWorker(Pipeline(self.processors), **edges.worker), self.processors)


class AudioTests(unittest.IsolatedAsyncioTestCase):
    async def test_a_22050_hz_reply_reaches_the_wire_rate_whole(self):
        for audio_format in (AudioFormat(), AudioFormat(rate=16000)):
            with self.subTest(rate=audio_format.rate):
                port = RecordingPort(socket=FakeSocket(audio_format))
                audio = tone(200, rate=22050)
                chunks = [
                    OutputAudioRawFrame(audio=audio[i : i + 882], sample_rate=22050, num_channels=1)
                    for i in range(0, len(audio), 882)
                ]
                await run_frames(
                    websocket_output(port), [TTSStartedFrame(), *chunks, TTSStoppedFrame()], gap=0, settle=0.5
                )
                packet = audio_format.rate // 10 * 2
                self.assertEqual({len(audio) for audio in port.packets()}, {packet})
                # 200 ms: the line's last samples, held in the resampler's filter, are flushed at its end.
                self.assertEqual(sum(len(audio) for audio in port.packets() if any(audio)), 2 * packet)

    def test_each_transport_runs_at_its_rates_in_its_chunks(self):
        captured = {}

        class FakeTransport:
            def __init__(self, websocket=None, params=None, webrtc_connection=None):
                captured.update(socket=websocket, params=params)

            def input(self):
                return "input"

            output = input

        socket = FakeSocket(AudioFormat(type="audio/pcmu"))
        for transport, target, rate, default in (
            ("webrtc", "pipecat.transports.smallwebrtc.transport.SmallWebRTCTransport", WEBRTC_AUDIO_OUT_RATE, 5),
            ("websocket", "pipecat.transports.websocket.fastapi.FastAPIWebsocketTransport", 8000, 10),
        ):
            for setting, chunks in ((None, default), ("8", 8)):
                with self.subTest(transport, setting=setting), patch(target, FakeTransport):
                    self.addCleanup(environ("AUDIO_OUT_10MS_CHUNKS", setting).stop)
                    built = edges(transport, port=RecordingPort(socket=socket), connection=object())
                    params = captured["params"]
                    rates = (params.audio_in_sample_rate, params.audio_out_sample_rate, params.audio_out_10ms_chunks)
                    self.assertEqual(rates, (LIVE_AUDIO_IN_RATE, rate, chunks))
                    self.assertEqual((built.inputs[0], built.outputs[0]), ("input", "input"))
                    if transport == "websocket":
                        self.assertIs(captured["socket"], socket)
                        self.assertIsInstance(params.serializer, LiveFrameSerializer)
                        # The gateway checked the origin; the session's own silence fills the time between lines.
                        self.assertEqual((params.allowed_origins, params.audio_out_end_silence_secs), ([], 0))

    async def test_the_clock_counts_all_caller_audio_even_muted(self):
        # Samples, not rounded milliseconds per frame, so the clock never falls behind (6 ms each would give 960 ms).
        port, probe = RecordingPort(), Probe()
        chunk = InputAudioRawFrame(audio=bytes(200), sample_rate=LIVE_AUDIO_IN_RATE, num_channels=1)
        await run_frames([LiveInputGate(port), probe], [chunk] * 160, gap=0.0)
        self.assertEqual([frame.audio for frame in probe.of(InputAudioRawFrame)], [bytes(200)] * 160)
        self.assertEqual(port.input_ms, 1000)
        port, probe = RecordingPort(), Probe()
        gate = LiveInputGate(port)
        gate.muted = True
        await run_frames([gate, probe], [InputAudioRawFrame(audio=bytes(3200), sample_rate=16000, num_channels=1)])
        self.assertEqual((probe.of(InputAudioRawFrame), port.input_ms), ([], 100))

    def test_the_pipeline_is_traced_only_when_tracing_is_on(self):
        for setting, traced in ((None, False), ("true", True)):
            with self.subTest(setting=setting):
                self.addCleanup(environ("ENABLE_TRACING", setting).stop)
                worker = edges(connection=None).worker
                self.assertEqual(worker["enable_tracing"], traced)
                if traced:
                    self.assertEqual(worker["conversation_id"], "live_traced")
                self.assertIsNone(worker["idle_timeout_secs"])
                self.assertEqual((worker["params"].enable_metrics, worker["params"].enable_usage_metrics), (True, True))

    async def test_the_audio_recorder_follows_the_transcript_when_on(self):
        with (
            patch.object(audio_recorder, "ENABLE_ASR_DUMP", True),
            patch.object(audio_recorder, "_validate_dump_dir"),
            patch.object(audio_recorder, "_write_wav"),
        ):
            host, port = TextHost(), RecordingPort(socket=FakeSocket(AudioFormat(rate=16000)))
            await host.start(port, None)
            self.addAsyncCleanup(host.stop_pipeline)
            self.assertIsInstance(host.recorder, AudioBufferProcessor)
            self.assertIs(host.processors[-1], host.recorder)
            self.assertIsInstance(host.processors[-2], LiveOutputTranscript)
            for index in range(3):
                port.socket.send(f"audio {index}", audio=tone(100, rate=16000))
            async with asyncio.timeout(2):
                while not host.recorder.has_audio():
                    await asyncio.sleep(0.01)


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
        self.assertEqual(set(port.packets()), {bytes(PACKET_BYTES)})

    async def test_a_short_reply_goes_out_whole_then_silence_again(self):
        port = RecordingPort()
        await run_frames(websocket_output(port), [TTSStartedFrame(), reply(150), TTSStoppedFrame()], gap=0, settle=0.6)
        sounds = port.sounds()
        first = sounds.index(True)
        self.assertEqual(sounds[first : first + 2], [True, True])
        voiced = b"".join(port.packets()[first : first + 2])
        self.assertEqual(voiced, tone(150) + bytes(len(tone(50))))
        self.assertGreaterEqual(len(sounds[first + 2 :]), 2)
        self.assertNotIn(True, sounds[first + 2 :])
        self.assertEqual({len(audio) for audio in port.packets()}, {PACKET_BYTES})

    async def test_no_silence_inside_a_reply_whose_audio_ran_dry(self):
        port = RecordingPort()
        await run_frames(websocket_output(port), [TTSStartedFrame(), reply(100)], gap=0, settle=0.6)
        self.assertEqual((port.sounds()[-1:], port.sounds().count(True)), ([True], 1))

    async def test_a_silent_line_or_a_cut_brings_the_silence_back(self):
        for ending in ([TTSStoppedFrame()], [reply(100), InterruptionFrame()]):
            with self.subTest(ending=[type(frame).__name__ for frame in ending]):
                port = RecordingPort()
                await run_frames(websocket_output(port), [TTSStartedFrame(), *ending], gap=0.05, settle=0.4)
                self.assertEqual(port.sounds()[-3:], [False] * 3)

    async def test_the_silence_never_reaches_what_reads_after_it(self):
        port, probe = RecordingPort(), Probe()
        await run_frames([*websocket_output(port), probe], [TTSStartedFrame(), reply(100), TTSStoppedFrame()])
        await asyncio.sleep(0.3)
        self.assertGreater(port.sounds().count(False), 0)
        self.assertEqual([frame.audio for frame in probe.of(OutputAudioRawFrame)], [tone(100)])

    async def test_a_g711_session_gets_its_silence_in_g711(self):
        for kind, code in (("audio/pcmu", b"\xff"), ("audio/pcma", b"\xd5")):
            with self.subTest(kind=kind):
                port = RecordingPort(socket=FakeSocket(AudioFormat(type=kind)))
                await run_frames(websocket_output(port), [], settle=0.35)
                self.assertGreaterEqual(len(port.packets()), 2)
                self.assertEqual(set(port.packets()), {code * 800})

    async def test_after_a_stall_the_clock_starts_again(self):
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
    async def test_a_cut_keeps_what_no_interruption_may_drop(self):
        kept = LLMMessagesAppendFrame(messages=[{"role": "developer", "content": "result"}], run_llm=True)
        kept.interruptible = False
        dropped = LLMMessagesAppendFrame(messages=[], run_llm=True)
        line = [TTSStartedFrame(), *[reply(100) for _ in range(20)], kept, dropped, InterruptionFrame()]
        probe = Probe()
        await run_frames([*websocket_output(RecordingPort()), probe], line, gap=0.005, settle=0.4)
        self.assertEqual(probe.of(LLMMessagesAppendFrame), [kept])
        probe = Probe()
        await run_frames([SpeechPreroll(10_000), probe], [*line[:3], kept, FLUSH, InterruptionFrame()])
        kinds = [type(frame) for frame in probe.of(InterruptionFrame, LLMMessagesAppendFrame)]
        self.assertEqual(kinds, [InterruptionFrame, LLMMessagesAppendFrame])
        self.assertFalse(probe.of(TTSStartedFrame))

    async def test_each_played_sentence_is_timed_from_its_first_audio(self):
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
