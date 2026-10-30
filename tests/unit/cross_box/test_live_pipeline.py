# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D100, D101, D102, D103

import asyncio
import base64
import unittest
from types import SimpleNamespace

from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    ErrorFrame,
    InputAudioRawFrame,
    InterruptionFrame,
    OutputAudioRawFrame,
    TranscriptionFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
    UserStartedSpeakingFrame,
)
from pipecat.observers.base_observer import FramePushed
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineWorker
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.workers.runner import WorkerRunner

from examples.frontend_backend_live.adapter.bridge import (
    LiveEventObserver,
    LiveInputGate,
    LiveWebSocketAudioOutput,
    audio_ms,
)
from live.gateway.websocket import wire_codec
from live.media.audio import WireCodec
from live.protocol import AudioFormat
from live.session import LiveProtocolSession
from tests.unit.live_helpers import Collector, FakeAdapter, responses_config


async def started_session(config=None):
    config = config or responses_config()
    adapter = FakeAdapter()  # the session's frontend adapter and, through the gateway's core hook, its core
    session = LiveProtocolSession(config, "websocket", adapter.parts(), core=adapter.core, codec=wire_codec(config))
    collector = Collector()
    collector.key = session.subscribe(collector.send, "primary")  # the connection's key, which its commands name
    await session.start()
    return session, collector


class Probe(FrameProcessor):
    """Records every frame that reaches it."""

    def __init__(self):
        """Start with no recorded frames."""
        super().__init__()
        self.frames = []

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        self.frames.append(frame)
        await self.push_frame(frame, direction)


async def run_pipeline(processors, frames, settle=0.15, observers=None):
    task = PipelineWorker(Pipeline(processors), observers=observers or [], idle_timeout_secs=None)
    runner = WorkerRunner(handle_sigint=False)
    await runner.add_workers(task)
    running = asyncio.create_task(runner.run())
    await asyncio.sleep(0.05)
    for frame in frames:
        await task.queue_frame(frame)
        await asyncio.sleep(0.02)
    await asyncio.sleep(settle)
    await task.cancel()
    await asyncio.gather(running, return_exceptions=True)


class AudioMathTests(unittest.TestCase):
    def test_audio_duration(self):
        self.assertEqual(audio_ms(bytes(32000), 16000), 1000)
        self.assertEqual(audio_ms(bytes(9600), 24000), 200)
        self.assertEqual(audio_ms(bytes(64000), 16000, channels=2), 1000)


class InputGateTests(unittest.IsolatedAsyncioTestCase):
    async def test_the_clock_counts_all_audio_and_muting_only_drops_it(self):
        session, _ = await started_session()
        self.addAsyncCleanup(session.close)
        gate, probe = LiveInputGate(session.adapter_view), Probe()
        chunk = InputAudioRawFrame(audio=bytes(3200), sample_rate=16000, num_channels=1)  # 100 ms
        await run_pipeline([gate, probe], [chunk, chunk])
        self.assertEqual((gate.input_ms, len([f for f in probe.frames if isinstance(f, InputAudioRawFrame)])), (200, 2))
        muted_gate, muted_probe = LiveInputGate(session.adapter_view), Probe()
        muted_gate.muted = True
        await run_pipeline([muted_gate, muted_probe], [chunk])
        self.assertEqual([f for f in muted_probe.frames if isinstance(f, InputAudioRawFrame)], [])
        self.assertEqual(muted_gate.input_ms, 100)  # muted audio still advances the clock

    async def test_an_append_is_acknowledged_at_once_on_the_input_clock_the_gate_advances(self):
        # An append's acknowledgment does not wait for later caller audio; its zero-length span is the input clock
        # when the adapter took the context, which the gate advances as audio comes.
        session, collector = await started_session()
        self.addAsyncCleanup(session.close)
        gate = LiveInputGate(session.adapter_view)
        await run_pipeline(
            [gate, Probe()],
            [InputAudioRawFrame(audio=bytes(3200), sample_rate=16000, num_channels=1)] * 3,
        )
        message = {"type": "session.thinking.append", "event_id": "t", "delegation_id": None, "content": "x"}
        await session.handle(message, collector.key)
        appended = await collector.wait_for("session.thinking.appended")
        self.assertEqual((appended["start_ms"], appended["end_ms"]), (300, 300))


class ObserverTests(unittest.IsolatedAsyncioTestCase):
    async def observed(self, session, frames):
        gate, stt, tts = SimpleNamespace(input_ms=500), object(), object()
        observer = LiveEventObserver(session.adapter_view, gate, stt, tts)
        sources = {"stt": stt, "tts": tts, "other": object()}
        for source, frame in frames:
            await observer.on_push_frame(
                FramePushed(
                    source=sources[source],
                    frame=frame,
                    destination=None,
                    direction=FrameDirection.DOWNSTREAM,
                    timestamp=0,
                )
            )
        await asyncio.sleep(0.05)

    async def test_a_frontend_that_is_both_speech_in_and_out_still_reports_when_speech_started(self):
        session, collector = await started_session()
        self.addAsyncCleanup(session.close)
        gate, frontend = SimpleNamespace(input_ms=300), object()
        observer = LiveEventObserver(session.adapter_view, gate, frontend, frontend)

        async def push(frame):
            await observer.on_push_frame(
                FramePushed(
                    source=frontend, frame=frame, destination=None, direction=FrameDirection.DOWNSTREAM, timestamp=0
                )
            )

        await push(UserStartedSpeakingFrame())
        gate.input_ms = 900
        await push(TranscriptionFrame(text="hello", user_id="u", timestamp="t", finalized=True))
        await asyncio.sleep(0.05)
        delta = next(e for e in collector.events if e["type"] == "session.input_transcript.delta")
        self.assertEqual((delta["start_ms"], delta["end_ms"]), (300, 900))

    async def test_transcripts_become_deltas_with_the_input_clock_and_ignore_other_sources(self):
        session, collector = await started_session()
        self.addAsyncCleanup(session.close)
        await self.observed(
            session,
            [
                ("stt", UserStartedSpeakingFrame()),
                ("stt", TranscriptionFrame(text="hello there", user_id="u", timestamp="t", finalized=True)),
                ("other", TranscriptionFrame(text="duplicate", user_id="u", timestamp="t")),  # a downstream re-push
                ("stt", TranscriptionFrame(text="   ", user_id="u", timestamp="t")),
            ],
        )
        deltas = [e for e in collector.events if e["type"] == "session.input_transcript.delta"]
        self.assertEqual([(d["delta"], d["start_ms"], d["end_ms"]) for d in deltas], [("hello there ", 0, 500)])

    async def test_a_nonfatal_pipeline_error_is_reported_to_the_client(self):
        session, collector = await started_session()
        self.addAsyncCleanup(session.close)
        await self.observed(session, [("other", ErrorFrame(error="TTS hiccup", fatal=False))])
        notice = next(e for e in collector.events if e["type"] == "info")  # the session goes on: a notice
        self.assertEqual((notice["code"], notice["message"]), ("pipeline_error", "TTS hiccup"))
        self.assertNotIn("error", collector.kinds())

    async def test_an_error_that_leaves_its_processor_unusable_is_reported_as_an_error(self):
        # The worker ends the pipeline for it (ProcessorUnusablePolicy.END), and the session closes after the error.
        session, collector = await started_session()
        self.addAsyncCleanup(session.close)
        tts = Probe()
        await tts.set_usable(False)
        run = ErrorFrame(error="3 consecutive TTS contexts completed with no audio", processor=tts)
        await self.observed(session, [("tts", run)])
        error = next(e for e in collector.events if e["type"] == "error")["error"]
        self.assertEqual((error["code"], error["message"]), ("pipeline_error", run.error))
        self.assertNotIn("info", collector.kinds())

    async def test_an_error_each_processor_pushes_on_is_reported_once(self):
        # The observer sees every push of a frame, and an error crosses the pipeline processor by
        # processor; it is one error for the client. Another error with the same text is reported again.
        session, collector = await started_session()
        self.addAsyncCleanup(session.close)
        first, again = ErrorFrame(error="TTS hiccup", fatal=False), ErrorFrame(error="TTS hiccup", fatal=False)
        await self.observed(session, [("tts", first), ("stt", first), ("other", first), ("tts", again)])
        errors = [e["message"] for e in collector.events if e["type"] == "info"]
        self.assertEqual(errors, ["TTS hiccup", "TTS hiccup"])

    async def test_one_silent_tts_context_is_not_reported_but_a_run_of_them_is(self):
        session, collector = await started_session()
        self.addAsyncCleanup(session.close)
        silent = ErrorFrame(error="TTS context 418dbda7-d63e-4c92-809d-144a06adf57c completed with no audio")
        run = ErrorFrame(error="3 consecutive TTS contexts completed with no audio")
        await self.observed(session, [("other", silent), ("other", run)])
        errors = [e["message"] for e in collector.events if e["type"] == "info"]
        self.assertEqual(errors, ["3 consecutive TTS contexts completed with no audio"])


class WebSocketOutputTests(unittest.IsolatedAsyncioTestCase):
    async def start(self, config=None):
        session, self.collector = await started_session(config)
        self.addAsyncCleanup(session.close)
        self.output, self.probe = LiveWebSocketAudioOutput(session.adapter_view), Probe()
        return session

    def audio_events(self):
        return [e for e in self.collector.events if e["type"] == "session.output_audio.delta"]

    async def test_it_emits_a_steady_clock_of_100ms_packets_including_silence(self):
        await self.start()
        await run_pipeline([self.output, self.probe], [], settle=0.55)
        packets = self.audio_events()
        self.assertGreaterEqual(len(packets), 4)
        for packet in packets:
            self.assertEqual(set(packet), {"type", "delta"})  # no event_id or timing, as on the primary socket
            self.assertEqual(len(base64.b64decode(packet["delta"])), 4800)  # 100 ms of 24 kHz PCM16
        self.assertTrue(all(not any(base64.b64decode(p["delta"])) for p in packets))  # all silence

    async def test_reply_audio_is_sent_in_order_and_reports_speaking_state(self):
        await self.start()
        tone = OutputAudioRawFrame(audio=b"\x10\x10" * 4800, sample_rate=24000, num_channels=1)  # 200 ms
        await run_pipeline(
            [self.output, self.probe],
            [TTSStartedFrame(), tone, TTSStoppedFrame()],
            settle=0.8,
        )
        payloads = [base64.b64decode(p["delta"]) for p in self.audio_events()]
        voiced = [i for i, p in enumerate(payloads) if any(p)]
        self.assertEqual(len(voiced), 2)
        self.assertEqual(voiced[1], voiced[0] + 1)
        kinds = [
            type(f).__name__
            for f in self.probe.frames
            if isinstance(f, (BotStartedSpeakingFrame, BotStoppedSpeakingFrame))
        ]
        self.assertEqual(kinds, ["BotStartedSpeakingFrame", "BotStoppedSpeakingFrame"])

    async def test_a_barge_in_drops_audio_that_was_not_sent_yet(self):
        await self.start()
        long_reply = OutputAudioRawFrame(audio=b"\x10\x10" * 24000, sample_rate=24000, num_channels=1)  # 1 s
        await run_pipeline(
            [self.output, self.probe],
            [TTSStartedFrame(), long_reply, InterruptionFrame()],
            settle=0.5,
        )
        voiced = [p for p in self.audio_events() if any(base64.b64decode(p["delta"]))]
        self.assertLess(len(voiced), 4)  # about ten were queued; the interruption cut it off

    async def test_g711_sessions_get_g711_packets(self):
        config = responses_config(audio={"format": {"type": "audio/pcmu", "rate": 8000}})
        session = await self.start(config)
        self.assertEqual(session.codec.format, AudioFormat(type="audio/pcmu", rate=8000))
        await run_pipeline([self.output, self.probe], [], settle=0.35)
        sizes = [len(base64.b64decode(p["delta"])) for p in self.audio_events()]
        # 100 ms at 8 kHz is 800 one-byte samples; a stateful resampler's first packet can be a few samples short.
        self.assertGreaterEqual(min(sizes), 700)
        self.assertLessEqual(max(sizes), 800)
        self.assertEqual(WireCodec(config.audio.format).format.type, "audio/pcmu")


if __name__ == "__main__":
    unittest.main()
