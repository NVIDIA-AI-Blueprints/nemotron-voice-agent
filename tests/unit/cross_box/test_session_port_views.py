# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D107

"""The views a session gives its two parts, what the adapter's observer sends on its view, and a request's settings."""

import asyncio
import base64
import unittest
from types import SimpleNamespace

from pipecat.frames.frames import ErrorFrame, TranscriptionFrame, UserStartedSpeakingFrame
from pipecat.observers.base_observer import FramePushed
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineWorker
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.workers.runner import WorkerRunner

from examples.frontend_backend_live.adapter.bridge import LiveEventObserver, LiveWebSocketSilence
from examples.frontend_backend_live.adapter.serializer import LiveFrameSerializer
from examples.frontend_backend_live.delegation_core.im.input import BackendTask, settings_for
from gateways.live.gateway.websocket import PrimarySocketView
from gateways.live.protocol import SessionConfig
from gateways.live.server_delegation.passthrough import request_settings
from gateways.live.session import LiveProtocolSession
from tests.unit.frontend_helpers import FakeSocket
from tests.unit.live_helpers import ClientSocket, Collector, FakeAdapter, responses_config

ADAPTER_SENDS = ("session.input_transcript.delta", "session.output_transcript.delta")
DELEGATION_SENDS = ("session.delegation.created", "response.event")
NEITHER_SENDS = ("session.started", "session.output_audio.delta", "session.closed", "error", "info", "response.created")


class SessionCase(unittest.IsolatedAsyncioTestCase):
    async def start(self, transport="websocket", adapter=None):
        """Start a session whose adapter (and core) is a fake; a sideband connection attaches once it is active."""
        config, self.adapter = responses_config(), adapter or FakeAdapter()
        self.primary, self.sideband = Collector(), Collector()
        parts, core = self.adapter.parts(), self.adapter.core
        if transport == "websocket":
            socket = PrimarySocketView(ClientSocket(self.primary), config.with_wire_format().audio.format)
            self.session = LiveProtocolSession(config, transport, parts, core=core, socket=socket)
            socket.attach(self.session)
        else:
            self.session = LiveProtocolSession(config, transport, parts, core=core)
            self.session.subscribe(self.primary.send, "primary")
        await self.session.start("start_1")
        self.session.subscribe(self.sideband.send, "sideband")
        self.addAsyncCleanup(self.session.close)


class ViewTests(SessionCase):
    async def test_each_view_sends_only_its_own_event_types(self):
        await self.start()
        views = ((self.adapter.session, ADAPTER_SENDS), (self.session.delegation_view, DELEGATION_SENDS))
        for view, own in views:
            for kind in (*NEITHER_SENDS, "", *ADAPTER_SENDS, *DELEGATION_SENDS):
                with self.subTest(kind=kind):
                    if kind in own:
                        await view.emit(kind, delta="Hi ", start_ms=0, end_ms=120)
                        continue
                    with self.assertRaises(ValueError) as refused:
                        await view.emit(kind, delta="Hi ", start_ms=0, end_ms=120)
                    self.assertIn(repr(kind), str(refused.exception))
        for collector, before in ((self.primary, ["session.started"]), (self.sideband, [])):
            await collector.wait_for(DELEGATION_SENDS[-1])
            self.assertEqual(collector.kinds(), [*before, *ADAPTER_SENDS, *DELEGATION_SENDS])

    async def test_reply_audio_goes_to_the_primary_only_as_it_came(self):
        await self.start()
        port = self.adapter.session
        await port.send_output_audio(b"")
        await port.send_output_audio(bytes(range(256)) * 4)
        await port.emit(ADAPTER_SENDS[0], delta="Hi ", start_ms=0, end_ms=120)
        await self.sideband.wait_for(ADAPTER_SENDS[0])
        await self.primary.wait_for(ADAPTER_SENDS[0])
        self.assertEqual(self.sideband.kinds(), [ADAPTER_SENDS[0]])
        audio = [event for event in self.primary.events if event["type"] == "session.output_audio.delta"]
        self.assertEqual([set(event) for event in audio], [{"type", "delta"}])
        self.assertEqual(base64.b64decode(audio[0]["delta"]), bytes(range(256)) * 4)

    async def test_a_failed_reply_audio_send_closes_the_session(self):
        await self.start(transport="webrtc")  # WebRTC carries reply audio on its media track: no event can carry it
        port = self.adapter.session
        silence = LiveWebSocketSilence(port, LiveFrameSerializer(FakeSocket(), port, 16000))
        worker = PipelineWorker(Pipeline([silence]), idle_timeout_secs=None)
        runner = WorkerRunner(handle_sigint=False)
        await runner.add_workers(worker)
        running = asyncio.create_task(runner.run())
        try:
            closed = await self.primary.wait_for("session.closed", timeout=5)
        finally:
            await worker.cancel()
            await asyncio.gather(running, return_exceptions=True)
        errors = [event["error"]["message"] for event in self.primary.events if event["type"] == "error"]
        self.assertEqual(errors, ["Output audio transport failed; start a new session."])
        self.assertEqual(closed["reason"], "connection_lost")
        self.assertNotIn("session.output_audio.delta", self.primary.kinds())

    async def test_the_adapters_view_is_active_until_its_close_request(self):
        # So a pipeline that finishes while its session starts asks for no close.
        class StartProbe(FakeAdapter):
            async def start(self, session, delegation):
                self.active_while_starting = session.is_active()
                await super().start(session, delegation)

        await self.start(adapter=StartProbe())
        port = self.adapter.session
        self.assertEqual((self.adapter.active_while_starting, port.is_active()), (False, True))
        self.assertEqual((port.id, self.session.delegation_view.id), (self.session.id, self.session.id))
        await port.advance_input_clock(250)
        await port.advance_input_clock(100)
        self.assertEqual((port.input_ms, self.session.delegation_view.input_ms), (250, 250))
        self.assertIsNone(port.request_close("connection_lost"))
        self.assertEqual(self.session.status, "active")
        self.assertEqual((await self.primary.wait_for("session.closed"))["reason"], "connection_lost")
        self.assertEqual((port.is_active(), self.adapter.closed), (False, True))


class ObserverTests(SessionCase):
    async def observed(self, frames):
        gate, stt, tts = SimpleNamespace(input_ms=0), object(), object()
        observer = LiveEventObserver(self.session.adapter_view, gate, stt)
        sources = {"stt": stt, "tts": tts, "other": object()}
        for source, frame in frames:
            gate.input_ms += 100
            pushed = FramePushed(
                source=sources[source], frame=frame, destination=None, direction=FrameDirection.DOWNSTREAM, timestamp=0
            )
            await observer.on_push_frame(pushed)
        await asyncio.sleep(0.05)

    def notices(self):
        return [(event["code"], event["message"]) for event in self.primary.events if event["type"] == "info"]

    async def test_transcripts_become_deltas_on_the_input_clock(self):
        await self.start()
        # A turn starts at the user aggregator's frame, not the STT's; only the STT's own transcripts count.
        await self.observed(
            [
                ("other", UserStartedSpeakingFrame()),
                ("stt", UserStartedSpeakingFrame()),
                ("stt", TranscriptionFrame(text="hello there", user_id="u", timestamp="t", finalized=True)),
                ("other", TranscriptionFrame(text="duplicate", user_id="u", timestamp="t")),
                ("stt", TranscriptionFrame(text="   ", user_id="u", timestamp="t")),
            ]
        )
        deltas = [e for e in self.primary.events if e["type"] == "session.input_transcript.delta"]
        self.assertEqual([(d["delta"], d["start_ms"], d["end_ms"]) for d in deltas], [("hello there ", 100, 300)])

    async def test_errors_are_notices_unless_their_processor_is_unusable(self):
        silent = "TTS context 418dbda7-d63e-4c92-809d-144a06adf57c completed with no audio"
        cases = {
            "a nonfatal pipeline error": [("other", ErrorFrame(error="TTS hiccup", fatal=False))],
            # The observer reports a frame at its first push only: one notice per error, even for the same text.
            "an error each processor pushes on, then another with its text": [
                ("tts", ErrorFrame(error="TTS hiccup", fatal=False)),
                ("tts", ErrorFrame(error="TTS hiccup", fatal=False)),
            ],
            "a silent TTS context, like any recoverable error": [("other", ErrorFrame(error=silent))],
        }
        for name, frames in cases.items():
            with self.subTest(name):
                await self.start()
                await self.observed(frames)
                self.assertEqual(self.notices(), [("pipeline_error", frame.error) for _, frame in frames])
                self.assertNotIn("error", self.primary.kinds())
        await self.start()
        tts = FrameProcessor()
        await tts.set_usable(False)
        written_off = ErrorFrame(error="3 consecutive TTS contexts completed with no audio", processor=tts)
        await self.observed([("tts", written_off)])
        error = next(e for e in self.primary.events if e["type"] == "error")["error"]
        self.assertEqual((error["code"], error["message"]), ("pipeline_error", written_off.error))
        self.assertEqual(self.notices(), [])


class BackendSettingsTests(unittest.TestCase):
    def task(self, delegation):
        config = SessionConfig.model_validate({"model": "live-1", "delegation": delegation})
        return settings_for(request_settings(config.delegation.responses), BackendTask("fallback", "static", []))

    def test_the_request_has_what_the_client_set_and_the_cores_rest(self):
        chosen = {"reasoning": {"effort": "high"}, "service_tier": "priority"}
        chosen["tool_choice"] = {"type": "function", "name": "get_menu"}
        block = {"model": "m", "instructions": "Look things up.", "tools": [{"type": "function", "name": "get_menu"}]}
        task = self.task({"type": "responses", "responses": {**block, **chosen}})
        self.assertEqual((task.model, task.instructions), ("m", "Look things up."))
        self.assertEqual({key: task.payload()[key] for key in chosen}, chosen)
        task = self.task({"type": "responses", "responses": {"model": "m"}})
        self.assertEqual((task.model, task.instructions), ("m", "static"))
        self.assertEqual({"reasoning", "service_tier", "parallel_tool_calls", "text"} & set(task.payload()), set())
        for nothing in (None, {"type": "client"}):
            task = self.task(nothing)
            self.assertEqual((task.model, task.instructions, task.tools), ("fallback", "static", []))
