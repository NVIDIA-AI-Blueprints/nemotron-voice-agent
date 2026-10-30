# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D107

"""The narrow views a session gives its two parts, the runtime companion of the import rules.

The frontend adapter gets a :class:`live.ports.SessionPort` and the server delegation a
:class:`live.ports.DelegationView`; each view sends only its own event types and refuses every other one.
"""

import asyncio
import base64
import unittest
from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient
from pipecat.frames.frames import EndFrame
from pipecat.processors.frame_processor import FrameProcessor

from examples.frontend_backend_live.adapter.bridge import LiveWebSocketAudioOutput
from examples.frontend_backend_live.adapter.pipeline_host import PipelineHost
from live.gateway.websocket import wire_codec
from live.media.audio import WireCodec
from live.mount import mount_live
from live.ports import DelegationView, SessionPort
from live.protocol import AudioFormat
from live.session import LiveProtocolSession
from tests.unit.live_helpers import Collector, FakeAdapter, responses_config

# The 22 server events of the Live wire, and names that are no Live server event at all.
LIVE_SERVER_EVENTS = (
    "transport.ringing",
    "transport.answered",
    "transport.failed",
    "session.started",
    "session.updated",
    "session.input_audio.muted",
    "session.input_audio.unmuted",
    "session.instructions.appended",
    "session.thinking.appended",
    "session.commentary.appended",
    "session.output_audio.delta",
    "session.input_transcript.delta",
    "session.output_transcript.delta",
    "session.delegation.created",
    "response.event",
    "session.usage.updated",
    "session.closed",
    "error",
    "info",
    "session.input_audio.append",  # the sideband's copy of the caller's audio
    "transport.dtmf.received",
    "transport.dtmf.send",
)
NOT_LIVE_EVENTS = ("response.created", "response.completed", "session.delegation.done", "")
ADAPTER_SENDS = {"session.input_transcript.delta", "session.output_transcript.delta"}
DELEGATION_SIDE_SENDS = {"session.delegation.created", "response.event"}


def members(view) -> set[str]:
    """The public names of a view."""
    return {name for name in dir(view) if not name.startswith("_")}


def port_members(port) -> set[str]:
    """What a port declares: its methods and its attributes."""
    return members(port) | set(port.__annotations__)


class ViewCase(unittest.IsolatedAsyncioTestCase):
    async def start(self, transport="websocket"):
        """Start a session whose one fake is its adapter and its core; a sideband observer attaches once active."""
        config = responses_config()
        self.engine = FakeAdapter()
        codec = wire_codec(config) if transport == "websocket" else None  # only the WebSocket edge hands one in
        self.session = LiveProtocolSession(config, transport, self.engine.parts(), core=self.engine.core, codec=codec)
        self.primary, self.sideband = Collector(), Collector()
        self.session.subscribe(self.primary.send, "primary")
        await self.session.start("start_1")
        await self.primary.wait_for("session.started")
        self.session.subscribe(self.sideband.send, "sideband")
        self.addAsyncCleanup(self.session.close)

    async def delivered(self):
        """Wait until every connection has been sent what was published for it."""
        for subscriber in tuple(self.session.fanout.subscribers.values()):
            await subscriber.queue.join()

    def new_events(self, collector, before):
        return [event["type"] for event in collector.events[before:]]


class WhatEachPartGetsTests(ViewCase):
    async def test_each_part_gets_its_own_view_and_never_the_session(self):
        await self.start()
        self.assertIs(self.session.delegation.view, self.session.delegation_view)  # open(view)
        self.assertIs(self.engine.session, self.session.adapter_view)  # start(session, delegation)
        for view in (self.session.delegation.view, self.engine.session):
            self.assertIsNot(view, self.session)

    async def test_each_view_has_its_ports_members_and_nothing_else_of_the_session(self):
        await self.start()
        self.assertEqual(members(self.session.adapter_view), port_members(SessionPort))
        self.assertEqual(members(self.session.delegation_view), port_members(DelegationView))
        # The view holds the session's id and input clock, its core client
        # (responses), the paced and the plain emit, the refusal report, and the acknowledgement delay the part serves.
        self.assertEqual(
            port_members(DelegationView),
            {"acknowledgement_delay_ms", "emit", "emit_paced", "id", "input_ms", "report_refusal", "responses"},
        )


class RefusalTests(ViewCase):
    async def assert_sends_only(self, view, allowed):
        for kind in LIVE_SERVER_EVENTS + NOT_LIVE_EVENTS:
            with self.subTest(kind=kind):
                before = len(self.primary.events), len(self.sideband.events)
                if kind in allowed:
                    await view.emit(kind, delta="Hi ", start_ms=0, end_ms=120)
                    await self.delivered()
                    self.assertEqual(self.new_events(self.primary, before[0]), [kind])
                    self.assertEqual(self.new_events(self.sideband, before[1]), [kind])
                else:
                    with self.assertRaises(ValueError) as refused:
                        await view.emit(kind, delta="Hi ", start_ms=0, end_ms=120)
                    self.assertIn(repr(kind), str(refused.exception))
                    await self.delivered()
                    self.assertEqual(self.new_events(self.primary, before[0]), [])
                    self.assertEqual(self.new_events(self.sideband, before[1]), [])

    async def test_the_adapters_view_sends_the_two_transcript_events_and_refuses_every_other_type(self):
        await self.start()
        await self.assert_sends_only(self.engine.session, ADAPTER_SENDS)

    async def test_the_delegation_sides_view_sends_the_two_delegation_events_and_refuses_every_other_type(self):
        await self.start()
        await self.assert_sends_only(self.session.delegation.view, DELEGATION_SIDE_SENDS)


class AdapterViewTests(ViewCase):
    async def test_reply_audio_goes_to_the_primary_connection_only_and_carries_no_event_id(self):
        await self.start()
        await self.engine.session.send_output_audio(bytes(4800), 24000)  # 100 ms at the exchange rate
        await self.delivered()
        audio = [event for event in self.primary.events if event["type"] == "session.output_audio.delta"]
        self.assertEqual(len(audio), 1)
        self.assertEqual(set(audio[0]), {"type", "delta"})
        self.assertEqual(len(base64.b64decode(audio[0]["delta"])), 4800)  # the default wire format: 24 kHz PCM16
        self.assertNotIn("session.output_audio.delta", self.new_events(self.sideband, 0))

    async def test_reply_audio_at_another_rate_is_refused_and_sends_nothing(self):
        await self.start()
        before = len(self.primary.events)
        with self.assertRaises(ValueError):
            await self.engine.session.send_output_audio(bytes(3200), 16000)
        await self.delivered()
        self.assertEqual(self.new_events(self.primary, before), [])

    async def test_a_session_without_a_websocket_codec_sends_no_reply_audio_events(self):
        await self.start(transport="webrtc")  # WebRTC carries reply audio on its media track
        with self.assertRaises(RuntimeError):
            await self.engine.session.send_output_audio(bytes(4800), 24000)

    async def test_an_error_report_is_a_pipeline_error_for_every_connection(self):
        await self.start()
        await self.engine.session.report_error("TTS hiccup", "c1")
        await self.delivered()
        for collector in (self.primary, self.sideband):
            error = next(event for event in collector.events if event["type"] == "error")
            self.assertTrue(error["event_id"].startswith("event_"))
            self.assertEqual(
                error["error"],
                {
                    "type": "server_error",
                    "code": "pipeline_error",
                    "message": "TTS hiccup",
                    "client_event_id": "c1",
                },  # an optional field the error has no value for is left out
            )

    async def test_a_close_request_returns_at_once_and_the_session_closes_with_its_reason(self):
        await self.start()
        view = self.engine.session
        self.assertTrue(view.is_active())
        self.assertIsNone(view.request_close("connection_lost"))
        self.assertEqual(self.session.status, "active")  # only scheduled
        closed = await self.primary.wait_for("session.closed")
        self.assertEqual(closed["reason"], "connection_lost")
        self.assertFalse(view.is_active())
        self.assertTrue(self.engine.closed)

    async def test_the_views_read_the_sessions_id_and_input_clock(self):
        await self.start()
        adapter, delegation = self.engine.session, self.session.delegation.view
        self.assertEqual(adapter.id, self.session.id)
        await adapter.advance_input_clock(250)
        await adapter.advance_input_clock(100)  # the clock never goes back
        self.assertEqual((adapter.input_ms, delegation.input_ms, self.session.input_ms), (250, 250, 250))

    async def test_the_session_is_active_for_the_adapter_only_once_the_adapter_has_started(self):
        # A pipeline that finishes while its session starts asks for no close (PipelineHost.run_pipeline).
        class StartProbe(FakeAdapter):
            async def start(self, session, delegation):
                self.active_while_starting = session.is_active()
                await super().start(session, delegation)

        engine = StartProbe()
        session = LiveProtocolSession(responses_config(), "websocket", engine.parts(), core=engine.core)
        session.subscribe(Collector().send, "primary")
        await session.start("start_1")
        self.addAsyncCleanup(session.close)
        self.assertEqual((engine.active_while_starting, engine.session.is_active()), (False, True))


class WebSocketEdgeTests(unittest.TestCase):
    """The primary WebSocket builds the session's codec from the session's own audio format."""

    def test_caller_audio_reaches_the_adapter_decoded_from_the_sessions_wire_format(self):
        engines: list[FakeAdapter] = []

        async def factory(config, transport, connection):
            engines.append(FakeAdapter())
            return engines[-1].parts()

        app = FastAPI()
        mount_live(app, factory=factory)
        g711 = {"type": "audio/pcmu", "rate": 8000}
        chunk = bytes(range(256)) * 3 + bytes(range(32))  # 100 ms of 8 kHz G.711: 800 one-byte samples
        start = {"type": "session.start", "event_id": "s1", "session": {"model": "live-1", "audio": {"format": g711}}}
        with TestClient(app) as client, client.websocket_connect("/v1/live/sessions") as ws:
            ws.send_json(start)
            self.assertEqual(ws.receive_json()["type"], "session.started")
            ws.send_json({"type": "session.input_audio.append", "audio": base64.b64encode(chunk).decode()})
            ws.send_json({"type": "session.input_audio.mute", "event_id": "m"})
            self.assertEqual(ws.receive_json()["type"], "session.input_audio.muted")
            ws.send_json({"type": "session.close", "event_id": "bye"})
            self.assertEqual(ws.receive_json()["type"], "session.closed")
        # 24 kHz PCM16 from a codec of the session's own format; a default 24 kHz PCM codec would pass the bytes on.
        self.assertEqual(engines[0].audio, [WireCodec(AudioFormat(**g711)).decode(chunk)])


class CommandFailureAtCloseTests(ViewCase):
    """A background command that fails as the close begins is reported before ``session.closed``.

    The adapter's view has no ``spawn``, so the report is the adapter's own task, which the adapter's close cancels
    (close step 4), not a session task, which close step 3 cancels.
    """

    async def test_a_command_that_fails_as_the_close_begins_is_still_reported_before_session_closed(self):
        await self.start()
        adapter = SimpleNamespace(session=self.engine.session, background=set())

        async def command():
            raise RuntimeError("boom")

        PipelineHost._spawn(adapter, command())
        await asyncio.create_task(self.session.close())  # the close begins in the loop turn in which the command fails
        self.assertEqual(self.primary.kinds(), ["session.started", "info", "session.closed"])
        self.assertEqual(self.primary.events[1]["message"], "A command could not be processed.")


class RecordingPort:
    """A frontend adapter's view of its session that notes what the adapter asks of it."""

    def __init__(self, audio_fails=False):
        self.id, self.input_ms = "live_test", 0
        self.active, self.audio_fails = True, audio_fails
        self.calls: list[tuple] = []

    def is_active(self):
        return self.active

    async def emit(self, kind, **fields):
        self.calls.append(("emit", kind))

    async def advance_input_clock(self, ms):
        self.calls.append(("advance_input_clock", ms))

    async def send_output_audio(self, pcm, sample_rate):
        if self.audio_fails:
            raise ConnectionError("the socket is gone")
        self.calls.append(("send_output_audio", len(pcm), sample_rate))

    async def report_error(self, message, client_event_id=None, *, fatal=False):
        self.calls.append(("report_error", message, client_event_id, fatal))

    def request_close(self, reason):
        self.calls.append(("request_close", reason))


class Passthrough(FrameProcessor):
    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        await self.push_frame(frame, direction)


class AdapterFailureTests(unittest.IsolatedAsyncioTestCase):
    """The adapter reports its own failures, and asks to close, only through its view."""

    async def test_a_failed_reply_audio_send_is_reported_then_the_adapter_asks_to_close(self):
        port = RecordingPort(audio_fails=True)
        async with asyncio.timeout(2):
            await LiveWebSocketAudioOutput(port)._clock()  # the first packet's send fails, which ends the clock
        self.assertEqual(
            port.calls,
            [
                ("report_error", "Output audio transport failed; start a new session.", None, True),
                ("request_close", "connection_lost"),
            ],
        )

    async def test_a_failed_background_command_is_reported_and_the_session_stays_open(self):
        port = RecordingPort()
        adapter = SimpleNamespace(session=port, background=set())

        async def command():
            raise RuntimeError("boom")

        PipelineHost._spawn(adapter, command())
        async with asyncio.timeout(2):
            while not port.calls or adapter.background:
                await asyncio.sleep(0.01)
        self.assertEqual(port.calls, [("report_error", "A command could not be processed.", None, False)])

    async def test_a_pipeline_that_finishes_asks_to_close_only_an_active_session(self):
        for active, requests in ((True, [("request_close", "connection_lost")]), (False, [])):
            with self.subTest(active=active):
                port = RecordingPort()
                host = SimpleNamespace(session=port, gate=None, task=None, runner_task=None)
                await PipelineHost.run_pipeline(host, [Passthrough()], stt=object(), tts=object())
                port.active = active
                await host.task.queue_frame(EndFrame())
                async with asyncio.timeout(5):
                    await host.runner_task
                self.assertEqual([call for call in port.calls if call[0] == "request_close"], requests)


if __name__ == "__main__":
    unittest.main()
