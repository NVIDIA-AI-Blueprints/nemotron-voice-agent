# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D107

"""The narrow views a session gives its two parts, and the server delegation's own rules against a core.

The frontend adapter gets a :class:`gateways.live.ports.SessionPort` and the server delegation a
:class:`gateways.live.ports.DelegationView`; each view sends only its own event types and refuses every other one. A
live session's requests to the core are checked in whole calls (``test_frontend_backend_live_engine.py``).
"""

import asyncio
import base64
import unittest

from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineWorker
from pipecat.workers.runner import WorkerRunner

from examples.frontend_backend_live.adapter.bridge import LiveWebSocketSilence
from examples.frontend_backend_live.adapter.serializer import LiveFrameSerializer
from examples.frontend_backend_live.delegation_core.im.input import BackendTask, settings_for
from gateways.live.gateway.websocket import PrimarySocketView
from gateways.live.protocol import SessionConfig
from gateways.live.server_delegation.passthrough import ServerDelegation, request_settings
from gateways.live.session import LiveProtocolSession
from tests.unit.frontend_backend_live.adapter.adapter_helpers import FakeSocket
from tests.unit.live_helpers import ClientSocket, Collector, FakeAdapter, responses_config

ADAPTER_SENDS = ("session.input_transcript.delta", "session.output_transcript.delta")
DELEGATION_SENDS = ("session.delegation.created", "response.event")
# Server events that neither part sends, and names that are no Live server event at all.
NEITHER_SENDS = ("session.started", "session.output_audio.delta", "session.closed", "error", "info", "response.created")
HI = [{"role": "user", "content": [{"type": "input_text", "text": "Hi."}]}]


class ViewTests(unittest.IsolatedAsyncioTestCase):
    async def start(self, transport="websocket", adapter=None):
        """Start a session whose adapter (and core) is a fake; a sideband connection attaches once it is active."""
        config, self.adapter = responses_config(), adapter or FakeAdapter()
        self.primary, self.sideband = Collector(), Collector()
        parts, core = self.adapter.parts(), self.adapter.core
        if transport == "websocket":  # as the WebSocket edge makes it: the socket the frontend adapter reads
            socket = PrimarySocketView(ClientSocket(self.primary), config.with_wire_format().audio.format)
            self.session = LiveProtocolSession(config, transport, parts, core=core, socket=socket)
            socket.attach(self.session)
        else:
            self.session = LiveProtocolSession(config, transport, parts, core=core)
            self.session.subscribe(self.primary.send, "primary")
        await self.session.start("start_1")
        self.session.subscribe(self.sideband.send, "sideband")
        self.addAsyncCleanup(self.session.close)

    async def test_each_view_sends_its_own_event_types_to_every_connection_and_refuses_every_other_one(self):
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
            await collector.wait_for(DELEGATION_SENDS[-1])  # each connection is sent its events in order
            self.assertEqual(collector.kinds(), [*before, *ADAPTER_SENDS, *DELEGATION_SENDS])

    async def test_reply_audio_goes_to_the_primary_connection_only_as_it_came(self):
        await self.start()
        port = self.adapter.session
        await port.send_output_audio(b"")  # an empty chunk sends nothing
        await port.send_output_audio(bytes(range(256)) * 4)  # audio in the wire format, which the adapter encoded
        await port.emit(ADAPTER_SENDS[0], delta="Hi ", start_ms=0, end_ms=120)
        await self.sideband.wait_for(ADAPTER_SENDS[0])
        await self.primary.wait_for(ADAPTER_SENDS[0])
        self.assertEqual(self.sideband.kinds(), [ADAPTER_SENDS[0]])
        audio = [event for event in self.primary.events if event["type"] == "session.output_audio.delta"]
        self.assertEqual([set(event) for event in audio], [{"type", "delta"}])  # one packet, with no event id
        self.assertEqual(base64.b64decode(audio[0]["delta"]), bytes(range(256)) * 4)

    async def test_a_reply_audio_send_that_fails_is_reported_and_the_adapter_asks_to_close(self):
        await self.start(transport="webrtc")  # WebRTC carries reply audio on its media track: no event can carry it
        port = self.adapter.session  # a WebSocket edge's output on it: the silence goes nowhere
        silence = LiveWebSocketSilence(port, LiveFrameSerializer(FakeSocket(), port, 16000))
        worker = PipelineWorker(Pipeline([silence]), idle_timeout_secs=None)
        runner = WorkerRunner(handle_sigint=False)
        await runner.add_workers(worker)
        running = asyncio.create_task(runner.run())
        try:
            closed = await self.primary.wait_for("session.closed", timeout=5)  # the first packet, silence, fails
        finally:
            await worker.cancel()
            await asyncio.gather(running, return_exceptions=True)
        errors = [event["error"]["message"] for event in self.primary.events if event["type"] == "error"]
        self.assertEqual(errors, ["Output audio transport failed; start a new session."])
        self.assertEqual(closed["reason"], "connection_lost")
        self.assertNotIn("session.output_audio.delta", self.primary.kinds())

    async def test_the_adapters_view_is_active_from_its_start_until_its_close_request_closes_the_session(self):
        # So a pipeline that finishes while its session starts asks for no close.
        class StartProbe(FakeAdapter):
            async def start(self, session, delegation):
                self.active_while_starting = session.is_active()
                await super().start(session, delegation)

        await self.start(adapter=StartProbe())
        port = self.adapter.session
        self.assertEqual((self.adapter.active_while_starting, port.is_active()), (False, True))
        # Both views read the session's id and its input clock, which never goes back.
        self.assertEqual((port.id, self.session.delegation_view.id), (self.session.id, self.session.id))
        await port.advance_input_clock(250)
        await port.advance_input_clock(100)
        self.assertEqual((port.input_ms, self.session.delegation_view.input_ms), (250, 250))
        # A close request returns at once; the session then closes with its reason.
        self.assertIsNone(port.request_close("connection_lost"))
        self.assertEqual(self.session.status, "active")
        self.assertEqual((await self.primary.wait_for("session.closed"))["reason"], "connection_lost")
        self.assertEqual((port.is_active(), self.adapter.closed), (False, True))


# --------------------------------------------------------------------- the server delegation against a core
class Recorder:
    """The session's delegation view and the frontend's listener in one: it notes the events each is handed."""

    id, input_ms, acknowledgement_delay_ms, responses = "live_test", 0, 0, None

    def __init__(self):
        self.relayed: list[str] = []  # to the client
        self.handed: list[str] = []  # to the frontend

    async def emit(self, kind, **fields):
        if kind == "response.event":
            self.relayed.append(fields["event"]["type"])

    async def emit_paced(self, kind, **fields):
        await self.emit(kind, **fields)

    def report_refusal(self, message, code, client_event_id, param=None):
        self.relayed.append(code)

    def on_delegation_event(self, delegation_id, event):
        self.handed.append(event["type"])

    def on_delegation_lost(self, delegation_id):
        self.handed.append("lost")


def call_item(call_id):
    """A completed call of the client's ``get_menu``."""
    return {"type": "function_call", "call_id": call_id, "name": "get_menu", "arguments": "{}", "status": "completed"}


class LateTerminalCore:
    """A core whose first response's terminal comes late, as over a network; its continuation's events come at once."""

    def __init__(self):
        self.gate = asyncio.Event()  # the first response's terminal is on its way until the test sets it
        self.requests: list[dict] = []

    async def create_conversation(self, metadata):
        return {"id": "conv_1", "object": "conversation", "created_at": 0, "metadata": metadata}

    async def delete_conversation(self, conversation_id):
        return {"id": conversation_id, "object": "conversation.deleted", "deleted": True}

    async def create_response(self, params):
        self.requests.append(params)
        first = len(self.requests) == 1
        return self.response("resp_1", "call_1", self.gate) if first else self.response("resp_2", "call_2")

    async def response(self, response_id, call_id, terminal_on_its_way=None):
        yield {"type": "response.created", "response": {"id": response_id}}
        yield {"type": "response.output_item.done", "item": call_item(call_id)}
        if terminal_on_its_way is not None:
            await terminal_on_its_way.wait()
        yield {"type": "response.completed", "response": {"id": response_id, "output": [call_item(call_id)]}}


class ServerDelegationTests(unittest.IsolatedAsyncioTestCase):
    async def side(self, live_session=True):
        """A server delegation on a :class:`LateTerminalCore`; one recorder is its frontend and its session's view."""
        self.core, self.recorder = LateTerminalCore(), Recorder()
        side = ServerDelegation(self.core)
        side.listen(self.recorder)
        await side.open(self.recorder if live_session else None)
        self.addAsyncCleanup(side.close)
        return side

    async def test_a_part_with_no_live_session_names_only_the_delegation_in_its_requests(self):
        # The UI bot's part has no live session; a live session's part also names the session (live_session_id).
        side = await self.side(live_session=False)
        delegation_id = await side.delegate(HI)
        self.core.gate.set()
        self.assertEqual([params["metadata"] for params in self.core.requests], [{"live_delegation_id": delegation_id}])

    async def test_a_continuations_events_never_overtake_the_end_of_the_response_it_continues(self):
        # A delegation has a reader for each of its responses. Over a network the waiting response's terminal may still
        # be on its way when the client's continuation is accepted; the next reader relays nothing before the previous
        # one has ended, so the client and the frontend see each response whole and in order.
        side, recorder = await self.side(), self.recorder
        await side.delegate(HI)
        async with asyncio.timeout(2):  # the call is relayed, so the client can answer it
            while "response.output_item.done" not in recorder.relayed:
                await asyncio.sleep(0.005)
        await side.item_create({"type": "function_call_output", "call_id": "call_1", "output": "{}"}, None)
        await side.response_create("continue")  # queued: the continuation's stream opens once the sender sends it
        await asyncio.sleep(0.05)  # time enough for the continuation's events to overtake, were they not held
        self.core.gate.set()
        async with asyncio.timeout(2):
            while len(recorder.relayed) < 6:
                await asyncio.sleep(0.005)
        each_response = ["response.created", "response.output_item.done", "response.completed"]
        self.assertEqual((recorder.relayed, recorder.handed), (each_response * 2, each_response * 2))


class BackendSettingsTests(unittest.TestCase):
    """What the client set reaches the request (the gateway's ``request_settings``), and the core fills the rest."""

    def task(self, delegation):
        config = SessionConfig.model_validate({"model": "live-1", "delegation": delegation})
        return settings_for(request_settings(config.delegation.responses), BackendTask("fallback", "static", []))

    def test_what_the_client_set_reaches_the_backend_request_and_the_core_fills_the_rest(self):
        chosen = {"reasoning": {"effort": "high"}, "service_tier": "priority"}
        chosen["tool_choice"] = {"type": "function", "name": "get_menu"}
        block = {"model": "m", "instructions": "Look things up.", "tools": [{"type": "function", "name": "get_menu"}]}
        task = self.task({"type": "responses", "responses": {**block, **chosen}})
        self.assertEqual((task.model, task.instructions), ("m", "Look things up."))
        self.assertEqual({key: task.payload()[key] for key in chosen}, chosen)
        # What the client left out: the core's prompt, and no setting left at its default is sent.
        task = self.task({"type": "responses", "responses": {"model": "m"}})
        self.assertEqual((task.model, task.instructions), ("m", "static"))
        self.assertEqual({"reasoning", "service_tier", "parallel_tool_calls", "text"} & set(task.payload()), set())
        # A request that sets nothing (a session not in responses mode) gets the default model and prompt, and no tools.
        for nothing in (None, {"type": "client"}):
            task = self.task(nothing)
            self.assertEqual((task.model, task.instructions, task.tools), ("fallback", "static", []))
