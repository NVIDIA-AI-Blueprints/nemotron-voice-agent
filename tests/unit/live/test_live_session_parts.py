# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D107

"""A session served by two parts: the frontend adapter and its delegation part (``live.ports``).

The fakes note every call in one shared log, so the tests see where each command goes and the order of start and close.
"""

import asyncio
import base64
import unittest
from unittest.mock import patch

import httpx
from fastapi import FastAPI

from live.gateway import LiveSettings, create_live_router
from live.gateway.websocket import wire_codec
from live.ports import SessionParts
from live.protocol import ProtocolError, SessionConfig
from live.server_delegation import passthrough as server_delegation
from live.session import LiveProtocolSession, lifecycle
from tests.unit.live_helpers import Collector, FakeAdapter, responses_config

HI = [{"role": "user", "content": [{"type": "input_text", "text": "Hi."}]}]


class FakeFrontend:
    """A frontend adapter and nothing else."""

    def __init__(self, log, collector=None):
        self.log, self.collector = log, collector
        self.start_gate: asyncio.Event | None = None  # set by a test to hold start until it is released
        self.close_error: Exception | None = None
        self.close_gate: asyncio.Event | None = None
        self.seen_at_start: list | None = None  # what the client had received when the frontend started

    async def start(self, session, delegation):
        self.log.append("frontend.start")
        self.seen_at_start = self.collector.kinds() if self.collector else None
        if self.start_gate:
            await self.start_gate.wait()

    async def close(self):
        self.log.append("frontend.close")
        if self.close_gate:
            await self.close_gate.wait()
        if self.close_error:
            raise self.close_error

    async def set_muted(self, muted):
        self.log.append(("frontend.set_muted", muted))

    async def feed_audio(self, pcm, sample_rate):
        self.log.append(("frontend.feed_audio", pcm, sample_rate))

    async def append_instructions(self, text, client_event_id=None, delegation_id=None):
        self.log.append(("frontend.append_instructions", text, client_event_id, delegation_id))

    async def append_thinking(self, text, client_event_id=None, delegation_id=None):
        self.log.append(("frontend.append_thinking", text, client_event_id, delegation_id))

    async def append_commentary(self, text, client_event_id=None, delegation_id=None):
        self.log.append(("frontend.append_commentary", text, client_event_id, delegation_id))


class TypingFrontend(FakeFrontend):
    """A frontend adapter that also reads the application's typed questions (``live.ports.TypedReader``)."""

    def typed(self, text):
        self.log.append(("frontend.typed", text))


class FakeDelegation:
    """A watch in place of the session's own server delegation: it notes each call the session makes, then makes it.

    The session builds its server delegation; the watch takes its place, so the log shows the calls.
    ``open_gate`` and ``open_error`` hold or fail the open, and ``item_error`` refuses each item. The update goes on to
    the server delegation; the other delegation commands are only noted: this file tests where they go.
    """

    def __init__(self, log, collector=None):
        self.log, self.collector = log, collector
        self.open_gate: asyncio.Event | None = None  # set by a test to hold open until it is released
        self.open_error: Exception | None = None
        self.item_error: Exception | None = None
        self.view = None
        self.seen_at_close: list | None = None  # what the client had received when the server delegation closed
        self.server = None  # the session's own server delegation, which the watch stands in for

    async def open(self, view):
        self.log.append("delegation.open")
        self.view = view
        if self.open_gate:
            await self.open_gate.wait()
        if self.open_error:
            raise self.open_error
        await self.server.open(view)

    def knows(self, delegation_id):
        return self.server.knows(delegation_id)

    async def update(self, config, patch, client_event_id):
        self.log.append(("delegation.update", patch, client_event_id))
        return await self.server.update(config, patch, client_event_id)  # the core does not serve "bad-model"

    async def item_create(self, item, client_event_id):
        self.log.append(("delegation.item_create", item, client_event_id))
        if self.item_error:
            raise self.item_error

    async def response_create(self, client_event_id):
        self.log.append(("delegation.response_create", client_event_id))

    def stop_handing(self):
        self.log.append("delegation.stop_handing")
        self.server.stop_handing()

    @property
    def close_seconds(self):
        return self.server.close_seconds

    async def close(self):
        self.log.append("delegation.close")
        self.seen_at_close = self.collector.kinds() if self.collector else None
        await self.server.close()


class SplitPartsCase(unittest.IsolatedAsyncioTestCase):
    def build(self, config=None, frontend=FakeFrontend):
        self.log: list = []
        self.primary = Collector()
        self.frontend = frontend(self.log, self.primary)
        self.delegation = FakeDelegation(self.log, self.primary)
        config = config or responses_config()
        self.session = LiveProtocolSession(
            config,
            "websocket",
            SessionParts(frontend=self.frontend),
            core=FakeAdapter().core,
            codec=wire_codec(config),
        )
        self.delegation.server, self.session.delegation = self.session.delegation, self.delegation
        self.connection = self.session.subscribe(self.primary.send, "primary")
        return self.session

    async def start(self, config=None, frontend=FakeFrontend):
        self.build(config, frontend)
        await self.session.start("start_1")
        await self.primary.wait_for("session.started")
        self.after_start = len(self.log)  # what the start logged: the open, the frontend's start, the settings check
        self.addAsyncCleanup(self.session.close)

    async def handle(self, message):
        """Run a command, then wait until the primary connection has received what it produced."""
        await self.session.handle(message, self.connection)
        for sub in tuple(self.session.fanout.subscribers.values()):
            await sub.queue.join()


class StartOrderTests(SplitPartsCase):
    async def test_the_delegation_side_opens_before_the_frontend_starts(self):
        await self.start()
        # After session.started, the settings the session started with are checked as an update that changes nothing,
        # with the start's event id.
        self.assertEqual(self.log, ["delegation.open", "frontend.start", ("delegation.update", {}, "start_1")])
        self.assertEqual(self.frontend.seen_at_start, [])
        self.assertEqual(self.primary.kinds(), ["session.started"])
        # The delegation side's view of the session gives it the input clock and the way to emit events.
        self.assertTrue(hasattr(self.delegation.view, "input_ms") and callable(self.delegation.view.emit))

    async def test_a_delegation_side_that_cannot_open_fails_the_start_and_both_parts_close(self):
        self.build()
        self.delegation.open_error = RuntimeError("no backend")
        with self.assertRaises(RuntimeError):
            await self.session.start("start_1")
        self.assertEqual(self.session.status, "closed")
        self.assertEqual(self.log, ["delegation.open", "delegation.stop_handing", "frontend.close", "delegation.close"])
        error = await self.primary.wait_for("error")
        self.assertEqual(error["error"]["code"], "pipeline_error")
        self.assertEqual((await self.primary.wait_for("session.closed"))["reason"], "connection_lost")
        self.assertNotIn("session.started", self.primary.kinds())

    async def test_a_session_closed_while_its_delegation_side_opens_closes_both_and_starts_no_frontend(self):
        self.build()
        self.delegation.open_gate = asyncio.Event()
        starting = asyncio.create_task(self.session.start("start_1"))
        await asyncio.sleep(0.05)
        self.assertEqual(self.log, ["delegation.open"])
        await self.session.close("remote_hangup")
        self.delegation.open_gate.set()
        await starting
        self.assertEqual(self.session.status, "closed")
        # The close closes both parts while the delegation side is still opening; once it has opened, the start
        # does not start the frontend and closes both parts again.
        self.assertEqual(
            self.log,
            [
                "delegation.open",
                "delegation.stop_handing",
                "frontend.close",
                "delegation.close",
                "frontend.close",
                "delegation.close",
            ],
        )
        self.assertNotIn("session.started", self.primary.kinds())
        self.assertEqual(self.primary.kinds().count("session.closed"), 1)

    async def test_a_session_closed_while_its_frontend_starts_closes_both_parts_again_after_the_start(self):
        self.build()
        self.frontend.start_gate = asyncio.Event()
        starting = asyncio.create_task(self.session.start("start_1"))
        await asyncio.sleep(0.05)
        await self.session.close("remote_hangup")
        self.frontend.start_gate.set()
        await starting
        self.assertEqual(self.session.status, "closed")
        self.assertEqual(
            self.log,
            [
                "delegation.open",
                "frontend.start",
                "delegation.stop_handing",
                "frontend.close",
                "delegation.close",
                "frontend.close",
                "delegation.close",
            ],
        )
        self.assertNotIn("session.started", self.primary.kinds())


class RoutingTests(SplitPartsCase):
    async def test_voice_commands_go_to_the_frontend_and_delegation_commands_to_the_delegation_side(self):
        await self.start()
        pcm = b"\x01\x00" * 480
        commands = [
            {"type": "session.input_audio.mute", "event_id": "m1"},
            {"type": "session.input_audio.unmute", "event_id": "m2"},
            {"type": "session.input_audio.append", "audio": base64.b64encode(pcm).decode()},
            {"type": "session.instructions.append", "event_id": "i1", "delegation_id": None, "content": "Be brief."},
            {"type": "session.thinking.append", "event_id": "t1", "delegation_id": None, "content": "A regular."},
            {"type": "session.commentary.append", "event_id": "c1", "delegation_id": None, "content": "It is sunny."},
            {
                "type": "session.update",
                "event_id": "u1",
                "session": {"delegation": {"responses": {"max_output_tokens": 64}}},
            },
            {"type": "response.item.create", "event_id": "o1", "item": {"type": "function_call_output"}},
            {"type": "response.create", "event_id": "r1"},
        ]
        for command in commands:
            await self.handle(command)
        self.assertNotIn("error", self.primary.kinds())
        self.assertEqual(
            self.log[self.after_start :],
            [
                ("frontend.set_muted", True),
                ("frontend.set_muted", False),
                # The default wire format is 24 kHz PCM, which the codec passes through at the exchange rate.
                ("frontend.feed_audio", pcm, 24000),
                ("frontend.append_instructions", "Be brief.", "i1", None),
                ("frontend.append_thinking", "A regular.", "t1", None),
                ("frontend.append_commentary", "It is sunny.", "c1", None),
                ("delegation.update", {"delegation": {"responses": {"max_output_tokens": 64}}}, "u1"),
                ("delegation.item_create", {"type": "function_call_output"}, "o1"),
                ("delegation.response_create", "r1"),
            ],
        )

    async def test_a_typed_question_the_delegation_side_took_is_read_by_a_frontend_that_reads_typed_questions(self):
        # The part takes the message first; then the frontend's memory gets its text parts, one per line.
        await self.start(frontend=TypingFrontend)
        typed = {
            "type": "message",
            "role": "user",
            "content": [
                {"type": "input_text", "text": "Is there parking?"},
                {"type": "input_image", "image_url": "data:image/png;base64,AA=="},
                {"type": "input_text", "text": "Behind the cafe?"},
            ],
        }
        result = {"type": "function_call_output", "call_id": "k1", "output": "{}"}
        for event_id, item in (("q1", typed), ("o1", result)):
            await self.handle({"type": "response.item.create", "event_id": event_id, "item": item})
        self.assertNotIn("error", self.primary.kinds())
        self.assertEqual(
            self.log[self.after_start :],
            [
                ("delegation.item_create", typed, "q1"),
                ("frontend.typed", "Is there parking?\nBehind the cafe?"),
                ("delegation.item_create", result, "o1"),  # a result is the backend's, never the voice's
            ],
        )

    async def test_a_typed_question_the_delegation_side_refuses_never_reaches_the_frontend(self):
        await self.start(frontend=TypingFrontend)
        self.delegation.item_error = ProtocolError("Too many queued response items", "rate_limit_exceeded")
        typed = {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "Hi."}]}
        await self.handle({"type": "response.item.create", "event_id": "q1", "item": typed})
        self.assertEqual((await self.primary.wait_for("error"))["error"]["code"], "rate_limit_exceeded")
        self.assertEqual(self.log[self.after_start :], [("delegation.item_create", typed, "q1")])

    async def test_a_model_the_core_refuses_is_reported_after_session_updated_and_kept(self):
        await self.start()
        bad = {"delegation": {"responses": {"model": "bad-model"}}}
        await self.handle({"type": "session.update", "event_id": "u1", "session": bad})
        error = (await self.primary.wait_for("error"))["error"]
        self.assertEqual(
            (error["code"], error["param"], error["client_event_id"]),
            ("responses_validation_failed", "session.delegation.responses.model", "u1"),
        )
        kinds = self.primary.kinds()
        self.assertLess(kinds.index("session.updated"), kinds.index("error"))  # session.updated does not wait
        self.assertEqual(self.log[self.after_start :], [("delegation.update", bad, "u1")])
        self.assertEqual(self.session.config.delegation.responses.model, "bad-model")

    async def test_the_usage_meter_reads_the_frontend_adapters_count_only(self):
        self.build()
        self.delegation.context_tokens = lambda: 64_000  # a delegation side's count is not the frontend's context
        fallback = lifecycle.context_ratio(self.session)
        self.assertLess(fallback, 0.05)
        self.frontend.context_tokens = lambda: 64_000
        self.frontend.context_window_tokens = lambda: None  # the ContextMeter port's other member: the default
        self.assertEqual(lifecycle.context_ratio(self.session), 0.5)


class CloseOrderTests(SplitPartsCase):
    async def test_close_stops_the_handing_then_the_tasks_then_closes_the_frontend_then_the_delegation_side(self):
        await self.start()

        async def session_task():
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                self.log.append("session task cancelled")
                raise

        self.session.spawn(session_task(), "worker")
        await asyncio.sleep(0)
        await self.session.close("close_requested")
        self.assertEqual(
            self.log[self.after_start :],
            ["delegation.stop_handing", "session task cancelled", "frontend.close", "delegation.close"],
        )
        self.assertNotIn("session.closed", self.delegation.seen_at_close)
        self.assertEqual(self.primary.kinds()[-1], "session.closed")

    async def test_a_frontend_whose_close_fails_still_lets_the_delegation_side_close(self):
        await self.start()
        self.frontend.close_error = RuntimeError("pipeline stuck")
        await self.session.close("close_requested")
        self.assertEqual(
            self.log[self.after_start :], ["delegation.stop_handing", "frontend.close", "delegation.close"]
        )
        self.assertEqual(self.session.status, "closed")
        self.assertEqual(self.primary.kinds().count("session.closed"), 1)

    async def test_a_frontend_close_that_hangs_uses_its_own_limit_and_the_delegation_side_still_closes(self):
        # Each part has its own close limit, and the delegation side always closes, so a hung voice
        # pipeline never leaves the core's work running.
        await self.start()
        self.frontend.close_gate = asyncio.Event()  # not set until the test ends: the frontend's close hangs
        self.addCleanup(self.frontend.close_gate.set)  # then a close that still hangs can finish
        with patch.object(lifecycle, "FRONTEND_CLOSE_SECONDS", 0.1):
            async with asyncio.timeout(5):  # a close with no limit at all fails here instead of hanging the test
                await self.session.close("close_requested")
        self.assertEqual(
            self.log[self.after_start :], ["delegation.stop_handing", "frontend.close", "delegation.close"]
        )
        self.assertEqual(self.session.status, "closed")
        self.assertEqual(self.primary.kinds()[-1], "session.closed")


class InspectableFrontend(FakeFrontend):
    """A frontend adapter that gives its half of ``/history``: the talker's view, and what it holds of the backend."""

    async def history(self):
        return {"frontend": {"messages": [{"role": "user", "content": "Hi."}]}, "backend": {"busy": False}}


class HistoryTests(unittest.IsolatedAsyncioTestCase):
    """``GET /v1/live/sessions/{id}/history`` composes the adapter's half and the delegation part's half."""

    async def history(self, frontend, config=None, update=None):
        core = FakeAdapter()

        async def factory(config, transport, connection):
            return SessionParts(frontend=frontend)

        router, gateway = create_live_router(factory, LiveSettings(), core=core.core)
        app = FastAPI()
        app.include_router(router)
        session = await gateway.register(config or responses_config(), "websocket", None)
        connection = session.subscribe(Collector().send, "primary")
        await session.start()
        self.addAsyncCleanup(session.close)
        if update is not None:
            await session.handle({"type": "session.update", "event_id": "u1", "session": update}, connection)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as http:
            response = await http.get(f"/v1/live/sessions/{session.id}/history")
        self.assertEqual(response.status_code, 200)
        return response.json()

    async def test_a_responses_session_composes_the_adapters_half_and_the_parts_half(self):
        history = await self.history(InspectableFrontend([]))
        self.assertEqual(history["frontend"], {"messages": [{"role": "user", "content": "Hi."}]})
        self.assertEqual(history["backend"], {"model": "org/org/model", "prompt": [], "items": [], "busy": False})

    async def test_the_parts_half_reads_the_sessions_stored_settings(self):
        update = {"delegation": {"responses": {"model": "org/other-model", "instructions": "Be brief."}}}
        backend = (await self.history(InspectableFrontend([]), update=update))["backend"]
        self.assertEqual(backend["model"], "org/other-model")
        self.assertEqual(backend["prompt"], [{"source": "delegation.responses.instructions", "text": "Be brief."}])

    async def test_a_client_mode_session_has_no_backend_half(self):
        config = SessionConfig(model="live-1", delegation={"type": "client"})
        history = await self.history(InspectableFrontend([]), config)
        self.assertEqual(history["backend"], None)
        self.assertEqual(history["frontend"], {"messages": [{"role": "user", "content": "Hi."}]})

    async def test_an_adapter_that_gives_no_half_still_has_the_parts_half(self):
        history = await self.history(FakeFrontend([]))
        self.assertEqual(history, {"frontend": None, "backend": {"model": "org/org/model", "prompt": [], "items": []}})

    async def test_an_adapter_that_holds_nothing_yet_gives_no_half_and_the_route_still_answers(self):
        class NotStarted(FakeFrontend):
            async def history(self):  # the cascade adapter's answer before its start
                return None

        history = await self.history(NotStarted([]))
        self.assertEqual(history, {"frontend": None, "backend": {"model": "org/org/model", "prompt": [], "items": []}})


class RecordingCore(FakeAdapter):
    """A delegation core that notes each conversation call in the shared log; its create may be held or fail."""

    def __init__(self, log):
        super().__init__()
        self.log = log
        self.create_gate: asyncio.Event | None = None
        self.create_error: Exception | None = None

    async def create_conversation(self, metadata):
        self.log.append(("core.create_conversation", metadata))
        if self.create_gate:
            await self.create_gate.wait()
        if self.create_error:
            raise self.create_error
        return await super().create_conversation(metadata)

    async def delete_conversation(self, conversation_id):
        self.log.append(("core.delete_conversation", conversation_id))
        return await super().delete_conversation(conversation_id)


class Listener:
    """The frontend's ``DelegationEvents``, which takes every event and every lost stream."""

    def on_delegation_event(self, delegation_id, event):
        pass

    def on_delegation_lost(self, delegation_id):
        pass


class OpenStream:
    """A response stream that is still open: its next event never comes."""

    def __aiter__(self):
        """The stream is its own iterator."""
        return self

    async def __anext__(self):
        """Wait for an event that never comes."""
        await asyncio.Event().wait()


class HungDeleteCore(RecordingCore):
    """A delegation core whose ``DELETE`` never returns, and whose responses stay open with no event."""

    async def delete_conversation(self, conversation_id):
        self.log.append(("core.delete_conversation", conversation_id))
        await asyncio.Event().wait()

    async def create_response(self, params):
        await super().create_response(params)
        return OpenStream()


class ConversationTests(unittest.IsolatedAsyncioTestCase):
    """The session's conversation with its delegation core."""

    def build(self, core=RecordingCore):
        self.log: list = []
        self.primary = Collector()
        self.frontend = FakeFrontend(self.log, self.primary)
        self.core = core(self.log)
        config = responses_config()
        parts = SessionParts(frontend=self.frontend)
        self.session = LiveProtocolSession(config, "websocket", parts, core=self.core.core, codec=wire_codec(config))
        self.session.subscribe(self.primary.send, "primary")
        self.created = ("core.create_conversation", {"live_session_id": self.session.id})
        self.deleted = ("core.delete_conversation", FakeAdapter.CONVERSATION)

    async def test_the_conversation_is_created_before_the_frontend_starts_and_deleted_after_the_frontend_closes(self):
        self.build()
        await self.session.start("start_1")
        self.assertEqual(self.session.delegation.conversation, FakeAdapter.CONVERSATION)
        await self.session.close("close_requested")
        self.assertEqual(self.log, [self.created, "frontend.start", "frontend.close", self.deleted])
        self.assertIsNone(self.session.delegation.conversation)
        self.assertEqual(self.primary.kinds()[-1], "session.closed")

    async def test_a_conversation_the_core_cannot_create_fails_the_start(self):
        self.build()
        self.core.create_error = RuntimeError("the core cannot create a conversation")
        with self.assertRaises(RuntimeError):
            await self.session.start("start_1")
        self.assertEqual(self.session.status, "closed")
        self.assertEqual(self.log, [self.created, "frontend.close"])  # no frontend started, nothing to delete
        self.assertEqual((await self.primary.wait_for("error"))["error"]["code"], "pipeline_error")
        self.assertEqual((await self.primary.wait_for("session.closed"))["reason"], "connection_lost")
        self.assertNotIn("session.started", self.primary.kinds())

    async def test_a_conversation_created_after_its_session_closed_is_deleted_and_no_frontend_starts(self):
        self.build()
        self.core.create_gate = asyncio.Event()
        starting = asyncio.create_task(self.session.start("start_1"))
        await asyncio.sleep(0.05)
        await self.session.close("remote_hangup")
        self.core.create_gate.set()
        await starting
        self.assertEqual(self.session.status, "closed")
        # The close found no conversation to delete; the open deletes the one the core created meanwhile. The start's
        # early-close path then closes the frontend again, and a second close of the server delegation does nothing.
        self.assertEqual(self.log, [self.created, "frontend.close", self.deleted, "frontend.close"])
        self.assertNotIn("session.started", self.primary.kinds())
        self.assertEqual(self.primary.kinds().count("session.closed"), 1)

    async def test_the_delegation_parts_close_limit_is_its_cores_budget_plus_the_relay(self):
        # The server part's limit is its core's close budget (the client says it) plus the relay of the ends, so a
        # slow stop is not cut; a client that says nothing gets the default limit.
        self.build()
        part = self.session.delegation
        self.assertEqual(part.close_seconds, server_delegation.CLOSE_SECONDS)  # the client says nothing
        part.client.close_seconds = 15.0
        self.assertEqual(part.close_seconds, 15.0 + server_delegation.RELAY_SECONDS)
        part.client.close_seconds = None  # a core with no bound of its own: the gateway's own limit still holds
        self.assertEqual(part.close_seconds, server_delegation.CLOSE_SECONDS)

    async def test_a_delete_that_outlasts_the_close_limit_cancels_the_readers_and_the_session_still_closes(self):
        # Close step 4: within its own close limit, the server delegation deletes the conversation and
        # its readers end; readers still running when the limit ends are cancelled. A DELETE that hangs (an in-process
        # core still stopping a round, or a slow core over HTTP) must not keep the session from closing.
        self.build(HungDeleteCore)
        await self.session.start("start_1")
        self.session.delegation.listen(Listener())  # listen() comes before the first delegate()
        await self.session.delegation.delegate(HI)
        [reader] = self.session.delegation.readers
        self.session.delegation.client.close_seconds = 0.1  # the core's close budget: the part adds its relay share
        with patch.object(server_delegation, "RELAY_SECONDS", 0.1):
            async with asyncio.timeout(5):  # a close that waits for its readers forever fails here instead of hanging
                await self.session.close("close_requested")
        await asyncio.wait({reader}, timeout=1)
        self.assertTrue(reader.cancelled())
        self.assertEqual(self.log, [self.created, "frontend.start", "frontend.close", self.deleted])
        self.assertEqual((self.session.status, self.primary.kinds()[-1]), ("closed", "session.closed"))


if __name__ == "__main__":
    unittest.main()
