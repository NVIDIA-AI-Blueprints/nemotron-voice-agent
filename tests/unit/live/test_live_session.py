# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D103, D107

"""One live session driven as its connections drive it: client commands in, server events out.

The frontend adapter is a fake that notes what it is handed and reaches the session through the views it got at start.
A responses session runs on a core that answers nothing (``tests/unit/live_helpers.py``).
"""

import asyncio
import base64
import unittest
from functools import partial
from unittest import mock

import anyio
from loguru import logger

from gateways.live.events import fanout
from gateways.live.gateway.websocket import TRY_AGAIN_LATER, close_dropped, wire_codec
from gateways.live.protocol import ProtocolError, SessionConfig
from gateways.live.session import LiveProtocolSession, lifecycle
from tests.unit.live_helpers import Collector, FakeAdapter, responses_config

CLIENT = SessionConfig(model="live-1", delegation={"type": "client"})
AUDIO = base64.b64encode(b"\x01\x00" * 480).decode()
APPENDS = ("session.instructions.append", "session.thinking.append", "session.commentary.append")
NOTE = {"type": "session.thinking.append", "delegation_id": None, "content": "x"}


class Frontend(FakeAdapter):
    """The frontend adapter: it notes each append with its ids, the rate of the audio it is fed and the typed questions.

    Its close waits for ``hold_close`` when it is set.
    """

    def __init__(self):
        super().__init__()
        self.rates, self.typed_texts = [], []
        self.hold_close, self.close_started = None, asyncio.Event()

    async def feed_audio(self, pcm, sample_rate):
        self.rates.append(sample_rate)
        await super().feed_audio(pcm, sample_rate)

    async def append_instructions(self, text, client_event_id=None, delegation_id=None):
        self.appended.append(("instructions", text, client_event_id, delegation_id))

    async def append_thinking(self, text, client_event_id=None, delegation_id=None):
        self.appended.append(("thinking", text, client_event_id, delegation_id))

    def typed(self, text):
        self.typed_texts.append(text)

    async def close(self):
        self.close_started.set()
        if self.hold_close is not None:
            await self.hold_close.wait()
        await super().close()


async def never(message):
    await asyncio.Event().wait()


async def broken(message):
    raise ConnectionError("peer gone")


class SessionCase(unittest.IsolatedAsyncioTestCase):
    async def start(self, config=None, transport="websocket", **options):
        self.frontend = Frontend()
        config = config or responses_config()
        codec = wire_codec(config) if transport == "websocket" else None  # only the WebSocket edge hands one in
        self.session = LiveProtocolSession(
            config, transport, self.frontend.parts(), core=self.frontend.core, codec=codec, **options
        )
        self.primary, self.sideband = Collector(), Collector()
        self.keys = {"primary": self.session.subscribe(self.primary.send, "primary")}
        await self.session.start("start_1")
        await self.primary.wait_for("session.started")
        self.keys["sideband"] = self.session.subscribe(self.sideband.send, "sideband")
        self.addAsyncCleanup(self.session.close)

    async def handle(self, message, source="primary"):
        """Run a command from a connection, then let every connection receive what it produced."""
        await self.session.handle(message, self.keys[source])
        await self.session.fanout.drain()

    def errors(self, collector=None):
        events = (collector or self.primary).events
        return [(e["error"]["code"], e["error"].get("param"), e.get("client_event_id")) for e in events if "error" in e]

    async def until(self, predicate):
        async with asyncio.timeout(3):
            while not predicate():
                await asyncio.sleep(0.005)


class StartTests(SessionCase):
    async def test_start_announces_the_resolved_session_once(self):
        await self.start(SessionConfig(model="live-1").with_wire_format())  # as the WebSocket edge hands it in
        started = self.primary.events[0]["session"]
        self.assertEqual((self.primary.events[0]["client_event_id"], started["status"]), ("start_1", "active"))
        self.assertTrue(started["id"].startswith("live_"))
        self.assertEqual(started["delegation"], {"type": "client", "acknowledgement_delay_ms": 0})
        self.assertEqual(started["audio"]["format"], {"type": "audio/pcm", "rate": 24000})
        self.assertNotIn("triggers", started)
        self.assertTrue(self.frontend.started)
        with self.assertRaises(ProtocolError) as again:
            await self.session.start()
        self.assertEqual(again.exception.code, "session_already_started")

    async def test_usage_reports_how_full_the_frontends_context_is(self):
        for count, window, ratio in ((64_000, None, 0.5), (64_000, 256_000, 0.25), (None, None, None)):
            with self.subTest(window=window, count=count), mock.patch.object(lifecycle, "USAGE_INTERVAL_SECONDS", 0.01):
                await self.start()
                if count:  # the frontend's own count, out of its own window or the gateway's default
                    self.frontend.context_tokens = mock.Mock(return_value=count)
                    self.frontend.context_window_tokens = mock.Mock(return_value=window)
                usage = await self.primary.wait_for("session.usage.updated")
                if ratio:
                    self.assertEqual(usage["context_window"]["usage_ratio"], ratio)
                else:  # with no count, the size of the startup settings
                    self.assertTrue(0 < usage["context_window"]["usage_ratio"] < 0.05)
                self.assertIn("seconds", usage["usage"])

    async def test_a_fault_in_the_frontend_is_a_server_error_without_detail_and_the_session_keeps_serving(self):
        records = []
        sink = logger.add(lambda message: records.append(message.record), level="ERROR")
        self.addCleanup(logger.remove, sink)
        for fault in (RuntimeError("the pipeline is stuck"), ValueError("an internal value went wrong")):
            with self.subTest(fault=fault):
                await self.start()
                self.frontend.set_muted = mock.AsyncMock(side_effect=fault)
                await self.handle({"type": "session.input_audio.mute", "event_id": "m"})
                [error] = [event["error"] for event in self.primary.events if event["type"] == "error"]
                self.assertEqual([error["type"], error["code"], error["client_event_id"]], ["server_error"] * 2 + ["m"])
                self.assertNotIn(str(fault), error["message"])  # the cause stays in the log
                await self.handle({**NOTE, "event_id": "t"})
                self.assertEqual(self.primary.events[-1]["type"], "session.thinking.appended")
                [logged] = [record for record in records if str(fault) in record["message"]]
                self.assertEqual(logged["extra"].get("stream_id"), self.session.id)  # the log line names the session


class CommandTests(SessionCase):
    async def test_mute_and_unmute_are_acknowledged_with_the_client_event(self):
        await self.start()
        for kind, muted in (("mute", True), ("unmute", False)):
            await self.handle({"type": f"session.input_audio.{kind}", "event_id": kind})
            self.assertEqual(self.frontend.muted, muted)
            ack = self.primary.events[-1]
            self.assertEqual((ack["type"], ack["client_event_id"]), (f"session.input_audio.{kind}d", kind))

    async def test_audio_is_a_primary_websocket_command_and_is_checked(self):
        await self.start()
        await self.handle({"type": "session.input_audio.append", "audio": AUDIO})
        self.assertEqual((len(self.frontend.audio[0]), self.frontend.rates), (960, [24000]))  # 24 kHz PCM as it came
        for bad in ("!!!not base64", base64.b64encode(b"\x01").decode(), 5):
            await self.handle({"type": "session.input_audio.append", "audio": bad})
        self.assertEqual([param for _, param, _ in self.errors()], ["audio"] * 3)
        await self.handle({"type": "session.input_audio.append", "audio": AUDIO}, "sideband")
        self.assertIn("only allowed on a primary WebSocket", self.sideband.events[-1]["error"]["message"])
        await self.start(transport="webrtc")
        await self.handle({"type": "session.input_audio.append", "audio": AUDIO})
        self.assertIn("only allowed on a primary WebSocket", self.primary.events[-1]["error"]["message"])
        self.assertEqual(self.frontend.audio, [])

    async def test_unknown_malformed_unsupported_and_repeated_start_commands_are_refused(self):
        await self.start()
        unsupported = ("session.output_audio.fixture.play", "decision.create", "decision.item.create")
        unsupported += ("decision.item.delete",)
        for message, param, words in (
            ({"type": "session.unknown"}, "type", "Unknown client event"),
            ({"nope": 1}, None, "Expected a JSON event object"),
            ("text", None, "Expected a JSON event object"),
            ({"type": "session.start"}, None, "already started"),
            *(({"type": kind}, "type", f"({kind}) is not supported by this server") for kind in unsupported),
        ):
            with self.subTest(message=message):
                await self.handle(message)
                error = self.primary.events[-1]["error"]
                self.assertEqual(error.get("param"), param)
                self.assertIn(words, error["message"])

    async def test_response_commands_need_responses_delegation_and_their_own_shape(self):
        typed = {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "Two lattes."}]}
        for config in (SessionConfig(model="live-1"), CLIENT):
            with self.subTest(config=config.delegation):
                await self.start(config)
                for command in (
                    {"type": "response.item.create", "event_id": "i1", "item": typed},
                    {"type": "response.create", "event_id": "r1"},
                    {"type": "response.create", "event_id": "r2", "extra": True},
                    {"type": "response.item.create", "event_id": "i2", "item": "nope"},
                ):
                    await self.handle(command)
                messages = [event["error"]["message"] for event in self.primary.events if event["type"] == "error"]
                refused = [("invalid_value", None, "i1"), ("invalid_value", None, "r1"), ("invalid_value", None, "r2")]
                self.assertEqual(self.errors(), [*refused, ("invalid_value", "item", "i2")])
                self.assertEqual(messages[:2], ["This command requires Responses delegation"] * 2)
                self.assertIn("accepts only type and event_id", messages[2])
                self.assertEqual(self.frontend.typed_texts, [])  # a refused question never reaches the voice


class UpdateTests(SessionCase):
    async def update(self, patch, event_id="u"):
        before = len(self.primary.events)
        await self.handle({"type": "session.update", "event_id": event_id, "session": patch})
        [answer] = self.primary.events[before:]
        return answer

    async def assert_refused(self, *cases):
        for patch, code, param in cases:
            with self.subTest(patch=patch):
                await self.update(patch, "bad")
                self.assertEqual(self.errors()[-1], (code, param, "bad"))

    async def test_a_client_session_changes_only_its_acknowledgement_delay(self):
        await self.start(CLIENT)
        port = self.frontend.delegation  # the frontend's port, which reads the settings when asked
        self.assertEqual((port.target, port.reports_events, port.acknowledgement_delay_ms), ("client", False, 0))
        for patch in ({"delegation": {"acknowledgement_delay_ms": 250}}, {}, {"delegation": None}, {"delegation": {}}):
            updated = await self.update(patch)
            self.assertEqual((updated["type"], updated["client_event_id"]), ("session.updated", "u"))
        self.assertEqual(updated["session"]["delegation"], {"type": "client", "acknowledgement_delay_ms": 250})
        await self.assert_refused(
            ({"delegation": {"type": "responses"}}, "immutable_field_update", "session.delegation.type"),
            ({"delegation": {"responses": {"model": "m"}}}, "unknown_parameter", "session.delegation.responses"),
            ({"delegation": {"decisions": {}}}, "unknown_parameter", "session.delegation.decisions"),
            ({"triggers": [{"event_type": "session.input_transcript.delta"}]}, "invalid_value", "session.triggers"),
            ({"instructions": "Be terse."}, "immutable_field_update", "session.instructions"),
            ({"audio": {"output": {"voice": "cedar"}}}, "immutable_field_update", "session.audio"),
            ({"store": True}, "immutable_field_update", "session.store"),
            ({"voice": "marin"}, "unknown_parameter", "session.voice"),
        )
        self.assertEqual(port.acknowledgement_delay_ms, 250)

    async def test_a_responses_session_merges_its_settings_and_keeps_its_type(self):
        await self.start()
        patch = {"type": "responses", "responses": {"reasoning": {"effort": "high"}, "max_output_tokens": 64}}
        updated = await self.update({"delegation": {**patch, "acknowledgement_delay_ms": 400}})
        responses = updated["session"]["delegation"]["responses"]
        self.assertEqual(responses["reasoning"], {"effort": "high", "summary": "auto"})  # a nested setting merges
        self.assertEqual((responses["max_output_tokens"], self.frontend.delegation.acknowledgement_delay_ms), (64, 400))
        await self.assert_refused(
            ({"delegation": None}, "immutable_field_update", "session.delegation"),
            ({"delegation": {"type": "client"}}, "immutable_field_update", "session.delegation.type"),
            ({"delegation": {"responses": {"bogus": 1}}}, "unknown_parameter", "session.delegation.responses.bogus"),
        )


class AppendTests(SessionCase):
    async def test_appends_reach_the_frontend_and_are_acknowledged_at_the_input_clock(self):
        await self.start()
        await self.frontend.session.advance_input_clock(150)
        await self.frontend.session.advance_input_clock(100)  # the clock never moves backwards
        for kind in APPENDS:
            await self.handle({"type": kind, "event_id": kind, "delegation_id": None, "content": "hello"})
        self.assertEqual(self.frontend.appended, [(kind.split(".")[1], "hello", kind, None) for kind in APPENDS])
        acks = [(e["type"], e["client_event_id"], e["start_ms"], e["end_ms"]) for e in self.primary.events[1:]]
        self.assertEqual(acks, [(f"{kind}ed", kind, 150, 150) for kind in APPENDS])  # appended context takes no time

    async def test_an_append_is_checked_before_it_reaches_the_frontend(self):
        await self.start()  # responses mode: no append may name a delegation
        for fields in (
            {"content": "x"},
            {"delegation_id": None, "content": "  "},
            {"delegation_id": None, "content": "x" * 3000},
            {"delegation_id": "item_known", "content": "x"},
        ):
            await self.handle({"type": "session.commentary.append", "event_id": "e", **fields})
        params = [param for _, param, _ in self.errors()]
        self.assertEqual(params, ["delegation_id", "content", "content", "delegation_id"])
        self.assertEqual(self.frontend.appended, [])

    async def test_a_client_mode_append_may_name_only_a_delegation_the_session_announced(self):
        await self.start(CLIENT)
        issued = await self.frontend.delegation.delegate([{"role": "user", "content": "sent nowhere"}])
        created = await self.primary.wait_for("session.delegation.created")
        self.assertEqual(created["delegation"], {"id": issued, "type": "delegation", "target": "client"})
        self.assertIn("offset_ms", created)
        for kind in APPENDS:
            await self.handle({"type": kind, "event_id": kind, "delegation_id": issued, "content": "ok"})
        self.assertEqual([appended[2:] for appended in self.frontend.appended], [(kind, issued) for kind in APPENDS])
        # An id it never announced, an empty one or one of another type is refused, before the content is checked.
        for delegation_id in ("item_1", "", 5, {"id": issued}):
            await self.handle({**{"type": APPENDS[1], "event_id": "a1"}, "delegation_id": delegation_id, "content": ""})
        self.assertEqual(self.errors(), [("invalid_value", "delegation_id", "a1")] * 4)
        self.assertEqual(len(self.frontend.appended), 3)

    async def test_appends_past_the_limit_in_any_minute_are_refused_and_never_delivered(self):
        await self.start(max_appends_per_minute=2)
        with mock.patch("gateways.live.session.appends.time") as clock:  # the appends' clock only
            for at, kind, event_id, content in (
                (0, APPENDS[0], "blank", " "),  # refused by its own check: not counted
                (0, APPENDS[0], "a", "x"),
                (50, APPENDS[1], "b", "x"),
                (59, APPENDS[2], "c", "x"),  # the three kinds count against one limit
                (59, APPENDS[2], "blank2", " "),  # past the limit, its own check comes first
                (61, APPENDS[2], "d", "x"),  # "a" has left the minute
                (62, APPENDS[0], "e", "x"),  # "b" is still in it: the minute slides with each append
            ):
                clock.monotonic.return_value = 1000.0 + at
                await self.handle({"type": kind, "event_id": event_id, "delegation_id": None, "content": content})
        acks = [event["client_event_id"] for event in self.primary.events if event["type"].endswith(".appended")]
        self.assertEqual(acks, ["a", "b", "d"])
        self.assertEqual([event_id for *_, event_id, _ in self.frontend.appended], ["a", "b", "d"])
        refused, limited = [(code, event_id) for code, _, event_id in self.errors()], "rate_limit_exceeded"
        self.assertEqual(
            refused, [("invalid_value", "blank"), (limited, "c"), ("invalid_value", "blank2"), (limited, "e")]
        )
        self.assertEqual(self.primary.events[4]["error"]["message"], "Too many appends: at most 2 per minute")


class ConnectionTests(SessionCase):
    async def test_errors_go_to_the_connection_whose_command_caused_them(self):
        await self.start()
        bad = {"type": "session.thinking.append", "event_id": "t1", "content": "no delegation_id"}
        await self.handle(bad, "sideband")
        self.assertEqual((self.errors(self.sideband), self.errors()), ([("invalid_value", "delegation_id", "t1")], []))

        async def take_thinking(text, client_event_id=None, delegation_id=None):
            await self.frontend.session.report_error("The context could not be taken.", client_event_id)

        self.frontend.append_thinking = take_thinking  # a part's failure of a command is that command's error
        await self.handle({**bad, "event_id": "t2", "delegation_id": None}, "sideband")
        self.assertEqual((self.errors(self.sideband)[-1], self.errors()), (("pipeline_error", None, "t2"), []))
        self.assertNotIn("info", self.sideband.kinds())
        await self.session.unsubscribe(self.keys["sideband"])
        await self.handle({"type": "session.input_audio.mute", "event_id": "m"}, "sideband")
        self.assertFalse(self.frontend.muted)  # a detached connection's command is dropped

    async def test_a_failure_no_command_caused_is_an_info_notice_unless_it_is_fatal(self):
        records = []
        sink = logger.add(lambda message: records.append(message.record), level="WARNING")
        self.addCleanup(logger.remove, sink)
        await self.start()
        await self.frontend.session.report_error("The speech service failed.")
        await self.frontend.session.report_error("The speech service is gone.", fatal=True)
        await self.session.fanout.drain()
        info = {"type": "info", "code": "pipeline_error", "message": "The speech service failed."}  # the official event
        for collector in (self.primary, self.sideband):  # every connection gets both
            notice, error = collector.events[-2:]
            self.assertEqual({key: value for key, value in notice.items() if key != "event_id"}, info)
            self.assertEqual((error["type"], error["error"]["code"]), ("error", "pipeline_error"))
        logged = [(record["level"].name, record["message"]) for record in records]
        self.assertIn(("WARNING", f"session={self.session.id} pipeline_error=The speech service failed."), logged)

    async def test_data_channel_permissions_filter_the_primarys_events_and_commands_but_not_a_sidebands(self):
        allowed = ["session.started", "info", "error", {"type": "response.event", "response_event": "x.y"}]
        permissions = {"allowed_client_events": ["session.input_audio.mute"], "allowed_server_events": allowed}
        await self.start(SessionConfig(model="live-1", client={"data_channel": permissions}), transport="webrtc")
        self.assertEqual((await self.primary.wait_for("info"))["code"], "data_channel_permissions")
        issued = await self.frontend.delegation.delegate([])
        answer = {"type": "session.commentary.append", "event_id": "c1", "delegation_id": issued, "content": "Done."}
        await self.handle(answer)
        self.assertEqual(self.errors(), [("permission_denied", None, "c1")])
        await self.handle(answer, "sideband")  # the trusted app server answers over the sideband
        self.assertEqual(self.frontend.appended, [("commentary", "Done.", "c1", issued)])
        for event in ({"type": "x.y"}, {"type": "other"}):
            await self.session.emit("response.event", delegation_id="d", event=event)
        await self.session.emit("session.usage.updated", usage={})
        await self.session.fanout.drain()
        self.assertEqual(self.primary.kinds(), ["session.started", "info", "error", "response.event"])
        delegation = ["session.delegation.created", "session.commentary.appended"]
        self.assertEqual(self.sideband.kinds(), [*delegation, *["response.event"] * 2, "session.usage.updated"])

    async def test_sideband_observers_see_what_follows_their_attach_and_at_most_eight_attach(self):
        await self.start()  # with one sideband
        observers = [Collector() for _ in range(7)]
        keys = [self.session.subscribe(observer.send, "sideband") for observer in observers]
        with self.assertRaises(ProtocolError) as full:
            self.session.subscribe(Collector().send, "sideband")
        self.assertEqual(full.exception.code, "rate_limit_exceeded")
        await self.session.emit("session.usage.updated", usage={"seconds": 1})
        await self.session.fanout.drain()
        # Starlette ends a WebSocket handler by cancelling its anyio scope, which absorbs only its own cancel.
        with anyio.CancelScope() as scope:
            asyncio.get_running_loop().call_soon(scope.cancel)
            await self.session.unsubscribe(keys[0])  # raises if the writer's own cancel escapes the scope
        await self.session.emit("session.usage.updated", usage={"seconds": 2})
        await self.session.fanout.drain()
        self.assertEqual([len(observer.events) for observer in observers], [1] + [2] * 6)

    async def test_a_sideband_that_falls_behind_or_fails_is_closed_with_the_reason_and_the_primary_goes_on(self):
        await self.start()

        class Socket:
            closed = None

            async def close(self, code=1000, reason=None):
                self.closed = (code, reason)

        behind, failed = Socket(), Socket()
        self.session.subscribe(never, "sideband", partial(close_dropped, behind))
        self.session.subscribe(broken, "sideband", partial(close_dropped, failed))
        for _ in range(200):
            await self.session.emit("session.usage.updated", usage={})
            await asyncio.sleep(0)  # the writers run as a server's do
        await self.until(lambda: behind.closed and failed.closed)
        again = "attach again to observe"
        self.assertEqual([behind.closed[0], failed.closed[0]], [TRY_AGAIN_LATER] * 2)
        self.assertEqual(behind.closed[1], "The sideband connection fell 128 events behind the session; " + again)
        self.assertEqual(failed.closed[1], "An event could not be sent to the sideband connection; " + again)
        await self.session.fanout.drain()
        self.assertEqual((self.primary.kinds().count("session.usage.updated"), self.session.status), (200, "active"))

    async def test_a_primary_that_falls_behind_stalls_or_fails_closes_the_session(self):
        for send, events, stall in ((never, 1100, 3), (never, 1, 0.05), (broken, 1, 3)):
            with self.subTest(events=events, stall=stall), mock.patch.object(fanout, "SEND_TIMEOUT_SECONDS", stall):
                frontend, observer = Frontend(), Collector()
                session = LiveProtocolSession(responses_config(), "websocket", frontend.parts(), core=frontend.core)
                session.subscribe(send, "primary")
                session.subscribe(observer.send, "sideband")
                await session.start()
                for _ in range(events):
                    await session.emit("session.usage.updated", usage={})
                    await asyncio.sleep(0)  # the observer keeps up
                closed = await observer.wait_for("session.closed", timeout=5)
                await session.close()
                self.assertEqual((closed["reason"], frontend.closed), ("connection_lost", True))


class CloseTests(SessionCase):
    async def test_close_sends_session_closed_once_after_what_came_before_and_logs_one_line(self):
        lines = []
        sink = logger.add(lambda message: lines.append(message.record), level="INFO")
        self.addCleanup(logger.remove, sink)
        await self.start()
        await self.handle({**NOTE, "event_id": "late"})
        await self.handle({"type": "session.close", "event_id": "bye"})
        await self.session.close("expired")  # a second close changes nothing
        await self.handle({"type": "session.input_audio.mute"})  # a command after the close reaches nothing
        self.assertEqual(self.primary.kinds(), ["session.started", "session.thinking.appended", "session.closed"])
        closed = self.primary.events[-1]
        self.assertEqual((closed["reason"], closed["client_event_id"]), ("close_requested", "bye"))
        self.assertEqual(closed["session"]["status"], "active")  # the final snapshot, as the protocol reports it
        self.assertIn("seconds", closed["usage"])
        self.assertEqual((self.frontend.closed, self.frontend.muted), (True, False))
        [line] = [record for record in lines if " closes: " in record["message"]]
        self.assertEqual(line["extra"].get("stream_id"), self.session.id)
        closes = rf"session={self.session.id} closes: reason=close_requested, lifetime=\d+\.\d s, "
        self.assertRegex(line["message"], closes + r"asked by the client \(primary connection\)")


class InterruptedCloseTests(SessionCase):
    """Whoever asked to close may be cancelled mid-close (a client that leaves); the close still finishes in order."""

    async def start_close(self):
        self.seen = []

        async def on_closed(session):
            self.seen.append(session.status)
            raise RuntimeError("the registry failed")  # logged: the close still finishes

        await self.start(on_closed=on_closed)
        self.frontend.hold_close = asyncio.Event()

    async def assert_closed_in_full(self):
        self.frontend.hold_close.set()
        await self.until(lambda: self.seen)
        self.assertEqual((self.seen, self.session.status, self.frontend.closed), (["closed"], "closed", True))
        self.assertEqual(self.primary.kinds().count("session.closed"), 1)

    async def test_a_close_whose_caller_is_cancelled_still_finishes(self):
        for how in ("anyio scope", "task"):
            with self.subTest(how=how):
                await self.start_close()
                scope = anyio.CancelScope()

                async def caller(scope=scope):
                    with scope:
                        await self.session.close("connection_lost")

                task = asyncio.create_task(caller() if how == "anyio scope" else self.session.close("connection_lost"))
                await self.frontend.close_started.wait()
                if how == "anyio scope":
                    scope.cancel()
                    await task  # the scope absorbs its own cancel; nothing else escapes
                else:
                    task.cancel()
                    with self.assertRaises(asyncio.CancelledError):
                        await task
                await self.assert_closed_in_full()

    async def test_a_second_caller_waits_for_the_first_close_whose_reason_it_keeps(self):
        await self.start_close()
        first = asyncio.create_task(self.session.close("remote_hangup"))
        await self.frontend.close_started.wait()
        second = asyncio.create_task(self.session.close("expired"))
        await asyncio.sleep(0.02)
        self.assertFalse(second.done())
        await self.assert_closed_in_full()
        await asyncio.gather(first, second)
        self.assertEqual(self.primary.events[-1]["reason"], "remote_hangup")


if __name__ == "__main__":
    unittest.main()
