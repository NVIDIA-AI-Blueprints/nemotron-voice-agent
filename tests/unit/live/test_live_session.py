# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D103

"""One live session driven as its connections drive it: client commands in, server events out."""

import asyncio
import base64
import json
import unittest
from functools import partial
from unittest import mock

import anyio
from loguru import logger

from gateways.live.events import fanout
from gateways.live.gateway.websocket import TRY_AGAIN_LATER, PrimarySocketView, close_dropped
from gateways.live.protocol import ProtocolError, SessionConfig
from gateways.live.session import LiveProtocolSession, lifecycle
from gateways.live.session.audio_input import MAX_AUDIO_CHARS
from tests.unit.live_helpers import ClientSocket, Collector, FakeAdapter, responses_config

CLIENT = SessionConfig(model="live-1", delegation={"type": "client"})
AUDIO = base64.b64encode(b"\x01\x00" * 480).decode()
APPENDS = ("session.instructions.append", "session.thinking.append", "session.commentary.append")
NOTE = {"type": "session.thinking.append", "delegation_id": None, "content": "x"}


async def never(message):
    await asyncio.Event().wait()


async def broken(message):
    raise ConnectionError("peer gone")


class SessionCase(unittest.IsolatedAsyncioTestCase):
    async def start(self, config=None, transport="websocket", **options):
        """Start a session on a fake frontend; a sideband attaches once it is active."""
        self.frontend, config = FakeAdapter(), config or responses_config()
        self.primary, self.sideband = Collector(), Collector()
        parts, core = self.frontend.parts(), self.frontend.core
        if transport == "websocket":
            self.socket = PrimarySocketView(ClientSocket(self.primary), config.with_wire_format().audio.format)
            self.session = LiveProtocolSession(config, transport, parts, core=core, socket=self.socket, **options)
            self.socket.attach(self.session)
            self.keys = {"primary": self.socket.connection}
        else:
            self.session = LiveProtocolSession(config, transport, parts, core=core, **options)
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


class LifecycleTests(SessionCase):
    async def test_start_announces_the_resolved_session_once(self):
        await self.start(SessionConfig(model="live-1").with_wire_format())
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
                if count:
                    self.frontend.context_tokens = mock.Mock(return_value=count)
                    self.frontend.context_window_tokens = mock.Mock(return_value=window)
                usage = await self.primary.wait_for("session.usage.updated")
                if ratio:
                    self.assertEqual(usage["context_window"]["usage_ratio"], ratio)
                else:
                    self.assertTrue(0 < usage["context_window"]["usage_ratio"] < 0.05)
                self.assertIn("seconds", usage["usage"])

    async def test_a_frontend_fault_is_a_server_error_without_detail(self):
        records = []
        self.addCleanup(logger.remove, logger.add(lambda message: records.append(message.record), level="ERROR"))
        for fault in (RuntimeError("the pipeline is stuck"), ValueError("an internal value went wrong")):
            with self.subTest(fault=fault):
                await self.start()
                self.frontend.set_muted = mock.AsyncMock(side_effect=fault)
                await self.handle({"type": "session.input_audio.mute", "event_id": "m"})
                [error] = [event["error"] for event in self.primary.events if event["type"] == "error"]
                self.assertEqual([error["type"], error["code"], error["client_event_id"]], ["server_error"] * 2 + ["m"])
                self.assertNotIn(str(fault), error["message"])
                await self.handle({**NOTE, "event_id": "t"})
                self.assertEqual(self.primary.events[-1]["type"], "session.thinking.appended")
                [logged] = [record for record in records if str(fault) in record["message"]]
                self.assertEqual(logged["extra"].get("stream_id"), self.session.id)

    async def test_close_sends_session_closed_once_and_logs_one_line(self):
        lines = []
        self.addCleanup(logger.remove, logger.add(lambda message: lines.append(message.record), level="INFO"))
        await self.start()
        await self.handle({**NOTE, "event_id": "late"})
        await self.handle({"type": "session.close", "event_id": "bye"})
        await self.session.close("expired")
        await self.handle({"type": "session.input_audio.mute"})
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

    async def test_a_cancelled_close_finishes_and_a_second_close_waits(self):
        # Whoever asked to close may be cancelled mid-close (a client that leaves); the close still finishes in order.
        for how in ("anyio scope", "task"):
            with self.subTest(how=how):
                seen = []

                async def on_closed(session, seen=seen):
                    seen.append(session.status)
                    raise RuntimeError("the registry failed")

                await self.start(on_closed=on_closed)
                self.frontend.hold["close"] = asyncio.Event()
                scope = anyio.CancelScope()

                async def caller(scope=scope):
                    with scope:
                        await self.session.close("connection_lost")

                task = asyncio.create_task(caller() if how == "anyio scope" else self.session.close("connection_lost"))
                await self.until(lambda: "close" in self.frontend.log)
                second = asyncio.create_task(self.session.close("expired"))
                if how == "anyio scope":
                    scope.cancel()
                    await task
                else:
                    task.cancel()
                    with self.assertRaises(asyncio.CancelledError):
                        await task
                await asyncio.sleep(0.02)
                self.assertFalse(second.done())
                self.frontend.hold["close"].set()
                await self.until(lambda seen=seen: seen)
                self.assertEqual((seen, self.session.status, self.frontend.closed), (["closed"], "closed", True))
                self.assertEqual(self.primary.kinds().count("session.closed"), 1)
                await second
                self.assertEqual(self.primary.events[-1]["reason"], "connection_lost")


class CommandTests(SessionCase):
    async def test_mute_and_unmute_are_acknowledged_with_the_client_event(self):
        await self.start()
        for kind, muted in (("mute", True), ("unmute", False)):
            await self.handle({"type": f"session.input_audio.{kind}", "event_id": kind})
            self.assertEqual(self.frontend.muted, muted)
            ack = self.primary.events[-1]
            self.assertEqual((ack["type"], ack["client_event_id"]), (f"session.input_audio.{kind}d", kind))

    async def test_audio_comes_checked_from_the_primary_websocket_only(self):
        await self.start()
        append = {"type": "session.input_audio.append", "audio": AUDIO}
        self.assertEqual(await self.socket.handle(json.dumps(append)), base64.b64decode(AUDIO))
        bad = {"b64": "!!!not base64", "odd": base64.b64encode(b"\x01").decode(), "type": 5}
        for event_id, audio in {**bad, "big": "A" * (MAX_AUDIO_CHARS + 8)}.items():
            self.assertIsNone(await self.socket.handle(json.dumps({**append, "event_id": event_id, "audio": audio})))
        await self.session.fanout.drain()
        refused = [("invalid_value", "audio", event_id) for event_id in ("b64", "odd", "type", "big")]
        self.assertEqual(self.errors(), refused)
        oversized = f"Invalid or oversized audio chunk: at most {MAX_AUDIO_CHARS} characters"
        self.assertEqual(self.primary.events[-1]["error"]["message"], oversized)
        self.assertEqual(self.sideband.kinds(), [])
        # G.711 takes one byte per sample, so every chunk is whole.
        await self.start(responses_config(audio={"format": {"type": "audio/pcmu"}}))
        self.assertEqual(await self.socket.handle(json.dumps({**append, "audio": "AQ=="})), b"\x01")
        await self.handle(append, "sideband")
        self.assertIn("only allowed on a primary WebSocket", self.sideband.events[-1]["error"]["message"])
        await self.start(transport="webrtc")
        await self.handle(append)
        self.assertIn("only allowed on a primary WebSocket", self.primary.events[-1]["error"]["message"])
        self.assertEqual(self.frontend.audio, [])

    async def test_unknown_malformed_and_unsupported_commands_are_refused(self):
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

    async def test_response_commands_need_responses_mode_and_their_shape(self):
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
                self.assertEqual(self.frontend.typed_texts, [])


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
        port = self.frontend.delegation
        self.assertEqual((port.target, port.reports_events, port.acknowledgement_delay_ms), ("client", False, 0))
        for patch in ({"delegation": {"acknowledgement_delay_ms": 250}}, {}, {"delegation": None}, {"delegation": {}}):
            updated = await self.update(patch)
            self.assertEqual((updated["type"], updated["client_event_id"]), ("session.updated", "u"))
        self.assertEqual(updated["session"]["delegation"], {"type": "client", "acknowledgement_delay_ms": 250})
        await self.assert_refused(
            ({"delegation": {"type": "responses"}}, "immutable_field_update", "session.delegation.type"),
            ({"delegation": {"responses": {"model": "m"}}}, "unknown_parameter", "session.delegation.responses"),
            ({"triggers": [{"event_type": "session.input_transcript.delta"}]}, "invalid_value", "session.triggers"),
            ({"instructions": "Be terse."}, "immutable_field_update", "session.instructions"),
            ({"voice": "marin"}, "unknown_parameter", "session.voice"),
        )
        self.assertEqual(port.acknowledgement_delay_ms, 250)

    async def test_a_responses_session_merges_settings_and_keeps_its_type(self):
        await self.start()
        patch = {"type": "responses", "responses": {"reasoning": {"effort": "high"}, "max_output_tokens": 64}}
        updated = await self.update({"delegation": {**patch, "acknowledgement_delay_ms": 400}})
        responses = updated["session"]["delegation"]["responses"]
        self.assertEqual(responses["reasoning"], {"effort": "high", "summary": "auto"})
        self.assertEqual((responses["max_output_tokens"], self.frontend.delegation.acknowledgement_delay_ms), (64, 400))
        await self.assert_refused(
            ({"delegation": None}, "immutable_field_update", "session.delegation"),
            ({"delegation": {"type": "client"}}, "immutable_field_update", "session.delegation.type"),
            ({"delegation": {"responses": {"bogus": 1}}}, "unknown_parameter", "session.delegation.responses.bogus"),
        )
        self.assertEqual(self.frontend.delegation.acknowledgement_delay_ms, 400)


class AppendTests(SessionCase):
    async def test_appends_are_checked_and_acked_at_the_input_clock(self):
        await self.start()
        await self.frontend.session.advance_input_clock(150)
        await self.frontend.session.advance_input_clock(100)
        for kind in APPENDS:
            await self.handle({"type": kind, "event_id": kind, "delegation_id": None, "content": "hello"})
        acks = [(e["type"], e["client_event_id"], e["start_ms"], e["end_ms"]) for e in self.primary.events[1:]]
        self.assertEqual(acks, [(f"{kind}ed", kind, 150, 150) for kind in APPENDS])
        for fields in (
            {"content": "x"},
            {"delegation_id": None, "content": None},
            {"delegation_id": None, "content": "x" * 3000},
            {"delegation_id": "item_known", "content": "x"},
        ):
            await self.handle({"type": "session.commentary.append", "event_id": "e", **fields})
        params = [param for _, param, _ in self.errors()]
        self.assertEqual(params, ["delegation_id", "content", "content", "delegation_id"])
        self.assertEqual(self.frontend.appended, [(kind.split(".")[1], "hello", kind, None) for kind in APPENDS])

    async def test_client_appends_may_be_empty_and_name_only_announced_ids(self):
        # Empty thinking that names a delegation is how a client cancels that delegation's pending acknowledgement.
        await self.start(CLIENT)
        issued = await self.frontend.delegation.delegate([{"role": "user", "content": "sent nowhere"}])
        created = await self.primary.wait_for("session.delegation.created")
        self.assertEqual(created["delegation"], {"id": issued, "type": "delegation", "target": "client"})
        sent = [(kind, kind, issued, "") for kind in APPENDS] + [("blank", APPENDS[1], None, " ")]
        for event_id, kind, delegation, text in sent:
            await self.handle({"type": kind, "event_id": event_id, "delegation_id": delegation, "content": text})
        for delegation_id in ("item_1", "", 5, {"id": issued}):
            await self.handle({"type": APPENDS[1], "event_id": "a1", "delegation_id": delegation_id, "content": ""})
        await self.handle({"type": APPENDS[1], "event_id": "t2", "delegation_id": None, "content": "x" * 3000})
        delivered = [(kind.split(".")[1], text, event_id, delegation) for event_id, kind, delegation, text in sent]
        self.assertEqual(self.frontend.appended, delivered)
        refused = [("invalid_value", "delegation_id", "a1")] * 4 + [("invalid_value", "content", "t2")]
        self.assertEqual(self.errors(), refused)
        self.assertEqual(self.primary.events[-1]["error"]["message"], "content must be a string of at most 500 tokens")

    async def test_appends_past_the_limit_in_any_minute_are_refused(self):
        await self.start(max_appends_per_minute=2)
        with mock.patch("gateways.live.session.appends.time") as clock:
            for at, kind, event_id, content in (
                (0, APPENDS[0], "bad", None),
                (0, APPENDS[0], "a", ""),
                (50, APPENDS[1], "b", "x"),
                (59, APPENDS[2], "c", "x"),
                (59, APPENDS[2], "bad2", None),
                (61, APPENDS[2], "d", "x"),
                (62, APPENDS[0], "e", "x"),  # "b" is still in it: the minute slides with each append
            ):
                clock.monotonic.return_value = 1000.0 + at
                await self.handle({"type": kind, "event_id": event_id, "delegation_id": None, "content": content})
        acks = [event["client_event_id"] for event in self.primary.events if event["type"].endswith(".appended")]
        self.assertEqual(acks, ["a", "b", "d"])
        self.assertEqual([event_id for *_, event_id, _ in self.frontend.appended], ["a", "b", "d"])
        refused, limited = [(code, event_id) for code, _, event_id in self.errors()], "rate_limit_exceeded"
        self.assertEqual(refused, [("invalid_value", "bad"), (limited, "c"), ("invalid_value", "bad2"), (limited, "e")])
        self.assertEqual(self.primary.events[4]["error"]["message"], "Too many appends: at most 2 per minute")


class ConnectionTests(SessionCase):
    async def test_errors_go_to_their_commands_connection_others_to_all(self):
        records = []
        self.addCleanup(logger.remove, logger.add(lambda message: records.append(message.record), level="WARNING"))
        await self.start()
        bad = {"type": "session.thinking.append", "event_id": "t1", "content": "no delegation_id"}
        await self.handle(bad, "sideband")
        self.assertEqual((self.errors(self.sideband), self.errors()), ([("invalid_value", "delegation_id", "t1")], []))

        async def take_thinking(text, client_event_id=None, delegation_id=None):
            await self.frontend.session.report_error("The context could not be taken.", client_event_id)

        self.frontend.append_thinking = take_thinking
        await self.handle({**bad, "event_id": "t2", "delegation_id": None}, "sideband")
        self.assertEqual((self.errors(self.sideband)[-1], self.errors()), (("pipeline_error", None, "t2"), []))
        self.assertNotIn("info", self.sideband.kinds())
        await self.frontend.session.report_error("The speech service failed.")
        await self.frontend.session.report_error("The speech service is gone.", fatal=True)
        await self.session.fanout.drain()
        info = {"type": "info", "code": "pipeline_error", "message": "The speech service failed."}
        for collector in (self.primary, self.sideband):
            notice, error = collector.events[-2:]
            self.assertEqual({key: value for key, value in notice.items() if key != "event_id"}, info)
            self.assertEqual((error["type"], error["error"]["code"]), ("error", "pipeline_error"))
        logged = [(record["level"].name, record["message"]) for record in records]
        self.assertIn(("WARNING", f"session={self.session.id} pipeline_error=The speech service failed."), logged)
        await self.session.unsubscribe(self.keys["sideband"])
        await self.handle({"type": "session.input_audio.mute", "event_id": "m"}, "sideband")
        self.assertFalse(self.frontend.muted)

    async def test_data_channel_permissions_filter_only_the_primary(self):
        allowed = ["session.started", "info", "error", {"type": "response.event", "response_event": "x.y"}]
        permissions = {"allowed_client_events": ["session.input_audio.mute"], "allowed_server_events": allowed}
        await self.start(SessionConfig(model="live-1", client={"data_channel": permissions}), transport="webrtc")
        self.assertEqual((await self.primary.wait_for("info"))["code"], "data_channel_permissions")
        issued = await self.frontend.delegation.delegate([])
        answer = {"type": "session.commentary.append", "event_id": "c1", "delegation_id": issued, "content": "Done."}
        await self.handle(answer)
        self.assertEqual(self.errors(), [("permission_denied", None, "c1")])
        await self.handle(answer, "sideband")
        self.assertEqual(self.frontend.appended, [("commentary", "Done.", "c1", issued)])
        for event in ({"type": "x.y"}, {"type": "other"}):
            await self.session.emit("response.event", delegation_id="d", event=event)
        await self.session.emit("session.usage.updated", usage={})
        await self.session.fanout.drain()
        self.assertEqual(self.primary.kinds(), ["session.started", "info", "error", "response.event"])
        delegation = ["session.delegation.created", "session.commentary.appended"]
        self.assertEqual(self.sideband.kinds(), [*delegation, *["response.event"] * 2, "session.usage.updated"])

    async def test_at_most_eight_sidebands_see_what_follows_their_attach(self):
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
            await self.session.unsubscribe(keys[0])
        await self.session.emit("session.usage.updated", usage={"seconds": 2})
        await self.session.fanout.drain()
        self.assertEqual([len(observer.events) for observer in observers], [1] + [2] * 6)

    async def test_a_sideband_that_falls_behind_or_fails_is_closed(self):
        await self.start()
        behind, failed = mock.Mock(close=mock.AsyncMock()), mock.Mock(close=mock.AsyncMock())
        self.session.subscribe(never, "sideband", partial(close_dropped, behind))
        self.session.subscribe(broken, "sideband", partial(close_dropped, failed))
        for _ in range(200):
            await self.session.emit("session.usage.updated", usage={})
            await asyncio.sleep(0)
        await self.until(lambda: behind.close.await_count and failed.close.await_count)
        again = "attach again to observe"
        reason = "The sideband connection fell 128 events behind the session; " + again
        self.assertEqual(behind.close.await_args, mock.call(code=TRY_AGAIN_LATER, reason=reason))
        reason = "An event could not be sent to the sideband connection; " + again
        self.assertEqual(failed.close.await_args, mock.call(code=TRY_AGAIN_LATER, reason=reason))
        await self.session.fanout.drain()
        self.assertEqual((self.primary.kinds().count("session.usage.updated"), self.session.status), (200, "active"))

    async def test_a_lagging_stalled_or_failed_primary_ends_the_session(self):
        for send, events, stall in ((never, 1100, 3), (never, 1, 0.05), (broken, 1, 3)):
            with self.subTest(events=events, stall=stall), mock.patch.object(fanout, "SEND_TIMEOUT_SECONDS", stall):
                frontend, observer = FakeAdapter(), Collector()
                session = LiveProtocolSession(responses_config(), "websocket", frontend.parts(), core=frontend.core)
                session.subscribe(send, "primary")
                session.subscribe(observer.send, "sideband")
                await session.start()
                for _ in range(events):
                    await session.emit("session.usage.updated", usage={})
                    await asyncio.sleep(0)
                closed = await observer.wait_for("session.closed", timeout=5)
                await session.close()
                self.assertEqual((closed["reason"], frontend.closed), ("connection_lost", True))


if __name__ == "__main__":
    unittest.main()
