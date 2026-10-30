# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D100, D101, D102, D103, D107

import asyncio
import base64
import unittest
from unittest import mock

import anyio
from pydantic import ValidationError

from live.events import error_event, event
from live.gateway import LiveGateway, LiveSettings
from live.gateway.websocket import run_primary_websocket, wire_codec
from live.media.audio import WireCodec
from live.protocol import AudioFormat, ProtocolError, SessionConfig
from live.session import LiveProtocolSession, lifecycle
from tests.unit.live_helpers import Collector, FakeAdapter, FakeWebSocket, responses_config


class SchemaTests(unittest.TestCase):
    def test_default_mode_is_client(self):
        for delegation in (None, {"type": "client"}, {}):  # {}: a delegation block with no type
            with self.subTest(delegation=delegation):
                self.assertEqual(SessionConfig(model="live-1", delegation=delegation).mode, "client")
        self.assertEqual(SessionConfig(model="live-1").mode, "client")
        self.assertEqual(responses_config().mode, "responses")

    def test_immutable_settings_cannot_change(self):
        for patch, param in (
            ({"instructions": "changed"}, "session.instructions"),
            ({"audio": {"output": {"voice": "cedar"}}}, "session.audio"),
            ({"delegation": None}, "session.delegation"),
            ({"delegation": {"type": "client"}}, "session.delegation.type"),
            ({"delegation": {"responses": {"model": "org/x/y"}}, "store": True}, "session.store"),
        ):
            with self.subTest(patch=patch), self.assertRaises(ProtocolError) as error:
                responses_config().updated(patch)
            self.assertEqual((error.exception.code, error.exception.param), ("immutable_field_update", param))

    def test_a_sparse_nested_update_merges(self):
        original = responses_config()
        updated = original.updated({"delegation": {"responses": {"reasoning": {"effort": "high"}}}})
        self.assertEqual(updated.delegation.responses.reasoning, {"effort": "high", "summary": "auto"})
        self.assertEqual(original.delegation.responses.reasoning["effort"], "low")

    def test_unknown_fields_and_bad_models_are_rejected(self):
        for bad in ({"model": ""}, {}, {"model": "live-1", "surprise": 1}):
            with self.subTest(bad=bad), self.assertRaises(ValidationError):
                SessionConfig(**bad)

    def test_illegal_audio_rates_are_rejected(self):
        for fmt in ({"type": "audio/pcm", "rate": 8000}, {"type": "audio/pcmu", "rate": 24000}):
            with self.subTest(fmt=fmt), self.assertRaises(ValidationError):
                SessionConfig(model="live-1", audio={"format": fmt})

    def test_transport_restrictions(self):
        with self.assertRaises(ProtocolError):
            SessionConfig(model="live-1", client={"data_channel": {}}).validate_transport("websocket")
        with self.assertRaises(ProtocolError):
            SessionConfig(model="live-1", audio={"format": {"type": "audio/pcm", "rate": 16000}}).validate_transport(
                "webrtc"
            )

    def test_responses_mode_needs_its_settings_and_official_tool_types(self):
        with self.assertRaises(ValidationError):
            SessionConfig(model="live-1", delegation={"type": "responses"})
        # Responses settings belong to responses mode only, also when the type is left to its client default: the client
        # delegation has no such key.
        for delegation in ({"type": "client", "responses": {"model": "m"}}, {"responses": {"model": "m"}}):
            with self.subTest(delegation=delegation), self.assertRaises(ValidationError) as error:
                SessionConfig(model="live-1", delegation=delegation)
            problem = error.exception.errors()[0]
            self.assertEqual((problem["type"], problem["loc"]), ("extra_forbidden", ("delegation", "responses")))
        # Every official tool type is accepted (whether the backend supports it is the background check's);
        # a type outside the schema is refused at once.
        SessionConfig(
            model="live-1",
            delegation={"type": "responses", "responses": {"model": "m", "tools": [{"type": "mcp"}]}},
        )
        with self.assertRaises(ValidationError):
            SessionConfig(
                model="live-1",
                delegation={"type": "responses", "responses": {"model": "m", "tools": [{"type": "teleport"}]}},
            )

    def test_input_history_limits(self):
        text = {"type": "input_text", "text": "hi"}
        SessionConfig(model="live-1", input=[{"type": "message", "role": "developer", "content": [text]}])
        for bad in (
            [{"type": "message", "role": "system", "content": [text]}],
            [{"type": "message", "role": "user", "content": [text, text]}],
            [{"type": "message", "role": "user", "content": [{"type": "input_image"}]}],
        ):
            with self.subTest(bad=bad), self.assertRaises(ValidationError):
                SessionConfig(model="live-1", input=bad)

    def test_events_carry_an_id_and_errors_carry_the_client_event(self):
        self.assertTrue(event("x")["event_id"].startswith("event_"))
        body = error_event(ProtocolError("nope", "bad", "param"), "client_1")["error"]
        self.assertEqual((body["code"], body["param"], body["client_event_id"]), ("bad", "param", "client_1"))


class CodecTests(unittest.TestCase):
    def test_pcm_round_trips_at_the_session_rate(self):
        codec = WireCodec(AudioFormat(type="audio/pcm", rate=16000))
        tone = b"\x10\x20" * 1600
        decoded = codec.decode(tone)
        self.assertEqual(len(decoded) % 2, 0)
        self.assertGreater(len(decoded), len(tone))  # 16 kHz -> 24 kHz

    def test_g711_decodes_and_encodes(self):
        for kind in ("audio/pcmu", "audio/pcma"):
            codec = WireCodec(AudioFormat(type=kind, rate=8000))
            pcm = codec.decode(bytes(range(160)))
            self.assertTrue(pcm)
            self.assertEqual(len(codec.encode(pcm)) % 1, 0)

    def test_incomplete_pcm_samples_are_rejected(self):
        with self.assertRaises(ValueError):
            WireCodec().decode(b"\x01")


class MalformedInputTests(unittest.TestCase):
    def test_a_malformed_input_item_is_a_validation_error_and_never_a_raw_exception(self):
        for item in (
            "text",
            {"role": [], "content": []},
            {"role": "user", "content": {"a": 1}},
            {"role": "user", "content": "x"},
            {"role": "user", "content": ["x"]},
            {"role": "user", "content": [{"type": "input_text", "text": 5}]},
        ):
            with self.subTest(item=item), self.assertRaises(ValidationError):
                SessionConfig(model="live-1", input=[item])

    def test_a_function_tool_needs_a_name_and_every_tool_must_be_an_object(self):
        for tools in ([{"type": "function"}], [{"type": "function", "name": ""}], ["get_menu"]):
            with self.subTest(tools=tools), self.assertRaises(ValidationError):
                responses_config_with(tools)


def responses_config_with(tools):
    return SessionConfig(model="live-1", delegation={"type": "responses", "responses": {"model": "m", "tools": tools}})


class SlowStartEngine(FakeAdapter):
    """An engine whose start waits for the test, so the session can be closed while it is starting."""

    def __init__(self):
        super().__init__()
        self.release = asyncio.Event()

    async def start(self, session, delegation):
        await self.release.wait()
        await super().start(session, delegation)


class SlowCloseEngine(FakeAdapter):
    """An engine whose close waits for the test, so whoever asked to close can be cancelled mid-close."""

    def __init__(self):
        super().__init__()
        self.closing, self.release = asyncio.Event(), asyncio.Event()

    async def close(self):
        self.closing.set()
        await self.release.wait()
        await super().close()


class MeteredEngine(FakeAdapter):
    def context_tokens(self):
        return 64_000

    def context_window_tokens(self):
        return None


class FaultyEngine(FakeAdapter):
    async def append_thinking(self, text, client_event_id=None, delegation_id=None):
        raise RuntimeError("boom")


class NamingEngine(FakeAdapter):
    """An adapter that notes the client event and the delegation each append names."""

    def __init__(self):
        super().__init__()
        self.named: list[tuple] = []

    async def append_instructions(self, text, client_event_id=None, delegation_id=None):
        self.named.append(("instructions", client_event_id, delegation_id))
        await super().append_instructions(text, client_event_id, delegation_id)

    async def append_thinking(self, text, client_event_id=None, delegation_id=None):
        self.named.append(("thinking", client_event_id, delegation_id))
        await super().append_thinking(text, client_event_id, delegation_id)

    async def append_commentary(self, text, client_event_id=None, delegation_id=None):
        self.named.append(("commentary", client_event_id, delegation_id))
        await super().append_commentary(text, client_event_id, delegation_id)


class SessionTests(unittest.IsolatedAsyncioTestCase):
    async def start(self, config=None, engine=None, transport="websocket", **options):
        self.engine = engine or FakeAdapter()
        config = config or responses_config()
        codec = wire_codec(config) if transport == "websocket" else None  # only the WebSocket edge hands one in
        self.session = LiveProtocolSession(
            config, transport, self.engine.parts(), core=self.engine.core, codec=codec, **options
        )
        self.primary, self.sideband = Collector(), Collector()
        self.connections = {"primary": self.session.subscribe(self.primary.send, "primary")}
        await self.session.start("start_1")
        await self.primary.wait_for("session.started")
        self.connections["sideband"] = self.session.subscribe(self.sideband.send, "sideband")
        self.addAsyncCleanup(self.session.close)

    async def handle(self, message, source="primary"):
        """Run a command from the primary or the sideband connection, then wait until every writer delivered it."""
        await self.session.handle(message, self.connections[source])
        for sub in tuple(self.session.fanout.subscribers.values()):
            await sub.queue.join()

    async def test_start_announces_an_active_session_and_echoes_the_start_event(self):
        await self.start()
        started = await self.primary.wait_for("session.started")
        self.assertEqual(started["client_event_id"], "start_1")
        self.assertEqual(started["session"]["status"], "active")
        self.assertTrue(started["session"]["id"].startswith("live_"))
        self.assertTrue(self.engine.started)

    async def test_a_fault_in_the_engine_is_reported_and_the_session_keeps_serving(self):
        await self.start(engine=FaultyEngine())
        await self.handle(
            {"type": "session.thinking.append", "event_id": "t1", "delegation_id": None, "content": "note"}
        )
        error = await self.primary.wait_for("error")
        self.assertEqual((error["error"]["code"], error["error"]["client_event_id"]), ("server_error", "t1"))
        self.assertNotIn("boom", str(error))  # the cause stays in the log, not on the wire
        self.assertEqual(self.session.status, "active")
        await self.handle({"type": "session.input_audio.mute", "event_id": "m"})
        await self.primary.wait_for("session.input_audio.muted")

    async def test_a_session_closed_while_its_engine_starts_stays_closed(self):
        engine = SlowStartEngine()
        session = LiveProtocolSession(responses_config(), "websocket", engine.parts(), core=engine.core)
        collector = Collector()
        session.subscribe(collector.send, "primary")
        starting = asyncio.create_task(session.start("start_1"))
        await asyncio.sleep(0.05)
        await session.close("remote_hangup")
        engine.release.set()
        await starting
        self.assertEqual(session.status, "closed")
        self.assertNotIn("session.started", collector.kinds())
        self.assertTrue(engine.closed)

    async def test_the_context_meter_uses_the_engines_own_count_when_it_has_one(self):
        await self.start(engine=MeteredEngine())
        self.assertEqual(lifecycle.context_ratio(self.session), 0.5)

    async def test_the_context_meter_divides_by_the_adapters_own_window_when_it_reports_one(self):
        engine = MeteredEngine()
        engine.context_window_tokens = lambda: 256_000  # the ContextMeter port's window, read by the gateway
        await self.start(engine=engine)
        self.assertEqual(lifecycle.context_ratio(self.session), 0.25)

    async def test_the_context_meter_falls_back_to_the_size_of_the_configuration(self):
        await self.start()
        self.assertGreater(lifecycle.context_ratio(self.session), 0)
        self.assertLess(lifecycle.context_ratio(self.session), 0.05)

    async def test_a_session_starts_only_once(self):
        await self.start()
        with self.assertRaises(ProtocolError) as error:
            await self.session.start()
        self.assertEqual(error.exception.code, "session_already_started")

    async def test_a_failed_pipeline_start_reports_an_error_and_closes(self):
        self.engine = FakeAdapter(fail_start=True)
        self.session = LiveProtocolSession(responses_config(), "websocket", self.engine.parts(), core=self.engine.core)
        self.primary = Collector()
        self.session.subscribe(self.primary.send, "primary")
        with self.assertRaises(RuntimeError):
            await self.session.start()
        self.assertEqual(self.session.status, "closed")
        self.assertIn("error", self.primary.kinds())
        self.assertEqual((await self.primary.wait_for("session.closed"))["reason"], "connection_lost")

    async def test_mute_and_unmute_acknowledge_with_the_client_event(self):
        await self.start()
        await self.handle({"type": "session.input_audio.mute", "event_id": "m1"})
        self.assertTrue(self.engine.muted)
        self.assertEqual((await self.primary.wait_for("session.input_audio.muted"))["client_event_id"], "m1")
        await self.handle({"type": "session.input_audio.unmute", "event_id": "m2"})
        self.assertFalse(self.engine.muted)
        await self.primary.wait_for("session.input_audio.unmuted")

    async def test_update_applies_sparse_settings_and_reports_unserved_models_afterwards(self):
        await self.start()
        patch = {"delegation": {"responses": {"max_output_tokens": 64}}}
        await self.handle({"type": "session.update", "event_id": "u1", "session": patch})
        updated = await self.primary.wait_for("session.updated")
        self.assertEqual(updated["session"]["delegation"]["responses"]["max_output_tokens"], 64)
        self.assertEqual(self.session.config.delegation.responses.max_output_tokens, 64)
        bad = {"delegation": {"responses": {"model": "bad-model"}}}
        await self.handle({"type": "session.update", "event_id": "u2", "session": bad})
        error = (await self.primary.wait_for("error"))["error"]
        self.assertEqual(
            (error["code"], error["param"], error["client_event_id"]),
            ("responses_validation_failed", "session.delegation.responses.model", "u2"),
        )
        self.assertEqual(self.session.config.delegation.responses.model, "bad-model")  # kept
        self.assertEqual(self.session.status, "active")

    async def test_client_mode_sessions_cannot_be_updated(self):
        await self.start(config=SessionConfig(model="live-1"))
        await self.handle({"type": "session.update", "session": {"delegation": {"responses": {}}}})
        self.assertIn("error", self.primary.kinds())

    async def test_appends_reach_the_engine_and_acknowledge_with_a_zero_length_span_on_the_input_clock(self):
        # Appended context takes no audio time; its acknowledgment comes once the adapter took it, with
        # end_ms equal to start_ms at the input clock, in the order of the appends.
        await self.start()
        await self.session.advance_input_clock(150)
        for kind, label in (
            ("session.instructions.append", "instructions"),
            ("session.thinking.append", "thinking"),
            ("session.commentary.append", "commentary"),
        ):
            await self.handle({"type": kind, "event_id": label, "delegation_id": None, "content": "hello"})
        self.assertEqual([item[0] for item in self.engine.appended], ["instructions", "thinking", "commentary"])
        acks = [e for e in self.primary.events if e["type"].endswith(".appended")]
        self.assertEqual(
            [(a["type"], a["client_event_id"], a["start_ms"], a["end_ms"]) for a in acks],
            [
                ("session.instructions.appended", "instructions", 150, 150),
                ("session.thinking.appended", "thinking", 150, 150),
                ("session.commentary.appended", "commentary", 150, 150),
            ],
        )

    async def test_append_validation(self):
        await self.start()
        cases = (
            ({"content": "x"}, "delegation_id"),
            ({"delegation_id": None, "content": "  "}, "content"),
            ({"delegation_id": None, "content": "x" * 3000}, "content"),
            ({"delegation_id": "item_known", "content": "x"}, "delegation_id"),  # responses mode: must be null
        )
        for fields, param in cases:
            before = len(self.primary.events)
            await self.handle({"type": "session.commentary.append", "event_id": "e", **fields})
            await asyncio.sleep(0.01)
            new = self.primary.events[before:]
            self.assertEqual([e["error"]["param"] for e in new], [param], fields)
        self.assertEqual(self.engine.appended, [])

    async def test_client_mode_appends_may_name_an_issued_delegation_only(self):
        await self.start(config=SessionConfig(model="live-1"))
        issued = await self.session.delegation.delegate([])  # the session's client delegation announces it
        await self.handle({"type": "session.commentary.append", "delegation_id": issued, "content": "ok"})
        self.assertEqual(self.engine.appended, [("commentary", "ok", None, issued)])
        await self.handle({"type": "session.commentary.append", "delegation_id": "invented", "content": "ok"})
        self.assertEqual(self.primary.events[-1]["error"]["param"], "delegation_id")

    async def test_a_sideband_may_answer_a_client_delegation_that_the_data_channel_may_not(self):
        # The trusted app server answers over the sideband; the browser's data channel keeps its own permissions.
        config = SessionConfig(
            model="live-1",
            delegation={"type": "client"},
            client={"data_channel": {"allowed_client_events": ["session.input_audio.mute"]}},
        )
        await self.start(config=config, transport="webrtc")
        issued = await self.session.delegation.delegate([])
        answer = {"type": "session.commentary.append", "event_id": "c1", "delegation_id": issued, "content": "Done."}
        await self.handle(answer)
        self.assertEqual(self.primary.events[-1]["error"]["code"], "permission_denied")
        await self.handle(answer, source="sideband")
        self.assertEqual(self.engine.appended, [("commentary", "Done.", "c1", issued)])

    async def test_each_append_may_name_an_issued_client_delegation_and_passes_it_on(self):
        # Every append carries its delegation_id to the frontend adapter, not only commentary.
        await self.start(config=SessionConfig(model="live-1", delegation={"type": "client"}), engine=NamingEngine())
        issued = await self.session.delegation.delegate([])
        for kind in ("session.instructions.append", "session.thinking.append", "session.commentary.append"):
            await self.handle({"type": kind, "event_id": kind, "delegation_id": issued, "content": "ok"})
        self.assertNotIn("error", self.primary.kinds())
        self.assertEqual(
            self.engine.named,
            [
                ("instructions", "session.instructions.append", issued),
                ("thinking", "session.thinking.append", issued),
                ("commentary", "session.commentary.append", issued),
            ],
        )
        # An id this session never announced, an empty one, or one of another type is refused; the ID check comes
        # before the content check.
        for delegation_id in ("item_1", "", 5, {"id": issued}):
            with self.subTest(delegation_id=delegation_id):
                await self.handle(
                    {"type": "session.thinking.append", "event_id": "a1", "delegation_id": delegation_id, "content": ""}
                )
                error = self.primary.events[-1]["error"]
                self.assertEqual(
                    (error["code"], error["param"], error["client_event_id"]), ("invalid_value", "delegation_id", "a1")
                )
        self.assertEqual(len(self.engine.appended), 3)

    async def test_an_append_past_the_limit_per_minute_is_refused_and_neither_delivered_nor_acknowledged(self):
        await self.start(max_appends_per_minute=2)
        note = {"type": "session.thinking.append", "delegation_id": None}
        await self.handle({**note, "event_id": "blank", "content": " "})  # refused by its checks: not counted
        for label in ("n1", "n2", "n3"):
            await self.handle({**note, "event_id": label, "content": label})
        await self.handle({**note, "event_id": "blank2", "content": " "})  # past the limit: its own check first
        self.assertEqual(self.engine.appended, [("thinking", "n1"), ("thinking", "n2")])
        acks = [e["client_event_id"] for e in self.primary.events if e["type"] == "session.thinking.appended"]
        self.assertEqual(acks, ["n1", "n2"])
        errors = [e["error"] for e in self.primary.events if e["type"] == "error"]
        self.assertEqual(
            [(e["code"], e["client_event_id"]) for e in errors],
            [("invalid_value", "blank"), ("rate_limit_exceeded", "n3"), ("invalid_value", "blank2")],
        )
        self.assertEqual(errors[1]["message"], "Too many appends: at most 2 per minute")

    async def test_appends_are_accepted_again_once_the_minute_has_passed(self):
        # The refused append is not counted: once the first two leave the window, two more fit.
        await self.start(max_appends_per_minute=2)
        note = {"type": "session.thinking.append", "delegation_id": None, "content": "note"}
        with mock.patch("live.session.appends.time") as clock:  # the appends module's clock only, not the event loop's
            for at, label in ((0, "a"), (0, "b"), (59, "c"), (61, "d"), (61, "e"), (61, "f")):
                clock.monotonic.return_value = 1000.0 + at
                await self.handle({**note, "event_id": label})
        acks = [e["client_event_id"] for e in self.primary.events if e["type"] == "session.thinking.appended"]
        self.assertEqual(acks, ["a", "b", "d", "e"])
        errors = [e["error"] for e in self.primary.events if e["type"] == "error"]
        self.assertEqual(
            [(e["code"], e["client_event_id"]) for e in errors],
            [("rate_limit_exceeded", "c"), ("rate_limit_exceeded", "f")],
        )
        self.assertEqual(len(self.engine.appended), 4)

    async def test_the_minute_slides_with_each_append(self):
        # "b" at 50 s is still within the minute at 62 s, so "d" is refused although "a" has left it.
        await self.start(max_appends_per_minute=2)
        note = {"type": "session.thinking.append", "delegation_id": None, "content": "note"}
        with mock.patch("live.session.appends.time") as clock:
            for at, label in ((0, "a"), (50, "b"), (61, "c"), (62, "d")):
                clock.monotonic.return_value = 1000.0 + at
                await self.handle({**note, "event_id": label})
        acks = [e["client_event_id"] for e in self.primary.events if e["type"] == "session.thinking.appended"]
        self.assertEqual(acks, ["a", "b", "c"])
        errors = [e["error"] for e in self.primary.events if e["type"] == "error"]
        self.assertEqual([(e["code"], e["client_event_id"]) for e in errors], [("rate_limit_exceeded", "d")])

    async def test_the_three_kinds_of_append_count_against_one_limit(self):
        await self.start(max_appends_per_minute=3)
        kinds = ("session.instructions.append", "session.thinking.append", "session.commentary.append")
        for kind in kinds * 2:
            await self.handle({"type": kind, "event_id": kind, "delegation_id": None, "content": "ok"})
        self.assertEqual([item[0] for item in self.engine.appended], ["instructions", "thinking", "commentary"])
        errors = [e["error"] for e in self.primary.events if e["type"] == "error"]
        self.assertEqual(
            [(e["code"], e["client_event_id"]) for e in errors], [("rate_limit_exceeded", kind) for kind in kinds]
        )

    async def test_response_commands_require_responses_mode_and_pass_through(self):
        await self.start()
        item = {"type": "function_call_output", "call_id": "c1", "output": "{}"}
        # The session's own server delegation takes a result only for a call it relayed: c1 was relayed.
        call = {"type": "function_call", "call_id": "c1", "name": "get_menu", "arguments": "{}"}
        self.session.delegation.routes.note_call("item_1", "resp_1", call)
        await self.handle({"type": "response.item.create", "item": item})
        await self.handle({"type": "response.create", "event_id": "r1"})
        async with asyncio.timeout(2):  # queued under the lock, then sent by the part's sender
            while not self.engine.creates:
                await asyncio.sleep(0.005)
        self.assertEqual((self.engine.items, self.engine.creates), ([item], 1))
        await self.handle({"type": "response.create", "event_id": "r2", "extra": True})
        self.assertIn("accepts only type and event_id", self.primary.events[-1]["error"]["message"])
        await self.handle({"type": "response.item.create", "item": "nope"})
        self.assertEqual(self.primary.events[-1]["error"]["param"], "item")

    async def test_an_engine_rejection_becomes_an_error_event_with_the_client_event_id(self):
        await self.start()  # the session's own server delegation refuses a result for a call it never relayed
        await self.handle({"type": "response.item.create", "event_id": "i1", "item": {"type": "function_call_output"}})
        error = self.primary.events[-1]["error"]
        self.assertEqual((error["param"], error["client_event_id"]), ("item.call_id", "i1"))

    async def test_a_client_mode_session_starts_and_builds_no_core(self):
        engines = []

        async def factory(config, transport, connection):
            engines.append(FakeAdapter())
            return engines[-1].parts()

        def core():
            raise AssertionError("a client-mode session never asks for the delegation core")

        gateway = LiveGateway(factory, LiveSettings(), core=core)
        ws = FakeWebSocket()
        serving = asyncio.create_task(run_primary_websocket(gateway, ws))
        ws.send(
            {
                "type": "session.start",
                "event_id": "s1",
                "session": {"model": "live-1", "delegation": {"type": "client"}},
            }
        )
        started = await ws.wait_for("session.started")
        self.assertEqual(started["session"]["delegation"], {"type": "client", "acknowledgement_delay_ms": 0})
        session = next(iter(gateway.sessions.values()))
        self.assertEqual((session.delegation.target, session.delegation.conversation), ("client", None))
        delegation_id = await engines[0].delegation.delegate([{"role": "user", "content": "dropped"}])
        created = await ws.wait_for("session.delegation.created")
        self.assertEqual(created["delegation"], {"id": delegation_id, "type": "delegation", "target": "client"})
        self.assertIn("offset_ms", created)
        ws.send({"type": "session.close", "event_id": "bye"})
        async with asyncio.timeout(5):
            await serving
        self.assertEqual((engines[0].creates, gateway.sessions), (0, {}))

    async def test_response_commands_are_refused_in_client_mode(self):
        for config in (SessionConfig(model="live-1"), SessionConfig(model="live-1", delegation={"type": "client"})):
            with self.subTest(config=config.delegation):
                await self.start(config=config)
                item = {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "Two lattes."}]}
                await self.handle({"type": "response.item.create", "event_id": "i1", "item": item})
                await self.handle({"type": "response.create", "event_id": "r1"})
                errors = [e["error"] for e in self.primary.events if e["type"] == "error"]
                self.assertEqual(
                    [(e["code"], e["client_event_id"], e["message"]) for e in errors],
                    [
                        ("invalid_value", "i1", "This command requires Responses delegation"),
                        ("invalid_value", "r1", "This command requires Responses delegation"),
                    ],
                )
                self.assertEqual((self.engine.items, self.engine.creates), ([], 0))

    async def test_audio_append_is_a_primary_websocket_command_and_is_validated(self):
        await self.start()
        good = base64.b64encode(b"\x01\x00" * 480).decode()
        await self.handle({"type": "session.input_audio.append", "audio": good})
        self.assertEqual(len(self.engine.audio), 1)
        for bad in ("!!!not base64", base64.b64encode(b"\x01").decode(), 5):
            before = len(self.primary.events)
            await self.handle({"type": "session.input_audio.append", "audio": bad})
            self.assertEqual(self.primary.events[before]["error"]["param"], "audio")
        await self.handle({"type": "session.input_audio.append", "audio": good}, source="sideband")
        self.assertIn("only allowed on a primary WebSocket", self.sideband.events[-1]["error"]["message"])
        await self.start(transport="webrtc")
        await self.handle({"type": "session.input_audio.append", "audio": good})
        self.assertIn("only allowed on a primary WebSocket", self.primary.events[-1]["error"]["message"])

    async def test_unknown_malformed_and_repeated_start_events_are_errors(self):
        await self.start()
        for message, fragment in (
            ({"type": "session.unknown"}, "Unknown client event"),
            ({"nope": 1}, "Expected a JSON event object"),
            ("text", "Expected a JSON event object"),
            ({"type": "session.start"}, "already started"),
        ):
            before = len(self.primary.events)
            await self.handle(message)
            self.assertIn(fragment, self.primary.events[before]["error"]["message"], message)

    async def test_commands_after_close_are_refused(self):
        await self.start()
        await self.session.close()
        collected = len(self.primary.events)
        await self.handle({"type": "session.input_audio.mute"})
        self.assertEqual(len(self.primary.events), collected)  # the writer is gone; nothing is delivered
        self.assertEqual(self.session.status, "closed")

    async def test_close_emits_the_final_event_once_and_stops_the_engine(self):
        await self.start()
        await self.handle({"type": "session.close", "event_id": "bye"})
        closed = await self.primary.wait_for("session.closed")
        self.assertEqual((closed["reason"], closed["client_event_id"]), ("close_requested", "bye"))
        self.assertEqual(closed["session"]["status"], "active")  # live-session reports the final snapshot as active
        self.assertIn("seconds", closed["usage"])
        self.assertTrue(self.engine.closed)
        self.assertEqual(self.primary.kinds().count("session.closed"), 1)
        await self.session.close("expired")  # idempotent
        self.assertEqual(self.primary.kinds().count("session.closed"), 1)

    async def test_an_append_is_acknowledged_before_a_close_that_follows_it(self):
        # Nothing is held for the clock, so no append's acknowledgment is lost to the close.
        await self.start()
        await self.handle(
            {"type": "session.thinking.append", "event_id": "late", "delegation_id": None, "content": "x"}
        )
        await self.session.close()
        self.assertEqual(self.primary.kinds()[-2:], ["session.thinking.appended", "session.closed"])
        self.assertNotIn("error", self.primary.kinds())

    async def test_sideband_observers_see_events_after_attaching_and_are_limited(self):
        await self.start()  # with one sideband attached
        observers = []
        for _ in range(7):
            collector = Collector()
            self.session.subscribe(collector.send, "sideband")
            observers.append(collector)
        with self.assertRaises(ProtocolError):
            self.session.subscribe(Collector().send, "sideband")
        await self.session.emit("session.usage.updated", usage={"seconds": 1})
        await asyncio.sleep(0.02)
        self.assertTrue(all(o.kinds() == ["session.usage.updated"] for o in observers))

    async def test_data_channel_permissions_filter_events_in_both_directions(self):
        config = SessionConfig(
            model="live-1",
            client={
                "data_channel": {
                    "allowed_client_events": ["session.input_audio.mute"],
                    "allowed_server_events": ["session.started", {"type": "response.event", "response_event": "x.y"}],
                }
            },
        )
        await self.start(config=config, transport="webrtc")
        sideband = Collector()
        self.session.subscribe(sideband.send, "sideband")
        await self.session.emit("session.usage.updated", usage={})
        await self.session.emit("response.event", delegation_id="d", event={"type": "x.y"})
        await self.session.emit("response.event", delegation_id="d", event={"type": "other"})
        await asyncio.sleep(0.02)
        self.assertEqual(self.primary.kinds(), ["session.started", "response.event"])
        self.assertEqual(sideband.kinds(), ["session.usage.updated", "response.event", "response.event"])
        await self.handle({"type": "session.thinking.append", "delegation_id": None, "content": "x"})
        await asyncio.sleep(0.02)
        self.assertEqual(self.engine.appended, [])  # a forbidden client event never reaches the engine

    async def test_a_slow_primary_connection_closes_the_session(self):
        self.engine = FakeAdapter()
        self.session = LiveProtocolSession(responses_config(), "websocket", self.engine.parts(), core=self.engine.core)
        gate = asyncio.Event()

        async def stuck(message):
            await gate.wait()

        self.session.subscribe(stuck, "primary")
        await self.session.start()
        for _ in range(300):
            await self.session.emit("session.usage.updated", usage={})
        async with asyncio.timeout(5):
            await self.session.closed.wait()
        gate.set()
        self.assertTrue(self.engine.closed)


class InterruptedCloseTests(unittest.IsolatedAsyncioTestCase):
    """Whoever asked to close may be cancelled mid-close (a client that leaves); the close still finishes in order."""

    async def start(self):
        self.engine = SlowCloseEngine()
        self.left_registry = asyncio.Event()

        async def on_closed(session):
            self.left_registry.set()

        self.session = LiveProtocolSession(
            responses_config(), "websocket", self.engine.parts(), core=self.engine.core, on_closed=on_closed
        )
        self.primary = Collector()
        self.session.subscribe(self.primary.send, "primary")
        await self.session.start("start_1")
        await self.primary.wait_for("session.started")

    async def assert_closed_in_full(self):
        self.engine.release.set()
        async with asyncio.timeout(2):
            await self.left_registry.wait()
        self.assertEqual(self.session.status, "closed")
        self.assertTrue(self.engine.closed)
        self.assertEqual(self.primary.kinds().count("session.closed"), 1)

    async def test_a_close_whose_anyio_scope_is_cancelled_still_finishes(self):
        await self.start()
        scope = anyio.CancelScope()

        async def caller():
            with scope:
                await self.session.close("connection_lost")

        task = asyncio.create_task(caller())
        await self.engine.closing.wait()
        scope.cancel()
        await task  # the scope absorbs its own cancel; nothing else escapes
        await self.assert_closed_in_full()

    async def test_a_close_whose_caller_task_is_cancelled_still_finishes(self):
        await self.start()
        task = asyncio.create_task(self.session.close("connection_lost"))
        await self.engine.closing.wait()
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        await self.assert_closed_in_full()

    async def test_a_second_caller_waits_for_the_first_close(self):
        await self.start()
        first = asyncio.create_task(self.session.close("remote_hangup"))
        await self.engine.closing.wait()
        second = asyncio.create_task(self.session.close("expired"))
        await asyncio.sleep(0.02)
        self.assertFalse(second.done())
        self.engine.release.set()
        await asyncio.gather(first, second)
        self.assertEqual([e["reason"] for e in self.primary.events if e["type"] == "session.closed"], ["remote_hangup"])

    async def test_the_final_event_is_delivered_before_the_session_reports_closed(self):
        # The WebSocket loops close their socket once `session.closed` is set, so the final event must be out first.
        await self.start()
        seen_when_closed = []

        async def watch():
            await self.session.closed.wait()
            seen_when_closed.extend(self.primary.kinds())

        watcher = asyncio.create_task(watch())
        self.engine.release.set()
        await self.session.close("close_requested")
        await watcher
        self.assertIn("session.closed", seen_when_closed)


if __name__ == "__main__":
    unittest.main()
