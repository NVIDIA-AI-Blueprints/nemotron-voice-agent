# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D100, D101, D102, D103, D107

import asyncio
import base64
import unittest

from pydantic import ValidationError

from live.events import error_event, event
from live.media.audio import WireCodec
from live.protocol import AudioFormat, ProtocolError, SessionConfig
from live.session import LiveProtocolSession, lifecycle
from live.session.updates import apply_update
from tests.unit.live_helpers import Collector, FakeEngine, responses_config


class SchemaTests(unittest.TestCase):
    def test_default_mode_is_client(self):
        for delegation in (None, {"type": "client"}):
            self.assertEqual(SessionConfig(model="live-1", delegation=delegation).mode, "client")

    def test_immutable_settings_cannot_change(self):
        for patch in (
            {"instructions": "changed"},
            {"audio": {"output": {"voice": "cedar"}}},
            {"delegation": None},
            {"delegation": {"type": "client"}},
            {"delegation": {"responses": {"model": "org/x/y"}}, "store": True},
        ):
            with self.subTest(patch=patch), self.assertRaises(ProtocolError) as error:
                apply_update(responses_config(), patch)
            self.assertEqual(error.exception.code, "immutable_field_update")

    def test_a_sparse_nested_update_merges(self):
        original = responses_config()
        updated = apply_update(original, {"delegation": {"responses": {"reasoning": {"effort": "high"}}}})
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

    def test_responses_mode_needs_its_settings_and_only_function_tools(self):
        with self.assertRaises(ValidationError):
            SessionConfig(model="live-1", delegation={"type": "responses"})
        with self.assertRaises(ValidationError):
            SessionConfig(
                model="live-1",
                delegation={"type": "responses", "responses": {"model": "m", "tools": [{"type": "mcp"}]}},
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


class SlowStartEngine(FakeEngine):
    """An engine whose start waits for the test, so the session can be closed while it is starting."""

    def __init__(self):
        super().__init__()
        self.release = asyncio.Event()

    async def start(self, session):
        await self.release.wait()
        await super().start(session)


class MeteredEngine(FakeEngine):
    def context_tokens(self):
        return 64_000


class FaultyEngine(FakeEngine):
    async def append_thinking(self, text):
        raise RuntimeError("boom")


class SessionTests(unittest.IsolatedAsyncioTestCase):
    async def start(self, config=None, engine=None, transport="websocket"):
        self.engine = engine or FakeEngine()
        self.session = LiveProtocolSession(config or responses_config(), transport, self.engine)
        self.primary = Collector()
        self.session.subscribe(self.primary.send, "primary")
        await self.session.start("start_1")
        await self.primary.wait_for("session.started")
        self.addAsyncCleanup(self.session.close)

    async def handle(self, message, source="primary"):
        """Run a command, then wait until every subscriber's writer has delivered what it produced."""
        await self.session.handle(message, source)
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
        session = LiveProtocolSession(responses_config(), "websocket", engine)
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
        self.engine = FakeEngine(fail_start=True)
        self.session = LiveProtocolSession(responses_config(), "websocket", self.engine)
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

    async def test_update_applies_sparse_settings_and_rejects_unserved_models(self):
        await self.start()
        patch = {"delegation": {"responses": {"max_output_tokens": 64}}}
        await self.handle({"type": "session.update", "event_id": "u1", "session": patch})
        updated = await self.primary.wait_for("session.updated")
        self.assertEqual(updated["session"]["delegation"]["responses"]["max_output_tokens"], 64)
        self.assertEqual(self.engine.updates[-1].delegation.responses.max_output_tokens, 64)
        bad = {"delegation": {"responses": {"model": "bad-model"}}}
        await self.handle({"type": "session.update", "event_id": "u2", "session": bad})
        error = next(e for e in self.primary.events if e["type"] == "error")["error"]
        self.assertEqual(error["param"], "session.delegation.responses.model")
        self.assertEqual(self.session.config.delegation.responses.model, "org/org/model")

    async def test_client_mode_sessions_cannot_be_updated(self):
        await self.start(config=SessionConfig(model="live-1"))
        await self.handle({"type": "session.update", "session": {"delegation": {"responses": {}}}})
        self.assertIn("error", self.primary.kinds())

    async def test_appends_reach_the_engine_and_acknowledge_on_the_input_clock(self):
        await self.start()
        for kind, label in (
            ("session.instructions.append", "instructions"),
            ("session.thinking.append", "thinking"),
            ("session.commentary.append", "commentary"),
        ):
            await self.handle({"type": kind, "event_id": label, "delegation_id": None, "content": "hello"})
        self.assertEqual([item[0] for item in self.engine.appended], ["instructions", "thinking", "commentary"])
        # Acknowledgments wait for the caller's audio clock to pass their end time.
        self.assertEqual(self.primary.kinds(), ["session.started"])
        await self.session.advance_input_clock(150)
        self.assertEqual(self.primary.kinds(), ["session.started"])
        await self.session.advance_input_clock(400)
        await asyncio.sleep(0.02)
        acks = [e for e in self.primary.events if e["type"].endswith(".appended")]
        self.assertEqual(
            [(a["type"], a["client_event_id"], a["end_ms"] - a["start_ms"]) for a in acks],
            [
                ("session.instructions.appended", "instructions", 200),
                ("session.thinking.appended", "thinking", 200),
                ("session.commentary.appended", "commentary", 200),
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
        await self.handle({"type": "session.commentary.append", "delegation_id": "item_known", "content": "ok"})
        self.assertEqual(self.engine.appended, [("commentary", "ok", "item_known")])
        await self.handle({"type": "session.commentary.append", "delegation_id": "invented", "content": "ok"})
        self.assertEqual(self.primary.events[-1]["error"]["param"], "delegation_id")

    async def test_too_many_pending_appends_are_refused(self):
        await self.start()
        for i in range(64):
            await self.handle(
                {"type": "session.thinking.append", "event_id": str(i), "delegation_id": None, "content": "x"}
            )
        await self.handle(
            {"type": "session.thinking.append", "event_id": "extra", "delegation_id": None, "content": "x"}
        )
        self.assertEqual(self.primary.events[-1]["error"]["code"], "rate_limit_exceeded")

    async def test_response_commands_require_responses_mode_and_pass_through(self):
        await self.start()
        item = {"type": "function_call_output", "call_id": "c1", "output": "{}"}
        await self.handle({"type": "response.item.create", "item": item})
        await self.handle({"type": "response.create", "event_id": "r1"})
        self.assertEqual((self.engine.items, self.engine.creates), ([item], 1))
        await self.handle({"type": "response.create", "event_id": "r2", "extra": True})
        self.assertIn("accepts only type and event_id", self.primary.events[-1]["error"]["message"])
        await self.handle({"type": "response.item.create", "item": "nope"})
        self.assertEqual(self.primary.events[-1]["error"]["param"], "item")

    async def test_an_engine_rejection_becomes_an_error_event_with_the_client_event_id(self):
        await self.start()
        self.engine.item_error = ProtocolError("Unknown pending function call", param="item.call_id")
        await self.handle({"type": "response.item.create", "event_id": "i1", "item": {"type": "function_call_output"}})
        error = self.primary.events[-1]["error"]
        self.assertEqual((error["param"], error["client_event_id"]), ("item.call_id", "i1"))

    async def test_response_commands_are_refused_in_client_mode(self):
        await self.start(config=SessionConfig(model="live-1"))
        await self.handle({"type": "response.create"})
        self.assertIn("requires Responses delegation", self.primary.events[-1]["error"]["message"])

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
        self.assertIn("only allowed on a primary WebSocket", self.primary.events[-1]["error"]["message"])
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

    async def test_appends_pending_at_close_fail_with_session_closed(self):
        await self.start()
        await self.handle(
            {"type": "session.thinking.append", "event_id": "late", "delegation_id": None, "content": "x"}
        )
        await self.session.close()
        error = next(e for e in self.primary.events if e["type"] == "error")["error"]
        self.assertEqual((error["code"], error["client_event_id"]), ("session_closed", "late"))

    async def test_sideband_observers_see_events_after_attaching_and_are_limited(self):
        await self.start()
        observers = []
        for _ in range(8):
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
        self.engine = FakeEngine()
        self.session = LiveProtocolSession(responses_config(), "websocket", self.engine)
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


if __name__ == "__main__":
    unittest.main()
