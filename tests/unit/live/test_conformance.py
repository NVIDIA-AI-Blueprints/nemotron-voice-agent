# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D107

"""Protocol conformance: a client written from the official schema starts and runs."""

import asyncio
import unittest

from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import ValidationError

from live.events import error_event
from live.gateway import create_live_router
from live.gateway.websocket import wire_codec
from live.protocol import LIVE_VOICES, ProtocolError, SessionConfig, validation_error
from live.session import LiveProtocolSession
from tests.unit.live_helpers import Collector, FakeAdapter, responses_config

# Nine of the documented voices, which the session config accepts.
NEW_VOICES = ("brise", "flitz", "harema", "juni", "nira", "noeul", "nuri", "shida", "sillage")


def start_error(session: dict) -> ProtocolError:
    """The protocol error a session.start with ``session`` gets."""
    try:
        SessionConfig.model_validate(session)
    except ValidationError as refused:
        return validation_error(refused, "session")
    raise AssertionError("the session was accepted")


class SessionCase(unittest.IsolatedAsyncioTestCase):
    async def start(self, config, transport="websocket"):
        self.engine = FakeAdapter()
        codec = wire_codec(config) if transport == "websocket" else None
        self.session = LiveProtocolSession(config, transport, self.engine.parts(), core=self.engine.core, codec=codec)
        self.primary, self.sideband = Collector(), Collector()
        self.primary_key = self.session.subscribe(self.primary.send, "primary")
        await self.session.start("start_1")
        await self.primary.wait_for("session.started")
        self.sideband_key = self.session.subscribe(self.sideband.send, "sideband")
        self.addAsyncCleanup(self.session.close)

    async def handle(self, message, connection):
        await self.session.handle(message, connection)
        for subscriber in tuple(self.session.fanout.subscribers.values()):
            await subscriber.queue.join()


class DelegationClassTests(unittest.TestCase):
    def test_both_delegation_types_take_an_acknowledgement_delay_at_start(self):
        for delegation in (
            {"type": "client", "acknowledgement_delay_ms": 800},
            {"type": "responses", "responses": {"model": "m"}, "acknowledgement_delay_ms": 800},
        ):
            with self.subTest(delegation=delegation["type"]):
                config = SessionConfig(model="live-1", delegation=delegation)
                self.assertEqual(config.delegation.acknowledgement_delay_ms, 800)
        self.assertEqual(
            start_error({"model": "m", "delegation": {"acknowledgement_delay_ms": -1}}).code, "invalid_value"
        )


class VoiceTests(unittest.TestCase):
    def test_every_documented_voice_and_a_custom_id_are_accepted(self):
        self.assertEqual(len(LIVE_VOICES), 31)
        for voice in (*NEW_VOICES, "marin", {"id": "voice_custom"}):
            with self.subTest(voice=voice):
                SessionConfig(model="live-1", audio={"output": {"voice": voice}})
        error = start_error({"model": "m", "audio": {"output": {"voice": "no-such-voice"}}})
        self.assertEqual((error.code, error.param), ("invalid_value", "session.audio.output"))


class ErrorRoutingTests(SessionCase):
    async def test_an_error_goes_only_to_the_connection_that_sent_the_command(self):
        await self.start(responses_config())
        bad = {"type": "session.thinking.append", "event_id": "t1", "content": "no delegation_id"}
        await self.handle(bad, self.sideband_key)
        self.assertEqual([e["type"] for e in self.sideband.events], ["error"])
        self.assertNotIn("error", self.primary.kinds())
        await self.handle(bad, self.primary_key)
        self.assertEqual(self.primary.kinds().count("error"), 1)
        self.assertEqual(self.sideband.kinds().count("error"), 1)

    async def test_an_error_no_command_caused_goes_to_every_connection(self):
        await self.start(responses_config())
        await self.session.adapter_view.report_error("The speech service failed.")
        for subscriber in tuple(self.session.fanout.subscribers.values()):
            await subscriber.queue.join()
        self.assertEqual(self.primary.kinds()[-1], "error")
        self.assertEqual(self.sideband.kinds()[-1], "error")

    async def test_a_detached_connections_command_is_dropped(self):
        await self.start(responses_config())
        await self.session.unsubscribe(self.sideband_key)
        await self.handle({"type": "session.input_audio.mute", "event_id": "m"}, self.sideband_key)
        self.assertFalse(self.engine.muted)


class WireDetailTests(unittest.TestCase):
    def test_optional_fields_are_omitted_and_the_type_follows_the_code(self):
        bare = error_event(ProtocolError("nope"))
        self.assertEqual(set(bare), {"type", "event_id", "error"})
        self.assertEqual(bare["error"], {"type": "invalid_request_error", "code": "invalid_value", "message": "nope"})
        full = error_event(ProtocolError("nope", "unknown_parameter", "session.voice"), "c1")
        self.assertEqual((full["client_event_id"], full["error"]["param"]), ("c1", "session.voice"))
        for code in ("server_error", "pipeline_error"):
            self.assertEqual(error_event(ProtocolError("down", code))["error"]["type"], "server_error")

    def test_an_unknown_key_at_start_is_unknown_parameter_with_its_param(self):
        for session, param in (
            ({"model": "m", "voice": "marin"}, "session.voice"),
            ({"model": "m", "audio": {"output": {"speed": 2}}}, "session.audio.output.speed"),
            (
                {"model": "m", "delegation": {"type": "client", "responses": {"model": "x"}}},
                "session.delegation.responses",
            ),
        ):
            with self.subTest(session=session):
                error = start_error(session)
                self.assertEqual((error.code, error.param), ("unknown_parameter", param))
                self.assertEqual(str(error), f"Unknown parameter: '{param}'.")


class UpdateTests(SessionCase):
    async def test_an_empty_update_changes_nothing_and_is_acknowledged(self):
        await self.start(responses_config())
        before = self.session.config
        await self.handle({"type": "session.update", "event_id": "u1", "session": {}}, self.primary_key)
        updated = await self.primary.wait_for("session.updated")
        self.assertEqual((updated["client_event_id"], self.session.config), ("u1", before))


class StartedTests(SessionCase):
    async def test_session_started_shows_the_resolved_delegation_and_wire_format(self):
        config = SessionConfig(model="live-1").with_wire_format()  # what the WebSocket edge hands the session
        await self.start(config)
        started = await self.primary.wait_for("session.started")
        self.assertEqual(started["session"]["delegation"], {"type": "client", "acknowledgement_delay_ms": 0})
        self.assertEqual(started["session"]["audio"]["format"], {"type": "audio/pcm", "rate": 24000})
        self.assertNotIn("triggers", started["session"])

    async def test_a_restricted_data_channel_is_announced_with_an_info_event(self):
        permissions = {"allowed_client_events": ["session.close"], "allowed_server_events": "all"}
        await self.start(SessionConfig(model="live-1", client={"data_channel": permissions}), transport="webrtc")
        info = await self.primary.wait_for("info")
        self.assertEqual(info["code"], "data_channel_permissions")
        self.assertEqual(self.primary.kinds()[:2], ["session.started", "info"])

    async def test_an_unrestricted_session_gets_no_info_event(self):
        await self.start(responses_config())
        await asyncio.sleep(0.01)
        self.assertNotIn("info", self.primary.kinds())


class BackendSettingTests(unittest.TestCase):
    def test_every_documented_service_tier_and_null_are_accepted(self):
        for tier in ("auto", "default", "fast_tier_temp_pilot", "flex", "priority", "ultrafast", None):
            with self.subTest(tier=tier):
                responses_config_with(service_tier=tier)

    def test_text_takes_only_verbosity(self):
        self.assertEqual(responses_config_with(text={"verbosity": "low"}).delegation.responses.text.verbosity, "low")
        error = start_error(
            {
                "model": "m",
                "delegation": {"type": "responses", "responses": {"model": "x", "text": {"format": {}}}},
            }
        )
        self.assertEqual((error.code, error.param), ("unknown_parameter", "session.delegation.responses.text.format"))

    def test_nullable_settings_take_null(self):
        responses_config_with(parallel_tool_calls=None, reasoning=None, text=None, instructions=None)


def responses_config_with(**settings):
    """A responses session whose backend block sets ``settings``."""
    return SessionConfig(
        model="live-1", delegation={"type": "responses", "responses": {"model": "org/model", **settings}}
    )


class SelectorTests(unittest.TestCase):
    def test_a_response_event_selector_needs_its_nested_type_and_only_it_may_have_one(self):
        for selector in ({"type": "response.event"}, {"type": "session.started", "response_event": "x"}):
            with self.subTest(selector=selector):
                error = start_error({"model": "m", "client": {"data_channel": {"allowed_server_events": [selector]}}})
                self.assertEqual(error.code, "invalid_value")
                self.assertTrue(error.param.startswith("session.client.data_channel.allowed_server_events"))
        SessionConfig(
            model="m",
            client={"data_channel": {"allowed_server_events": [{"type": "response.event", "response_event": "a.b"}]}},
        )


class UnsupportedFeatureTests(SessionCase):
    async def test_each_official_feature_this_server_does_not_run_gets_one_clean_refusal(self):
        await self.start(responses_config())
        for kind in (
            "session.output_audio.fixture.play",
            "decision.create",
            "decision.item.create",
            "decision.item.delete",
        ):
            with self.subTest(kind=kind):
                await self.handle({"type": kind, "event_id": "x"}, self.primary_key)
                error = self.primary.events[-1]["error"]
                self.assertEqual((error["code"], error["param"]), ("invalid_value", "type"))
                self.assertIn(kind, error["message"])
                self.assertIn("not supported by this server", error["message"])

    def test_decisions_delegation_and_triggers_are_refused_at_start_by_name(self):
        error = start_error({"model": "m", "delegation": {"type": "decisions", "decisions": {}}})
        self.assertEqual((error.code, error.param), ("invalid_value", "session.delegation"))
        self.assertIn("'decisions'", str(error))
        error = start_error({"model": "m", "triggers": [{"type": "decisions"}]})
        self.assertEqual((error.code, error.param), ("invalid_value", "session.triggers"))
        self.assertIn("Decisions triggers are not supported", str(error))
        SessionConfig(model="m", triggers=[])  # no triggers: nothing to refuse


class HttpStartTests(unittest.TestCase):
    def test_a_webrtc_start_with_an_unknown_key_is_a_400_with_unknown_parameter(self):
        async def factory(config, transport, connection):
            raise AssertionError("never built")

        router, _ = create_live_router(factory)
        app = FastAPI()
        app.include_router(router)
        with TestClient(app) as client:
            body = {"transport": {"type": "webrtc", "sdp": "v=0\r\nm=audio"}, "session": {"model": "m", "voice": "x"}}
            response = client.post("/v1/live/sessions", json=body)
        self.assertEqual(response.status_code, 400)
        self.assertEqual(
            (response.json()["error"]["code"], response.json()["error"]["param"]),
            ("unknown_parameter", "session.voice"),
        )


if __name__ == "__main__":
    unittest.main()
