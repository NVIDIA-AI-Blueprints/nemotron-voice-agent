# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D100, D101, D102, D103

import asyncio
import base64
import os
import unittest
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from live.gateway import LiveSettings
from live.gateway.webrtc import expire_unconnected
from live.mount import mount_live
from live.protocol import SessionConfig
from tests.unit.live_helpers import FakeAdapter

START = {
    "type": "session.start",
    "event_id": "start_1",
    "session": {"model": "live-1", "instructions": "Be brief."},
}


def make_app(settings=None):
    engines: list[FakeAdapter] = []

    async def factory(config, transport, connection):
        engine = FakeAdapter()
        engines.append(engine)
        return engine.parts()

    app = FastAPI()
    core = FakeAdapter()  # the gateway's one delegation core
    gateway = mount_live(app, factory=factory, core=core.core, settings=settings)
    return app, gateway, engines


class PrimaryWebSocketTests(unittest.TestCase):
    def test_start_commands_audio_and_close(self):
        app, gateway, engines = make_app()
        with TestClient(app) as client, client.websocket_connect("/v1/live/sessions") as ws:
            ws.send_json(START)
            started = ws.receive_json()
            self.assertEqual((started["type"], started["client_event_id"]), ("session.started", "start_1"))
            self.assertEqual(started["session"]["instructions"], "Be brief.")
            ws.send_json({"type": "session.input_audio.append", "audio": base64.b64encode(b"\x01\x00" * 480).decode()})
            ws.send_json({"type": "session.input_audio.mute", "event_id": "m"})
            self.assertEqual(ws.receive_json()["type"], "session.input_audio.muted")
            self.assertEqual(len(engines[0].audio[0]), 960)
            ws.send_json({"type": "session.close", "event_id": "bye"})
            closed = ws.receive_json()
            self.assertEqual((closed["type"], closed["reason"]), ("session.closed", "close_requested"))
        self.assertTrue(engines[0].closed)
        self.assertEqual(gateway.sessions, {})

    def test_the_first_message_must_be_session_start(self):
        app, _, engines = make_app()
        with TestClient(app) as client, client.websocket_connect("/v1/live/sessions") as ws:
            ws.send_json({"type": "session.input_audio.mute"})
            self.assertIn("session.start must be the first message", ws.receive_json()["error"]["message"])
            with self.assertRaises(WebSocketDisconnect):
                ws.receive_json()
        self.assertEqual(engines, [])

    def test_an_invalid_session_is_reported_with_the_start_event_id(self):
        app, _, _ = make_app()
        with TestClient(app) as client, client.websocket_connect("/v1/live/sessions") as ws:
            ws.send_json({"type": "session.start", "event_id": "s9", "session": {"model": ""}})
            error = ws.receive_json()["error"]
            self.assertEqual(error["client_event_id"], "s9")

    def test_websocket_sessions_reject_webrtc_only_settings(self):
        app, _, _ = make_app()
        with TestClient(app) as client, client.websocket_connect("/v1/live/sessions") as ws:
            ws.send_json({**START, "session": {"model": "live-1", "client": {"data_channel": {}}}})
            self.assertEqual(ws.receive_json()["error"]["param"], "session.client")

    def test_invalid_json_is_an_error_event_not_a_disconnect(self):
        app, _, _ = make_app()
        with TestClient(app) as client, client.websocket_connect("/v1/live/sessions") as ws:
            ws.send_json(START)
            ws.receive_json()
            ws.send_text("not json")
            self.assertEqual(ws.receive_json()["error"]["message"], "Invalid JSON event")
            ws.send_json({"type": "session.input_audio.mute"})
            self.assertEqual(ws.receive_json()["type"], "session.input_audio.muted")

    def test_the_session_limit_is_enforced_with_an_error_event_before_the_close(self):
        # A full server answers the start with rate_limit_exceeded, then closes 1008.
        app, _, _ = make_app(LiveSettings(max_sessions=1))
        with TestClient(app) as client, client.websocket_connect("/v1/live/sessions") as first:
            first.send_json(START)
            first.receive_json()
            with client.websocket_connect("/v1/live/sessions") as second:
                second.send_json(START)
                error = second.receive_json()
                self.assertEqual(
                    (error["type"], error["error"]["code"], error["client_event_id"]),
                    ("error", "rate_limit_exceeded", "start_1"),
                )
                with self.assertRaises(WebSocketDisconnect) as closed:
                    second.receive_json()
                self.assertEqual(closed.exception.code, 1008)


class SecurityTests(unittest.TestCase):
    def test_a_configured_token_is_required(self):
        app, _, _ = make_app(LiveSettings(api_token="s3cret"))
        with TestClient(app) as client:
            with self.assertRaises(WebSocketDisconnect), client.websocket_connect("/v1/live/sessions") as ws:
                ws.send_json(START)
                ws.receive_json()
            headers = {"Authorization": "Bearer s3cret"}
            with client.websocket_connect("/v1/live/sessions", headers=headers) as ws:
                ws.send_json(START)
                self.assertEqual(ws.receive_json()["type"], "session.started")
            self.assertEqual(client.post("/v1/live/sessions", json={}).status_code, 401)

    def test_cross_site_browser_origins_are_refused(self):
        app, _, _ = make_app()
        with TestClient(app) as client:
            response = client.post("/v1/live/sessions", json={}, headers={"Origin": "https://evil.example"})
            self.assertEqual(response.status_code, 403)
            evil = {"Origin": "https://evil.example"}
            with (
                self.assertRaises(WebSocketDisconnect),
                client.websocket_connect("/v1/live/sessions", headers=evil) as ws,
            ):
                ws.send_json(START)
                ws.receive_json()


class HttpTests(unittest.TestCase):
    def test_http_creation_needs_a_webrtc_offer(self):
        app, _, engines = make_app()
        with TestClient(app) as client:
            cases = (
                ({}, 400),
                ({"transport": {"type": "websocket"}, "session": {"model": "live-1"}}, 400),
                ({"transport": {"type": "webrtc"}, "session": {"model": "live-1"}}, 400),  # no SDP
                ({"transport": {"type": "webrtc", "sdp": "x"}, "session": {"model": ""}}, 400),
                (
                    {
                        "transport": {"type": "webrtc", "sdp": "x"},
                        "session": {"model": "live-1", "audio": {"format": {}}},
                    },
                    400,
                ),
            )
            for body, status in cases:
                with self.subTest(body=body):
                    self.assertEqual(client.post("/v1/live/sessions", json=body).status_code, status)
            self.assertEqual(client.post("/v1/live/sessions", content=b"[1]").status_code, 400)
            self.assertEqual(client.post("/v1/live/sessions", content=b"x" * (2**20 + 1)).status_code, 413)
        self.assertEqual(engines, [])


class SidebandTests(unittest.TestCase):
    def test_an_observer_sees_later_events_and_can_send_trusted_commands(self):
        app, gateway, engines = make_app()
        with TestClient(app) as client, client.websocket_connect("/v1/live/sessions") as primary:
            primary.send_json(START)
            session_id = primary.receive_json()["session"]["id"]
            with client.websocket_connect(f"/v1/live/sessions/{session_id}/attach") as observer:
                observer.send_json(
                    {"type": "session.thinking.append", "event_id": "t", "delegation_id": None, "content": "note"}
                )
                self.assertEqual(primary.receive_json()["type"], "session.thinking.appended")
                self.assertEqual(observer.receive_json()["type"], "session.thinking.appended")
                primary.send_json({"type": "session.input_audio.mute", "event_id": "m"})
                self.assertEqual(primary.receive_json()["type"], "session.input_audio.muted")
                self.assertEqual(observer.receive_json()["type"], "session.input_audio.muted")
                self.assertEqual(engines[0].appended, [("thinking", "note")])

    def test_an_observer_that_sends_bad_json_gets_an_error_and_stays_attached(self):
        app, gateway, engines = make_app()
        with TestClient(app) as client, client.websocket_connect("/v1/live/sessions") as primary:
            primary.send_json(START)
            session_id = primary.receive_json()["session"]["id"]
            with client.websocket_connect(f"/v1/live/sessions/{session_id}/attach") as observer:
                observer.send_text("this is not json")
                self.assertEqual(observer.receive_json()["error"]["message"], "Invalid JSON event")
                observer.send_json({"type": "session.input_audio.mute", "event_id": "m"})
                self.assertEqual(observer.receive_json()["type"], "session.input_audio.muted")

    def test_an_unknown_session_is_refused(self):
        app, _, _ = make_app()
        with (
            TestClient(app) as client,
            self.assertRaises(WebSocketDisconnect),
            client.websocket_connect("/v1/live/sessions/live_unknown/attach") as ws,
        ):
            ws.receive_json()


class HistoryTests(unittest.TestCase):
    def test_history_is_served_for_any_running_session_with_the_adapters_half_when_it_gives_one(self):
        app, _, engines = make_app()
        with TestClient(app) as client, client.websocket_connect("/v1/live/sessions") as primary:
            primary.send_json(START)  # a client-mode session: the server runs no backend for it
            session_id = primary.receive_json()["session"]["id"]
            plain = client.get(f"/v1/live/sessions/{session_id}/history")

            async def history():
                return {"frontend": {"messages": []}}

            engines[0].history = history
            served = client.get(f"/v1/live/sessions/{session_id}/history")
            unknown = client.get("/v1/live/sessions/live_unknown/history")
        self.assertEqual((plain.status_code, plain.json()), (200, {"frontend": None, "backend": None}))
        self.assertEqual(served.json(), {"frontend": {"messages": []}, "backend": None})
        self.assertEqual(unknown.status_code, 404)

    def test_history_requires_the_bearer_token_when_one_is_configured(self):
        app, _, _ = make_app(LiveSettings(api_token="secret"))
        with TestClient(app) as client:
            self.assertEqual(client.get("/v1/live/sessions/live_x/history").status_code, 401)


class MountTests(unittest.TestCase):
    def test_server_shutdown_closes_open_sessions(self):
        app, gateway, engines = make_app()

        async def run():
            async with app.router.lifespan_context(app):
                session = await gateway.register(SessionConfig(model="live-1"), "websocket", None)
                await session.start()
                self.assertEqual(len(gateway.sessions), 1)
            return session

        session = asyncio.run(run())
        self.assertEqual(session.status, "closed")
        self.assertTrue(engines[0].closed)
        self.assertEqual(gateway.sessions, {})


async def FakeAdapterFactory(config, transport, connection):  # noqa: N802
    return FakeAdapter().parts()


class TokenTests(unittest.TestCase):
    def test_a_non_ascii_authorization_header_is_a_401_and_never_a_crash(self):
        app = FastAPI()
        with patch.dict(os.environ, {"LIVE_API_TOKEN": "secret"}):
            mount_live(app, factory=FakeAdapterFactory)
        with TestClient(app) as client:
            for header in ("Bearer é".encode("latin-1"), b"Bearer \xff\xfe"):
                self.assertEqual(client.get("/v1/live/info", headers={"Authorization": header}).status_code, 401)


class ExposureTests(unittest.TestCase):
    # The server decides whether Live is served, for which example (src/server.py); with no factory the mount
    # serves no live route.
    def test_with_no_factory_no_live_route_is_served_whatever_the_environment_says(self):
        app = FastAPI()
        with patch.dict(os.environ, {"LIVE_ENABLED": "true"}), TestClient(app) as client:
            self.assertIsNone(mount_live(app))
            self.assertEqual(client.get("/v1/live/info").status_code, 404)


class UnconnectedPeerTests(unittest.IsolatedAsyncioTestCase):
    async def run_deadline(self, connected, status="active"):
        closed = []

        class Session:
            async def close(self, reason, client_event_id=None, *, asked_by="the gateway"):
                closed.append(reason)

        session = Session()
        session.status = status

        class Connection:
            def is_connected(self):
                return connected

        await expire_unconnected(session, Connection(), timeout=0)
        return closed

    async def test_an_active_session_whose_peer_never_connected_is_closed(self):
        self.assertEqual(await self.run_deadline(connected=False), ["connection_lost"])

    async def test_a_connected_session_is_left_alone(self):
        self.assertEqual(await self.run_deadline(connected=True), [])

    async def test_a_session_still_starting_at_the_deadline_is_closed(self):
        self.assertEqual(await self.run_deadline(connected=True, status="starting"), ["connection_lost"])


class OpenByDefaultTests(unittest.TestCase):
    def test_a_server_without_a_token_warns_at_startup_and_one_with_a_token_does_not(self):
        with patch("live.mount.logger") as logger:
            with patch.dict(os.environ, {"LIVE_API_TOKEN": ""}):
                mount_live(FastAPI(), factory=FakeAdapterFactory)
            self.assertIn("LIVE_API_TOKEN is not set", logger.warning.call_args.args[0])
            logger.warning.reset_mock()
            with patch.dict(os.environ, {"LIVE_API_TOKEN": "secret"}):
                mount_live(FastAPI(), factory=FakeAdapterFactory)
            logger.warning.assert_not_called()


if __name__ == "__main__":
    unittest.main()
