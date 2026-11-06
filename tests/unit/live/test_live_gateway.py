# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D103, D105, D107

"""The live routes as clients reach them: WebSockets, WebRTC, the HTTP routes, the delegate route and the mount.

``mount_live`` serves them on an app whose sessions run on fake frontend adapters and a fake delegation core.
"""

import asyncio
import base64
import json
import os
import unittest
from dataclasses import replace
from unittest.mock import patch

import httpx
from aiortc import RTCPeerConnection, RTCSessionDescription
from aiortc.mediastreams import AudioStreamTrack
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from loguru import logger
from starlette.websockets import WebSocketDisconnect

from examples.frontend_backend_live.adapter.connection import LiveWebRTCConnection
from gateways.live.gateway import LiveSettings
from gateways.live.gateway.webrtc import expire_unconnected
from gateways.live.gateway.websocket import run_primary_websocket
from gateways.live.mount import mount_live
from gateways.live.ports import SessionParts
from gateways.live.protocol import ProtocolError, SessionConfig
from tests.unit.live_helpers import Collector, FakeAdapter

SESSIONS = "/v1/live/sessions"
START = {"type": "session.start", "event_id": "start_1", "session": {"model": "live-1", "instructions": "Be brief."}}
NOTE = {"type": "session.thinking.append", "delegation_id": None, "content": "note"}
MUTE = {"type": "session.input_audio.mute", "event_id": "m"}
OFFER = "v=0\r\no=- 1 1 IN IP4 127.0.0.1\r\ns=-\r\nt=0 0\r\nm=audio 9 UDP/TLS/RTP/SAVPF 111\r\n"
INPUT = [{"role": "user", "content": [{"type": "input_text", "text": "What teas do you have?"}]}]
TOOLS = [{"type": "function", "name": "get_menu", "parameters": {"type": "object", "properties": {}}}]
FUNCTION_CALL = {"type": "function_call", "call_id": "call_1", "name": "get_menu", "arguments": '{"category":"tea"}'}


def round_with(*items, finish="response.completed"):
    response = {"id": "resp_1", "model": "org/backend", "status": "completed", "output": list(items)}
    return [
        *({"type": "response.output_item.done", "item": item} for item in items),
        {"type": finish, "response": response},
    ]


class RoundStream:
    """A delegate round's stream: after ``go`` and ``delay``, its events; it notes whether the route closed it."""

    def __init__(self, events, delay=0.0, go=None):
        self.events, self.delay, self.go, self.closed = list(events), delay, go, False

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self.go is not None:
            await self.go.wait()
        await asyncio.sleep(self.delay)
        if not self.events:
            raise StopAsyncIteration
        return self.events.pop(0)

    async def aclose(self):
        self.closed = True


class RoundCore(FakeAdapter):
    """The gateway's one core for delegate rounds: it answers each with ``events``, or refuses it with ``refusal``."""

    def __init__(self, events=None, refusal=None, delay=0.0, go=None):
        super().__init__()
        self.events = round_with(FUNCTION_CALL) if events is None else events
        self.refusal, self.delay, self.go = refusal, delay, go
        self.rounds, self.streams, self.checked_models = [], [], []

    async def retrieve_model(self, model):
        self.checked_models.append(model)
        return await super().retrieve_model(model)

    async def create_response(self, params):
        self.rounds.append(params)
        if self.refusal is not None:
            raise self.refusal
        self.streams.append(RoundStream(self.events, self.delay, self.go))
        return self.streams[-1]


def make_app(settings=None, **hooks):
    """The live routes on an app; the test client's ``Host: testserver`` is allowed unless the settings name hosts."""
    engines = []

    async def factory(config, transport, ice_servers):
        engines.append(FakeAdapter())
        peer = LiveWebRTCConnection.from_ice_servers(ice_servers) if transport == "webrtc" else None
        return SessionParts(frontend=engines[-1], connection=peer)

    settings = settings or LiveSettings()
    settings = replace(settings, allowed_hosts=settings.allowed_hosts or ("testserver",))
    app = FastAPI()
    gateway = mount_live(app, settings=settings, **{"factory": factory, "core": FakeAdapter().core, **hooks})
    return app, gateway, engines


class ScriptedSocket:
    """The server side of a primary WebSocket: it receives the scripted frames, then waits.

    It notes the events it had sent when it was first closed.
    """

    headers = {"host": "localhost"}  # noqa: RUF012

    def __init__(self, *frames):
        self.frames, self.sent, self.sent_at_close = list(frames), [], None

    async def accept(self):
        pass

    async def receive(self):
        if self.frames:
            return self.frames.pop(0)
        await asyncio.Event().wait()

    async def send_json(self, message):
        self.sent.append(message)

    async def close(self, code=1000):
        if self.sent_at_close is None:
            self.sent_at_close = [event["type"] for event in self.sent]


def text_frame(message):
    """A client's text frame, as Starlette's ``WebSocket.receive()`` gives it."""
    return {"type": "websocket.receive", "text": message if isinstance(message, str) else json.dumps(message)}


class SlowStart(FakeAdapter):
    """A frontend adapter whose transport reads the socket while its start goes on, as a pipeline's does."""

    async def start(self, session, delegation):
        await super().start(session, delegation)
        await asyncio.sleep(0.05)


class WebSocketTests(unittest.TestCase):
    def test_a_session_starts_takes_commands_and_closes(self):
        app, _, engines = make_app(LiveSettings(max_appends_per_minute=1))
        with TestClient(app) as client, client.websocket_connect(SESSIONS) as ws:
            ws.send_json(START)
            started = ws.receive_json()
            self.assertEqual((started["type"], started["client_event_id"]), ("session.started", "start_1"))
            self.assertEqual(started["session"]["instructions"], "Be brief.")
            ws.send_json({"type": "session.input_audio.append", "audio": base64.b64encode(b"\x01\x00" * 480).decode()})
            ws.send_text("not json")  # an error event, and the session stays open
            self.assertEqual(ws.receive_json()["error"]["message"], "Invalid JSON event")
            ws.send_bytes(b"\x00\x01")
            self.assertIn("Binary frames are not supported", ws.receive_json()["error"]["message"])
            for event_id in ("t1", "t2"):  # the gateway's append limit holds for each session
                ws.send_json({**NOTE, "event_id": event_id})
            self.assertEqual(ws.receive_json()["type"], "session.thinking.appended")
            refused = ws.receive_json()
            self.assertEqual((refused["error"]["code"], refused["client_event_id"]), ("rate_limit_exceeded", "t2"))
            ws.send_json({"type": "session.close", "event_id": "bye"})
            closed = ws.receive_json()
            self.assertEqual((closed["type"], closed["reason"]), ("session.closed", "close_requested"))
        engine = engines[0]
        self.assertEqual((len(engine.audio[0]), engine.appended, engine.closed), (960, [("thinking", "note")], True))

    def test_a_refused_start_gets_an_error_event_then_a_policy_close(self):
        async def refuse(config, transport, connection):
            raise HTTPException(429, "a factory's own HTTP answer")  # a factory refuses only with a ProtocolError

        first = "session.start must be the first message"
        for frame, hooks, code, param, event_id, words in (
            ({"type": "session.input_audio.mute"}, {}, "invalid_value", None, None, first),
            (b"{}", {}, "invalid_value", None, None, first),  # a binary frame is no session.start
            ({**START, "event_id": "s9", "session": {"model": ""}}, {}, "invalid_value", "session.model", "s9", ""),
            ({**START, "session": {"model": "m", "client": {}}}, {}, "invalid_value", "session.client", "start_1", ""),
            (START, {"factory": refuse}, "server_error", None, "start_1", ""),
            (START, {"settings": LiveSettings(max_sessions=0)}, "rate_limit_exceeded", None, "start_1", ""),
        ):
            with self.subTest(frame=frame, hooks=hooks):
                app, _, engines = make_app(**hooks)
                with TestClient(app) as client, client.websocket_connect(SESSIONS) as ws:
                    ws.send_bytes(frame) if isinstance(frame, bytes) else ws.send_json(frame)
                    error = ws.receive_json()
                    self.assertEqual((error["error"]["code"], error["error"].get("param")), (code, param))
                    self.assertEqual(error.get("client_event_id"), event_id)
                    self.assertIn(words, error["error"]["message"])
                    with self.assertRaises(WebSocketDisconnect) as closed:
                        ws.receive_json()
                    self.assertEqual(closed.exception.code, 1008)
                self.assertEqual(engines, [])

    def test_a_client_that_closes_its_websocket_ends_the_session_by_its_close_code(self):
        # Driven frame by frame: Starlette's TestClient may cancel the handler as it leaves, which a server does not.
        for code, reason in ((1000, "remote_hangup"), (1001, "remote_hangup"), (1006, "connection_lost")):
            with self.subTest(code=code):
                _, gateway, engines = make_app()
                start = {"type": "websocket.receive", "text": json.dumps(START)}
                ws = ScriptedSocket(start, {"type": "websocket.disconnect", "code": code})
                asyncio.run(run_primary_websocket(gateway, ws))
                self.assertEqual([event["reason"] for event in ws.sent if event["type"] == "session.closed"], [reason])
                self.assertTrue(engines[0].closed)

    def test_a_frame_over_the_size_limit_gets_its_error_then_ends_the_session(self):
        _, gateway, engines = make_app()
        oversized = text_frame(json.dumps({**NOTE, "content": "x" * 2**20}))
        ws = ScriptedSocket(text_frame(START), oversized, text_frame(MUTE))
        asyncio.run(run_primary_websocket(gateway, ws))
        self.assertEqual([event["type"] for event in ws.sent], ["session.started", "error", "session.closed"])
        error, closed = ws.sent[1], ws.sent[2]
        self.assertEqual(
            (error["error"]["message"], error["client_event_id"]), ("WebSocket event exceeds 1 MiB", "start_1")
        )
        self.assertEqual(closed["reason"], "connection_lost")
        self.assertEqual((engines[0].muted, engines[0].closed), (False, True))  # nothing after it is taken
        self.assertEqual(ws.sent_at_close, ["session.started", "error", "session.closed"])

    def test_a_command_sent_with_the_start_waits_for_the_session_to_become_active(self):
        engines = []

        async def factory(config, transport, ice_servers):
            engines.append(SlowStart())
            return engines[-1].parts()

        _, gateway, _ = make_app(factory=factory)
        bye = {"type": "websocket.disconnect", "code": 1000}
        ws = ScriptedSocket(text_frame(START), text_frame(MUTE), bye)
        asyncio.run(run_primary_websocket(gateway, ws))
        kinds = ["session.started", "session.input_audio.muted", "session.closed"]
        self.assertEqual([event["type"] for event in ws.sent], kinds)
        self.assertEqual((ws.sent[-1]["reason"], engines[0].muted), ("remote_hangup", True))
        self.assertEqual(ws.sent_at_close, kinds)  # the socket closes once session.closed has gone out

    def test_server_shutdown_closes_every_session_as_connection_lost(self):
        app, gateway, engines = make_app()

        async def run():
            client = Collector()
            async with app.router.lifespan_context(app):
                session = await gateway.register(SessionConfig(model="live-1"), "websocket", None)
                session.subscribe(client.send, "primary")
                await session.start()
            return [event["reason"] for event in client.events if event["type"] == "session.closed"]

        self.assertEqual(asyncio.run(run()), ["connection_lost"])
        self.assertTrue(engines[0].closed)

    def test_an_observer_sees_later_events_sends_trusted_commands_and_survives_bad_json(self):
        app, _, engines = make_app()
        with TestClient(app) as client, client.websocket_connect(SESSIONS) as primary:
            primary.send_json(START)
            session_id = primary.receive_json()["session"]["id"]
            with client.websocket_connect(f"{SESSIONS}/{session_id}/attach") as observer:
                observer.send_text("this is not json")
                self.assertEqual(observer.receive_json()["error"]["message"], "Invalid JSON event")
                for sender, command, ack in (
                    (observer, NOTE, "session.thinking.appended"),
                    (primary, MUTE, "session.input_audio.muted"),
                ):
                    sender.send_json(command)
                    self.assertEqual([ws.receive_json()["type"] for ws in (primary, observer)], [ack, ack])
            with (
                self.assertRaises(WebSocketDisconnect),
                client.websocket_connect(f"{SESSIONS}/live_x/attach") as unknown,
            ):
                unknown.receive_json()
        self.assertEqual(engines[0].appended, [("thinking", "note")])


class RouteTests(unittest.TestCase):
    def test_history_composes_the_frontends_half_and_the_delegation_parts_half(self):
        app, _, engines = make_app()
        responses = {"model": "live-1", "delegation": {"type": "responses", "responses": {"model": "org/org/model"}}}
        frontend = {"messages": [{"role": "user", "content": "Hi."}]}

        async def held():
            return {"frontend": frontend, "backend": {"busy": False}}

        with TestClient(app) as client:
            with client.websocket_connect(SESSIONS) as plain, client.websocket_connect(SESSIONS) as backed:
                plain.send_json(START)  # client mode: the server runs no backend for it
                backed.send_json({**START, "session": responses})
                urls = [f"{SESSIONS}/{ws.receive_json()['session']['id']}/history" for ws in (plain, backed)]
                # An adapter that holds nothing yet gives no half, and the route still answers.
                self.assertEqual(client.get(urls[0]).json(), {"frontend": None, "backend": None})
                backend = {"model": "org/org/model", "prompt": [], "items": []}
                self.assertEqual(client.get(urls[1]).json(), {"frontend": None, "backend": backend})
                for engine in engines:
                    engine.history = held
                update = {"delegation": {"responses": {"model": "org/other-model", "instructions": "Be brief."}}}
                backed.send_json({"type": "session.update", "event_id": "u1", "session": update})
                backed.receive_json()
                self.assertEqual(client.get(urls[0]).json(), {"frontend": frontend, "backend": None})
                prompt = [{"source": "delegation.responses.instructions", "text": "Be brief."}]
                backend = {"model": "org/other-model", "prompt": prompt, "items": [], "busy": False}
                self.assertEqual(client.get(urls[1]).json(), {"frontend": frontend, "backend": backend})
            self.assertEqual(client.get(f"{SESSIONS}/live_unknown/history").status_code, 404)

    def test_info_gives_the_voices_the_session_limit_and_what_the_root_describes(self):
        async def describe():
            return {"frontend": {"model": "talker"}, "warnings": []}

        for hook, described in ((describe, {"frontend": {"model": "talker"}, "warnings": []}), (None, {})):
            app, _, _ = make_app(LiveSettings(max_sessions=3), describe=hook)
            with TestClient(app) as client:
                info = client.get("/v1/live/info").json()
            self.assertIn("marin", info.pop("voices"))
            self.assertEqual(info, {"max_sessions": 3, **described})

    def test_http_session_creation_needs_a_webrtc_offer_it_can_answer(self):
        async def refuse(config, transport, connection):
            raise ProtocolError("No session for you", "server_error")

        webrtc = {"type": "webrtc", "sdp": OFFER}
        app, _, engines = make_app()
        with TestClient(app) as client:
            for body, status in (
                ({}, 400),
                ({"transport": {"type": "websocket"}, "session": {"model": "live-1"}}, 400),
                ({"transport": {"type": "webrtc"}, "session": {"model": "live-1"}}, 400),  # no SDP
                ({"transport": {"type": "webrtc", "sdp": "not an sdp"}, "session": {"model": "live-1"}}, 400),
                ({"transport": webrtc, "session": {"model": ""}}, 400),
                ({"transport": webrtc, "session": {"model": "live-1", "audio": {"format": {}}}}, 400),  # SDP decides
                (b"[1]", 400),
                (b"x" * (2**20 + 1), 413),
            ):
                with self.subTest(body=str(body)[:80]):
                    sent = {"content": body} if isinstance(body, bytes) else {"json": body}
                    self.assertEqual(client.post(SESSIONS, **sent).status_code, status)
            unknown = client.post(SESSIONS, json={"transport": webrtc, "session": {"model": "m", "voice": "x"}})
            error = unknown.json()["error"]
            self.assertEqual(
                (unknown.status_code, error["code"], error["param"]), (400, "unknown_parameter", "session.voice")
            )
        self.assertEqual(engines, [])
        app, _, _ = make_app(factory=refuse)
        with TestClient(app) as client:
            self.assertEqual(
                client.post(SESSIONS, json={"transport": webrtc, "session": {"model": "m"}}).status_code, 503
            )


class SecurityTests(unittest.TestCase):
    def test_a_configured_token_is_required_on_every_route(self):
        app, _, _ = make_app(LiveSettings(api_token="s3cret"), core=RoundCore().core)
        bearer = {"Authorization": "Bearer s3cret"}
        with TestClient(app) as client:
            with self.assertRaises(WebSocketDisconnect), client.websocket_connect(SESSIONS) as refused:
                refused.send_json(START)
                refused.receive_json()
            with client.websocket_connect(SESSIONS, headers=bearer) as primary:
                primary.send_json(START)
                session_id = primary.receive_json()["session"]["id"]
                with (
                    self.assertRaises(WebSocketDisconnect),
                    client.websocket_connect(f"{SESSIONS}/{session_id}/attach") as refused,
                ):
                    refused.receive_json()
                with client.websocket_connect(f"{SESSIONS}/{session_id}/attach", headers=bearer) as observer:
                    observer.send_json(MUTE)
                    self.assertEqual(observer.receive_json()["type"], "session.input_audio.muted")
                routes = (("POST", SESSIONS), ("GET", f"{SESSIONS}/{session_id}/history"))
                for method, path in (*routes, ("GET", "/v1/live/info"), ("POST", "/v1/live/delegate")):
                    for header in ("", "Bearer é".encode("latin-1"), b"Bearer \xff\xfe"):  # never a crash
                        with self.subTest(route=path, header=header):
                            response = client.request(
                                method, path, headers={"Authorization": header}, json={"input": INPUT}
                            )
                            self.assertEqual(response.status_code, 401)
            self.assertEqual(client.get("/v1/live/info", headers=bearer).status_code, 200)
            self.assertEqual(client.post("/v1/live/delegate", json={"input": INPUT}, headers=bearer).status_code, 200)

    def test_cross_site_origins_and_unknown_host_names_are_refused(self):
        app, _, engines = make_app(core=RoundCore().core)
        evil = {"Origin": "https://evil.example"}
        # A page on another site whose name now resolves to this server (DNS rebinding): its origin matches its host.
        rebound = {"Host": "attacker.example:7860", "Origin": "http://attacker.example:7860"}
        with TestClient(app) as client, client.websocket_connect(SESSIONS) as ws:
            ws.send_json(START)
            session_id = ws.receive_json()["session"]["id"]
            routes = (("POST", SESSIONS), ("GET", f"{SESSIONS}/{session_id}/history"))
            for headers in (evil, rebound):
                for method, path in (*routes, ("GET", "/v1/live/info"), ("POST", "/v1/live/delegate")):
                    with self.subTest(route=f"{method} {path}", headers=headers):
                        response = client.request(method, path, headers=headers, json={"input": INPUT})
                        self.assertEqual(response.status_code, 403)
                for path in (SESSIONS, f"{SESSIONS}/{session_id}/attach"):
                    with (
                        self.assertRaises(WebSocketDisconnect),
                        client.websocket_connect(path, headers=headers) as refused,
                    ):
                        refused.send_json(START)
                        refused.receive_json()
            self.assertIn("allowed_hosts", client.get("/v1/live/info", headers=rebound).json()["detail"])
        self.assertEqual(len(engines), 1)  # only the session of the allowed host was built

    def test_with_no_token_the_routes_answer_an_ip_literal_localhost_or_an_allowed_name(self):
        token = LiveSettings(api_token="secret")
        for host, settings, headers, status in (
            *(
                (host, None, {}, 200)
                for host in ("127.0.0.1", "10.0.0.5:8000", "[::1]:8000", "localhost:7860", "LOCALHOST")
            ),
            ("Voice.Lab:8000", LiveSettings(allowed_hosts=("voice.lab",)), {}, 200),
            ("evil.example", None, {}, 403),
            ("evil.example", token, {"Authorization": "Bearer secret"}, 200),  # with a token, the token decides
            ("evil.example", token, {}, 401),
        ):
            with self.subTest(host=host, settings=settings):
                app, _, _ = make_app(settings)
                with TestClient(app) as client:
                    self.assertEqual(client.get("/v1/live/info", headers={"host": host, **headers}).status_code, status)


class MountTests(unittest.TestCase):
    def test_the_settings_are_the_examples_values_with_the_token_and_the_workers_from_the_environment(self):
        limits = {"max_sessions": 3, "max_session_seconds": 60, "max_appends_per_minute": 5, "max_delegate_rounds": 1}
        # Each string is stripped, and a blank one dropped.
        values = {**limits, "ice_urls": ["stun:stun.example:3478", " "], "allowed_hosts": [" Voice.Lab ", ""]}
        with patch.dict(os.environ, {"LIVE_API_TOKEN": " s3cr3t-token \n", "UVICORN_WORKERS": "1"}, clear=True):
            settings = LiveSettings.from_env(values)
        lists = {"ice_urls": ("stun:stun.example:3478",), "allowed_hosts": ("voice.lab",)}
        self.assertEqual(settings, LiveSettings(api_token="s3cr3t-token", **limits, **lists))
        self.assertNotIn("s3cr3t-token", repr(settings))
        # A blank token is no token, and a key the values lack keeps its default.
        with patch.dict(os.environ, {"LIVE_API_TOKEN": "   "}, clear=True):
            self.assertEqual(LiveSettings.from_env(), LiveSettings())
            self.assertEqual(LiveSettings.from_env({"max_sessions": 2}), LiveSettings(max_sessions=2))
        for value in ("many", "0"):
            with patch.dict(os.environ, {"UVICORN_WORKERS": value}), self.assertRaisesRegex(ValueError, "UVICORN"):
                mount_live(FastAPI(), factory=lambda *args: None)

    def test_the_mount_warns_without_a_token_and_serves_no_live_route_with_several_workers(self):
        logged = []
        sink = logger.add(lambda message: logged.append(message.record["message"]), level="WARNING")
        self.addCleanup(logger.remove, sink)
        for token in ("", "secret"):
            with patch.dict(os.environ, {"LIVE_API_TOKEN": token, "UVICORN_WORKERS": "1"}):
                mount_live(FastAPI(), factory=lambda *args: None)
        app = FastAPI()
        with patch.dict(os.environ, {"UVICORN_WORKERS": "2"}):
            self.assertIsNone(mount_live(app, factory=lambda *args: None))
        with TestClient(app) as client:
            self.assertEqual(client.post(SESSIONS, json={}).status_code, 404)
        [warning, error] = logged
        self.assertIn("LIVE_API_TOKEN is not set", warning)
        self.assertIn("POST /v1/live/delegate", warning)
        self.assertIn("UVICORN_WORKERS=2", error)

    def test_a_disabled_or_unknown_live_path_is_a_404_never_the_web_page(self):
        disabled = FastAPI()
        with patch.dict(os.environ, {"LIVE_ENABLED": "false"}):
            mount_live(disabled)
        enabled, _, _ = make_app()
        for app, path in ((disabled, "/v1/live/info"), (disabled, f"{SESSIONS}/x/y"), (enabled, "/v1/live/nothing")):

            @app.get("/{path:path}")
            async def web_page(path: str):  # the web app's catch-all, added after the live routes
                return {"page": "index.html"}

            with self.subTest(path=path), TestClient(app) as client:
                self.assertEqual(client.get(path).status_code, 404)
                self.assertEqual(client.get("/somewhere").json(), {"page": "index.html"})
        with TestClient(disabled) as client:
            self.assertEqual(client.post("/v1/live/delegate", json={"input": INPUT}).status_code, 404)
        with TestClient(enabled) as client:
            self.assertEqual(client.get(f"{SESSIONS}/x/y").status_code, 404)
            self.assertEqual(client.get("/v1/live/info").status_code, 200)  # a real live route still answers


class DelegateRouteTests(unittest.TestCase):
    """``POST /v1/live/delegate``: one stateless backend round for a client that owns its delegation."""

    def post(self, body, core=None, settings=None, headers=None):
        self.core = core or RoundCore()
        app, _, _ = make_app(settings, core=self.core.core)
        with TestClient(app) as client:
            return client.post("/v1/live/delegate", json=body, headers=headers or {})

    def test_a_round_returns_its_output_items_so_the_client_can_run_the_function_calls(self):
        response = self.post({"input": INPUT, "tools": TOOLS})
        sent = {"model": "org/backend", "tools": ["get_menu"]}
        self.assertEqual(
            response.json(), {"response": {"id": "resp_1", "output": [FUNCTION_CALL]}, "sent_request": sent}
        )
        # Stateless, with no conversation: the core fills the model and the instructions, and no model is checked.
        request = {"input": INPUT, "tools": TOOLS, "tool_choice": "auto", "stream": True, "store": False}
        self.assertEqual(
            (self.core.rounds, self.core.checked_models, self.core.streams[0].closed), ([request], [], True)
        )
        self.post({"input": INPUT, "model": "org/other", "instructions": "Be terse."})  # the client may choose them
        self.assertEqual(
            (self.core.rounds[0]["model"], self.core.rounds[0]["instructions"]), ("org/other", "Be terse.")
        )
        self.assertEqual(self.core.checked_models, ["org/other"])

    def test_a_bad_request_is_a_400_and_runs_no_round(self):
        nested = "A request may nest at most 64 levels"
        for body, core, detail in (
            ({}, None, "input must be a non-empty list of items"),
            ({"input": []}, None, "input must be a non-empty list of items"),
            ({"input": ["text"]}, None, "input must be a non-empty list of items"),
            ({"input": INPUT, "tools": [{"type": "function"}]}, None, "tools must be function or web_search tools"),
            ({"input": INPUT, "model": 7}, None, "model must be a string"),
            ({"input": INPUT, "model": "bad-model"}, None, "Unknown model 'bad-model'"),
            ({"input": INPUT}, RoundCore(refusal=ValueError(nested)), nested),  # the core refuses it
        ):
            with self.subTest(body=body):
                response = self.post(body, core)
                self.assertEqual(
                    (response.status_code, response.json()["detail"], self.core.streams), (400, detail, [])
                )

    def test_a_round_the_backend_could_not_finish_is_a_502_without_its_detail_and_its_work_is_stopped(self):
        logged = []
        sink = logger.add(logged.append, level="DEBUG", diagnose=True)  # as the server's sink: each traceback's values
        self.addCleanup(logger.remove, sink)
        token = LiveSettings(api_token="s3cr3t-token-value", delegate_round_seconds=0.05)
        for core, settings, headers in (
            (RoundCore(events=round_with(finish="response.failed")), None, None),
            (RoundCore(events=round_with(finish="response.incomplete")), None, None),
            (RoundCore(refusal=RuntimeError("/srv/config.yaml is unreadable")), None, None),
            (RoundCore(delay=1.0), token, {"Authorization": "Bearer s3cr3t-token-value"}),  # past the backstop
        ):
            with self.subTest(settings=settings):
                response = self.post({"input": INPUT}, core, settings, headers)
                self.assertEqual(response.status_code, 502)
                self.assertEqual(response.json()["detail"], "The backend could not complete the round")
                self.assertTrue(all(stream.closed for stream in core.streams))
        self.assertIn("A delegate round failed", "".join(logged))
        self.assertNotIn("s3cr3t-token-value", "".join(logged))

    def test_the_route_needs_a_core_and_limits_its_body_and_the_rounds_at_once(self):
        app, _, _ = make_app(core=None)
        with TestClient(app) as client:
            response = client.post("/v1/live/delegate", json={"input": INPUT})
        self.assertEqual(
            (response.status_code, response.json()["detail"]), (503, "The delegation core is not available")
        )
        big = [{"role": "user", "content": [{"type": "input_text", "text": "x" * 2048}]}]
        self.assertEqual(self.post({"input": big}, settings=LiveSettings(delegate_body_bytes=1024)).status_code, 413)
        self.assertEqual(self.core.rounds, [])
        go = asyncio.Event()
        core = RoundCore(events=round_with(), go=go)
        app, _, _ = make_app(LiveSettings(max_delegate_rounds=1), core=core.core)

        async def rounds():
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1") as http:
                first = asyncio.create_task(http.post("/v1/live/delegate", json={"input": INPUT}))
                while not core.rounds:
                    await asyncio.sleep(0.005)
                async with asyncio.timeout(5):  # a round over the cap that runs would wait here for ``go``
                    over = await http.post("/v1/live/delegate", json={"input": INPUT})
                go.set()
                return over, await first, await http.post("/v1/live/delegate", json={"input": INPUT})

        over, done, again = asyncio.run(rounds())
        self.assertEqual(
            (over.status_code, over.json()["detail"]), (429, "Too many delegate rounds: at most 1 at once")
        )
        self.assertEqual((done.status_code, again.status_code, len(core.rounds)), (200, 200, 2))  # a finished one frees


class WebRTCTests(unittest.IsolatedAsyncioTestCase):
    """Real SDP, DTLS and SCTP on loopback, through the real HTTP endpoint."""

    async def connect(self, label="oai-events"):
        app, _, self.engines = make_app()
        self.client = RTCPeerConnection()
        self.client.addTrack(AudioStreamTrack())
        self.channel = self.client.createDataChannel(label)
        self.seen, self.seen_at_close = [], None

        @self.channel.on("message")
        def on_message(raw):
            self.seen.append(json.loads(raw))

        @self.channel.on("close")
        def on_close():
            self.seen_at_close = [event["type"] for event in self.seen]

        self.addAsyncCleanup(self.client.close)
        await self.client.setLocalDescription(await self.client.createOffer())
        self.http = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1")
        self.addAsyncCleanup(self.http.aclose)
        offer = {"type": "webrtc", "sdp": self.client.localDescription.sdp}
        response = await self.http.post(SESSIONS, json={"session": START["session"], "transport": offer})
        self.assertEqual(response.status_code, 201, response.text)
        body = response.json()
        await self.client.setRemoteDescription(RTCSessionDescription(sdp=body["transport"]["sdp"], type="answer"))
        self.history = f"{SESSIONS}/{body['session']['id']}/history"
        return body

    async def until(self, predicate):
        async with asyncio.timeout(15):
            while not predicate():
                await asyncio.sleep(0.05)

    async def event(self, kind):
        await self.until(lambda: any(event["type"] == kind for event in self.seen))
        return next(event for event in self.seen if event["type"] == kind)

    async def gone(self):
        """Wait until the session has left the gateway: its history is no longer served."""
        async with asyncio.timeout(15):
            while (await self.http.get(self.history)).status_code != 404:
                await asyncio.sleep(0.05)

    async def test_a_session_runs_over_the_event_channel_whatever_its_label(self):
        for label in ("oai-events", ""):  # clients name the event channel differently
            with self.subTest(label=label):
                body = await self.connect(label)
                # The official answer: the session's id and the SDP answer, nothing more.
                self.assertEqual((set(body), set(body["session"])), ({"session", "transport"}, {"id"}))
                self.assertEqual(set(body["transport"]), {"type", "sdp"})
                started = (await self.event("session.started"))["session"]
                self.assertEqual((started["id"], started["instructions"]), (body["session"]["id"], "Be brief."))
                self.channel.send(json.dumps({"type": "session.input_audio.mute", "event_id": "mute"}))
                self.assertEqual((await self.event("session.input_audio.muted"))["client_event_id"], "mute")
                self.channel.send(json.dumps({"type": "session.close", "event_id": "close"}))
                await self.gone()
                await self.until(lambda: self.seen_at_close is not None)
                self.assertIn("session.closed", self.seen_at_close)  # the final event comes before the channel closes
                self.assertEqual((self.engines[-1].muted, self.engines[-1].closed), (True, True))
                await self.client.close()

    async def test_a_peer_that_leaves_without_session_close_ends_its_session_as_a_hang_up(self):
        for leave in ("peer", "data channel"):
            with self.subTest(leave=leave):
                lines = []
                self.addCleanup(logger.remove, logger.add(lines.append, level="INFO"))
                await self.connect()
                await self.event("session.started")
                if leave == "peer":
                    await self.client.close()
                else:
                    self.channel.close()
                await self.gone()
                self.assertTrue(self.engines[-1].closed)
                self.assertIn("reason=remote_hangup", next(line for line in lines if " closes: " in line))

    async def test_a_session_whose_peer_never_connected_is_closed_at_the_deadline(self):
        class Peer:
            """A session and its WebRTC peer, as the deadline sees them."""

            def __init__(self, status, connected):
                self.status, self.connected, self.reasons = status, connected, []

            def is_connected(self):
                return self.connected

            async def close(self, reason, client_event_id=None, *, asked_by="the gateway"):
                self.reasons.append(reason)

        for status, connected, reasons in (
            ("active", False, ["connection_lost"]),
            ("active", True, []),
            ("starting", True, ["connection_lost"]),
        ):
            peer = Peer(status, connected)
            await expire_unconnected(peer, peer, timeout=0)
            self.assertEqual(peer.reasons, reasons, (status, connected))


if __name__ == "__main__":
    unittest.main()
