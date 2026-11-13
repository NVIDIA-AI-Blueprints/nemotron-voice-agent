# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D103, D107

"""The live routes as clients reach them, the mount, and the gateway's one core and session registry."""

import asyncio
import base64
import json
import os
import unittest
from dataclasses import replace
from unittest import mock

import httpx
from aiortc import RTCPeerConnection, RTCSessionDescription
from aiortc.mediastreams import AudioStreamTrack
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from loguru import logger
from starlette.websockets import WebSocketDisconnect

from examples.frontend_backend_live.adapter.connection import LivePeer
from gateways.live.gateway import LiveGateway, LiveSettings
from gateways.live.gateway.webrtc import expire_unconnected
from gateways.live.gateway.websocket import run_primary_websocket
from gateways.live.mount import mount_live
from gateways.live.ports import SessionParts
from gateways.live.protocol import ProtocolError, SessionConfig
from tests.unit.live_helpers import Collector, FakeAdapter, FakeWebSocket, Stream, response, responses_config

SESSIONS, DELEGATE = "/v1/live/sessions", "/v1/live/delegate"
START = {"type": "session.start", "event_id": "start_1", "session": {"model": "live-1", "instructions": "Be brief."}}
NOTE = {"type": "session.thinking.append", "delegation_id": None, "content": "note"}
MUTE = {"type": "session.input_audio.mute", "event_id": "m"}
# A message of Pipecat's own on the data channel, which its connection takes: a track's status, not a live event.
SIGNALLING = {"type": "signalling", "message": {"type": "trackStatus", "receiver_index": 0, "enabled": True}}
OFFER = "v=0\r\no=- 1 1 IN IP4 127.0.0.1\r\ns=-\r\nt=0 0\r\nm=audio 9 UDP/TLS/RTP/SAVPF 111\r\n"
INPUT = [{"role": "user", "content": [{"type": "input_text", "text": "What teas do you have?"}]}]
CLIENT = SessionConfig(model="live-1")


def make_app(settings=None, **hooks):
    """Mount the live routes on an app that allows the test client's host unless the settings name hosts."""
    engines = []

    async def factory(config, transport, ice_servers):
        peer = LivePeer.from_ice_servers(ice_servers) if transport == "webrtc" else None
        engines.append(FakeAdapter(peer))
        return SessionParts(frontend=engines[-1], connection=peer)

    settings = settings or LiveSettings()
    settings = replace(settings, allowed_hosts=settings.allowed_hosts or ("testserver",))
    app = FastAPI()
    gateway = mount_live(app, settings=settings, **{"factory": factory, "core": FakeAdapter().core, **hooks})
    return app, gateway, engines


def run_socket(gateway, *messages, leave=None):
    """Serve one primary WebSocket whose client sends ``messages``, then closes with the code ``leave``, if given."""
    ws = FakeWebSocket()
    for message in messages:
        ws.send(message)
    if leave is not None:
        ws.leave(leave)
    asyncio.run(run_primary_websocket(gateway, ws))
    return ws


class WebSocketTests(unittest.TestCase):
    def test_a_session_starts_takes_commands_and_closes(self):
        app, _, engines = make_app(LiveSettings(max_appends_per_minute=1))
        with TestClient(app) as client, client.websocket_connect(SESSIONS) as ws:
            ws.send_json(START)
            started = ws.receive_json()
            self.assertEqual((started["type"], started["client_event_id"]), ("session.started", "start_1"))
            self.assertEqual(started["session"]["instructions"], "Be brief.")
            ws.send_json({"type": "session.input_audio.append", "audio": base64.b64encode(b"\x01\x00" * 480).decode()})
            ws.send_text("not json")
            self.assertEqual(ws.receive_json()["error"]["message"], "Invalid JSON event")
            ws.send_bytes(b"\x00\x01")
            self.assertIn("Binary frames are not supported", ws.receive_json()["error"]["message"])
            for event_id in ("t1", "t2"):
                ws.send_json({**NOTE, "event_id": event_id})
            self.assertEqual(ws.receive_json()["type"], "session.thinking.appended")
            refused = ws.receive_json()
            self.assertEqual((refused["error"]["code"], refused["client_event_id"]), ("rate_limit_exceeded", "t2"))
            ws.send_json({"type": "session.close", "event_id": "bye"})
            closed = ws.receive_json()
            self.assertEqual((closed["type"], closed["reason"]), ("session.closed", "close_requested"))
        engine = engines[0]
        appended = [("thinking", "note", "t1", None)]
        self.assertEqual((len(engine.audio[0]), engine.appended, engine.closed), (960, appended, True))

    def test_a_refused_start_gets_an_error_event_then_a_policy_close(self):
        async def refuse(config, transport, connection):
            raise HTTPException(429, "a factory's own HTTP answer")  # a factory refuses only with a ProtocolError

        first = "session.start must be the first message"
        for frame, hooks, code, param, event_id, words in (
            ({"type": "session.input_audio.mute"}, {}, "invalid_value", None, None, first),
            (b"{}", {}, "invalid_value", None, None, first),
            ({**START, "event_id": "s9", "session": {"model": ""}}, {}, "invalid_value", "session.model", "s9", ""),
            ({**START, "session": {"model": "m", "client": {}}}, {}, "invalid_value", "session.client", "start_1", ""),
            (START, {"factory": refuse}, "server_error", None, "start_1", ""),
            # A start over the event limit, which every later event has too: it is not read, so no id is known.
            ({**START, "session": {"model": "m", "instructions": "x" * 2**20}}, {}, "invalid_value", None, None, "MiB"),
        ):
            with self.subTest(code=code, param=param, words=words, binary=isinstance(frame, bytes)):
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

    def test_the_close_code_gives_the_close_reason(self):
        # Driven frame by frame: Starlette's TestClient may cancel the handler as it leaves, which a server does not.
        for code, reason in ((1000, "remote_hangup"), (1001, "remote_hangup"), (1006, "connection_lost")):
            with self.subTest(code=code):
                _, gateway, engines = make_app()
                ws = run_socket(gateway, START, leave=code)
                closed = [event["reason"] for event in ws.from_server if event["type"] == "session.closed"]
                self.assertEqual((closed, engines[0].closed), ([reason], True))

    def test_an_oversized_frame_gets_its_error_then_ends_the_session(self):
        _, gateway, engines = make_app()
        ws = run_socket(gateway, START, {**NOTE, "content": "x" * 2**20}, MUTE)
        self.assertEqual(ws.sent_at_close, ["session.started", "error", "session.closed"])
        self.assertEqual(ws.kinds(), ws.sent_at_close)
        error, closed = ws.from_server[1:]
        self.assertEqual(error["error"]["message"], "WebSocket event exceeds 1 MiB")
        self.assertIsNone(error.get("client_event_id"))
        self.assertEqual((closed["reason"], engines[0].muted, engines[0].closed), ("connection_lost", False, True))

    def test_a_failed_start_gets_one_error_then_session_closed(self):
        def core():
            raise RuntimeError("bad backend block")

        _, gateway, _ = make_app(core=core)
        delegation = {"type": "responses", "responses": {"model": "org/model"}}
        start = {**START, "session": {"model": "live-1", "delegation": delegation}}
        ws = run_socket(gateway, start)
        self.assertEqual(ws.kinds(), ["error", "session.closed"])
        error = ws.from_server[0]
        self.assertEqual((error["error"]["code"], error["client_event_id"]), ("server_error", "start_1"))

    def test_an_observer_sees_later_events_and_sends_commands(self):
        app, _, engines = make_app()
        with TestClient(app) as client, client.websocket_connect(SESSIONS) as primary:
            primary.send_json(START)
            session_id = primary.receive_json()["session"]["id"]
            with client.websocket_connect(f"{SESSIONS}/{session_id}/attach") as observer:
                observer.send_text("this is not json")
                self.assertEqual(observer.receive_json()["error"]["message"], "Invalid JSON event")
                for sender, command, ack in (
                    (observer, NOTE, "thinking.appended"),
                    (primary, MUTE, "input_audio.muted"),
                ):
                    sender.send_json(command)
                    self.assertEqual([ws.receive_json()["type"] for ws in (primary, observer)], [f"session.{ack}"] * 2)
            with self.assertRaises(WebSocketDisconnect), client.websocket_connect(f"{SESSIONS}/x/attach") as unknown:
                unknown.receive_json()
        self.assertEqual(engines[0].appended, [("thinking", "note", None, None)])


class RouteTests(unittest.TestCase):
    def test_history_composes_the_frontend_and_delegation_halves(self):
        app, _, engines = make_app()
        responses = {"model": "live-1", "delegation": {"type": "responses", "responses": {"model": "org/org/model"}}}
        frontend = {"messages": [{"role": "user", "content": "Hi."}]}
        with TestClient(app) as client, client.websocket_connect(SESSIONS) as ws:
            ws.send_json({**START, "session": responses})
            url = f"{SESSIONS}/{ws.receive_json()['session']['id']}/history"
            backend = {"model": "org/org/model", "prompt": [], "items": []}
            self.assertEqual(client.get(url).json(), {"frontend": None, "backend": backend})
            engines[0].history = mock.AsyncMock(return_value={"frontend": frontend, "backend": {"busy": False}})
            update = {"delegation": {"responses": {"model": "org/other-model", "instructions": "Be brief."}}}
            ws.send_json({"type": "session.update", "event_id": "u1", "session": update})
            ws.receive_json()
            prompt = [{"source": "delegation.responses.instructions", "text": "Be brief."}]
            backend = {"model": "org/other-model", "prompt": prompt, "items": [], "busy": False}
            self.assertEqual(client.get(url).json(), {"frontend": frontend, "backend": backend})
            self.assertEqual(client.get(f"{SESSIONS}/live_unknown/history").status_code, 404)

    def test_info_gives_the_voices_the_limit_and_the_roots_part(self):
        described = {"frontend": {"model": "talker"}, "warnings": []}

        async def describe():
            return described

        for hook, extra in ((describe, described), (None, {})):
            app, _, _ = make_app(LiveSettings(max_sessions=3), describe=hook)
            with TestClient(app) as client:
                info = client.get("/v1/live/info").json()
            self.assertIn("marin", info.pop("voices"))
            self.assertEqual(info, {"max_sessions": 3, **extra})

    def test_an_http_session_needs_a_webrtc_offer_it_can_answer(self):
        webrtc, session = {"type": "webrtc", "sdp": OFFER}, {"model": "live-1"}
        app, _, engines = make_app()
        with TestClient(app) as client:
            for body, status in (
                ({"transport": {"type": "websocket"}, "session": session}, 400),
                ({"transport": {"type": "webrtc"}, "session": session}, 400),
                ({"transport": {"type": "webrtc", "sdp": "not an sdp"}, "session": session}, 400),
                ({"transport": webrtc, "session": {**session, "audio": {"format": {}}}}, 400),
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
        app, _, _ = make_app(factory=mock.AsyncMock(side_effect=ProtocolError("No session for you", "server_error")))
        with TestClient(app) as client:
            self.assertEqual(client.post(SESSIONS, json={"transport": webrtc, "session": session}).status_code, 503)

    def test_the_roots_ice_servers_reach_the_peers_factory(self):
        turn = {"urls": ["turn:turn.example:3478?transport=udp"], "username": "user", "credential": "secret"}
        asked, handed = [], []
        peer = mock.Mock(["initialize", "close_after_drain"])
        peer.initialize, peer.close_after_drain = mock.AsyncMock(side_effect=RuntimeError("no peer")), mock.AsyncMock()

        def ice_servers(request):
            asked.append(request.url.path)
            return [turn]

        async def factory(config, transport, servers):
            handed.append(servers)
            return SessionParts(frontend=FakeAdapter(), connection=peer)

        app = FastAPI()
        settings = LiveSettings(ice_urls=("stun:stun.example:3478",))
        gateway = mount_live(app, factory=factory, settings=settings, ice_servers=ice_servers)
        offer = {"session": {"model": "live-1"}, "transport": {"type": "webrtc", "sdp": OFFER}}
        answer = TestClient(app, base_url="http://127.0.0.1").post(SESSIONS, json=offer)
        self.assertEqual((answer.status_code, asked), (400, [SESSIONS]))
        self.assertEqual(handed, [[turn, {"urls": ["stun:stun.example:3478"]}]])
        self.assertEqual((peer.close_after_drain.await_count, gateway.sessions), (1, {}))

    def delegate(self, body, core=None, settings=None, headers=None):
        self.core = core or FakeAdapter()
        app, _, _ = make_app(settings, core=self.core.core)
        with TestClient(app) as client:
            return client.post(DELEGATE, json=body, headers=headers or {})

    def test_a_bad_request_is_a_400_and_runs_no_round(self):
        for body, detail in (
            ({"input": []}, "input must be a non-empty list of items"),
            ({"input": ["text"]}, "input must be a non-empty list of items"),
            ({"input": INPUT, "tools": [{"type": "function"}]}, "tools must be function or web_search tools"),
            ({"input": INPUT, "model": 7}, "model must be a string"),
            ({"input": INPUT, "model": "bad-model"}, "Unknown model 'bad-model'"),
        ):
            with self.subTest(body=body):
                reply = self.delegate(body)
                self.assertEqual((reply.status_code, reply.json()["detail"], self.core.requests), (400, detail, []))

    def test_a_failed_round_is_a_502_without_detail_and_is_stopped(self):
        logged = []
        self.addCleanup(logger.remove, logger.add(logged.append, level="DEBUG", diagnose=True))
        refused, stalled = FakeAdapter(), Stream(until=asyncio.Event())
        refused.fail["response"] = RuntimeError("/srv/settings.yaml is unreadable")
        token = LiveSettings(api_token="s3cr3t-token-value", delegate_round_seconds=0.05)
        bearer, failed = {"Authorization": "Bearer s3cr3t-token-value"}, "The backend could not complete the round"
        for core, settings, headers in ((refused, None, None), (FakeAdapter(streams=[stalled]), token, bearer)):
            with self.subTest(settings=settings):
                reply = self.delegate({"input": INPUT}, core, settings, headers)
                self.assertEqual((reply.status_code, reply.json()["detail"]), (502, failed))
        self.assertTrue(stalled.closed)
        self.assertIn("A delegate round failed", "".join(logged))
        self.assertNotIn("s3cr3t-token-value", "".join(logged))

    def test_delegate_needs_a_core_and_limits_its_body_and_rounds(self):
        app, _, _ = make_app(core=None)
        with TestClient(app) as client:
            missing = client.post(DELEGATE, json={"input": INPUT})
        self.assertEqual((missing.status_code, missing.json()["detail"]), (503, "The delegation core is not available"))
        big = [{"role": "user", "content": [{"type": "input_text", "text": "x" * 2048}]}]
        reply = self.delegate({"input": big}, settings=LiveSettings(delegate_body_bytes=1024))
        self.assertEqual((reply.status_code, self.core.requests), (413, []))
        go = asyncio.Event()
        core = FakeAdapter(streams=[Stream(until=go, then=response("resp_1"))])
        app, _, _ = make_app(LiveSettings(max_delegate_rounds=1), core=core.core)

        async def rounds():
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1") as http:
                first = asyncio.create_task(http.post(DELEGATE, json={"input": INPUT}))
                while not core.requests:
                    await asyncio.sleep(0.005)
                async with asyncio.timeout(5):  # a round over the cap that runs would wait here for ``go``
                    over = await http.post(DELEGATE, json={"input": INPUT})
                go.set()
                return over, await first, await http.post(DELEGATE, json={"input": INPUT})

        over, done, again = asyncio.run(rounds())
        cap = "Too many delegate rounds: at most 1 at once"
        self.assertEqual((over.status_code, over.json()["detail"]), (429, cap))
        self.assertEqual((done.status_code, again.status_code, len(core.requests)), (200, 200, 2))


class SecurityTests(unittest.TestCase):
    def test_a_configured_token_is_required_on_every_route(self):
        app, _, _ = make_app(LiveSettings(api_token="s3cret"))
        bearer = {"Authorization": "Bearer s3cret"}
        with TestClient(app) as client:
            with self.assertRaises(WebSocketDisconnect), client.websocket_connect(SESSIONS) as refused:
                refused.send_json(START)
                refused.receive_json()
            with client.websocket_connect(SESSIONS, headers=bearer) as primary:
                primary.send_json(START)
                session_id = primary.receive_json()["session"]["id"]
                attach = f"{SESSIONS}/{session_id}/attach"
                with self.assertRaises(WebSocketDisconnect), client.websocket_connect(attach) as refused:
                    refused.receive_json()
                with client.websocket_connect(attach, headers=bearer) as observer:
                    observer.send_json(MUTE)
                    self.assertEqual(observer.receive_json()["type"], "session.input_audio.muted")
                routes = (("POST", SESSIONS), ("GET", f"{SESSIONS}/{session_id}/history"))
                for method, path in (*routes, ("GET", "/v1/live/info"), ("POST", DELEGATE)):
                    for header in ("", "Bearer é".encode("latin-1"), b"Bearer \xff\xfe"):
                        with self.subTest(route=path, header=header):
                            reply = client.request(
                                method, path, headers={"Authorization": header}, json={"input": INPUT}
                            )
                            self.assertEqual(reply.status_code, 401)
            self.assertEqual(client.get("/v1/live/info", headers=bearer).status_code, 200)
            self.assertEqual(client.post(DELEGATE, json={"input": INPUT}, headers=bearer).status_code, 200)

    def test_cross_site_origins_and_unknown_host_names_are_refused(self):
        app, _, engines = make_app()
        evil = {"Origin": "https://evil.example"}
        # A page on another site whose name resolves to this server (DNS rebinding): its origin matches its host.
        rebound = {"Host": "attacker.example:7860", "Origin": "http://attacker.example:7860"}
        with TestClient(app) as client, client.websocket_connect(SESSIONS) as ws:
            ws.send_json(START)
            session_id = ws.receive_json()["session"]["id"]
            routes = (("POST", SESSIONS), ("GET", f"{SESSIONS}/{session_id}/history"))
            for headers in (evil, rebound):
                for method, path in (*routes, ("GET", "/v1/live/info"), ("POST", DELEGATE)):
                    with self.subTest(route=f"{method} {path}", headers=headers):
                        reply = client.request(method, path, headers=headers, json={"input": INPUT})
                        self.assertEqual(reply.status_code, 403)
                for path in (SESSIONS, f"{SESSIONS}/{session_id}/attach"):
                    with (
                        self.assertRaises(WebSocketDisconnect),
                        client.websocket_connect(path, headers=headers) as refused,
                    ):
                        refused.send_json(START)
                        refused.receive_json()
            self.assertIn("allowed_hosts", client.get("/v1/live/info", headers=rebound).json()["detail"])
        self.assertEqual(len(engines), 1)

    def test_without_a_token_only_local_or_allowed_hosts_are_served(self):
        token, local = LiveSettings(api_token="secret"), ("127.0.0.1", "10.0.0.5:8000", "[::1]:8000", "localhost:7860")
        for host, settings, headers, status in (
            *((host, None, {}, 200) for host in (*local, "LOCALHOST")),
            ("Voice.Lab:8000", LiveSettings(allowed_hosts=("voice.lab",)), {}, 200),
            ("evil.example", None, {}, 403),
            ("evil.example", token, {"Authorization": "Bearer secret"}, 200),
            ("evil.example", token, {}, 401),
        ):
            with self.subTest(host=host, settings=settings):
                app, _, _ = make_app(settings)
                with TestClient(app) as client:
                    self.assertEqual(client.get("/v1/live/info", headers={"host": host, **headers}).status_code, status)


class MountTests(unittest.TestCase):
    def test_the_settings_come_from_the_example_and_the_environment(self):
        limits = {"max_sessions": 3, "max_session_seconds": 60, "max_appends_per_minute": 5, "max_delegate_rounds": 1}
        values = {**limits, "ice_urls": ["stun:stun.example:3478", " "], "allowed_hosts": [" Voice.Lab ", ""]}
        with mock.patch.dict(os.environ, {"LIVE_API_TOKEN": " s3cr3t-token \n", "UVICORN_WORKERS": "1"}, clear=True):
            settings = LiveSettings.from_env(values)
        lists = {"ice_urls": ("stun:stun.example:3478",), "allowed_hosts": ("voice.lab",)}
        self.assertEqual(settings, LiveSettings(api_token="s3cr3t-token", **limits, **lists))
        self.assertNotIn("s3cr3t-token", repr(settings))
        with mock.patch.dict(os.environ, {"LIVE_API_TOKEN": "   "}, clear=True):
            self.assertEqual(LiveSettings.from_env(), LiveSettings())
            self.assertEqual(LiveSettings.from_env({"max_sessions": 2}), LiveSettings(max_sessions=2))
        for value in ("many", "0"):
            with mock.patch.dict(os.environ, {"UVICORN_WORKERS": value}), self.assertRaisesRegex(ValueError, "UVICORN"):
                mount_live(FastAPI(), factory=lambda *args: None)

    def test_no_token_warns_and_several_workers_serve_no_route(self):
        logged = []
        sink = logger.add(lambda message: logged.append(message.record["message"]), level="WARNING")
        self.addCleanup(logger.remove, sink)
        for token in ("", "secret"):
            with mock.patch.dict(os.environ, {"LIVE_API_TOKEN": token, "UVICORN_WORKERS": "1"}):
                mount_live(FastAPI(), factory=lambda *args: None)
        app = FastAPI()
        with mock.patch.dict(os.environ, {"UVICORN_WORKERS": "2"}):
            self.assertIsNone(mount_live(app, factory=lambda *args: None))
        with TestClient(app) as client:
            self.assertEqual(client.post(SESSIONS, json={}).status_code, 404)
        [warning, error] = logged
        self.assertIn("LIVE_API_TOKEN is not set", warning)
        self.assertIn("POST /v1/live/delegate", warning)
        self.assertIn("UVICORN_WORKERS=2", error)

    def test_a_disabled_or_unknown_path_is_a_404_not_the_web_page(self):
        disabled = FastAPI()
        with mock.patch.dict(os.environ, {"LIVE_ENABLED": "false"}):
            mount_live(disabled)
        enabled, _, _ = make_app()
        for app, path in ((disabled, "/v1/live/info"), (disabled, f"{SESSIONS}/x/y"), (enabled, "/v1/live/nothing")):

            @app.get("/{path:path}")
            async def web_page(path: str):  # the web app's catch-all, added after the live routes
                return {"page": "index.html"}

            with self.subTest(path=path), TestClient(app) as client:
                self.assertEqual(client.get(path).status_code, 404)
                self.assertEqual(client.get("/somewhere").json(), {"page": "index.html"})


class WebRTCTests(unittest.IsolatedAsyncioTestCase):
    async def connect(self, **hooks):
        app, _, self.engines = make_app(**hooks)
        self.client = RTCPeerConnection()
        self.client.addTrack(AudioStreamTrack())
        self.channel = self.client.createDataChannel("oai-events")
        self.seen, self.seen_at_close = [], None
        self.channel.on("message", lambda raw: self.seen.append(json.loads(raw)))

        @self.channel.on("close")
        def on_close():
            self.seen_at_close = [event["type"] for event in self.seen]

        self.addAsyncCleanup(self.client.close)
        await self.client.setLocalDescription(await self.client.createOffer())
        self.http = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1")
        self.addAsyncCleanup(self.http.aclose)
        offer = {"type": "webrtc", "sdp": self.client.localDescription.sdp}
        answer = await self.http.post(SESSIONS, json={"session": START["session"], "transport": offer})
        self.assertEqual(answer.status_code, 201, answer.text)
        body = answer.json()
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
        """Wait until the session has left the gateway and its history is not served."""
        async with asyncio.timeout(15):
            while (await self.http.get(self.history)).status_code != 404:
                await asyncio.sleep(0.05)

    async def test_a_session_runs_over_the_event_channel(self):
        body = await self.connect()
        self.assertEqual((set(body), set(body["session"])), ({"session", "transport"}, {"id"}))
        self.assertEqual(set(body["transport"]), {"type", "sdp"})
        started = (await self.event("session.started"))["session"]
        self.assertEqual((started["id"], started["instructions"]), (body["session"]["id"], "Be brief."))
        self.channel.send(json.dumps(SIGNALLING))
        self.channel.send(json.dumps({"type": "session.input_audio.mute", "event_id": "mute"}))
        self.assertEqual((await self.event("session.input_audio.muted"))["client_event_id"], "mute")
        self.channel.send(json.dumps({"type": "session.close", "event_id": "close"}))
        await self.gone()
        await self.until(lambda: self.seen_at_close is not None)
        self.assertIn("session.closed", self.seen_at_close)
        self.assertEqual(self.seen_at_close.count("session.input_audio.muted"), 1)
        self.assertNotIn("error", self.seen_at_close)
        self.assertEqual((self.engines[-1].muted, self.engines[-1].closed), (True, True))

    async def test_a_peer_that_leaves_ends_its_session_as_a_hang_up(self):
        lines = []
        self.addCleanup(logger.remove, logger.add(lines.append, level="INFO"))
        await self.connect()
        await self.event("session.started")
        self.channel.close()
        await self.gone()
        self.assertTrue(self.engines[-1].closed)
        self.assertIn("reason=remote_hangup", next(line for line in lines if " closes: " in line))

    async def test_an_early_command_gets_its_own_error_if_the_start_fails(self):
        arrived = asyncio.Event()

        async def factory(config, transport, ice_servers):
            peer = LivePeer.from_ice_servers(ice_servers)

            @peer.event_handler("client-event")
            async def on_client_event(_peer, event):
                arrived.set()

            adapter = FakeAdapter(peer)
            adapter.hold["start"] = arrived
            adapter.fail["start"] = RuntimeError("the voice pipeline could not be built")
            return SessionParts(frontend=adapter, connection=peer)

        await self.connect(factory=factory)
        await self.until(lambda: self.channel.readyState == "open")
        self.channel.send(json.dumps(MUTE))
        await self.until(lambda: self.seen_at_close is not None)
        self.assertEqual(self.seen_at_close, ["error", "error", "session.closed"])
        errors = [(event["error"]["code"], event.get("client_event_id")) for event in self.seen[:2]]
        self.assertEqual(errors, [("pipeline_error", None), ("session_closed", "m")])

    async def test_a_peer_that_never_connects_is_closed_at_the_deadline(self):
        for status, connected, reasons in (
            ("active", False, ["connection_lost"]),
            ("active", True, []),
            ("starting", True, ["connection_lost"]),
        ):
            session = mock.Mock(status=status, close=mock.AsyncMock())
            await expire_unconnected(session, mock.Mock(is_connected=mock.Mock(return_value=connected)), timeout=0)
            self.assertEqual([call.args[0] for call in session.close.await_args_list], reasons, (status, connected))


async def adapter_factory(config, transport, connection):
    return FakeAdapter().parts()


class CountingCore:
    """The root's core hook as a coroutine function: it counts its builds, and may fail or take time."""

    def __init__(self, fails=0, delay=0.0, core=None):
        self.builds, self.fails, self.delay, self.core = 0, fails, delay, core or FakeAdapter()

    async def __call__(self):
        self.builds += 1
        await asyncio.sleep(self.delay)
        if self.fails:
            self.fails -= 1
            raise RuntimeError("/srv/settings.yaml could not be read")
        return self.core


class GatewayTests(unittest.IsolatedAsyncioTestCase):
    async def start(self, gateway, config):
        session = await gateway.register(config, "websocket", None)
        self.client = Collector()
        session.subscribe(self.client.send, "primary")
        await session.start("start_1")
        self.addAsyncCleanup(session.close)
        return session

    async def test_responses_sessions_share_one_core_built_once(self):
        hook = CountingCore(delay=0.01)
        gateway = LiveGateway(adapter_factory, LiveSettings(), core=hook)
        await self.start(gateway, CLIENT)
        self.assertEqual(hook.builds, 0)
        first, second = await asyncio.gather(
            self.start(gateway, responses_config()), self.start(gateway, responses_config())
        )
        self.assertEqual(hook.builds, 1)
        self.assertIs(first.delegation.client, second.delegation.client)
        self.assertEqual(first.delegation.conversation, FakeAdapter.CONVERSATION)

    async def test_a_stored_session_attaches_to_its_conversation_through_the_one_core(self):
        hook = CountingCore()
        hook.core.kept = {"conv_kept"}
        gateway = LiveGateway(adapter_factory, LiveSettings(), core=hook)
        session = await self.start(gateway, responses_config(conversation="conv_kept", store=True))
        metadata = {"live_session_id": session.id, "live_store": "true"}
        self.assertEqual((hook.core.log, hook.core.metadata), (["update"], metadata))
        self.assertEqual(session.delegation.conversation, "conv_kept")

    async def test_a_core_build_failure_is_said_without_detail_and_retried(self):
        hook = CountingCore(fails=1)
        gateway = LiveGateway(adapter_factory, LiveSettings(), core=hook)
        session, client = await gateway.register(responses_config(), "websocket", None), Collector()
        session.subscribe(client.send, "primary")
        with self.assertRaises(ProtocolError) as refused:
            await session.start("start_1")
        self.assertEqual((refused.exception.code, session.status), ("server_error", "closed"))
        self.assertNotIn("settings.yaml", f"{refused.exception} {client.events}")
        await self.start(gateway, responses_config())
        self.assertEqual(hook.builds, 2)

    async def test_client_mode_sessions_need_no_core_not_even_at_shutdown(self):
        hook = CountingCore()
        gateway = LiveGateway(adapter_factory, LiveSettings(), core=hook)
        session = await self.start(gateway, CLIENT)
        await gateway.close_all()
        self.assertEqual((session.status, hook.builds, hook.core.log), ("closed", 0, []))
        gateway = LiveGateway(adapter_factory, LiveSettings())
        with self.assertRaises(ProtocolError) as refused:
            await gateway.register(responses_config(), "websocket", None)
        self.assertEqual(refused.exception.param, "session.delegation.type")
        self.assertEqual((gateway.sessions, gateway.starting), ({}, 0))
        await self.start(gateway, CLIENT)
        await gateway.close_all()

    async def test_shutdown_closes_the_core_once_after_every_session(self):
        for fails in (False, True):
            with self.subTest(fails=fails), mock.patch("gateways.live.gateway.gateway.logger") as logged:
                core, closes = FakeAdapter(), []
                gateway = LiveGateway(adapter_factory, LiveSettings(), core=CountingCore(core=core))
                sessions = [await self.start(gateway, responses_config()), await self.start(gateway, CLIENT)]

                async def aclose(sessions=sessions, closes=closes, fails=fails):
                    closes.append([session.status for session in sessions])
                    if fails:
                        raise RuntimeError("the tool bridge did not stop")

                core.aclose = aclose
                await gateway.close_all()
                await gateway.close_all()
                self.assertEqual(closes, [["closed", "closed"]])
                self.assertEqual([session.status for session in sessions], ["closed", "closed"])
                closed = [event["reason"] for event in self.client.events if event["type"] == "session.closed"]
                self.assertEqual(closed, ["connection_lost"])
                self.assertEqual(logged.exception.call_count, int(fails))
                if fails:
                    self.assertIn("could not be closed", logged.exception.call_args.args[0])

    async def test_parts_being_built_count_against_the_session_limit(self):
        release, built = asyncio.Event(), []

        async def factory(config, transport, connection):
            if not built:
                built.append("slow")
                await release.wait()
            return FakeAdapter().parts()

        gateway = LiveGateway(factory, LiveSettings(max_sessions=2))
        slow = asyncio.create_task(gateway.register(CLIENT, "websocket", None))
        await asyncio.sleep(0.01)
        quick = await gateway.register(CLIENT, "websocket", None)  # not held up: parts are built outside the lock
        with self.assertRaises(ProtocolError) as full:
            await gateway.register(CLIENT, "websocket", None)
        self.assertEqual(full.exception.code, "rate_limit_exceeded")
        release.set()
        for session in (await slow, quick):
            await session.close()


if __name__ == "__main__":
    unittest.main()
