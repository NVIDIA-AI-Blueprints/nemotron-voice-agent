# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D103, D107

"""Gateway edge cases, one test each: frames, close reasons, WebRTC readiness, settings, codecs and fault reports."""

import asyncio
import json
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import httpx
import numpy as np
from aiortc import RTCPeerConnection, RTCSessionDescription
from aiortc.mediastreams import AudioStreamTrack
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from loguru import logger
from starlette.websockets import WebSocketDisconnect

from live.gateway import LiveGateway, LiveSettings
from live.gateway.settings import positive_setting
from live.gateway.websocket import close_reason, run_primary_websocket
from live.media.audio import ALAW, SAMPLE_CODECS, ULAW, ALaw, LinearPcm, MuLaw, WireCodec
from live.media.connection import LiveWebRTCConnection
from live.mount import mount_live
from live.protocol import AudioFormat, ProtocolError, SessionConfig
from live.session import LiveProtocolSession
from tests.unit.live_helpers import Collector, FakeAdapter, responses_config

START = {"type": "session.start", "event_id": "start_1", "session": {"model": "live-1"}}


class ScriptedSocket:
    """The server side of a primary WebSocket that receives the scripted frames, then waits."""

    headers: dict = {}  # noqa: RUF012

    def __init__(self, *frames):
        self.frames, self.sent, self.closed_with = list(frames), [], None

    async def accept(self):
        return None

    async def receive(self):
        if self.frames:
            return self.frames.pop(0)
        await asyncio.Event().wait()

    async def send_json(self, message):
        self.sent.append(message)

    async def close(self, code=1000):
        self.closed_with = self.closed_with or code


def make_app(factory=None):
    engines = []

    async def default_factory(config, transport, connection):
        engines.append(FakeAdapter())
        return engines[-1].parts()

    app = FastAPI()
    core = FakeAdapter()  # the gateway's one delegation core
    gateway = mount_live(app, factory=factory or default_factory, core=core.core)
    return app, gateway, engines


# --------------------------------------------------------------------------------------------- the WebSocket edge
class BinaryFrameTests(unittest.TestCase):
    def test_a_binary_frame_gets_an_error_event_and_the_session_stays_open(self):
        app, gateway, _ = make_app()
        with TestClient(app) as client, client.websocket_connect("/v1/live/sessions") as ws:
            ws.send_json(START)
            self.assertEqual(ws.receive_json()["type"], "session.started")
            ws.send_bytes(b"\x00\x01")
            error = ws.receive_json()["error"]
            self.assertIn("Binary frames are not supported", error["message"])
            ws.send_json({"type": "session.input_audio.mute", "event_id": "m"})
            self.assertEqual(ws.receive_json()["type"], "session.input_audio.muted")
            [session] = gateway.sessions.values()
            self.assertEqual(session.status, "active")

    def test_a_binary_first_frame_is_no_session_start(self):
        app, _, engines = make_app()
        with TestClient(app) as client, client.websocket_connect("/v1/live/sessions") as ws:
            ws.send_bytes(b"{}")
            self.assertIn("session.start must be the first message", ws.receive_json()["error"]["message"])
            with self.assertRaises(WebSocketDisconnect) as closed:
                ws.receive_json()
            self.assertEqual(closed.exception.code, 1008)
        self.assertEqual(engines, [])


class CloseReasonTests(unittest.TestCase):
    def test_a_clean_websocket_close_is_a_remote_hangup_and_any_other_a_lost_connection(self):
        self.assertEqual([close_reason(code) for code in (1000, 1001)], ["remote_hangup", "remote_hangup"])
        self.assertEqual([close_reason(code) for code in (1006, 1011, None)], ["connection_lost"] * 3)

    def test_a_websocket_client_that_closes_ends_its_session_by_its_close_code(self):
        # Driven frame by frame: Starlette's TestClient may cancel the handler as it leaves, which a server does not.
        for code, reason in ((1000, "remote_hangup"), (1001, "remote_hangup"), (1006, "connection_lost")):
            with self.subTest(code=code):
                _, gateway, engines = make_app()
                ws = ScriptedSocket(
                    {"type": "websocket.receive", "text": json.dumps(START)},
                    {"type": "websocket.disconnect", "code": code},
                )
                asyncio.run(run_primary_websocket(gateway, ws))
                closed = [event for event in ws.sent if event["type"] == "session.closed"]
                self.assertEqual([event["reason"] for event in closed], [reason])
                self.assertEqual((engines[0].closed, gateway.sessions), (True, {}))

    def test_server_shutdown_closes_sessions_as_connection_lost(self):
        app, gateway, _ = make_app()

        async def run():
            async with app.router.lifespan_context(app):
                session = await gateway.register(SessionConfig(model="live-1"), "websocket", None)
                collector = Collector()
                session.subscribe(collector.send, "primary")
                await session.start()
            return collector

        collector = asyncio.run(run())
        closed = next(event for event in collector.events if event["type"] == "session.closed")
        self.assertEqual(closed["reason"], "connection_lost")


class WebRTCHangUpTests(unittest.IsolatedAsyncioTestCase):
    """Real SDP, DTLS and SCTP on loopback: a client that leaves without session.close."""

    async def connect(self):
        self.app, self.gateway, self.engines = make_app()
        self.client = RTCPeerConnection()
        self.client.addTrack(AudioStreamTrack())
        self.channel = self.client.createDataChannel("oai-events")
        self.events: asyncio.Queue = asyncio.Queue()
        self.channel.on("message", lambda raw: self.events.put_nowait(json.loads(raw)))
        self.addAsyncCleanup(self.client.close)
        await self.client.setLocalDescription(await self.client.createOffer())
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=self.app), base_url="http://test") as http:
            response = await http.post(
                "/v1/live/sessions",
                json={
                    "session": {"model": "live-1"},
                    "transport": {"type": "webrtc", "sdp": self.client.localDescription.sdp},
                },
            )
        self.assertEqual(response.status_code, 201, response.text)
        await self.client.setRemoteDescription(
            RTCSessionDescription(sdp=response.json()["transport"]["sdp"], type="answer")
        )
        async with asyncio.timeout(15):
            while (await self.events.get())["type"] != "session.started":
                pass
        [self.session] = self.gateway.sessions.values()
        self.reasons = []
        original = self.session.close

        async def noting(reason="close_requested", client_event_id=None, *, asked_by="the gateway"):
            self.reasons.append(reason)
            await original(reason, client_event_id, asked_by=asked_by)

        self.session.close = noting

    async def gone(self):
        async with asyncio.timeout(15):
            while self.gateway.sessions:
                await asyncio.sleep(0.05)

    async def test_a_peer_that_hangs_up_without_session_close_ends_the_session_and_leaves_the_registry(self):
        await self.connect()
        await self.client.close()
        await self.gone()
        self.assertEqual(self.reasons[0], "remote_hangup")
        self.assertEqual((self.session.status, self.engines[0].closed), ("closed", True))

    async def test_a_session_ends_when_only_its_data_channel_closes(self):
        await self.connect()
        self.channel.close()
        await self.gone()
        self.assertEqual(self.reasons[0], "remote_hangup")


# --------------------------------------------------------------------------------------------- settings and mount
class SettingsTests(unittest.TestCase):
    def test_an_empty_setting_is_the_default_and_a_bad_one_stops_startup_with_its_name(self):
        with patch.dict(os.environ, {"LIVE_MAX_SESSIONS": "", "LIVE_MAX_SESSION_SECONDS": " 60 "}):
            settings = LiveSettings.from_env()
        self.assertEqual((settings.max_sessions, settings.max_session_seconds), (LiveSettings.max_sessions, 60))
        for value in ("eight", "0", "-3", "2.5"):
            with (
                self.subTest(value=value),
                patch.dict(os.environ, {"LIVE_MAX_SESSIONS": value}),
                self.assertRaisesRegex(ValueError, "LIVE_MAX_SESSIONS"),
            ):
                LiveSettings.from_env()
        with patch.dict(os.environ, {"LIVE_MAX_SESSIONS": "eight"}), self.assertRaises(ValueError):
            mount_live(FastAPI(), factory=lambda *args: None)  # the server does not start
        self.assertEqual(positive_setting("LIVE_UNSET_FOR_TEST", 7), 7)

    def test_the_defaults_are_written_once_as_the_dataclass_fields(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(LiveSettings.from_env(), LiveSettings())


class WorkerTests(unittest.TestCase):
    def test_several_workers_serve_no_live_route_and_say_why(self):
        app = FastAPI()
        with patch.dict(os.environ, {"UVICORN_WORKERS": "2"}), patch("live.mount.logger") as log:
            self.assertIsNone(mount_live(app, factory=lambda *args: None))
        self.assertIn("UVICORN_WORKERS=2", log.error.call_args.args[0])
        with TestClient(app) as client:
            self.assertEqual(client.post("/v1/live/sessions", json={}).status_code, 404)


class NotFoundTests(unittest.TestCase):
    def test_a_disabled_or_unknown_live_path_is_a_404_never_the_web_page(self):
        disabled = FastAPI()
        with patch.dict(os.environ, {"LIVE_ENABLED": "false"}):
            mount_live(disabled)
        enabled, _, _ = make_app()
        for app in (disabled, enabled):

            @app.get("/{path:path}")
            async def web_page(path: str):  # the web app's catch-all, added after the live routes
                return {"page": "index.html"}

            with TestClient(app) as client:
                for path in ("/v1/live/info" if app is disabled else "/v1/live/nothing", "/v1/live/sessions/x/y"):
                    with self.subTest(app="disabled" if app is disabled else "enabled", path=path):
                        self.assertEqual(client.get(path).status_code, 404)
                self.assertEqual(client.get("/somewhere").json(), {"page": "index.html"})
        with TestClient(enabled) as client:
            self.assertEqual(client.get("/v1/live/info").status_code, 200)  # a real live route still answers


class TokenlessWarningTests(unittest.TestCase):
    def test_the_startup_warning_names_the_delegate_route(self):
        with patch.dict(os.environ, {"LIVE_API_TOKEN": ""}), patch("live.mount.logger") as log:
            mount_live(FastAPI(), factory=lambda *args: None)
        self.assertIn("POST /v1/live/delegate", log.warning.call_args.args[0])


# ------------------------------------------------------------------------------------------------------ G.711
SEGMENT_ENDS_ULAW = (0x3F, 0x7F, 0xFF, 0x1FF, 0x3FF, 0x7FF, 0xFFF, 0x1FFF)
SEGMENT_ENDS_ALAW = (0x1F, 0x3F, 0x7F, 0xFF, 0x1FF, 0x3FF, 0x7FF, 0xFFF)


def segment(value, ends):
    for index, end in enumerate(ends):
        if value <= end:
            return index
    return len(ends)


def reference_ulaw(sample):
    """ITU-T G.711 mu-law of one 16-bit sample: the CCITT reference (Sun g711.c, linear2ulaw on 14 bits)."""
    pcm = sample >> 2
    pcm, mask = (-pcm, 0x7F) if pcm < 0 else (pcm, 0xFF)
    pcm = min(pcm, 8159) + 33
    seg = segment(pcm, SEGMENT_ENDS_ULAW)
    return (0x7F if seg >= 8 else (seg << 4) | ((pcm >> (seg + 1)) & 0xF)) ^ mask


def reference_alaw(sample):
    """ITU-T G.711 A-law of one 16-bit sample: the CCITT reference (Sun g711.c, linear2alaw on 13 bits)."""
    pcm = sample >> 3
    pcm, mask = (pcm, 0xD5) if pcm >= 0 else (-pcm - 1, 0x55)
    seg = segment(pcm, SEGMENT_ENDS_ALAW)
    if seg >= 8:
        return 0x7F ^ mask
    return ((seg << 4) | ((pcm >> (1 if seg < 2 else seg)) & 0xF)) ^ mask


class G711Tests(unittest.TestCase):
    def test_g711_takes_8000_hz_without_an_explicit_rate(self):
        for kind in ("audio/pcmu", "audio/pcma"):
            self.assertEqual(AudioFormat(type=kind).rate, 8000)
            self.assertEqual(SessionConfig(model="m", audio={"format": {"type": kind}}).audio.format.rate, 8000)
        self.assertEqual(AudioFormat().rate, 24000)

    def test_the_encoders_match_the_g711_reference_on_every_16_bit_sample(self):
        samples = np.arange(-32768, 32768, dtype="<i2")
        for codec, reference in ((MuLaw(), reference_ulaw), (ALaw(), reference_alaw)):
            with self.subTest(codec=type(codec).__name__):
                encoded = np.frombuffer(codec.encode(samples.tobytes()), dtype=np.uint8)
                expected = np.array([reference(int(sample)) for sample in samples], dtype=np.uint8)
                self.assertEqual(int(np.count_nonzero(encoded != expected)), 0)

    def test_known_g711_codes_and_a_lossless_round_trip_of_every_code(self):
        known = {MuLaw(): {0: 0xFF, 32767: 0x80, -32768: 0x00}, ALaw(): {0: 0xD5, -1: 0x55, 32767: 0xAA}}
        for codec, codes in known.items():
            for sample, code in codes.items():
                self.assertEqual(codec.encode(np.array([sample], dtype="<i2").tobytes())[0], code, (codec, sample))
        for codec, table in ((MuLaw(), ULAW), (ALaw(), ALAW)):
            decoded = codec.decode(bytes(range(256)))
            self.assertEqual(codec.decode(codec.encode(decoded)), decoded)
            self.assertEqual(decoded, table.tobytes())

    def test_each_wire_format_has_its_own_sample_codec(self):
        self.assertEqual(SAMPLE_CODECS, {"audio/pcm": LinearPcm, "audio/pcmu": MuLaw, "audio/pcma": ALaw})
        self.assertIsInstance(WireCodec(AudioFormat(type="audio/pcmu")).samples, MuLaw)
        self.assertIsInstance(WireCodec().samples, LinearPcm)


# ------------------------------------------------------------------------------------------------------- WebRTC
class ReadinessTests(unittest.IsolatedAsyncioTestCase):
    async def test_once_a_pipeline_transport_runs_on_the_connection_pipecats_own_check_decides(self):
        connection = LiveWebRTCConnection()
        self.addAsyncCleanup(connection._close)
        peer = connection._pc
        self.addCleanup(setattr, connection, "_pc", peer)  # cleanups run last first: the real peer is closed
        connection._pc = SimpleNamespace(connectionState="connected")  # the peer connected before the pipeline
        self.assertTrue(connection.is_connected())  # early commands are dispatched as they arrive
        connection.add_event_handler("connected", lambda *_: None)  # a pipeline transport registers itself
        # Pipecat's transport then calls connect(), which runs its setup (it skips it while is_connected() is True).
        self.assertFalse(connection.is_connected())


class RefusedWebRTCTests(unittest.IsolatedAsyncioTestCase):
    async def test_a_refused_session_never_answers_its_offer_so_no_ice_task_is_left(self):
        initialized = []

        async def refuse(config, transport, connection):
            raise ProtocolError("No session for you", "server_error")

        with patch.object(LiveWebRTCConnection, "initialize", lambda *args: initialized.append(args)):
            app = FastAPI()
            mount_live(app, factory=refuse)
            client = RTCPeerConnection()
            client.addTrack(AudioStreamTrack())
            client.createDataChannel("oai-events")
            self.addAsyncCleanup(client.close)
            await client.setLocalDescription(await client.createOffer())
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as http:
                response = await http.post(
                    "/v1/live/sessions",
                    json={
                        "session": {"model": "m"},
                        "transport": {"type": "webrtc", "sdp": client.localDescription.sdp},
                    },
                )
        self.assertEqual(response.status_code, 503)
        self.assertEqual(initialized, [])


# ------------------------------------------------------------------------------------------- sessions and logs
class LogContextTests(unittest.IsolatedAsyncioTestCase):
    async def test_a_sessions_log_lines_carry_its_id_and_each_close_is_one_line(self):
        records = []
        sink = logger.add(lambda message: records.append(message.record), level="DEBUG")
        self.addCleanup(logger.remove, sink)

        class Faulty(FakeAdapter):
            async def set_muted(self, muted):
                raise RuntimeError("the pipeline is stuck")

        faulty = Faulty()
        session = LiveProtocolSession(responses_config(), "websocket", faulty.parts(), core=faulty.core)
        connection = session.subscribe(Collector().send, "primary")
        await session.start()
        await session.handle({"type": "session.input_audio.mute", "event_id": "m"}, connection)
        await session.close("close_requested", asked_by="the client (primary connection)")
        fault = next(record for record in records if "failed" in record["message"] and record["exception"])
        self.assertEqual(fault["extra"].get("stream_id"), session.id)
        [closing] = [record for record in records if " closes: " in record["message"]]
        self.assertEqual(closing["extra"].get("stream_id"), session.id)
        self.assertRegex(
            closing["message"],
            rf"session={session.id} closes: reason=close_requested, lifetime=\d+\.\d s, "
            r"asked by the client \(primary connection\)",
        )


class FaultTests(unittest.IsolatedAsyncioTestCase):
    async def test_a_parts_value_error_is_a_server_fault_not_the_clients_mistake(self):
        class Faulty(FakeAdapter):
            async def set_muted(self, muted):
                raise ValueError("an internal value went wrong")

        faulty = Faulty()
        session = LiveProtocolSession(responses_config(), "websocket", faulty.parts(), core=faulty.core)
        primary = Collector()
        connection = session.subscribe(primary.send, "primary")
        await session.start()
        self.addAsyncCleanup(session.close)
        await session.handle({"type": "session.input_audio.mute", "event_id": "m"}, connection)
        error = (await primary.wait_for("error"))["error"]
        self.assertEqual(
            (error["type"], error["code"], error["client_event_id"]), ("server_error", "server_error", "m")
        )
        self.assertNotIn("internal value", error["message"])

    async def test_a_factory_refuses_only_with_a_protocol_error(self):
        async def http_refusal(config, transport, connection):
            raise HTTPException(429, "a factory's own HTTP answer")

        gateway = LiveGateway(http_refusal, LiveSettings())
        with self.assertRaises(ProtocolError) as refused:
            await gateway.register(SessionConfig(model="m"), "websocket", None)
        self.assertEqual(refused.exception.code, "server_error")


if __name__ == "__main__":
    unittest.main()
