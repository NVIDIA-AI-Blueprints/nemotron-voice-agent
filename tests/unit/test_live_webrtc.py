# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D100, D101, D102, D103

import asyncio
import json
import unittest

import httpx
from aiortc import RTCPeerConnection, RTCSessionDescription
from aiortc.mediastreams import AudioStreamTrack
from fastapi import FastAPI

from live.mount import mount_live
from tests.unit.live_helpers import FakeEngine

SESSION = {"model": "live-1", "instructions": "Be brief."}


class WebRTCLoopbackTests(unittest.IsolatedAsyncioTestCase):
    """Real SDP, DTLS, SCTP and media on loopback, through the real HTTP endpoint."""

    async def connect(self, label, session=None):
        self.engines = []

        async def factory(config, transport, connection):
            engine = FakeEngine()
            self.engines.append(engine)
            return engine

        self.app = FastAPI()
        self.gateway = mount_live(self.app, factory)
        self.client = RTCPeerConnection()
        self.client.addTrack(AudioStreamTrack())
        self.channel = self.client.createDataChannel(label)
        self.incoming: asyncio.Queue = asyncio.Queue()
        self.order: list[str] = []
        self.channel_closed = asyncio.Event()

        @self.channel.on("message")
        def on_message(raw):
            event = json.loads(raw)
            self.incoming.put_nowait(event)
            if event["type"] == "session.closed":
                self.order.append("final_event")

        @self.channel.on("close")
        def on_close():
            self.order.append("channel_closed")
            self.channel_closed.set()

        self.addAsyncCleanup(self.client.close)
        await self.client.setLocalDescription(await self.client.createOffer())
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=self.app), base_url="http://test") as http:
            response = await http.post(
                "/v1/live/sessions",
                json={
                    "session": session or SESSION,
                    "transport": {"type": "webrtc", "sdp": self.client.localDescription.sdp},
                },
            )
        self.assertEqual(response.status_code, 201, response.text)
        body = response.json()
        await self.client.setRemoteDescription(RTCSessionDescription(sdp=body["transport"]["sdp"], type="answer"))
        return body

    async def next_event(self, kind, timeout=15):
        async with asyncio.timeout(timeout):
            while True:
                event = await self.incoming.get()
                if event["type"] == kind:
                    return event

    async def test_session_lifecycle_over_the_event_channel(self):
        # Clients name the event channel differently: "oai-events", or an empty label.
        for label in ("oai-events", ""):
            with self.subTest(label=label):
                body = await self.connect(label)
                self.assertEqual(body["session"]["status"], "active")
                self.assertEqual(body["sent_session_config"]["instructions"], "Be brief.")
                started = await self.next_event("session.started")
                self.assertEqual(started["session"]["id"], body["session"]["id"])
                self.channel.send(json.dumps({"type": "session.input_audio.mute", "event_id": "mute"}))
                self.assertEqual((await self.next_event("session.input_audio.muted"))["client_event_id"], "mute")
                self.assertTrue(self.engines[-1].muted)
                self.channel.send(json.dumps({"type": "session.close", "event_id": "close"}))
                await self.next_event("session.closed")
                async with asyncio.timeout(15):
                    await self.channel_closed.wait()
                # The final event must reach the client before the channel closes.
                self.assertEqual(self.order, ["final_event", "channel_closed"])
                self.assertTrue(self.engines[-1].closed)
                for _ in range(100):
                    if not self.gateway.sessions:
                        break
                    await asyncio.sleep(0.05)
                self.assertEqual(self.gateway.sessions, {})
                await self.client.close()

    async def test_server_events_follow_the_data_channel_permissions(self):
        session = {
            **SESSION,
            "client": {"data_channel": {"allowed_server_events": ["session.started", "session.closed"]}},
        }
        await self.connect("oai-events", session)
        await self.next_event("session.started")
        self.channel.send(json.dumps({"type": "session.input_audio.mute"}))
        self.channel.send(json.dumps({"type": "session.close"}))
        # The mute acknowledgment is filtered out; the next event the client sees is the final one.
        self.assertEqual((await self.incoming.get())["type"], "session.closed")

    async def test_a_bad_offer_is_a_400_and_leaves_no_session(self):
        app = FastAPI()
        gateway = mount_live(app, lambda *args: None)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as http:
            response = await http.post(
                "/v1/live/sessions",
                json={"session": SESSION, "transport": {"type": "webrtc", "sdp": "not an sdp"}},
            )
        self.assertEqual(response.status_code, 400)
        self.assertEqual(gateway.sessions, {})


if __name__ == "__main__":
    unittest.main()
