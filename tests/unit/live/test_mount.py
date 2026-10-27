# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D103, D105, D107

"""The mount: one shared core client, the session registry and the WebRTC peers' ICE servers.

The gateway builds one delegation core client from the ``core`` hook on first use, shares it with every ``responses``
session and console round, and closes it once at shutdown, only if it was built. Each WebRTC peer gets the root's ICE
servers, TURN credentials included.
"""

import asyncio
import unittest
from unittest.mock import patch

import httpx
from fastapi import FastAPI

from live.gateway import LiveGateway, LiveSettings
from live.gateway import webrtc as webrtc_edge
from live.mount import mount_live
from live.protocol import ProtocolError, SessionConfig
from tests.unit.live_helpers import Collector, FakeAdapter, responses_config

OFFER = "v=0\r\no=- 1 1 IN IP4 127.0.0.1\r\ns=-\r\nt=0 0\r\nm=audio 9 UDP/TLS/RTP/SAVPF 111\r\n"


class CountingCore:
    """The root's core hook, a coroutine function as the root's is: it counts the builds, and may fail or take time."""

    def __init__(self, fails=0, delay=0.0, core=None):
        self.builds, self.fails, self.delay = 0, fails, delay
        self.core = core or FakeAdapter()

    async def __call__(self):
        self.builds += 1
        await asyncio.sleep(self.delay)
        if self.fails:
            self.fails -= 1
            raise RuntimeError("/srv/core.yaml could not be read")
        return self.core


class ClosingCore(FakeAdapter):
    """The one core's client, which notes at each close the status of the sessions it watches; its close may fail."""

    def __init__(self, fails=False):
        super().__init__()
        self.fails = fails
        self.watched = []  # the sessions whose status it notes when it is closed
        self.closes: list[list[str]] = []  # at each close, the status of each watched session

    async def aclose(self):
        self.closes.append([session.status for session in self.watched])
        if self.fails:
            raise RuntimeError("the tool bridge did not stop")


async def adapter_factory(config, transport, connection):
    return FakeAdapter().parts()


class OneCoreTests(unittest.IsolatedAsyncioTestCase):
    async def start(self, gateway, config):
        session = await gateway.register(config, "websocket", None)
        session.subscribe(Collector().send, "primary")
        await session.start("start_1")
        self.addAsyncCleanup(session.close)
        return session

    async def test_client_mode_builds_no_core_and_every_responses_session_shares_one(self):
        hook = CountingCore(delay=0.01)
        gateway = LiveGateway(adapter_factory, LiveSettings(), core=hook)
        await self.start(gateway, SessionConfig(model="live-1"))
        self.assertEqual(hook.builds, 0)  # a client-mode session never asks for the core
        first, second = await asyncio.gather(  # two first uses at once wait for the one build
            self.start(gateway, responses_config()), self.start(gateway, responses_config())
        )
        self.assertEqual(hook.builds, 1)  # one core client per gateway, awaited once
        self.assertIs(first.delegation.client, second.delegation.client)
        self.assertEqual(first.delegation.conversation, FakeAdapter.CONVERSATION)  # opened on the one core

    async def test_a_core_that_cannot_be_built_fails_the_start_without_detail_and_is_tried_again(self):
        hook = CountingCore(fails=1)
        gateway = LiveGateway(adapter_factory, LiveSettings(), core=hook)
        session = await gateway.register(responses_config(), "websocket", None)
        client = Collector()
        session.subscribe(client.send, "primary")
        with self.assertRaises(ProtocolError) as refused:
            await session.start("start_1")
        self.assertEqual(refused.exception.code, "server_error")
        self.assertNotIn("core.yaml", str(refused.exception))
        self.assertEqual(session.status, "closed")
        self.assertNotIn("core.yaml", str(client.events))
        await self.start(gateway, responses_config())  # the next use builds it
        self.assertEqual(hook.builds, 2)

    async def test_a_server_with_no_core_refuses_a_responses_session_and_serves_client_mode(self):
        gateway = LiveGateway(adapter_factory, LiveSettings())
        with self.assertRaises(ProtocolError) as refused:
            await gateway.register(responses_config(), "websocket", None)
        self.assertEqual(refused.exception.param, "session.delegation.type")
        self.assertEqual((gateway.sessions, gateway.starting), ({}, 0))
        await self.start(gateway, SessionConfig(model="live-1"))

    async def test_server_shutdown_closes_the_built_core_once_after_every_session(self):
        core = ClosingCore()
        gateway = LiveGateway(adapter_factory, LiveSettings(), core=CountingCore(core=core))
        core.watched = [
            await self.start(gateway, responses_config()),
            await self.start(gateway, SessionConfig(model="live-1")),  # a client-mode session closes before it too
        ]
        await gateway.close_all()
        await gateway.close_all()  # a second shutdown has nothing left to close
        self.assertEqual(core.closes, [["closed", "closed"]])  # once, when every session had closed

    async def test_server_shutdown_builds_no_core_to_close_it(self):
        hook = CountingCore(core=ClosingCore())
        gateway = LiveGateway(adapter_factory, LiveSettings(), core=hook)
        session = await self.start(gateway, SessionConfig(model="live-1"))  # client mode never asks for the core
        await gateway.close_all()
        self.assertEqual((session.status, hook.builds, hook.core.closes), ("closed", 0, []))
        await LiveGateway(adapter_factory, LiveSettings()).close_all()  # a server with no core has none to close

    async def test_a_core_whose_close_fails_is_logged_and_the_shutdown_goes_on(self):
        core = ClosingCore(fails=True)
        gateway = LiveGateway(adapter_factory, LiveSettings(), core=CountingCore(core=core))
        session = await self.start(gateway, responses_config())
        with patch("live.gateway.gateway.logger") as logger:
            await gateway.close_all()  # returns: nothing is raised to the server's shutdown
        self.assertEqual((session.status, len(core.closes)), ("closed", 1))
        logger.exception.assert_called_once()
        self.assertIn("could not be closed", logger.exception.call_args.args[0])


class RegistryTests(unittest.IsolatedAsyncioTestCase):
    async def test_parts_are_built_outside_the_registry_lock_and_count_against_the_limit(self):
        release = asyncio.Event()
        built = []

        async def factory(config, transport, connection):
            if not built:
                built.append("slow")
                await release.wait()  # the first session's adapter takes its time
            return FakeAdapter().parts()

        gateway = LiveGateway(factory, LiveSettings(max_sessions=2))
        slow = asyncio.create_task(gateway.register(SessionConfig(model="live-1"), "websocket", None))
        await asyncio.sleep(0.01)
        quick = await gateway.register(SessionConfig(model="live-1"), "websocket", None)  # not held up
        with self.assertRaises(ProtocolError) as full:  # the one being built counts
            await gateway.register(SessionConfig(model="live-1"), "websocket", None)
        self.assertEqual(full.exception.code, "rate_limit_exceeded")
        release.set()
        for session in (await slow, quick):
            await session.close()


class IceServerTests(unittest.IsolatedAsyncioTestCase):
    async def test_the_roots_ice_servers_and_their_turn_credentials_reach_the_peer(self):
        turn = {"urls": ["turn:turn.example:3478?transport=udp"], "username": "user", "credential": "secret"}
        asked, built = [], []

        def ice_servers(request):
            asked.append(request.url.path)
            return [turn]

        class Recording(webrtc_edge.LiveWebRTCConnection):
            def __init__(self, ice_servers=None, **kwargs):
                built.append(ice_servers)
                super().__init__(ice_servers=ice_servers, **kwargs)

            async def initialize(self, sdp, type):
                raise RuntimeError("no real peer in this test")

        app = FastAPI()
        settings = LiveSettings(ice_urls=("stun:stun.example:3478",))
        gateway = mount_live(app, factory=adapter_factory, settings=settings, ice_servers=ice_servers)
        with patch.object(webrtc_edge, "LiveWebRTCConnection", Recording):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as http:
                response = await http.post(
                    "/v1/live/sessions",
                    json={"session": {"model": "live-1"}, "transport": {"type": "webrtc", "sdp": OFFER}},
                )
        self.assertEqual(response.status_code, 400)  # the offer is not answered here
        self.assertEqual(asked, ["/v1/live/sessions"])
        [servers] = built
        self.assertEqual(
            [(server.urls, server.username, server.credential) for server in servers],
            [(turn["urls"], "user", "secret"), (["stun:stun.example:3478"], None, None)],
        )
        self.assertEqual(gateway.sessions, {})


if __name__ == "__main__":
    unittest.main()
