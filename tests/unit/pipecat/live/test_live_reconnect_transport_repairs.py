# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D103

"""Actual localhost TCP/ASGI reconnect, using no provider, native runtime, audio or business executor."""

import asyncio
import json
import socket
import unittest
from unittest.mock import patch

import uvicorn
from fastapi import FastAPI
from websockets.asyncio.client import connect

from nemotron_voice_agent.pipecat.runtime.gateways.live.gateway import LiveSettings
from nemotron_voice_agent.pipecat.runtime.gateways.live.gateway.routes import create_live_router
from tests.unit.pipecat.live.test_live_reconnect import ANSWER, CONFIG, RestorableAdapter
from tests.unit.pipecat.live_helpers import response


class ReconnectTransportRepairTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.core, self.adapters, self.clients = RestorableAdapter(), [], []

        async def factory(config, transport, ice_servers):
            adapter = RestorableAdapter()
            adapter.config = config
            self.adapters.append(adapter)
            return adapter.parts()

        app = FastAPI()
        router, self.gateway = create_live_router(factory, LiveSettings(), core=self.core.core)
        app.include_router(router)
        self.listener = socket.socket()
        self.addCleanup(self.listener.close)
        self.listener.bind(("127.0.0.1", 0))
        self.listener.listen()
        self.server = uvicorn.Server(uvicorn.Config(app, log_level="critical", lifespan="off", ws="websockets-sansio"))
        self.runner = asyncio.create_task(self.server.serve(sockets=[self.listener]))
        async with asyncio.timeout(30):
            while not self.server.started:
                await asyncio.sleep(0)
        self.address = f"ws://127.0.0.1:{self.listener.getsockname()[1]}/v1/live/sessions"

    async def asyncTearDown(self):
        await self.gateway.close_all()
        for ws in self.clients:
            await ws.close()
        self.server.should_exit = True
        await asyncio.wait_for(self.runner, 30)

    async def open(self, message):
        ws = await connect(self.address, proxy=None)
        self.clients.append(ws)
        await ws.send(json.dumps(message))
        return ws

    async def receive(self, ws, kind):
        async with asyncio.timeout(30):
            while True:
                row = json.loads(await ws.recv())
                if row["type"] == kind:
                    return row

    async def start(self):
        ws = await self.open({"type": "session.start", "session": CONFIG, "reconnect": {"version": 1}})
        ready = await self.receive(ws, "session.reconnect.ready")
        return ws, next(iter(self.gateway.sessions.values())), ready["reconnect"]

    async def close_call(self, ws):
        await ws.send(json.dumps({"type": "session.close"}))
        await self.receive(ws, "session.closed")
        await ws.wait_closed()
        self.assertEqual(ws.close_code, 1000)

    async def test_server_retained_close_is_1013_and_old_owner_cannot_end_the_resumed_call(self):
        ws, session, authority = await self.start()
        old = session.socket
        await session.primary_lost(old)
        await ws.wait_closed()
        self.assertEqual(ws.close_code, 1013)
        self.assertTrue(ws.close_reason)
        resumed = await self.open({"type": "session.start", "resume": {**authority, "request_id": "resume1"}})
        receipt = await self.receive(resumed, "session.resumed")
        self.assertEqual(receipt["session"]["id"], session.id)
        self.assertNotEqual(receipt["reconnect"]["token"], authority["token"])
        await session.primary_lost(old)
        self.assertEqual((session.status, session.attachment_state), ("active", "active"))
        self.assertFalse(self.core.deleted.is_set())
        self.assertEqual(self.core.requests, [])
        self.assertEqual(len(self.adapters), 2)
        await self.close_call(resumed)
        self.assertEqual(self.gateway.resume_leases, {})

    async def test_definite_refusal_then_exact_same_operation_admits_one_prepared_continuation(self):
        ws, session, _ = await self.start()
        call = {"id": "read-item", "call_id": "read1", "type": "function_call", "name": "read_value", "arguments": "{}"}
        self.core.streams.append(response("first", call))
        await session.delegation.delegate([{"role": "user", "content": "Read the value."}])
        await asyncio.gather(*session.delegation.readers)
        await self.receive(ws, "response.event")
        create = {"type": "response.create", "operation_id": "continue"}
        await ws.send(json.dumps(create))
        self.assertEqual((await self.receive(ws, "error"))["error"]["code"], "function_call_outputs_required")
        await ws.send(
            json.dumps(
                {
                    "type": "response.item.create",
                    "operation_id": "read-result",
                    "item": {
                        "type": "function_call_output",
                        "call_id": "read1",
                        "output": "The read value is 7.",
                    },
                }
            )
        )
        self.assertEqual((await self.receive(ws, "session.operation.receipt"))["status"], "recorded")
        self.core.streams.append(response("second", ANSWER))
        with patch.object(session.delegation.routes, "requests", wraps=session.delegation.routes.requests) as prepare:
            await ws.send(json.dumps(create))
            accepted = completed = False
            async with asyncio.timeout(30):
                while not (accepted and completed):
                    row = json.loads(await ws.recv())
                    if row["type"] == "session.operation.receipt":
                        self.assertEqual(row["status"], "recorded")
                        accepted = True
                    elif row["type"] == "response.event":
                        completed |= row["event"].get("response", {}).get("id") == "second"
            await ws.send(json.dumps(create))
            self.assertEqual((await self.receive(ws, "session.operation.receipt"))["status"], "recorded")
            self.assertEqual(prepare.call_count, 1, "commit and retries must reuse the prepared continuation")
        await self.close_call(ws)
        self.assertEqual(len(self.core.requests), 2)
        self.assertEqual(
            self.core.requests[-1]["input"],
            [{"type": "function_call_output", "call_id": "read1", "output": "The read value is 7."}],
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
