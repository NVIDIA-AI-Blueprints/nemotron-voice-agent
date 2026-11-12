# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D103, D107

"""The tool bridge: MCP over streamable HTTP, spoken as a runtime would in plain JSON-RPC over ``httpx``."""

import asyncio
import gc
import json
import socket
import unittest
import warnings
from unittest import mock

import httpx

from examples.frontend_backend_live.delegation_core.backend.harness.bridge import ConductorDoor, McpBridge, WorkerDoor
from examples.frontend_backend_live.delegation_core.backend.harness.protocol import AppCall, CallRunning, CallSettled
from examples.frontend_backend_live.delegation_core.backend.harness.settings import Bridge
from tests.unit.delegation_core.harness.doubles import vocabulary
from tests.unit.delegation_core.helpers import MENU, TEXTS, FakeTaskContext

ACCEPT = "application/json, text/event-stream"


class Tools:
    """A tool source: ``lookup`` answers, ``broken`` raises, ``slow`` never answers."""

    label = "shop"

    def tools(self):
        schema = {"type": "object", "properties": {}}
        return [{"type": "function", "name": name, "parameters": schema} for name in ("lookup", "broken", "slow")]

    async def execute(self, call_id, name, arguments):
        if name == "broken":
            raise RuntimeError("the shop is closed")
        if name == "slow":
            await asyncio.sleep(10)
        return json.dumps({"found": name})


class McpClient:
    """A runtime's side of the bridge: JSON-RPC over HTTP with a bearer token."""

    def __init__(self, url, token):
        self.url, self.token, self.ids = url, token, 0

    def headers(self):
        return {"authorization": f"Bearer {self.token}", "accept": ACCEPT, "mcp-protocol-version": "2025-06-18"}

    async def post(self, client, method, params=None, notify=False):
        payload = {"jsonrpc": "2.0", "method": method, **({"params": params} if params is not None else {})}
        if not notify:
            self.ids += 1
            payload["id"] = self.ids
        response = await client.post(self.url, headers=self.headers(), json=payload)
        if notify or response.status_code != 200:
            return response
        data = [line[5:].strip() for line in response.text.splitlines() if line.startswith("data:")]
        return json.loads(data[-1]) if data else response.json()

    async def connect(self, client):
        init = {"protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "runtime", "version": "1"}}
        await self.post(client, "initialize", init)
        await self.post(client, "notifications/initialized", notify=True)

    async def call(self, client, name, arguments):
        return (await self.post(client, "tools/call", {"name": name, "arguments": arguments}))["result"]

    async def names(self, client):
        return [tool["name"] for tool in (await self.post(client, "tools/list", {}))["result"]["tools"]]


class BridgeTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.bridge = McpBridge(Bridge(bind="127.0.0.1:0"))
        await self.bridge.start()
        self.http = httpx.AsyncClient(timeout=10)

    async def asyncTearDown(self):
        await self.http.aclose()
        await self.bridge.aclose()

    async def open(self, door):
        server = self.bridge.open(door)
        client = McpClient(server.url, server.token)
        await client.connect(self.http)
        return client

    async def worker(self, task=None):
        self.task = task or FakeTaskContext()
        self.mailbox = asyncio.Queue()
        door = WorkerDoor("w1", self.task, (Tools(),), self.mailbox, tool_timeout_seconds=0.3)
        return door, await self.open(door)

    async def test_only_post_with_a_live_doors_token_is_served(self):
        # Stateless HTTP opens no server-sent stream. A runtime may probe the URL before it connects (Hermes sends
        # HEAD, then GET on a 405, waiting up to 5 s on each), so a stream left open would hold every session's start.
        door, client = await self.worker()
        ping = {"jsonrpc": "2.0", "id": 1, "method": "ping"}
        self.assertEqual((await self.http.post(client.url, headers={"accept": ACCEPT}, json=ping)).status_code, 401)
        for method in ("HEAD", "GET", "DELETE"):
            response = await self.http.request(method, client.url, headers=client.headers(), timeout=2)
            self.assertEqual((response.status_code, response.headers.get("allow")), (405, "POST"))
        self.assertEqual((await client.names(self.http))[:1], ["get_menu"])
        self.bridge.close(door, "")
        self.assertEqual((await client.post(self.http, "tools/list", {})).status_code, 401)

    async def test_the_worker_door_lists_each_tool_name_once(self):
        task = FakeTaskContext(declared=[MENU, {"type": "function", "name": "lookup", "parameters": {}}])
        _, client = await self.worker(task)
        self.assertEqual(await client.names(self.http), ["get_menu", "lookup", "broken", "slow"])

    async def test_a_backend_call_is_answered_in_the_cores_words(self):
        _, client = await self.worker()
        found = json.dumps({"found": "lookup"})
        failed = TEXTS.failed_output.format(name="broken", error="RuntimeError: the shop is closed")
        unknown = TEXTS.unknown_tool_output.format(name="nothing", offered="get_menu, lookup, broken, slow")
        cases = {
            "lookup": ("shop", "done", found),
            "broken": ("shop", "failed", json.dumps({"error": failed})),
            "slow": ("shop", "unclear", TEXTS.owed_output),
            "nothing": ("harness", "failed", json.dumps({"error": unknown})),
        }
        for name, (label, outcome, reply) in cases.items():
            with self.subTest(name):
                result = await client.call(self.http, name, {"q": "teas"})
                # Never isError: three error results pause the server name in Hermes.
                self.assertEqual(result, {"content": [{"type": "text", "text": reply}], "isError": False})
                call_id = self.task.started[-1][0]
                self.assertEqual((self.task.started[-1][1], self.task.outcomes[call_id][1]), (label, outcome))
        running, settled = self.mailbox.get_nowait(), self.mailbox.get_nowait()
        self.assertEqual((type(running), type(settled)), (CallRunning, CallSettled))
        self.assertEqual(json.loads(running.call["arguments"]), {"q": "teas"})

    async def test_a_call_whose_routing_raises_is_not_run(self):
        # A failed_output of its own shows whose text the answer is.
        task = FakeTaskContext()
        task.texts = task.texts.model_copy(update={"failed_output": "{name} broke: {error}"})
        task.route = mock.Mock(side_effect=RuntimeError("the gate broke"))
        _, client = await self.worker(task)
        result = await client.call(self.http, "lookup", {})
        self.assertEqual(
            (result["content"][0]["text"], result["isError"]), ("lookup broke: RuntimeError: the gate broke", False)
        )
        self.assertEqual(task.started, [])

    async def test_a_rebound_door_serves_only_the_same_toolset(self):
        door, client = await self.worker()
        later = FakeTaskContext(work_id="W2", task_id="W2.1")
        self.assertTrue(door.rebind(later, (Tools(),)))
        self.assertEqual((await client.call(self.http, "lookup", {}))["content"][0]["text"], '{"found": "lookup"}')
        self.assertEqual(([label for _, label in later.started], self.task.started), (["shop"], []))
        self.assertFalse(door.rebind(FakeTaskContext(declared=()), (Tools(),)))
        menu = asyncio.ensure_future(client.call(self.http, "get_menu", {}))
        while not isinstance(event := await asyncio.wait_for(self.mailbox.get(), 5), AppCall):
            pass
        self.assertFalse(door.rebind(FakeTaskContext(), (Tools(),)))
        event.held.reply("{}")
        await menu
        self.bridge.close(door, "")
        self.assertFalse(door.rebind(FakeTaskContext(), (Tools(),)))

    async def test_the_conductor_door_places_each_call_as_a_decision(self):
        placed = []

        async def place(decision):
            placed.append(decision)
            if len(placed) > 1:
                raise RuntimeError("the IM broke")
            return "Started W1 with W1.1."

        words = vocabulary()
        # A template of its own shows whose text the answer is.
        words.failed = "{name} broke: {error}"
        door = ConductorDoor("c1", words)
        client = await self.open(door)
        self.assertEqual(await client.names(self.http), ["start_work", "steer_work", "cancel_work", "answer_status"])

        async def answer(name, arguments):
            result = await client.call(self.http, name, arguments)
            self.assertFalse(result["isError"])
            return result["content"][0]["text"]

        start = {"delegations": ["d1"]}
        self.assertEqual(await answer("start_work", start), TEXTS.refusal("idle"))
        door.bind(place)
        self.assertEqual(await answer("start_work", start), "Started W1 with W1.1.")
        self.assertEqual(json.loads(await answer("cancel_work", {})), {"error": "cancel_work: missing work_id"})
        self.assertEqual(await answer("start_work", start), "start_work broke: RuntimeError: the IM broke")
        self.assertEqual(len(placed), 2)


def ipv6_loopback() -> bool:
    try:
        with socket.socket(socket.AF_INET6, socket.SOCK_STREAM) as probe:
            probe.bind(("::1", 0))
    except OSError:
        return False
    return True


class BridgeBindTests(unittest.IsolatedAsyncioTestCase):
    @unittest.skipUnless(ipv6_loopback(), "this host has no IPv6 loopback")
    async def test_an_ipv6_bind_is_reached_at_its_url_in_brackets(self):
        bridge = McpBridge(Bridge(bind="[::1]:0"))
        await bridge.start()
        self.addAsyncCleanup(bridge.aclose)
        self.assertEqual(bridge.settings.host, "::1")
        self.assertRegex(bridge.url, r"^http://\[::1\]:\d+/mcp$")
        async with httpx.AsyncClient(timeout=10) as http:
            response = await http.post(bridge.url, headers={"accept": ACCEPT}, json={"jsonrpc": "2.0", "id": 1})
        self.assertEqual(response.status_code, 401)

    async def test_a_bind_that_fails_raises_and_closes_its_socket(self):
        taken = socket.create_server(("127.0.0.1", 0))
        self.addCleanup(taken.close)
        in_use, out_of_range = f"127.0.0.1:{taken.getsockname()[1]}", "127.0.0.1:70000"
        for bind, error in ((in_use, OSError), (out_of_range, OverflowError)):
            with self.subTest(bind=bind), warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", ResourceWarning)
                with self.assertRaises(error):
                    await McpBridge(Bridge(bind=bind)).start()
                gc.collect()
                self.assertEqual([str(w.message) for w in caught if issubclass(w.category, ResourceWarning)], [])
