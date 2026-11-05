# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D103, D107

"""Tests for the tool bridge: MCP over streamable HTTP, spoken with ``httpx`` as a runtime would.

The core's tests may not import ``mcp``, so the client side is plain JSON-RPC over HTTP.
"""

import asyncio
import dataclasses
import json
import unittest

import httpx

from examples.frontend_backend_live.delegation_core.backend.harness.bridge import ConductorDoor, McpBridge, WorkerDoor
from examples.frontend_backend_live.delegation_core.backend.harness.protocol import AppCall, CallRunning, CallSettled
from examples.frontend_backend_live.delegation_core.backend.harness.settings import Bridge
from tests.unit.contracts.backend import MENU_FUNCTION, TEXTS, FakeTaskContext
from tests.unit.delegation_core.backend.harness.doubles import vocabulary

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


class Raising(FakeTaskContext):
    """A task whose ``route()`` raises, with a failed_output of its own: the door's answer shows whose text it is."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.texts = dataclasses.replace(self.texts, failed_output="{name} broke: {error}")

    def route(self, call):
        raise RuntimeError("the gate broke")


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
        self.bridge = McpBridge(Bridge("127.0.0.1", 0, None))
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
        self.task = task or FakeTaskContext(declared=[MENU_FUNCTION])
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
        self.assertEqual((await client.post(self.http, "tools/list", {})).status_code, 401)  # a retired door: nothing

    async def test_the_worker_door_lists_the_apps_functions_the_tool_sources_and_the_work_tools_once_each(self):
        task = FakeTaskContext(declared=[MENU_FUNCTION, {"type": "function", "name": "lookup", "parameters": {}}])
        _, client = await self.worker(task)
        self.assertEqual(await client.names(self.http), ["get_menu", "lookup", "broken", "slow"])  # the app's lookup

    async def test_a_backend_call_is_reported_and_answered_in_the_cores_words_never_with_an_error_flag(self):
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
                self.assertEqual(result, {"content": [{"type": "text", "text": reply}], "isError": False})
                call_id = self.task.started[-1][0]  # three error results pause the server name in Hermes
                self.assertEqual((self.task.started[-1][1], self.task.outcomes[call_id][1]), (label, outcome))
        running, settled = self.mailbox.get_nowait(), self.mailbox.get_nowait()
        self.assertEqual((type(running), type(settled)), (CallRunning, CallSettled))
        self.assertEqual(json.loads(running.call["arguments"]), {"q": "teas"})

    async def test_a_call_whose_routing_raises_is_not_run_and_answered_with_its_tasks_failed_output(self):
        task = Raising(declared=[MENU_FUNCTION])
        _, client = await self.worker(task)
        result = await client.call(self.http, "lookup", {})
        self.assertEqual(
            (result["content"][0]["text"], result["isError"]), ("lookup broke: RuntimeError: the gate broke", False)
        )
        self.assertEqual(task.started, [])

    async def test_an_app_call_is_held_until_its_run_answers_and_a_retired_door_answers_it(self):
        door, client = await self.worker()
        call = asyncio.ensure_future(client.call(self.http, "get_menu", {"page": 1}))
        event = await asyncio.wait_for(self.mailbox.get(), 5)
        self.assertIsInstance(event, AppCall)
        self.assertEqual(event.held.call["name"], "get_menu")
        self.assertFalse(call.done())
        event.held.reply('{"teas": 3}')
        self.assertEqual((await call)["content"][0]["text"], '{"teas": 3}')
        second = asyncio.ensure_future(client.call(self.http, "get_menu", {"page": 2}))
        await asyncio.wait_for(self.mailbox.get(), 5)
        self.bridge.close(door, "Stopped.")
        self.assertEqual((await second)["content"][0]["text"], "Stopped.")

    async def test_a_rebound_door_keeps_its_name_and_token_and_serves_only_the_same_toolset(self):
        # A session kept for the next task keeps its door: the runtime holds its token and the tools it listed.
        door, client = await self.worker()
        later = FakeTaskContext(declared=[MENU_FUNCTION], work_id="W2", task_id="W2.1")
        self.assertTrue(door.rebind(later, (Tools(),)))
        self.assertEqual((await client.call(self.http, "lookup", {}))["content"][0]["text"], '{"found": "lookup"}')
        self.assertEqual(([label for _, label in later.started], self.task.started), (["shop"], []))
        self.assertFalse(door.rebind(FakeTaskContext(), (Tools(),)))  # without get_menu: another toolset
        menu = asyncio.ensure_future(client.call(self.http, "get_menu", {}))
        while not isinstance(event := await asyncio.wait_for(self.mailbox.get(), 5), AppCall):
            pass
        self.assertFalse(door.rebind(FakeTaskContext(declared=[MENU_FUNCTION]), (Tools(),)))  # it holds a call
        event.held.reply("{}")
        await menu
        self.bridge.close(door, "")
        self.assertFalse(door.rebind(FakeTaskContext(declared=[MENU_FUNCTION]), (Tools(),)))  # retired

    async def test_the_conductor_door_turns_each_call_into_a_decision_answered_with_the_outcome(self):
        placed = []

        async def place(decision):
            placed.append(decision)
            if len(placed) > 1:
                raise RuntimeError("the IM broke")
            return "Started W1 with W1.1."

        words = vocabulary()
        words.failed = "{name} broke: {error}"  # a template of its own: the answer shows whose text it is
        door = ConductorDoor("c1", words)
        client = await self.open(door)
        self.assertEqual(await client.names(self.http), ["start_work", "steer_work", "cancel_work", "answer_status"])

        async def answer(name, arguments):
            result = await client.call(self.http, name, arguments)
            self.assertFalse(result["isError"])
            return result["content"][0]["text"]

        start = {"delegations": ["d1"]}
        self.assertEqual(await answer("start_work", start), TEXTS.refusal("idle"))  # no decide() takes calls now
        door.bind(place)
        self.assertEqual(await answer("start_work", start), "Started W1 with W1.1.")
        self.assertEqual(json.loads(await answer("cancel_work", {})), {"error": "cancel_work: missing work_id"})
        self.assertEqual(await answer("start_work", start), "start_work broke: RuntimeError: the IM broke")
        self.assertEqual(len(placed), 2)


if __name__ == "__main__":
    unittest.main()
