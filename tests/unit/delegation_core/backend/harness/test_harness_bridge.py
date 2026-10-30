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

from delegation_core.backend.harness.bridge import ConductorDoor, McpBridge, WorkerDoor
from delegation_core.backend.harness.protocol import AppCall, CallRunning, CallSettled
from delegation_core.backend.harness.settings import Bridge
from tests.unit.contracts.backend import MENU_FUNCTION, TEXTS, FakeTaskContext
from tests.unit.delegation_core.backend.harness.doubles import until, vocabulary

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


class Refusing(FakeTaskContext):
    def route(self, call):
        return "refused" if call.get("name") == "lookup" else super().route(call)

    def refusal(self, call_id):
        return self.texts.stopped_output  # the IM refused it because its work is stopping


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
        reply = await self.post(client, "tools/call", {"name": name, "arguments": arguments})
        return reply["result"]


class BridgeTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.bridge = McpBridge(Bridge("127.0.0.1", 0, None))
        await self.bridge.start()
        self.http = httpx.AsyncClient(timeout=10)

    async def asyncTearDown(self):
        await self.http.aclose()
        await self.bridge.aclose()

    def worker(self, task=None):
        self.task = task or FakeTaskContext(declared=[MENU_FUNCTION])
        self.mailbox = asyncio.Queue()
        door = WorkerDoor("w1", self.task, (Tools(),), self.mailbox, tool_timeout_seconds=0.3)
        server = self.bridge.open(door)
        return door, McpClient(server.url, server.token)

    async def test_a_request_without_a_live_doors_token_is_refused(self):
        door, client = self.worker()
        response = await self.http.post(
            client.url, headers={"accept": ACCEPT}, json={"jsonrpc": "2.0", "id": 1, "method": "ping"}
        )
        self.assertEqual(response.status_code, 401)
        await client.connect(self.http)
        self.bridge.close(door, "")
        response = await client.post(self.http, "tools/list", {})
        self.assertEqual(response.status_code, 401)  # a retired door serves nothing

    async def test_only_post_is_served_and_any_other_method_is_answered_405_at_once(self):
        # Stateless HTTP opens no server-sent stream. A runtime may probe the URL before it connects (Hermes sends
        # HEAD, then GET on a 405, waiting up to 5 s on each), so a stream left open would hold every session's start.
        door, client = self.worker()
        for method in ("HEAD", "GET", "DELETE"):
            with self.subTest(method=method):
                response = await self.http.request(method, client.url, headers=client.headers(), timeout=2)
                self.assertEqual((response.status_code, response.headers.get("allow")), (405, "POST"))
        await client.connect(self.http)
        listed = await client.post(self.http, "tools/list", {})
        self.assertEqual([tool["name"] for tool in listed["result"]["tools"]][:1], ["get_menu"])  # POST still serves

    async def test_the_worker_door_serves_the_apps_functions_the_tool_sources_and_the_work_tools_once_each(self):
        task = FakeTaskContext(declared=[MENU_FUNCTION, {"type": "function", "name": "lookup", "parameters": {}}])
        door, client = self.worker(task)
        await client.connect(self.http)
        self.assertFalse(door.listed.is_set())
        listed = await client.post(self.http, "tools/list", {})
        names = [tool["name"] for tool in listed["result"]["tools"]]
        self.assertEqual(names, ["get_menu", "lookup", "broken", "slow"])  # the app's lookup wins; offered once
        self.assertTrue(door.listed.is_set())  # the runtime listed the tools: its connection works

    async def test_a_backend_call_is_routed_reported_and_answered_plainly(self):
        door, client = self.worker()
        await client.connect(self.http)
        result = await client.call(self.http, "lookup", {"q": "teas"})
        self.assertEqual(result, {"content": [{"type": "text", "text": '{"found": "lookup"}'}], "isError": False})
        [(call_id, label)] = self.task.started
        self.assertEqual(label, "shop")
        self.assertEqual(self.task.outcomes[call_id], ('{"found": "lookup"}', "done"))
        running, settled = self.mailbox.get_nowait(), self.mailbox.get_nowait()
        self.assertEqual((type(running), type(settled)), (CallRunning, CallSettled))
        self.assertEqual(json.loads(running.call["arguments"]), {"q": "teas"})

    async def test_a_failure_is_a_plain_result_in_the_cores_words_never_an_error_flag(self):
        door, client = self.worker()
        await client.connect(self.http)
        failed = TEXTS.failed_output.format(name="broken", error="RuntimeError: the shop is closed")
        offered = "get_menu, lookup, broken, slow"  # the door's tools
        unknown = TEXTS.unknown_tool_output.format(name="nothing", offered=offered)
        cases = {
            "broken": ("failed", json.dumps({"error": failed})),
            "slow": ("unclear", TEXTS.owed_output),
            "nothing": ("failed", json.dumps({"error": unknown})),
        }
        for name, (outcome, reply) in cases.items():
            with self.subTest(name):
                result = await client.call(self.http, name, {})
                self.assertFalse(result["isError"])  # three error results pause the server name in Hermes
                self.assertEqual(result["content"][0]["text"], reply)
                call_id = self.task.started[-1][0]
                self.assertEqual(self.task.outcomes[call_id][1], outcome)

    async def test_a_call_that_raises_in_the_worker_door_is_answered_with_its_tasks_failed_output(self):
        door, client = self.worker(Raising(declared=[MENU_FUNCTION]))
        await client.connect(self.http)
        result = await client.call(self.http, "lookup", {})
        self.assertFalse(result["isError"])
        self.assertEqual(result["content"][0]["text"], "lookup broke: RuntimeError: the gate broke")

    async def test_a_call_the_im_refuses_is_not_run_and_answered_with_the_ims_reason(self):
        door, client = self.worker(Refusing(declared=[MENU_FUNCTION]))
        await client.connect(self.http)
        result = await client.call(self.http, "lookup", {})
        self.assertEqual(result["content"][0]["text"], TEXTS.stopped_output)
        self.assertEqual(self.task.started, [])  # never reported: the IM gave it its output
        self.assertTrue(self.mailbox.get_nowait().refused)

    async def test_an_app_call_is_held_until_its_run_answers_and_a_retired_door_answers_it(self):
        door, client = self.worker()
        await client.connect(self.http)
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
        door, client = self.worker()
        await client.connect(self.http)
        name, token = door.name, door.token
        later = FakeTaskContext(declared=[MENU_FUNCTION], work_id="W2", task_id="W2.1")
        self.assertTrue(door.rebind(later, (Tools(),)))
        self.assertEqual((door.name, door.token, self.bridge.doors[token]), (name, token, door))
        result = await client.call(self.http, "lookup", {})
        self.assertEqual(result["content"][0]["text"], '{"found": "lookup"}')
        self.assertEqual(([label for _, label in later.started], self.task.started), (["shop"], []))
        self.assertFalse(door.rebind(FakeTaskContext(), (Tools(),)))  # without get_menu: another toolset
        menu = asyncio.ensure_future(client.call(self.http, "get_menu", {}))
        await until(lambda: door.held, timeout=5)
        self.assertFalse(door.rebind(FakeTaskContext(declared=[MENU_FUNCTION]), (Tools(),)))  # it holds a call
        door.held[0].reply("{}")
        await menu
        self.bridge.close(door, "")
        self.assertFalse(door.rebind(FakeTaskContext(declared=[MENU_FUNCTION]), (Tools(),)))  # retired
        self.assertIs(door.task, later)

    async def test_the_conductor_door_turns_each_call_into_a_decision_answered_with_the_outcome(self):
        placed = []

        async def place(decision):
            placed.append(decision)
            return "Started W1 with W1.1."

        door = ConductorDoor("c1", vocabulary())
        server = self.bridge.open(door)
        client = McpClient(server.url, server.token)
        await client.connect(self.http)
        listed = await client.post(self.http, "tools/list", {})
        self.assertEqual(
            [tool["name"] for tool in listed["result"]["tools"]],
            ["start_work", "steer_work", "cancel_work", "answer_status"],
        )
        idle = await client.call(self.http, "start_work", {"delegations": ["d1"]})
        self.assertEqual(idle["content"][0]["text"], "Not applied: no request waits to be placed now.")
        door.bind(place)
        result = await client.call(self.http, "start_work", {"delegations": ["d1"]})
        self.assertEqual(result["content"][0]["text"], "Started W1 with W1.1.")
        self.assertEqual(len(placed), 1)
        bad = await client.call(self.http, "cancel_work", {})
        self.assertFalse(bad["isError"])
        self.assertEqual(json.loads(bad["content"][0]["text"]), {"error": "cancel_work: missing work_id"})

    async def test_a_control_call_that_raises_is_answered_with_the_vocabularys_failed_answer(self):
        async def place(decision):
            raise RuntimeError("the IM broke")

        words = vocabulary()
        words.failed = "{name} broke: {error}"  # a template of its own: the answer shows whose text it is
        door = ConductorDoor("c1", words)
        door.bind(place)
        server = self.bridge.open(door)
        client = McpClient(server.url, server.token)
        await client.connect(self.http)
        result = await client.call(self.http, "start_work", {"delegations": ["d1"]})
        self.assertFalse(result["isError"])
        self.assertEqual(result["content"][0]["text"], "start_work broke: RuntimeError: the IM broke")


if __name__ == "__main__":
    unittest.main()
