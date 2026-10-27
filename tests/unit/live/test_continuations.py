# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D103, D105, D107

"""Call results and the ``response.create`` that continues a waiting response.

A ``response.create`` continues every waiting response whose results are all present, one request each. It is refused
only when it would continue nothing while results are owed; with nothing waiting it is a typed run.
"""

import asyncio
import unittest

from live.gateway.websocket import wire_codec
from live.session import LiveProtocolSession
from tests.unit.live_helpers import Collector, FakeAdapter, responses_config

HI = [{"role": "user", "content": [{"type": "input_text", "text": "Hi."}]}]
TYPED = {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "Make it two."}]}


def function_call(call_id):
    return {"type": "function_call", "call_id": call_id, "name": "get_menu", "arguments": "{}"}


def output(call_id, text="{}"):
    return {"type": "function_call_output", "call_id": call_id, "output": text}


def waiting(response_id, *calls):
    """A response that ends ``completed`` with calls for the client: it waits for their results."""
    done = [{"type": "response.output_item.done", "item": call} for call in calls]
    return [
        {"type": "response.created", "response": {"id": response_id}},
        *done,
        {"type": "response.completed", "response": {"id": response_id, "status": "completed"}},
    ]


class Stream:
    def __init__(self, events):
        self.events = list(events)

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self.events:
            raise StopAsyncIteration
        return self.events.pop(0)


class ContinuationCore(FakeAdapter):
    """A core that answers the frontend's delegations from its script, and every other request with an empty one."""

    def __init__(self, *script):
        super().__init__()
        self.script = list(script)
        self.requests: list[dict] = []

    async def create_response(self, params):
        self.requests.append(params)
        number = len(self.requests)
        if self.script and not params["metadata"].get("live_typed_run") and not continues(params):
            return Stream(self.script.pop(0))
        return Stream(waiting(f"resp_{number}"))


def continues(params):
    return any(item.get("type", "message") != "message" for item in params["input"])


class Listener:
    def on_delegation_event(self, delegation_id, event):
        pass

    def on_delegation_lost(self, delegation_id):
        pass


class ContinuationCase(unittest.IsolatedAsyncioTestCase):
    async def start(self, *script, core=ContinuationCore):
        self.core = core(*script)
        config = responses_config().with_wire_format()
        self.session = LiveProtocolSession(
            config, "websocket", self.core.parts(), core=self.core.core, codec=wire_codec(config)
        )
        self.client = Collector()
        self.key = self.session.subscribe(self.client.send, "primary")
        await self.session.start("start_1")
        await self.client.wait_for("session.started")
        self.session.delegation.listen(Listener())
        self.addAsyncCleanup(self.session.close)

    async def delegate(self):
        """Delegate, and wait until the client has seen the delegation's response end."""
        delegation_id = await self.session.delegation.delegate(HI)
        await self.until(lambda: self.ended(delegation_id))
        return delegation_id

    def ended(self, delegation_id):
        return any(
            e["type"] == "response.event"
            and e["delegation_id"] == delegation_id
            and e["event"]["type"] == "response.completed"
            for e in self.client.events
        )

    async def until(self, predicate, timeout=3.0):
        async with asyncio.timeout(timeout):
            while not predicate():
                await asyncio.sleep(0.005)

    async def send(self, message):
        await self.session.handle(message, self.key)

    async def create(self, event_id):
        """Send response.create; return how many requests the core received once it settled."""
        before = len(self.core.requests)
        await self.send({"type": "response.create", "event_id": event_id})
        await asyncio.sleep(0.05)
        return self.core.requests[before:]

    def errors(self):
        return [e["error"] for e in self.client.events if e["type"] == "error"]


class TwoWorksTests(ContinuationCase):
    async def test_two_waiting_responses_are_continued_one_at_a_time_in_either_order_without_an_error(self):
        for order in (("c1", "c2"), ("c2", "c1")):
            with self.subTest(order=order):
                await self.start(waiting("resp_a", function_call("c1")), waiting("resp_b", function_call("c2")))
                owner = {"c1": await self.delegate(), "c2": await self.delegate()}
                for number, call_id in enumerate(order):
                    await self.send({"type": "response.item.create", "item": output(call_id)})
                    [request] = await self.create(f"rc{number}")
                    self.assertEqual(request["metadata"]["live_delegation_id"], owner[call_id])
                    self.assertEqual(request["input"], [output(call_id)])
                self.assertEqual(self.errors(), [])
                await self.session.close()

    async def test_a_partial_response_waits_without_an_error_while_a_complete_one_is_continued(self):
        await self.start(
            waiting("resp_a", function_call("c1"), function_call("c2")), waiting("resp_b", function_call("c3"))
        )
        await self.delegate()
        second = await self.delegate()
        await self.send({"type": "response.item.create", "item": output("c1")})
        await self.send({"type": "response.item.create", "item": output("c3")})
        [request] = await self.create("rc1")
        self.assertEqual((request["metadata"]["live_delegation_id"], request["input"]), (second, [output("c3")]))
        self.assertEqual(self.errors(), [])

    async def test_results_go_in_call_order_and_the_typed_messages_with_the_newest(self):
        await self.start(
            waiting("resp_a", function_call("c1"), function_call("c2")), waiting("resp_b", function_call("c3"))
        )
        first, second = await self.delegate(), await self.delegate()
        for item in (output("c3"), TYPED, output("c2"), output("c1")):
            await self.send({"type": "response.item.create", "item": item})
        requests = await self.create("rc1")
        self.assertEqual(
            [(r["metadata"]["live_delegation_id"], r["input"]) for r in requests],
            [(first, [output("c1"), output("c2")]), (second, [output("c3"), TYPED])],
        )


class RefusalTests(ContinuationCase):
    async def test_text_while_function_outputs_are_owed_is_refused_and_kept(self):
        await self.start(waiting("resp_a", function_call("c1")))
        delegation_id = await self.delegate()
        await self.send({"type": "response.item.create", "item": TYPED})
        self.assertEqual(await self.create("rc1"), [])
        [error] = self.errors()
        self.assertEqual((error["code"], error["client_event_id"]), ("function_call_outputs_required", "rc1"))
        self.assertIn("c1", error["message"])
        await self.send({"type": "response.item.create", "item": output("c1")})
        [request] = await self.create("rc2")
        self.assertEqual(request["metadata"]["live_delegation_id"], delegation_id)
        self.assertEqual(request["input"], [output("c1"), TYPED])  # the text was kept

    async def test_a_partial_response_alone_is_refused(self):
        await self.start(waiting("resp_a", function_call("c1"), function_call("c2")))
        await self.delegate()
        await self.send({"type": "response.item.create", "item": output("c1")})
        self.assertEqual(await self.create("rc1"), [])
        self.assertEqual(self.errors()[0]["code"], "function_call_outputs_required")

    async def test_a_missing_result_of_another_tool_is_tool_outputs_required(self):
        custom = {"type": "custom_tool_call", "call_id": "c9", "name": "draw", "input": "a cat"}
        await self.start(waiting("resp_a", custom))
        await self.delegate()
        self.assertEqual(await self.create("rc1"), [])
        self.assertEqual(self.errors()[0]["code"], "tool_outputs_required")

    async def test_a_result_for_a_call_never_relayed_is_refused(self):
        await self.start()
        for item in (output("c404"), {"type": "mcp_approval_response", "approval_request_id": "x404", "approve": True}):
            await self.send({"type": "response.item.create", "event_id": "i1", "item": item})
        await self.until(lambda: len(self.errors()) == 2)
        self.assertEqual(
            [(error["code"], error["param"]) for error in self.errors()],
            [("invalid_value", "item.call_id"), ("invalid_value", "item.approval_request_id")],
        )


class ResultItemTests(ContinuationCase):
    async def test_every_official_result_item_is_passed_on_for_a_relayed_call(self):
        calls = {
            "function_call_output": function_call("k1"),
            "custom_tool_call_output": {"type": "custom_tool_call", "call_id": "k2", "name": "draw", "input": ""},
            "computer_call_output": {"type": "computer_call", "call_id": "k3", "pending_safety_checks": []},
            "shell_call_output": {"type": "shell_call", "call_id": "k4", "action": {"commands": ["ls"]}},
            "apply_patch_call_output": {"type": "apply_patch_call", "call_id": "k5", "operation": {}},
            "tool_search_output": {"type": "tool_search_call", "call_id": "k6", "execution": "client"},
            "mcp_approval_response": {"type": "mcp_approval_request", "id": "k7", "name": "x", "server_label": "crm"},
        }
        results = [
            {"type": "function_call_output", "call_id": "k1", "output": [{"type": "input_text", "text": "a list"}]},
            {"type": "custom_tool_call_output", "call_id": "k2", "output": "drawn"},
            {"type": "computer_call_output", "call_id": "k3", "output": {"type": "computer_screenshot"}},
            {"type": "shell_call_output", "call_id": "k4", "output": []},
            {"type": "apply_patch_call_output", "call_id": "k5", "status": "completed"},
            {"type": "tool_search_output", "call_id": "k6", "execution": "client", "tools": []},
            {"type": "mcp_approval_response", "approval_request_id": "k7", "approve": True},
        ]
        await self.start(waiting("resp_a", *calls.values()))
        delegation_id = await self.delegate()
        for item in results:
            await self.send({"type": "response.item.create", "item": item})
        [request] = await self.create("rc1")
        self.assertEqual((request["metadata"]["live_delegation_id"], request["input"]), (delegation_id, results))
        self.assertEqual(self.errors(), [])

    async def test_a_repeat_is_passed_on_for_the_core_to_judge(self):
        await self.start(waiting("resp_a", function_call("c1")))
        await self.delegate()
        await self.send({"type": "response.item.create", "item": output("c1", "first")})
        await self.send({"type": "response.item.create", "item": output("c1", "second")})
        [request] = await self.create("rc1")
        self.assertEqual(request["input"], [output("c1", "first"), output("c1", "second")])
        self.assertEqual(self.errors(), [])

    async def test_a_repeat_the_core_refuses_reaches_the_client_as_its_mistake_with_the_cores_reason(self):
        class JudgingCore(ContinuationCore):
            async def create_response(self, params):
                calls = [item["call_id"] for item in params["input"] if item.get("type") == "function_call_output"]
                if len(calls) != len(set(calls)):  # as the IM judges a repeat: an invalid request
                    raise ValueError(f"Duplicate result for call {calls[0]}")
                return await super().create_response(params)

        await self.start(waiting("resp_a", function_call("c1")), core=JudgingCore)
        await self.delegate()
        await self.send({"type": "response.item.create", "item": output("c1", "first")})
        await self.send({"type": "response.item.create", "item": output("c1", "second")})
        await self.create("rc1")
        await self.until(lambda: self.errors())
        [refused] = self.errors()
        self.assertEqual(
            (refused["type"], refused["code"], refused["message"], refused["client_event_id"]),
            ("invalid_request_error", "invalid_value", "Duplicate result for call c1", "rc1"),
        )

    async def test_an_answered_call_owes_nothing_until_its_route_closes(self):
        await self.start(waiting("resp_a", function_call("c1")))
        await self.delegate()
        await self.send({"type": "response.item.create", "item": output("c1")})
        routes = self.session.delegation.routes
        [continuation] = routes.requests()
        routes.sending(continuation)
        routes.sent(continuation)  # the core accepted the continuation; its response has not ended yet
        self.assertEqual(routes.answered, {"c1"})
        [typed_run] = routes.requests()  # nothing is owed: text may run
        self.assertIsNone(typed_run.delegation_id)
        await self.send({"type": "response.item.create", "item": output("c1", "again")})  # still routed: passed on
        self.assertEqual([result.item for result in routes.results], [output("c1", "again")])

    async def test_a_result_whose_route_closed_is_dropped(self):
        await self.start(waiting("resp_a", function_call("c1")))
        delegation_id = await self.delegate()
        await self.send({"type": "response.item.create", "item": output("c1")})
        routes = self.session.delegation.routes
        routes.response_ended(delegation_id, "resp_later")  # a later response of the delegation ended
        self.assertEqual((routes.routes, routes.results), ({}, []))


class TypedRunTests(ContinuationCase):
    async def test_a_bare_response_create_starts_a_typed_run(self):
        await self.start()
        [request] = await self.create("rc1")
        self.assertEqual((request["input"], request["metadata"]["live_typed_run"]), ([], "true"))
        created = await self.client.wait_for("session.delegation.created")
        self.assertEqual(created["client_event_id"], "rc1")

    async def test_a_typed_runs_events_carry_its_delegation_id_and_the_commands_client_event_id(self):
        await self.start()
        await self.send({"type": "response.item.create", "item": TYPED})
        await self.create("rc1")
        created = await self.client.wait_for("session.delegation.created")
        delegation_id = created["delegation"]["id"]
        relayed = [e for e in self.client.events if e["type"] == "response.event"]
        self.assertTrue(relayed)
        self.assertTrue(all(e["delegation_id"] == delegation_id for e in relayed))
        self.assertTrue(all(e["client_event_id"] == "rc1" for e in relayed))

    async def test_the_frontends_delegation_events_carry_no_client_event_id(self):
        await self.start(waiting("resp_a"))
        await self.delegate()
        relayed = [e for e in self.client.events if e["type"] in ("response.event", "session.delegation.created")]
        self.assertTrue(relayed)
        self.assertTrue(all("client_event_id" not in e for e in relayed))

    async def test_a_continuations_events_carry_the_commands_client_event_id(self):
        await self.start(waiting("resp_a", function_call("c1")))
        delegation_id = await self.delegate()
        await self.send({"type": "response.item.create", "item": output("c1")})
        await self.create("rc7")
        await self.until(lambda: any(e.get("client_event_id") == "rc7" for e in self.client.events))
        continued = [e for e in self.client.events if e.get("client_event_id") == "rc7"]
        self.assertTrue(all(e["delegation_id"] == delegation_id for e in continued))
        self.assertNotIn("session.delegation.created", [e["type"] for e in continued])  # announced once, by delegate()


if __name__ == "__main__":
    unittest.main()
