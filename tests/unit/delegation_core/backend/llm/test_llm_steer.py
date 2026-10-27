# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D103, D107

"""Tests for steering: new words reach the work at its next model boundary, with delivery reports."""

import asyncio
import unittest

import httpx
from openai import APIConnectionError

from delegation_core.backend.llm import LlmSettings
from tests.unit.contracts.backend import call_round, check_events, text_round
from tests.unit.delegation_core.backend.llm.doubles import (
    IMTask,
    RecordingTools,
    ScriptedProvider,
    adapter,
    items_of,
    kinds,
)

TWO = [{"type": "message", "role": "user", "content": "Make it two."}]
OAT = [{"type": "message", "role": "user", "content": "With oat milk."}]
DECAF = [{"type": "message", "role": "user", "content": "Actually make it decaf."}]
SET_DECAF = {"type": "function", "name": "set_decaf", "parameters": {"type": "object", "properties": {}}}


class TypedRunTask(IMTask):
    """A task whose steering is a typed run with its own settings: once it lands, they are the Work's."""

    def __init__(self, brings: dict, **kwargs):
        super().__init__(**kwargs)
        self.brings = brings

    def settle(self, steering, delivery):
        super().settle(steering, delivery)
        if delivery == "applied":
            self.instructions, self.declared = self.brings["instructions"], list(self.brings["tools"])


def gated(text, response_id="resp_1"):
    gate = asyncio.Event()
    round_ = text_round(text, response_id)
    round_.insert(1, gate)
    return gate, round_


async def read_until_created(events, task):
    event = await anext(events)
    task.see(event)
    return event


async def read_rest(events, task):
    rest = []
    async for event in events:
        task.see(event)
        rest.append(event)
    return rest


def count(payload, items):
    return sum(item in payload["input"] for item in items)


class SteerTests(unittest.IsolatedAsyncioTestCase):
    async def test_a_round_still_streaming_is_restarted_with_the_steering_once(self):
        gate, slow = gated("One latte.")
        provider = ScriptedProvider(slow, text_round("Two lattes.", "resp_2"))
        task = IMTask()
        run = adapter(provider).start(task)
        events = run.events()
        await read_until_created(events, task)
        task.steer_with(TWO)
        run.steer()
        gate.set()
        rest = await read_rest(events, task)
        self.assertEqual(kinds(rest)[:2], ["response.incomplete", "response.created"])
        self.assertEqual(rest[0]["response"]["incomplete_details"], {"reason": "steered"})
        self.assertEqual(check_events([{"type": "response.created", "response": {}}, *rest]), [])
        self.assertEqual([m["content"][0]["text"] for m in items_of(rest, "message")], ["Two lattes."])
        self.assertEqual(task.settled, [("steer_1", "applied")])
        self.assertEqual((count(provider.payloads[1], TWO), provider.opened), (1, 2))

    async def test_a_typed_run_that_lands_brings_its_settings_to_the_round_that_takes_it_in(self):
        # A typed run brings the session's settings of its time; the IM makes them the Work's when it lands, so the
        # round that reads the typed words is sent them.
        gate, slow = gated("One latte.")
        provider = ScriptedProvider(slow, text_round("One decaf latte.", "resp_2"))
        brings = {"instructions": "NEW instructions.", "tools": [SET_DECAF]}
        task = TypedRunTask(brings, instructions="OLD instructions.")
        run = adapter(provider).start(task)
        events = run.events()
        await read_until_created(events, task)
        task.steer_with(DECAF)
        run.steer()
        gate.set()
        await read_rest(events, task)
        landing = provider.payloads[1]
        self.assertEqual(task.settled, [("steer_1", "applied")])
        self.assertEqual(count(landing, DECAF), 1)
        self.assertEqual(landing["instructions"], "NEW instructions.")
        self.assertIn("set_decaf", [tool.get("name") for tool in landing["tools"]])

    async def test_one_restart_per_work_the_next_steering_waits_for_the_round_to_end(self):
        first_gate, first = gated("One latte.")
        second_gate, second = gated("Two lattes.", "resp_2")
        provider = ScriptedProvider(first, second, text_round("Two oat lattes.", "resp_3"))
        task = IMTask()
        run = adapter(provider).start(task)
        events = run.events()
        await read_until_created(events, task)
        task.steer_with(TWO, "steer_1")
        run.steer()
        first_gate.set()
        self.assertEqual((await anext(events))["type"], "response.incomplete")
        task.see(await anext(events))  # the restarted round is open
        task.steer_with(OAT, "steer_2")
        run.steer()  # no second restart: this round goes on
        second_gate.set()
        rest = await read_rest(events, task)
        self.assertEqual(
            [(m["content"][0]["text"], m["phase"]) for m in items_of(rest, "message")],
            [("Two lattes.", "commentary"), ("Two oat lattes.", "final_answer")],
        )
        self.assertEqual(task.settled, [("steer_1", "applied"), ("steer_2", "applied")])
        self.assertEqual(provider.opened, 3)
        self.assertEqual((count(provider.payloads[2], TWO), count(provider.payloads[2], OAT)), (1, 1))

    async def test_steering_after_a_call_lands_at_the_next_round_and_the_call_runs(self):
        provider = ScriptedProvider(call_round("lookup", {}), text_round("Done.", "resp_2"))
        shop, task = RecordingTools(delay=0.1), IMTask()
        run = adapter(provider, tools=[shop]).start(task)
        events = run.events()
        async for event in events:
            task.see(event)
            if event["type"] == "response.completed":
                break
        task.steer_with(TWO)
        run.steer()  # the round has ended: nothing to restart
        rest = await read_rest(events, task)
        # The commit gate decided this call's route at the round's completion, before the steering came.
        self.assertEqual(task.outcomes["call_1"][1], "done")
        self.assertEqual(task.settled, [("steer_1", "applied")])
        self.assertEqual(count(provider.payloads[1], TWO), 1)
        self.assertNotIn("response.incomplete", kinds(rest))

    async def test_steering_while_waiting_for_the_app_lands_with_its_outputs(self):
        provider = ScriptedProvider(call_round("get_menu", {}), text_round("Done.", "resp_2"))
        task = IMTask()
        run = adapter(provider).start(task)
        async for event in run.events():
            task.see(event)
        task.steer_with(TWO)
        run.steer()
        task.give_outputs([{"type": "function_call_output", "call_id": "call_1", "output": "{}"}])
        await run.resume()
        await read_rest(run.events(), task)
        sent = provider.payloads[1]["input"]
        self.assertEqual(
            [item.get("type") or "message" for item in sent[-3:]], ["function_call", "function_call_output", "message"]
        )
        self.assertEqual((sent[-1], task.settled), (TWO[0], [("steer_1", "applied")]))

    async def test_every_batch_waiting_at_a_safe_point_lands_in_the_next_round_together(self):
        # Two requests joined while the work waited for the app: both reach the next round, not one round each.
        provider = ScriptedProvider(call_round("get_menu", {}), text_round("Two, with oat milk.", "resp_2"))
        task = IMTask()
        run = adapter(provider).start(task)
        async for event in run.events():
            task.see(event)
        task.steer_with(TWO)
        task.steer_with(OAT, "steer_2")
        run.steer()
        task.give_outputs([{"type": "function_call_output", "call_id": "call_1", "output": "{}"}])
        await run.resume()
        await read_rest(run.events(), task)
        self.assertEqual(len(provider.payloads), 2)
        self.assertEqual(provider.payloads[1]["input"][-2:], [TWO[0], OAT[0]])
        self.assertEqual(task.settled, [("steer_1", "applied"), ("steer_2", "applied")])

    async def test_two_pulls_take_a_batch_once_and_a_retry_does_not_send_it_twice(self):
        gate, slow = gated("One latte.")
        dropped = APIConnectionError(request=httpx.Request("POST", "http://localhost:1/v1"))
        provider = ScriptedProvider(slow, [dropped], text_round("Two lattes.", "resp_2"))
        task = IMTask()
        run = adapter(provider, settings=LlmSettings(provider_retries=0)).start(task)
        events = run.events()
        await read_until_created(events, task)
        task.steer_with(TWO)
        run.steer()
        gate.set()
        await read_rest(events, task)
        self.assertEqual(task.settled, [("steer_1", "applied")])
        self.assertEqual([count(payload, TWO) for payload in provider.payloads], [0, 1, 1])


if __name__ == "__main__":
    unittest.main()
