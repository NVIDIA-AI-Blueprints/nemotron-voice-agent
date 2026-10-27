# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D103

"""Tests for answer phases: a completed ``final_answer``, empty rounds and crossed answers."""

import asyncio
import unittest

from delegation_core.backend.llm.run import RoundFailedError
from tests.unit.contracts.backend import a_round, call_round, check_events, message_events, text_round
from tests.unit.delegation_core.backend.llm.doubles import (
    IMTask,
    ScriptedProvider,
    adapter,
    drive,
    items_of,
    kinds,
    source,
)

EMPTY = a_round()  # completed, with nothing in it
STEERING = [{"type": "message", "role": "user", "content": "Actually, two."}]


class PhaseTests(unittest.IsolatedAsyncioTestCase):
    async def test_every_message_carries_its_phase_and_calls_come_last(self):
        provider = ScriptedProvider(call_round("lookup", {}, said="Let me look."), text_round("Done.", "resp_2"))
        task = IMTask()
        events = await drive(adapter(provider).start(task), task)
        self.assertEqual(check_events(events), [])
        self.assertEqual([m["phase"] for m in items_of(events, "message")], ["commentary", "final_answer"])
        first_round = [e for e in events[: kinds(events).index("response.completed") + 1]]
        done = [e["item"]["type"] for e in first_round if e["type"] == "response.output_item.done"]
        self.assertEqual(done, ["message", "function_call"])

    async def test_a_phase_the_model_gave_is_kept_and_a_round_of_commentary_answers_with_its_last_message(self):
        given = a_round(
            message_events("Working on it.", item_id="m1", phase="commentary"),
            message_events("Here it is.", item_id="m2", index=1, phase="final_answer"),
        )
        only_commentary = a_round(
            message_events("First.", item_id="m1", phase="commentary"),
            message_events("Second.", item_id="m2", index=1, phase="commentary"),
        )
        for script, expected in (
            (given, ["commentary", "final_answer"]),
            (only_commentary, ["commentary", "final_answer"]),
        ):
            task = IMTask()
            events = await drive(adapter(ScriptedProvider(script)).start(task), task)
            self.assertEqual([m["phase"] for m in items_of(events, "message")], expected)

    async def test_an_empty_round_is_retried_once(self):
        provider = ScriptedProvider(EMPTY, text_round("Done."))
        task = IMTask()
        events = await drive(adapter(provider).start(task), task)
        self.assertEqual(provider.opened, 2)
        self.assertEqual(
            kinds(events)[:3], ["response.created", "response.failed", "response.created"]
        )  # the empty attempt is closed failed, never completed
        self.assertEqual(events[1]["response"]["error"]["message"], "The backend's round answered nothing")
        self.assertEqual(check_events(events), [])
        self.assertEqual(items_of(events, "message")[0]["phase"], "final_answer")

    async def test_an_empty_round_twice_fails_the_task(self):
        provider = ScriptedProvider(EMPTY, EMPTY)
        task, events = IMTask(), []
        with self.assertRaisesRegex(RoundFailedError, "answered nothing"):
            async for event in adapter(provider).start(task).events():
                events.append(event)
        self.assertEqual(kinds(events), ["response.created", "response.failed"] * 2)

    async def test_a_one_shot_session_never_retries(self):
        provider = ScriptedProvider(EMPTY, text_round("Done."))
        run = source(provider).new_one_shot().start(IMTask())
        with self.assertRaises(RoundFailedError):
            async for _ in run.events():
                pass
        self.assertEqual(provider.opened, 1)

    async def test_an_answer_that_crosses_waiting_steering_becomes_commentary(self):
        gate = asyncio.Event()
        old = text_round("One coffee is on its way.")
        old.insert(1, gate)
        provider = ScriptedProvider(old, text_round("Two coffees are on their way.", "resp_2"))
        task = IMTask()
        run = adapter(provider).start(task)
        events = run.events()
        task.see(await anext(events))  # the round streams; steering arrives with no wake-up (its restart was used)
        run.redone = True
        task.steer_with(STEERING)
        gate.set()
        rest = []
        async for event in events:
            task.see(event)
            rest.append(event)
        self.assertEqual(
            [(m["content"][0]["text"], m["phase"]) for m in items_of(rest, "message")],
            [("One coffee is on its way.", "commentary"), ("Two coffees are on their way.", "final_answer")],
        )
        self.assertEqual(task.settled, [("steer_1", "applied")])
        self.assertIn(STEERING[0], provider.payloads[1]["input"])


if __name__ == "__main__":
    unittest.main()
