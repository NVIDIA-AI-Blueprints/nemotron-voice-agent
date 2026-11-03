# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D103

"""Tests for a confirmed stop: "stopped" means stopped."""

import asyncio
import time
import unittest

from delegation_core.backend.llm import LlmSettings
from tests.unit.contracts.backend import call_round, text_round
from tests.unit.delegation_core.backend.llm.doubles import (
    FAST,
    Hang,
    IMTask,
    RecordingTools,
    ScriptedProvider,
    adapter,
    drive,
    kinds,
)

STEERING = [{"type": "message", "role": "user", "content": "Stop that."}]


def gated(text="Old."):
    gate = asyncio.Event()
    round_ = text_round(text)
    round_.insert(1, gate)
    return gate, round_


class StopTests(unittest.IsolatedAsyncioTestCase):
    async def test_a_stop_before_the_first_round_runs_nothing(self):
        provider, task = ScriptedProvider(text_round("Done.")), IMTask()
        run = adapter(provider).start(task)
        self.assertEqual(await run.stop(), "stopped")
        self.assertEqual((await drive(run, task), provider.opened), ([], 0))

    async def test_a_stop_cuts_a_streaming_round_and_nothing_follows_it(self):
        gate, slow = gated()
        provider, task = ScriptedProvider(slow), IMTask()
        run = adapter(provider).start(task)
        events = run.events()
        self.assertEqual((await anext(events))["type"], "response.created")
        self.assertEqual(await asyncio.wait_for(run.stop(), 2), "stopped")
        gate.set()
        self.assertEqual([event async for event in events], [])
        self.assertTrue(run.stopped)

    async def test_a_stop_lets_started_calls_finish_and_runs_no_next_round(self):
        provider = ScriptedProvider(call_round("lookup", {}), text_round("Done.", "resp_2"))
        shop, task = RecordingTools(delay=0.2), IMTask()
        run = adapter(provider, tools=[shop]).start(task)
        reading = asyncio.create_task(drive(run, task))
        while not shop.calls:
            await asyncio.sleep(0.01)
        self.assertEqual(await run.stop(), "stopped")
        self.assertEqual(task.outcomes["call_1"][1], "done")  # it finished before stop() returned
        events = await reading
        self.assertEqual(kinds(events)[-1], "response.completed")
        self.assertEqual(provider.opened, 1)

    async def test_nothing_is_dispatched_after_a_stop(self):
        provider = ScriptedProvider(call_round("lookup", {}), text_round("Done.", "resp_2"))
        shop, task = RecordingTools(), IMTask()
        run = adapter(provider, tools=[shop]).start(task)
        events = run.events()
        async for event in events:
            if event["type"] == "response.completed":
                break  # the round with the call is passed on; its call is not dispatched yet
        self.assertEqual(await run.stop(), "stopped")
        self.assertEqual([event async for event in events], [])
        self.assertEqual((shop.calls, task.started), ([], []))

    async def test_a_stop_between_an_attempt_and_its_retry_opens_nothing(self):
        failed = [
            {"type": "response.created", "response": {"id": "resp_f", "status": "in_progress"}},
            {
                "type": "response.failed",
                "response": {"id": "resp_f", "status": "failed", "error": {"code": "server_error"}},
            },
        ]
        provider, task = ScriptedProvider(failed, text_round("Done.")), IMTask()
        run = adapter(provider).start(task)
        seen = []
        async for event in run.events():
            seen.append(event["type"])
            if event["type"] == "response.failed":
                self.assertEqual(await run.stop(), "stopped")  # the safe point before the retry
        self.assertEqual((seen, provider.opened), (["response.created", "response.failed"], 1))

    async def test_a_stop_while_waiting_for_the_app_or_after_the_answer_is_confirmed_at_once(self):
        provider, task = ScriptedProvider(call_round("get_menu", {})), IMTask()
        run = adapter(provider).start(task)
        await drive(run, task)
        self.assertEqual(await run.stop(), "stopped")
        await run.resume()
        self.assertEqual(await drive(run, task), [])
        answered, task = ScriptedProvider(text_round("Done.")), IMTask()
        run = adapter(answered).start(task)
        await drive(run, task)
        self.assertEqual(await run.stop(), "stopped")

    async def test_a_hung_provider_gives_unconfirmed_at_the_limit(self):
        round_ = text_round("Old.")
        round_.insert(1, Hang(1.0))
        provider, task = ScriptedProvider(round_), IMTask()
        run = adapter(provider, settings=LlmSettings(stop_timeout_seconds=0.1)).start(task)
        events = run.events()
        await anext(events)
        started = time.monotonic()
        self.assertEqual(await run.stop(), "unconfirmed")
        self.assertLess(time.monotonic() - started, 0.5)
        self.assertEqual([event async for event in events], [])

    async def test_a_round_cut_for_steering_that_does_not_let_go_still_holds_the_stop(self):
        hung = text_round("Old.")
        hung.insert(1, Hang(1.0))
        gate, slow = gated("New.")
        provider, task = ScriptedProvider(hung, slow), IMTask()
        run = adapter(provider, settings=LlmSettings(stop_timeout_seconds=0.1)).start(task)
        events = run.events()
        await anext(events)
        task.steer_with(STEERING)
        run.steer()  # the first round is cut but its reader hangs; the restarted round streams
        self.assertEqual((await anext(events))["type"], "response.incomplete")
        await anext(events)
        self.assertEqual(await run.stop(), "unconfirmed")
        gate.set()
        self.assertEqual([event async for event in events], [])

    async def test_steering_that_waits_when_the_run_answers_and_stops_is_never_taken(self):
        gate, slow = gated("One coffee.")
        provider, task = ScriptedProvider(slow, text_round("Two.", "resp_2")), IMTask()
        run = adapter(provider, settings=FAST).start(task)
        events = run.events()
        await anext(events)
        run.redone = True  # its one restart was used: the steering waits for the round's end
        task.steer_with(STEERING)
        gate.set()
        async for event in events:
            if event["type"] == "response.completed":
                await run.stop()
        self.assertEqual((task.settled, len(task._steering)), ([], 1))  # the IM keeps it for the next request
        self.assertEqual(provider.opened, 1)


if __name__ == "__main__":
    unittest.main()
