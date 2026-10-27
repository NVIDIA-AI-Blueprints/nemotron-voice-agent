# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D103, D107

"""The emulator: both contracts of the backend port, and what its scripts do."""

import asyncio
import json
import unittest

from delegation_core.backend.port import Answer, Cancel, ConductorInput, Start, Steer
from tests.unit.contracts.backend import (
    BackendAdapterContract,
    ConductorContract,
    FakeDecisionTarget,
    FakeTaskContext,
    a_request,
    an_ended_work,
    an_open_work,
    check_events,
)
from tests.unit.contracts.base import collect
from tests.unit.delegation_core.emulator import EmulatedFailure, EmulatorSource, emulator_source

# Each scenario of the adapter contract, as the emulator's script. The emulator never runs an attempt again, so the
# contract's "tail_error" (a failure after a completed attempt) has nothing to script.
SCENARIOS = {
    "answer": "",
    "own_call": "[own:lookup]",
    "tool_fails": "[own-fail:lookup]",
    "app_call": "[say:Let me look.] [client:get_menu {}]",
    "slow": "[slow:30]",
    "tail_error": "",
}
MENU = {"type": "function", "name": "get_menu", "parameters": {"type": "object", "properties": {}}}


def user(*parts):
    return {"type": "message", "role": "user", "content": [{"type": "input_text", "text": part} for part in parts]}


def task(*texts, **kwargs):
    return FakeTaskContext(history=[user(text) for text in texts], declared=[MENU], **kwargs)


def items(events, kind):
    return [e["item"] for e in events if e["type"] == "response.output_item.done" and e["item"]["type"] == kind]


def answer(events):
    messages = [item for item in items(events, "message") if item.get("phase") == "final_answer"]
    return messages[-1]["content"][0]["text"] if messages else None


class ShowTool:
    """The IM's work tools, as a task gets them: ``show`` only."""

    label = "voiceclaw"

    def __init__(self):
        self.calls = []

    def tools(self):
        return [{"type": "function", "name": "show", "parameters": {"type": "object", "properties": {}}}]

    async def execute(self, call_id, name, arguments):
        self.calls.append((call_id, name, json.loads(arguments)))
        return "shown"


# ------------------------------------------------------------------------------------------------- the contracts
class EmulatorAdapterContractTests(BackendAdapterContract, unittest.IsolatedAsyncioTestCase):
    def make_adapter(self, scenario):
        self.source = emulator_source({"script": SCENARIOS[scenario]})
        return self.source.new()

    def release(self):
        self.source.release()

    def sent_instructions(self):
        return [
            request["instructions"]
            for adapter in self.source.adapters
            for run in adapter.runs
            for request in run.requests
        ]


class EmulatorConductorContractTests(ConductorContract, unittest.IsolatedAsyncioTestCase):
    def make_conductor(self):
        return emulator_source().new().conductor()


# ------------------------------------------------------------------------------------------------- the source
class SourceTests(unittest.IsolatedAsyncioTestCase):
    async def test_its_block_is_validated_and_names_what_it_serves(self):
        source = emulator_source({"max_running": 2, "models": ["org/a"], "unsupported": {"text": "No verbosity."}})
        self.assertEqual(source.describe(), {"adapter": "emulator", "max_running": 2, "models": ["org/a"]})
        self.assertEqual(source.new().max_running, 2)
        await source.check_model("org/a")
        with self.assertRaisesRegex(ValueError, "not 'org/b'"):
            await source.check_model("org/b")
        await emulator_source().check_model("any/model")
        self.assertEqual(source.unsupported({"model": "org/a", "text": {}}), [("text", "No verbosity.")])
        for block in (
            {"max_runing": 2},
            {"max_running": 0},
            {"max_running": True},
            {"delivery": "maybe"},
            {"stop_outcome": "gone"},
            {"stop_delay_seconds": -1},
            {"models": 5},
            {"script": 3},
        ):
            with self.subTest(block=block), self.assertRaises(ValueError):
                EmulatorSource.from_settings(block, endpoints=None, tools=tuple, vocabulary=None)

    async def test_one_shot_sessions_get_an_adapter_of_their_own(self):
        source = emulator_source()
        self.assertIsNot(source.new_one_shot(), source.new())
        self.assertEqual(len(source.adapters), 2)
        await source.aclose()
        self.assertTrue(source.closed)


# ------------------------------------------------------------------------------------------------- the conductor
def given(*arrivals, works=()):
    return ConductorInput(tuple(arrivals), tuple(works), (), 0)


class ConductorTests(unittest.IsolatedAsyncioTestCase):
    async def test_each_directive_places_its_decision_and_the_outcome_lines_are_kept(self):
        conductor, target = emulator_source().new().conductor(), FakeDecisionTarget()
        arrivals = (
            a_request("d1", "Plan my day [start:menu|[own:hours]]"),
            a_request("d2", "Make it green [steer:W1.2]"),
            a_request("d3", "Never mind [cancel:W1] [answer:Stopped it.]"),
            a_request("d4", "Also this [steer:W1]"),
            a_request("d5", "Whatever [cancel:W2.1]"),
            a_request("d6", "Hold it [skip]"),
            a_request("d7", "And one more [steer]"),
        )
        await conductor.decide(given(*arrivals, works=(an_open_work("W1", "d0"),)), target.place)
        self.assertEqual(
            target.decisions,
            [
                Start(("d1",), ("menu", "[own:hours]")),
                Steer("W1", ("d2",), "W1.2"),
                Cancel("W1"),
                Answer("d3", "Stopped it."),
                Steer("W1", ("d4",)),
                Cancel("W2", "W2.1"),
                Start(("d5",)),
                Steer("W1", ("d7",)),
            ],
        )
        self.assertEqual(len(conductor.outcomes), len(target.decisions))
        self.assertEqual(conductor.outcomes[0], "Started W1 with 2 task(s).")
        self.assertEqual(len(conductor.given), 1)

    async def test_steer_joins_the_newest_open_work_and_starts_when_only_ended_ones_are_listed(self):
        conductor, target = emulator_source().new().conductor(), FakeDecisionTarget()
        ended = an_ended_work("W0")
        await conductor.decide(given(a_request("d1", "Find teas [steer]"), works=(ended,)), target.place)
        await conductor.decide(given(a_request("d2", "Also [steer]"), works=(an_open_work("W1"), ended)), target.place)
        self.assertEqual(target.decisions, [Start(("d1",)), Steer("W1", ("d2",))])

    async def test_raise_fails_the_decide(self):
        conductor, target = emulator_source().new().conductor(), FakeDecisionTarget()
        with self.assertRaisesRegex(RuntimeError, "d2"):
            await conductor.decide(given(a_request("d2", "Boom [raise]")), target.place)

    async def test_think_holds_the_decide_before_it_places_anything(self):
        conductor, target = emulator_source().new().conductor(), FakeDecisionTarget()
        decide = asyncio.create_task(conductor.decide(given(a_request("d1", "Slowly [think:30]")), target.place))
        await asyncio.sleep(0.01)
        self.assertEqual(target.decisions, [])  # nothing placed yet: the IM may still restart it
        decide.cancel()
        await asyncio.gather(decide, return_exceptions=True)
        self.assertTrue(decide.cancelled())


# ---------------------------------------------------------------------------------------------------- a task's run
class RunTests(unittest.IsolatedAsyncioTestCase):
    async def run_task(self, context, block=None):
        source = emulator_source(block)
        run = source.new().start(context)
        return run, await collect(run.events())

    async def test_the_answer_says_the_newest_request_without_its_directives(self):
        run, events = await self.run_task(task("Earlier.", "Find teas [own:lookup] please"))
        self.assertEqual(check_events(events), [])
        self.assertEqual(answer(events), "Done: Find teas please")
        self.assertEqual(run.round, 2)

    async def test_a_brief_given_as_its_own_message_is_the_tasks_script(self):
        _, events = await self.run_task(task("Plan my day [start:[reply:x]|[reply:y]]", "Task: [reply:Open at nine.]"))
        self.assertEqual(answer(events), "Open at nine.")

    async def test_the_records_block_in_a_second_part_is_never_read_as_a_script(self):
        context = FakeTaskContext(history=[user("Then the hours.", "Work so far: W1 Find teas [own:lookup] done")])
        _, events = await self.run_task(context)
        self.assertEqual(items(events, "function_call"), [])
        self.assertEqual(answer(events), "Done: Then the hours.")

    async def test_route_decides_who_runs_each_call(self):
        tools = ShowTool()
        context = task('Go [client:get_menu {"size": "large"}] [own:lookup] [show:A table] [own:never]', tools=tools)
        context.refused = {"call_4"}
        run, first = await self.run_task(context)
        calls = items(first, "function_call")
        self.assertEqual([call["name"] for call in calls], ["get_menu", "lookup", "show", "never"])
        self.assertEqual(json.loads(calls[0]["arguments"]), {"size": "large"})
        self.assertEqual(first[-1]["type"], "response.completed")  # one round, ended for the app's call
        self.assertEqual(
            context.started, [("call_2", "emulator"), ("call_3", "voiceclaw")]
        )  # the refused one never ran
        self.assertEqual(tools.calls, [("call_3", "show", {"display": "A table"})])
        self.assertEqual(context.order, ["started:call_2", "outcome:call_2", "started:call_3", "outcome:call_3"])
        context.give_outputs([{"type": "function_call_output", "call_id": "call_1", "output": "{}"}])
        await run.resume()
        rest = await collect(run.events())
        self.assertEqual(check_events(first + rest), [])
        self.assertEqual(answer(rest), "Done: Go")
        self.assertIn({"type": "function_call_output", "call_id": "call_1", "output": "{}"}, context.items)

    async def test_commentary_comes_before_the_calls_and_a_message_after_them_opens_the_next_round(self):
        _, events = await self.run_task(task("Go [say:Looking.] [own:a] [say:Found it.] [reply:Here.]"))
        self.assertEqual(check_events(events), [])
        phases = [(item.get("phase"), item["content"][0]["text"]) for item in items(events, "message")]
        self.assertEqual(phases, [("commentary", "Looking."), ("commentary", "Found it."), ("final_answer", "Here.")])
        self.assertEqual(sum(e["type"] == "response.created" for e in events), 2)

    async def test_notes_failures_and_an_empty_round(self):
        context = task("Go [progress:Checking the menu.] [fail:429]")
        with self.assertRaises(EmulatedFailure) as failed:
            await self.run_task(context)
        self.assertEqual((failed.exception.status_code, context.notes), (429, ["Checking the menu."]))
        _, events = await self.run_task(task("Go [own:a] [empty]"))
        self.assertEqual(events[-1]["response"]["output"], [])
        self.assertIsNone(answer(events))

    async def test_steering_not_applied_leaves_the_answer_alone(self):
        context = task("Find teas [slow:30]")
        run = emulator_source({"delivery": "not_applied"}).new().start(context)
        events = run.events()
        await anext(events)
        steering = context.steer_with([user("Green only [own:filter]")])
        run.steer()
        rest = await collect(events)
        self.assertEqual(context.settled, [(steering.id, "not_applied")])
        self.assertEqual(answer(rest), "Done: Find teas")
        self.assertEqual(items(rest, "function_call"), [])

    async def test_applied_steering_redoes_the_attempt_and_its_directives_join_the_script(self):
        context = task("Find teas [slow:30]")
        source = emulator_source()
        run = source.new().start(context)
        events = run.events()
        opened = await anext(events)
        context.steer_with([user("Green only [own:filter]")])
        run.steer()
        rest = await collect(events)
        self.assertEqual(rest[0]["type"], "response.incomplete")  # the attempt that ran nothing is dropped
        self.assertEqual(rest[0]["response"]["incomplete_details"], {"reason": "steered"})
        self.assertEqual(check_events([opened, *rest]), [])
        self.assertEqual([call["name"] for call in items(rest, "function_call")], ["filter"])
        self.assertEqual(answer(rest), "Done: Green only")
        self.assertEqual(run.round, 2)  # the redone attempt is the same round

    async def test_a_slow_stop_returns_its_outcome_after_its_delay(self):
        context = task("Find teas [slow:30]")
        run = emulator_source({"stop_delay_seconds": 0.05, "stop_outcome": "unconfirmed"}).new().start(context)
        events = run.events()
        await anext(events)
        loop = asyncio.get_running_loop()
        started = loop.time()
        self.assertEqual(await run.stop(), "unconfirmed")
        self.assertGreaterEqual(loop.time() - started, 0.04)
        self.assertEqual(await collect(events), [])

    async def test_call_ids_are_unique_across_the_runs_of_one_source(self):
        source = emulator_source({"script": "[own:a]"})
        adapter = source.new()
        first, second = task("One."), task("Two.")
        await collect(adapter.start(first).events())
        await collect(source.new().start(second).events())
        self.assertEqual([call_id for call_id, _ in first.started + second.started], ["call_1", "call_2"])


if __name__ == "__main__":
    unittest.main()
