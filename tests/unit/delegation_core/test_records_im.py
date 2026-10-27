# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D103

"""The IM moves the work ledger's records."""

import asyncio

from delegation_core.backend.port import Start
from delegation_core.im.results import SpeakDisplay
from delegation_core.im.stop import ForNewerRequest
from delegation_core.im.stream import message_text
from tests.unit.delegation_core.helpers import (
    MENU,
    TEXTS,
    IMCase,
    ScriptedAdapter,
    ScriptedConductor,
    answer,
    app_call,
    ask,
    checkpoint,
    own_call,
    read_all,
    show,
    wait,
)

SPEAK_DISPLAY = SpeakDisplay(TEXTS.result_formats["speak_display"])


class TaskRecordTests(IMCase):
    async def test_a_tasks_record_moves_with_its_run_and_keeps_its_times(self):
        gate = asyncio.Event()
        im = self.session(ScriptedAdapter([wait(gate), answer("Two teas.")]))
        stream = im.admit(ask("Teas?"), delegation_id="D1").stream
        await self.turns()
        task = im.ledger.tasks["W1.1"]
        self.assertEqual((task.state, task.work, task.result.state), ("running", "W1", "none"))
        self.assertIsNotNone(task.started_at)
        gate.set()
        await read_all(stream)
        self.assertEqual((task.state, task.outcome, task.stop.state), ("answered", "none", "none"))
        self.assertGreaterEqual(task.ended_at, task.started_at)
        summary = im.ended_works[-1].summary().tasks[0]
        self.assertEqual((summary.state, summary.seconds), ("answered", task.ended_at - task.started_at))

    async def test_a_tasks_result_is_what_it_showed_then_its_answer_pushed_once_written(self):
        adapter = ScriptedAdapter([show("Earl Grey 3.25"), answer("Earl Grey, three twenty-five.")])
        im = self.session(adapter, result_format=SPEAK_DISPLAY)
        await read_all(im.admit(ask("Teas?"), delegation_id="D1").stream)
        result = im.ledger.tasks["W1.1"].result
        shown, said = result.items
        self.assertEqual(
            (shown["name"], shown["output"], shown["server_label"]), ("show", "Earl Grey 3.25", "voiceclaw")
        )
        self.assertEqual((result.state, message_text(said)), ("pushed", "Earl Grey, three twenty-five."))

    async def test_a_stopped_tasks_own_stop_record_says_how_the_stop_went(self):
        gate = asyncio.Event()
        im = self.session(ScriptedAdapter([wait(gate), answer()]))
        im.admit(ask("Book it."), delegation_id="D1")
        await self.turns()
        await im.works["W1"].stop(ForNewerRequest())
        task = im.ledger.tasks["W1.1"]
        self.assertEqual((task.state, task.stop.state, task.stop.outcome), ("cancelled", "stopped", "none"))


class SteeringRecordTests(IMCase):
    async def test_steering_for_a_running_task_is_a_steering_record_whose_id_the_adapter_settles(self):
        gate = asyncio.Event()
        adapter = ScriptedAdapter([wait(gate), checkpoint(), answer()])
        im = self.session(adapter)
        im.admit(ask("Book Denver."), delegation_id="D1")
        await self.turns()
        im.admit(ask("Make it Tuesday."), delegation_id="D2")
        await self.turns()
        (record,) = im.ledger.steerings.values()
        self.assertEqual((record.id, record.task, record.delegations, record.state), ("S1", "W1.1", ["D2"], "waiting"))
        gate.set()
        await self.turns()
        self.assertEqual([steering.id for steering in adapter.runs[0].applied], ["S1"])
        self.assertEqual(record.state, "applied")

    async def test_steering_no_task_applied_ends_follow_up_and_starts_a_follow_up_work(self):
        gate = asyncio.Event()
        adapter = ScriptedAdapter([wait(gate), checkpoint(), answer("Old.")], [answer("New.")])
        adapter.delivery = "not_applied"
        im = self.session(adapter)
        first = im.admit(ask("Book Denver."), delegation_id="D1")
        await self.turns()
        second = im.admit(ask("Make it Tuesday."), delegation_id="D2")
        await self.turns()
        gate.set()
        await read_all(first.stream)
        await read_all(second.stream)
        self.assertEqual(im.ledger.steerings["S1"].state, "follow_up")
        self.assertEqual((im.ledger.delegations["D2"].work, im.ledger.works["W2"].requests), ("W2", ["D2"]))


class ActionRecordTests(IMCase):
    async def test_each_action_names_who_runs_it(self):
        adapter = ScriptedAdapter([own_call("lookup"), show("Menu"), app_call("get_menu", call_id="call_menu")])
        im = self.session(adapter, result_format=SPEAK_DISPLAY)
        im.admit(ask("Teas?"), delegation_id="D1", settings={"tools": [MENU]})
        await self.turns()
        runners = {action.name: (action.task, action.work, action.runner) for action in im.ledger.actions.values()}
        self.assertEqual(
            runners,
            {"lookup": ("W1.1", "W1", "backend"), "show": ("W1.1", "W1", "im"), "get_menu": ("W1.1", "W1", "app")},
        )


class WorkRecordTests(IMCase):
    async def test_a_start_of_several_requests_carries_them_oldest_first_and_the_newest_carries_it(self):
        async def plan(given, place):
            await place(Start(tuple(arrival.delegation_id for arrival in given.arrivals)))

        im = self.session(ScriptedAdapter([answer()], conductor=ScriptedConductor(plan)))
        streams = [im.admit(ask(f"Part {n}."), delegation_id=f"D{n}").stream for n in (1, 2, 3)]
        for stream in streams:
            await read_all(stream)
        work = im.ledger.works["W1"]
        self.assertEqual((work.requests, work.carrier, work.state), (["D1", "D2", "D3"], "D3", "answered"))
        self.assertEqual([im.ledger.delegations[f"D{n}"].handed_to for n in (1, 2)], ["D2", "D3"])
        self.assertIn("Part 1.", str(im.ended_works[-1].requested))  # the input the Work's task read
        self.assertIn("Part 3.", str(im.ended_works[-1].requested))
