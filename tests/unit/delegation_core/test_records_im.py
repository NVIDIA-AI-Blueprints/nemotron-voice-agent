# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D103

"""The IM moves the work ledger's records."""

import asyncio

from delegation_core.im.results import SpeakDisplay
from delegation_core.im.stop import ForNewerRequest
from delegation_core.im.stream import message_text
from tests.unit.delegation_core.helpers import (
    MENU,
    TEXTS,
    IMCase,
    ScriptedAdapter,
    answer,
    app_call,
    ask,
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
            (shown["name"], shown["output"], shown["server_label"]), ("show", "Earl Grey 3.25", "delegation_core")
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
