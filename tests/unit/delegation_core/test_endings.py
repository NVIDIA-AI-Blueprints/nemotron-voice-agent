# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D103

"""The IM writes every scenario of the endings fixture (``tests/unit/contracts/endings.py``) exactly."""

import asyncio

from delegation_core.backend.port import Answer, Cancel, Start
from delegation_core.im.results import SpeakDisplay
from tests.unit.contracts.endings import load
from tests.unit.delegation_core.helpers import (
    MENU,
    TEXTS,
    IMCase,
    ScriptedAdapter,
    StartEach,
    answer,
    app_call,
    app_outputs,
    ask,
    checkpoint,
    cut,
    fail,
    one_shot_session,
    own_call,
    planned,
    read_all,
    say,
    show,
    wait,
)

SCENARIOS = load()
MENU_SETTINGS = {"tools": [MENU]}


class EndingsTests(IMCase):
    def assertScenario(self, name: str) -> None:
        self.assertEqual(self.timeline(), SCENARIOS[name]["timeline"])

    async def test_answered(self):
        im = self.session(ScriptedAdapter([answer()]))
        await read_all(im.admit(ask("Find teas."), delegation_id="D1").stream)
        self.assertScenario("answered")

    async def test_answered_with_progress_and_tool_items(self):
        im = self.session(ScriptedAdapter([say("Let me look."), own_call("get_menu", label="cafe"), answer()]))
        await read_all(im.admit(ask("Find teas."), delegation_id="D1").stream)
        self.assertScenario("answered_with_progress_and_tool_items")

    async def test_waits_for_the_app_then_answers(self):
        im = self.session(ScriptedAdapter([say("Let me look."), app_call("get_menu"), answer()]))
        first = await read_all(im.admit(ask("Find teas."), delegation_id="D1", settings=MENU_SETTINGS).stream)
        again = im.admit_client_request(app_outputs(first), delegation_id="D1", settings=MENU_SETTINGS)
        await read_all(again.stream)
        self.assertScenario("waits_for_the_app_then_answers")

    async def test_queued_then_running(self):
        im = self.session(ScriptedAdapter([answer("One.")], [answer("Two.")], conductor=StartEach()))
        first = im.admit(ask("Find teas."), delegation_id="D1")
        second = im.admit(ask("And the hours."), delegation_id="D2")
        await read_all(first.stream)
        await read_all(second.stream)
        self.assertScenario("queued_then_running")

    async def test_handed_over_with_the_taker_named(self):
        gate = asyncio.Event()
        im = self.session(ScriptedAdapter([wait(gate), answer()]))
        first = im.admit(ask("Find teas."), delegation_id="D1")
        await self.turns()
        second = im.admit(ask("Green tea only."), delegation_id="D2")
        await self.turns()
        gate.set()
        await read_all(first.stream)
        await read_all(second.stream)
        self.assertScenario("handed_over_with_the_taker_named")

    async def test_a_chain_of_handovers(self):
        first_gate, second_gate = asyncio.Event(), asyncio.Event()
        im = self.session(ScriptedAdapter([wait(first_gate), checkpoint(), wait(second_gate), checkpoint(), answer()]))
        first = im.admit(ask("Find teas."), delegation_id="D1")
        await self.turns()
        second = im.admit(ask("Green tea only."), delegation_id="D2")
        await self.turns()
        first_gate.set()
        await read_all(first.stream)
        third = im.admit(ask("Make it two."), delegation_id="D3")
        await self.turns()
        second_gate.set()
        await read_all(second.stream)
        await read_all(third.stream)
        self.assertScenario("a_chain_of_handovers")

    async def test_answered_by_the_conductor(self):
        gate = asyncio.Event()

        async def start(given, place):
            await place(Start(("D1",)))

        async def status(given, place):
            await place(Answer("D2", "Still looking at the teas."))

        im = self.session(ScriptedAdapter([wait(gate), answer()], conductor=planned(start, status)))
        first = im.admit(ask("Find teas."), delegation_id="D1")
        await self.turns()
        second = im.admit(ask("How is it going?"), delegation_id="D2")
        await read_all(second.stream)
        gate.set()
        await read_all(first.stream)
        self.assertScenario("answered_by_the_conductor")

    async def cancelled_for_a_newer_request(self, stop_outcome: str) -> None:
        gate = asyncio.Event()

        async def start(given, place):
            await place(Start(("D1",)))

        async def cancel(given, place):
            await place(Cancel("W1"))
            await place(Answer("D2", "Stopped the tea search."))

        adapter = ScriptedAdapter([wait(gate), answer()], conductor=planned(start, cancel), stop_outcome=stop_outcome)
        im = self.session(adapter)
        first = im.admit(ask("Find teas."), delegation_id="D1")
        await self.turns()
        second = im.admit(ask("Never mind."), delegation_id="D2")
        await read_all(second.stream)
        await read_all(first.stream)

    async def test_cancelled_for_a_newer_request_stopped(self):
        await self.cancelled_for_a_newer_request("stopped")
        self.assertScenario("cancelled_for_a_newer_request_stopped")

    async def test_cancelled_for_a_newer_request_unconfirmed(self):
        await self.cancelled_for_a_newer_request("unconfirmed")
        self.assertScenario("cancelled_for_a_newer_request_unconfirmed")

    async def test_failed_with_an_error(self):
        im = self.session(ScriptedAdapter([fail()]))
        await read_all(im.admit(ask("Find teas."), delegation_id="D1").stream)
        self.assertScenario("failed_with_an_error")

    async def test_late_continuation(self):
        im = self.session(ScriptedAdapter([app_call("get_menu"), answer()]), client_tool_timeout_seconds=0.05)
        first = await read_all(im.admit(ask("Find teas."), delegation_id="D1", settings=MENU_SETTINGS).stream)
        await asyncio.sleep(0.1)
        late = im.admit_client_request(app_outputs(first), delegation_id="D1", settings=MENU_SETTINGS)
        await read_all(late.stream)
        self.assertScenario("late_continuation")

    async def test_deadline_while_waiting(self):
        im = self.session(ScriptedAdapter([app_call("get_menu"), answer()]), client_tool_timeout_seconds=0.05)
        await read_all(im.admit(ask("Find teas."), delegation_id="D1", settings=MENU_SETTINGS).stream)
        await asyncio.sleep(0.1)
        self.assertEqual(im.ledger.works["W1"].state, "failed")
        self.assertScenario("deadline_while_waiting")

    async def test_cancelled_at_close(self):
        never = asyncio.Event()
        adapter = ScriptedAdapter([wait(never), answer()], [answer()], conductor=StartEach())
        im = self.session(adapter, close_grace_seconds=0.05)
        first = im.admit(ask("Find teas."), delegation_id="D1")
        await self.turns()
        second = im.admit(ask("And the hours."), delegation_id="D2")
        await self.turns()
        await im.close()
        await read_all(first.stream)
        await read_all(second.stream)
        self.assertScenario("cancelled_at_close")

    async def test_a_typed_run(self):
        im = self.session(ScriptedAdapter([answer()]))
        await read_all(im.admit_client_request(ask("Typed question."), delegation_id="D1").stream)
        self.assertScenario("a_typed_run")

    async def test_stateless(self):
        im = one_shot_session(ScriptedAdapter([answer()]))
        im.start()
        self.addAsyncCleanup(im.close)
        await read_all(im.admit(ask("Find teas."), delegation_id="D1").stream)
        self.assertScenario("stateless")

    async def test_show_then_the_answer(self):
        im = self.session(
            ScriptedAdapter([show("Earl Grey 3.25"), answer("Earl Grey is 3.25.")]),
            result_format=SpeakDisplay(TEXTS.result_formats["speak_display"]),
        )
        await read_all(im.admit(ask("Find teas."), delegation_id="D1").stream)
        self.assertScenario("show_then_the_answer")

    async def test_two_works_answering_at_once(self):
        first_gate, first_hold, second_hold = asyncio.Event(), asyncio.Event(), asyncio.Event()
        adapter = ScriptedAdapter(
            [wait(first_gate), answer("One.", hold=first_hold)],
            [answer("Two.", hold=second_hold)],
            conductor=StartEach(),
            max_running=2,
        )
        im = self.session(adapter)
        first = im.admit(ask("Find teas."), delegation_id="D1")
        await self.turns()
        second = im.admit(ask("And the hours."), delegation_id="D2")
        await self.turns()
        first_gate.set()
        await self.turns()
        second_hold.set()
        await read_all(second.stream)
        first_hold.set()
        await read_all(first.stream)
        self.assertScenario("two_works_answering_at_once")

    async def test_one_work_two_tasks(self):
        async def split(given, place):
            await place(Start(("D1",), ("Find teas.", "Find the hours.")))

        adapter = ScriptedAdapter([answer("Teas.")], [answer("Hours.")], conductor=planned(split), max_running=2)
        im = self.session(adapter)
        await read_all(im.admit(ask("Find teas and the hours."), delegation_id="D1").stream)
        self.assertScenario("one_work_two_tasks")

    async def test_a_task_waits_while_another_runs(self):
        gate = asyncio.Event()

        async def split(given, place):
            await place(Start(("D1",), ("Show the menu.", "Find the hours.")))

        adapter = ScriptedAdapter(
            [app_call("get_menu"), answer("Menu.")],
            [wait(gate), answer("Hours.")],
            conductor=planned(split),
            max_running=2,
        )
        im = self.session(adapter)
        first = await read_all(im.admit(ask("Menu and hours."), delegation_id="D1", settings=MENU_SETTINGS).stream)
        gate.set()
        await self.turns()
        again = im.admit_client_request(app_outputs(first), delegation_id="D1", settings=MENU_SETTINGS)
        await read_all(again.stream)
        self.assertScenario("a_task_waits_while_another_runs")

    async def test_cut_by_max_output_tokens(self):
        im = self.session(ScriptedAdapter([cut("max_output_tokens")]))
        await read_all(im.admit(ask("Find teas."), delegation_id="D1").stream)
        self.assertScenario("cut_by_max_output_tokens")

    async def test_two_works_waiting_for_the_app(self):
        first_gate, second_gate = asyncio.Event(), asyncio.Event()
        adapter = ScriptedAdapter(
            [wait(first_gate), app_call("get_menu"), answer("One.")],
            [wait(second_gate), app_call("get_menu"), answer("Two.")],
            conductor=StartEach(),
            max_running=2,
        )
        im = self.session(adapter)
        first = im.admit(ask("Find teas."), delegation_id="D1", settings=MENU_SETTINGS)
        await self.turns()
        second = im.admit(ask("And coffee."), delegation_id="D2", settings=MENU_SETTINGS)
        await self.turns()
        first_gate.set()
        first_events = await read_all(first.stream)
        second_gate.set()
        second_events = await read_all(second.stream)
        third = im.admit_client_request(app_outputs(first_events), delegation_id="D1", settings=MENU_SETTINGS)
        await read_all(third.stream)
        fourth = im.admit_client_request(app_outputs(second_events), delegation_id="D2", settings=MENU_SETTINGS)
        await read_all(fourth.stream)
        self.assertScenario("two_works_waiting_for_the_app")

    def test_every_scenario_has_its_test(self):
        tested = {name.removeprefix("test_") for name in dir(self) if name.startswith("test_")}
        self.assertLessEqual(set(SCENARIOS), tested)
