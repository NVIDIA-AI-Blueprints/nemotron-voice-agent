# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D103

"""Per-call routing, reports and the commit gate."""

import asyncio
import json

from delegation_core.backend.port import Answer, Start, Steer
from delegation_core.im.results import SpeakDisplay
from delegation_core.im.stop import ForNewerRequest
from tests.unit.delegation_core.helpers import (
    CORE_SETTINGS,
    MENU,
    TEXTS,
    IMCase,
    ScriptedAdapter,
    ScriptStep,
    answer,
    app_outputs,
    ask,
    call_item,
    finished_items,
    own_call,
    planned,
    read_all,
    show,
    wait,
)

CORE_DISPLAY_LABEL = CORE_SETTINGS.delegation.display_label


class MixedRound(ScriptStep):
    """One round with two backend calls around an app call; the backend reports them out of order."""

    ends = True

    async def play(self, run):
        calls = [
            call_item("lookup", {}, "call_a"),
            call_item("get_menu", {}, "call_b"),
            call_item("price", {}, "call_c"),
        ]
        for event in run.round_events(calls):
            yield event
        run.routes = [run.task.route(call) for call in calls]
        for call in (calls[2], calls[0]):  # out of call order
            run.task.tool_started(call, "cafe")
            run.task.tool_outcome(call["call_id"], f"output {call['call_id']}", "done")


class StreamedRound(ScriptStep):
    """A round that streams until ``streamed`` is set, then routes ``calls`` as the backend LLM's run does."""

    def __init__(self, streamed: asyncio.Event, *calls: dict):
        """Stream until ``streamed`` is set, then call ``calls``."""
        self.streamed, self.calls = streamed, list(calls)

    async def play(self, run):
        run.boundary()
        await self.streamed.wait()  # newer input may arrive while the round streams
        routes = [run.task.route(call) for call in self.calls]
        run.routes += routes
        self.ends = "app" in routes
        for event in run.round_events(self.calls, boundary=False):
            yield event


class BackendCallThenAppCall(ScriptStep):
    """A round whose backend call runs until ``ran`` is set before its app call ends the iteration."""

    ends = True

    def __init__(self, ran: asyncio.Event):
        """Run the backend's call until ``ran`` is set."""
        self.ran = ran

    async def play(self, run):
        calls = [call_item("lookup", {}, "call_lookup"), call_item("get_menu", {}, "call_menu")]
        for event in run.round_events(calls):
            yield event
        run.routes += [run.task.route(call) for call in calls]
        run.task.tool_started(calls[0], "cafe")
        await self.ran.wait()  # newer input may arrive while the backend's call runs
        run.task.tool_outcome("call_lookup", "output call_lookup", "done")


class CancelOwnWork(ScriptStep):
    """A round that calls the work tool cancel_work, run through the task's tools; then a call it may not run."""

    async def play(self, run):
        call = call_item("cancel_work", {}, "call_cancel")
        for event in run.round_events([call]):
            yield event
        tools = run.task.tools()
        run.task.tool_started(call, tools.label)
        run.task.tool_outcome("call_cancel", await tools.execute("call_cancel", "cancel_work", "{}"), "done")
        later = call_item("lookup", {}, "call_later")
        for event in run.round_events([later]):
            yield event
        run.routes.append(run.task.route(later))


class RoutingTests(IMCase):
    async def test_a_mixed_round_routes_each_call_and_keeps_every_output_in_call_order(self):
        adapter = ScriptedAdapter([MixedRound(), answer()])
        im = self.session(adapter)
        menu = {"tools": [MENU]}
        first = await read_all(im.admit(ask("Price the teas."), delegation_id="D1", settings=menu).stream)
        run = adapter.runs[0]
        self.assertEqual(run.routes, ["backend", "app", "backend"])
        self.assertEqual([item["type"] for item in finished_items(first)], ["mcp_call", "mcp_call", "function_call"])
        await read_all(im.admit_client_request(app_outputs(first, "menu"), delegation_id="D1", settings=menu).stream)
        outputs = [item for item in run.histories[-1] if item.get("type") == "function_call_output"]
        self.assertEqual([item["call_id"] for item in outputs], ["call_a", "call_b", "call_c"])
        self.assertEqual([item["output"] for item in outputs], ["output call_a", "menu", "output call_c"])
        self.assertEqual(run.boundaries[-2].outputs, tuple(outputs))  # resume() handed them over, in call order

    async def test_a_call_after_a_stop_was_asked_is_refused_and_answered_by_the_im(self):
        gate = asyncio.Event()
        im = self.session(ScriptedAdapter([wait(gate), answer()]))
        im.admit(ask("Book it."), delegation_id="D1")
        await self.turns()
        task = im.works["W1"].tasks[0]
        task.request_stop(ForNewerRequest())
        self.assertEqual(task.route(call_item("book", {}, "call_book")), "refused")
        given = task.take_boundary_input()
        self.assertEqual(given.outputs[0]["output"], TEXTS.stopped_output)
        self.assertEqual(im.ledger.actions["call_book"].state, "not_run")
        self.assertFalse(im.ledger.tasks["W1.1"].acted)

    async def test_a_call_routed_while_steering_waits_is_refused_until_the_steering_lands(self):
        gate = asyncio.Event()
        im = self.session(ScriptedAdapter([wait(gate), answer()]))
        im.admit(ask("Order two lattes."), delegation_id="D1", settings={"tools": [MENU]})
        await self.turns()
        im.admit(ask("Make it one."), delegation_id="D2")
        await self.turns()
        task = im.works["W1"].tasks[0]
        self.assertEqual(task.route(call_item("get_menu", {}, "c1")), "refused")  # the app's function
        self.assertEqual(task.route(call_item("place_order", {}, "c2")), "refused")  # a backend tool
        given = task.take_boundary_input()
        self.assertEqual([output["output"] for output in given.outputs], [TEXTS.not_run_output] * 2)
        self.assertEqual(task.route(call_item("get_menu", {}, "c3")), "refused")  # taken, not applied yet
        task.settle(given.steering, "not_applied")
        self.assertEqual(task.route(call_item("get_menu", {}, "c4")), "refused")  # waiting again
        task.settle(task.take_boundary_input().steering, "applied")
        self.assertEqual(task.route(call_item("get_menu", {}, "c5")), "app")
        self.assertEqual(task.route(call_item("place_order", {}, "c6")), "backend")
        self.assertEqual([im.ledger.actions[c].state for c in ("c1", "c2", "c3", "c4")], ["not_run"] * 4)
        gate.set()

    async def test_a_round_that_newer_input_reached_runs_none_of_its_calls(self):
        streamed = asyncio.Event()
        calls = [call_item("get_menu", {}, "call_menu"), call_item("place_order", {"lattes": 2}, "call_order")]
        adapter = ScriptedAdapter([StreamedRound(streamed, *calls), answer("Nothing was ordered.")])
        im = self.session(adapter)
        first = im.admit(ask("Order two lattes."), delegation_id="D1", settings={"tools": [MENU]})
        await self.turns()
        second = im.admit(ask("Never mind."), delegation_id="D2")
        await self.turns()
        streamed.set()
        older, taker = await read_all(first.stream), await read_all(second.stream)
        run = adapter.runs[0]
        self.assertEqual(run.routes, ["refused", "refused"])
        self.assertEqual(run.steers, 1)
        self.assertNotIn("function_call", str(older + taker))  # nothing reached the app
        self.assertNotIn("mcp_call", str(older + taker))  # and nothing ran
        history = run.histories[-1]
        outputs = [item for item in history if item.get("type") == "function_call_output"]
        self.assertEqual(
            [(item["call_id"], item["output"]) for item in outputs],
            [("call_menu", TEXTS.not_run_output), ("call_order", TEXTS.not_run_output)],
        )
        newer = next(n for n, item in enumerate(history) if "Never mind." in str(item))
        self.assertLess(history.index(outputs[-1]), newer)  # the next round reads the outputs, then the new words
        self.assertEqual([im.ledger.actions[c].state for c in ("call_menu", "call_order")], ["not_run"] * 2)
        self.assertFalse(im.ledger.tasks["W1.1"].acted)
        self.assertEqual(im.ended_works[-1].tasks[0].step_lines(), ("get_menu: not_run", "place_order: not_run"))
        self.assertEqual(taker[-1]["response"]["output"][-1]["content"][0]["text"], "Nothing was ordered.")
        self.assertEqual(im.ledger.delegations["D1"].handed_to, "D2")

    async def test_newer_input_that_comes_while_a_rounds_backend_calls_run_keeps_its_app_calls_back(self):
        # A round's app call goes out only after its backend call ran; a request that arrives meanwhile holds it
        # back, and the next round reads its not-run output, then the new words. The call that ran is never cut.
        ran = asyncio.Event()
        adapter = ScriptedAdapter([BackendCallThenAppCall(ran), answer("Nothing was ordered.")])
        im = self.session(adapter)
        first = im.admit(ask("Look it up, then get the menu."), delegation_id="D1", settings={"tools": [MENU]})
        await self.turns()
        second = im.admit(ask("Never mind."), delegation_id="D2")
        await self.turns()
        ran.set()
        older = await asyncio.wait_for(read_all(first.stream), 2)
        self.assertEqual([item["type"] for item in finished_items(older)], ["mcp_call", "message"])  # no function_call
        taker = await asyncio.wait_for(read_all(second.stream), 2)
        run = adapter.runs[0]
        self.assertEqual(run.routes, ["backend", "app"])  # decided once, when the round completed
        self.assertEqual(run.resumes, 1)  # the outputs are in: the IM's own, for the call it kept back
        self.assertEqual(
            (im.ledger.actions["call_lookup"].state, im.ledger.actions["call_menu"].state), ("done", "not_run")
        )
        history = run.histories[-1]
        outputs = [item for item in history if item.get("type") == "function_call_output"]
        self.assertEqual(
            [(item["call_id"], item["output"]) for item in outputs],
            [("call_lookup", "output call_lookup"), ("call_menu", TEXTS.not_run_output)],
        )
        newer = next(n for n, item in enumerate(history) if "Never mind." in str(item))
        self.assertLess(history.index(outputs[-1]), newer)  # the next round reads the outputs, then the new words
        self.assertEqual(im.ledger.steerings["S1"].state, "applied")
        self.assertEqual(taker[-1]["response"]["output"][-1]["content"][0]["text"], "Nothing was ordered.")
        self.assertEqual(im.ledger.delegations["D1"].handed_to, "D2")

    async def test_a_call_routed_while_the_conductor_decides_on_newer_input_is_held_until_it_decided(self):
        # Until a model conductor has placed a newer request, no new call of any open Work is dispatched; the call gets
        # its not-run output and its next round reads the output, then the new words.
        ready, deciding = asyncio.Event(), asyncio.Event()

        async def start(given, place):
            await place(Start(("D1",)))

        async def steer_once_decided(given, place):
            await deciding.wait()  # a model conductor's call: the newer request waits for it
            await place(Steer("W1", ("D2",)))

        adapter = ScriptedAdapter(
            [wait(ready), own_call("book_table", call_id="call_book"), answer("Booked, by the window.")],
            conductor=planned(start, steer_once_decided),
        )
        im = self.session(adapter)
        im.admit(ask("Book a table."), delegation_id="D1")
        await self.turns()
        second = im.admit(ask("By the window."), delegation_id="D2")
        await self.turns()
        ready.set()  # the round ends with its call while the conductor still decides
        await self.turns()
        run = adapter.runs[0]
        self.assertEqual(im.ledger.actions["call_book"].state, "not_run")
        self.assertEqual(len(run.histories), 1)  # held: no next round before the decision
        deciding.set()
        taker = await asyncio.wait_for(read_all(second.stream), 2)
        self.assertEqual(run.routes, ["refused"])
        history = run.histories[-1]
        outputs = [item for item in history if item.get("type") == "function_call_output"]
        self.assertEqual([(item["call_id"], item["output"]) for item in outputs], [("call_book", TEXTS.not_run_output)])
        newer = next(n for n, item in enumerate(history) if "By the window." in str(item))
        self.assertLess(history.index(outputs[-1]), newer)  # the next round reads the output, then the new words
        self.assertEqual(taker[-1]["response"]["output"][-1]["content"][0]["text"], "Booked, by the window.")
        self.assertEqual(im.ledger.delegations["D1"].handed_to, "D2")

    async def test_a_status_answer_from_the_conductor_lets_the_held_work_plan_again_with_no_new_words(self):
        # A request the conductor answers itself concerns no Work, so the held round's call is refused for nothing
        # and the Work plans its next round from the not-run output alone.
        ready, deciding = asyncio.Event(), asyncio.Event()

        async def start(given, place):
            await place(Start(("D1",)))

        async def answer_once_decided(given, place):
            await deciding.wait()
            await place(Answer("D2", "It is booking the table."))

        adapter = ScriptedAdapter(
            [wait(ready), own_call("book_table", call_id="call_book"), own_call("book_table"), answer("Booked.")],
            conductor=planned(start, answer_once_decided),
        )
        im = self.session(adapter)
        first = im.admit(ask("Book a table."), delegation_id="D1")
        await self.turns()
        second = im.admit(ask("How is it going?"), delegation_id="D2")
        await self.turns()
        ready.set()
        await self.turns()
        deciding.set()
        status = await asyncio.wait_for(read_all(second.stream), 2)
        older = await asyncio.wait_for(read_all(first.stream), 2)
        self.assertEqual(status[-1]["response"]["output"][-1]["content"][0]["text"], "It is booking the table.")
        self.assertEqual(adapter.runs[0].routes, ["refused", "backend"])  # planned again; then the call ran
        self.assertEqual(older[-1]["response"]["output"][-1]["content"][0]["text"], "Booked.")
        self.assertIsNone(im.ledger.delegations["D1"].handed_to)

    async def test_the_workers_own_cancel_work_refuses_later_calls_and_its_next_answer_counts(self):
        adapter = ScriptedAdapter([CancelOwnWork(), answer("Stopped as asked.")])
        im = self.session(adapter, work_tools=("cancel_work",))
        events = await read_all(im.admit(ask("Book it."), delegation_id="D1").stream)
        self.assertEqual(adapter.runs[0].routes, ["refused"])
        self.assertEqual(im.ledger.actions["call_later"].state, "not_run")
        self.assertEqual(events[-1]["response"]["output"][-1]["content"][0]["text"], "Stopped as asked.")
        self.assertEqual(im.ledger.works["W1"].state, "answered")

    async def test_the_workers_own_cancel_work_writes_no_item_while_show_writes_the_display(self):
        # cancel_work is a control call and never reaches the wire; only the display tool, show, is an mcp_call under
        # the display label, so the reader never shows a control outcome as the display.
        adapter = ScriptedAdapter([show("Earl Grey"), CancelOwnWork(), answer("Stopped as asked.")])
        speak_display = SpeakDisplay(TEXTS.result_formats["speak_display"])
        im = self.session(adapter, result_format=speak_display, work_tools=("cancel_work",))
        events = await read_all(im.admit(ask("Book it."), delegation_id="D1").stream)
        calls = [(item["server_label"], item["name"]) for item in finished_items(events) if item["type"] == "mcp_call"]
        self.assertEqual(calls, [(CORE_DISPLAY_LABEL, "show")])
        self.assertEqual(im.ledger.actions["call_cancel"].state, "done")  # the IM records it all the same
        self.assertEqual(events[-1]["response"]["output"][-1]["content"][0]["text"], "Stopped as asked.")

    async def test_a_function_the_app_and_a_work_tool_both_name_is_refused(self):
        gate = asyncio.Event()
        speak_display = SpeakDisplay(TEXTS.result_formats["speak_display"])
        show = {"type": "function", "name": "show", "parameters": {"type": "object", "properties": {}}}
        im = self.session(ScriptedAdapter([wait(gate), answer()]), result_format=speak_display)
        im.admit(ask("Show it."), delegation_id="D1", settings={"tools": [show, MENU]})
        await self.turns()
        task = im.works["W1"].tasks[0]
        self.assertEqual(task.route(call_item("show", {"display": "x"}, "c1")), "refused")
        self.assertEqual(task.route(call_item("get_menu", {}, "c2")), "app")
        self.assertEqual(task.route(call_item("lookup", {}, "c3")), "backend")
        gate.set()

    async def test_each_refusal_gives_the_backend_its_own_reason(self):
        # The not-run output names the reason: newer input waits, the work is stopping (a stop request or its own
        # cancel_work), or the app and a work tool name the same function.
        gates = [asyncio.Event() for _ in range(4)]
        speak_display = SpeakDisplay(TEXTS.result_formats["speak_display"])
        show = {"type": "function", "name": "show", "parameters": {"type": "object", "properties": {}}}
        reasons = {}
        for number, (name, refuse) in enumerate(
            (
                ("a stop", lambda task: task.request_stop(ForNewerRequest())),
                ("its own cancel_work", lambda task: task.cancel_own()),
                ("a name clash", lambda task: None),
                ("newer input", lambda task: task.session.admit(ask("Make it one."), delegation_id="D2")),
            )
        ):
            im = self.session(ScriptedAdapter([wait(gates[number]), answer()]), result_format=speak_display)
            im.admit(ask("Book it."), delegation_id="D1", settings={"tools": [show]})
            await self.turns()
            task = im.works["W1"].tasks[0]
            refuse(task)
            await self.turns()
            call = call_item("show" if name == "a name clash" else "book", {}, f"call_{number}")
            self.assertEqual(task.route(call), "refused", name)
            reasons[name] = task.take_boundary_input().outputs[0]["output"]
            gates[number].set()
        self.assertEqual(
            reasons,
            {
                "a stop": TEXTS.stopped_output,
                "its own cancel_work": TEXTS.stopped_output,
                "a name clash": TEXTS.clash_output,
                "newer input": TEXTS.not_run_output,
            },
        )
        self.assertEqual(len(set(reasons.values())), 3)

    async def test_acted_is_set_when_a_call_starts(self):
        gate = asyncio.Event()
        im = self.session(ScriptedAdapter([wait(gate), answer()]))
        im.admit(ask("Book it."), delegation_id="D1")
        await self.turns()
        task, record = im.works["W1"].tasks[0], im.ledger.tasks["W1.1"]
        call = call_item("book", {"table": 4}, "call_book")
        self.assertEqual(task.route(call), "backend")
        self.assertFalse(record.acted)
        task.tool_started(call, "cafe")
        self.assertTrue(record.acted)
        task.tool_outcome("call_book", json.dumps({"booked": True}), "done")
        task.tool_outcome("call_book", "again", "done")  # a call_id never reports twice
        self.assertEqual(im.ledger.actions["call_book"].output, json.dumps({"booked": True}))
        self.assertEqual(task.summary().steps, ("book: done",))
        gate.set()


class UnclearCallTests(IMCase):
    async def test_a_call_that_may_have_run_gets_the_owed_output_and_shows_as_incomplete(self):
        # A call cut at its time limit may have had its effect, so the backend is told the outcome
        # is unknown (texts.owed_output), the action is unclear, and its mcp_call did not complete.
        gate = asyncio.Event()
        im = self.session(ScriptedAdapter([wait(gate), answer()]))
        stream = im.admit(ask("Book it."), delegation_id="D1").stream
        await self.turns()
        task = im.works["W1"].tasks[0]
        call = call_item("book", {"table": 4}, "call_book")
        self.assertEqual(task.route(call), "backend")
        task.tool_started(call, "cafe")
        task.tool_outcome("call_book", "book did not answer within 60 s", "unclear")
        self.assertEqual(im.ledger.actions["call_book"].state, "unclear")
        self.assertEqual(task.take_boundary_input().outputs[0]["output"], TEXTS.owed_output)
        self.assertEqual(task.summary().steps, ("book: unclear",))
        gate.set()
        events = await read_all(stream)
        (item,) = [item for item in finished_items(events) if item["type"] == "mcp_call"]
        self.assertEqual((item["status"], item["output"]), ("incomplete", TEXTS.owed_output))
        self.assertIn("response.mcp_call.failed", [event["type"] for event in events])
