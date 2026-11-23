# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D103, D107

"""An IM session's calls: routing and the commit gate, the app's outputs and continuations, and repeated calls."""

import asyncio
import json

from nemotron_voice_agent.pipecat.examples.frontend_backend_live.delegation_core.backend.port import (
    CallerMessage,
    Cancel,
    Start,
    Steer,
    TaskSpec,
    WorkTarget,
)
from nemotron_voice_agent.pipecat.examples.frontend_backend_live.delegation_core.im.results import SpeakDisplay
from nemotron_voice_agent.pipecat.examples.frontend_backend_live.delegation_core.im.session import (
    FUNCTION_CALL_OUTPUTS_REQUIRED,
    DelegationError,
)
from nemotron_voice_agent.pipecat.examples.frontend_backend_live.work_provenance import decode_provenance
from tests.unit.pipecat.core_helpers import load
from tests.unit.pipecat.delegation_core.helpers import (
    MARK,
    MENU,
    QUESTION,
    TEXTS,
    Act,
    IMCase,
    OwnCall,
    ScriptedAdapter,
    ScriptedRun,
    ScriptStep,
    StartEach,
    WorkTool,
    answer,
    app_call,
    app_outputs,
    ask,
    asking,
    call_item,
    checkpoint,
    finished_items,
    function_calls,
    items_of,
    placing,
    planned,
    read_all,
    said,
    texts,
    wait,
    wire_decision,
    without_ids,
)

SPEAK_DISPLAY = SpeakDisplay(TEXTS.result_formats["speak_display"])
USER = {"type": "function", "name": "get_member", "parameters": {"type": "object", "properties": {}}}
ORDER = {**USER, "name": "get_order"}
CART = {"type": "function", "name": "manage_cart", "parameters": {"type": "object", "properties": {}}}
APP, MENU_APP = {"tools": [ORDER, CART]}, {"tools": [MENU]}
LOOKUP, LATTE, NOT_FOUND = {"order_id": "BB-1042"}, {"action": "add", "item_id": "lat"}, "Error: order not found"
DECAF = {"type": "message", "role": "user", "content": "Make the drink decaf."}
MILK = {"type": "message", "role": "user", "content": "With milk, please."}
ENDINGS = load()
STOCK = call_item("lookup_stock", {}, "call_stock")
WHERE = {"work_id": "W1", "task_id": "W1.1", "delegation_id": "D1"}


def outputs(history: list[dict]) -> dict[str, str]:
    return {item["call_id"]: item["output"] for item in history if item.get("type") == "function_call_output"}


def asks(call_id: str) -> dict:
    return call_item("caller_message", {"kind": "question", "message": QUESTION}, call_id)


BAD_OUTPUT_PARTS = (
    {"type": "input_text", "text": None},
    {"type": "input_image", "image_url": 467},
    {"type": "input_file", "file_id": {"id": "file-demo"}},
)


class Round(ScriptStep):
    """A round of ``calls``: it streams until ``streamed`` is set, and its backend calls run until ``ran`` is set.

    The task's events end when one of its calls goes to the app.
    """

    def __init__(self, *calls: dict, streamed: asyncio.Event | None = None, ran: asyncio.Event | None = None):
        self.calls, self.streamed, self.ran = list(calls), streamed, ran

    async def play(self, run):
        if self.streamed is not None:
            run.boundary()
            await self.streamed.wait()
        for event in run.round_events(self.calls, boundary=self.streamed is None):
            yield event
        routes = [run.task.route(call) for call in self.calls]
        run.routes += routes
        self.ends = "app" in routes
        backend = [call for call, route in zip(self.calls, routes, strict=True) if route == "backend"]
        for call in backend:
            run.task.tool_started(call, "cafe")
        if self.ran is not None:
            await self.ran.wait()
        # The outcomes come out of call order.
        for call in reversed(backend):
            run.task.tool_outcome(call["call_id"], f"output {call['call_id']}", "done")


def app(name: str, arguments: dict, call_id: str) -> Round:
    return Round(call_item(name, arguments, call_id))


class DiesWhileItRuns(ScriptStep):
    """A round whose backend call ``call`` starts; once ``gate`` is set the run raises, the call still running."""

    def __init__(self, call: dict, gate: asyncio.Event):
        self.call, self.gate = call, gate

    async def play(self, run):
        for event in run.round_events([self.call]):
            yield event
        run.task.route(self.call)
        run.task.tool_started(self.call, "cafe")
        await self.gate.wait()
        raise RuntimeError("the tool's host died")


class FinishesItsCall(ScriptedRun):
    """A run whose stop lets its started call finish: the call reports its output before the stop is confirmed."""

    async def stop(self):
        self.task.tool_outcome("call_book", "Booked.", "done")
        return await super().stop()


class Marked:
    """An app output step that marks every output it reads, and keeps the tools each continuation brought."""

    def __init__(self, mark: str):
        self.mark, self.tools = mark, []

    def outputs(self, calls, tools):
        self.tools.append(tools)
        return {output["call_id"]: {**output, "output": f"{output['output']} {self.mark}"} for _, output in calls}


class CallTests(IMCase):
    async def waiting(self, *scripts, request="Menu?", settings=MENU_APP, **changes) -> tuple:
        """Return the adapter, a session on ``scripts`` with ``changes``, and the events of its first request."""
        adapter = ScriptedAdapter(*scripts)
        im = self.session(adapter, **changes)
        return adapter, im, await read_all(im.admit(ask(request), delegation_id="D1", settings=settings).stream)

    def refusals(self) -> list[tuple[str, str]]:
        """Return the call and the reason of each refusal traced, in order."""
        return [(fields["call_id"], fields["reason"]) for fields in self.traced("tools.refused")]

    async def test_calls_wait_for_confirmed_steering(self):
        gate = asyncio.Event()
        adapter = ScriptedAdapter([wait(gate), answer()])
        im = self.session(adapter)
        await self.admit_each(im, "Order two lattes.", "Make it one.", settings=MENU_APP)
        task = adapter.runs[0].task
        self.assertEqual(task.route(call_item("get_menu", {}, "c1")), "refused")
        self.assertEqual(task.route(call_item("place_order", {}, "c2")), "refused")
        given = task.take_boundary_input()
        self.assertEqual(given.outputs, ())
        self.assertEqual([task.refusal(key) for key in ("c1", "c2")], [TEXTS.not_run_output] * 2)
        self.assertEqual(task.route(call_item("get_menu", {}, "c3")), "refused")
        task.settle(given.steering, "not_applied")
        self.assertEqual(task.route(call_item("get_menu", {}, "c4")), "refused")
        task.settle(task.take_boundary_input().steering, "unknown")
        self.assertEqual(task.route(call_item("get_menu", {}, "c5")), "refused")
        self.assertEqual(task.refusal_details("c5").reason, "routing_unresolved")
        im.admit(ask("Yes, just one latte."), delegation_id="D3", settings=MENU_APP)
        await self.until(lambda: bool(task.steering), "a fresh caller update")
        task.settle(task.take_boundary_input().steering, "applied")
        self.assertEqual(task.route(call_item("get_menu", {}, "c5")), "refused")
        self.assertEqual(task.route(call_item("get_menu", {}, "c6")), "app")
        self.assertEqual(task.route(call_item("place_order", {}, "c7")), "backend")
        gate.set()

    async def test_newer_input_keeps_back_the_calls_of_its_round(self):
        menu, order = call_item("get_menu", {}, "call_menu"), call_item("place_order", {"lattes": 2}, "call_order")
        lookup = call_item("lookup", {}, "call_lookup")
        cases = {
            "streamed": ((menu, order), ["refused", "refused"], {"call_order": TEXTS.not_run_output}),
            "ran": ((lookup, menu), ["backend", "app"], {"call_lookup": "output call_lookup"}),
        }
        for gated, (calls, routes, kept) in cases.items():
            with self.subTest(gated):
                self.traces.clear()
                gate = asyncio.Event()
                adapter = ScriptedAdapter([Round(*calls, **{gated: gate}), answer("Nothing was ordered.")])
                streams = await self.admit_each(self.session(adapter), "Order two.", "Never mind.", settings=MENU_APP)
                gate.set()
                older, taker = [await read_all(stream, timeout=2) for stream in streams]
                run = adapter.runs[0]
                self.assertEqual(run.routes, routes)
                self.assertEqual((function_calls(older + taker), texts(taker)), ([], ["Nothing was ordered."]))
                history = run.histories[-1]
                self.assertEqual(outputs(history), {**kept, "call_menu": TEXTS.not_run_output})
                newer = next(n for n, item in enumerate(history) if "Never mind." in str(item))
                last_output = max(n for n, item in enumerate(history) if item.get("type") == "function_call_output")
                self.assertLess(last_output, newer)
                # Every call but the lookup, which ran before the newer input came.
                kept_back = [call["call_id"] for call in calls if call is not lookup]
                self.assertEqual(self.refusals(), [(call_id, "not_run") for call_id in kept_back])

    async def test_a_call_waits_while_the_conductor_decides(self):
        handover = TEXTS.handover_line.format(delegation="D2")
        for decision, older, taker in (
            (Steer(("D2",), (WorkTarget("W1", None),), clarifications=()), [handover], ["Booked."]),
            (CallerMessage(("D2",), "status", "It is booking the table."), ["Booked."], ["It is booking the table."]),
        ):
            with self.subTest(type(decision).__name__):
                ready, deciding = asyncio.Event(), asyncio.Event()

                async def once_decided(given, place, decision=decision, deciding=deciding):
                    # A model conductor's call: the newer request waits for it.
                    await deciding.wait()
                    await place(decision)

                script = [wait(ready), OwnCall("book", call_id="call_book"), OwnCall("book"), answer("Booked.")]
                adapter = ScriptedAdapter(script, conductor=planned(Start(("D1",)), once_decided))
                streams = await self.admit_each(self.session(adapter), "Book a table.", "By the window.")
                ready.set()
                await self.turns()
                run = adapter.runs[0]
                self.assertEqual(len(run.histories), 1)
                deciding.set()
                said = [texts(await read_all(stream, timeout=2)) for stream in reversed(streams)]
                self.assertEqual(said, [taker, older])
                self.assertEqual(run.routes, ["refused", "backend"])
                self.assertEqual(outputs(run.histories[1])["call_book"], TEXTS.not_run_output)

    async def test_the_tasks_own_cancel_work_refuses_later_calls(self):
        cancel = WorkTool("cancel_work", {}, (call_item("lookup", {}, "call_later"),))
        adapter = ScriptedAdapter([WorkTool("show", {"display": "Earl Grey"}), cancel, answer("Stopped as asked.")])
        im = self.session(adapter, result_format=SPEAK_DISPLAY)
        events = await self.request(im, "Book it.", "D1")
        self.assertEqual(adapter.runs[0].routes, ["backend", "backend", "refused"])
        self.assertEqual(outputs(adapter.runs[0].histories[-1])["call_later"], TEXTS.stopped_output)
        calls = [(item["server_label"], item["name"]) for item in items_of(events, "mcp_call")]
        self.assertEqual(calls, [("delegation_core", "show")])
        self.assertEqual((texts(events), events[-1]["type"]), ([TEXTS.cancelled_line], "response.completed"))

    async def test_an_ask_writes_its_question_and_the_task_runs_on(self):
        gate = asyncio.Event()
        im, adapter = self.scripted_session([asking(), wait(gate), answer("A latte, then.")])
        stream = im.admit(ask("A muffin and a coffee."), delegation_id="D1").stream
        await self.until(lambda: self.traced("im.question.written"), "the question")
        await self.turns()
        written = [event for _, event in self.wire.events]
        self.assertEqual((said(written), function_calls(written)), ([("final_answer", QUESTION)], []))
        (response,) = im.ledger.delegations["D1"].responses
        self.assertEqual(im.ledger.responses[response].state, "in_progress")
        record = im.ledger.tasks["W1.1"]
        self.assertEqual((record.state, record.result.state, record.question), ("running", "none", QUESTION))
        self.assertEqual(
            self.traced("im.question.written"),
            [
                {
                    **WHERE,
                    "question_id": "W1.1.Q1",
                    "call_id": "call_W1.1_caller_message_1",
                    "input_revision": 1,
                    "item_id": items_of(written, "message")[0]["id"],
                    "response_id": response,
                }
            ],
        )
        gate.set()
        events = await read_all(stream)
        self.assertEqual(
            (said(events), events[-1]["type"]),
            ([("final_answer", QUESTION), ("final_answer", "A latte, then.")], "response.completed"),
        )
        receipt = json.loads(outputs(adapter.runs[0].histories[-1])["call_W1.1_caller_message_1"])
        self.assertEqual(receipt, {"kind": "question", "question_id": "W1.1.Q1", "publication": "written"})
        self.assertEqual(record.result.state, "pushed")

    async def test_an_ask_after_the_tasks_own_cancel_work_is_refused(self):
        adapter = ScriptedAdapter([WorkTool("cancel_work", {}, (asks("call_ask"),)), answer("Stopped as asked.")])
        events = await self.request(self.session(adapter), "Stop.", "D1")
        run = adapter.runs[0]
        self.assertEqual(run.routes, ["backend", "refused"])
        self.assertEqual(outputs(run.histories[-1])["call_ask"], TEXTS.stopped_output)
        self.assertEqual((texts(events), run.task.record.question), ([TEXTS.cancelled_line], None))
        refused = {**WHERE, "call_id": "call_ask", "name": "caller_message", "reason": "stopped"}
        self.assertEqual(self.traced("tools.refused"), [refused])

    async def asking_while_the_app_works(self, go_on: asyncio.Event, *later, before_answer=(), **settings) -> tuple:
        """Split D1 into a task whose app call ends D1's response, and a task that then asks, its question held.

        The asking task runs ``before_answer`` once ``go_on`` is set.
        ``later`` holds the conductor's plans for the next arrivals.
        Return the session and D1's events.
        """
        parked = asyncio.Event()
        menu = [app_call(call_id="call_menu"), answer("Menu.")]
        asker = [wait(parked), asking(), wait(go_on), *before_answer, answer("Noted.")]
        conductor = planned(Start(("D1",), (TaskSpec("menu"), TaskSpec("coffee"))), *later)
        im = self.session(ScriptedAdapter(menu, asker, conductor=conductor, max_running=2), **settings)
        first = await self.request(im, "The menu, and a coffee.", "D1", settings=MENU_APP)
        parked.set()
        actions, asked = im.ledger.actions, "call_W1.2_caller_message_1"
        await self.until(lambda: asked in actions and actions[asked].ended, "the ask to return")
        self.assertEqual(im.ledger.tasks["W1.2"].questions[0].dispatch, "pending")
        self.assertEqual(self.traced("im.question.written"), [])
        self.assertNotIn(QUESTION, texts([event for _, event in self.wire.events]))
        return im, first

    async def test_a_question_held_while_the_app_works_goes_out_first(self):
        go_on = asyncio.Event()
        im, first = await self.asking_while_the_app_works(go_on)
        continuation = im.admit_client_request(app_outputs(first), delegation_id="D1", settings=MENU_APP)
        self.assertEqual(im.ledger.tasks["W1.2"].question, QUESTION)
        go_on.set()
        rest = await read_all(continuation.stream, timeout=2)
        self.assertEqual(said(rest)[0], ("final_answer", QUESTION))
        self.assertEqual(sorted(texts(rest)[1:]), ["Menu.", "Noted."])
        self.assertEqual(len(self.traced("im.question.written")), 1)

    async def test_a_held_question_is_never_said_when_its_work_stops_at_the_deadline(self):
        im, _ = await self.asking_while_the_app_works(asyncio.Event(), client_tool_timeout_seconds=0.2)
        await self.until(lambda: not im.works, "the deadline")
        self.assertEqual((im.ledger.works["W1"].state, im.ledger.tasks["W1.2"].question), ("failed", None))
        self.assertEqual(self.traced("im.question.written"), [])
        self.assertNotIn(QUESTION, texts([event for _, event in self.wire.events]))

    async def test_a_held_question_is_never_said_when_a_newer_request_stops_its_task(self):
        stop = placing(
            Cancel((), (WorkTarget("W1", ("W1.2",)),), clarifications=()),
            CallerMessage(("D2",), "status", "No coffee, then."),
        )
        im, first = await self.asking_while_the_app_works(asyncio.Event(), stop)
        await self.request(im, "Skip the coffee.", "D2", timeout=2)
        await self.until(lambda: im.ledger.tasks["W1.2"].ended, "the coffee task's stop")
        rest = await self.answer_calls(im, first, settings=MENU_APP)
        self.assertEqual(texts(rest), ["Menu."])
        self.assertEqual((im.ledger.tasks["W1.2"].question, self.traced("im.question.written")), (None, []))

    async def test_continue_after_call_is_offered_to_a_kept_conversation_above_one_slot(self):
        offered, kept = (
            ["cancel_work", "set_work_state", "caller_message", "resolve_questions"],
            ["cancel_work", "set_work_state", "caller_message", "resolve_questions", "continue_after_call"],
        )
        for continuable, max_running, names in ((True, 2, kept), (False, 2, offered), (True, 1, offered)):
            with self.subTest(continuable=continuable, max_running=max_running):
                gate = asyncio.Event()
                adapter = ScriptedAdapter([wait(gate), answer()], max_running=max_running)
                await self.admit_each(self.session(adapter, continuable=continuable), "Hi.")
                self.assertEqual([tool["name"] for tool in adapter.runs[0].task.tools().tools()], names)
                gate.set()

    async def test_after_the_call_a_kept_works_app_calls_and_asks_are_refused(self):
        hung_up, attached = asyncio.Event(), asyncio.Event()
        job = [
            MARK,
            wait(hung_up),
            Round(call_item("get_menu", {}, "call_menu_1"), asks("call_ask_1")),
            wait(attached),
            Round(call_item("get_menu", {}, "call_menu_2"), asks("call_ask_2")),
            answer("The comparison is ready."),
        ]
        adapter = ScriptedAdapter(job, max_running=2)
        im = self.session(adapter, continuable=True)
        im.admit(ask("Compare every supplier."), delegation_id="D1", settings=MENU_APP)
        await self.until(lambda: self.traced("im.work.after_call"), "the mark")
        self.assertTrue(await self.hang_up(im))
        hung_up.set()
        await self.until(lambda: len(self.traced("tools.refused")) == 2, "the calls after the hang-up")
        # A later call: no request of it has taken the work over.
        await self.call_again(im)
        attached.set()
        await self.until(lambda: not im.works, "the job's end")
        calls = ("call_menu_1", "call_ask_1", "call_menu_2", "call_ask_2")
        self.assertEqual(self.refusals(), [(call_id, "after_call") for call_id in calls])
        read = outputs(adapter.runs[0].histories[-1])
        self.assertEqual({read[call_id] for call_id in calls}, {TEXTS.after_call_output})
        self.assertEqual((im.ledger.works["W1"].state, im.ledger.tasks["W1.1"].question), ("answered", None))
        self.assertEqual(function_calls([event for _, event in self.wire.events]), [])

    async def test_a_kept_works_calls_are_refused_once_the_call_ends(self):
        with self.subTest("made in the hang-up's close grace"):
            gate, other = asyncio.Event(), asyncio.Event()
            job = [MARK, wait(gate), Round(call_item("get_menu", {}, "call_menu"), asks("call_ask")), wait(other)]
            scripts = ([*job, answer("Done.")], [wait(other), answer("Calendar.")])
            conductor = planned(Start(("D1",)), Start(("D2",)))
            adapter = ScriptedAdapter(*scripts, conductor=conductor, max_running=3)
            im = self.session(adapter, continuable=True, close_grace_seconds=5)
            im.admit(ask("Compare every supplier."), delegation_id="D1", settings=MENU_APP)
            await self.until(lambda: self.traced("im.work.after_call"), "the mark")
            im.admit(ask("What is on my calendar?"), delegation_id="D2")
            await self.until(lambda: im.ledger.tasks.get("W2.1") and im.ledger.tasks["W2.1"].state == "running", "W2")
            leaving = asyncio.ensure_future(im.leave())
            await self.until(lambda: im.detached, "the hang-up")
            gate.set()
            await self.until(lambda: len(self.traced("tools.refused")) == 2, "the refusals")
            self.assertEqual({fields["reason"] for fields in self.traced("tools.refused")}, {"after_call"})
            self.assertEqual((im.parked, im.ledger.tasks["W1.1"].question, leaving.done()), ({}, None, False))
            other.set()
            self.assertTrue(await asyncio.wait_for(leaving, 2))
            await self.until(lambda: not im.works, "the job's end")
            self.assertEqual(im.ledger.works["W1"].state, "answered")
        self.traces.clear()
        with self.subTest("routed before the hang-up"):
            ran = asyncio.Event()
            rounds = [MARK, Round(call_item("lookup", {}, "call_own"), call_item("get_menu", {}, "call_app"), ran=ran)]
            im = self.session(ScriptedAdapter([*rounds, answer("Done.")], max_running=2), continuable=True)
            im.admit(ask("Compare every supplier."), delegation_id="D1", settings=MENU_APP)
            await self.until(lambda: "call_own" in im.ledger.actions, "the lookup, its round's app call routed")
            self.assertEqual(im.ledger.actions["call_app"].runner, "app")
            self.assertTrue(await self.hang_up(im))
            ran.set()
            await self.until(lambda: not im.works, "the job's end")
            self.assertEqual([fields["reason"] for fields in self.traced("tools.refused")], ["after_call"])
            self.assertEqual((im.ledger.works["W1"].state, im.parked), ("answered", {}))
        self.assertEqual(function_calls([event for _, event in self.wire.events]), [])

    async def test_a_refused_call_gets_its_reason(self):
        show_tool = {"type": "function", "name": "show", "parameters": {"type": "object", "properties": {}}}
        plain = {"tools": [show_tool]}
        listed = {**plain, "tool_choice": {"type": "allowed_tools", "tools": []}}
        stopped, clash = TEXTS.stopped_output, TEXTS.clash_output
        rows = (("book", ["Stop."], plain, stopped), ("show", ["Stop."], listed, stopped))
        for tool, newer, settings, reason in (*rows, ("show", [], plain, clash), ("show", [], listed, clash)):
            with self.subTest(reason, tool=tool, listed=settings is listed):
                self.traces.clear()
                gate, conductor = (
                    asyncio.Event(),
                    planned(Start(("D1",)), Cancel((), (WorkTarget("W1", None),), clarifications=())),
                )
                adapter = ScriptedAdapter([wait(gate), answer()], conductor=conductor, stop_delay=0.05)
                im = self.session(adapter, result_format=SPEAK_DISPLAY)
                await self.admit_each(im, "Book it.", *newer, settings=settings)
                task = adapter.runs[0].task
                self.assertEqual(task.route(call_item(tool, {}, "call_1")), "refused")
                self.assertEqual(task.refusal("call_1"), reason)
                self.assertEqual(self.refusals(), [("call_1", "stopped" if reason == stopped else "clash")])
                gate.set()

    async def test_only_an_app_function_the_list_leaves_out_is_refused(self):
        decaf = {"type": "function", "name": "set_decaf", "parameters": {"type": "object", "properties": {}}}
        choice = {"type": "allowed_tools", "mode": "auto", "tools": [{"type": "function", "name": "get_menu"}]}
        gate, settings = asyncio.Event(), {"tools": [MENU, decaf], "tool_choice": choice}
        adapter = ScriptedAdapter([wait(gate), answer()])
        await self.admit_each(self.session(adapter), "Make it decaf.", "With oat milk.", settings=settings)
        task = adapter.runs[0].task
        for name in ("set_decaf", "get_menu"):
            self.assertEqual(task.route(call_item(name, {}, f"early_{name}")), "refused")
        given = task.take_boundary_input()
        self.assertEqual(
            [task.refusal(key) for key in ("early_set_decaf", "early_get_menu")],
            [TEXTS.unlisted_output, TEXTS.not_run_output],
        )
        task.settle(given.steering, "applied")
        names = ("set_decaf", "get_menu", "lookup", "cancel_work")
        routes = [task.route(call_item(name, {}, f"call_{name}")) for name in names]
        self.assertEqual(routes, ["refused", "app", "backend", "backend"])
        self.assertEqual(task.refusal("call_set_decaf"), TEXTS.unlisted_output)
        refused = [("early_set_decaf", "unlisted"), ("early_get_menu", "not_run"), ("call_set_decaf", "unlisted")]
        self.assertEqual(self.refusals(), refused)
        gate.set()

    async def test_stop_retains_backend_result_and_refuses_an_unpublished_app_call(self):
        book, menu = call_item("book_table", {}, "call_book"), call_item("get_menu", {}, "call_menu")
        scripts = ([Round(book, menu, ran=asyncio.Event())], [answer()])
        adapter = ScriptedAdapter(*scripts, conductor=StartEach(), run=FinishesItsCall)
        # The backstop stops the Work.
        im = self.session(adapter, work_timeout_seconds=0.05)
        before = await self.request(im, "Book a table and show the menu.", "D1", settings=MENU_APP)
        self.assertNotIn("call_menu", function_calls(before))
        await self.request(im, "Did it book?", "D2")
        read = outputs(adapter.runs[1].histories[0])
        self.assertEqual(read, {"call_book": "Booked.", "call_menu": TEXTS.stopped_output})

    async def test_a_call_that_may_have_run_gets_the_owed_output(self):
        gate = asyncio.Event()
        adapter = ScriptedAdapter([wait(gate), answer()])
        (stream,) = await self.admit_each(self.session(adapter), "Book it.")
        task, call = adapter.runs[0].task, call_item("book", {"table": 4}, "call_book")
        self.assertEqual(task.route(call), "backend")
        task.tool_started(call, "cafe")
        task.tool_outcome("call_book", "book did not answer within 60 s", "unclear")
        self.assertEqual(task.take_boundary_input().outputs, ())
        self.assertEqual(task._execution_facts()[0]["outcome"], "unclear")
        # An uncertain effect is never retried. A later actual result reconciles its evidence.
        task.tool_outcome("call_book", "Booked.", "done")
        self.assertEqual(task.take_boundary_input().outputs, ())
        self.assertEqual(task._execution_facts()[0]["output"], "Booked.")
        action = task.session.ledger.actions["call_book"]
        self.assertEqual(
            (action.state, action.output, action.late_outcome, action.late_output),
            ("unclear", TEXTS.owed_output, "done", "Booked."),
        )
        gate.set()
        events = await read_all(stream)
        (item,) = items_of(events, "mcp_call")
        self.assertEqual((item["status"], item["output"]), ("incomplete", TEXTS.owed_output))
        self.assertIn("response.mcp_call.failed", [event["type"] for event in events])

    async def test_bad_outputs_are_refused_and_change_nothing(self):
        adapter, im, first = await self.waiting(
            [Round(call_item("get_menu", {}, "c1"), call_item("get_menu", {}, "c2")), answer()]
        )
        output, second = app_outputs(first)
        with self.assertRaises(DelegationError) as refused:
            im.admit_client_request([output], delegation_id="D1")
        self.assertEqual(refused.exception.code, FUNCTION_CALL_OUTPUTS_REQUIRED)
        for items, error in (
            ([output, output], "Duplicate"),
            ([{**output, "type": "custom_tool_call_output"}], "function_call_output"),
            ([{"type": "function_call_output", "output": "{}"}], "function_call_output"),
            ([{"type": "function_call_output", "call_id": output["call_id"]}, second], "no output"),
            ([{**output, "output": None}, second], "must be text or a list"),
        ):
            with self.subTest(error), self.assertRaisesRegex(ValueError, error):
                im.admit_client_request(items, delegation_id="D1")
        for part in BAD_OUTPUT_PARTS:
            with self.subTest(kind=part["type"]), self.assertRaisesRegex(ValueError, part["type"]):
                im.admit_client_request([output, {**second, "output": [part]}], delegation_id="D1")
        content = [
            {"type": "input_text", "text": "Tea is available."},
            {"type": "input_image", "image_url": "https://example.com/menu.png"},
            {"type": "input_file", "file_id": "file-menu"},
        ]
        data = [{"drink": "coffee"}, 3]
        retry = im.admit_client_request([{**output, "output": content}, {**second, "output": data}], delegation_id="D1")
        self.assertEqual((await read_all(retry.stream, timeout=2))[-1]["type"], "response.completed")
        self.assertEqual(outputs(adapter.runs[0].histories[-1]), {"c1": content, "c2": data})

    async def test_two_waiting_responses_continue_one_at_a_time(self):
        scripts = ([app_call(), answer("Teas.")], [app_call(), answer("Coffee.")])
        im = self.session(ScriptedAdapter(*scripts, conductor=StartEach(), max_running=2))
        first = await self.request(im, "Teas?", "D1", settings=MENU_APP)
        second = await self.request(im, "Coffee?", "D2", settings=MENU_APP)
        with self.assertRaisesRegex(ValueError, "more than one response"):
            im.admit_client_request(app_outputs(first) + app_outputs(second))
        later = await self.answer_calls(im, second, delegation_id="D2")
        self.assertEqual((texts(later), texts(await self.answer_calls(im, first))), (["Coffee."], ["Teas."]))

    async def test_outputs_for_an_ended_work_get_an_empty_response(self):
        scripts = ([app_call(), answer()], [answer("Green teas.")])
        _, im, first = await self.waiting(*scripts, client_tool_timeout_seconds=0.02)
        # The client-tool deadline ends the Work.
        await asyncio.sleep(0.05)
        late = ["response.created", "response.queued", "response.in_progress", "response.completed"]
        output = [{"type": "input_text", "text": "Late result."}]
        (evidence,) = app_outputs(first, output)
        action = im.ledger.actions[evidence["call_id"]]
        with self.assertRaisesRegex(DelegationError, "published application call"):
            im.admit_client_request([evidence, {**evidence, "call_id": "call_unknown"}], delegation_id="D1")
        self.assertIsNone(action.late_outcome)
        self.assertEqual([e["type"] for e in await self.answer_calls(im, first, output)], late)
        self.assertEqual((action.state, action.late_outcome, action.late_output), ("unclear", "done", output))
        typed = {"type": "message", "role": "user", "content": "Just the green teas then."}
        with self.assertRaisesRegex(DelegationError, "conflicts"):
            im.admit_client_request([{**evidence, "output": "Conflicting result."}, typed], delegation_id="D1")
        ran = await read_all(im.admit_client_request([evidence, typed], delegation_id="D1").stream)
        self.assertEqual(texts(ran), ["Green teas."])
        empty = await read_all(im.admit_client_request([], delegation_id="T1").stream)
        self.assertEqual(texts(empty), ["Done."])

    async def split_waiting(self, food: list, *more, after_call=False) -> tuple:
        """Split D1 into a drink task, which waits for the app's menu, and a food task that plays ``food``.

        ``more`` is what the conductor places for D2, which then comes. Return the session and D1's events.
        """
        start = wire_decision(
            {"action": "start", "delegations": ["D1"], "tasks": [{"brief": "Find a drink."}, {"brief": "Find food."}]}
        )
        plans = (start, *([placing(*more)] if more else []))
        drink = [app_call(call_id="call_drink"), answer("Drink.")]
        scripts = ([MARK, *drink] if after_call else drink, food, [answer("Skipped.")])
        im = self.session(ScriptedAdapter(*scripts, conductor=planned(*plans), max_running=3), continuable=after_call)
        first = await self.request(im, "A drink and some food.", "D1", settings=MENU_APP)
        if more:
            await self.request(im, "Skip the drink.", "D2")
        return im, first

    async def test_typed_words_reach_only_the_task_that_owns_the_outputs(self):
        gate = asyncio.Event()
        im, first = await self.split_waiting([wait(gate), checkpoint(), answer("Food.")])
        rest = im.admit_client_request([*app_outputs(first), DECAF], delegation_id="D1", settings=MENU_APP)
        await self.turns()
        gate.set()
        self.assertEqual(texts(await read_all(rest.stream, timeout=2)), ["Drink.", "Food."])
        drink, food = im.backend.runs[:2]
        history = drink.histories[-1]
        typed = next(item for item in history if without_ids([item]) == [DECAF])
        output = next(item for item in history if item.get("type") == "function_call_output")
        self.assertLess(history.index(output), history.index(typed))
        self.assertNotIn(typed.get("id"), (None, "D1"))
        self.assertNotIn(DECAF, without_ids(food.histories[-1]))

    async def test_typed_words_of_a_stopped_task_stay_in_history(self):
        skip = (Cancel((), (WorkTarget("W1", ("W1.1",)),), clarifications=()), Start(("D2",)))
        for after_call in (False, True):
            with self.subTest(after_call=after_call):
                gate = asyncio.Event()
                im, first = await self.split_waiting(
                    [wait(gate), checkpoint(), answer("Food.")], *skip, after_call=after_call
                )
                self.assertEqual(im.ledger.works["W1"].after_call, after_call)
                (output,) = app_outputs(first, "Drink menu.")
                rest = im.admit_client_request([output, DECAF], delegation_id="D1", settings=MENU_APP)
                gate.set()
                self.assertEqual(texts(await read_all(rest.stream, timeout=2)), ["Food."])
                self.assertNotIn(DECAF, without_ids(im.backend.runs[1].histories[-1]))
                self.assertIn(DECAF, without_ids(im.history))
                action = im.ledger.actions[output["call_id"]]
                self.assertEqual((action.task, action.state, action.output), ("W1.1", "done", output["output"]))

    async def test_typed_words_no_task_takes_join_the_conversation(self):
        gate = asyncio.Event()
        im, first = await self.split_waiting(
            [wait(gate), answer("Food.")], Cancel((), (WorkTarget("W1", ("W1.1",)),), clarifications=()), Start(("D2",))
        )
        gate.set()
        await self.turns()
        rest = im.admit_client_request([*app_outputs(first), DECAF], delegation_id="D1", settings=MENU_APP)
        self.assertEqual(texts(await read_all(rest.stream, timeout=2)), ["Food."])
        self.assertIn(DECAF, without_ids(im.history))

    async def test_typed_words_never_applied_join_and_start_no_work(self):
        for delivery in ("not_applied", "unknown"):
            with self.subTest(delivery=delivery):
                script = [app_call(), checkpoint(), answer("Teas.")]
                if delivery == "unknown":
                    script.insert(2, app("get_menu", {}, "call_stale"))
                im = self.session(ScriptedAdapter(script, delivery=delivery))
                first = await self.request(im, "Find teas.", "D1", settings=MENU_APP)
                (output,) = app_outputs(first, "Tea menu.")
                rest = im.admit_client_request([output, MILK], delegation_id="D1", settings=MENU_APP)
                im._prune()
                events = await read_all(rest.stream, timeout=2)
                action = im.ledger.actions[output["call_id"]]
                self.assertEqual((action.state, action.output), ("done", output["output"]))
                if delivery == "unknown":
                    self.assertEqual(function_calls(events), [])
                    stale = im.ledger.actions["call_stale"]
                    self.assertEqual((stale.state, stale.app_published), ("not_run", False))
                else:
                    self.assertEqual(texts(events), [])
                    task = im.ledger.tasks["W1.1"]
                    self.assertEqual((task.answer, task.result.state), ("Teas.", "final"))
                    await self.turns()
                    self.assertIn(MILK, without_ids(im.history))
                self.assertFalse(self.traced("im.follow_up"))

    async def test_typed_words_follow_outputs_and_the_earlier_spoken_request(self):
        adapter, im, first = await self.waiting([app_call(), answer("Green: two.")])
        second = im.admit(ask("Green only."), delegation_id="D2", settings=MENU_APP)
        await self.turns()
        rest = im.admit_client_request([*app_outputs(first), MILK], delegation_id="D1", settings=MENU_APP)
        await read_all(rest.stream, timeout=2)
        self.assertEqual(texts(await read_all(second.stream, timeout=2)), ["Green: two."])
        history = without_ids(adapter.runs[0].histories[-1])
        output = next(item for item in history if item.get("type") == "function_call_output")
        typed = next(item for item in history if item == MILK)
        steered = next(item for item in history if item.get("role") == "user" and "Green only." in str(item))
        self.assertLess(history.index(output), history.index(typed))
        self.assertLess(history.index(output), history.index(steered))
        self.assertLess(history.index(steered), history.index(typed))
        parts = [part["text"] for part in steered["content"]]
        self.assertEqual(parts, ["Green only."])
        self.assertIn(TEXTS.closing_line, str(history[history.index(steered) + 1]))

    async def test_the_apps_outputs_pass_the_output_steps_in_order(self):
        lookup = {"tools": [{**USER, "name": "get_member"}]}
        first_step = Marked("(a)")
        for steps, read in (((first_step, Marked("(b)")), "Not found. (a) (b)"), ((), "Not found.")):
            with self.subTest(steps=len(steps)):
                adapter = ScriptedAdapter([app("get_member", {"user_id": "maya_ross_3817"}, "call_1"), answer()])
                im = self.session(adapter, output_steps=steps)
                first = await self.request(im, "Find my account.", "D1", settings=lookup)
                await self.answer_calls(im, first, "Not found.", settings=lookup)
                self.assertEqual(outputs(adapter.runs[0].histories[-1]), {"call_1": read})
        self.assertEqual(first_step.tools, [lookup["tools"]])  # the continuation's own functions

    async def test_an_invalid_output_step_does_not_consume_the_app_result(self):
        class Broken:
            def __init__(self, failure):
                self.failure = failure

            def outputs(self, calls, tools):
                _, output = calls[0]
                output["output"] = "Changed by failed step."
                if self.failure == "exception":
                    raise RuntimeError("Cannot format the result")
                if self.failure == "missing":
                    return {}
                if self.failure == "invalid_value":
                    return {output["call_id"]: {**output, "output": None}}
                if isinstance(self.failure, dict):
                    return {output["call_id"]: {**output, "output": [self.failure]}}
                return {"call_1": {**output, "call_id": "another_call"}}

        lookup = {"tools": [{**USER, "name": "get_member"}]}
        for failure in ("exception", "missing", "mismatched", "invalid_value", *BAD_OUTPUT_PARTS):
            with self.subTest(failure=failure):
                adapter = ScriptedAdapter([app("get_member", {"user_id": "maya_ross_3817"}, "call_1"), answer("Done.")])
                im = self.session(adapter, output_steps=(Marked("(a)"), Broken(failure), Marked("(b)")))
                first = await self.request(im, "Find my account.", "D1", settings=lookup)
                rest = await self.answer_calls(im, first, "Not found.", settings=lookup)
                self.assertEqual(texts(rest), ["Done."])
                self.assertEqual(outputs(adapter.runs[0].histories[-1]), {"call_1": "Not found. (a) (b)"})
                self.assertEqual(list(im.ledger.actions), ["call_1"])
                self.assertEqual(im.ledger.actions["call_1"].output, "Not found.")

    async def test_a_repeat_of_the_newest_call_is_answered_from_record(self):
        both = Round(call_item("get_order", LOOKUP, "call_1"), call_item("get_order", LOOKUP, "call_2"))
        steps = [both, app("get_order", LOOKUP, "call_3"), answer("Spell it.")]
        adapter, im, first = await self.waiting(steps, request="Find my order.", settings=APP)
        self.assertEqual(function_calls(first), ["call_1", "call_2"])
        rest = await self.answer_calls(im, first, NOT_FOUND, settings=APP)
        self.assertEqual((function_calls(rest), adapter.runs[0].routes), ([], ["app", "app", "refused"]))
        repeated = TEXTS.repeated_call_output.format(output=NOT_FOUND)
        read = outputs(adapter.runs[0].histories[-1])
        self.assertEqual(read, {"call_1": NOT_FOUND, "call_2": NOT_FOUND, "call_3": repeated})
        # Its trace names the reason, never the output, which holds the app's earlier answer.
        refused = {**WHERE, "call_id": "call_3", "name": "get_order", "reason": "repeated"}
        self.assertEqual(self.traced("tools.refused"), [refused])

    async def test_calls_routed_before_their_round_all_reach_the_app(self):
        calls = [call_item("get_order", LOOKUP, "call_1"), call_item("get_order", LOOKUP, "call_2")]
        routed = Act(lambda run: run.routes.extend(run.task.route(call) for call in calls))
        adapter, _, first = await self.waiting([routed, Round(*calls)], request="Find my order.", settings=APP)
        self.assertEqual((adapter.runs[0].routes[:2], function_calls(first)), (["app", "app"], ["call_1", "call_2"]))

    async def test_a_repeat_with_other_data_or_a_call_between_is_sent(self):
        same = {"order_id": "a", "detail": 1}
        reordered = app("get_order", {"detail": 1, "order_id": "a"}, "call_2")
        cases = {
            "another value after the same data": (same, reordered, {**same, "detail": True}, "refused"),
            "a backend call in between": (LOOKUP, OwnCall("lookup", call_id="call_2"), LOOKUP, "backend"),
        }
        for name, (arguments, between, again, route) in cases.items():
            with self.subTest(name):
                steps = [
                    app("get_order", arguments, "call_1"),
                    between,
                    app("get_order", again, "call_3"),
                ]
                adapter, im, first = await self.waiting(steps, request="Find my order.", settings=APP)
                rest = await self.answer_calls(im, first, NOT_FOUND, settings=APP)
                self.assertEqual((adapter.runs[0].routes, function_calls(rest)), (["app", route, "app"], ["call_3"]))

    async def test_a_request_since_the_call_sends_its_repeat(self):
        for when in ("spoken while it runs", "typed while it runs", "while the call waits", "with the outputs"):
            with self.subTest(when):
                gate, running = asyncio.Event(), when.endswith("it runs")
                steps = [app("get_order", LOOKUP, "call_1"), *([wait(gate)] if running else [])]
                script = [*steps, app("get_order", LOOKUP, "call_2")]
                adapter, im, events = await self.waiting(script, request="Find my order.", settings=APP)
                given = app_outputs(events, NOT_FOUND)
                if when == "with the outputs":
                    joined = im.admit_client_request([*given, *ask("Try it again.")], delegation_id="D1", settings=APP)
                else:
                    if running:
                        im.admit_client_request(given, delegation_id="D1", settings=APP)
                        await self.turns()
                    admit = im.admit_client_request if when.startswith("typed") else im.admit
                    joined = admit(ask("Try it again."), delegation_id="D2", settings=APP)
                    await self.turns()
                    if not running:
                        im.admit_client_request(given, delegation_id="D1", settings=APP)
                gate.set()
                events = await read_all(joined.stream, timeout=2)
                self.assertEqual((adapter.runs[0].routes, function_calls(events)), (["app", "app"], ["call_2"]))

    async def test_two_asks_for_the_same_add_send_both_adds(self):
        streamed = asyncio.Event()
        script = [
            Round(call_item("manage_cart", LATTE, "call_1"), streamed=streamed),
            app("manage_cart", LATTE, "call_2"),
        ]
        adapter = ScriptedAdapter([*script, app("manage_cart", LATTE, "call_3"), answer("Two lattes are in the cart.")])
        im = self.session(adapter)
        first, second = await self.admit_each(im, "Add a latte.", "Add another latte.", settings=APP)
        streamed.set()
        await read_all(first, timeout=2)
        events = await read_all(second, timeout=2)
        events = await self.answer_calls(im, events, '{"cart": ["lat"]}', delegation_id="D2", settings=APP)
        self.assertEqual(function_calls(events), ["call_3"])
        rest = await self.answer_calls(im, events, '{"cart": ["lat", "lat"]}', delegation_id="D2", settings=APP)
        assert (texts(rest), adapter.runs[0].routes) == (["Two lattes are in the cart."], ["refused", "app", "app"])


class OpenCallTests(IMCase):
    """A response that ends while its work goes on (the app's calls, a handover) ends after its open backend calls."""

    async def stock_and_menu(self, stock: list, menu: list, *later, run=ScriptedRun, **settings) -> tuple:
        """Split D1 into a stock task playing ``stock`` and a menu task playing ``menu`` once the lookup started.

        ``later`` holds the conductor's plans for the next arrivals. Return the session and D1's events,
        read in the background.
        """
        started = asyncio.Event()
        conductor = planned(Start(("D1",), (TaskSpec("stock"), TaskSpec("menu"))), *later)
        adapter = ScriptedAdapter(stock, [wait(started), *menu], conductor=conductor, max_running=2, run=run)
        im = self.session(adapter, **settings)
        first = im.admit(ask("Stock and menu."), delegation_id="D1", settings=MENU_APP).stream
        reading = asyncio.ensure_future(read_all(first))
        await self.until(lambda: "call_stock" in im.ledger.actions, "the stock lookup to start")
        started.set()
        return im, reading

    async def parked(self, im, call_id: str, reading) -> None:
        """Wait until ``call_id`` waits for the app; D1's response, its function call written, waits for the lookup."""
        await self.until(lambda: call_id in im.parked, f"{call_id} to wait for the app")
        await self.turns()
        self.assertIn(call_id, function_calls([event for _, event in self.wire.events]))
        self.assertFalse(reading.done())

    async def test_an_app_call_goes_out_at_once_its_response_ends_later(self):
        release = asyncio.Event()
        menu = [app_call(call_id="call_menu"), answer("Menu.")]
        im, reading = await self.stock_and_menu([Round(STOCK, ran=release), answer("Stock.")], menu)
        await self.parked(im, "call_menu", reading)
        release.set()
        rest = await self.answer_calls(im, await asyncio.wait_for(reading, 2), settings=MENU_APP)
        self.assertEqual(texts(rest), ["Stock.", "Menu."])
        self.assertEqual(self.timeline(), ENDINGS["a_task_waits_while_another_tasks_call_runs"]["timeline"])

    async def test_a_handover_ends_the_older_response_after_its_open_call(self):
        release, gate = asyncio.Event(), asyncio.Event()
        stock, menu = [Round(STOCK, ran=release), answer("Stock.")], [wait(gate), checkpoint(), answer("Menu.")]
        im, reading = await self.stock_and_menu(
            stock, menu, placing(Steer(("D2",), (WorkTarget("W1", ("W1.2",)),), clarifications=()))
        )
        taker = asyncio.ensure_future(read_all(im.admit(ask("Only green ones."), delegation_id="D2").stream))
        await self.until(lambda: im.ledger.steerings, "D2 to join the menu task")
        gate.set()
        await self.until(lambda: im.ledger.delegations["D1"].state == "handed_over", "the handover")
        await self.turns()
        self.assertFalse(reading.done())
        release.set()
        first, second = await asyncio.wait_for(asyncio.gather(reading, taker), 2)
        self.assertEqual(
            self.timeline(),
            [
                ["R1", "created queued"],
                ["R1", "queued"],
                ["R1", "in_progress"],
                ["R1", "mcp_call.in_progress"],
                ["R2", "created queued"],
                ["R2", "queued"],
                ["R2", "in_progress"],
                ["R2", "message final_answer"],
                ["R1", "mcp_call.completed"],
                ["R1", "mcp_call cafe/lookup_stock completed"],
                ["R1", "completed"],
                ["R2", "message final_answer"],
                ["R2", "completed"],
            ],
        )
        # The held call keeps its original attribution; the new owner receives the branch's later results.
        old_call = next(event for event in first if event["type"] == "response.output_item.done")
        self.assertEqual(decode_provenance(old_call).item_attribution.owner_ref, "D1")
        messages = [event for event in second if event["type"] == "response.output_item.done"]
        self.assertEqual([decode_provenance(event).item_attribution.owner_ref for event in messages], ["D2", "D2"])
        self.assertIsNone(second[-1]["response"].get("previous_response_id"))

    async def test_the_client_tool_deadline_counts_from_the_responses_end(self):
        release = asyncio.Event()
        menu = [app_call(call_id="call_menu"), answer("Menu.")]
        stock = [Round(STOCK, ran=release), answer("Stock.")]
        im, reading = await self.stock_and_menu(stock, menu, client_tool_timeout_seconds=0.2)
        await self.parked(im, "call_menu", reading)
        # Past the deadline while the response waits for the lookup.
        await asyncio.sleep(0.3)
        self.assertEqual(im.ledger.works["W1"].state, "waiting_client")
        release.set()
        events = await asyncio.wait_for(reading, 2)
        # Within the deadline, counted from the response's end.
        await asyncio.sleep(0.1)
        self.assertEqual(texts(await self.answer_calls(im, events, settings=MENU_APP)), ["Stock.", "Menu."])

    async def test_outputs_sent_while_the_response_waits_continue_it(self):
        release = asyncio.Event()
        menu = [app_call(call_id="call_menu"), app_call(arguments={"page": 2}, call_id="call_page"), answer("Menu.")]
        stock = [Round(STOCK, ran=release), answer("Stock.")]
        im, reading = await self.stock_and_menu(stock, menu, client_tool_timeout_seconds=0.2)
        await self.parked(im, "call_menu", reading)
        early = [{"type": "function_call_output", "call_id": "call_menu", "output": "{}"}]
        stream = im.admit_client_request(early, delegation_id="D1", settings=MENU_APP).stream
        continued = asyncio.ensure_future(read_all(stream))
        await self.parked(im, "call_page", reading)
        # Past the deadline while both responses wait.
        await asyncio.sleep(0.3)
        self.assertEqual((continued.done(), im.ledger.works["W1"].state), (False, "waiting_client"))
        release.set()
        first, second = await asyncio.wait_for(asyncio.gather(reading, continued), 2)
        (stock_call,) = [item for item in finished_items(first) if item["type"] == "mcp_call"]
        self.assertEqual((stock_call["status"], function_calls(second)), ("completed", ["call_page"]))
        self.assertEqual(texts(await self.answer_calls(im, second, settings=MENU_APP)), ["Stock.", "Menu."])

    async def test_a_stop_or_failure_with_the_call_open_ends_it_incomplete(self):
        await self.check_stock_termination(menu_already_parked=True)

    async def test_a_stop_or_failure_before_the_wait_lets_the_call_end_it(self):
        await self.check_stock_termination(menu_already_parked=False)

    async def check_stock_termination(self, *, menu_already_parked):
        for stopped in (False, True):
            with self.subTest(stopped=stopped):
                release, gate = asyncio.Event(), asyncio.Event()
                confirm = asyncio.Event()

                class GatedStop(ScriptedRun):
                    async def stop(self, confirmation=confirm):
                        await confirmation.wait()
                        return await super().stop()

                menu = [app_call(call_id="call_menu"), answer("Menu.")]
                if not menu_already_parked:
                    menu.insert(0, wait(gate))
                if stopped:
                    stock = [Round(STOCK, ran=release), answer("Stock.")]
                    notice = Start(("D2",)) if menu_already_parked else CallerMessage(("D2",), "status", "OK.")
                    later = (placing(Cancel((), (WorkTarget("W1", ("W1.1",)),), clarifications=()), notice),)
                else:
                    stock, later = [DiesWhileItRuns(STOCK, release)], ()
                pending_stop = stopped and menu_already_parked
                run = GatedStop if pending_stop else ScriptedRun
                im, reading = await self.stock_and_menu(stock, menu, *later, run=run)
                try:
                    if menu_already_parked:
                        await self.parked(im, "call_menu", reading)
                    if stopped:
                        im.admit(ask("Skip the stock."), delegation_id="D2")
                    else:
                        release.set()
                    if not menu_already_parked:
                        await self.until(lambda im=im: im.ledger.works["W1"].tasks[0].ended, "stock task ended")
                        gate.set()
                    events = await asyncio.wait_for(reading, 2)
                    if pending_stop:
                        self.assertEqual(im.ledger.works["W1"].tasks[0].stop.state, "stop_requested")
                finally:
                    confirm.set()
                items = finished_items(events)
                if not menu_already_parked:
                    items = [item for item in items if item["type"] != "message"]
                    self.assertIsNotNone(im.parked["call_menu"].timer)
                self.assertEqual(
                    [(item["type"], item["status"]) for item in items],
                    [("function_call", "completed"), ("mcp_call", "incomplete")],
                )
                self.assertEqual(events[-1]["type"], "response.completed")

    async def test_a_handover_after_a_stop_ends_the_older_response_at_once(self):
        gate = asyncio.Event()
        stock, menu = [Round(STOCK, ran=asyncio.Event()), answer("Stock.")], [wait(gate), checkpoint(), answer("Menu.")]
        later = placing(
            Cancel((), (WorkTarget("W1", ("W1.1",)),), clarifications=()),
            Steer(("D2",), (WorkTarget("W1", ("W1.2",)),), clarifications=()),
        )
        im, reading = await self.stock_and_menu(stock, menu, later)
        taker = asyncio.ensure_future(read_all(im.admit(ask("Just the menu."), delegation_id="D2").stream))
        await self.until(lambda: im.ledger.works["W1"].tasks[0].ended, "the stock task to stop")
        gate.set()
        first, second = await asyncio.wait_for(asyncio.gather(reading, taker), 2)
        items = [(item["type"], item["status"]) for item in finished_items(first)]
        self.assertEqual(items, [("message", "completed"), ("mcp_call", "incomplete")])
        self.assertEqual(texts(first), [TEXTS.cancelled_line])
        old_call = next(
            event
            for event in first
            if event["type"] == "response.output_item.done" and event["item"]["type"] == "mcp_call"
        )
        self.assertEqual(decode_provenance(old_call).item_attribution.owner_ref, "D1")
        self.assertIsNone(items_of(first, "mcp_call")[0]["output"])
        self.assertEqual(im.ledger.actions["call_stock"].state, "unclear")
        result = next(
            event for event in second if event["type"] == "response.output_item.done" and "Menu." in texts([event])
        )
        self.assertEqual(decode_provenance(result).item_attribution.owner_ref, "D2")
        assert (first[-1]["type"], texts(second)) == ("response.completed", [TEXTS.stop_confirmed_line, "Menu."])

    async def test_a_stopped_tasks_held_call_ends_with_the_next_app_call(self):
        parked = asyncio.Event()
        menu = [app_call(call_id="call_menu"), app_call(arguments={"page": 2}, call_id="call_page"), answer("Menu.")]
        stock = [wait(parked), Round(STOCK, ran=asyncio.Event()), answer("Stock.")]
        stop = placing(
            Cancel((), (WorkTarget("W1", ("W1.2",)),), clarifications=()), CallerMessage(("D2",), "status", "OK.")
        )
        conductor = planned(Start(("D1",), (TaskSpec("menu"), TaskSpec("stock"))), stop)
        im = self.session(ScriptedAdapter(menu, stock, conductor=conductor, max_running=2))
        first = await self.request(im, "Menu and stock.", "D1", settings=MENU_APP)
        parked.set()
        await self.until(lambda: "call_stock" in im.ledger.actions, "the stock lookup to start")
        await self.request(im, "Skip the stock.", "D2")
        await self.until(lambda: im.ledger.works["W1"].tasks[1].ended, "the stock task to stop")
        rest = await self.answer_calls(im, first, settings=MENU_APP)
        items = [(item["type"], item["status"]) for item in finished_items(rest)]
        self.assertEqual(items, [("function_call", "completed"), ("mcp_call", "incomplete")])
        self.assertEqual((function_calls(rest), rest[-1]["type"]), (["call_page"], "response.completed"))
        self.assertIsNotNone(im.parked["call_page"].timer)

    async def test_at_close_a_response_waiting_for_a_call_ends_cancelled(self):
        await self.check_close_with_app_call(before_close=True)

    async def test_a_wait_begun_in_the_close_grace_ends_cancelled_too(self):
        await self.check_close_with_app_call(before_close=False)

    async def check_close_with_app_call(self, *, before_close):
        gate = asyncio.Event()
        menu = [app_call(call_id="call_menu"), answer("Menu.")]
        options = {}
        if not before_close:
            menu.insert(0, wait(gate))
            options["close_grace_seconds"] = 0.5
        stock = [Round(STOCK, ran=asyncio.Event()), answer("Stock.")]
        im, reading = await self.stock_and_menu(stock, menu, **options)
        if before_close:
            await self.parked(im, "call_menu", reading)
            await im.close()
        else:
            closing = asyncio.create_task(im.close())
            await self.turns()
            gate.set()
            await self.parked(im, "call_menu", reading)
            await closing
        ended = (await asyncio.wait_for(reading, 2))[-1]
        self.assertEqual((ended["type"], ended["response"]["status"]), ("response.incomplete", "cancelled"))
        self.assertIsNone(ended["response"]["incomplete_details"])
        items = [(item["type"], item["status"]) for item in ended["response"]["output"]]
        self.assertEqual(items, [("function_call", "completed"), ("mcp_call", "incomplete")])
