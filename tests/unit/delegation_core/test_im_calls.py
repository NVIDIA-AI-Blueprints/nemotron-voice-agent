# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D103, D107

"""An IM session's calls: routing and the commit gate, the app's outputs and continuations, and repeated calls."""

import asyncio

from examples.frontend_backend_live.delegation_core.backend.port import Answer, Cancel, Start, Steer
from examples.frontend_backend_live.delegation_core.im.results import SpeakDisplay
from examples.frontend_backend_live.delegation_core.im.session import FUNCTION_CALL_OUTPUTS_REQUIRED, DelegationError
from tests.unit.core_helpers import load
from tests.unit.delegation_core.helpers import (
    MENU,
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
    call_item,
    checkpoint,
    finished_items,
    function_calls,
    items_of,
    placing,
    planned,
    read_all,
    texts,
    wait,
)

SPEAK_DISPLAY = SpeakDisplay(TEXTS.result_formats["speak_display"])
USER = {"type": "function", "name": "get_member", "parameters": {"type": "object", "properties": {}}}
ORDER = {**USER, "name": "get_order"}
CART = {"type": "function", "name": "manage_cart", "parameters": {"type": "object", "properties": {}}}
APP, MENU_APP = {"tools": [ORDER, CART]}, {"tools": [MENU]}
LOOKUP, LATTE, NOT_FOUND = {"order_id": "BB-1042"}, {"action": "add", "item_id": "lat"}, "Error: order not found"
VEGAN = {"type": "message", "role": "user", "content": "Make the food vegan."}
MILK = {"type": "message", "role": "user", "content": "With milk, please."}
ENDINGS = load()
STOCK = call_item("lookup_stock", {}, "call_stock")


def outputs(history: list[dict]) -> dict[str, str]:
    return {item["call_id"]: item["output"] for item in history if item.get("type") == "function_call_output"}


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

    async def test_a_call_while_steering_waits_is_refused_until_it_lands(self):
        gate = asyncio.Event()
        adapter = ScriptedAdapter([wait(gate), answer()])
        await self.admit_each(self.session(adapter), "Order two lattes.", "Make it one.", settings=MENU_APP)
        task = adapter.runs[0].task
        self.assertEqual(task.route(call_item("get_menu", {}, "c1")), "refused")
        self.assertEqual(task.route(call_item("place_order", {}, "c2")), "refused")
        given = task.take_boundary_input()
        self.assertEqual([output["output"] for output in given.outputs], [TEXTS.not_run_output] * 2)
        self.assertEqual(task.route(call_item("get_menu", {}, "c3")), "refused")
        task.settle(given.steering, "not_applied")
        self.assertEqual(task.route(call_item("get_menu", {}, "c4")), "refused")
        task.settle(task.take_boundary_input().steering, "applied")
        self.assertEqual(task.route(call_item("get_menu", {}, "c5")), "app")
        self.assertEqual(task.route(call_item("place_order", {}, "c6")), "backend")
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
                gate = asyncio.Event()
                adapter = ScriptedAdapter([Round(*calls, **{gated: gate}), answer("Nothing was ordered.")])
                streams = await self.admit_each(self.session(adapter), "Order two.", "Never mind.", settings=MENU_APP)
                gate.set()
                older, taker = [await asyncio.wait_for(read_all(stream), 2) for stream in streams]
                run = adapter.runs[0]
                self.assertEqual(run.routes, routes)
                self.assertEqual((function_calls(older + taker), texts(taker)), ([], ["Nothing was ordered."]))
                history = run.histories[-1]
                self.assertEqual(outputs(history), {**kept, "call_menu": TEXTS.not_run_output})
                newer = next(n for n, item in enumerate(history) if "Never mind." in str(item))
                last_output = max(n for n, item in enumerate(history) if item.get("type") == "function_call_output")
                self.assertLess(last_output, newer)

    async def test_a_call_waits_while_the_conductor_decides(self):
        handover = TEXTS.handover_line.format(delegation="D2")
        for decision, older, taker in (
            (Steer("W1", ("D2",)), [handover], ["Booked."]),
            (Answer("D2", "It is booking the table."), ["Booked."], ["It is booking the table."]),
        ):
            with self.subTest(type(decision).__name__):
                ready, deciding = asyncio.Event(), asyncio.Event()

                async def once_decided(given, place, decision=decision, deciding=deciding):
                    # A model conductor's call: the newer request waits for it.
                    await deciding.wait()
                    await place(decision)

                script = [wait(ready), OwnCall("book", call_id="call_book"), OwnCall("book"), answer("Booked.")]
                adapter = ScriptedAdapter(script, conductor=planned(placing(Start(("D1",))), once_decided))
                streams = await self.admit_each(self.session(adapter), "Book a table.", "By the window.")
                ready.set()
                await self.turns()
                run = adapter.runs[0]
                self.assertEqual(len(run.histories), 1)
                deciding.set()
                said = [texts(await asyncio.wait_for(read_all(stream), 2)) for stream in reversed(streams)]
                self.assertEqual(said, [taker, older])
                self.assertEqual(run.routes, ["refused", "backend"])
                self.assertEqual(outputs(run.histories[1])["call_book"], TEXTS.not_run_output)

    async def test_the_tasks_own_cancel_work_refuses_later_calls(self):
        cancel = WorkTool("cancel_work", {}, (call_item("lookup", {}, "call_later"),))
        adapter = ScriptedAdapter([WorkTool("show", {"display": "Earl Grey"}), cancel, answer("Stopped as asked.")])
        im = self.session(adapter, result_format=SPEAK_DISPLAY)
        events = await read_all(im.admit(ask("Book it."), delegation_id="D1").stream)
        self.assertEqual(adapter.runs[0].routes, ["backend", "backend", "refused"])
        self.assertEqual(outputs(adapter.runs[0].histories[-1])["call_later"], TEXTS.stopped_output)
        calls = [(item["server_label"], item["name"]) for item in items_of(events, "mcp_call")]
        self.assertEqual(calls, [("delegation_core", "show")])
        self.assertEqual((texts(events), events[-1]["type"]), (["Stopped as asked."], "response.completed"))

    async def test_a_refused_call_gets_its_reason(self):
        show_tool = {"type": "function", "name": "show", "parameters": {"type": "object", "properties": {}}}
        plain = {"tools": [show_tool]}
        listed = {**plain, "tool_choice": {"type": "allowed_tools", "tools": []}}
        stopped, clash = TEXTS.stopped_output, TEXTS.clash_output
        rows = (("book", ["Stop."], plain, stopped), ("show", ["Stop."], listed, stopped))
        for tool, newer, settings, reason in (*rows, ("show", [], plain, clash), ("show", [], listed, clash)):
            with self.subTest(reason, tool=tool, listed=settings is listed):
                gate, conductor = asyncio.Event(), planned(placing(Start(("D1",))), placing(Cancel("W1")))
                adapter = ScriptedAdapter([wait(gate), answer()], conductor=conductor, stop_delay=0.05)
                im = self.session(adapter, result_format=SPEAK_DISPLAY)
                await self.admit_each(im, "Book it.", *newer, settings=settings)
                task = adapter.runs[0].task
                self.assertEqual(task.route(call_item(tool, {}, "call_1")), "refused")
                self.assertEqual(task.take_boundary_input().outputs[0]["output"], reason)
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
        self.assertEqual([output["output"] for output in given.outputs], [TEXTS.unlisted_output, TEXTS.not_run_output])
        task.settle(given.steering, "applied")
        names = ("set_decaf", "get_menu", "lookup", "cancel_work")
        routes = [task.route(call_item(name, {}, f"call_{name}")) for name in names]
        self.assertEqual(routes, ["refused", "app", "backend", "backend"])
        self.assertEqual(task.take_boundary_input().outputs[0]["output"], TEXTS.unlisted_output)
        gate.set()

    async def test_a_stopped_tasks_unanswered_call_is_owed(self):
        book, menu = call_item("book_table", {}, "call_book"), call_item("get_menu", {}, "call_menu")
        scripts = ([Round(book, menu, ran=asyncio.Event())], [answer()])
        adapter = ScriptedAdapter(*scripts, conductor=StartEach(), run=FinishesItsCall)
        # The backstop stops the Work.
        im = self.session(adapter, work_timeout_seconds=0.05)
        await read_all(im.admit(ask("Book a table and show the menu."), delegation_id="D1", settings=MENU_APP).stream)
        await read_all(im.admit(ask("Did it book?"), delegation_id="D2").stream)
        read = outputs(adapter.runs[1].histories[0])
        self.assertEqual(read, {"call_book": "Booked.", "call_menu": TEXTS.owed_output})

    async def test_a_call_that_may_have_run_gets_the_owed_output(self):
        gate = asyncio.Event()
        adapter = ScriptedAdapter([wait(gate), answer()])
        (stream,) = await self.admit_each(self.session(adapter), "Book it.")
        task, call = adapter.runs[0].task, call_item("book", {"table": 4}, "call_book")
        self.assertEqual(task.route(call), "backend")
        task.tool_started(call, "cafe")
        task.tool_outcome("call_book", "book did not answer within 60 s", "unclear")
        task.tool_outcome("call_book", "again", "done")
        self.assertEqual(task.take_boundary_input().outputs[0]["output"], TEXTS.owed_output)
        gate.set()
        events = await read_all(stream)
        (item,) = items_of(events, "mcp_call")
        self.assertEqual((item["status"], item["output"]), ("incomplete", TEXTS.owed_output))
        self.assertIn("response.mcp_call.failed", [event["type"] for event in events])

    async def test_bad_outputs_are_refused_and_change_nothing(self):
        _, im, first = await self.waiting(
            [Round(call_item("get_menu", {}, "c1"), call_item("get_menu", {}, "c2")), answer()]
        )
        with self.assertRaises(DelegationError) as refused:
            im.admit_client_request(app_outputs(first)[:1], delegation_id="D1")
        self.assertEqual(refused.exception.code, FUNCTION_CALL_OUTPUTS_REQUIRED)
        output = app_outputs(first)[0]
        for items, error in (
            ([output, output], "Duplicate"),
            ([{**output, "type": "custom_tool_call_output"}], "function_call_output"),
            ([{"type": "function_call_output", "output": "{}"}], "function_call_output"),
            ([{"type": "function_call_output", "call_id": output["call_id"]}, *app_outputs(first)[1:]], "no output"),
        ):
            with self.subTest(error), self.assertRaisesRegex(ValueError, error):
                im.admit_client_request(items, delegation_id="D1")
        self.assertEqual((await self.answer_calls(im, first))[-1]["type"], "response.completed")

    async def test_two_waiting_responses_continue_one_at_a_time(self):
        scripts = ([app_call(), answer("Teas.")], [app_call(), answer("Coffee.")])
        im = self.session(ScriptedAdapter(*scripts, conductor=StartEach(), max_running=2))
        first = await read_all(im.admit(ask("Teas?"), delegation_id="D1", settings=MENU_APP).stream)
        second = await read_all(im.admit(ask("Coffee?"), delegation_id="D2", settings=MENU_APP).stream)
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
        self.assertEqual([e["type"] for e in await self.answer_calls(im, first, "late")], late)
        typed = {"type": "message", "role": "user", "content": "Just the green teas then."}
        ran = await read_all(im.admit_client_request([*app_outputs(first), typed], delegation_id="D1").stream)
        self.assertEqual(texts(ran), ["Green teas."])
        empty = await read_all(im.admit_client_request([], delegation_id="T1").stream)
        self.assertEqual(texts(empty), ["Done."])

    async def split_waiting(self, food: list, *more) -> tuple:
        """Split D1 into a drink task, which waits for the app's menu, and a food task that plays ``food``.

        ``more`` is what the conductor places for D2, which then comes. Return the session and D1's events.
        """
        plans = (placing(Start(("D1",), ("drink", "food"))), *([placing(*more)] if more else []))
        scripts = ([app_call(call_id="call_drink"), answer("Drink.")], food, [answer("Skipped.")])
        im = self.session(ScriptedAdapter(*scripts, conductor=planned(*plans), max_running=3))
        first = await read_all(im.admit(ask("A drink and some food."), delegation_id="D1", settings=MENU_APP).stream)
        if more:
            await read_all(im.admit(ask("Skip the drink."), delegation_id="D2").stream)
        return im, first

    async def test_typed_words_reach_every_open_task_after_the_outputs(self):
        gate = asyncio.Event()
        im, first = await self.split_waiting([wait(gate), checkpoint(), answer("Food.")])
        rest = im.admit_client_request([*app_outputs(first), VEGAN], delegation_id="D1", settings=MENU_APP)
        await self.turns()
        gate.set()
        self.assertEqual(texts(await asyncio.wait_for(read_all(rest.stream), 2)), ["Drink.", "Food."])
        drink, food = im.backend.runs[:2]
        *_, output, typed = drink.histories[-1]
        self.assertEqual((output["type"], typed), ("function_call_output", VEGAN))
        self.assertIn(VEGAN, food.histories[-1])

    async def test_typed_words_of_a_stopped_task_reach_the_open_tasks(self):
        gate = asyncio.Event()
        skip = (Cancel("W1", "W1.1"), Start(("D2",)))
        im, first = await self.split_waiting([wait(gate), checkpoint(), answer("Food.")], *skip)
        await self.turns()
        rest = im.admit_client_request([*app_outputs(first), VEGAN], delegation_id="D1", settings=MENU_APP)
        gate.set()
        self.assertEqual(texts(await asyncio.wait_for(read_all(rest.stream), 2)), ["Food."])
        self.assertIn(VEGAN, im.backend.runs[1].histories[-1])

    async def test_typed_words_no_task_takes_join_the_conversation(self):
        gate = asyncio.Event()
        im, first = await self.split_waiting([wait(gate), answer("Food.")], Cancel("W1", "W1.1"), Start(("D2",)))
        gate.set()
        await self.turns()
        rest = im.admit_client_request([*app_outputs(first), VEGAN], delegation_id="D1", settings=MENU_APP)
        self.assertEqual(texts(await asyncio.wait_for(read_all(rest.stream), 2)), ["Food."])
        self.assertIn(VEGAN, im.history)

    async def test_typed_words_never_applied_join_and_start_no_work(self):
        im = self.session(ScriptedAdapter([app_call(), checkpoint(), answer("Teas.")], delivery="not_applied"))
        first = await read_all(im.admit(ask("Find teas."), delegation_id="D1", settings=MENU_APP).stream)
        rest = im.admit_client_request([*app_outputs(first), MILK], delegation_id="D1", settings=MENU_APP)
        self.assertEqual(texts(await asyncio.wait_for(read_all(rest.stream), 2)), ["Teas."])
        await self.turns()
        self.assertIn(MILK, im.history)
        self.assertNotIn("im.follow_up", [kind for kind, _ in self.traces])

    async def test_typed_words_go_after_the_outputs_before_a_steered_one(self):
        adapter, im, first = await self.waiting([app_call(), answer("Green: two.")])
        second = im.admit(ask("Green only."), delegation_id="D2", settings=MENU_APP)
        await self.turns()
        rest = im.admit_client_request([*app_outputs(first), MILK], delegation_id="D1", settings=MENU_APP)
        await asyncio.wait_for(read_all(rest.stream), 2)
        self.assertEqual(texts(await asyncio.wait_for(read_all(second.stream), 2)), ["Green: two."])
        *_, output, typed, steered = adapter.runs[0].histories[-1]
        self.assertEqual((output["type"], typed), ("function_call_output", MILK))
        parts = [part["text"] for part in steered["content"]]
        self.assertEqual((parts[0], parts[-1]), ("Green only.", TEXTS.closing_line))

    async def test_the_apps_outputs_pass_the_output_steps_in_order(self):
        lookup = {"tools": [{**USER, "name": "get_member"}]}
        first_step = Marked("(a)")
        for steps, read in (((first_step, Marked("(b)")), "Not found. (a) (b)"), ((), "Not found.")):
            with self.subTest(steps=len(steps)):
                adapter = ScriptedAdapter([app("get_member", {"user_id": "maya_ross_3817"}, "call_1"), answer()])
                im = self.session(adapter, output_steps=steps)
                first = await read_all(im.admit(ask("Find my account."), delegation_id="D1", settings=lookup).stream)
                await self.answer_calls(im, first, "Not found.", settings=lookup)
                self.assertEqual(outputs(adapter.runs[0].histories[-1]), {"call_1": read})
        self.assertEqual(first_step.tools, [lookup["tools"]])  # the continuation's own functions

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
                events = await asyncio.wait_for(read_all(joined.stream), 2)
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
        await asyncio.wait_for(read_all(first), 2)
        events = await asyncio.wait_for(read_all(second), 2)
        events = await self.answer_calls(im, events, '{"cart": ["lat"]}', delegation_id="D2", settings=APP)
        self.assertEqual(function_calls(events), ["call_3"])
        rest = await self.answer_calls(im, events, '{"cart": ["lat", "lat"]}', delegation_id="D2", settings=APP)
        self.assertEqual(
            (texts(rest), adapter.runs[0].routes), (["Two lattes are in the cart."], ["refused", "app", "app"])
        )


class OpenCallTests(IMCase):
    """A response that ends while its work goes on (the app's calls, a handover) ends after its open backend calls."""

    async def stock_and_menu(self, stock: list, menu: list, *later, stop_delay: float = 0.0, **settings) -> tuple:
        """Split D1 into a stock task playing ``stock`` and a menu task playing ``menu`` once the lookup started.

        ``later`` holds the conductor's plans for the next arrivals, and ``stop_delay`` is how long a run takes to
        confirm its stop. Return the session and D1's events, read in the background.
        """
        started = asyncio.Event()
        conductor = planned(placing(Start(("D1",), ("stock", "menu"))), *later)
        adapter = ScriptedAdapter(
            stock, [wait(started), *menu], conductor=conductor, max_running=2, stop_delay=stop_delay
        )
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
        im, reading = await self.stock_and_menu(stock, menu, placing(Steer("W1", ("D2",), "W1.2")))
        taker = asyncio.ensure_future(read_all(im.admit(ask("Only green ones."), delegation_id="D2").stream))
        await self.until(lambda: im.ledger.steerings, "D2 to join the menu task")
        gate.set()
        await self.until(lambda: im.ledger.delegations["D1"].state == "handed_over", "the handover")
        await self.turns()
        self.assertFalse(reading.done())
        release.set()
        await asyncio.wait_for(asyncio.gather(reading, taker), 2)
        self.assertEqual(self.timeline(), ENDINGS["handed_over_while_a_call_runs"]["timeline"])

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
        menu = [app_call(call_id="call_menu"), answer("Menu.")]
        for case in ("stopped", "failed"):
            with self.subTest(case):
                release = asyncio.Event()
                if case == "stopped":
                    stock = [Round(STOCK, ran=release), answer("Stock.")]
                    later = (placing(Cancel("W1", "W1.1"), Start(("D2",))),)
                else:
                    stock, later = [DiesWhileItRuns(STOCK, release)], ()
                im, reading = await self.stock_and_menu(stock, menu, *later, stop_delay=0.3)
                await self.parked(im, "call_menu", reading)
                if case == "stopped":
                    im.admit(ask("Skip the stock."), delegation_id="D2")
                else:
                    release.set()
                events = await asyncio.wait_for(reading, 2)
                items = [(item["type"], item["status"]) for item in finished_items(events)]
                self.assertEqual(items, [("function_call", "completed"), ("mcp_call", "incomplete")])
                self.assertEqual(events[-1]["type"], "response.completed")
                if case == "stopped":
                    self.assertEqual(im.ledger.works["W1"].tasks[0].stop.state, "stop_requested")

    async def test_a_stop_or_failure_before_the_wait_lets_the_call_end_it(self):
        for case in ("stopped", "failed"):
            with self.subTest(case):
                release, gate = asyncio.Event(), asyncio.Event()
                menu = [wait(gate), app_call(call_id="call_menu"), answer("Menu.")]
                if case == "stopped":
                    stock = [Round(STOCK, ran=release), answer("Stock.")]
                    later = (placing(Cancel("W1", "W1.1"), Answer("D2", "OK.")),)
                else:
                    stock, later = [DiesWhileItRuns(STOCK, release)], ()
                im, reading = await self.stock_and_menu(stock, menu, *later)
                if case == "stopped":
                    im.admit(ask("Skip the stock."), delegation_id="D2")
                else:
                    release.set()
                await self.until(lambda im=im: im.ledger.works["W1"].tasks[0].ended, "the stock task to end")
                gate.set()
                events = await asyncio.wait_for(reading, 2)
                calls = [(item["type"], item["status"]) for item in finished_items(events) if item["type"] != "message"]
                self.assertEqual(calls, [("function_call", "completed"), ("mcp_call", "incomplete")])
                self.assertEqual(events[-1]["type"], "response.completed")
                self.assertIsNotNone(im.parked["call_menu"].timer)

    async def test_a_handover_after_a_stop_ends_the_older_response_at_once(self):
        gate = asyncio.Event()
        stock, menu = [Round(STOCK, ran=asyncio.Event()), answer("Stock.")], [wait(gate), checkpoint(), answer("Menu.")]
        later = placing(Cancel("W1", "W1.1"), Steer("W1", ("D2",), "W1.2"))
        im, reading = await self.stock_and_menu(stock, menu, later)
        taker = asyncio.ensure_future(read_all(im.admit(ask("Just the menu."), delegation_id="D2").stream))
        await self.until(lambda: im.ledger.works["W1"].tasks[0].ended, "the stock task to stop")
        gate.set()
        first, second = await asyncio.wait_for(asyncio.gather(reading, taker), 2)
        items = [(item["type"], item["status"]) for item in finished_items(first)]
        self.assertEqual(items, [("message", "completed"), ("mcp_call", "incomplete"), ("message", "completed")])
        self.assertEqual(texts(first), [TEXTS.cancelled_line, TEXTS.handover_line.format(delegation="D2")])
        self.assertEqual((first[-1]["type"], texts(second)), ("response.completed", ["Menu."]))

    async def test_a_stopped_tasks_held_call_ends_with_the_next_app_call(self):
        parked = asyncio.Event()
        menu = [app_call(call_id="call_menu"), app_call(arguments={"page": 2}, call_id="call_page"), answer("Menu.")]
        stock = [wait(parked), Round(STOCK, ran=asyncio.Event()), answer("Stock.")]
        stop = placing(Cancel("W1", "W1.2"), Answer("D2", "OK."))
        conductor = planned(placing(Start(("D1",), ("menu", "stock"))), stop)
        im = self.session(ScriptedAdapter(menu, stock, conductor=conductor, max_running=2))
        first = await read_all(im.admit(ask("Menu and stock."), delegation_id="D1", settings=MENU_APP).stream)
        parked.set()
        await self.until(lambda: "call_stock" in im.ledger.actions, "the stock lookup to start")
        await read_all(im.admit(ask("Skip the stock."), delegation_id="D2").stream)
        await self.until(lambda: im.ledger.works["W1"].tasks[1].ended, "the stock task to stop")
        rest = await self.answer_calls(im, first, settings=MENU_APP)
        items = [(item["type"], item["status"]) for item in finished_items(rest)]
        self.assertEqual(items, [("function_call", "completed"), ("mcp_call", "incomplete")])
        self.assertEqual((function_calls(rest), rest[-1]["type"]), (["call_page"], "response.completed"))
        self.assertIsNotNone(im.parked["call_page"].timer)

    async def test_at_close_a_response_waiting_for_a_call_ends_cancelled(self):
        menu = [app_call(call_id="call_menu"), answer("Menu.")]
        im, reading = await self.stock_and_menu([Round(STOCK, ran=asyncio.Event()), answer("Stock.")], menu)
        await self.parked(im, "call_menu", reading)
        await im.close()
        ended = (await asyncio.wait_for(reading, 2))[-1]
        self.assertEqual(ended["type"], "response.incomplete")
        self.assertEqual((ended["response"]["status"], ended["response"]["incomplete_details"]), ("cancelled", None))
        items = [(item["type"], item["status"]) for item in ended["response"]["output"]]
        self.assertEqual(items, [("function_call", "completed"), ("mcp_call", "incomplete")])

    async def test_a_wait_begun_in_the_close_grace_ends_cancelled_too(self):
        gate = asyncio.Event()
        menu = [wait(gate), app_call(call_id="call_menu"), answer("Menu.")]
        stock = [Round(STOCK, ran=asyncio.Event()), answer("Stock.")]
        im, reading = await self.stock_and_menu(stock, menu, close_grace_seconds=0.5)
        closing = asyncio.ensure_future(im.close())
        await self.turns()
        gate.set()
        await self.parked(im, "call_menu", reading)
        await closing
        ended = (await asyncio.wait_for(reading, 2))[-1]
        self.assertEqual((ended["type"], ended["response"]["status"]), ("response.incomplete", "cancelled"))
        items = [(item["type"], item["status"]) for item in ended["response"]["output"]]
        self.assertEqual(items, [("function_call", "completed"), ("mcp_call", "incomplete")])
