# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D103

"""An IM session's calls: routing and the commit gate, the app's outputs and continuations, and repeated calls."""

import asyncio

from examples.frontend_backend_live.delegation_core.backend.port import Answer, Cancel, Start, Steer
from examples.frontend_backend_live.delegation_core.im.results import SpeakDisplay
from examples.frontend_backend_live.delegation_core.im.session import FUNCTION_CALL_OUTPUTS_REQUIRED, DelegationError
from tests.unit.delegation_core.helpers import (
    MENU,
    TEXTS,
    CancelOwnWork,
    IMCase,
    OwnCall,
    ScriptedAdapter,
    ScriptedRun,
    ScriptStep,
    Show,
    StartEach,
    answer,
    app_call,
    app_outputs,
    ask,
    call_item,
    finished_items,
    function_calls,
    placing,
    planned,
    read_all,
    texts,
    wait,
)

SPEAK_DISPLAY = SpeakDisplay(TEXTS.result_formats["speak_display"])
# The app's functions: the menu, a lookup and a write (the cafe's cart).
USER = {"type": "function", "name": "get_user_details", "parameters": {"type": "object", "properties": {}}}
CART = {"type": "function", "name": "manage_cart", "parameters": {"type": "object", "properties": {}}}
APP, MENU_APP = {"tools": [USER, CART]}, {"tools": [MENU]}
LOOKUP, LATTE, NOT_FOUND = {"user_id": "noh_mlr_9847"}, {"action": "add", "item_id": "lat"}, "Error: user not found"


def outputs(history: list[dict]) -> dict[str, str]:
    """The function outputs a history holds, by call id."""
    return {item["call_id"]: item["output"] for item in history if item.get("type") == "function_call_output"}


class Round(ScriptStep):
    """A round of ``calls``; it streams until ``streamed`` is set, and its backend calls run until ``ran`` is set.

    The task's events end when one of its calls goes to the app.
    """

    def __init__(self, *calls: dict, streamed: asyncio.Event | None = None, ran: asyncio.Event | None = None):
        """Call ``calls``; wait for ``streamed`` before the round's items, and for ``ran`` before reporting."""
        self.calls, self.streamed, self.ran = list(calls), streamed, ran

    async def play(self, run):
        if self.streamed is not None:
            run.boundary()
            await self.streamed.wait()  # newer input may arrive while the round streams
        for event in run.round_events(self.calls, boundary=self.streamed is None):
            yield event
        routes = [run.task.route(call) for call in self.calls]
        run.routes += routes
        self.ends = "app" in routes
        backend = [call for call, route in zip(self.calls, routes, strict=True) if route == "backend"]
        for call in backend:
            run.task.tool_started(call, "cafe")
        if self.ran is not None:
            await self.ran.wait()  # newer input may arrive while the backend's calls run
        for call in reversed(backend):  # out of call order
            run.task.tool_outcome(call["call_id"], f"output {call['call_id']}", "done")


def app(name: str, arguments: dict, call_id: str) -> Round:
    """A round that calls the app's function ``name``."""
    return Round(call_item(name, arguments, call_id))


class FinishesItsCall(ScriptedRun):
    """A run whose stop lets its started call finish: the call reports its output before the stop is confirmed."""

    async def stop(self):
        self.task.tool_outcome("call_book", "Booked.", "done")
        return await super().stop()


class Marked:
    """An app output step that adds its mark to every output it reads, and keeps the tools each continuation brought."""

    def __init__(self, mark: str):
        """Add ``mark``."""
        self.mark, self.tools = mark, []

    def outputs(self, calls, tools):
        self.tools.append(tools)
        return {output["call_id"]: {**output, "output": f"{output['output']} {self.mark}"} for _, output in calls}


class CallTests(IMCase):
    async def test_a_mixed_round_routes_each_call_and_keeps_every_output_in_call_order(self):
        calls = [call_item(name, {}, f"call_{name}") for name in ("lookup", "get_menu", "price")]
        adapter = ScriptedAdapter([Round(*calls), answer()])
        im = self.session(adapter)
        first = await read_all(im.admit(ask("Price the teas."), delegation_id="D1", settings=MENU_APP).stream)
        run = adapter.runs[0]
        self.assertEqual(run.routes, ["backend", "app", "backend"])
        self.assertEqual([item["type"] for item in finished_items(first)], ["mcp_call", "mcp_call", "function_call"])
        await self.answer_calls(im, first, "menu")
        expected = {"call_lookup": "output call_lookup", "call_get_menu": "menu", "call_price": "output call_price"}
        self.assertEqual(list(outputs(run.histories[-1]).items()), list(expected.items()))
        self.assertEqual([output["call_id"] for output in run.boundaries[-2].outputs], list(expected))

    async def test_a_call_routed_while_steering_waits_is_refused_until_the_steering_lands(self):
        gate = asyncio.Event()
        adapter = ScriptedAdapter([wait(gate), answer()])
        await self.admit_each(self.session(adapter), "Order two lattes.", "Make it one.", settings=MENU_APP)
        task = adapter.runs[0].task
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
        gate.set()

    async def test_newer_input_keeps_back_the_calls_of_a_round_it_reached(self):
        # A round's calls are refused when newer input came while it streamed; while its backend calls run, its app
        # calls are kept back and the backend's calls end as they ran. The next round reads the outputs, then the words.
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
                self.assertLess(last_output, newer)  # the next round reads the outputs, then the new words

    async def test_a_call_routed_while_the_conductor_decides_on_newer_input_is_held_until_it_decided(self):
        # Until a model conductor has placed a newer request, no new call of any open Work runs: the call gets its
        # not-run output, and the next round reads it with the new words, or alone when the request was answered.
        handover = TEXTS.handover_line.format(delegation="D2")
        for decision, older, taker in (
            (Steer("W1", ("D2",)), [handover], ["Booked."]),
            (Answer("D2", "It is booking the table."), ["Booked."], ["It is booking the table."]),
        ):
            with self.subTest(type(decision).__name__):
                ready, deciding = asyncio.Event(), asyncio.Event()

                async def once_decided(given, place, decision=decision, deciding=deciding):
                    await deciding.wait()  # a model conductor's call: the newer request waits for it
                    await place(decision)

                script = [wait(ready), OwnCall("book", call_id="call_book"), OwnCall("book"), answer("Booked.")]
                adapter = ScriptedAdapter(script, conductor=planned(placing(Start(("D1",))), once_decided))
                streams = await self.admit_each(self.session(adapter), "Book a table.", "By the window.")
                ready.set()  # the round ends with its call while the conductor still decides
                await self.turns()
                run = adapter.runs[0]
                self.assertEqual(len(run.histories), 1)  # held: no next round before the decision
                deciding.set()
                said = [texts(await asyncio.wait_for(read_all(stream), 2)) for stream in reversed(streams)]
                self.assertEqual(said, [taker, older])
                self.assertEqual(run.routes, ["refused", "backend"])  # planned again; then the call ran
                self.assertEqual(outputs(run.histories[1])["call_book"], TEXTS.not_run_output)

    async def test_the_workers_own_cancel_work_refuses_later_calls_writes_no_item_and_its_next_answer_counts(self):
        cancel = CancelOwnWork((call_item("lookup", {}, "call_later"),))  # then a call it may not run
        adapter = ScriptedAdapter([Show("Earl Grey"), cancel, answer("Stopped as asked.")])
        im = self.session(adapter, result_format=SPEAK_DISPLAY)
        events = await read_all(im.admit(ask("Book it."), delegation_id="D1").stream)
        self.assertEqual(adapter.runs[0].routes, ["backend", "backend", "refused"])  # show, cancel_work, the call after
        self.assertEqual(outputs(adapter.runs[0].histories[-1])["call_later"], TEXTS.stopped_output)
        calls = [(item["server_label"], item["name"]) for item in finished_items(events) if item["type"] == "mcp_call"]
        self.assertEqual(calls, [("delegation_core", "show")])  # a control call never reaches the wire
        self.assertEqual((texts(events), events[-1]["type"]), (["Stopped as asked."], "response.completed"))

    async def test_a_stopping_work_or_a_name_the_app_and_a_work_tool_share_refuses_a_call_with_its_reason(self):
        show_tool = {"type": "function", "name": "show", "parameters": {"type": "object", "properties": {}}}
        for tool, newer, reason in (("book", ["Stop."], TEXTS.stopped_output), ("show", [], TEXTS.clash_output)):
            with self.subTest(reason):
                gate, conductor = asyncio.Event(), planned(placing(Start(("D1",))), placing(Cancel("W1")))
                adapter = ScriptedAdapter([wait(gate), answer()], conductor=conductor, stop_delay=0.05)
                im = self.session(adapter, result_format=SPEAK_DISPLAY)
                await self.admit_each(im, "Book it.", *newer, settings={"tools": [show_tool]})
                task = adapter.runs[0].task
                self.assertEqual(task.route(call_item(tool, {}, "call_1")), "refused")
                self.assertEqual(task.take_boundary_input().outputs[0]["output"], reason)
                gate.set()

    async def test_a_call_that_ended_while_its_task_stopped_keeps_its_output_and_one_never_answered_is_owed(self):
        book, menu = call_item("book_table", {}, "call_book"), call_item("get_menu", {}, "call_menu")
        scripts = ([Round(book, menu, ran=asyncio.Event())], [answer()])  # book_table still runs when the Work stops
        adapter = ScriptedAdapter(*scripts, conductor=StartEach(), run=FinishesItsCall)
        im = self.session(adapter, work_timeout_seconds=0.05)  # the backstop stops it
        await read_all(im.admit(ask("Book a table and show the menu."), delegation_id="D1", settings=MENU_APP).stream)
        await read_all(im.admit(ask("Did it book?"), delegation_id="D2").stream)
        read = outputs(adapter.runs[1].histories[0])
        self.assertEqual(read, {"call_book": "Booked.", "call_menu": TEXTS.owed_output})

    async def test_a_call_that_may_have_run_gets_the_owed_output_and_shows_incomplete(self):
        # A call cut at its time limit may have had its effect: the backend is told the outcome is unknown, and its
        # mcp_call did not complete. A call id reports once.
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
        (item,) = [item for item in finished_items(events) if item["type"] == "mcp_call"]
        self.assertEqual((item["status"], item["output"]), ("incomplete", TEXTS.owed_output))
        self.assertIn("response.mcp_call.failed", [event["type"] for event in events])

    async def waiting(self, *scripts, **settings) -> tuple:
        """A session whose first request waits for the app; ``settings`` change the IM's settings."""
        im = self.session(ScriptedAdapter(*scripts), **settings)
        return im, await read_all(im.admit(ask("Menu?"), delegation_id="D1", settings=MENU_APP).stream)

    async def test_outputs_that_are_missing_duplicated_or_of_another_kind_are_refused_and_change_nothing(self):
        im, first = await self.waiting(
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
        ):
            with self.subTest(error), self.assertRaisesRegex(ValueError, error):
                im.admit_client_request(items, delegation_id="D1")
        self.assertEqual((await self.answer_calls(im, first))[-1]["type"], "response.completed")  # nothing changed

    async def test_two_waiting_responses_are_continued_in_either_order_but_never_at_once(self):
        scripts = ([app_call(), answer("Teas.")], [app_call(), answer("Coffee.")])
        im = self.session(ScriptedAdapter(*scripts, conductor=StartEach(), max_running=2))
        first = await read_all(im.admit(ask("Teas?"), delegation_id="D1", settings=MENU_APP).stream)
        second = await read_all(im.admit(ask("Coffee?"), delegation_id="D2", settings=MENU_APP).stream)
        with self.assertRaisesRegex(ValueError, "more than one response"):
            im.admit_client_request(app_outputs(first) + app_outputs(second))
        later = await self.answer_calls(im, second, delegation_id="D2")
        self.assertEqual((texts(later), texts(await self.answer_calls(im, first))), (["Coffee."], ["Teas."]))

    async def test_a_continuation_brings_new_settings_and_typed_words_after_its_outputs(self):
        im, first = await self.waiting([app_call(), answer()], client_tool_timeout_seconds=0.2)
        await asyncio.sleep(0.1)  # the deadline counts from when the calls reached the app
        typed = {"type": "message", "role": "user", "content": "Only green ones."}
        after = {"tools": [MENU], "model": "model/two", "instructions": "Answer in French."}
        continuation = im.admit_client_request([*app_outputs(first), typed], delegation_id="D1", settings=after)
        self.assertEqual(texts(await read_all(continuation.stream)), ["Done."])
        run = im.backend.runs[0]
        self.assertEqual([item.get("type") for item in run.histories[-1][-2:]], ["function_call_output", "message"])
        self.assertEqual(run.requests[-1]["model"], "model/two")
        self.assertTrue(run.requests[-1]["instructions"].startswith("Answer in French."))

    async def test_outputs_for_an_ended_work_get_an_empty_completed_and_typed_words_still_run(self):
        scripts = ([app_call(), answer()], [answer("Green teas.")])
        im, first = await self.waiting(*scripts, client_tool_timeout_seconds=0.02)
        await asyncio.sleep(0.05)  # the client-tool deadline ends the Work
        late = ["response.created", "response.queued", "response.in_progress", "response.completed"]
        self.assertEqual([e["type"] for e in await self.answer_calls(im, first, "late")], late)
        typed = {"type": "message", "role": "user", "content": "Just the green teas then."}
        ran = await read_all(im.admit_client_request([*app_outputs(first), typed], delegation_id="D1").stream)
        self.assertEqual(texts(ran), ["Green teas."])
        empty = await read_all(im.admit_client_request([], delegation_id="T1").stream)  # a typed run with no items
        self.assertEqual(texts(empty), ["Done."])

    async def test_the_task_reads_the_apps_outputs_after_the_session_output_steps_in_order(self):
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


class RepeatedCallTests(IMCase):
    async def first(self, *scripts, request="Find my account.", **options) -> tuple:
        """A session on ``scripts`` and the events of its first request."""
        adapter = ScriptedAdapter(*scripts, **options)
        im = self.session(adapter)
        return adapter, im, await read_all(im.admit(ask(request), delegation_id="D1", settings=APP).stream)

    async def test_an_app_call_that_repeats_the_newest_call_is_answered_from_the_record(self):
        # Two identical calls of one round both reach the app, which answered neither; a later round's repeat does not.
        both = Round(call_item("get_user_details", LOOKUP, "call_1"), call_item("get_user_details", LOOKUP, "call_2"))
        adapter, im, first = await self.first([both, app("get_user_details", LOOKUP, "call_3"), answer("Spell it.")])
        self.assertEqual(function_calls(first), ["call_1", "call_2"])
        rest = await self.answer_calls(im, first, NOT_FOUND, settings=APP)
        self.assertEqual((function_calls(rest), adapter.runs[0].routes), ([], ["app", "app", "refused"]))  # not sent
        repeated = TEXTS.repeated_call_output.format(output=NOT_FOUND)
        read = outputs(adapter.runs[0].histories[-1])
        self.assertEqual(read, {"call_1": NOT_FOUND, "call_2": NOT_FOUND, "call_3": repeated})

    async def test_a_repeat_reaches_the_app_when_its_arguments_differ_as_data_or_a_call_ran_in_between(self):
        same = {"user_id": "a", "detail": 1}
        reordered = app("get_user_details", {"detail": 1, "user_id": "a"}, "call_2")  # the same data, so not sent
        cases = {
            "another value after the same data": (same, reordered, {**same, "detail": True}, "refused"),
            "a backend call in between": (LOOKUP, OwnCall("lookup", call_id="call_2"), LOOKUP, "backend"),
        }
        for name, (arguments, between, again, route) in cases.items():
            with self.subTest(name):
                steps = [
                    app("get_user_details", arguments, "call_1"),
                    between,
                    app("get_user_details", again, "call_3"),
                ]
                adapter, im, first = await self.first(steps)
                rest = await self.answer_calls(im, first, NOT_FOUND, settings=APP)
                self.assertEqual((adapter.runs[0].routes, function_calls(rest)), (["app", route, "app"], ["call_3"]))

    async def test_a_new_request_asks_anew_so_its_identical_add_reaches_the_app(self):
        # "Add another latte.": the caller asked again, so the add reaches the app; the model's repeat of it within that
        # request is answered from the record, so the cart gets two lattes.
        first = [app("manage_cart", LATTE, "call_1"), answer("Added a latte.")]
        second = [app("manage_cart", LATTE, "call_2"), app("manage_cart", LATTE, "call_3"), answer("Added another.")]
        adapter, im, events = await self.first(first, second, request="Add a latte.")
        await self.answer_calls(im, events, '{"cart": ["lat"]}', settings=APP)
        events = await read_all(im.admit(ask("Add another latte."), delegation_id="D2", settings=APP).stream)
        self.assertEqual(function_calls(events), ["call_2"])
        rest = await self.answer_calls(im, events, '{"cart": ["lat", "lat"]}', delegation_id="D2", settings=APP)
        self.assertEqual((function_calls(rest), adapter.runs[1].routes), ([], ["app", "refused"]))

    async def test_a_request_since_the_call_ran_sends_its_repeat_to_the_app(self):
        # "Try it again." spoken or typed while the Work runs, while its call waits for the app, or typed with the app's
        # outputs: the caller asked for the same call again, so the repeat reaches the app.
        for when in ("spoken while it runs", "typed while it runs", "while the call waits", "with the outputs"):
            with self.subTest(when):
                gate, running = asyncio.Event(), when.endswith("it runs")
                steps = [app("get_user_details", LOOKUP, "call_1"), *([wait(gate)] if running else [])]
                adapter, im, events = await self.first([*steps, app("get_user_details", LOOKUP, "call_2")])
                given = app_outputs(events, NOT_FOUND)
                if when == "with the outputs":
                    joined = im.admit_client_request([*given, *ask("Try it again.")], delegation_id="D1", settings=APP)
                else:
                    if running:  # the outputs are in, and the Work runs on
                        im.admit_client_request(given, delegation_id="D1", settings=APP)
                        await self.turns()
                    admit = im.admit_client_request if when.startswith("typed") else im.admit
                    joined = admit(ask("Try it again."), delegation_id="D2", settings=APP)
                    await self.turns()
                    if not running:  # the call waited for the app until now
                        im.admit_client_request(given, delegation_id="D1", settings=APP)
                gate.set()
                events = await asyncio.wait_for(read_all(joined.stream), 2)
                self.assertEqual((adapter.runs[0].routes, function_calls(events)), (["app", "app"], ["call_2"]))

    async def test_an_identical_call_of_another_task_reaches_the_app(self):
        gate = asyncio.Event()
        first = [app("manage_cart", LATTE, "call_1"), answer("Added a latte.")]
        second = [wait(gate), app("manage_cart", LATTE, "call_2"), answer("Added another latte.")]
        adapter, im, events = await self.first(first, second, max_running=2, conductor=StartEach())
        two = im.admit(ask("Add another latte."), delegation_id="D2", settings=APP)
        await self.answer_calls(im, events, '{"cart": ["lat"]}', settings=APP)
        gate.set()
        events = await asyncio.wait_for(read_all(two.stream), 2)
        self.assertEqual((adapter.runs[1].routes, function_calls(events)), (["app"], ["call_2"]))

    async def test_a_work_that_took_two_asks_for_the_same_add_sends_both_adds_to_the_app(self):
        # "Add another latte." comes while the first round still streams: the commit gate refuses that round's add, and
        # the task plans again with both asks, so its second identical add reaches the app.
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
