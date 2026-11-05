# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D103

"""An IM session's responses: admission, the items and notes it writes, its instructions, failures, close, history."""

import asyncio
import json
from dataclasses import replace
from unittest import mock

from examples.frontend_backend_live.delegation_core.backend.port import Start
from examples.frontend_backend_live.delegation_core.im.results import CallNotes, SpeakDisplay
from examples.frontend_backend_live.delegation_core.im.session import DelegationError
from tests.unit.delegation_core.helpers import (
    CORE_SETTINGS,
    DEFAULTS,
    MENU,
    TEXTS,
    Cut,
    Empty,
    Fail,
    FailedAttempt,
    IMCase,
    OwnCall,
    Progress,
    Say,
    ScriptedAdapter,
    ScriptedConductor,
    ScriptStep,
    Show,
    StartEach,
    answer,
    app_call,
    ask,
    call_item,
    finished_items,
    message_item,
    placing,
    planned,
    read_all,
    records_of,
    said,
    texts,
    wait,
)

SPEAK_DISPLAY = SpeakDisplay(TEXTS.result_formats["speak_display"])


class TwoAnswers(ScriptStep):
    """A round with two final_answer messages."""

    ends = True

    async def play(self, run):
        items = [message_item("First.", "final_answer", "m1"), message_item("Last.", "final_answer", "m2")]
        for event in run.round_events(items):
            yield event


class ResponseTests(IMCase):
    async def test_a_response_opens_at_admission_and_reports_each_item_and_the_newest_rounds_usage(self):
        usage = {"input_tokens": 12, "output_tokens": 3, "total_tokens": 15}
        steps = [Say("Looking."), OwnCall(), Say("Almost."), OwnCall(), answer(usage=usage)]
        events = await read_all(self.session(ScriptedAdapter(steps)).admit(ask("Find teas.")).stream)
        opened = ["response.created", "response.queued", "response.in_progress"]
        self.assertEqual([e["type"] for e in events[:3]], opened)
        self.assertEqual([e["response"]["status"] for e in events[:3]], ["queued", "queued", "in_progress"])
        self.assertEqual(len({e["response"]["id"] for e in events if "response" in e}), 1)
        indexes = [e["output_index"] for e in events if e["type"] == "response.output_item.done"]
        self.assertEqual(indexes, list(range(len(indexes))))  # each item has its own index
        self.assertEqual(events[-1]["response"]["usage"], usage)

    async def test_admission_refuses_a_full_queue_an_id_in_use_and_a_closing_session(self):
        im = self.session(ScriptedAdapter([wait(asyncio.Event())], conductor=StartEach()))
        await self.admit_each(im, *(f"Request {n}." for n in range(1, 18)))  # one runs, sixteen wait
        with self.assertRaisesRegex(DelegationError, "full"):
            im.admit(ask("One too many."), delegation_id="D18")
        with self.assertRaisesRegex(DelegationError, "already exists"):
            im.admit(ask("Again."), delegation_id="D1")
        await im.close()
        with self.assertRaisesRegex(DelegationError, "closing"):
            im.admit(ask("Late."), delegation_id="D19")

    async def test_a_reader_that_stops_reading_leaves_its_response_running(self):
        gate = asyncio.Event()
        adapter = ScriptedAdapter([wait(gate), OwnCall("book_table"), answer()], [answer()], conductor=StartEach())
        im = self.session(adapter)
        await im.admit(ask("Book it."), delegation_id="D1").stream.aclose()
        gate.set()
        await read_all(im.admit(ask("Booked?"), delegation_id="D2").stream)
        (record,) = records_of(adapter.runs[1].histories[0][-1])
        self.assertEqual((record["state"], record["steps"]), ("answered", ["book_table: done"]))

    async def test_the_answer_is_the_last_final_answer_and_a_failed_attempt_writes_nothing(self):
        im = self.session(ScriptedAdapter([TwoAnswers()], [FailedAttempt(), answer()], conductor=StartEach()))
        two = await read_all(im.admit(ask("Find teas."), delegation_id="D1").stream)
        retried = await read_all(im.admit(ask("Find coffee."), delegation_id="D2").stream)
        self.assertEqual(said(two), [("commentary", "First."), ("final_answer", "Last.")])
        self.assertEqual((said(retried), retried[-1]["type"]), ([("final_answer", "Done.")], "response.completed"))
        self.assertNotIn("Half an answ", str(retried))

    async def test_a_backend_call_is_an_mcp_call_with_its_official_events(self):
        failed = {"type": "mcp_tool_execution_error", "content": "No teas."}
        for fails, ending, error in ((False, "completed", None), (True, "failed", failed)):
            with self.subTest(ending):
                call = OwnCall(output="No teas.", fails=fails)
                events = await read_all(self.session(ScriptedAdapter([call, answer()])).admit(ask("Teas?")).stream)
                mcp = [e for e in events if "mcp" in e["type"] or e.get("item", {}).get("type") == "mcp_call"]
                kinds = ["output_item.added", "mcp_call.in_progress", f"mcp_call.{ending}", "output_item.done"]
                self.assertEqual([e["type"] for e in mcp], [f"response.{kind}" for kind in kinds])
                item = mcp[-1]["item"]
                self.assertEqual((item["server_label"], item["name"], item["status"]), ("cafe", "lookup", ending))
                self.assertEqual((item["output"], item["error"]), ("No teas.", error))
                self.assertEqual(len({e["item_id"] for e in mcp if "item_id" in e} | {item["id"]}), 1)

    async def test_backend_calls_as_notes_show_only_a_note_and_the_display_still_shows_under_its_label(self):
        adapter = ScriptedAdapter([OwnCall(), Show("Earl Grey 3.25\nSencha 4.10"), answer("Two green teas.")])
        reports = {"call_reports": CallNotes(), "progress_note_seconds": 0.01, "display_label": "screen"}
        im = self.session(adapter, result_format=SPEAK_DISPLAY, **reports)
        note, shown, spoken = finished_items(await read_all(im.admit(ask("Find teas."), delegation_id="D1").stream))
        self.assertEqual((note["phase"], note["content"][0]["text"]), ("commentary", "lookup: done"))
        self.assertEqual((shown["type"], shown["server_label"], shown["name"]), ("mcp_call", "screen", "show"))
        self.assertEqual(shown["output"], "Earl Grey 3.25\nSencha 4.10")
        self.assertEqual((spoken["phase"], spoken["content"][0]["text"]), ("final_answer", "Two green teas."))

    async def test_progress_notes_come_one_per_interval_the_newest_waiting_wins_and_none_outlives_its_response(self):
        cases = (
            (0.05, [Progress("First."), Progress("Second."), Progress("Third.")], ["First.", "Third."]),
            (5, [Progress("First."), Progress("Second.")], ["First."]),
            (0, [Progress("First.")], []),
        )
        for seconds, notes, written in cases:
            with self.subTest(seconds=seconds):
                gate = asyncio.Event()
                im = self.session(ScriptedAdapter([*notes, wait(gate), answer()]), progress_note_seconds=seconds)
                stream = im.admit(ask("Find teas."), delegation_id="D1").stream
                await asyncio.sleep(0.1)
                gate.set()
                written = [*(("commentary", note) for note in written), ("final_answer", "Done.")]
                self.assertEqual(said(await read_all(stream)), written)

    async def test_notes_stay_out_of_every_history_and_the_models_commentary_stays_in(self):
        gate = asyncio.Event()
        adapter = ScriptedAdapter([Progress("Opening the menu."), Say("Let me look."), OwnCall(), wait(gate), answer()])
        (stream,) = await self.admit_each(self.session(adapter), "Find teas.")
        gate.set()
        await read_all(stream)
        history = str(adapter.runs[0].histories[-1])
        self.assertNotIn("Opening the menu.", history)
        self.assertIn("Let me look.", history)

    async def test_the_work_tools_follow_the_result_format_and_the_work_tools_setting(self):
        cases = (
            (SPEAK_DISPLAY, ("cancel_work",), ["show", "cancel_work"]),
            (None, ("cancel_work",), ["cancel_work"]),
            (None, (), []),
            (SPEAK_DISPLAY, (), ["show"]),
        )
        for result_format, work_tools, offered in cases:
            with self.subTest(offered=offered):
                gate = asyncio.Event()
                adapter = ScriptedAdapter([wait(gate), answer()])
                await self.admit_each(self.session(adapter, result_format=result_format, work_tools=work_tools), "Hi.")
                tools = adapter.runs[0].task.tools()
                self.assertEqual([tool["name"] for tool in tools.tools()], offered)
                gate.set()
        schema, (tool,) = TEXTS.work_tools["show"], tools.tools()
        self.assertEqual((tool["description"], tool["parameters"]), (schema["description"], dict(schema["parameters"])))
        with self.assertRaises(ValueError):  # a show call without a display
            await tools.execute("c1", "show", "{}")
        unknown = TEXTS.unknown_tool_output.format(name="cancel_work", offered="show")  # the tools it offers
        self.assertEqual(await tools.execute("c2", "cancel_work", "{}"), json.dumps({"error": unknown}))

    async def test_each_round_and_every_snapshot_read_the_composed_instructions(self):
        notes = (TEXTS.caller_notes["default"].strip(), TEXTS.voice_notes["default"].strip())
        adapter = ScriptedAdapter([answer()], [answer()], conductor=StartEach())
        im = self.session(adapter, result_format=SPEAK_DISPLAY)
        brief = {"instructions": "Be brief."}
        events = await read_all(im.admit(ask("Find teas."), delegation_id="D1", settings=brief).stream)
        await read_all(im.admit(ask("Hours?"), delegation_id="D2").stream)
        composed = "\n".join(["Be brief.", *notes, TEXTS.result_formats["speak_display"]])
        self.assertEqual(adapter.runs[0].requests[0]["instructions"], composed)
        self.assertEqual({event["response"]["instructions"] for event in events if "response" in event}, {composed})
        default = "\n".join([DEFAULTS.instructions, *notes, TEXTS.result_formats["speak_display"]])
        self.assertEqual(adapter.runs[1].requests[0]["instructions"], default)
        adapter = ScriptedAdapter([answer()])
        no_notes = replace(TEXTS, caller_notes={"default": ""}, voice_notes={"default": " "})
        await read_all(self.session(adapter, texts=no_notes).admit(ask("Find teas."), delegation_id="D1").stream)
        self.assertEqual(adapter.runs[0].requests[0]["instructions"], DEFAULTS.instructions)  # byte for byte


class RateLimited(Exception):
    status_code = 429


class FailedWithCode(ScriptStep):
    """An attempt the backend ended failed with its own official code, then the run gives up."""

    async def play(self, run):
        error = {"error": {"code": "vector_store_timeout", "message": "The store timed out."}}
        for event in run.round_events([], terminal=("failed", error)):
            yield event
        raise RuntimeError("The backend gave up")


class Dies(ScriptStep):
    """A round whose backend call starts, then the run raises."""

    async def play(self, run):
        call = call_item("lookup", {}, "call_x")
        for event in run.round_events([call]):
            yield event
        run.task.route(call)
        run.task.tool_started(call, "cafe")
        raise RuntimeError("the tool's host died")


class EndingTests(IMCase):
    async def test_a_failure_ends_failed_with_the_official_code_or_incomplete_with_the_official_reason(self):
        cases = {
            "a raising run": ([Fail(RuntimeError("provider exploded"))], {"code": "server_error"}, "provider exploded"),
            "a 429": ([Fail(RateLimited("slow down"))], {"code": "rate_limit_exceeded"}, ""),
            "a 429 after a retry": ([FailedAttempt(), Fail(RateLimited())], {"code": "rate_limit_exceeded"}, ""),
            "the backend's own code": ([FailedWithCode()], {"code": "vector_store_timeout"}, ""),
            "an empty answer": ([OwnCall("book_table"), Empty()], {"code": "server_error"}, ""),
            "a cut by the token limit": ([Cut("max_output_tokens")], {"reason": "max_output_tokens"}, ""),
            "a cut by the content filter": ([Cut("content_filter")], {"reason": "content_filter"}, ""),
        }
        for name, (script, official, detail) in cases.items():
            with self.subTest(name):
                ended = (await read_all(self.session(ScriptedAdapter(script)).admit(ask("Find teas.")).stream))[-1]
                status = "failed" if "code" in official else "incomplete"
                field = ended["response"]["error"] if "code" in official else ended["response"]["incomplete_details"]
                self.assertEqual((ended["type"], ended["response"]["status"]), (f"response.{status}", status))
                self.assertEqual({key: field[key] for key in official}, official)
                self.assertIn(detail, field.get("message", ""))

    async def test_a_failure_closes_an_open_mcp_call_and_one_task_failing_fails_the_response_after_the_answers(self):
        ended = (await read_all(self.session(ScriptedAdapter([Dies()])).admit(ask("Teas?")).stream))[-1]
        (item,) = ended["response"]["output"]
        self.assertEqual((item["type"], item["status"]), ("mcp_call", "incomplete"))
        gate, split = asyncio.Event(), planned(placing(Start(("D1",), ("teas", "hours"))))
        im = self.session(ScriptedAdapter([answer("Teas.")], [wait(gate), Fail()], conductor=split, max_running=2))
        (stream,) = await self.admit_each(im, "Teas and hours.")
        gate.set()
        events = await read_all(stream)
        self.assertEqual((texts(events), events[-1]["type"]), (["Teas."], "response.failed"))

    async def test_at_close_a_queued_request_and_a_running_work_past_the_grace_end_cancelled(self):
        adapter = ScriptedAdapter([wait(asyncio.Event())], [answer()], conductor=StartEach())
        im = self.session(adapter, close_grace_seconds=0.02)
        streams = await self.admit_each(im, "Busy.", "Next.")
        await im.close()
        running, queued = [await read_all(stream) for stream in streams]
        self.assertEqual([e["type"] for e in queued], ["response.created", "response.queued", "response.incomplete"])
        for events in (running, queued):
            ended = events[-1]["response"]
            self.assertEqual((ended["status"], ended["incomplete_details"]), ("cancelled", None))
        self.assertEqual((len(adapter.runs), adapter.runs[0].stops, adapter.closed), (1, 1, True))

    async def test_running_work_may_answer_within_the_grace_or_with_no_grace(self):
        for grace in (1.0, None):
            with self.subTest(grace=grace):
                gate = asyncio.Event()
                adapter = ScriptedAdapter([wait(gate), answer("Made it.")])
                im = self.session(adapter, close_grace_seconds=grace)
                (stream,) = await self.admit_each(im, "Almost done.")
                closing = asyncio.create_task(im.close())
                await asyncio.sleep(0.05)
                self.assertFalse(closing.done())
                gate.set()
                await closing
                ended = (await read_all(stream))[-1]
                self.assertEqual((ended["type"], adapter.runs[0].stops), ("response.completed", 0))

    async def test_a_work_waiting_for_the_app_stops_at_once_and_an_arrival_being_placed_ends_cancelled(self):
        adapter = ScriptedAdapter([app_call("get_menu"), answer()])
        im = self.session(adapter, close_grace_seconds=60)
        await read_all(im.admit(ask("Menu?"), delegation_id="D1", settings={"tools": [MENU]}).stream)
        await asyncio.wait_for(im.close(), 5)
        self.assertEqual(adapter.runs[0].stops, 1)
        im = self.session(ScriptedAdapter(conductor=ScriptedConductor(lambda given, place: asyncio.Event().wait())))
        (stream,) = await self.admit_each(im, "Hello?")
        await im.close()
        ended = (await read_all(stream))[-1]
        self.assertEqual((ended["type"], ended["response"]["status"]), ("response.incomplete", "cancelled"))

    async def test_the_adapter_closes_even_when_a_stop_fails_and_close_is_safe_twice(self):
        class BrokenStop(ScriptedAdapter):
            def start(self, task):
                run = super().start(task)
                run.stop = mock.AsyncMock(side_effect=RuntimeError("stop failed"))
                return run

        adapter = BrokenStop([wait(asyncio.Event())])
        im = self.session(adapter, close_grace_seconds=0.01)
        (stream,) = await self.admit_each(im, "Long.")
        await im.close()
        await im.close()
        self.assertEqual((adapter.closed, (await read_all(stream))[-1]["type"]), (True, "response.incomplete"))


class HistoryTests(IMCase):
    async def test_a_request_carries_the_records_of_the_newest_ended_works_then_the_closing_line(self):
        count = CORE_SETTINGS.delegation.records_max_works + 2
        scripts = [[OwnCall("book_table"), answer("Booked.")], *([answer()] for _ in range(count - 1))]
        adapter = ScriptedAdapter(*scripts, conductor=StartEach())
        im = self.session(adapter)
        for n in range(count):
            await read_all(im.admit(ask(f"Question {n}."), delegation_id=f"D{n}").stream)
        first, second = (run.histories[0][-1]["content"] for run in adapter.runs[:2])
        self.assertEqual(first, [{"type": "input_text", "text": "Question 0."}])  # no records yet
        self.assertEqual((second[0]["text"], second[-1]["text"]), ("Question 1.", TEXTS.closing_line))
        record = {"work": "W1", "state": "answered", "requests": ["Question 0."], "steps": ["book_table: done"]}
        self.assertEqual(records_of(adapter.runs[1].histories[0][-1]), [record])
        works = [record["work"] for record in records_of(adapter.runs[-1].histories[0][-1])]
        self.assertEqual(works, [f"W{n}" for n in range(2, count)])

    async def test_each_work_reads_its_own_segment_and_an_ended_works_calls_with_their_outputs(self):
        gate = asyncio.Event()
        scripts = ([app_call("get_menu"), answer("Menu.")], [OwnCall("lookup"), wait(gate), answer()], [answer()])
        adapter = ScriptedAdapter(*scripts, conductor=StartEach(), max_running=2)
        im = self.session(adapter)
        await read_all(im.admit(ask("Menu?"), delegation_id="D1", settings={"tools": [MENU]}).stream)
        second = im.admit(ask("Hours?"), delegation_id="D2").stream
        await self.turns()
        gate.set()
        await read_all(second)
        await read_all(im.admit(ask("Third?"), delegation_id="D3").stream)
        own, later = adapter.runs[1].histories[-1], adapter.runs[2].histories[0]
        self.assertEqual([item["name"] for item in own if item.get("type") == "function_call"], ["lookup"])
        self.assertEqual(records_of(own[0])[0]["steps"], ["get_menu: started"])  # W1's record, not its call
        shared = [(item["type"], item["call_id"]) for item in later if "call_id" in item]
        self.assertEqual(shared, [("function_call", "call_W2.1_1"), ("function_call_output", "call_W2.1_1")])

    async def test_a_failed_works_calls_get_the_owed_output_in_the_next_works_history(self):
        adapter = ScriptedAdapter([app_call("get_menu")], [answer()], conductor=StartEach())
        im = self.session(adapter, client_tool_timeout_seconds=0.02)
        await read_all(im.admit(ask("Menu?"), delegation_id="D1", settings={"tools": [MENU]}).stream)
        await asyncio.sleep(0.05)
        await read_all(im.admit(ask("Again?"), delegation_id="D2").stream)
        owed = [item["output"] for item in adapter.runs[1].histories[0] if item.get("type") == "function_call_output"]
        self.assertEqual(owed, [TEXTS.owed_output])

    async def test_the_history_is_trimmed_to_its_budget_and_a_long_tool_output_is_cut(self):
        output = "H" * 60_000 + "T" * 40_000
        script = [OwnCall("book_table", output=output), answer("Booked.")]
        adapter = ScriptedAdapter(script, *([answer()] for _ in range(3)), conductor=StartEach())
        im = self.session(adapter, context_tokens=3000)
        await read_all(im.admit(ask("Book a table."), delegation_id="D1").stream)
        (kept,) = [item["output"] for item in adapter.runs[0].histories[-1] if item["type"] == "function_call_output"]
        half = CORE_SETTINGS.delegation.tool_output_chars // 2
        self.assertEqual(kept, output[:half] + TEXTS.cut_output.format(chars=len(output) - 2 * half) + output[-half:])
        for n in range(3):
            await read_all(im.admit(ask(f"Question {n}." + "x" * 3000), delegation_id=f"D{n + 2}").stream)
        last = adapter.runs[-1].histories[0]
        self.assertNotIn("book_table", str(last[:-1]))  # whole old requests left the history
        self.assertIn("book_table: done", str(records_of(last[-1])))  # their records did not
        im = self.session(ScriptedAdapter([answer()]), context_tokens=50)
        events = await read_all(im.admit(ask("x" * 2000), delegation_id="D1").stream)
        self.assertEqual(events[-1]["type"], "response.failed")  # the newest request alone is over the budget
