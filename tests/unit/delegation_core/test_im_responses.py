# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D103

"""An IM session's responses: admission, the items and notes it writes, its instructions, history, failures, close."""

import asyncio
import json
from unittest import mock

from examples.frontend_backend_live.delegation_core.backend.port import Start
from examples.frontend_backend_live.delegation_core.im.conversation import item_size
from examples.frontend_backend_live.delegation_core.im.results import SpeakDisplay
from examples.frontend_backend_live.delegation_core.im.session import DelegationError
from tests.unit.delegation_core.helpers import (
    CORE_SETTINGS,
    DEFAULTS,
    MENU,
    TEXTS,
    Attempt,
    Fail,
    IMCase,
    OwnCall,
    ScriptedAdapter,
    ScriptedConductor,
    ScriptStep,
    StartEach,
    answer,
    app_call,
    ask,
    call_item,
    placing,
    planned,
    progress,
    read_all,
    records_of,
    said,
    say,
    texts,
    wait,
)

SPEAK_DISPLAY = SpeakDisplay(TEXTS.result_formats["speak_display"])


def failed_attempt():
    """Return an attempt that streams a message and fails; the run goes on with its next step (a retry)."""
    error = {"error": {"code": "server_error", "message": "The attempt failed."}}
    return Attempt(("Half an answ",), ("failed", error), ends=False)


def cut(reason):
    """Return an attempt cut short (``response.incomplete`` with ``reason``), after which the run raises."""
    details = {"incomplete_details": {"reason": reason}}
    return Attempt(("Half an answ",), ("incomplete", details), RuntimeError(f"The round was cut: {reason}"))


class ResponseTests(IMCase):
    async def test_a_response_reports_its_items_and_the_newest_usage(self):
        usage = {"input_tokens": 12, "output_tokens": 3, "total_tokens": 15}
        steps = [say("Looking."), OwnCall(), say("Almost."), OwnCall(), answer(usage=usage)]
        events = await read_all(self.session(ScriptedAdapter(steps)).admit(ask("Find teas.")).stream)
        indexes = [e["output_index"] for e in events if e["type"] == "response.output_item.done"]
        self.assertEqual(indexes, list(range(len(indexes))))
        self.assertEqual(events[-1]["response"]["usage"], usage)

    async def test_admission_refuses_a_full_queue_a_used_id_and_a_close(self):
        im = self.session(ScriptedAdapter([wait(asyncio.Event())], conductor=StartEach()))
        # One runs and sixteen wait.
        await self.admit_each(im, *(f"Request {n}." for n in range(1, 18)))
        with self.assertRaisesRegex(DelegationError, "full"):
            im.admit(ask("One too many."), delegation_id="D18")
        with self.assertRaisesRegex(DelegationError, "already exists"):
            im.admit(ask("Again."), delegation_id="D1")
        await im.close()
        with self.assertRaisesRegex(DelegationError, "closing"):
            im.admit(ask("Late."), delegation_id="D19")

    async def test_a_reader_that_stops_leaves_its_response_running(self):
        gate = asyncio.Event()
        adapter = ScriptedAdapter([wait(gate), OwnCall("book_table"), answer()], [answer()], conductor=StartEach())
        im = self.session(adapter)
        await im.admit(ask("Book it."), delegation_id="D1").stream.aclose()
        gate.set()
        await read_all(im.admit(ask("Booked?"), delegation_id="D2").stream)
        (record,) = records_of(adapter.runs[1].histories[0][-1])
        self.assertEqual((record["state"], record["steps"]), ("answered", ["book_table: done"]))

    async def test_the_last_answer_counts_and_a_failed_attempt_is_unsaid(self):
        im = self.session(
            ScriptedAdapter([Attempt(("First.", "Last."))], [failed_attempt(), answer()], conductor=StartEach())
        )
        two = await read_all(im.admit(ask("Find teas."), delegation_id="D1").stream)
        retried = await read_all(im.admit(ask("Find coffee."), delegation_id="D2").stream)
        self.assertEqual(said(two), [("commentary", "First."), ("final_answer", "Last.")])
        self.assertEqual((said(retried), retried[-1]["type"]), ([("final_answer", "Done.")], "response.completed"))
        self.assertNotIn("Half an answ", str(retried))

    async def test_a_backend_call_is_an_mcp_call_with_official_events(self):
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

    async def test_progress_notes_come_one_per_interval(self):
        cases = (
            (0.05, [progress("First."), progress("Second."), progress("Third.")], ["First.", "Third."]),
            (5, [progress("First."), progress("Second.")], ["First."]),
            (0, [progress("First.")], []),
        )
        for seconds, notes, written in cases:
            with self.subTest(seconds=seconds):
                said_by_then = await self.said_with_notes(seconds, notes, due=len(written))
                self.assertEqual(said_by_then, [*(("commentary", note) for note in written), ("final_answer", "Done.")])

    async def said_with_notes(self, seconds: float, notes: list[ScriptStep], due: int) -> list[tuple[str, str]]:
        """Return what a response says when its run gives ``notes`` at once and answers once ``due`` are written."""
        gate = asyncio.Event()
        adapter = ScriptedAdapter([*notes, wait(gate), answer()])
        im = self.session(adapter, progress_note_seconds=seconds)
        wired = len(self.wire.events)
        stream = im.admit(ask("Find teas."), delegation_id="D1").stream
        # The answer waits for the run at its gate, with only the answer left, and for the notes due by then: a slow
        # loop makes it wait longer, never cuts an interval short.
        await self.until(
            lambda: (
                bool(adapter.runs)
                and len(adapter.runs[0].steps) == 1
                and len(said([event for _, event in self.wire.events[wired:]])) >= due
            ),
            "the run at its gate with its notes written",
        )
        gate.set()
        return said(await read_all(stream))

    async def test_notes_stay_out_of_the_history_and_commentary_stays(self):
        gate = asyncio.Event()
        adapter = ScriptedAdapter([progress("Opening the menu."), say("Let me look."), OwnCall(), wait(gate), answer()])
        (stream,) = await self.admit_each(self.session(adapter), "Find teas.")
        gate.set()
        await read_all(stream)
        history = str(adapter.runs[0].histories[-1])
        self.assertNotIn("Opening the menu.", history)
        self.assertIn("Let me look.", history)

    async def test_the_work_tools_follow_the_result_format(self):
        plain, envelope = ["cancel_work", "ask_caller"], ["show", "cancel_work", "ask_caller"]
        for result_format, offered in ((None, plain), (SPEAK_DISPLAY, envelope)):
            with self.subTest(offered=offered):
                gate = asyncio.Event()
                adapter = ScriptedAdapter([wait(gate), answer()])
                await self.admit_each(self.session(adapter, result_format=result_format), "Hi.")
                tools = adapter.runs[0].task.tools()
                self.assertEqual([tool["name"] for tool in tools.tools()], offered)
                gate.set()
        schema, (tool, *_) = TEXTS.work_tools["show"], tools.tools()
        self.assertEqual((tool["description"], tool["parameters"]), (schema["description"], dict(schema["parameters"])))
        with self.assertRaises(ValueError):
            await tools.execute("c1", "show", "{}")
        unknown = TEXTS.unknown_tool_output.format(name="book_table", offered="show, cancel_work, ask_caller")
        self.assertEqual(await tools.execute("c2", "book_table", "{}"), json.dumps({"error": unknown}))

    async def test_rounds_and_snapshots_read_the_composed_instructions(self):
        notes = (TEXTS.caller_note.strip(), TEXTS.voice_note.strip())
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
        no_notes = TEXTS.model_copy(update={"caller_note": "", "voice_note": " "})
        await read_all(self.session(adapter, texts=no_notes).admit(ask("Find teas."), delegation_id="D1").stream)
        self.assertEqual(adapter.runs[0].requests[0]["instructions"], DEFAULTS.instructions)

    async def test_a_request_carries_the_newest_ended_works_records(self):
        count = CORE_SETTINGS.delegation.records_max_works + 2
        scripts = [[OwnCall("book_table"), answer("Booked.")], *([answer()] for _ in range(count - 1))]
        adapter = ScriptedAdapter(*scripts, conductor=StartEach())
        im = self.session(adapter)
        for n in range(count):
            await read_all(im.admit(ask(f"Question {n}."), delegation_id=f"D{n}").stream)
        first, second = (run.histories[0][-1]["content"] for run in adapter.runs[:2])
        self.assertEqual(first, [{"type": "input_text", "text": "Question 0."}])
        self.assertEqual((second[0]["text"], second[-1]["text"]), ("Question 1.", TEXTS.closing_line))
        record = {"work": "W1", "state": "answered", "requests": ["Question 0."], "steps": ["book_table: done"]}
        self.assertEqual(records_of(adapter.runs[1].histories[0][-1]), [record])
        works = [record["work"] for record in records_of(adapter.runs[-1].histories[0][-1])]
        self.assertEqual(works, [f"W{n}" for n in range(2, count)])

    async def test_each_work_reads_its_own_segment_of_the_history(self):
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
        self.assertEqual(records_of(own[0])[0]["steps"], ["get_menu: started"])
        shared = [(item["type"], item["call_id"]) for item in later if "call_id" in item]
        self.assertEqual(shared, [("function_call", "call_W2.1_1"), ("function_call_output", "call_W2.1_1")])

    async def test_the_history_is_trimmed_and_a_long_output_is_cut(self):
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
        self.assertNotIn("book_table", str(last[:-1]))
        self.assertIn("book_table: done", str(records_of(last[-1])))
        im = self.session(ScriptedAdapter([answer()]), context_tokens=50)
        events = await read_all(im.admit(ask("x" * 2000), delegation_id="D1").stream)
        self.assertEqual(events[-1]["type"], "response.failed")

    def test_the_budget_counts_the_id_of_no_input_message(self):
        said = {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "Teas."}]}
        self.assertEqual(item_size({**said, "id": "D1.2"}), item_size(said))
        # An output message counts its id.
        output = {**said, "status": "completed"}
        self.assertGreater(item_size({**output, "id": "msg_1"}), item_size(output))


class RateLimited(Exception):
    status_code = 429


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
    async def test_a_failure_ends_with_its_official_code_or_reason(self):
        own = ("failed", {"error": {"code": "vector_store_timeout", "message": "The store timed out."}})
        cases = {
            "a raising run": ([Fail(RuntimeError("provider exploded"))], {"code": "server_error"}, "provider exploded"),
            "a 429": ([Fail(RateLimited("slow down"))], {"code": "rate_limit_exceeded"}, ""),
            "a 429 after a retry": ([failed_attempt(), Fail(RateLimited())], {"code": "rate_limit_exceeded"}, ""),
            "the backend's own code": (
                [Attempt((), own, RuntimeError("The backend gave up"))],
                {"code": "vector_store_timeout"},
                "",
            ),
            "an empty answer": ([OwnCall("book_table"), Attempt()], {"code": "server_error"}, ""),
            "a cut by the token limit": ([cut("max_output_tokens")], {"reason": "max_output_tokens"}, ""),
            "a cut by the content filter": ([cut("content_filter")], {"reason": "content_filter"}, ""),
        }
        for name, (script, official, detail) in cases.items():
            with self.subTest(name):
                ended = (await read_all(self.session(ScriptedAdapter(script)).admit(ask("Find teas.")).stream))[-1]
                status = "failed" if "code" in official else "incomplete"
                field = ended["response"]["error"] if "code" in official else ended["response"]["incomplete_details"]
                self.assertEqual((ended["type"], ended["response"]["status"]), (f"response.{status}", status))
                self.assertEqual({key: field[key] for key in official}, official)
                self.assertIn(detail, field.get("message", ""))

    async def test_a_failure_closes_an_open_mcp_call_and_its_response(self):
        ended = (await read_all(self.session(ScriptedAdapter([Dies()])).admit(ask("Teas?")).stream))[-1]
        (item,) = ended["response"]["output"]
        self.assertEqual((item["type"], item["status"]), ("mcp_call", "incomplete"))
        gate, split = asyncio.Event(), planned(placing(Start(("D1",), ("teas", "hours"))))
        im = self.session(ScriptedAdapter([answer("Teas.")], [wait(gate), Fail()], conductor=split, max_running=2))
        (stream,) = await self.admit_each(im, "Teas and hours.")
        gate.set()
        events = await read_all(stream)
        self.assertEqual((texts(events), events[-1]["type"]), (["Teas."], "response.failed"))

    async def test_at_close_queued_and_running_work_end_cancelled(self):
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

    async def test_at_close_waiting_work_stops_and_placing_ends(self):
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

    async def test_the_adapter_closes_when_a_stop_fails_and_twice(self):
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
