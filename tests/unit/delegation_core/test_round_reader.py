# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D103

"""The round reader and tools on the wire."""

import asyncio

from delegation_core.im.results import CallNotes, SpeakDisplay
from delegation_core.im.stream import message_text
from tests.unit.delegation_core.helpers import (
    MENU,
    IMCase,
    ScriptedAdapter,
    ScriptStep,
    answer,
    app_call,
    ask,
    failed_attempt,
    finished_items,
    message_item,
    own_call,
    progress,
    read_all,
    say,
    show,
    wait,
)


class TwoAnswers(ScriptStep):
    """A round with two final_answer messages."""

    ends = True

    async def play(self, run):
        first, last = message_item("First.", "final_answer", "m1"), message_item("Last.", "final_answer", "m2")
        for event in run.round_events([first, last]):
            yield event


def texts(events: list[dict]) -> list[tuple[str, str]]:
    return [(item["phase"], item["content"][0]["text"]) for item in finished_items(events) if item["type"] == "message"]


class RoundReaderTests(IMCase):
    async def test_a_failed_attempt_writes_nothing(self):
        im = self.session(ScriptedAdapter([failed_attempt(), answer("Done.")]))
        events = await read_all(im.admit(ask("Find teas."), delegation_id="D1").stream)
        self.assertEqual(texts(events), [("final_answer", "Done.")])
        self.assertNotIn("Half an answ", str(events))
        self.assertEqual(events[-1]["type"], "response.completed")

    async def test_messages_carry_their_phase_and_the_last_final_answer_is_the_answer(self):
        im = self.session(ScriptedAdapter([TwoAnswers()]))
        events = await read_all(im.admit(ask("Find teas."), delegation_id="D1").stream)
        self.assertEqual(texts(events), [("commentary", "First."), ("final_answer", "Last.")])
        result = im.ledger.tasks["W1.1"].result  # the task's own result: its last final answer
        self.assertEqual((result.state, message_text(result.items[-1])), ("pushed", "Last."))

    async def test_the_apps_calls_and_the_terminal_are_written_in_one_step(self):
        im = self.session(ScriptedAdapter([say("Checking."), app_call("get_menu"), answer()]))
        stream = im.admit(ask("Find teas."), delegation_id="D1", settings={"tools": [MENU]}).stream
        async for event in stream:
            if event["type"] == "response.output_item.done" and event["item"]["type"] == "function_call":
                break
        self.assertEqual(stream.unread, 1)  # the terminal is already there: no await came between them
        terminal = await anext(stream)
        self.assertEqual(terminal["type"], "response.completed")
        self.assertEqual([item["type"] for item in terminal["response"]["output"]], ["message", "function_call"])

    async def test_a_backend_call_is_an_mcp_call_with_its_official_events(self):
        im = self.session(ScriptedAdapter([own_call("lookup", output='{"teas": 2}', label="cafe"), answer()]))
        events = await read_all(im.admit(ask("Find teas."), delegation_id="D1").stream)
        mcp = [e for e in events if "mcp" in e["type"] or e.get("item", {}).get("type") == "mcp_call"]
        self.assertEqual(
            [e["type"] for e in mcp],
            [
                "response.output_item.added",
                "response.mcp_call.in_progress",
                "response.mcp_call.completed",
                "response.output_item.done",
            ],
        )
        item = mcp[-1]["item"]
        self.assertEqual((item["server_label"], item["name"], item["status"]), ("cafe", "lookup", "completed"))
        self.assertEqual((item["output"], item["error"]), ('{"teas": 2}', None))
        self.assertEqual(len({e["item_id"] for e in mcp if "item_id" in e} | {item["id"]}), 1)

    async def test_a_failed_backend_call_carries_the_official_error_object(self):
        im = self.session(ScriptedAdapter([own_call("lookup", output="timed out", fails=True), answer()]))
        events = await read_all(im.admit(ask("Find teas."), delegation_id="D1").stream)
        self.assertIn("response.mcp_call.failed", [e["type"] for e in events])
        item = next(item for item in finished_items(events) if item["type"] == "mcp_call")
        self.assertEqual(item["status"], "failed")
        self.assertEqual(item["error"], {"type": "mcp_tool_execution_error", "content": "timed out"})
        self.assertEqual(im.ledger.actions["call_W1.1_1"].state, "failed")

    async def test_notes_stay_out_of_every_history_and_the_models_commentary_stays_in(self):
        gate = asyncio.Event()
        adapter = ScriptedAdapter(
            [progress("Opening the menu."), say("Let me look."), own_call(), wait(gate), answer()]
        )
        im = self.session(adapter)
        stream = im.admit(ask("Find teas."), delegation_id="D1").stream
        await self.turns()
        gate.set()
        await read_all(stream)
        last_history = str(adapter.runs[0].histories[-1])
        self.assertNotIn("Opening the menu.", last_history)
        self.assertIn("Let me look.", last_history)

    async def test_backend_calls_as_notes_show_only_a_note_and_the_display_still_shows(self):
        adapter = ScriptedAdapter([own_call("lookup", label="cafe"), show("Menu: tea"), answer("Tea it is.")])
        im = self.session(adapter, call_reports=CallNotes(), result_format=SpeakDisplay(""), progress_note_seconds=0.01)
        events = await read_all(im.admit(ask("Find teas."), delegation_id="D1").stream)
        kinds = [(item["type"], item.get("phase") or item.get("name")) for item in finished_items(events)]
        self.assertEqual(kinds, [("message", "commentary"), ("mcp_call", "show"), ("message", "final_answer")])
        self.assertEqual(texts(events)[0], ("commentary", "lookup: done"))
        self.assertNotIn("response.mcp_call.completed", [e["type"] for e in events][:5])

    async def test_the_terminal_reports_the_newest_rounds_usage(self):
        usage = {"input_tokens": 12, "output_tokens": 3, "total_tokens": 15}
        im = self.session(ScriptedAdapter([answer("Done.", usage=usage)]))
        events = await read_all(im.admit(ask("Find teas."), delegation_id="D1").stream)
        self.assertEqual(events[-1]["response"]["usage"], usage)

    async def test_each_item_has_its_own_output_index(self):
        im = self.session(ScriptedAdapter([say("Looking."), own_call(), say("Almost."), own_call(), answer()]))
        events = await read_all(
            im.admit(ask("Find teas."), delegation_id="D1").stream,
        )
        indexes = [e["output_index"] for e in events if e["type"] == "response.output_item.done"]
        self.assertEqual(indexes, sorted(set(indexes)))
