# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D103

"""The conversation: requests, per-task histories and compaction."""

import asyncio
import json
import unittest
from unittest import mock

from delegation_core.im import conversation
from delegation_core.im.conversation import Conversation, TaskHistory
from tests.unit.contracts.backend import a_round, call_events, text_round
from tests.unit.delegation_core.backend.llm.doubles import RecordingTools, ScriptedProvider
from tests.unit.delegation_core.backend.llm.doubles import source as llm_source
from tests.unit.delegation_core.helpers import (
    CORE_SETTINGS,
    MENU,
    TEXTS,
    IMCase,
    ScriptedAdapter,
    StartEach,
    answer,
    app_call,
    ask,
    own_call,
    read_all,
    wait,
)

CHARS = CORE_SETTINGS.delegation.tool_output_chars


def request(text: str, size: int = 0) -> list[dict]:
    return ask(text + "x" * size)


class TrimTests(unittest.TestCase):
    def test_trims_drop_whole_old_requests_and_keep_the_newest(self):
        view = TaskHistory([], 200, CHARS, TEXTS.cut_output)
        view.join([*request("old", 1000), {"type": "message", "role": "assistant", "content": "old answer"}])
        view.join(request("new"))
        history = view.history()
        self.assertEqual(len(history), 1)
        self.assertTrue(history[0]["content"][0]["text"].startswith("new"))
        self.assertEqual(view.since(0)[0]["content"][0]["text"][:3], "old")  # still kept for a cursor

    def test_the_newest_request_alone_over_the_budget_fails(self):
        view = TaskHistory(request("huge", 2000), 100, CHARS, TEXTS.cut_output)
        with self.assertRaisesRegex(RuntimeError, "budget"):
            view.history()

    def test_a_long_tool_output_is_cut_to_its_head_and_tail(self):
        output = "H" * 60_000 + "T" * 40_000
        view = TaskHistory([], 100_000, CHARS, TEXTS.cut_output)
        view.join([{"type": "function_call_output", "call_id": "c1", "output": output}])
        kept = view.history()[0]["output"]
        self.assertLess(len(kept), CHARS + 100)
        self.assertTrue(kept.startswith("H" * 100) and kept.endswith("T" * 100))
        half = CHARS // 2
        self.assertEqual(kept, output[:half] + TEXTS.cut_output.format(chars=len(output) - 2 * half) + output[-half:])

    def test_each_items_size_is_estimated_once(self):
        with mock.patch.object(conversation, "item_size", wraps=conversation.item_size) as size:
            view = TaskHistory(request("one"), 10_000, CHARS, TEXTS.cut_output)
            view.join(request("two"))
            for _ in range(5):
                view.history()
        self.assertEqual(size.call_count, 2)

    def test_the_shared_conversation_forgets_what_left_every_window_and_keeps_positions(self):
        shared = Conversation(100, CHARS, TEXTS.cut_output)
        shared.join(request("old", 500))
        shared.join(request("new"))
        start, items = shared.window()
        self.assertEqual((start, len(items), shared.end), (1, 1, 2))

    def test_owed_outputs_answer_every_call_left_without_one_and_a_reported_output_its_call_in_call_order(self):
        calls = [
            {"type": "function_call", "call_id": "a", "name": "x", "arguments": "{}"},
            {"type": "function_call", "call_id": "b", "name": "y", "arguments": "{}"},
        ]
        answered = [*calls, {"type": "function_call_output", "call_id": "a", "output": "{}"}]
        owed = conversation.owed_outputs(answered, TEXTS.owed_output)
        self.assertEqual(owed, [{"type": "function_call_output", "call_id": "b", "output": TEXTS.owed_output}])
        reported = {
            "b": {"type": "function_call_output", "call_id": "b", "output": '{"booked": true}'},
            "z": {"type": "function_call_output", "call_id": "z", "output": "{}"},  # no call of its own: left out
        }
        owed = conversation.owed_outputs(calls, TEXTS.owed_output, reported)
        self.assertEqual(
            owed, [{"type": "function_call_output", "call_id": "a", "output": TEXTS.owed_output}, reported["b"]]
        )


class SegmentTests(IMCase):
    async def test_two_works_keep_separate_segments(self):
        gate = asyncio.Event()
        adapter = ScriptedAdapter(
            [app_call("get_menu"), answer("Menu.")],
            [own_call("lookup"), wait(gate), answer("Hours.")],
            [answer("Third.")],
            conductor=StartEach(),
            max_running=2,
        )
        im = self.session(adapter)
        await read_all(im.admit(ask("Menu?"), delegation_id="D1", settings={"tools": [MENU]}).stream)
        second = im.admit(ask("Hours?"), delegation_id="D2").stream
        await self.turns()
        gate.set()
        await read_all(second)
        second_view = adapter.runs[1].histories[-1]
        calls = [item["name"] for item in second_view if item.get("type") == "function_call"]
        self.assertEqual(calls, ["lookup"])  # W1's unanswered get_menu call is not W2's
        self.assertIn('"steps": ["get_menu: started"]', second_view[0]["content"][1]["text"])  # only its record
        await read_all(im.admit(ask("Third?"), delegation_id="D3").stream)
        third_view = adapter.runs[2].histories[0]
        calls = [item["name"] for item in third_view if item.get("type") == "function_call"]
        call_ids = [item["call_id"] for item in third_view if item.get("type") == "function_call"]
        outputs = [item["call_id"] for item in third_view if item.get("type") == "function_call_output"]
        self.assertEqual(calls, ["lookup"])  # W2 ended: its segment is shared; W1 still waits for the app
        self.assertEqual(call_ids, outputs)  # every shared call has its output

    async def test_a_failed_works_calls_get_owed_outputs_in_the_shared_conversation(self):
        adapter = ScriptedAdapter([app_call("get_menu")], [answer()], conductor=StartEach())
        im = self.session(adapter, client_tool_timeout_seconds=0.02)
        await read_all(im.admit(ask("Menu?"), delegation_id="D1", settings={"tools": [MENU]}).stream)
        await asyncio.sleep(0.05)
        await read_all(im.admit(ask("Again?"), delegation_id="D2").stream)
        view = adapter.runs[1].histories[0]
        owed = [item for item in view if item.get("type") == "function_call_output"]
        self.assertEqual([item["output"] for item in owed], [TEXTS.owed_output])

    async def test_a_call_that_ended_while_its_task_stopped_keeps_its_output_and_a_call_that_never_ran_is_owed(self):
        # The backstop stops W1 while book_table runs. The stop lets the call finish, so its output is known; the app's
        # get_menu of the same round never got an output, so it gets texts.owed_output.
        tools = RecordingTools(["book_table"], delay=0.4)
        both = a_round(
            call_events("book_table", {}, call_id="call_book", index=0),
            call_events("get_menu", {}, call_id="call_menu", index=1),
        )
        provider = ScriptedProvider(both, text_round("It was booked."))
        im = self.session(llm_source(provider, tools=[tools]).new(), work_timeout_seconds=0.2)
        menu = {"tools": [MENU]}
        await read_all(im.admit(ask("Book a table and show the menu."), delegation_id="D1", settings=menu).stream)
        await read_all(im.admit(ask("Did it book?"), delegation_id="D2").stream)
        later = provider.payloads[-1]["input"]
        outputs = {item["call_id"]: item["output"] for item in later if item.get("type") == "function_call_output"}
        booked = json.dumps({"ok": True, "tool": "book_table"})
        self.assertEqual(outputs, {"call_book": booked, "call_menu": TEXTS.owed_output})
        actions = im.ledger.actions
        self.assertEqual((actions["call_book"].state, actions["call_menu"].state), ("done", "unclear"))

    async def test_a_bookings_record_survives_a_trim(self):
        script = [own_call("book_table", output=json.dumps({"booked": True})), answer("Booked.")]
        later = [[answer(f"Answer {n}.")] for n in range(3)]
        adapter = ScriptedAdapter(script, *later, conductor=StartEach())
        im = self.session(adapter, context_tokens=900)
        await read_all(im.admit(request("Book a table.", 1000), delegation_id="D1").stream)
        for n in range(3):
            await read_all(im.admit(request(f"Question {n}.", 1000), delegation_id=f"D{n + 2}").stream)
        last = adapter.runs[-1].histories[0]
        self.assertNotIn("book_table", str(last[:-1]))  # the booking's items were trimmed
        records = last[-1]["content"][1]["text"]
        self.assertIn('"steps": ["book_table: done"]', records)  # its record was not
