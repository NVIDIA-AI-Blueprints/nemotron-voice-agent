# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D103

"""One-shot sessions, the IM side."""

import unittest

from delegation_core.im.session import DelegationError
from tests.unit.delegation_core.helpers import (
    MENU,
    TEXTS,
    ScriptedAdapter,
    answer,
    app_call,
    ask,
    fail,
    finished_items,
    one_shot_session,
    progress,
    read_all,
    say,
)

WHOLE_CONTEXT = [
    *ask("Two lattes."),
    {"type": "function_call", "call_id": "c0", "name": "get_menu", "arguments": "{}"},
    {"type": "function_call_output", "call_id": "c0", "output": "{}"},
    *ask("Make it oat milk." + "x" * 200_000),
]


class OneShotTests(unittest.IsolatedAsyncioTestCase):
    async def serve(self, *steps, items=None, settings=None) -> tuple:
        adapter = ScriptedAdapter(list(steps))
        im = one_shot_session(adapter)
        im.start()
        self.addAsyncCleanup(im.close)
        events = await read_all(im.admit(items or ask("Two lattes."), delegation_id="D1", settings=settings).stream)
        return im, adapter, events

    async def test_the_input_is_used_as_sent_never_trimmed_and_given_no_records(self):
        _, adapter, events = await self.serve(answer(), items=WHOLE_CONTEXT)
        self.assertEqual(adapter.runs[0].histories[0], WHOLE_CONTEXT)
        self.assertEqual(events[-1]["type"], "response.completed")

    async def test_the_voice_note_reaches_the_instructions_and_nothing_else_does(self):
        _, adapter, _ = await self.serve(answer(), settings={"instructions": "Be brief."})
        self.assertEqual(adapter.runs[0].requests[0]["instructions"], "Be brief.\n" + TEXTS.voice_notes["v2"].strip())

    async def test_the_apps_calls_end_the_response_and_nothing_waits(self):
        im, adapter, events = await self.serve(app_call("get_menu"), settings={"tools": [MENU]})
        self.assertEqual([item["type"] for item in events[-1]["response"]["output"]], ["function_call"])
        self.assertEqual(im.ledger.delegations["D1"].state, "answered")
        await im.close()
        self.assertTrue(adapter.closed)
        self.assertEqual(adapter.runs[0].resumes, 0)

    async def test_the_rounds_messages_are_written_as_they_are_and_no_note_is_written(self):
        _, _, events = await self.serve(progress("Working."), say("Checking the menu."), app_call("get_menu"))
        messages = [(item["phase"], item["content"][0]["text"]) for item in finished_items(events) if "phase" in item]
        self.assertEqual(messages, [("commentary", "Checking the menu.")])

    async def test_it_opens_in_progress_and_a_failure_ends_failed(self):
        _, _, events = await self.serve(fail())
        self.assertEqual([e["type"] for e in events], ["response.created", "response.in_progress", "response.failed"])
        self.assertEqual(events[0]["response"]["status"], "in_progress")

    async def test_it_serves_one_request_and_ends_itself(self):
        im, adapter, _ = await self.serve(answer())
        with self.assertRaisesRegex(DelegationError, "exactly one request"):
            im.admit(ask("Again."))
        with self.assertRaisesRegex(DelegationError, "exactly one request"):
            im.admit_client_request(ask("Typed."))
        await im._closing
        self.assertTrue(adapter.closed)
        self.assertEqual(adapter.conductors, 0)  # a one-shot adapter needs no conductor
