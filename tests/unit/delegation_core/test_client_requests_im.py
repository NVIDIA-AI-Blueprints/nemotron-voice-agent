# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D103

"""The client's continuations and typed requests, the IM side."""

import asyncio

from delegation_core.im.session import FUNCTION_CALL_OUTPUTS_REQUIRED, DelegationError
from delegation_core.im.stop import AtDeadline
from tests.unit.delegation_core.helpers import (
    MENU,
    IMCase,
    ScriptedAdapter,
    ScriptStep,
    StartEach,
    answer,
    app_call,
    app_outputs,
    ask,
    call_item,
    finished_items,
    read_all,
)

MENU_SETTINGS = {"tools": [MENU]}
# A late continuation's response: it opens, goes in progress like every response, and ends empty.
LATE = ["response.created", "response.queued", "response.in_progress", "response.completed"]


class TwoCalls(ScriptStep):
    """A round that calls the app's function twice; the task's events end until resumed."""

    ends = True

    async def play(self, run):
        calls = [call_item("get_menu", {"category": c}, f"call_{c}") for c in ("tea", "coffee")]
        for event in run.round_events(calls):
            yield event


class ContinuationTests(IMCase):
    async def waiting(self, *scripts, **settings) -> tuple:
        """A session whose first request waits for the app; ``settings`` change the IM's settings."""
        im = self.session(ScriptedAdapter(*scripts), **settings)
        first = await read_all(im.admit(ask("Menu?"), delegation_id="D1", settings=MENU_SETTINGS).stream)
        return im, first

    async def test_missing_outputs_are_refused_with_the_official_code(self):
        im, first = await self.waiting([TwoCalls(), answer()])
        with self.assertRaises(DelegationError) as refused:
            im.admit_client_request(app_outputs(first)[:1], delegation_id="D1", settings=MENU_SETTINGS)
        self.assertEqual(refused.exception.code, FUNCTION_CALL_OUTPUTS_REQUIRED)
        self.assertEqual(len(im.ledger.delegations["D1"].responses), 1)  # nothing changed
        events = await read_all(im.admit_client_request(app_outputs(first), delegation_id="D1").stream)
        self.assertEqual(events[-1]["type"], "response.completed")

    async def test_duplicates_other_result_types_and_two_responses_at_once_are_refused(self):
        im, first = await self.waiting([app_call("get_menu"), answer()])
        (output,) = app_outputs(first)
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            im.admit_client_request([output, output], delegation_id="D1")
        with self.assertRaisesRegex(ValueError, "function_call_output"):
            im.admit_client_request([{**output, "type": "custom_tool_call_output"}], delegation_id="D1")
        with self.assertRaisesRegex(ValueError, "function_call_output"):
            im.admit_client_request([{"type": "function_call_output", "output": "{}"}], delegation_id="D1")

    async def test_results_for_two_waiting_responses_at_once_are_refused(self):
        adapter = ScriptedAdapter(
            [app_call("get_menu"), answer()], [app_call("get_menu"), answer()], conductor=StartEach(), max_running=2
        )
        im = self.session(adapter)
        first = await read_all(im.admit(ask("Teas?"), delegation_id="D1", settings=MENU_SETTINGS).stream)
        second = await read_all(im.admit(ask("Coffee?"), delegation_id="D2", settings=MENU_SETTINGS).stream)
        with self.assertRaisesRegex(ValueError, "more than one response"):
            im.admit_client_request(app_outputs(first) + app_outputs(second))

    async def test_two_waiting_responses_are_continued_in_either_order(self):
        adapter = ScriptedAdapter(
            [app_call("get_menu"), answer("Teas.")],
            [app_call("get_menu"), answer("Coffee.")],
            conductor=StartEach(),
            max_running=2,
        )
        im = self.session(adapter)
        first = await read_all(im.admit(ask("Teas?"), delegation_id="D1", settings=MENU_SETTINGS).stream)
        second = await read_all(im.admit(ask("Coffee?"), delegation_id="D2", settings=MENU_SETTINGS).stream)
        later = await read_all(im.admit_client_request(app_outputs(second), delegation_id="D2").stream)
        sooner = await read_all(im.admit_client_request(app_outputs(first), delegation_id="D1").stream)
        self.assertEqual([item["content"][0]["text"] for item in finished_items(later)], ["Coffee."])
        self.assertEqual([item["content"][0]["text"] for item in finished_items(sooner)], ["Teas."])

    async def test_typed_messages_sent_with_the_outputs_join_after_them(self):
        im, first = await self.waiting([app_call("get_menu"), answer()])
        typed = {"type": "message", "role": "user", "content": "Only green ones."}
        await read_all(im.admit_client_request([*app_outputs(first), typed], delegation_id="D1").stream)
        history = im.backend.runs[0].histories[-1]
        self.assertEqual([item.get("type") for item in history[-2:]], ["function_call_output", "message"])

    async def test_the_deadline_counts_from_when_the_calls_reached_the_app(self):
        im, first = await self.waiting([app_call("get_menu"), answer()], client_tool_timeout_seconds=0.2)
        await asyncio.sleep(0.1)
        events = await read_all(im.admit_client_request(app_outputs(first), delegation_id="D1").stream)
        self.assertIn("response.in_progress", [e["type"] for e in events])

    async def test_outputs_for_an_ended_work_get_an_empty_completed_and_typed_text_still_runs(self):
        im, first = await self.waiting([app_call("get_menu"), answer()], [answer("Green teas.")])
        await im.works["W1"].stop(AtDeadline())  # the client-tool deadline ends the Work
        late = await read_all(im.admit_client_request(app_outputs(first, "late"), delegation_id="D1").stream)
        self.assertEqual(
            [e["type"] for e in late],
            LATE,
        )
        call_id = app_outputs(first)[0]["call_id"]
        self.assertEqual(im.ledger.actions[call_id].output, "late")
        typed = {"type": "message", "role": "user", "content": "Just the green teas then."}
        ran = await read_all(im.admit_client_request([*app_outputs(first), typed], delegation_id="D1").stream)
        self.assertEqual([item["content"][0]["text"] for item in finished_items(ran)], ["Green teas."])


class TypedRunTests(IMCase):
    async def test_a_typed_request_with_no_items_is_an_arrival(self):
        adapter = ScriptedAdapter([answer("Continuing.")])
        im = self.session(adapter)
        events = await read_all(im.admit_client_request([], delegation_id="T1").stream)
        self.assertEqual([item["content"][0]["text"] for item in finished_items(events)], ["Continuing."])
        self.assertEqual(im.ledger.delegations["T1"].state, "answered")
