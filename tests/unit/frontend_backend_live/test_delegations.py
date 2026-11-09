# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D103, D107

"""The delegation reader and its work note: every ending read from official fields, by structure."""

import asyncio
import unittest
from collections import Counter
from functools import partial

from examples.frontend_backend_live.delegation_core.items import message_text
from examples.frontend_backend_live.delegations import DelegationReader
from examples.frontend_backend_live.prompts import load_prompts
from examples.frontend_backend_live.worknote import WorkNote
from tests.unit.core_helpers import load, official_events

LOADED = load_prompts()
LINES = LOADED.lines
TEXTS = {
    "handover_line": "Handled together with your other request (D2).",
    "cancelled_line": "Stopped for your newer request.",
    "unconfirmed_line": "I could not confirm that the earlier request stopped; check before trying it again.",
}
LABEL = "delegation_core"


def response(kind, response_id, *, status, **fields):
    body = {"id": response_id, "object": "response", "status": status, "output": [], **fields}
    return {"type": f"response.{kind}", "response": body}


created, queued = partial(response, "created", status="queued"), partial(response, "queued", status="queued")
completed = partial(response, "completed", status="completed")
failed = partial(response, "failed", status="failed", error={"code": "server_error", "message": "The backend failed."})


def in_progress(response_id, previous=None):
    named = {"previous_response_id": previous} if previous else {}
    return response("in_progress", response_id, status="in_progress", **named)


def incomplete(response_id, reason=None):
    if reason is None:
        return response("incomplete", response_id, status="cancelled", incomplete_details=None)
    return response("incomplete", response_id, status="incomplete", incomplete_details={"reason": reason})


def done(item):
    return {"type": "response.output_item.done", "item": item}


def message(text, phase="final_answer", item_id="msg_1", status="completed"):
    part = {"type": "output_text", "text": text, "annotations": []}
    item = {"id": item_id, "type": "message", "role": "assistant", "status": status, "content": [part]}
    return done({**item, "phase": phase} if phase else item)


def function_call(name, call_id="call_1"):
    return done({"id": f"fc_{call_id}", "type": "function_call", "call_id": call_id, "name": name, "arguments": "{}"})


def mcp_call(label, name, item_id="mcp_1", output="{}", status="completed"):
    item = {"id": item_id, "type": "mcp_call", "server_label": label, "name": name, "arguments": "{}"}
    return done({**item, "output": output, "status": status})


def mcp_started(label, name, item_id="mcp_1"):
    """Return the event that starts a backend call, as the IM writes it: its item in progress, with no output yet."""
    item = {"id": item_id, "type": "mcp_call", "server_label": label, "name": name, "arguments": "{}"}
    return {"type": "response.output_item.added", "item": {**item, "output": None, "status": "in_progress"}}


HANDOVER = message(TEXTS["handover_line"], phase="commentary", item_id="handover")


class Results:
    """The reader's output, as the voice takes it."""

    def __init__(self):
        self.calls: list[tuple] = []

    def ready(self, result):
        self.calls.append(("ready", result.delegation_id, result.speak))

    def notice(self, delegation_id, text):
        self.calls.append(("notice", delegation_id, text))

    def ended(self, delegation_id):
        self.calls.append(("results.ended", delegation_id))

    def handed_over(self, delegation_id, taker):
        self.calls.append(("handed_over", delegation_id, taker))


class Display(Results):
    """The reader's display side; each card's ending names how it ended."""

    def started(self, delegation_id, request):
        self.calls.append(("started", delegation_id, request))

    def progress(self, delegation_id, line):
        self.calls.append(("progress", delegation_id, line))

    def shown(self, delegation_id, item):
        self.calls.append(("shown", delegation_id, item.get("name")))

    def answered(self, result):
        self.calls.append(("answered", result.delegation_id, result.speak))

    def ended(self, delegation_id, how):
        self.calls.append(("ended", delegation_id, how.value))


class Reading:
    """A reader with recording doubles around it, for the frontend's own delegations ``delegation_ids``."""

    def __init__(self, *delegation_ids, app_wait_seconds=None, client_tool_timeout_seconds=None):
        self.events: list[tuple[str, dict]] = []
        self.results, self.display, self.arrived = Results(), Display(), []
        note = WorkNote(LOADED.cascade, tracks=True, clock=lambda: 0.0, observe=self.observe)
        for delegation_id in delegation_ids:
            note.confirm(note.reserve(f"request {delegation_id}"), delegation_id)
        self.reader = DelegationReader(
            note,
            self.results,
            self.display,
            LINES,
            display_label=LABEL,
            app_wait_seconds=app_wait_seconds,
            item_text=message_text,
            observe=self.observe,
            outcome_arrived=self.arrived.append,
            client_tool_timeout_seconds=client_tool_timeout_seconds,
        )

    def observe(self, kind, **fields):
        self.events.append((kind, fields))

    def hand(self, delegation_id, *events):
        for event in events:
            self.reader.on_delegation_event(delegation_id, event)

    def calls(self, kind):
        """Return what the reader told the voice or the display of ``kind``, in order."""
        return [call[1:] for call in self.results.calls + self.display.calls if call[0] == kind]

    def notices(self):
        return [text for _, text in self.calls("notice")]

    def endings(self) -> dict:
        return dict(self.calls("ended"))

    def outcome(self) -> dict:
        """Return the reader's outcome in the fixture's terms."""

        def traced(kind):
            return [fields["delegation_id"] for name, fields in self.events if name == kind]

        def count(calls):
            return dict(Counter(delegation_id for delegation_id, *_ in calls))

        return {
            "results": count(self.calls("ready")),
            "endings": self.endings(),
            "takers": dict(self.calls("handed_over")),
            "waits": sorted(set(traced("delegation.waits_for_app"))),
            "typed": sorted(set(traced("delegation.typed"))),
            "progress": count(self.calls("progress")),
            "steps": dict(Counter(traced("delegation.backend_step"))),
            "shown": count(self.calls("shown")),
        }


class EndingTests(unittest.IsolatedAsyncioTestCase):
    async def test_every_fixture_scenario_ends_as_expected(self):
        # Only the structure tells the endings apart: the outcomes hold when the IM's lines say anything at all.
        for texts in (TEXTS, dict.fromkeys(TEXTS, "anything at all")):
            for name, scenario in load().items():
                with self.subTest(name, words=texts["handover_line"]):
                    expected = scenario["reader"]
                    typed = set(expected.get("typed", ()))
                    reading = Reading(*sorted(set(scenario["responses"].values()) - typed))
                    for delegation_id, event in official_events(scenario, texts):
                        reading.hand(delegation_id, event)
                    outcome = reading.outcome()
                    for key in ("results", "endings", "takers", "steps", "shown"):
                        self.assertEqual(outcome[key], expected.get(key, {}), key)
                    for key in ("waits", "typed"):
                        self.assertEqual(outcome[key], sorted(expected.get(key, ())), key)
                    if "progress" in expected:
                        self.assertEqual(outcome["progress"], expected["progress"], "progress")
                    reading.reader.close()

    async def test_a_failure_says_the_apology_or_the_unclear_line(self):
        for events, notices, ending in (
            ([failed("r1")], [LINES.apology], "failed"),
            ([mcp_call("cafe", "manage_cart"), failed("r1")], [LINES.unclear], "failed"),  # once a tool ran
            ([incomplete("r1", "max_output_tokens")], [LINES.apology], "failed"),
            ([incomplete("r1")], [], "cancelled"),
        ):
            with self.subTest(ending=ending, notices=notices):
                reading = Reading("D1")
                reading.hand("D1", created("r1"), in_progress("r1"), *events)
                self.assertEqual((reading.notices(), reading.endings()), (notices, {"D1": ending}))

    async def test_a_lost_stream_says_its_line_once(self):
        reading = Reading("D1", "D2")
        reading.hand("D1", created("r1"), in_progress("r1"))
        reading.reader.on_delegation_lost("D1")
        reading.reader.on_delegation_lost("D1")
        reading.hand("D1", message("Too late."), completed("r1"))
        self.assertEqual((reading.notices(), reading.endings()), ([LINES.lost], {"D1": "lost"}))
        reading.reader.close()
        reading.hand("D2", created("r2"), in_progress("r2"), message("Done."), completed("r2"))
        self.assertEqual(reading.calls("ready"), [])

    async def test_the_app_wait_line_then_the_unclear_one_at_the_deadline(self):
        for case, (continued, deadline, notices, endings) in {
            "the app-wait time runs out": (False, None, [LINES.app_wait_line], {}),
            "a continuation comes first": (True, None, [], {}),
            # The core writes nothing at its deadline, so the reader ends the wait at the same deadline.
            "the core's deadline passes": (False, 0.1, [LINES.app_wait_line, LINES.unclear], {"D1": "failed"}),
        }.items():
            with self.subTest(case):
                reading = Reading("D1", app_wait_seconds=0.05, client_tool_timeout_seconds=deadline)
                reading.hand("D1", created("r1"), in_progress("r1"), function_call("get_menu"), completed("r1"))
                if continued:
                    reading.hand("D1", created("r2"))
                await asyncio.sleep(0.2)
                self.assertEqual((reading.notices(), reading.endings()), (notices, endings))
                open_work = [work.id for work in reading.reader.worknote.open()]
                self.assertEqual(open_work, [] if endings else ["D1"])

    async def test_a_taker_named_after_the_deadline_takes_the_work(self):
        for previous, notices, ending in (("r1", [], "handed_over"), (None, [LINES.unclear], "failed")):
            with self.subTest(previous=previous):
                reading = Reading("D1", "D2", client_tool_timeout_seconds=0.05)
                reading.hand("D1", created("r1"), in_progress("r1"), function_call("get_menu"), completed("r1"))
                reading.hand("D2", created("r2"), queued("r2"))
                await asyncio.sleep(0.1)
                self.assertEqual((reading.notices(), reading.endings()), ([], {}))
                reading.hand("D2", in_progress("r2", previous=previous))
                self.assertEqual((reading.notices(), reading.endings()), (notices, {"D1": ending}))
                reading.hand("D1", created("r3"), in_progress("r3"), completed("r3"))
                self.assertEqual((reading.notices(), reading.endings()), (notices, {"D1": ending}))


class ResultTests(unittest.IsolatedAsyncioTestCase):
    async def test_each_final_answer_is_a_result_and_a_partial_is_not(self):
        reading = Reading("D1")
        reading.hand("D1", created("r1"), in_progress("r1"), message("Tea: yes.", item_id="m1"))
        self.assertEqual(reading.arrived, ["D1"])
        reading.hand("D1", message("Half", phase=None, status="incomplete", item_id="m2"))
        reading.hand("D1", message("Cake: no.", phase=None, item_id="m3"), completed("r1"))
        self.assertEqual(reading.calls("ready"), [("D1", "Tea: yes."), ("D1", "Cake: no.")])

    async def test_only_a_completed_display_item_is_shown_never_said(self):
        reading = Reading("D1")
        error = '{"error": "show failed: display must be non-empty text"}'
        reading.hand("D1", created("r1"), in_progress("r1"), mcp_call(LABEL, "show", output=error, status="failed"))
        self.assertEqual(reading.display.calls, [])
        reading.hand("D1", mcp_call(LABEL, "show", item_id="mcp_2"))
        self.assertEqual(reading.display.calls, [("shown", "D1", "show")])
        reading.hand("D1", message("Done."), completed("r1"))
        self.assertEqual([call[0] for call in reading.display.calls], ["shown", "answered", "ended"])
        self.assertNotIn("show", str(reading.results.calls))

    async def test_a_line_waits_while_another_delegation_may_name_it(self):
        # A commentary line waits while D2 or D3 may name r1, unless its response went on after it.
        reading = Reading("D1", "D2", "D3")
        reading.hand("D2", created("r2"), queued("r2"))
        reading.hand("D3", created("r3"), queued("r3"))
        reading.hand("D1", created("r1"), in_progress("r1"), message("Checking the menu.", phase="commentary"))
        self.assertEqual(reading.calls("progress"), [])
        reading.hand("D1", mcp_started("cafe", "get_menu"))
        reading.hand("D1", message("Pricing your order.", phase="commentary", item_id="m2"))
        reading.hand("D2", in_progress("r2"))
        self.assertEqual(reading.calls("progress"), [("D1", "Checking the menu.")])
        reading.hand("D3", in_progress("r3"))
        self.assertEqual(reading.calls("progress"), [("D1", "Checking the menu."), ("D1", "Pricing your order.")])


class HandoverTests(unittest.IsolatedAsyncioTestCase):
    async def test_a_handovers_line_is_never_progress(self):
        # Each response has its own stream, so the naming can come before or after the older line and its end.
        for taker_first in (True, False):
            with self.subTest(taker_first=taker_first):
                reading = Reading("D1", "D2")
                reading.hand("D2", created("r2"), queued("r2"))
                reading.hand("D1", created("r1"), in_progress("r1"))
                naming, older = ("D2", [in_progress("r2", previous="r1")]), ("D1", [HANDOVER, completed("r1")])
                for delegation_id, events in [naming, older] if taker_first else [older, naming]:
                    reading.hand(delegation_id, *events)
                self.assertEqual((reading.calls("progress"), reading.notices()), ([], []))
                self.assertEqual(reading.endings(), {"D1": "handed_over"})
                self.assertEqual(reading.calls("handed_over"), [("D1", "D2")])

    async def test_an_answered_handover_keeps_its_answer(self):
        # D2 carries that a tool ran, so its failure leaves the outcome unclear.
        for older_first in (False, True):
            with self.subTest(older_first=older_first):
                reading = Reading("D1", "D2")
                reading.hand("D2", created("r2"), queued("r2"))
                reading.hand("D1", created("r1"), in_progress("r1"), mcp_call("cafe", "lookup"))
                reading.hand("D1", message("Task A done.", item_id="a"))
                naming, older = ("D2", [in_progress("r2", previous="r1")]), ("D1", [HANDOVER, completed("r1")])
                for delegation_id, events in [older, naming] if older_first else [naming, older]:
                    reading.hand(delegation_id, *events)
                self.assertEqual(reading.calls("handed_over"), [("D1", "D2")])
                self.assertEqual(reading.endings(), {"D1": "answered"})
                reading.hand("D2", failed("r2"))
                self.assertEqual(reading.notices(), [LINES.unclear])
                self.assertEqual(reading.calls("ready"), [("D1", "Task A done.")])

    async def test_an_early_naming_waits_for_its_response(self):
        reading = Reading("D1", "D2")
        reading.hand("D2", created("r2"), in_progress("r2", previous="r1"))
        reading.hand("D1", created("r1"), in_progress("r1"), HANDOVER, completed("r1"))
        self.assertEqual((reading.notices(), reading.endings()), ([], {"D1": "handed_over"}))

    async def test_a_response_no_taker_names_ends_when_none_can(self):
        for answer, previous, notices, ending in (
            ([message("Tea: yes.")], None, [], "answered"),
            ([], None, [LINES.unclear], "late"),
            ([], "unknown", [LINES.unclear], "late"),
        ):
            with self.subTest(ending=ending, previous=previous):
                reading = Reading("D1", "D2")
                reading.hand("D2", created("r2"), queued("r2"))
                reading.hand("D1", created("r1"), in_progress("r1"), *answer, completed("r1"))
                self.assertEqual(reading.endings(), {})
                reading.hand("D2", in_progress("r2", previous=previous))
                self.assertEqual((reading.notices(), reading.endings()), (notices, {"D1": ending}))
                self.assertEqual(reading.calls("handed_over"), [])
