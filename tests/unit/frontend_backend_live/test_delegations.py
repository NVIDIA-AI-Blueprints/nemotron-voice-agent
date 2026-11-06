# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D103, D107

"""The reader reads every ending from official fields, by structure, with the display at once."""

import asyncio
import unittest
from collections import Counter

from examples.frontend_backend_live.delegation_core.items import message_text
from examples.frontend_backend_live.delegations import DelegationReader
from examples.frontend_backend_live.worknote import WorkNote
from tests.unit.contracts.endings import load, official_events
from tests.unit.frontend_backend_live.doubles import (
    LINES,
    PROMPTS,
    RecordingDisplay,
    RecordingResults,
    completed,
    created,
    failed,
    function_call,
    in_progress,
    incomplete,
    mcp_call,
    mcp_started,
    message,
    queued,
)

# The IM's own lines, as the fixture marks them ([key]): the reader never reads them, only says them.
TEXTS = {
    "handover_line": "Handled together with your newer request (D2).",  # as the IM writes it when D2 takes over
    "cancelled_line": "Stopped for your newer request.",
    "unconfirmed_line": "I could not confirm that the earlier request stopped; check before trying it again.",
}
HANDOVER = message(TEXTS["handover_line"], phase="commentary", item_id="handover")
LABEL = "delegation_core"


class Reading:
    """A reader with recording doubles around it, for the frontend's own delegations ``delegation_ids``."""

    def __init__(self, *delegation_ids, app_wait_seconds=None):
        self.events: list[tuple[str, dict]] = []
        self.results, self.display, self.arrived = RecordingResults(), RecordingDisplay(), []
        note = WorkNote(PROMPTS, tracks=True, clock=lambda: 0.0, observe=self.observe)
        for delegation_id in delegation_ids:  # reserved, then confirmed, before any event
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
        )

    def observe(self, kind, **fields):
        self.events.append((kind, fields))

    def hand(self, delegation_id, *events):
        for event in events:
            self.reader.on_delegation_event(delegation_id, event)

    def calls(self, kind):
        """What the reader told the voice or the display of ``kind`` (not ``ended``: :meth:`endings`), in order."""
        return [call[1:] for call in self.results.calls + self.display.calls if call[0] == kind]

    def notices(self):
        return [text for _, text in self.calls("notice")]

    def endings(self) -> dict:
        return {call[1]: call[2] for call in self.display.calls if call[0] == "ended"}

    def outcome(self) -> dict:
        """The reader's outcome in the fixture's terms."""

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


class EndingsFixtureTests(unittest.IsolatedAsyncioTestCase):
    async def test_every_scenario_gives_its_expected_outcome_whatever_words_the_ims_lines_carry(self):
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
                    # The fixture counts progress only where it is the point; every commentary line is progress.
                    if "progress" in expected:
                        self.assertEqual(outcome["progress"], expected["progress"], "progress")
                    reading.reader.close()


class EndingLineTests(unittest.IsolatedAsyncioTestCase):
    """Each ending says the voice's own line, as written, decided by structure."""

    async def test_a_failure_says_the_apology_or_unclear_once_a_tool_ran_and_a_close_says_nothing(self):
        for events, notices, ending in (
            ([failed("r1")], [LINES.apology], "failed"),
            ([mcp_call("cafe", "manage_cart"), failed("r1")], [LINES.unclear], "failed"),
            ([incomplete("r1", "max_output_tokens")], [LINES.apology], "failed"),
            ([incomplete("r1")], [], "cancelled"),
        ):
            with self.subTest(ending=ending, notices=notices):
                reading = Reading("D1")
                reading.hand("D1", created("r1"), in_progress("r1"), *events)
                self.assertEqual((reading.notices(), reading.endings()), (notices, {"D1": ending}))

    async def test_a_lost_stream_says_the_lost_line_once_and_events_after_an_end_or_the_close_are_ignored(self):
        reading = Reading("D1", "D2")
        reading.hand("D1", created("r1"), in_progress("r1"))
        reading.reader.on_delegation_lost("D1")
        reading.reader.on_delegation_lost("D1")
        reading.hand("D1", message("Too late."), completed("r1"))
        self.assertEqual((reading.notices(), reading.endings()), ([LINES.lost], {"D1": "lost"}))
        reading.reader.close()
        reading.hand("D2", created("r2"), in_progress("r2"), message("Done."), completed("r2"))
        self.assertEqual(reading.calls("ready"), [])

    async def test_the_app_wait_line_is_said_once_its_time_runs_out_unless_a_continuation_comes_first(self):
        for continued, notices in ((False, [LINES.app_wait_line]), (True, [])):
            with self.subTest(continued=continued):
                reading = Reading("D1", app_wait_seconds=0.05)
                reading.hand("D1", created("r1"), in_progress("r1"), function_call("get_menu"), completed("r1"))
                if continued:
                    reading.hand("D1", created("r2"))
                await asyncio.sleep(0.15)
                self.assertEqual(reading.notices(), notices)


class ResultTests(unittest.IsolatedAsyncioTestCase):
    async def test_every_final_answer_is_one_ready_result_and_a_partial_message_is_nothing(self):
        reading = Reading("D1")
        reading.hand("D1", created("r1"), in_progress("r1"), message("Tea: yes.", item_id="m1"))
        self.assertEqual(reading.arrived, ["D1"])  # its pending lead-in is dropped
        reading.hand("D1", message("Half", phase=None, status="incomplete", item_id="m2"))
        reading.hand("D1", message("Cake: no.", phase=None, item_id="m3"), completed("r1"))  # no phase: a final answer
        self.assertEqual(reading.calls("ready"), [("D1", "Tea: yes."), ("D1", "Cake: no.")])

    async def test_only_a_completed_display_item_is_shown_at_once_and_it_is_never_said(self):
        # A show that failed (an empty display) is no display: its error output never reaches the screen.
        reading = Reading("D1")
        error = '{"error": "show failed: display must be non-empty text"}'
        reading.hand("D1", created("r1"), in_progress("r1"), mcp_call(LABEL, "show", output=error, status="failed"))
        self.assertEqual(reading.display.calls, [])
        reading.hand("D1", mcp_call(LABEL, "show", item_id="mcp_2"))
        self.assertEqual(reading.display.calls, [("shown", "D1", "show")])
        reading.hand("D1", message("Done."), completed("r1"))
        self.assertEqual([call[0] for call in reading.display.calls], ["shown", "answered", "ended"])
        self.assertNotIn("show", str(reading.results.calls))


class ProgressTests(unittest.IsolatedAsyncioTestCase):
    """A commentary line is progress, never a handover's: one on a response another delegation may still name waits."""

    async def test_a_line_another_delegation_may_still_name_waits_until_its_response_goes_on_or_none_can(self):
        reading = Reading("D1", "D2", "D3")
        reading.hand("D2", created("r2"), queued("r2"))
        reading.hand("D3", created("r3"), queued("r3"))
        reading.hand("D1", created("r1"), in_progress("r1"), message("Checking flights.", phase="commentary"))
        self.assertEqual(reading.calls("progress"), [])  # D2 or D3 may still name r1
        reading.hand("D1", mcp_started("airline", "search_flights"))  # its response went on: the line was progress
        reading.hand("D1", message("Comparing prices.", phase="commentary", item_id="m2"))
        reading.hand("D2", in_progress("r2"))  # D2 started without naming r1; D3 still may
        self.assertEqual(reading.calls("progress"), [("D1", "Checking flights.")])
        reading.hand("D3", in_progress("r3"))  # no delegation can name r1 any more
        self.assertEqual(reading.calls("progress"), [("D1", "Checking flights."), ("D1", "Comparing prices.")])

    async def test_a_handovers_line_is_never_progress_whether_its_taker_is_named_before_or_after_it(self):
        # Each response has a stream of its own, so the older response's line and end may come before the taker's
        # in_progress that names it: nothing is said, and the naming hands the work over.
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


class HandoverTests(unittest.IsolatedAsyncioTestCase):
    async def test_an_answered_response_handed_over_keeps_its_answer_and_its_taker_carries_that_a_tool_ran(self):
        # Task A's answer is on D1's response, then a newer request lands in task B and the IM names D2 the taker, on
        # D2's own stream. Whichever stream the reader gets first, D1's answer is a result and D1 ends answered; D2
        # carries that a tool ran, so its failure leaves the outcome unclear.
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

    async def test_a_naming_that_arrives_before_the_response_it_names_is_kept_until_that_response_opens(self):
        reading = Reading("D1", "D2")
        reading.hand("D2", created("r2"), in_progress("r2", previous="r1"))  # r1 is still on its way
        reading.hand("D1", created("r1"), in_progress("r1"), HANDOVER, completed("r1"))
        self.assertEqual((reading.notices(), reading.endings()), ([], {"D1": "handed_over"}))

    async def test_a_response_no_taker_names_ends_once_no_delegation_can_name_it(self):
        # Answered, it ends answered; with no answer it came too late. A response the reader does not know names none.
        for answer, previous, notices, ending in (
            ([message("Tea: yes.")], None, [], "answered"),
            ([], None, [LINES.unclear], "late"),
            ([], "unknown", [LINES.unclear], "late"),
        ):
            with self.subTest(ending=ending, previous=previous):
                reading = Reading("D1", "D2")
                reading.hand("D2", created("r2"), queued("r2"))
                reading.hand("D1", created("r1"), in_progress("r1"), *answer, completed("r1"))
                self.assertEqual(reading.endings(), {})  # D2 may still name it
                reading.hand("D2", in_progress("r2", previous=previous))
                self.assertEqual((reading.notices(), reading.endings()), (notices, {"D1": ending}))
                self.assertEqual(reading.calls("handed_over"), [])

    async def test_a_taker_named_at_the_deadline_takes_over_work_that_waits_for_the_app(self):
        # The app never answered D1's call and a newer request landed in another task of D1's Work: at the client-tool
        # deadline the IM names D2 the taker of D1's last response. D1 ends handed over with no app-wait line, and D2's
        # answers are said like any result.
        reading = Reading("D1", "D2", app_wait_seconds=0.05)
        reading.hand("D1", created("r1"), in_progress("r1"), function_call("get_menu"), completed("r1"))
        reading.hand("D2", created("r2"), queued("r2"), in_progress("r2", previous="r1"))
        reading.hand("D2", message("Open on Sundays."), message("Outcome unknown.", item_id="m2"), completed("r2"))
        await asyncio.sleep(0.1)
        reading.hand("D1", created("r3"), queued("r3"), in_progress("r3"), completed("r3"))  # the app's late answer
        self.assertEqual(reading.endings(), {"D1": "handed_over", "D2": "answered"})
        self.assertEqual(reading.calls("ready"), [("D2", "Open on Sundays."), ("D2", "Outcome unknown.")])
        self.assertEqual(reading.notices(), [])
