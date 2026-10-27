# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D103, D107

"""The reader reads every ending from official fields, by structure, with the display at once."""

import asyncio
import unittest
from collections import Counter

from examples.frontend_backend_live.cascade.delegations import DelegationReader
from examples.frontend_backend_live.cascade.lines import catalog_lines
from examples.frontend_backend_live.cascade.ports import Ending
from examples.frontend_backend_live.cascade.worknote import WorkNote
from tests.unit.contracts.endings import load, official_events
from tests.unit.frontend_backend_live.cascade.doubles import (
    RecordingDisplay,
    RecordingResults,
    completed,
    created,
    failed,
    function_call,
    in_progress,
    incomplete,
    mcp_call,
    message,
    prompt_set,
    queued,
)

# The IM's own lines, as the fixture marks them ([key]): the reader never reads them, only says them.
TEXTS = {
    "handover_line": "Handled together with your newer request.",
    "cancelled_line": "Stopped for your newer request.",
    "unconfirmed_line": "I could not confirm that the earlier request stopped; check before trying it again.",
}
LABEL = "voiceclaw"


class Reading:
    """A reader with recording doubles around it."""

    def __init__(self, *, app_wait_seconds=None):
        self.events: list[tuple[str, dict]] = []
        self.results, self.display = RecordingResults(), RecordingDisplay()
        self.note = WorkNote(prompt_set(), tracks=True, clock=lambda: 0.0, observe=self.observe)
        self.lines = catalog_lines()
        self.arrived: list[str] = []
        self.reader = DelegationReader(
            self.note,
            self.results,
            self.display,
            self.lines,
            display_label=LABEL,
            app_wait_seconds=app_wait_seconds,
            observe=self.observe,
            outcome_arrived=self.arrived.append,
        )

    def observe(self, kind, **fields):
        self.events.append((kind, fields))

    def delegated(self, *delegation_ids):
        """The frontend's own delegations: reserved, then confirmed, before any event."""
        for delegation_id in delegation_ids:
            self.note.confirm(self.note.reserve(f"request {delegation_id}"), delegation_id)

    def hand(self, delegation_id, *events):
        for event in events:
            self.reader.on_delegation_event(delegation_id, event)

    def outcome(self) -> dict:
        """The reader's outcome in the fixture's terms."""
        count = Counter
        calls = self.display.calls
        return {
            "results": dict(count(call[1] for call in self.results.calls if call[0] == "ready")),
            "endings": {call[1]: call[2] for call in calls if call[0] == "ended"},
            "takers": {call[1]: call[2] for call in self.results.calls if call[0] == "handed_over"},
            "waits": sorted(
                {fields["delegation_id"] for kind, fields in self.events if kind == "delegation.waits_for_app"}
            ),
            "typed": sorted({fields["delegation_id"] for kind, fields in self.events if kind == "delegation.typed"}),
            "progress": dict(count(call[1] for call in calls if call[0] == "progress")),
            "steps": dict(
                count(fields["delegation_id"] for kind, fields in self.events if kind == "delegation.backend_step")
            ),
            "shown": dict(count(call[1] for call in calls if call[0] == "shown")),
        }


class EndingsFixtureTests(unittest.IsolatedAsyncioTestCase):
    async def test_every_scenario_gives_its_expected_outcome(self):
        for name, scenario in load().items():
            with self.subTest(name):
                reading = Reading()
                expected = scenario["reader"]
                typed = set(expected.get("typed", ()))
                reading.delegated(*sorted(set(scenario["responses"].values()) - typed))
                for delegation_id, event in official_events(scenario, TEXTS):
                    reading.reader.on_delegation_event(delegation_id, event)
                outcome = reading.outcome()
                for key in ("results", "endings", "takers", "steps", "shown"):
                    self.assertEqual(outcome[key], expected.get(key, {}), key)
                for key in ("waits", "typed"):
                    self.assertEqual(outcome[key], sorted(expected.get(key, ())), key)
                # The fixture counts progress only where it is the point; every commentary line is progress.
                if "progress" in expected:
                    self.assertEqual(outcome["progress"], expected["progress"], "progress")
                reading.reader.close()

    async def test_a_cancel_line_is_said_like_any_result_and_the_handover_line_is_never_said(self):
        # An unconfirmed stop is said as such. The cancel is the older delegation's answer; a handover's one
        # commentary line is progress, and the taker carries the work.
        scenarios = load()
        for name, said in (
            (
                "cancelled_for_a_newer_request_stopped",
                [("ready", "D2", "Done."), ("ready", "D1", TEXTS["cancelled_line"])],
            ),
            (
                "cancelled_for_a_newer_request_unconfirmed",
                [("ready", "D2", "Done."), ("ready", "D1", TEXTS["unconfirmed_line"])],
            ),
            ("handed_over_with_the_taker_named", [("ready", "D2", "Done.")]),
        ):
            with self.subTest(name):
                reading = Reading()
                reading.delegated("D1", "D2")
                for delegation_id, event in official_events(scenarios[name], TEXTS):
                    reading.reader.on_delegation_event(delegation_id, event)
                self.assertEqual([call for call in reading.results.calls if call[0] == "ready"], said)


class EndingLineTests(unittest.IsolatedAsyncioTestCase):
    """Each ending says the voice's own line, as written, decided by structure."""

    def notices(self, reading):
        return [call[2] for call in reading.results.calls if call[0] == "notice"]

    async def test_a_failure_says_the_apology_or_unclear_when_a_tool_ran(self):
        lines = catalog_lines()
        for events, line in (
            ([created("r1"), in_progress("r1"), failed("r1")], lines.apology),
            ([created("r1"), in_progress("r1"), mcp_call("cafe", "manage_cart"), failed("r1")], lines.unclear),
            ([created("r1"), in_progress("r1"), incomplete("r1", "max_output_tokens")], lines.apology),
        ):
            with self.subTest(line=line):
                reading = Reading()
                reading.delegated("D1")
                reading.hand("D1", *events)
                self.assertEqual(self.notices(reading), [line])
                self.assertEqual(reading.display.calls[-1], ("ended", "D1", Ending.FAILED.value))
                self.assertNotIn("D1", reading.note)

    async def test_a_late_continuation_says_unclear_and_close_says_nothing(self):
        reading = Reading()
        reading.delegated("D1", "D2")
        reading.hand("D1", created("r1"), in_progress("r1"), function_call("get_menu"), completed("r1"))
        reading.hand("D1", created("r2"), queued("r2"), completed("r2"))
        reading.hand("D2", created("r3"), in_progress("r3"), incomplete("r3"))
        self.assertEqual(self.notices(reading), [catalog_lines().unclear])
        endings = {call[1]: call[2] for call in reading.display.calls if call[0] == "ended"}
        self.assertEqual(endings, {"D1": "late", "D2": "cancelled"})

    async def test_a_lost_stream_says_the_lost_line_once(self):
        reading = Reading()
        reading.delegated("D1")
        reading.hand("D1", created("r1"), in_progress("r1"))
        reading.reader.on_delegation_lost("D1")
        reading.reader.on_delegation_lost("D1")
        self.assertEqual(self.notices(reading), [catalog_lines().lost])
        self.assertEqual(reading.display.calls[-1], ("ended", "D1", "lost"))
        reading.hand("D1", message("Too late."), completed("r1"))  # an ended delegation's events are ignored
        self.assertEqual([call for call in reading.results.calls if call[0] == "ready"], [])

    async def test_no_text_decides_an_ending(self):
        # The handover line, the cancelled line or any other words: only the structure tells the endings apart.
        for text in (TEXTS["handover_line"], TEXTS["cancelled_line"], "anything at all"):
            reading = Reading()
            reading.delegated("D1", "D2")
            reading.hand("D1", created("r1"), in_progress("r1"))
            reading.hand("D2", created("r2"), in_progress("r2", previous="r1"))
            reading.hand("D1", message(text, phase="commentary"), completed("r1"))
            self.assertEqual(reading.display.calls[-1], ("ended", "D1", "handed_over"))
            reading = Reading()  # no taker named it: no final answer, no taker, so it came too late
            reading.delegated("D1")
            reading.hand("D1", created("r1"), in_progress("r1"), message(text, phase="commentary"), completed("r1"))
            self.assertEqual(reading.display.calls[-1], ("ended", "D1", "late"))
            reading = Reading()  # as a final answer, any words are a result
            reading.delegated("D1")
            reading.hand("D1", created("r1"), in_progress("r1"), message(text), completed("r1"))
            self.assertEqual(reading.display.calls[-1], ("ended", "D1", "answered"))


class ReaderPartTests(unittest.IsolatedAsyncioTestCase):
    notices = EndingLineTests.notices

    async def test_queued_until_in_progress_then_running_with_its_steps_and_newest_line(self):
        reading = Reading()
        reading.delegated("D1")
        reading.hand("D1", created("r1"), queued("r1"))
        self.assertIn('waiting to start: "request D1"', reading.note.line())
        reading.hand("D1", in_progress("r1"), message("Looking at the menu.", phase="commentary"))
        reading.hand("D1", mcp_call("cafe", "get_menu"), function_call("check_hours"))
        line = reading.note.line()
        self.assertIn("running for 0 s, now using get menu, check hours", line)
        self.assertIn('latest update: "Looking at the menu."', line)

    async def test_every_final_answer_is_one_ready_result_and_the_entry_reads_said_next(self):
        reading = Reading()
        reading.delegated("D1")
        reading.hand("D1", created("r1"), in_progress("r1"), message("Tea: yes.", item_id="m1"))
        self.assertEqual(reading.arrived, ["D1"])  # its pending acknowledgement is dropped
        self.assertIn('answer ready, said next: "request D1"', reading.note.line())
        reading.hand("D1", message("Cake: no.", item_id="m2"), completed("r1"))
        ready = [call for call in reading.results.calls if call[0] == "ready"]
        self.assertEqual(ready, [("ready", "D1", "Tea: yes."), ("ready", "D1", "Cake: no.")])
        self.assertIn("D1", reading.note)  # it leaves once its results are said
        reading.note.said("D1")
        reading.note.said("D1")
        self.assertNotIn("D1", reading.note)

    async def test_a_message_with_no_phase_is_a_final_answer_and_a_partial_one_is_nothing(self):
        reading = Reading()
        reading.delegated("D1")
        reading.hand("D1", created("r1"), in_progress("r1"), message("Half", phase=None, status="incomplete"))
        reading.hand("D1", message("Whole.", phase=None), completed("r1"))
        self.assertEqual([call[2] for call in reading.results.calls if call[0] == "ready"], ["Whole."])

    async def test_the_display_item_is_shown_at_once_and_never_said(self):
        reading = Reading()
        reading.delegated("D1")
        reading.hand("D1", created("r1"), in_progress("r1"), mcp_call(LABEL, "show"))
        self.assertEqual(reading.display.calls, [("shown", "D1", "show")])
        reading.hand("D1", message("Done."), completed("r1"))
        self.assertEqual(reading.display.kinds(), ["shown", "answered", "ended"])
        self.assertNotIn("show", str(reading.results.calls))

    async def test_only_a_completed_display_item_is_shown(self):
        # A show that failed (an empty display) is no display: its error output never reaches the screen, and it is no
        # tool step either.
        reading = Reading()
        reading.delegated("D1")
        failed_show = mcp_call(LABEL, "show", output='{"error": "show failed: display must be non-empty text"}')
        failed_show["item"]["status"] = "failed"
        reading.hand("D1", created("r1"), in_progress("r1"), failed_show)
        self.assertEqual(reading.display.calls, [])
        self.assertEqual([entry.step for entry in reading.note.open()], [()])
        reading.hand("D1", mcp_call(LABEL, "show", item_id="mcp_2"), message("Done."), completed("r1"))
        self.assertEqual(reading.display.kinds(), ["shown", "answered", "ended"])

    async def test_a_taker_named_for_a_response_the_reader_does_not_know_changes_nothing(self):
        reading = Reading()
        reading.delegated("D1", "D2")
        reading.hand("D1", created("r1"), in_progress("r1"))
        reading.hand("D2", created("r2"), in_progress("r2", previous="unknown"))
        reading.hand("D1", completed("r1"))
        self.assertEqual(reading.display.calls[-1], ("ended", "D1", "late"))  # no taker named it: no handover

    async def test_a_handover_joins_the_words_and_the_clock_to_the_named_taker_not_the_newest(self):
        reading = Reading()
        reading.delegated("D1", "D2", "D3")
        reading.hand("D1", created("r1"), in_progress("r1"))
        reading.hand("D3", created("r3"), queued("r3"))
        reading.hand("D2", created("r2"), in_progress("r2", previous="r1"))
        reading.hand("D1", message(TEXTS["handover_line"], phase="commentary"), completed("r1"))
        requests = [(entry.id, entry.request) for entry in reading.note.open()]
        self.assertEqual(requests, [("D2", "request D1 / request D2"), ("D3", "request D3")])
        self.assertEqual(
            [call for call in reading.results.calls if call[0] == "handed_over"], [("handed_over", "D1", "D2")]
        )

    async def test_the_handover_line_is_never_a_progress_line_a_quiet_spell_could_say(self):
        # The IM names the taker before the older response's one commentary line, so that line is the handover's: once
        # the taker is named, the older delegation's words go to it, and its line is never progress a quiet spell says.
        reading = Reading()
        reading.delegated("D1", "D2")
        reading.hand("D1", created("r1"), in_progress("r1"), message("Checking flights.", phase="commentary"))
        reading.hand("D2", created("r2"), in_progress("r2", previous="r1"))
        self.assertEqual([entry.id for entry in reading.note.open()], ["D2"])
        reading.hand("D1", message(TEXTS["handover_line"], phase="commentary", item_id="h1"))
        progress = [call for call in reading.display.calls if call[0] == "progress"]
        self.assertEqual(progress, [("progress", "D1", "Checking flights.")])
        reading.hand("D1", completed("r1"))
        self.assertEqual(reading.display.calls[-1], ("ended", "D1", "handed_over"))

    async def test_a_handover_line_read_before_its_takers_naming_is_never_a_progress_line(self):
        # The IM names the taker, on the taker's stream, before it writes the older response's handover line and end,
        # on the older one's. A line on a response another delegation may still name waits for that response's next
        # event, and the handover's line is the last item before its response's end: an end next drops it.
        reading = Reading()
        reading.delegated("D1", "D2")
        reading.hand("D2", created("r2"), queued("r2"))
        reading.hand("D1", created("r1"), in_progress("r1"), message(TEXTS["handover_line"], phase="commentary"))
        self.assertEqual([call for call in reading.display.calls if call[0] == "progress"], [])
        self.assertNotIn(TEXTS["handover_line"], reading.note.line())
        reading.hand("D1", completed("r1"))
        reading.hand("D2", in_progress("r2", previous="r1"), message("Done."), completed("r2"))
        self.assertEqual([call for call in reading.display.calls if call[0] == "progress"], [])
        self.assertIn(("ended", "D1", "handed_over"), reading.display.calls)

    async def test_a_progress_line_waits_only_while_another_delegation_may_still_name_its_response(self):
        alone = Reading()
        alone.delegated("D1")
        alone.hand("D1", created("r1"), in_progress("r1"), message("Checking flights.", phase="commentary"))
        self.assertEqual(alone.display.calls[-1], ("progress", "D1", "Checking flights."))  # at once
        reading = Reading()
        reading.delegated("D1", "D2", "D3")
        reading.hand("D2", created("r2"), queued("r2"))
        reading.hand("D3", created("r3"), queued("r3"))
        reading.hand("D1", created("r1"), in_progress("r1"), message("Checking flights.", phase="commentary"))
        self.assertNotIn("Checking flights.", reading.note.line())  # D2 or D3 may still name r1
        reading.hand("D1", function_call("search_flights"))  # its response goes on: the line was progress
        self.assertIn('latest update: "Checking flights."', reading.note.line())
        reading.hand("D1", message("Comparing prices.", phase="commentary", item_id="m2"))
        reading.hand("D2", in_progress("r2"))  # D2 started without naming r1; D3 still may
        self.assertNotIn("Comparing prices.", reading.note.line())
        reading.hand("D3", in_progress("r3"))  # no delegation can name r1 any more: the line is progress
        self.assertIn('latest update: "Comparing prices."', reading.note.line())

    async def test_an_older_response_that_ends_before_its_takers_naming_arrives_is_handed_over_and_says_nothing(self):
        # Each response has a stream of its own, so the IM's order across them may not hold at the reader: here the
        # older response's line and end come before the taker's in_progress that names it. While the taker
        # has not started it may still name it, so nothing is said; its naming hands the work over.
        reading = Reading()
        reading.delegated("D1", "D2")
        reading.hand("D2", created("r2"), queued("r2"))
        reading.hand("D1", created("r1"), in_progress("r1"))
        reading.hand("D1", message(TEXTS["handover_line"], phase="commentary"), completed("r1"))
        self.assertEqual(self.notices(reading), [])
        self.assertIn("D1", reading.reader.open)
        reading.hand("D2", in_progress("r2", previous="r1"), message("Done."), completed("r2"))
        self.assertEqual(self.notices(reading), [])
        self.assertIn(("ended", "D1", "handed_over"), reading.display.calls)
        handed = [call for call in reading.results.calls if call[0] in ("handed_over", "ready")]
        self.assertEqual(handed, [("handed_over", "D1", "D2"), ("ready", "D2", "Done.")])  # the told go first

    async def test_a_response_that_answered_then_was_handed_over_gives_its_taker_in_either_stream_order(self):
        # A Work with two tasks: task A's answer is on D1's response, then a newer request lands in task B and the
        # IM names D2 the taker before D1's handover line and end, but on D2's own stream. Whichever stream the
        # reader gets first, D1's answer is a result, D2 carries the words, the told and that a tool ran, and D1 ends
        # answered.
        lines = catalog_lines()
        for older_first in (False, True):
            with self.subTest(older_first=older_first):
                reading = Reading()
                reading.delegated("D1", "D2")
                reading.hand("D2", created("r2"), queued("r2"))
                reading.hand("D1", created("r1"), in_progress("r1"), mcp_call("cafe", "lookup"))
                reading.hand("D1", message("Task A done.", item_id="a"))
                naming = ("D2", [in_progress("r2", previous="r1")])
                older_end = ("D1", [message(TEXTS["handover_line"], phase="commentary", item_id="h"), completed("r1")])
                for delegation_id, events in [older_end, naming] if older_first else [naming, older_end]:
                    reading.hand(delegation_id, *events)
                self.assertEqual(
                    [call for call in reading.results.calls if call[0] == "handed_over"], [("handed_over", "D1", "D2")]
                )
                self.assertEqual(
                    [(entry.id, entry.request) for entry in reading.note.open()], [("D2", "request D1 / request D2")]
                )
                self.assertIn(("ended", "D1", "answered"), reading.display.calls)
                reading.hand("D2", failed("r2"))  # the work the taker carries ran a tool: its failure is unclear
                self.assertEqual(self.notices(reading), [lines.unclear])
                self.assertEqual([call[2] for call in reading.results.calls if call[0] == "ready"], ["Task A done."])

    async def test_an_answered_response_no_taker_names_ends_answered_once_none_can(self):
        reading = Reading()
        reading.delegated("D1", "D2")
        reading.hand("D2", created("r2"), queued("r2"))
        reading.hand("D1", created("r1"), in_progress("r1"), message("Tea: yes."), completed("r1"))
        self.assertNotIn(("ended", "D1", "answered"), reading.display.calls)  # D2 may still name it
        reading.hand("D2", in_progress("r2"))  # D2 started without naming it
        self.assertEqual([call for call in reading.display.calls if call[0] == "ended"], [("ended", "D1", "answered")])
        self.assertEqual(self.notices(reading), [])
        self.assertEqual([call for call in reading.results.calls if call[0] == "handed_over"], [])

    async def test_a_naming_that_arrives_before_the_response_it_names_is_kept_until_that_response_opens(self):
        reading = Reading()
        reading.delegated("D1", "D2")
        reading.hand("D2", created("r2"), in_progress("r2", previous="r1"))  # r1 is still on its way
        reading.hand("D1", created("r1"), in_progress("r1"), message(TEXTS["handover_line"], phase="commentary"))
        reading.hand("D1", completed("r1"))
        self.assertEqual(self.notices(reading), [])
        self.assertEqual(reading.display.calls[-1], ("ended", "D1", "handed_over"))

    async def test_a_taker_named_at_the_deadline_for_a_response_that_ended_with_calls_takes_the_work_over(self):
        # The app never answered D1's call and a newer request landed in another task of D1's Work: at the client-tool
        # deadline the IM names D2 the taker of D1's last response. D1 ends handed over, its app wait stops, and its
        # words, its clock and that a tool ran go to D2, whose answers are said like any result.
        reading = Reading(app_wait_seconds=0.05)
        reading.delegated("D1", "D2")
        reading.hand("D1", created("r1"), in_progress("r1"), function_call("get_menu"), completed("r1"))
        reading.hand("D2", created("r2"), queued("r2"))
        reading.hand("D2", in_progress("r2", previous="r1"))
        self.assertIn(("ended", "D1", "handed_over"), reading.display.calls)
        self.assertEqual(reading.reader.app_waits, {})
        self.assertEqual(
            [(entry.id, entry.request) for entry in reading.note.open()], [("D2", "request D1 / request D2")]
        )
        reading.hand("D2", message("Open on Sundays."), message("Outcome unknown.", item_id="m2"), completed("r2"))
        ready = [call[2] for call in reading.results.calls if call[0] == "ready"]
        self.assertEqual(ready, ["Open on Sundays.", "Outcome unknown."])
        self.assertIn(("ended", "D2", "answered"), reading.display.calls)
        await asyncio.sleep(0.1)
        self.assertEqual(self.notices(reading), [])  # no app-wait line for work the taker carries
        reading.hand("D1", created("r3"), queued("r3"), in_progress("r3"), completed("r3"))  # the app's late answer
        self.assertEqual(self.notices(reading), [])

    async def test_a_taker_whose_work_ran_a_tool_before_it_took_over_says_unclear_when_it_fails(self):
        # The work the taker carries called the app before the handover, so its failure leaves the outcome unknown.
        reading = Reading()
        reading.delegated("D1", "D2")
        reading.hand("D1", created("r1"), in_progress("r1"), function_call("get_menu"), completed("r1"))
        reading.hand("D2", created("r2"), queued("r2"))
        reading.hand("D1", created("r3"), in_progress("r3"))  # the app answered: the work goes on
        reading.hand("D2", in_progress("r2", previous="r3"))
        reading.hand("D1", message(TEXTS["handover_line"], phase="commentary"), completed("r3"))
        reading.hand("D2", failed("r2"))
        self.assertEqual(self.notices(reading), [catalog_lines().unclear])

    async def test_an_end_with_no_answer_came_too_late_once_no_delegation_can_name_its_taker(self):
        reading = Reading()
        reading.delegated("D1", "D2")
        reading.hand("D2", created("r2"), queued("r2"))
        reading.hand("D1", created("r1"), in_progress("r1"), completed("r1"))
        self.assertEqual(self.notices(reading), [])  # D2 may still name it
        reading.hand("D2", in_progress("r2"))  # D2 started without naming it: no taker can come now
        self.assertEqual(self.notices(reading), [catalog_lines().unclear])
        self.assertIn(("ended", "D1", "late"), reading.display.calls)

    async def test_a_typed_run_is_a_delegation_the_frontend_did_not_start(self):
        reading = Reading()
        reading.hand("typed_1", created("r1"), in_progress("r1"))
        self.assertIn('"a typed request"', reading.note.line())
        self.assertEqual(reading.display.calls[0], ("started", "typed_1", ""))

    async def test_events_after_close_are_ignored(self):
        reading = Reading()
        reading.delegated("D1")
        reading.reader.close()
        reading.hand("D1", created("r1"), in_progress("r1"), message("Done."), completed("r1"))
        self.assertEqual(reading.results.calls, [])


class AppWaitTests(unittest.IsolatedAsyncioTestCase):
    """The app wait runs on its own clock: its line once, and the delegation stays watched."""

    async def test_the_app_wait_line_is_said_once_and_a_late_result_is_still_said(self):
        reading = Reading(app_wait_seconds=0.05)
        reading.delegated("D1")
        reading.hand("D1", created("r1"), in_progress("r1"), function_call("get_menu"), completed("r1"))
        await asyncio.sleep(0.15)
        notices = [call for call in reading.results.calls if call[0] == "notice"]
        self.assertEqual(notices, [("notice", "D1", catalog_lines().app_wait_line)])
        self.assertIn("D1", reading.note)  # still watched
        reading.hand("D1", created("r2"), queued("r2"), in_progress("r2"), message("Tea."), completed("r2"))
        self.assertEqual(reading.results.calls[-1], ("ready", "D1", "Tea."))

    async def test_a_continuation_before_the_wait_runs_out_stops_it(self):
        reading = Reading(app_wait_seconds=0.05)
        reading.delegated("D1")
        reading.hand("D1", created("r1"), in_progress("r1"), function_call("get_menu"), completed("r1"))
        reading.hand("D1", created("r2"))
        await asyncio.sleep(0.15)
        self.assertEqual([call for call in reading.results.calls if call[0] == "notice"], [])

    async def test_no_app_wait_when_it_is_off(self):
        reading = Reading(app_wait_seconds=None)
        reading.delegated("D1")
        reading.hand("D1", created("r1"), in_progress("r1"), function_call("get_menu"), completed("r1"))
        self.assertEqual(reading.reader.app_waits, {})
