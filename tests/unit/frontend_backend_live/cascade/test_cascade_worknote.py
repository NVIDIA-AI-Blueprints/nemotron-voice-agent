# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D103, D107

"""The work note: one record of open work, tracked by capability."""

import asyncio
import unittest

from examples.frontend_backend_live.cascade.settings import CascadeSettings
from examples.frontend_backend_live.cascade.worknote import WorkNote
from examples.frontend_backend_live.models.decision import TalkerDecision
from tests.unit.contracts.delegation_port import ScriptedDelegationPort
from tests.unit.frontend_backend_live.cascade.doubles import (
    ANSWER_READY_LABEL,
    NO_WORK_IN_PROGRESS,
    PROMPTS,
    WORK,
    WORK_IN_PROGRESS,
    Call,
    answered_events,
    completed,
    created,
    function_call,
    in_progress,
    message,
    progress_update,
    prompt_set,
    queued,
    running_state,
    runtime_data,
    settle,
    work_item,
)


class Clock:
    def __init__(self):
        self.now = 100.0

    def __call__(self):
        return self.now


def note(tracks=True, clock=None):
    return WorkNote(prompt_set(), tracks=tracks, clock=clock or Clock(), observe=lambda kind, **fields: None)


class EntryTests(unittest.TestCase):
    def test_a_port_that_reports_no_events_gets_no_entries_and_no_line(self):
        untracked = note(tracks=False)
        self.assertIsNone(untracked.reserve("Book a table."))
        untracked.typed("item_1")
        self.assertEqual((untracked.open(), untracked.line()), ([], ""))

    def test_the_states_waiting_running_said_next_and_gone(self):
        clock = Clock()
        work = note(clock=clock)
        work.confirm(work.reserve("Teas?"), "D1")
        self.assertIn(work_item(1, WORK.waiting, "Teas?"), work.line())
        work.running("D1")
        clock.now += 12
        work.call_started("D1", "r1", "mcp_1", "get_menu")
        work.progress("D1", "Checking the menu.")
        running = running_state(12, "get menu")
        self.assertIn(work_item(1, running, "Teas?", progress_update("Checking the menu.")), work.line())
        work.answered("D1")
        self.assertIn(work_item(1, ANSWER_READY_LABEL, "Teas?"), work.line())
        work.ended("D1")
        self.assertIn("D1", work)  # it stays until its result is said
        work.said("D1")
        self.assertEqual(work.open(), [])

    def test_a_call_reads_as_running_until_it_ends_then_with_the_official_status_it_ended_with(self):
        work = note()
        work.confirm(work.reserve("Change my flight."), "D1")
        work.running("D1")
        work.call_started("D1", "r1", "mcp_1", "get_user_details")
        work.call_started("D1", "r1", "mcp_2", "get_reservation_details")
        both = running_state(0, "get user details, get reservation details")
        self.assertIn(work_item(1, both, "Change my flight."), work.line())
        work.call_ended("D1", "r1", "mcp_1", "get_user_details", "failed")  # a failed call never reads as done
        one = running_state(0, "get reservation details", ended=[("get user details", "failed")])
        self.assertIn(work_item(1, one, "Change my flight."), work.line())
        work.call_ended("D1", "r1", "mcp_2", "get_reservation_details", "completed")
        none = running_state(0, ended=[("get user details", "failed"), ("get reservation details", "completed")])
        self.assertIn(work_item(1, none, "Change my flight."), work.line())  # nothing runs now
        work.call_started("D1", "r2", "mcp_3", "search_flights")  # the next response's calls start anew
        self.assertIn(work_item(1, running_state(0, "search flights"), "Change my flight."), work.line())

    def test_a_join_keeps_the_older_start_and_a_continuation_clears_the_step(self):
        # After a join the taker carries the older delegation's start, not its own; once the app answered, the tool
        # its call ran leaves the running state.
        clock = Clock()
        work = note(clock=clock)
        work.confirm(work.reserve("Teas?"), "D1")
        work.running("D1")
        work.call_started("D1", "r1", "fc_1", "get_menu")  # the app's call: it runs until the app answers
        work.waits_for_app("D1", True)
        self.assertIn(WORK.steps.format(steps="get menu"), work.line())
        clock.now += 30
        work.waits_for_app("D1", False)  # the app's outputs came: its continuation runs
        self.assertNotIn("get menu", work.line())
        work.confirm(work.reserve("Green only."), "D2")
        clock.now += 5
        work.running("D2")
        work.hand_over("D1", "D2")
        self.assertIn(work_item(1, running_state(35), "Teas? / Green only."), work.line())

    def test_a_confirmed_reservation_keeps_its_place(self):
        work = note()
        first = work.reserve("Teas?")
        work.confirm(work.reserve("Cakes?"), "D2")
        work.confirm(first, "D1")
        self.assertEqual([entry.id for entry in work.open()], ["D1", "D2"])

    def test_the_newer_requests_are_the_open_ones_made_after_the_oldest_answered(self):
        work = note()
        for delegation_id, request in (("D1", "Teas?"), ("D2", "Oat milk?"), ("D3", "Soy milk?")):
            work.confirm(work.reserve(request), delegation_id)
        work.answered("D3")
        self.assertEqual(work.newer_requests(["D1"]), ("Oat milk?",))  # D3's answer is said next: not in progress
        self.assertEqual(work.newer_requests(["D2", "D1"]), ())

    def test_each_joined_request_is_shortened_on_its_own(self):
        work = note()
        older = "Move my Tuesday dentist visit to the earliest morning slot next week, and keep the same dentist, " * 2
        work.confirm(work.reserve(older), "D1")
        work.confirm(work.reserve("Oh, and add a reminder the day before."), "D2")
        work.hand_over("D1", "D2")
        self.assertIn('... / Oh, and add a reminder the day before."', work.line())
        self.assertEqual(work.entries["D2"].request, f"{older} / Oh, and add a reminder the day before.")


class ProgressOfferTests(unittest.TestCase):
    def running(self):
        work = note()
        work.confirm(work.reserve("Change my flight."), "D1")
        work.running("D1")
        return work

    def test_an_offer_counts_against_the_cap_and_gives_the_entrys_line_alone(self):
        work = self.running()
        work.confirm(work.reserve("And my seat."), "D2")
        self.assertIs(work.quiet_candidate(1), work.entries["D1"])  # new work has news
        line = WORK_IN_PROGRESS.format(items=work_item(1, running_state(0), "Change my flight."))
        self.assertEqual(work.offer("D1"), line)
        self.assertEqual(work.entries["D1"].lines_said, 1)
        self.assertIs(work.quiet_candidate(1), work.entries["D2"])  # D1 reached its cap

    def test_a_backend_step_that_ended_or_a_new_progress_line_is_news_and_an_app_call_is_not(self):
        work = self.running()
        work.worded("D1")
        work.call_started("D1", "r1", "fc_1", "lookup_order")  # the app's function call
        work.waits_for_app("D1", True)
        work.waits_for_app("D1", False)  # its outputs came
        self.assertIsNone(work.quiet_candidate(3))
        work.call_ended("D1", "r2", "mcp_1", "search_flights", "completed")
        self.assertIs(work.quiet_candidate(3), work.entries["D1"])
        work.worded("D1")
        work.progress("D1", "Searching flights.")
        self.assertIs(work.quiet_candidate(3), work.entries["D1"])

    def test_quiet_reads_whether_the_delegation_is_open_and_may_be_spoken_of(self):
        work = self.running()
        self.assertTrue(work.quiet("D1"))
        work.answered("D1")  # said next
        self.assertFalse(work.quiet("D1"))
        self.assertFalse(work.quiet("D2"))  # not open


class AppWaitCallTests(Call):
    async def test_a_short_app_wait_queues_its_line_once_and_a_late_result_is_still_said(self):
        settings = CascadeSettings.parse({"delegation": {"app_wait_seconds": 0.05, "progress_after_seconds": None}})
        await self.call(
            [TalkerDecision("delegate", "Let me check."), TalkerDecision("speak", "Earl Grey it is.")],
            settings=settings,
        )
        await self.turn("Teas?")
        self.emit("item_1", created("r1"), queued("r1"), in_progress("r1"), function_call("get_menu"), completed("r1"))
        await self.said(2)
        await asyncio.sleep(0.1)
        self.assertEqual(self.spoken, ["Let me check.", self.lines.app_wait_line])
        self.assertIn("item_1", self.session.worknote)
        self.emit("item_1", *answered_events("r2", "Earl Grey."))
        await self.said(3)
        self.assertEqual(self.spoken[-1], "Earl Grey it is.")

    async def test_the_talker_never_answers_a_result_about_to_be_said(self):
        port = ScriptedDelegationPort()
        await self.call(
            [TalkerDecision("delegate", "One moment."), TalkerDecision("speak", "One moment more.")], port=port
        )
        await self.turn("Teas?")
        self.session.caller_started()  # the caller speaks while the answer arrives: it waits, said next
        self.emit("item_1", created("r1"), in_progress("r1"), message("Earl Grey."), completed("r1"))
        self.buffer.turn("Well?")
        self.session.caller_stopped()
        await self.session.on_turn()
        self.assertIn('answer ready, said next: "Teas?"', runtime_data(self.frontend.calls[1]))


AS_WRITTEN = CascadeSettings.parse({"delegation": {"result_wording": "as_written", "progress_after_seconds": None}})
READY = work_item(1, ANSWER_READY_LABEL, "Teas?")


class HeardAnswerTests(Call):
    """An answered delegation leaves the note once the caller hears its answer, not when it is handed to speech."""

    async def asked(self, *script, say=None):
        """The caller asks for teas and the lead-in plays; the talker then decides ``script``."""
        await self.call(
            [TalkerDecision("delegate", "Let me check."), *script], reports_playout=True, settings=AS_WRITTEN, say=say
        )
        self.playout = self.session.playout
        await self.turn("Teas?")
        self.playout.started()
        self.playout.finished()

    async def answer_handed(self, *script):
        """The delegation answers and ends; its answer is handed to speech and has not played yet."""
        await self.asked(*script)
        self.emit("item_1", *answered_events("r1", "Earl Grey is in stock. Mint is not."))
        await self.said(2)

    def heard(self):
        """The line in progress plays to its end."""
        self.playout.started()
        self.playout.finished()

    async def test_handed_then_heard(self):
        await self.answer_handed()
        self.assertIn(READY, self.session.worknote.line())
        self.heard()
        self.assertEqual(self.session.worknote.line(), NO_WORK_IN_PROGRESS)

    async def test_handed_then_not_heard_then_requeued_then_heard(self):
        await self.answer_handed()
        self.session.caller_started()
        self.session.interrupted()
        self.playout.cut()  # cut before a word of it played: nobody heard it, and it is requeued
        self.assertIn(READY, self.session.worknote.line())
        self.session.caller_stopped()
        await self.said(3)  # said again at idle
        self.assertIn(READY, self.session.worknote.line())
        self.heard()
        self.assertEqual(self.session.worknote.line(), NO_WORK_IN_PROGRESS)

    async def test_handed_then_cut_after_a_sentence_leaves_and_is_not_said_again(self):
        await self.answer_handed()
        self.session.utterances.reports_sentences = True  # per-sentence speech
        self.session.interrupted()
        self.playout.started()
        self.playout.played("Earl Grey is in stock.")
        self.assertIn(READY, self.session.worknote.line())  # playing
        self.playout.cut()
        self.assertEqual(self.session.worknote.line(), NO_WORK_IN_PROGRESS)
        await asyncio.sleep(0.05)
        self.assertEqual(len(self.spoken), 2)

    async def test_a_delegation_with_two_results_leaves_once_both_are_heard(self):
        await self.asked()
        self.emit("item_1", created("r1"), queued("r1"), in_progress("r1"), message("Earl Grey.", item_id="m1"))
        await self.said(2)  # the first result plays
        self.assertIsNone(self.session.worknote.quiet_candidate(1))  # no progress line for work whose answer is ready
        self.emit("item_1", message("Mint.", item_id="m2"), completed("r1"))
        self.heard()
        self.assertIn(READY, self.session.worknote.line())  # the second waits to be said
        await self.said(3)
        self.assertEqual(self.spoken[1:], ["Earl Grey.", "Mint."])
        self.assertIn(READY, self.session.worknote.line())
        self.heard()
        self.assertEqual(self.session.worknote.line(), NO_WORK_IN_PROGRESS)

    async def answer_fails(self, times):
        """The caller asks, then the speech output raises on the answer's first ``times`` deliveries."""
        left = [times]

        async def say(text):
            if self.spoken and left[0]:  # every line after the lead-in
                left[0] -= 1
                raise RuntimeError("the speech service failed")
            self.spoken.append(text)

        await self.asked(say=say)
        self.emit("item_1", *answered_events("r1", "Earl Grey is in stock."))

    async def test_a_delivery_that_raises_keeps_the_answer_ready_until_heard(self):
        await self.answer_fails(1)
        await self.said(2)  # requeued once, then handed
        self.assertIn(READY, self.session.worknote.line())
        self.heard()
        self.assertEqual(self.session.worknote.line(), NO_WORK_IN_PROGRESS)

    async def test_a_delivery_that_raises_twice_drops_the_answer_from_the_note(self):
        await self.answer_fails(2)
        await settle(lambda: self.kinds().count("answer.failed") == 2)
        self.assertEqual(self.session.worknote.line(), NO_WORK_IN_PROGRESS)

    async def test_an_answer_that_joined_a_delivery_that_raises_goes_back_with_it(self):
        raised = []

        async def say(text):
            if self.spoken and not raised:  # the first line after the lead-in raises once
                raised.append(text)
                raise RuntimeError("the speech service failed")
            self.spoken.append(text)

        await self.call(
            [TalkerDecision("delegate", "Let me check.")], reports_playout=True, settings=AS_WRITTEN, say=say
        )
        self.playout = self.session.playout
        await self.turn("Teas?")  # the lead-in is in flight: an answer waits for idle
        self.emit("item_1", *answered_events("r1", "Earl Grey."))
        await settle(lambda: not self.session.presenter.inbox)  # taken, waiting for idle
        self.emit("item_2", *answered_events("r2", "Mint."))  # another delegation answers meanwhile
        self.heard()  # the lead-in played: both answers are worded into one line, whose delivery raises
        await self.said(2)
        self.assertEqual((raised, self.spoken[1]), (["Earl Grey. Mint."], "Earl Grey. Mint."))
        self.heard()
        self.assertEqual(self.session.worknote.line(), NO_WORK_IN_PROGRESS)

    async def test_a_turn_decided_while_the_answer_plays_reads_it_as_answer_ready(self):
        await self.answer_handed(TalkerDecision("wait"))
        await self.turn("Okay.")
        self.assertIn(READY, runtime_data(self.frontend.calls[1]))

    async def test_after_the_caller_talks_over_the_answer_okay_reads_it_as_answer_ready(self):
        await self.answer_handed(TalkerDecision("wait"))
        self.session.utterances.reports_sentences = True  # per-sentence speech
        self.session.caller_started()  # the caller talks over the answer before its first sentence finished
        self.session.interrupted()
        self.playout.started()
        self.playout.cut()
        self.buffer.turn("Okay.")
        self.session.caller_stopped()
        await self.session.on_turn()
        self.assertIn(READY, runtime_data(self.frontend.calls[1]))
        await self.said(3)  # the answer is said next, and no new work starts
        self.assertEqual(self.spoken[-1], "Earl Grey is in stock. Mint is not.")
        self.assertEqual(len(self.port.requests), 1)

    async def test_a_delegation_sent_while_the_answer_waits_tells_the_backend_it_is_not_heard(self):
        # The caller's "Okay." came before the answer played: the backend must not read it as a reply to it.
        unheard = PROMPTS.transcript.unheard
        await self.answer_handed(TalkerDecision("delegate"), TalkerDecision("delegate"))
        await self.turn("Okay.")
        self.assertEqual(self.port.requests[1][0]["content"][0]["text"].splitlines()[-1], unheard)
        self.heard()  # the answer played: nothing waits to be heard
        await self.turn("Yes, the mint.")
        self.assertNotIn(unheard, self.port.requests[2][0]["content"][0]["text"])
        self.assertNotIn(unheard, self.port.requests[0][0]["content"][0]["text"])
