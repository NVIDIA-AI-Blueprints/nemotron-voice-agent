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
    NO_WORK_IN_PROGRESS,
    Call,
    answered_events,
    completed,
    created,
    function_call,
    in_progress,
    message,
    prompt_set,
    queued,
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

    def test_a_port_that_reports_events_reads_none_with_nothing_open(self):
        self.assertEqual(note().line(), NO_WORK_IN_PROGRESS)

    def test_the_states_waiting_running_said_next_and_gone(self):
        clock = Clock()
        work = note(clock=clock)
        work.confirm(work.reserve("Teas?"), "D1")
        self.assertIn('1. waiting to start: "Teas?"', work.line())
        work.running("D1")
        clock.now += 12
        work.called("D1", "r1", "get_menu")
        work.progress("D1", "Checking the menu.")
        self.assertIn(
            '1. running for 12 s, now using get menu: "Teas?"; latest update: "Checking the menu."', work.line()
        )
        work.answered("D1")
        self.assertIn('1. answer ready, said next: "Teas?"', work.line())
        work.ended("D1")
        self.assertIn("D1", work)  # it stays until its result is said
        work.said("D1")
        self.assertEqual(work.open(), [])

    def test_a_join_keeps_the_older_start_and_a_continuation_clears_the_step(self):
        # After a join the taker carries the older delegation's start, not "running for 0 s"; once the app answered,
        # the call it ran leaves "now using".
        clock = Clock()
        work = note(clock=clock)
        work.confirm(work.reserve("Teas?"), "D1")
        work.running("D1")
        work.called("D1", "r1", "get_menu")
        work.waits_for_app("D1", True)
        self.assertIn("now using get menu", work.line())
        clock.now += 30
        work.waits_for_app("D1", False)  # the app's outputs came: its continuation runs
        self.assertNotIn("get menu", work.line())
        work.confirm(work.reserve("Green only."), "D2")
        clock.now += 5
        work.running("D2")
        work.hand_over("D1", "D2")
        self.assertIn('1. running for 35 s: "Teas? / Green only."', work.line())

    def test_a_reservation_released_on_refusal_leaves_no_entry(self):
        work = note()
        reservation = work.reserve("Teas?")
        work.release(reservation)
        self.assertEqual(work.open(), [])

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
        older = "Change my Houston to Denver return to the fastest flight on May twenty seventh, stay in economy, " * 2
        work.confirm(work.reserve(older), "D1")
        work.confirm(work.reserve("Oh, and add a checked bag."), "D2")
        work.hand_over("D1", "D2")
        self.assertIn('... / Oh, and add a checked bag."', work.line())
        self.assertEqual(work.entries["D2"].request, f"{older} / Oh, and add a checked bag.")


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
        self.assertIn('answer ready, said next: "Teas?"', self.frontend.calls[1]["instructions"])
