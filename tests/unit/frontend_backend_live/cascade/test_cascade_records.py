# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D103

"""Honest records: written when a line is handed to speech, corrected by playout."""

import asyncio

from examples.frontend_backend_live.cascade.ports import ReadyResult
from examples.frontend_backend_live.cascade.utterances import LineKind, Outcome
from examples.frontend_backend_live.models.decision import TalkerDecision
from tests.unit.frontend_backend_live.cascade.doubles import Call, settle


class PlayoutCall(Call):
    async def playout_call(self, script=(), **kwargs):
        await self.call(script, reports_playout=True, **kwargs)
        self.playout = self.session.playout

    def agent_lines(self):
        return [entry.text for entry in self.session.heard_log.entries if entry.kind.value == "Agent"]


class OutcomeTests(PlayoutCall):
    async def test_a_played_line_stays_as_said(self):
        await self.playout_call([TalkerDecision("speak", "We open at seven.")])
        await self.turn("When do you open?")
        self.playout.started()
        self.playout.played("We open at seven.")
        self.playout.finished()
        line = self.session.utterances.recent[-1]
        self.assertEqual((line.outcome, line.heard), (Outcome.HEARD, "We open at seven."))
        self.assertEqual(self.memory()[-1], ("assistant", "We open at seven."))
        self.assertFalse(self.session.floor.in_flight)

    async def test_a_cut_line_keeps_what_played_and_told_keeps_only_that(self):
        await self.playout_call([TalkerDecision("delegate", "Let me pull up your booking and check the dates.")])
        await self.turn("Change my booking.")
        self.session.interrupted()
        self.playout.started()
        self.playout.played("Let me pull up")
        self.playout.cut()
        self.assertEqual(self.memory()[-1], ("assistant", "Let me pull up [cut off]"))
        self.assertEqual(self.agent_lines(), ["Let me pull up [cut off]"])  # what the next delegation carries
        self.assertEqual(self.session.utterances.told_text("item_1"), "Let me pull up")

    async def test_a_line_that_started_with_no_words_reported_stays_cut_whole(self):
        # A stitched text-to-speech holds a line's text to the line's end, so a cut stitched line reports started()
        # and cut() with no played(): part of it was heard, so it is not taken back or said again.
        await self.playout_call([TalkerDecision("speak", "We open at seven.")])
        await self.turn("When do you open?")
        self.session.interrupted()
        self.playout.started()
        self.playout.cut()
        self.assertEqual(self.session.utterances.recent[-1].outcome, Outcome.CUT)
        self.assertEqual(self.memory()[-1], ("assistant", "We open at seven. [cut off]"))
        self.assertEqual(self.agent_lines(), ["We open at seven. [cut off]"])
        self.assertFalse(await self.session.utterances.resume())  # where it was cut is not known: nothing is repeated

    async def test_a_line_nobody_heard_leaves_the_records(self):
        await self.playout_call([TalkerDecision("speak", "We open at seven.")])
        await self.turn("When do you open?")
        self.assertEqual(self.agent_lines(), ["We open at seven."])
        self.session.interrupted()
        self.playout.cut()  # cut before any word played
        self.assertEqual(self.memory()[1:], [("user", "When do you open?")])
        self.assertEqual(self.agent_lines(), [])
        self.assertEqual(self.session.utterances.recent[-1].outcome, Outcome.NOT_HEARD)

    async def test_a_result_line_nobody_heard_is_requeued_and_said_at_the_next_idle(self):
        await self.playout_call()
        self.session.presenter.ready(ReadyResult("D1", "m1", "Latte added."))
        await self.said(1)
        self.assertIn(("user", "Backend result (reference data): Latte added."), self.memory())
        self.session.interrupted()
        self.playout.cut()
        self.assertNotIn(("user", "Backend result (reference data): Latte added."), self.memory())
        await self.said(2)
        self.assertEqual(self.spoken, ["Latte added.", "Latte added."])
        self.assertEqual(self.memory().count(("user", "Backend result (reference data): Latte added.")), 1)

    async def test_a_cut_settles_only_the_lines_handed_before_the_interruption(self):
        await self.playout_call([TalkerDecision("speak", "First line."), TalkerDecision("speak", "Second line.")])
        await self.turn("One.")
        self.session.interrupted()  # the interruption passes the binding
        await self.turn("Two.")  # the reply to the new turn is handed after it
        self.playout.cut()  # the tracker reports the cut at the output
        first, second = list(self.session.utterances.recent)[-2:]
        self.assertEqual((first.text, first.outcome), ("First line.", Outcome.NOT_HEARD))
        self.assertEqual((second.text, second.outcome), ("Second line.", Outcome.HANDED))
        self.assertEqual(self.memory()[-1], ("assistant", "Second line."))

    async def test_a_cut_settles_no_line_handed_after_its_interruption_even_with_none_before(self):
        # An instruction interrupts with nothing playing, and its reply is handed before the interruption reaches the
        # output: the cut is that interruption's, so the reply stays, in the talker's memory too.
        await self.playout_call([TalkerDecision("speak", "Welcome back, Sam.")])
        self.session.interrupted()  # the instruction's interruption passes the binding
        await self.turn("Greet Sam.")  # its reply is handed after it
        self.playout.cut()  # the tracker reports that interruption at the output
        line = self.session.utterances.recent[-1]
        self.assertEqual((line.text, line.outcome), ("Welcome back, Sam.", Outcome.HANDED))
        self.assertEqual(self.memory()[-1], ("assistant", "Welcome back, Sam."))

    async def test_a_cut_that_did_not_pass_the_binding_settles_every_line_in_flight(self):
        await self.playout_call([TalkerDecision("speak", "We open at seven.")])
        await self.turn("When do you open?")
        self.playout.cut()  # an interruption made after the binding, which never counted it
        self.assertEqual(self.session.utterances.recent[-1].outcome, Outcome.NOT_HEARD)

    async def test_without_a_tracker_a_line_is_heard_when_its_hand_off_returns(self):
        await self.call([TalkerDecision("speak", "Hi.")])
        await self.turn("Hello.")
        self.assertEqual(self.session.utterances.recent[-1].outcome, Outcome.HEARD)
        self.assertEqual(self.session.floor.in_flight, 0)


class SilenceTests(PlayoutCall):
    async def test_a_line_that_never_starts_counts_as_silent_and_is_said_once_more(self):
        from examples.frontend_backend_live.cascade.settings import CascadeSettings

        settings = CascadeSettings.parse(
            {"delegation": {"progress_after_seconds": None}, "speech": {"start_timeout_seconds": 0.05}}
        )
        await self.playout_call([TalkerDecision("speak", "Hello there.")], settings=settings)
        await self.turn("Hi.")
        await settle(lambda: len(self.spoken) == 2)
        self.assertEqual(self.spoken, ["Hello there.", "Hello there."])
        self.assertIn("line.start_timeout", self.kinds())
        self.assertEqual(self.memory().count(("assistant", "Hello there.")), 1)  # the first records were taken back
        await asyncio.sleep(0.15)  # the second never starts either: not heard, and not said a third time
        self.assertEqual(len(self.spoken), 2)
        self.assertNotIn(("assistant", "Hello there."), self.memory())

    async def test_a_silent_result_line_is_said_once_more_then_requeued(self):
        await self.playout_call()
        self.session.presenter.ready(ReadyResult("D1", "m1", "Latte added."))
        await self.said(1)
        self.playout.silent()
        await self.said(2)
        retried = self.session.utterances.recent[-1]
        self.assertTrue(retried.retried)
        self.assertEqual(retried.kind, LineKind.RESULT)
