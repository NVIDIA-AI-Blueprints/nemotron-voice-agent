# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D103

"""Status from progress notes, and the progress line the writer words after a quiet spell."""

import asyncio

from examples.frontend_backend_live.cascade.settings import CascadeSettings
from examples.frontend_backend_live.models.decision import TalkerDecision
from tests.unit.frontend_backend_live.cascade.doubles import (
    PROMPTS,
    Call,
    completed,
    created,
    in_progress,
    mcp_call,
    message,
    progress_update,
    runtime_data,
    settle,
)

SEARCHING = "Searching flights."  # the backend's newest progress note
LINE = "I'm going through the flights now."  # the progress line the writer words


def quiet(after=0.1, cap=1, **speech):
    delegation = {"progress_after_seconds": after, "progress_max_per_delegation": cap}
    return CascadeSettings.parse({"delegation": delegation, "speech": speech})


class StatusTests(Call):
    async def test_the_newest_progress_line_wins_in_the_work_line(self):
        await self.call([TalkerDecision("delegate", "One moment."), TalkerDecision("speak", "Checking flights now.")])
        await self.turn("Change my flight.")
        self.emit("item_1", created("r1"), in_progress("r1"))
        self.emit("item_1", message("Looking up your booking.", phase="commentary", item_id="c1"))
        self.emit("item_1", message("Searching flights.", phase="commentary", item_id="c2"))
        await self.turn("Any updates?")  # the talker reads the work line and answers
        work = runtime_data(self.frontend.calls[1])
        self.assertIn(progress_update("Searching flights."), work)
        self.assertNotIn("Looking up your booking.", work)
        self.assertEqual(self.spoken[-1], "Checking flights now.")

    async def start_work(self, *script, settings=None, reports_playout=False):
        """The caller asks for a flight change, its work runs, and the backend's newest note is ``SEARCHING``.

        ``script`` follows the talker's delegation: the writer's progress line first.
        """
        await self.call(
            [TalkerDecision("delegate", "One moment."), *script],
            settings=settings or quiet(),
            reports_playout=reports_playout,
        )
        await self.turn("Change my flight.")
        if reports_playout:
            self.session.playout.started()  # the lead-in plays out
            self.session.playout.finished()
        self.emit("item_1", created("r1"), in_progress("r1"))
        self.emit("item_1", message(SEARCHING, phase="commentary", item_id="c1"))

    async def test_after_a_quiet_spell_the_writer_words_one_line_from_the_work_note(self):
        # The note's facts and what the caller heard go in the update; the backend's note itself is never said.
        await self.start_work(TalkerDecision("speak", LINE))
        await self.said(2, timeout=3)
        await asyncio.sleep(0.4)  # several quiet spells: the cap of one holds
        self.assertEqual(self.spoken, ["One moment.", LINE])
        call = self.frontend.calls[1]
        self.assertEqual((call["role"], call["actions"]), ("writer", ("speak",)))
        self.assertIn(PROMPTS.progress, call["instructions"])
        update = call["history"][-1]["content"]
        for fact in ('"Change my flight."', progress_update(SEARCHING), 'the caller already heard: "One moment."'):
            with self.subTest(fact=fact):
                self.assertIn(fact, update)
        self.assertEqual(self.session.utterances.told_text("item_1"), f"One moment. {LINE}")
        self.assertIn("progress.said", self.kinds())

    async def test_a_progress_line_that_cannot_be_worded_says_nothing(self):
        # A failed call, unusable output, or the backend's note given back as written: no fixed line takes its place.
        for step in (RuntimeError("model down"), TalkerDecision("delegate", ""), TalkerDecision("speak", SEARCHING)):
            with self.subTest(step=repr(step)):
                await self.start_work(step)
                await settle(lambda: "progress.silent" in self.kinds(), timeout=3)
                await asyncio.sleep(0.2)
                self.assertEqual(self.spoken, ["One moment."])
                self.assertIn("progress.recovered", self.kinds())
                self.assertFalse(self.session.worknote.entries["item_1"].news)  # no retry at the next quiet spell

    async def test_no_words_leave_the_quiet_spell_silent(self):
        await self.start_work(TalkerDecision("speak", ""))
        await settle(lambda: "progress.silent" in self.kinds(), timeout=3)
        await asyncio.sleep(0.2)
        self.assertEqual(self.spoken, ["One moment."])
        self.assertNotIn("progress.recovered", self.kinds())

    async def test_a_result_that_comes_while_the_line_is_worded_drops_it(self):
        pending = asyncio.get_running_loop().create_future()
        await self.start_work(pending, TalkerDecision("speak", "Your flight is changed."))
        await settle(lambda: len(self.frontend.calls) == 2, timeout=3)  # the progress line is being worded
        self.emit("item_1", message("Changed to the nine o'clock flight."), completed("r1"))
        await self.said(2, timeout=3)
        self.assertEqual(self.spoken, ["One moment.", "Your flight is changed."])
        self.assertIn("progress.dropped", self.kinds())
        self.assertTrue(pending.cancelled())

    async def test_the_caller_speaking_while_the_line_is_worded_drops_it_and_its_news_waits(self):
        pending = asyncio.get_running_loop().create_future()
        await self.start_work(pending)
        await settle(lambda: len(self.frontend.calls) == 2, timeout=3)
        self.session.caller_started()
        await settle(lambda: "progress.dropped" in self.kinds(), timeout=3)
        self.assertTrue(pending.cancelled())
        entry = self.session.worknote.entries["item_1"]
        self.assertEqual((entry.news, entry.lines_said), (True, 1))  # the offer counts; its news is for the next spell
        self.assertEqual(self.spoken, ["One moment."])

    async def test_a_later_line_needs_a_backend_step_that_ended(self):
        await self.start_work(
            TalkerDecision("speak", LINE), TalkerDecision("speak", "Now on to your seat."), settings=quiet(cap=3)
        )
        await self.said(2, timeout=3)
        await asyncio.sleep(0.3)  # quiet spells with nothing new: no offer
        self.assertEqual(len(self.frontend.calls), 2)
        self.emit("item_1", mcp_call("airline", "search_flights"))
        await self.said(3, timeout=3)
        self.assertEqual(self.spoken, ["One moment.", LINE, "Now on to your seat."])

    async def test_a_progress_line_cut_before_its_audio_is_not_resumed_and_the_held_result_plays(self):
        result = TalkerDecision("speak", "Your flight is changed.")
        await self.start_work(TalkerDecision("speak", LINE), result, result, reports_playout=True)
        await self.said(2, timeout=3)  # handed, not playing yet
        self.emit("item_1", message("Changed to the nine o'clock flight."), completed("r1"))  # held behind it
        self.session.caller_started()  # a cough cuts the line before any audio
        self.session.interrupted()
        self.session.playout.cut()
        self.session.caller_stopped()
        await self.session.on_turn()  # a turn with no words: nothing to resume
        await self.said(3, timeout=3)
        self.assertEqual(self.spoken, ["One moment.", LINE, "Your flight is changed."])
        self.assertNotIn("line.resumed", self.kinds())

    async def test_a_progress_line_that_never_starts_is_not_said_again(self):
        await self.start_work(
            TalkerDecision("speak", LINE), settings=quiet(start_timeout_seconds=0.05), reports_playout=True
        )
        await self.said(2, timeout=3)
        await asyncio.sleep(0.2)  # it never starts: written off, never said once more
        self.assertEqual(self.spoken, ["One moment.", LINE])
        self.assertIn("line.start_timeout", self.kinds())
        self.assertNotIn("line.retried", self.kinds())

    async def test_no_line_is_offered_while_the_caller_speaks_or_when_it_is_off(self):
        await self.call([TalkerDecision("delegate", "One moment.")], settings=quiet())
        await self.turn("Change my flight.")
        self.session.caller_started()
        await asyncio.sleep(0.3)
        self.assertEqual(self.spoken, ["One moment."])
        await self.call([TalkerDecision("delegate", "One moment.")], settings=quiet(after=None))
        await self.turn("Change my flight.")
        await asyncio.sleep(0.2)
        self.assertEqual(self.spoken, ["One moment."])
