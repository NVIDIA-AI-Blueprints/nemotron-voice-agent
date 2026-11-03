# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102

"""With ``speech.heard_words: estimated``, the talker's memory keeps the words a cut line's audio likely carried."""

import unittest

from examples.frontend_backend_live.cascade.heard_log import Heard
from examples.frontend_backend_live.cascade.ports import ReadyResult
from examples.frontend_backend_live.cascade.settings import CascadeSettings
from examples.frontend_backend_live.cascade.utterances import EstimatedHeard, Line, LineKind, Outcome
from examples.frontend_backend_live.models.decision import TalkerDecision
from tests.unit.frontend_backend_live.cascade.doubles import PROMPTS, QUIET, Call

LINE = "We open at seven. The kitchen closes at ten."
FIRST, SECOND = "We open at seven.", "The kitchen closes at ten."  # FIRST: 17 characters
# The estimate, at the default 15 characters a second until a sentence of the call has played.
ESTIMATE = CascadeSettings.parse(
    {"delegation": {"progress_after_seconds": None}, "speech": {"heard_words": "estimated"}}
)


class EstimatedHeardTests(unittest.TestCase):
    def test_only_whole_words_of_the_text_the_audio_belongs_to_and_never_more_than_it(self):
        line, estimate = Line(1, LineKind.REPLY, LINE), EstimatedHeard(chars_per_second=15)
        for heard_ms, kept in ((100, ""), (200, "We"), (450, "We"), (500, "We open"), (60_000, FIRST)):
            with self.subTest(heard_ms=heard_ms):
                self.assertEqual(estimate.kept(line, "", heard_ms, FIRST), kept)  # never into the next sentence
        self.assertEqual(estimate.kept(line, FIRST, 60_000, SECOND), LINE)
        self.assertEqual(estimate.kept(line, "", 60_000, ""), LINE)  # a host that names none: the rest of the line

    def test_the_rate_is_the_calls_own_once_a_sentence_played(self):
        line, estimate = Line(1, LineKind.REPLY, LINE), EstimatedHeard(chars_per_second=15)
        self.assertEqual(estimate.kept(line, FIRST, 700, SECOND), f"{FIRST} The")  # 10.5 characters at 15 a second
        estimate.played(FIRST, 1000)  # 17 characters a second
        self.assertEqual(estimate.kept(line, FIRST, 700, SECOND), f"{FIRST} The kitchen")  # 11.9 characters


class MemoryOnlyTests(Call):
    async def cut_call(self, settings, *, sentences=True):
        await self.call([TalkerDecision("speak", LINE)], settings=settings, reports_playout=True)
        self.session.utterances.reports_sentences = sentences  # what the host passes for its speech mode
        await self.turn("When do you open?")
        self.session.interrupted()
        self.playout = self.session.playout

    def agent_lines(self):
        return [entry.text for entry in self.session.heard_log.entries if entry.kind == Heard.AGENT]

    async def test_the_memory_alone_keeps_the_words_of_the_cut_sentence_that_likely_played(self):
        await self.cut_call(ESTIMATE)
        self.playout.started()
        self.playout.played(FIRST, 1000)
        self.playout.cut(700, SECOND)
        self.assertEqual(self.memory()[-1], ("assistant", f"{FIRST} The kitchen{PROMPTS.cut_mark}"))
        self.assertEqual(self.agent_lines(), [f"{FIRST}{PROMPTS.cut_mark}"])  # what the next delegation carries
        line = self.session.utterances.recent[-1]
        self.assertEqual((line.outcome, line.heard), (Outcome.CUT, FIRST))

    async def test_as_recorded_the_memory_keeps_the_sentences_that_played(self):
        await self.cut_call(QUIET)
        self.playout.started()
        self.playout.played(FIRST, 1000)
        self.playout.cut(700, SECOND)
        self.assertEqual(self.memory()[-1], ("assistant", f"{FIRST}{PROMPTS.cut_mark}"))
        self.assertEqual(self.agent_lines(), [f"{FIRST}{PROMPTS.cut_mark}"])

    async def test_a_first_sentence_cut_keeps_its_words_at_the_configured_rate_and_is_still_not_heard(self):
        await self.cut_call(ESTIMATE)
        self.playout.started()
        self.playout.cut(1000, FIRST)  # 15 characters
        self.assertEqual(self.memory()[-1], ("assistant", "We open at" + PROMPTS.cut_mark))
        self.assertEqual(self.agent_lines(), [])
        self.assertEqual(self.session.utterances.recent[-1].outcome, Outcome.NOT_HEARD)

    async def test_a_stitched_line_keeps_the_estimate_in_the_memory_and_stays_whole_in_the_heard_log(self):
        await self.cut_call(ESTIMATE, sentences=False)
        self.playout.started()
        self.playout.cut(1000, LINE)  # stitched speech took the whole line
        self.assertEqual(self.memory()[-1], ("assistant", "We open at" + PROMPTS.cut_mark))
        self.assertEqual(self.agent_lines(), [f"{LINE}{PROMPTS.cut_mark}"])

    async def test_a_result_line_said_again_keeps_its_estimate_and_its_result_once(self):
        await self.call(settings=ESTIMATE, reports_playout=True)
        self.session.utterances.reports_sentences = True
        self.session.presenter.ready(ReadyResult("D1", "m1", "Latte added."))
        await self.said(1)
        self.session.interrupted()
        self.session.playout.started()
        self.session.playout.cut(400, "Latte added.")  # 6 characters
        self.assertEqual(self.memory()[-1], ("assistant", "Latte" + PROMPTS.cut_mark))
        self.assertNotIn(("user", PROMPTS.result_label + "Latte added."), self.memory())  # it goes back to be said
        await self.said(2)
        self.assertEqual(self.memory().count(("user", PROMPTS.result_label + "Latte added.")), 1)
        self.assertEqual(self.memory()[-1], ("assistant", "Latte added."))
