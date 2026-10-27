# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102

"""With per-sentence speech, a cut line keeps only the sentences that finished playing."""

from examples.frontend_backend_live.adapter.services import PerSentence, Stitched
from examples.frontend_backend_live.cascade.ports import ReadyResult
from examples.frontend_backend_live.cascade.utterances import Outcome
from examples.frontend_backend_live.models.decision import TalkerDecision
from tests.unit.frontend_backend_live.cascade.doubles import Call

LINE = "We open at seven. The kitchen closes at ten."


class SentenceCutTests(Call):
    async def per_sentence_call(self, script=()):
        await self.call(script, reports_playout=True)
        self.session.utterances.reports_sentences = True  # what the adapter passes for per-sentence speech
        self.playout = self.session.playout

    def agent_lines(self):
        return [entry.text for entry in self.session.heard_log.entries if entry.kind.value == "Agent"]

    async def test_a_line_cut_before_its_first_sentence_finished_was_not_heard(self):
        await self.per_sentence_call([TalkerDecision("speak", LINE)])
        await self.turn("When do you open?")
        self.session.interrupted()
        self.playout.started()
        self.playout.cut()
        self.assertEqual(self.session.utterances.recent[-1].outcome, Outcome.NOT_HEARD)
        self.assertEqual(self.memory()[1:], [("user", "When do you open?")])
        self.assertEqual(self.agent_lines(), [])

    async def test_a_line_cut_after_a_sentence_keeps_that_sentence(self):
        await self.per_sentence_call([TalkerDecision("speak", LINE)])
        await self.turn("When do you open?")
        self.session.interrupted()
        self.playout.started()
        self.playout.played("We open at seven.")
        self.playout.cut()
        self.assertEqual(self.agent_lines(), ["We open at seven. [cut off]"])

    async def test_a_result_line_cut_before_its_first_sentence_finished_is_said_again(self):
        await self.per_sentence_call()
        self.session.presenter.ready(ReadyResult("D1", "m1", "Latte added."))
        await self.said(1)
        self.session.interrupted()
        self.playout.started()
        self.playout.cut()
        await self.said(2)
        self.assertEqual(self.spoken, ["Latte added.", "Latte added."])

    def test_only_per_sentence_speech_reports_sentences(self):
        self.assertTrue(PerSentence.reports_sentences)
        self.assertFalse(Stitched.reports_sentences)
