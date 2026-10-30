# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102

"""With per-sentence speech, a cut line keeps only the sentences that finished playing."""

from examples.frontend_backend_live.cascade.heard_log import Heard
from examples.frontend_backend_live.cascade.utterances import Outcome
from examples.frontend_backend_live.models.decision import TalkerDecision
from tests.unit.frontend_backend_live.cascade.doubles import Call

LINE = "We open at seven. The kitchen closes at ten."


class SentenceCutTests(Call):
    async def per_sentence_call(self, script):
        await self.call(script, reports_playout=True)
        self.session.utterances.reports_sentences = True  # what the adapter passes for per-sentence speech
        self.playout = self.session.playout

    def agent_lines(self):
        return [entry.text for entry in self.session.heard_log.entries if entry.kind == Heard.AGENT]

    async def test_a_line_cut_before_its_first_sentence_finished_was_not_heard(self):
        await self.per_sentence_call([TalkerDecision("speak", LINE)])
        await self.turn("When do you open?")
        self.session.interrupted()
        self.playout.started()
        self.playout.cut()
        self.assertEqual(self.session.utterances.recent[-1].outcome, Outcome.NOT_HEARD)
        self.assertEqual(self.memory()[1:], [("user", "When do you open?")])
        self.assertEqual(self.agent_lines(), [])
