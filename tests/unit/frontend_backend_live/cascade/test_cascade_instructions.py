# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D103

"""Instructions decided in turn."""

from examples.frontend_backend_live.cascade.settings import CascadeSettings
from examples.frontend_backend_live.cascade.utterances import Outcome
from examples.frontend_backend_live.models.decision import TalkerDecision
from tests.unit.frontend_backend_live.cascade.doubles import PROMPTS, Call


class Waits:
    """A talker that answers ``wait`` to everything."""

    async def decide(self, instructions, history, work_in_progress="", *, work_open=False):
        return TalkerDecision("wait")


class InstructionTests(Call):
    async def test_stop_speaking_in_the_middle_of_a_line_gives_silence(self):
        cuts = []

        async def interrupt():
            cuts.append("interrupted")
            self.session.interrupted()

        await self.call([TalkerDecision("speak", "Your flight leaves at nine from gate twelve.")], reports_playout=True)
        self.session.application.interrupt = interrupt
        await self.turn("When is my flight?")
        self.session.playout.started()
        self.session.playout.played("Your flight")
        self.session.agent.talker = Waits()
        await self.session.on_instruction("Stop speaking immediately.")
        self.session.playout.cut()  # the tracker reports the cut the interruption made
        self.assertEqual(cuts, ["interrupted"])
        self.assertEqual(self.spoken, ["Your flight leaves at nine from gate twelve."])  # nothing more is said
        self.assertEqual(self.session.utterances.recent[-1].outcome, Outcome.CUT)
        self.assertEqual(
            self.memory()[-2:],
            [("assistant", "Your flight" + PROMPTS.cut_mark), ("developer", "Stop speaking immediately.")],
        )

    async def test_a_turn_with_no_words_after_an_instructions_cut_resumes_nothing(self):
        # "Stop speaking immediately" stays stopped: a cough afterwards is no reason to say the rest of the line,
        # whether the tracker's cut lands before the instruction goes on or after it.
        for tracker_first in (False, True):
            with self.subTest(tracker_first=tracker_first):

                async def interrupt(tracker_first=tracker_first):
                    self.session.interrupted()
                    if tracker_first:
                        self.session.playout.cut()

                await self.call(
                    [TalkerDecision("speak", "Your flight leaves at nine from gate twelve.")], reports_playout=True
                )
                self.session.application.interrupt = interrupt
                await self.turn("When is my flight?")
                self.session.playout.started()
                self.session.playout.played("Your flight")
                self.session.agent.talker = Waits()
                await self.session.on_instruction("Stop speaking immediately.")
                if not tracker_first:
                    self.session.playout.cut()
                self.session.caller_started()  # a cough
                self.session.caller_stopped()
                await self.session.on_turn()
                self.assertEqual(self.spoken, ["Your flight leaves at nine from gate twelve."])
                self.assertNotIn("line.resumed", self.kinds())
                await self.session.close()

    async def test_an_instruction_is_pinned_noted_for_the_backend_and_decided(self):
        await self.call([TalkerDecision("delegate", "Un momento.")])
        await self.session.on_instruction("Reply in Spanish from now on.")
        self.assertIn("Reply in Spanish from now on.", self.frontend.calls[0]["instructions"])
        transcript = self.port.requests[0][0]["content"][0]["text"]
        self.assertIn(f"{PROMPTS.transcript.instruction}: Reply in Spanish from now on.", transcript)
        self.assertEqual(self.spoken, ["Un momento."])

    async def test_an_instruction_survives_every_compaction(self):
        settings = CascadeSettings.parse(
            {"delegation": {"progress_after_seconds": None}, "memory": {"context_tokens": 400}}
        )
        await self.call([TalkerDecision("speak", f"Reply {n}.") for n in range(40)], settings=settings)
        talker = self.session.agent.talker
        self.session.agent.talker = Waits()
        await self.session.on_instruction("Always confirm prices.")
        self.session.agent.talker = talker
        for n in range(40):
            await self.turn(f"Turn {n}: tell me about the menu please.")
        self.assertIn("Always confirm prices.", self.frontend.calls[-1]["instructions"])
        self.assertNotIn("Turn 0:", str(self.frontend.calls[-1]["history"]))  # old turns went
        self.assertLessEqual(self.session.memory.tokens(), 0.8 * 400)
