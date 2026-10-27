# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D103

"""Resume after noise; repeat a silent line."""

from examples.frontend_backend_live.cascade.settings import CascadeSettings
from examples.frontend_backend_live.models.decision import TalkerDecision
from tests.unit.frontend_backend_live.cascade.doubles import Call

ANSWER = "Your flight leaves at nine from gate twelve, and boarding starts at eight thirty."


class ResumeTests(Call):
    async def cut_reply(self, settings=None):
        kwargs = {"settings": settings} if settings else {}
        await self.call([TalkerDecision("speak", ANSWER)], reports_playout=True, **kwargs)
        await self.turn("When is my flight?")
        self.session.caller_started()  # a cough starts a turn and its interruption cuts the reply
        self.session.interrupted()
        self.session.playout.started()
        self.session.playout.played("Your flight leaves at nine")
        self.session.playout.cut()
        self.session.caller_stopped()  # no words came

    async def test_a_turn_with_no_words_right_after_a_cut_says_the_unheard_rest_once(self):
        await self.cut_reply()
        await self.session.on_turn()
        self.assertEqual(self.spoken, [ANSWER, "from gate twelve, and boarding starts at eight thirty."])
        self.assertIn("line.resumed", self.kinds())
        await self.session.on_turn()  # a second turn with no words says nothing
        self.assertEqual(len(self.spoken), 2)
        self.assertEqual(
            self.memory()[-2:],
            [
                ("assistant", "Your flight leaves at nine [cut off]"),
                ("assistant", "from gate twelve, and boarding starts at eight thirty."),
            ],
        )

    async def test_a_turn_with_words_after_a_cut_is_decided_and_nothing_resumes(self):
        await self.cut_reply()
        self.frontend.script.append(TalkerDecision("speak", "Sure, which seat?"))
        self.buffer.turn("Can I change my seat?")
        await self.session.on_turn()
        self.assertEqual(self.spoken, [ANSWER, "Sure, which seat?"])
        await self.session.on_turn()
        self.assertEqual(len(self.spoken), 2)

    async def test_words_after_a_cut_that_the_talker_waits_on_leave_nothing_to_resume(self):
        # The caller broke in with words and the talker chose to say nothing: a cough afterwards is no turn with no
        # words right after the cut, so the line the caller stopped is not said again.
        class Waits:
            async def decide(self, instructions, history, work_in_progress="", *, work_open=False):
                return TalkerDecision("wait")

        await self.call([TalkerDecision("speak", ANSWER)], reports_playout=True)
        await self.turn("When is my flight?")
        self.session.agent.talker = Waits()
        self.session.caller_started()
        self.session.interrupted()
        self.session.playout.started()
        self.session.playout.played("Your flight leaves at nine")
        self.session.playout.cut()
        self.buffer.turn("Hold on.")
        self.session.caller_stopped()
        await self.session.on_turn()  # decided: wait
        self.session.caller_started()  # then a cough
        self.session.caller_stopped()
        await self.session.on_turn()
        self.assertEqual(self.spoken, [ANSWER])
        self.assertNotIn("line.resumed", self.kinds())

    async def test_an_empty_turn_with_no_cut_says_nothing(self):
        await self.call([TalkerDecision("speak", ANSWER)])
        await self.turn("When is my flight?")
        await self.session.on_turn()
        self.assertEqual(self.spoken, [ANSWER])

    async def test_resuming_can_be_turned_off(self):
        settings = CascadeSettings.parse(
            {"delegation": {"progress_after_seconds": None}, "speech": {"resume_after_noise": False}}
        )
        await self.cut_reply(settings)
        await self.session.on_turn()
        self.assertEqual(self.spoken, [ANSWER])


class SilentLineTests(Call):
    async def test_a_silent_line_is_said_once_more_and_a_second_silence_is_not_heard(self):
        await self.call([TalkerDecision("speak", "We open at seven.")], reports_playout=True)
        await self.turn("When do you open?")
        self.session.playout.silent()
        await self.said(2)
        self.assertEqual(self.spoken, ["We open at seven.", "We open at seven."])
        self.session.playout.silent()
        self.assertEqual(len(self.spoken), 2)
        self.assertNotIn(("assistant", "We open at seven."), self.memory())

    async def test_retrying_can_be_turned_off(self):
        settings = CascadeSettings.parse(
            {"delegation": {"progress_after_seconds": None}, "speech": {"retry_silent_lines": False}}
        )
        await self.call([TalkerDecision("speak", "We open at seven.")], reports_playout=True, settings=settings)
        await self.turn("When do you open?")
        self.session.playout.silent()
        self.assertEqual(self.spoken, ["We open at seven."])
