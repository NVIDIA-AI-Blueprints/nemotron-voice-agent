# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D103

"""Resume after noise or after a turn the talker waits on; repeat a silent line."""

import asyncio

from examples.frontend_backend_live.cascade.settings import CascadeSettings
from examples.frontend_backend_live.cascade.utterances import LineKind
from examples.frontend_backend_live.models.decision import TalkerDecision
from tests.unit.frontend_backend_live.cascade.doubles import PROMPTS, Call, settle

ANSWER = "Your flight leaves at nine from gate twelve, and boarding starts at eight thirty."
REST = "from gate twelve, and boarding starts at eight thirty."  # what a cut after "nine" left unheard


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

    async def cut_by(self, words, *script, settings=None):
        """The caller's ``words`` cut the reply: heard, they wait for their decision, the next of ``script``."""
        kwargs = {"settings": settings} if settings else {}
        await self.call([TalkerDecision("speak", ANSWER), *script], reports_playout=True, **kwargs)
        await self.turn("When is my flight?")
        self.session.caller_started()
        self.session.interrupted()
        self.session.playout.started()
        self.session.playout.played("Your flight leaves at nine")
        self.session.playout.cut()
        self.buffer.turn(words)
        self.session.caller_stopped()

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
                ("assistant", "Your flight leaves at nine" + PROMPTS.cut_mark),
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

    async def test_words_after_a_cut_that_the_talker_waits_on_say_the_unheard_rest_once(self):
        # The caller broke in with "Okay." and the talker chose to say nothing: the line it stopped goes on, once, and a
        # cough afterwards says nothing more.
        await self.cut_by("Okay.", TalkerDecision("wait"))
        await self.session.on_turn()  # decided: wait
        self.assertEqual(self.spoken, [ANSWER, REST])
        self.assertIn("line.resumed", self.kinds())
        self.session.caller_started()  # then a cough
        self.session.caller_stopped()
        await self.session.on_turn()
        self.assertEqual(len(self.spoken), 2)

    async def test_a_turn_the_talker_delegates_after_a_cut_resumes_nothing(self):
        # The delegation answers the turn, even with no lead-in; a cough afterwards does not bring the line back either.
        await self.cut_by("Can I change my seat?", TalkerDecision("delegate", ""))
        await self.session.on_turn()
        self.session.caller_started()
        self.session.caller_stopped()
        await self.session.on_turn()
        self.assertEqual((self.spoken, len(self.port.requests)), ([ANSWER], 1))
        self.assertNotIn("line.resumed", self.kinds())

    async def test_a_line_said_after_the_cut_leaves_the_waited_turn_nothing_to_resume(self):
        await self.cut_by("Okay.", TalkerDecision("wait"))
        await self.session.utterances.say("Your gate changed to fourteen.", LineKind.NOTICE)  # said before the decision
        await self.session.on_turn()
        self.assertEqual(self.spoken, [ANSWER, "Your gate changed to fourteen."])
        self.assertNotIn("line.resumed", self.kinds())

    async def test_words_joined_to_a_cut_off_decision_resume_the_line_the_first_words_cut(self):
        # "Okay." cut the reply and "Go on." cut off its decision: one decision covers both, and the reply goes on once.
        slow = asyncio.get_running_loop().create_future()
        await self.cut_by("Okay.", slow, TalkerDecision("wait"))
        deciding = asyncio.create_task(self.session.on_turn())
        await settle(lambda: len(self.frontend.calls) == 2)
        deciding.cancel()  # the caller says more: Pipecat cancels the decision
        await asyncio.gather(deciding, return_exceptions=True)
        await self.turn("Go on.")
        self.assertEqual(self.spoken, [ANSWER, REST])

    async def test_words_joined_to_a_discarded_decision_resume_the_line_the_first_words_cut(self):
        # "Okay." cut the reply, and the caller went on before its decision began, so no interruption cut that decision
        # off: it is discarded when it completes, one decision covers both turns, and the reply goes on once.
        slow = asyncio.get_running_loop().create_future()
        await self.cut_by("Okay.", slow, TalkerDecision("wait"))
        self.session.caller_started()
        deciding = asyncio.create_task(self.session.on_turn())
        await settle(lambda: len(self.frontend.calls) == 2)
        slow.set_result(TalkerDecision("speak", "Sure."))
        await deciding
        self.assertEqual(self.spoken, [ANSWER])
        self.buffer.turn("Go on.")
        self.session.caller_stopped()
        await self.session.on_turn()
        self.assertEqual(self.spoken, [ANSWER, REST])

    async def test_resuming_can_be_turned_off(self):
        settings = CascadeSettings.parse(
            {"delegation": {"progress_after_seconds": None}, "speech": {"resume_after_noise": False}}
        )
        await self.cut_reply(settings)
        await self.session.on_turn()
        self.assertEqual(self.spoken, [ANSWER])

    async def test_resuming_after_a_waited_turn_can_be_turned_off(self):
        settings = CascadeSettings.parse(
            {"delegation": {"progress_after_seconds": None}, "speech": {"resume_after_noise": False}}
        )
        await self.cut_by("Okay.", TalkerDecision("wait"), settings=settings)
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
