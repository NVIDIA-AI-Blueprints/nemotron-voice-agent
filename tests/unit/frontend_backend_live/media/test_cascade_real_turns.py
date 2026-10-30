# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D103, D107

"""The cascade's Pipecat binding, with Pipecat's own user aggregator and interruptions on, as in production."""

import asyncio
import unittest
from collections.abc import Callable
from dataclasses import dataclass, replace

from pipecat.frames.frames import (
    InterruptionFrame,
    LLMContextFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
    ProposedUserStartedSpeakingFrame,
    ProposedUserStoppedSpeakingFrame,
    SystemFrame,
    TranscriptionFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair, LLMUserAggregatorParams
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.tests.utils import SleepFrame, run_test
from pipecat.turns.user_start import ExternalUserTurnStartStrategy
from pipecat.turns.user_stop import ExternalUserTurnStopStrategy
from pipecat.turns.user_turn_strategies import UserTurnStrategies

from examples.frontend_backend_live.cascade.memory import ContextConversation
from examples.frontend_backend_live.cascade.ports import ReadyResult
from examples.frontend_backend_live.cascade.session import CascadeSession
from examples.frontend_backend_live.cascade.settings import CascadeSettings
from examples.frontend_backend_live.media.processor import LiveTalkerProcessor
from examples.frontend_backend_live.models.decision import TalkerDecision
from tests.unit.contracts.delegation_port import ScriptedDelegationPort
from tests.unit.frontend_backend_live.cascade.doubles import (
    CONFIG,
    FAST,
    PROMPTS,
    QUIET,
    ScriptedFrontend,
    agent,
    conversation,
)

# Only the interruption ends a held decision: no attempt timeout runs out first on a slow machine.
PATIENT = replace(FAST, talker_attempt_timeout_seconds=30.0)


def turn(text):
    words = [TranscriptionFrame(text=text, user_id="caller", timestamp="0", finalized=True)] if text else []
    return [ProposedUserStartedSpeakingFrame(), *words, ProposedUserStoppedSpeakingFrame()]


@dataclass
class Act(SystemFrame):
    """Something the test does when this frame reaches the head of the pipeline; no interruption drops it."""

    call: Callable[[], object] = lambda: None


class Acts(FrameProcessor):
    """Runs each :class:`Act` as it arrives, in order with the caller's frames."""

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        if isinstance(frame, Act):
            frame.call()
            return
        await self.push_frame(frame, direction)


class Binding(unittest.IsolatedAsyncioTestCase):
    async def bind(self, script, *, stop_timeout=None, port=None, settings=QUIET, reports_playout=False):
        self.context = LLMContext([{"role": "system", "content": "You are Remy."}])
        params = LLMUserAggregatorParams(
            user_turn_strategies=UserTurnStrategies(
                start=[ExternalUserTurnStartStrategy()], stop=[ExternalUserTurnStopStrategy()]
            ),
            **({"user_turn_stop_timeout": stop_timeout} if stop_timeout else {}),
        )
        self.aggregator = LLMContextAggregatorPair(self.context, user_params=params).user()
        self.frontend = ScriptedFrontend(script, reliability=PATIENT)
        self.port = port or ScriptedDelegationPort()
        self.processor = LiveTalkerProcessor()
        self.session = CascadeSession(
            agent(self.frontend),
            CONFIG,
            ContextConversation(self.context),
            delegation=self.port,
            say=self.processor.speak,
            interrupt=self.processor.interrupt,
            settings=settings,
            reports_playout=reports_playout,
        )
        self.processor.attach(self.session)
        self.session.start()
        self.addAsyncCleanup(self.session.close)

    async def play(self, *frames, processors=None):
        await run_test(
            Pipeline(processors or [self.aggregator, self.processor]),
            frames_to_send=list(frames),
            expected_down_frames=None,
        )


class RealTurnTests(Binding):
    async def test_the_caller_speaking_again_while_the_talker_decides_loses_no_words(self):
        slow = asyncio.get_running_loop().create_future()
        await self.bind([slow, TalkerDecision("delegate", "One moment.")], stop_timeout=0.3)
        added = []  # what Pipecat writes into the context; the cascade trims the buffer to its prompt once it heard it
        add_message = self.context.add_message
        self.context.add_message = lambda message: (added.append(message), add_message(message))[1]
        await self.play(
            *turn("My user ID is Q nine one"),
            SleepFrame(0.3),  # the talker is deciding the first turn
            *turn("underscore T"),
            SleepFrame(0.3),
            *turn(""),  # a cough after the delegation: Pipecat's empty-turn recovery runs the processor
            SleepFrame(0.8),
        )
        self.assertTrue(slow.cancelled())  # the second turn's interruption cut off the first decision
        self.assertTrue(any(m["role"] == "developer" for m in added))  # the recovery ran
        self.assertEqual(self.context.get_messages(), [{"role": "system", "content": "You are Remy."}])  # bounded
        talker_calls = [call for call in self.frontend.calls if call["role"] == "talker"]
        self.assertEqual(len(talker_calls), 2)  # the recovery brought no new turn, so nothing was decided again
        self.assertEqual(
            conversation(talker_calls[1]),
            [
                {"role": "user", "content": "My user ID is Q nine one"},
                {"role": "user", "content": "underscore T"},
            ],
        )
        self.assertNotIn("Briefly ask them", talker_calls[1]["instructions"])  # Pipecat's note never reaches it
        transcript = self.port.requests[0][0]["content"][0]["text"]
        caller = PROMPTS.transcript.caller
        self.assertIn(f"{caller}: My user ID is Q nine one\n{caller}: underscore T", transcript)

    async def test_lines_enter_the_memory_once_when_handed_to_speech_and_a_caller_breaking_in_keeps_them(self):
        slow = asyncio.get_running_loop().create_future()
        settings = CascadeSettings.parse(
            {"delegation": {"result_wording": "as_written", "progress_after_seconds": None}}
        )
        await self.bind([slow, TalkerDecision("speak", "Sorry, go ahead.")], settings=settings)
        ready = self.session.presenter.ready
        await self.play(
            Act(
                lambda: (
                    ready(_result("D1", "No user Q91_T exists.")),
                    ready(_result("D2", "Parcel 42 is on time.")),
                )
            ),
            SleepFrame(0.2),  # both results are said together, as one line
            *turn("Find user Q91_T, and is parcel 42 on time?"),
            SleepFrame(0.1),
            *turn("Wait, what?"),  # the caller breaks in
            SleepFrame(0.3),
            processors=[Acts(), self.aggregator, self.processor],
        )
        self.assertTrue(slow.cancelled())
        self.assertEqual(
            [(m["role"], m["content"]) for m in self.session.memory.messages()[1:]],
            [
                ("user", PROMPTS.result_label + "No user Q91_T exists."),
                ("user", PROMPTS.result_label + "Parcel 42 is on time."),
                ("assistant", "No user Q91_T exists. Parcel 42 is on time."),
                ("user", "Find user Q91_T, and is parcel 42 on time?"),
                ("user", "Wait, what?"),
                ("assistant", "Sorry, go ahead."),
            ],
        )

    async def test_a_cough_right_after_a_cut_says_the_unheard_rest_once(self):
        await self.bind(
            [TalkerDecision("speak", "Your flight leaves at nine from gate twelve.")],
            stop_timeout=0.2,
            reports_playout=True,
        )
        playout = self.session.playout

        def played():
            playout.started()
            playout.played("Your flight leaves")

        await self.play(
            *turn("When is my flight?"),
            SleepFrame(0.1),
            Act(played),
            ProposedUserStartedSpeakingFrame(),  # a cough: its interruption passes the binding
            SleepFrame(0.05),
            Act(playout.cut),  # the tracker reports the cut at the output
            ProposedUserStoppedSpeakingFrame(),  # no words: Pipecat's empty-turn recovery runs the processor
            SleepFrame(0.5),
            *turn(""),  # a second cough: nothing more is said
            SleepFrame(0.5),
            processors=[Acts(), self.aggregator, self.processor],
        )
        lines = [line.text for line in self.session.utterances.recent]
        self.assertEqual(lines, ["Your flight leaves at nine from gate twelve.", "at nine from gate twelve."])

    async def test_a_decision_dropped_by_speech_with_no_words_is_made_once_the_caller_is_silent(self):
        # The turn frames come decided elsewhere, as a speech service with its own turn detection sends them: the
        # aggregator adopts them, broadcasts no interruption, and runs no turn after one with no words.
        slow = asyncio.get_running_loop().create_future()
        await self.bind([slow, TalkerDecision("speak", "Yes, we have oat milk.")])
        added = []
        add_message = self.context.add_message
        self.context.add_message = lambda message: (added.append(message), add_message(message))[1]
        await self.play(
            UserStartedSpeakingFrame(),
            TranscriptionFrame(text="Do you have oat milk?", user_id="caller", timestamp="0", finalized=True),
            SleepFrame(0.05),  # the stop, a system frame, would overtake the words
            UserStoppedSpeakingFrame(),
            SleepFrame(0.2),  # the talker is deciding the turn
            UserStartedSpeakingFrame(),  # the caller goes on: nothing cuts the decision off
            SleepFrame(0.1),
            Act(lambda: slow.set_result(TalkerDecision("speak", "Sure."))),  # it comes while they speak: dropped
            SleepFrame(0.1),
            UserStoppedSpeakingFrame(),  # a cough: no words, so no context frame follows
            SleepFrame(0.3),
            processors=[Acts(), self.aggregator, self.processor],
        )
        self.assertFalse(any(m["role"] == "developer" for m in added))  # Pipecat ran no turn after the cough
        talker_calls = [call for call in self.frontend.calls if call["role"] == "talker"]
        self.assertEqual(len(talker_calls), 2)
        self.assertEqual(conversation(talker_calls[1]), [{"role": "user", "content": "Do you have oat milk?"}])
        lines = [line.text for line in self.session.utterances.recent]
        self.assertEqual((lines, self.session.turns.waiting), (["Yes, we have oat milk."], []))


class InstructionInterruptTests(Binding):
    async def test_an_instructions_interruption_keeps_the_caller_turn_queued_behind_it(self):
        # The interruption goes downstream only: a broadcast would reset the processor's own queue and drop the caller's
        # queued turn.
        await self.bind([TalkerDecision("speak", "Sure, a table for two."), TalkerDecision("wait")])
        sent: list[tuple[str, str]] = []
        push = self.processor.push_frame

        async def record(frame, direction=FrameDirection.DOWNSTREAM):
            sent.append((type(frame).__name__, direction.name))
            await push(frame, direction)

        self.processor.push_frame = record
        await self.processor.interrupt()
        self.assertEqual(sent, [("InterruptionFrame", "DOWNSTREAM")])
        self.assertEqual(self.session.utterances.generation, 1)  # the lines handed so far may be cut


class ProcessorTests(Binding):
    async def test_the_processor_closes_nothing_when_its_pipeline_ends(self):
        # The host that built the cascade closes it, once; the processor's pipeline ending leaves the quiet-spell timer
        # running and the talker's model open.
        await self.bind([TalkerDecision("speak", "We open at seven.")], settings=CascadeSettings.parse({}))
        closed = []

        async def noted_close():
            closed.append("talker model")

        self.frontend.close = noted_close
        self.context.add_message({"role": "user", "content": "When do you open?"})
        await run_test(
            self.processor,
            frames_to_send=[LLMContextFrame(context=self.context)],
            expected_down_frames=[LLMFullResponseStartFrame, LLMTextFrame, LLMFullResponseEndFrame],
        )
        timers = [task for task in asyncio.all_tasks() if task.get_name() == "delegation-progress" and not task.done()]
        self.assertEqual((closed, len(timers)), ([], 1))

    async def test_a_turn_is_spoken_as_one_llm_response(self):
        await self.bind([TalkerDecision("speak", "We open at seven.")])
        self.context.add_message({"role": "user", "content": "When do you open?"})
        down, _ = await run_test(
            self.processor,
            frames_to_send=[LLMContextFrame(context=self.context)],
            expected_down_frames=[LLMFullResponseStartFrame, LLMTextFrame, LLMFullResponseEndFrame],
        )
        self.assertEqual(down[1].text, "We open at seven.")

    async def test_utterances_spoken_at_the_same_time_do_not_interleave_their_frames(self):
        await self.bind([])
        sent = []

        async def record(frame, direction=FrameDirection.DOWNSTREAM):
            sent.append(type(frame).__name__)
            await asyncio.sleep(0)

        self.processor.push_frame = record
        await asyncio.gather(*(self.processor.speak(text) for text in ("A lead-in.", "An answer.", "An update.")))
        self.assertEqual(sent, ["LLMFullResponseStartFrame", "LLMTextFrame", "LLMFullResponseEndFrame"] * 3)

    async def test_a_speculative_context_is_ignored(self):
        await self.bind([])
        self.context.add_message({"role": "user", "content": "When do you open?"})
        await run_test(
            self.processor,
            frames_to_send=[LLMContextFrame(context=self.context, speculation=True)],
            expected_down_frames=[],
        )
        self.assertEqual(self.frontend.calls, [])

    async def test_an_interruption_passing_the_binding_starts_a_new_generation(self):
        await self.bind([])
        await run_test(self.processor, frames_to_send=[InterruptionFrame()], expected_down_frames=[InterruptionFrame])
        self.assertEqual(self.session.utterances.generation, 1)


def _result(delegation_id, text):
    return ReadyResult(delegation_id, f"msg_{delegation_id}", text)
