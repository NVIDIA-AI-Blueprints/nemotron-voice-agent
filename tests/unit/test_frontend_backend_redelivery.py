# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

# ruff: noqa: D101, D102

"""Held and cut backend answers are given again after an acknowledgement (Rec. 2b, 2e)."""

from __future__ import annotations

import json
import unittest
from collections.abc import AsyncIterator
from types import SimpleNamespace

from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    InterruptionFrame,
    UserStartedSpeakingFrame,
)
from pipecat.processors.aggregators import async_tool_messages
from pipecat.processors.aggregators.llm_context import LLMContext

from examples.frontend_backend_agent.src.barge_in import BargeInState
from examples.frontend_backend_agent.src.redelivery import (
    ANSWER_REDELIVERY_NOTE,
    HELD_ANSWER_NOTE,
    DeliveryTracker,
)
from examples.frontend_backend_agent.src.reliable_talker import ReliableNvidiaLLMService
from realtime.vad import REALTIME_VAD_START_CONFIRMATION_SECONDS

QUESTION = {
    "type": "response_hint",
    "reason": "params_missing",
    "action": "req_params",
    "params_needed": ["user_id"],
    "response_text": "Please tell me your user ID.",
    "context": "get_user_details",
}
ANSWER = {
    "type": "tool_result",
    "tool": "get_reservation_details",
    "status": "success",
    "response_text": "Your baggage allowance is two suitcases total.",
    "context": "get_reservation_details",
    "data": {"owner": "client", "result": {"bags": 2}},
}
CONSENT = {
    "type": "response_hint",
    "reason": "confirmation_needed",
    "action": "req_confirmation",
    "response_text": "Just to confirm, I will cancel the reservation, with reservation id: 59XX6W. Shall I go ahead?",
    "context": "cancel_reservation",
}


def _cut_before_audio(tracker: DeliveryTracker, payload: dict) -> None:
    """Task 23: the reply starts at turn end and caller speech cancels it before any audio."""
    tracker.note_payload(payload)
    assert tracker.begin_talker_run("tool", "").note is None
    tracker.talker_output(spoke=True, called_function=False)
    tracker.observe(InterruptionFrame())


class DeliveryTrackerTests(unittest.TestCase):
    def test_reply_cut_before_audio_is_given_again_once_after_progress_check(self) -> None:
        tracker = DeliveryTracker()
        _cut_before_audio(tracker, QUESTION)
        self.assertEqual(tracker.state, "cut")

        plan = tracker.begin_talker_run("user", "Okay, any update?")

        self.assertTrue(plan.allow_replay)
        self.assertIn(json.dumps(QUESTION["response_text"]), plan.note or "")
        # Cut again: never given a third time.
        tracker.talker_output(spoke=True, called_function=False)
        tracker.observe(InterruptionFrame())
        again = tracker.begin_talker_run("user", "Okay.")
        self.assertIsNone(again.note)
        self.assertFalse(again.allow_replay)

    def test_answer_cut_while_playing_is_redelivered_after_acknowledgement(self) -> None:
        tracker = DeliveryTracker()
        tracker.note_payload(ANSWER)
        tracker.begin_talker_run("tool", "")
        tracker.talker_output(spoke=True, called_function=False)
        tracker.observe(BotStartedSpeakingFrame())
        tracker.observe(InterruptionFrame())

        plan = tracker.begin_talker_run("user", "Hello")

        self.assertEqual(plan.note, ANSWER_REDELIVERY_NOTE)

    def test_consent_question_is_asked_again_in_full(self) -> None:
        tracker = DeliveryTracker()
        tracker.note_payload(CONSENT)
        tracker.begin_talker_run("tool", "")
        tracker.observe(BotStartedSpeakingFrame())
        tracker.observe(InterruptionFrame())

        plan = tracker.begin_talker_run("user", "yes")

        self.assertIn("in full", plan.note or "")
        self.assertIn(json.dumps(CONSENT["response_text"]), plan.note or "")

    def test_substantive_words_drop_a_cut_answer(self) -> None:
        tracker = DeliveryTracker()
        _cut_before_audio(tracker, ANSWER)

        plan = tracker.begin_talker_run("user", "Actually, never mind, cancel that.")

        self.assertIsNone(plan.note)
        self.assertEqual(tracker.state, "done")

    def test_heard_answer_is_not_repeated(self) -> None:
        tracker = DeliveryTracker()
        tracker.note_payload(ANSWER)
        tracker.begin_talker_run("tool", "")
        tracker.talker_output(spoke=True, called_function=False)
        tracker.observe(BotStartedSpeakingFrame())
        tracker.observe(BotStoppedSpeakingFrame())
        tracker.observe(InterruptionFrame())

        self.assertEqual(tracker.state, "heard")
        self.assertIsNone(tracker.begin_talker_run("user", "Okay.").note)

    def test_talker_that_delegates_again_drops_the_answer(self) -> None:
        tracker = DeliveryTracker()
        tracker.note_payload(ANSWER)
        tracker.begin_talker_run("tool", "")
        tracker.talker_output(spoke=False, called_function=True)
        tracker.observe(InterruptionFrame())

        self.assertEqual(tracker.state, "done")
        self.assertIsNone(tracker.begin_talker_run("user", "Okay.").note)

    def test_answer_held_during_caller_speech_is_given_after_acknowledgement(self) -> None:
        tracker = DeliveryTracker()
        tracker.note_payload(ANSWER)
        # The caller is still speaking; no reply has started.
        tracker.observe(InterruptionFrame())
        self.assertEqual(tracker.state, "held")

        plan = tracker.begin_talker_run("user", "Are you still there?")

        self.assertEqual(plan.note, HELD_ANSWER_NOTE)
        self.assertTrue(plan.allow_replay)

    def test_held_answer_with_new_words_is_left_to_the_talker(self) -> None:
        tracker = DeliveryTracker()
        tracker.note_payload(ANSWER)

        plan = tracker.begin_talker_run("user", "Also add a checked bag.")

        self.assertIsNone(plan.note)
        self.assertEqual(tracker.state, "started")

    def test_status_notices_are_not_tracked(self) -> None:
        tracker = DeliveryTracker()
        for reason in ("no_action_needed", "aborted", "planner_error", "timeout"):
            tracker.note_payload({"type": "response_hint", "reason": reason, "response_text": "x"})
        tracker.note_payload({**ANSWER, "speakable": False})
        self.assertIsNone(tracker.state)

    def test_switches_disable_each_behaviour(self) -> None:
        tracker = DeliveryTracker(hold_answers=True, redeliver_cut_answers=False)
        _cut_before_audio(tracker, ANSWER)
        self.assertIsNone(tracker.begin_talker_run("user", "Okay.").note)

        tracker = DeliveryTracker(hold_answers=False, redeliver_cut_answers=True)
        tracker.note_payload(ANSWER)
        self.assertIsNone(tracker.begin_talker_run("user", "Okay.").note)
        self.assertFalse(DeliveryTracker(hold_answers=False, redeliver_cut_answers=False).enabled)

    def test_barge_in_state_forwards_frames(self) -> None:
        tracker = DeliveryTracker()
        state = BargeInState(tracker)
        tracker.note_payload(ANSWER)
        tracker.begin_talker_run("tool", "")
        state.observe(BotStartedSpeakingFrame())
        state.observe(UserStartedSpeakingFrame())
        state.observe(InterruptionFrame())

        self.assertEqual(tracker.state, "cut")
        self.assertTrue(state.consume_interrupted_speech())


def test_realtime_vad_start_confirmation_already_exceeds_prototype_minimum_speech() -> None:
    """Rec. 2c: speech shorter than the prototype's 120 ms never starts a Realtime turn."""
    assert REALTIME_VAD_START_CONFIRMATION_SECONDS >= 0.12


def _chunk(*, content: str | None = None, tool_calls: list | None = None):
    delta = SimpleNamespace(content=content, tool_calls=tool_calls)
    return SimpleNamespace(choices=[SimpleNamespace(delta=delta)])


async def _stream(chunks: list) -> AsyncIterator:
    for chunk in chunks:
        yield chunk


class _ScriptedTalker(ReliableNvidiaLLMService):
    def __init__(self, responses: list[list]) -> None:
        self._responses = list(responses)
        self.contexts: list[LLMContext] = []
        self.fallbacks: list[str] = []
        self._delivery_tracker = None

    async def _start_completion_stream(self, context: LLMContext) -> AsyncIterator:
        self.contexts.append(context)
        return _stream(self._responses.pop(0))

    async def _push_llm_text(self, text: str) -> None:
        self.fallbacks.append(text)


def _context(*user_texts: str) -> LLMContext:
    messages = [
        {"role": "user", "content": "How many bags can I bring?"},
        async_tool_messages.build_final_result_message("call-1", json.dumps(ANSWER)),
    ]
    messages.extend({"role": "user", "content": text} for text in user_texts)
    return LLMContext(messages, tools=[], tool_choice="auto")


class TalkerRedeliveryTests(unittest.IsolatedAsyncioTestCase):
    async def _run(self, talker: _ScriptedTalker, context: LLMContext) -> list:
        stream = await talker.get_chat_completions(context)
        return [chunk async for chunk in stream]

    async def test_redelivered_answer_is_not_rejected_as_a_cached_replay(self) -> None:
        tracker = DeliveryTracker()
        talker = _ScriptedTalker(
            [
                [_chunk(content="Your baggage allowance is two suitcases total.")],
                [_chunk(content="Your baggage allowance is two suitcases total.")],
            ]
        )
        talker.bind_delivery_tracker(tracker)
        talker.remember_backend_result(ANSWER)

        await self._run(talker, _context())
        self.assertEqual(tracker.state, "speaking")
        tracker.observe(InterruptionFrame())

        chunks = await self._run(talker, _context("Okay."))

        self.assertEqual(len(chunks), 1)
        self.assertEqual(len(talker.contexts), 2)
        self.assertEqual(talker.contexts[1].get_messages()[-1], {"role": "system", "content": ANSWER_REDELIVERY_NOTE})
        self.assertEqual(talker.fallbacks, [])

    async def test_without_an_unheard_answer_a_replay_is_still_corrected(self) -> None:
        tracker = DeliveryTracker()
        talker = _ScriptedTalker(
            [
                [_chunk(content="Your baggage allowance is two suitcases total.")],
                [_chunk(content="Your baggage allowance is two suitcases total.")],
                [_chunk(content="Anything else?")],
            ]
        )
        talker.bind_delivery_tracker(tracker)
        talker.remember_backend_result(ANSWER)
        await self._run(talker, _context())
        tracker.observe(BotStartedSpeakingFrame())
        tracker.observe(BotStoppedSpeakingFrame())

        await self._run(talker, _context("Okay."))

        # The replay was rejected and retried without a delivery note.
        self.assertEqual(len(talker.contexts), 3)
        self.assertNotIn(ANSWER_REDELIVERY_NOTE, json.dumps(talker.contexts[1].get_messages()))
