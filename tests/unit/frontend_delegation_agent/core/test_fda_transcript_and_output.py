# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D100, D101, D102, D103, D107

"""SharedTranscript routes/outcomes/projections and the OutputQueue hold and stale rules."""

from __future__ import annotations

import unittest

from examples.frontend_delegation_agent.core.engine.output_scheduler import CallsItem, OutputQueue, SpeechItem
from examples.frontend_delegation_agent.core.engine.transcript import (
    CONTEXT,
    HEARD,
    NATIVE,
    NOT_HEARD,
    PARTIAL,
    RUN_INPUT,
    SharedTranscript,
)


class TranscriptTests(unittest.TestCase):
    def test_routes_are_fixed_at_creation(self) -> None:
        t = SharedTranscript()
        self.assertEqual(t.add_user("check order", turn_id=1, delegated=True).route, RUN_INPUT)
        self.assertEqual(t.add_user("okay", turn_id=2, delegated=False).route, CONTEXT)
        self.assertEqual(t.add_spoken("backend_answer", "hermes", "It shipped.").route, NATIVE)
        self.assertEqual(t.add_spoken("frontend_speech", "frontend", "Sure.").route, CONTEXT)

    def test_context_is_taken_once_in_order_and_blocks_on_pending(self) -> None:
        t = SharedTranscript()
        t.seed_greeting("Hi!")
        filler = t.add_spoken("frontend_speech", "frontend", "Sure.", turn_id=1)
        t.add_user("okay", turn_id=2, delegated=False)
        self.assertEqual([e.seq for e in t.take_context()], [0])  # the pending filler blocks the ack
        t.settle(filler, HEARD)
        self.assertEqual([e.kind for e in t.take_context()], ["frontend_speech", "user"])
        self.assertEqual(t.take_context(), [])  # exactly once

    def test_not_heard_frontend_lines_never_travel(self) -> None:
        t = SharedTranscript()
        filler = t.add_spoken("frontend_speech", "frontend", "Sure.")
        t.settle(filler, NOT_HEARD, reason="dropped_stale")
        self.assertEqual(t.take_context(), [])
        self.assertEqual(t.frontend_messages(max_groups=5), [])

    def test_cut_or_undelivered_answer_gets_a_delivery_note_not_an_edit(self) -> None:
        t = SharedTranscript()
        answer = t.add_spoken("backend_answer", "hermes", "Your order shipped. It arrives Friday.", answer_id="a1")
        note = t.settle(answer, PARTIAL, heard_text="Your order shipped.")
        self.assertIsNotNone(note)
        self.assertEqual(answer.text, "Your order shipped. It arrives Friday.")  # never rewritten
        wire = [e.wire() for e in t.take_context()]
        self.assertEqual(wire[0]["kind"], "delivery_note")
        self.assertEqual(
            (wire[0]["outcome"], wire[0]["heard_text"], wire[0]["answer_id"]), (PARTIAL, "Your order shipped.", "a1")
        )
        undelivered = t.add_spoken("backend_answer", "hermes", "Also 1235 shipped.", answer_id="a2")
        note2 = t.settle(undelivered, PARTIAL, heard_text="  ")  # nothing heard -> not heard
        self.assertEqual(note2.outcome, NOT_HEARD)
        self.assertIsNone(t.settle(undelivered, HEARD))  # outcomes are final

    def test_frontend_projection_uses_heard_text_and_alternates(self) -> None:
        t = SharedTranscript()
        t.add_user("check order 1", turn_id=1, delegated=True)
        t.settle(t.add_spoken("frontend_speech", "frontend", "Sure."), HEARD)
        t.settle(
            t.add_spoken("backend_answer", "hermes", "It shipped. Anything else?"), PARTIAL, heard_text="It shipped."
        )
        t.add_user("thanks", turn_id=2, delegated=False)
        self.assertEqual(
            t.frontend_messages(max_groups=10),
            [
                {"role": "user", "content": "check order 1"},
                {"role": "assistant", "content": "Sure. It shipped. [cut off by the user]"},
                {"role": "user", "content": "thanks"},
            ],
        )
        self.assertEqual(t.frontend_messages(max_groups=1), [{"role": "user", "content": "thanks"}])

    def test_backend_asked_question_uses_heard_backend_text(self) -> None:
        t = SharedTranscript()
        answer = t.add_spoken("backend_answer", "hermes", "Shall I cancel it?")
        self.assertFalse(t.backend_asked_question())  # not heard yet
        t.settle(answer, HEARD)
        self.assertTrue(t.backend_asked_question())


class _Clock:
    def __init__(self) -> None:
        self.now = 100.0

    def __call__(self) -> float:
        return self.now


class OutputQueueTests(unittest.TestCase):
    def queue(self, clock: _Clock) -> OutputQueue:
        return OutputQueue(
            hold_max_ms=4000,
            on_stale={"backend_answer": "keep", "status": "drop", "filler": "drop", "apology": "keep"},
            monotonic=clock,
        )

    def test_calls_are_never_held(self) -> None:
        clock = _Clock()
        q = self.queue(clock)
        q.push(SpeechItem(kind="answer", text="x", queued_mono=clock.now))
        q.push(CallsItem(batch_id="b", calls=[("c1", "get_order", "{}")]))
        item, _ = q.next(user_speaking=True, deciding=True, current_turn=None)
        self.assertIsInstance(item, CallsItem)

    def test_audio_never_released_while_the_user_speaks_even_when_stale(self) -> None:
        clock = _Clock()
        q = self.queue(clock)
        q.push(SpeechItem(kind="answer", text="x", queued_mono=clock.now))
        clock.now += 60
        self.assertEqual(q.next(user_speaking=True, deciding=False, current_turn=None), (None, []))
        item, dropped = q.next(user_speaking=False, deciding=False, current_turn=None)
        self.assertEqual((item.kind, dropped), ("answer", []))  # stale answers are kept

    def test_new_turn_filler_goes_first_and_backend_speech_waits_for_the_decision(self) -> None:
        clock = _Clock()
        q = self.queue(clock)
        q.push(SpeechItem(kind="answer", text="old", turn_id=1, queued_mono=clock.now))
        self.assertEqual(q.next(user_speaking=False, deciding=True, current_turn=2), (None, []))
        q.push(SpeechItem(kind="filler", text="Sure.", turn_id=2, queued_mono=clock.now))
        item, _ = q.next(user_speaking=False, deciding=False, current_turn=2)
        self.assertEqual(item.text, "Sure.")

    def test_stale_status_and_filler_are_dropped(self) -> None:
        clock = _Clock()
        q = self.queue(clock)
        q.push(SpeechItem(kind="status", text="s", queued_mono=clock.now))
        q.push(SpeechItem(kind="filler", text="f", turn_id=1, queued_mono=clock.now))
        q.push(SpeechItem(kind="apology", text="a", queued_mono=clock.now))
        clock.now += 5
        item, dropped = q.next(user_speaking=False, deciding=False, current_turn=None)
        self.assertEqual(item.kind, "apology")
        self.assertEqual(sorted(d.item.kind for d in dropped), ["filler", "status"])
        self.assertTrue(all(d.reason == "dropped_stale" for d in dropped))


if __name__ == "__main__":
    unittest.main()
