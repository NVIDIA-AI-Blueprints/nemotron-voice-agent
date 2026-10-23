# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: D100, D101, D102, D103, D107

"""ContextQueue: queued → in_flight → committed only on a committed settle; requeue order; dedup; ack."""

from __future__ import annotations

import unittest

from examples.frontend_delegation_hermes.core.backend.context_queue import (
    COMMITTED,
    IN_FLIGHT,
    QUEUED,
    ContextEntry,
    ContextQueue,
)


def user(seq: int, text: str = "") -> ContextEntry:
    return ContextEntry(seq=seq, kind="user", text=text or f"u{seq}")


class ContextQueueTest(unittest.TestCase):
    def test_take_marks_in_flight_and_commit_only_on_settle(self) -> None:
        q = ContextQueue()
        q.add(user(1))
        q.add(user(2))
        taken = q.take(epoch=1, carrier="1:start")
        self.assertEqual([e.seq for e in taken], [1, 2])
        self.assertTrue(all(e.state == IN_FLIGHT and e.epoch == 1 for e in taken))
        self.assertEqual(q.committed_upto, 0)  # nothing committed yet (seq 0 unseen counts as a gap)
        self.assertEqual(q.commit(1), 2)
        self.assertTrue(all(e.state == COMMITTED for e in taken))
        self.assertEqual(q.queued(), [])

    def test_requeue_puts_entries_back_in_order_ahead_of_newer(self) -> None:
        q = ContextQueue()
        q.add(user(1))
        q.add(user(2))
        q.take(1, "1:start")
        q.add_run_input([user(3, "please cancel order 5")], 1, "1:start")
        q.add(user(4))  # arrived while the run was in flight
        note = ContextEntry(seq=None, kind="side_effect_note", text="", origin="derived", extra={"calls": []})
        q.requeue(1, [note])
        queued = q.queued()
        self.assertEqual([e.kind for e in queued], ["side_effect_note", "user", "user", "requeued_request", "user"])
        self.assertEqual([e.seq for e in queued], [None, 1, 2, 3, 4])
        self.assertTrue(all(e.state == QUEUED for e in queued))
        # Next take delivers them once, in that order.
        self.assertEqual([e.seq for e in q.take(2, "2:start")], [None, 1, 2, 3, 4])
        self.assertEqual(q.queued(), [])

    def test_duplicates_are_ignored_even_after_commit(self) -> None:
        q = ContextQueue()
        self.assertTrue(q.add(user(1)))
        self.assertFalse(q.add(user(1)))
        q.take(1, "c")
        q.commit(1)
        self.assertFalse(q.add(user(1)))
        self.assertEqual(q.queued(), [])

    def test_silent_entries_never_travel_but_commit(self) -> None:
        q = ContextQueue()
        q.add(ContextEntry(seq=1, kind="frontend_speech", text="Let me check", outcome="not_heard", origin="frontend"))
        q.add(ContextEntry(seq=2, kind="delivery_note", text="", outcome="heard", origin="derived"))
        q.add(
            ContextEntry(
                seq=3,
                kind="frontend_speech",
                text="Sure thing",
                outcome="partial",
                heard_text="Sure",
                origin="frontend",
            )
        )
        taken = q.take(1, "c")
        self.assertEqual([e.seq for e in taken], [3])
        self.assertEqual(q.committed_upto, 2)

    def test_move_keeps_pending_steer_entries_in_flight(self) -> None:
        q = ContextQueue()
        q.add_run_input([user(1)], 1, "1:start")
        q.add_run_input([user(2)], 1, "1:steer1")
        q.move(1, 2, ["1:steer1"])
        q.commit(1)
        self.assertEqual([e.seq for e in q.in_flight(2)], [2])
        self.assertEqual(q.committed_upto, 1)
        q.commit(2)
        self.assertEqual(q.committed_upto, 2)

    def test_requeue_carrier_returns_words_and_releases_their_seq(self) -> None:
        q = ContextQueue()
        q.add(user(1))
        q.add_run_input([user(2, "and order 9")], 1, "1:steer1")
        q.take(1, "1:steer1")
        words = q.requeue_carrier(1, "1:steer1")
        self.assertEqual([e.seq for e in words], [2])
        self.assertEqual([e.seq for e in q.queued()], [1])
        self.assertTrue(q.add_run_input(words, 2, "2:start"))  # the seq can be registered again

    def test_watermark_skips_gaps_below_the_oldest_uncommitted(self) -> None:
        q = ContextQueue()
        q.add(user(5))
        q.add(user(7))
        q.take(1, "c")
        q.commit(1)
        self.assertEqual(q.committed_upto, 7)
        q.add(user(9))
        self.assertEqual(q.committed_upto, 7)


if __name__ == "__main__":
    unittest.main()
