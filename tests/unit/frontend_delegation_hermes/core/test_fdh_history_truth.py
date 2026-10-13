# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: D100, D101, D102, D103, D107

"""Truthful backend history (plan sections 7.3-7.4), backend half.

Every origin × playback outcome reaches Hermes only as labelled context with its real
outcome, or not at all; Hermes answers are corrected only by delivery notes; the
settled history only ever grows by Hermes' own results (no row is edited or inserted).
"""

from __future__ import annotations

import unittest

from _fdh_backend_fakes import Harness, history_after

SPOKEN = [
    ("frontend", "frontend_speech", "[the voice frontend told the user"),
    ("frontend", "status_speech", "[the voice frontend gave the user a progress update"),
    ("controller", "controller_speech", "[the system told the user"),
]


class HistoryTruthTest(unittest.IsolatedAsyncioTestCase):
    async def run_message_for(self, entries: list[dict]) -> str:
        h = Harness()
        await h.start()
        h.controller.append(entries)
        await h.delegate(1, "next request", seq=99)
        return h.worker.of_type("run")[0]["user_message"]

    async def test_spoken_lines_by_outcome(self) -> None:
        for origin, kind, label in SPOKEN:
            with self.subTest(kind=kind):
                heard = await self.run_message_for(
                    [{"seq": 1, "origin": origin, "kind": kind, "text": "One moment please.", "outcome": "heard"}]
                )
                self.assertIn(f'{label}] "One moment please."', heard)
                partial = await self.run_message_for(
                    [
                        {
                            "seq": 1,
                            "origin": origin,
                            "kind": kind,
                            "text": "One moment please.",
                            "outcome": "partial",
                            "heard_text": "One moment",
                        }
                    ]
                )
                self.assertIn(f'{label}, cut off by the user] "One moment"', partial)
                self.assertNotIn("please", partial.split("next request")[0])
                unheard = await self.run_message_for(
                    [{"seq": 1, "origin": origin, "kind": kind, "text": "One moment please.", "outcome": "not_heard"}]
                )
                self.assertEqual(unheard, "next request")

    async def test_delivery_notes(self) -> None:
        cut = await self.run_message_for(
            [
                {
                    "seq": 1,
                    "origin": "derived",
                    "kind": "delivery_note",
                    "text": "Your order was cancelled today.",
                    "outcome": "partial",
                    "heard_text": "Your order was",
                }
            ]
        )
        self.assertIn('[your last answer was cut off; the user heard only: "Your order was"]', cut)
        lost = await self.run_message_for(
            [
                {
                    "seq": 1,
                    "origin": "derived",
                    "kind": "delivery_note",
                    "text": "x",
                    "outcome": "not_heard",
                    "answer_id": "a1",
                }
            ]
        )
        self.assertIn("[your previous answer was not delivered to the user]", lost)

    async def test_user_context_labels_depend_on_the_backend_state(self) -> None:
        h = Harness()
        await h.start()
        h.controller.append([{"seq": 1, "origin": "user", "kind": "user", "text": "hello", "outcome": "heard"}])
        await h.delegate(1, "order 1", seq=2)
        h.controller.append([{"seq": 3, "origin": "user", "kind": "user", "text": "okay sure", "outcome": "heard"}])
        self.assertIn('[earlier] user: "hello"', h.worker.of_type("run")[0]["user_message"])
        await h.delegate(2, "and order 2", seq=4)
        self.assertIn('[while you were working] user: "okay sure"', h.worker.of_type("steer")[0]["text"])

    async def test_settled_history_only_grows_by_hermes_results(self) -> None:
        h = Harness()
        await h.start()
        await h.delegate(1, "order 1", seq=1)
        first = history_after("order 1", "Your order was cancelled today.")
        h.worker.outcome(1, "ok", final_response=first[-1]["content"], messages=first)
        await h.settle()
        snapshot = [dict(m) for m in h.controller.history]
        # The answer was cut off: only a delivery note may say so.
        h.controller.append(
            [
                {
                    "seq": 2,
                    "origin": "derived",
                    "kind": "delivery_note",
                    "text": first[-1]["content"],
                    "outcome": "partial",
                    "heard_text": "Your order",
                }
            ]
        )
        await h.delegate(2, "what?", seq=3)
        run = h.worker.of_type("run")[-1]
        self.assertEqual(run["conversation_history"], snapshot)  # nothing edited or inserted
        self.assertIn("cut off", run["user_message"])
        second = history_after(run["user_message"], "It was cancelled.", snapshot)
        h.worker.outcome(2, "ok", final_response="It was cancelled.", messages=second)
        await h.settle()
        self.assertEqual(h.controller.history[: len(snapshot)], snapshot)

    async def test_apology_answers_are_marked_as_apologies(self) -> None:
        h = Harness()
        await h.start()
        await h.delegate(1, "x", seq=1)
        h.worker.outcome(1, "ok", final_response="", messages=history_after("x", ""))
        await h.settle()
        answer = h.of_type("answer")[-1]
        self.assertEqual((answer["kind"], answer["reason"]), ("apology", "empty_response"))


if __name__ == "__main__":
    unittest.main()
