# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D100, D101, D102, D103, D107

"""The run <-> turn reporting adapter (plan section 14)."""

from __future__ import annotations

import unittest

from examples.frontend_delegation_hermes.cli.report_adapter import adapt


def rec(ts: float, kind: str, **data):
    return {"timestamp": ts, "kind": kind, "session_id": "s1", **data}


def decision(ts, turn, delegate, request="task", filler="", ms=500):
    return rec(ts, "delegation_decision", turn_id=turn, delegate=delegate, request=request, filler=filler,
               latency_ms=ms, usage={"input_tokens": 100, "output_tokens": 10}, text=f"turn {turn}")  # fmt: skip


LOG = [
    rec(0.0, "session_start", model="pine-fdh-voice-dlg-mock-control"),
    decision(1.0, 1, True, filler="Sure."),
    rec(1.1, "backend_action", action="start", turn_id=1, run_id="r1", epoch=1),
    rec(1.5, "filler_timing", turn_id=1, user_stop_to_first_audio_ms=700),
    decision(3.0, 2, False),  # "okay sure"
    decision(4.0, 3, True, filler="And that one."),
    rec(4.1, "backend_action", action="steer", turn_id=3, run_id="r1", epoch=1),
    rec(5.0, "tool_output_in", call_id="c1", output="ok"),
    decision(6.0, 4, True, request="status", filler="Let me check."),
    rec(6.1, "backend_action", action="status", turn_id=4, run_id="r1", epoch=1),
    rec(9.1, "backend_answer", answer_id="a1", run_id="r1", answer_kind="hermes", turn_ids=[1, 3]),
    rec(
        9.1,
        "backend_run_done",
        run_id="r1",
        epoch=1,
        status="ok",
        history="committed",
        turn_ids=[1, 3],
        usage={"input_tokens": 2000, "output_tokens": 80, "api_calls": 3},
        tool_call_ids=["c1"],
    ),  # fmt: skip
    rec(9.5, "turn_latency", turn_id=1, user_stop_to_first_audio_ms=8900),
]


class AdapterTests(unittest.TestCase):
    def test_runs_are_attributed_to_their_starting_turn(self) -> None:
        legacy, problems = adapt(LOG)
        self.assertEqual(problems, [])
        done = {r["turn_id"]: r for r in legacy if r["kind"] == "agent_turn_done"}
        self.assertEqual([done[t]["outcome"] for t in (1, 2, 3, 4)], ["text", "no_reply", "steer", "status"])
        self.assertEqual(done[1]["backend"]["prompt_tokens"], 2000)
        self.assertEqual(done[1]["backend"]["calls"], 3)
        self.assertAlmostEqual(done[1]["backend"]["latency_ms"], 8000.0)
        self.assertEqual(done[3]["backend"]["calls"], 0)  # a steer turn carries frontend usage only
        self.assertEqual(done[3]["frontend"]["prompt_tokens"], 100)
        fillers = [r for r in legacy if r["kind"] == "filler_timing"]
        self.assertEqual([(r["turn_id"], r["outcome"]) for r in fillers], [(1, "answer")])
        kinds = {r["kind"] for r in legacy}
        self.assertTrue({"session_start", "tool_output_in", "turn_latency", "agent_turn_start"} <= kinds)

    def test_self_checks(self) -> None:
        orphan_call = [*LOG, rec(10.0, "tool_output_in", call_id="c9", output="x")]
        _, problems = adapt(orphan_call)
        self.assertEqual(len(problems), 1)
        self.assertIn("c9", problems[0])


if __name__ == "__main__":
    unittest.main()
