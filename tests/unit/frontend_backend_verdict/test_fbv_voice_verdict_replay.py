# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D100, D101, D102, D103, D107

"""The verdict replay CLI (barge-in plan, section 11.7), with a fake frontend."""

from __future__ import annotations

import contextlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from _fbv_voice_fakes import PACKAGE_DIR, REPO_ROOT, FakeChatClient, delegate_response, text_response

from examples.frontend_backend_verdict.voice.agent.runner import AgentClients
from examples.frontend_backend_verdict.voice.cli import verdict_replay

PROFILE = PACKAGE_DIR / "config" / "profiles" / "browser_demo_frontend_verdict.yaml"
QUERY = "The user is asking for the status of order 1231."
LOG = [
    {"kind": "session_start", "session_id": "s1", "model": "pine-browser"},
    {"kind": "filler", "session_id": "s1", "text": "Let me check on that."},
    {"kind": "barge_in_review", "session_id": "s1", "turn_id": 2, "filler_spoken": True},
    {
        "kind": "barge_in_verdict",
        "session_id": "s1",
        "turn_id": 2,
        "utterance": "Okay .",
        "merged_text": "status of order one two three one? Okay .",
        "verdict": "new",
        "reason": "model",
        "running_query": QUERY,
    },
    {"kind": "barge_in_verdict", "session_id": "s1", "turn_id": 3, "verdict": "continue", "reason": "no_transcript"},
    {"kind": "session_start", "session_id": "s2", "model": "other"},
    {"kind": "barge_in_verdict", "session_id": "s2", "utterance": "x", "merged_text": "y x", "running_query": "q"},
]


class VerdictReplayTests(unittest.TestCase):
    def setUp(self) -> None:
        patcher = mock.patch.dict(os.environ, {"NVIDIA_API_KEY": "test-key"})
        patcher.start()
        self.addCleanup(patcher.stop)
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.events = self.root / "events.jsonl"
        self.events.write_text("".join(json.dumps(r) + "\n" for r in LOG), encoding="utf-8")

    def run_replay(self, *extra: str, frontend: FakeChatClient) -> tuple[dict, list[dict]]:
        out = self.root / "out.jsonl"
        clients = AgentClients(backend=FakeChatClient(), frontend=frontend)
        argv = ["--events", str(self.events), "--config", str(PROFILE), "--model", "pine-browser", "--out", str(out)]
        with contextlib.redirect_stderr(io.StringIO()):
            report = verdict_replay.main([*argv, *extra], clients=clients)
        return report, [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]

    def test_guard_on_and_off(self) -> None:
        report, results = self.run_replay(
            "--guard", "on", frontend=FakeChatClient([delegate_response(QUERY, task="new")])
        )
        self.assertEqual(len(results), 1)  # the no-transcript verdict and the other model's session are skipped
        (result,) = results
        self.assertEqual(
            (result["old_verdict"], result["verdict"], result["reason"]), ("new", "continue", "same_query")
        )
        self.assertEqual((report["log_changed"], report["same_query_overrides"]), (1, 1))
        report, results = self.run_replay(
            "--guard", "off", frontend=FakeChatClient([delegate_response(QUERY, task="new")])
        )
        self.assertEqual((results[0]["verdict"], results[0]["model_task"]), ("new", "new"))
        self.assertFalse(report["guard"])

    def test_the_probe_sees_the_recorded_context(self) -> None:
        frontend = FakeChatClient([delegate_response(QUERY, task="continue")])
        self.run_replay(frontend=frontend)
        system, *_, user = frontend.calls[0]["messages"]
        self.assertIn(QUERY, system.content)
        self.assertIn('you told the user: "Let me check on that."', system.content)
        self.assertIn('The user has just said: "Okay ."', system.content)
        self.assertEqual(user.content, "status of order one two three one? Okay .")

    def test_cases_file_accuracy(self) -> None:
        cases = REPO_ROOT / "tests" / "unit" / "frontend_backend_verdict" / "fixtures" / "verdict_cases.jsonl"
        count = len(cases.read_text(encoding="utf-8").splitlines())
        frontend = FakeChatClient([text_response("It is running.")] * count)
        out = self.root / "cases.jsonl"
        clients = AgentClients(backend=FakeChatClient(), frontend=frontend)
        with contextlib.redirect_stderr(io.StringIO()):
            report = verdict_replay.main(
                ["--cases", str(cases), "--config", str(PROFILE), "--out", str(out)], clients=clients
            )
        # A direct answer is always "new": only the cases that expect "new" are right.
        expected_new = sum(json.loads(line)["expected"] == "new" for line in cases.read_text().splitlines())
        self.assertEqual((report["cases_expected"], report["cases_correct"]), (count, expected_new))

    def test_a_profile_without_the_verdict_is_refused(self) -> None:
        with self.assertRaises(SystemExit):
            verdict_replay.main(
                [
                    "--events",
                    str(self.events),
                    "--config",
                    str(PACKAGE_DIR / "config" / "profiles" / "browser_demo.yaml"),
                ],
                clients=AgentClients(backend=FakeChatClient(), frontend=FakeChatClient()),
            )
