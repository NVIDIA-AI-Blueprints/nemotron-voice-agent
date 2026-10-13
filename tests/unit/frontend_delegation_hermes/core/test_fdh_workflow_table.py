# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: D100, D101, D102, D103, D107

"""All 12 rows of workflow.csv against the controller (plan section 3, RC1).

The CSV is the user's spec and is read here, so spec and tests cannot drift apart.
For each row the backend is put into the row's state, the frontend decision the row
prescribes is applied (``delegate(true, …, request)`` or ``delegate(false, …)``, which
never reaches the backend), and the Hermes command the worker receives is compared with
the row's "Hermes Backend Action".
"""

from __future__ import annotations

import csv
import unittest

from _fdh_backend_fakes import WORKFLOW_CSV, Harness, history_after

from examples.frontend_delegation_hermes.core.backend import protocol as proto

STATES = {
    "No Agent Session is started": proto.NO_SESSION,
    "Agent Session is mid-conversation and idle": proto.IDLE,
    "Agent Session is working on a task": proto.WORKING,
}


def load_rows() -> list[dict[str, str]]:
    with WORKFLOW_CSV.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def expected_command(action: str) -> str:
    action = action.strip()
    if action.startswith("agent = AIAgent"):
        return "new_session_run"
    if "run_conversation" in action:
        return "continue_run"
    if "steer" in action or "redirect" in action:
        return "steer"
    if "get_activity_summary" in action:
        return "status"
    if action == "NA":
        return "none"
    raise AssertionError(f"unmapped CSV action {action!r}")


class WorkflowTableTest(unittest.IsolatedAsyncioTestCase):
    def test_csv_has_the_twelve_cells(self) -> None:
        rows = load_rows()
        self.assertEqual(len(rows), 12)
        self.assertEqual({r["Backend State"] for r in rows}, set(STATES))
        self.assertEqual(len({(r["User Queries"], r["Backend State"]) for r in rows}), 12)

    async def test_every_row(self) -> None:
        for index, row in enumerate(load_rows()):
            with self.subTest(row=index + 2, query=row["User Queries"], state=row["Backend State"]):
                await self._check_row(row)

    async def _check_row(self, row: dict[str, str]) -> None:
        state = STATES[row["Backend State"]]
        query = row["User Queries"]
        delegate = row["Frontend Tool Call"].strip() == "delegate"
        # RC1: "status" only for an update question while a task is running.
        request = proto.REQUEST_STATUS if query.startswith("Asks for updates") else proto.REQUEST_TASK
        h = Harness()
        await h.start()
        if state in (proto.IDLE, proto.WORKING):
            await h.delegate(1, "What is the status of order 1234?", seq=1)
            if state == proto.IDLE:
                h.worker.outcome(
                    1, "ok", final_response="It is cancelled.", messages=history_after("x", "It is cancelled.")
                )
                await h.settle()
        self.assertEqual(h.controller.state, state)
        before = list(h.worker.sent)

        text = {"Asks for updates": "What is the update so far?"}.get(
            query.split(" (")[0], "Please check order 1235 too."
        )
        if delegate:
            await h.delegate(2, text, seq=2, request=request)
        else:  # delegate=false: the turn is context only
            h.controller.append([{"seq": 2, "origin": "user", "kind": "user", "text": "Okay sure", "outcome": "heard"}])

        new = [m for m in h.worker.sent if m not in before]
        kinds = [m["type"] for m in new]
        expected = expected_command(row["Hermes Backend Action"])
        if expected == "new_session_run":
            self.assertEqual(kinds, ["run"])
            self.assertEqual((new[0]["epoch"], new[0]["conversation_history"]), (1, []))
            self.assertEqual(h.of_type("action")[-1]["kind"], "start")
        elif expected == "continue_run":
            self.assertEqual(kinds, ["run"])
            self.assertTrue(new[0]["conversation_history"])
            self.assertEqual(h.of_type("action")[-1]["kind"], "continue")
        elif expected == "steer":
            self.assertEqual(kinds, ["steer"])
            self.assertIn(h.of_type("action")[-1]["kind"], ("steer", "redirect"))
        elif expected == "status":
            self.assertEqual(kinds, ["status"])
            self.assertEqual(h.of_type("status")[-1]["turn_id"], 2)
        else:
            self.assertEqual(kinds, [])
            self.assertFalse(delegate)
        await h.controller.close()


if __name__ == "__main__":
    unittest.main()
