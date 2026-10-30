# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D103, D107

"""The harness's worker runs under the real IM with ``delegation.hold_answers_while_deciding``.

A turn that answers while the conductor decides on a newer request is held. Steered into the task, its answer is
progress, and the task runs one more round in a fresh session of the agent, whose first prompt carries the held answer
and then the request; the worker's own session ended with its answer.
"""

import asyncio

from delegation_core.backend.harness.worker import WorkerRun
from delegation_core.backend.port import Start, Steer, TaskContext
from tests.unit.delegation_core.backend.harness.doubles import FakeHost, until
from tests.unit.delegation_core.helpers import IMCase, ask, finished_items, planned, read_all

USAGE = {"inputTokens": 80, "outputTokens": 20, "totalTokens": 100}
# A response that does not end within this many seconds fails its test instead of hanging the suite.
READ_LIMIT_SECONDS = 5


class HostAdapter:
    """The IM's backend here: the harness's worker runs on a fake host, under a scripted conductor."""

    max_running = 3

    def __init__(self, host: FakeHost, conductor):
        self.host, self._conductor = host, conductor

    def conductor(self):
        return self._conductor

    def start(self, task: TaskContext) -> WorkerRun:
        return WorkerRun(task, self.host)

    async def close(self) -> None:
        pass


def said(events: list[dict], phase: str) -> list[str]:
    return [
        item["content"][0]["text"]
        for item in finished_items(events)
        if item["type"] == "message" and item["phase"] == phase
    ]


class HeldAnswerTests(IMCase):
    async def test_a_steered_held_answer_is_progress_and_a_fresh_session_runs_the_newer_request(self):
        deciding = asyncio.Event()

        async def start(given, place):
            await place(Start(("D1",)))

        async def steer_once_let_go(given, place):
            await deciding.wait()  # the conductor's turn: the newer request waits for it
            await place(Steer("W1", ("D2",)))

        host = FakeHost()
        adapter = HostAdapter(host, planned(start, steer_once_let_go))
        im = self.session(adapter, hold_answers_while_deciding=True)
        first = im.admit(ask("Book a table for 6 pm."), delegation_id="D1")
        await until(lambda: host.runtime.client.sessions)
        old = host.runtime.client.sessions[0]
        second = im.admit(ask("Make it 7."), delegation_id="D2")
        await self.turns()
        old.say("m1", "Booked for ", "6 pm.")
        old.end(1, usage=USAGE)  # the turn answers while the conductor decides
        await until(lambda: old.retired)  # the worker's session ended with its answer
        self.assertEqual(im.ledger.tasks["W1.1"].state, "running")  # its answer waits for the decision
        deciding.set()
        await until(lambda: len(host.runtime.client.sessions) == 2)
        fresh = host.runtime.client.sessions[1]
        [prompt] = fresh.texts("prompt")
        self.assertLess(prompt.index("Booked for 6 pm."), prompt.index("Make it 7."))
        fresh.say("m1", "Booked for ", "7 pm.")
        fresh.end(1, usage=USAGE)
        older = await asyncio.wait_for(read_all(first.stream), READ_LIMIT_SECONDS)
        newer = await asyncio.wait_for(read_all(second.stream), READ_LIMIT_SECONDS)
        self.assertEqual((said(older, "final_answer"), said(newer, "final_answer")), ([], ["Booked for 7 pm."]))
        self.assertIn("Booked for 6 pm.", said(older, "commentary"))  # progress, never an answer
        self.assertEqual(im.ledger.delegations["D1"].handed_to, "D2")
        self.assertEqual(im.ledger.steerings["S1"].state, "applied")
