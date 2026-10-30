# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D103

"""``delegation.hold_answers_while_deciding`` on the real IM with the agent's and the backend LLM's real runs.

A worker that answers while the agent's conductor decides on a newer request is held; the conductor's steer makes the
answer progress and the task runs one more round with the request. The backend LLM's rule conductor decides at once,
so an answer that crosses a request is the answer, and the request starts a follow-up Work, as with the setting off.
"""

import asyncio

from delegation_core.backend.llm import LlmSource
from tests.unit.contracts.backend import text_round
from tests.unit.delegation_core.backend.agent.doubles import (
    FAST_WORKER,
    Model,
    decisions,
    lookup,
    source,
    start,
    steer,
    texts_of,
)
from tests.unit.delegation_core.helpers import IMCase, ask, read_all

# A response that does not end within this many seconds fails its test instead of hanging the suite.
READ_LIMIT_SECONDS = 10


async def read(admission) -> list[dict]:
    return await asyncio.wait_for(read_all(admission.stream), READ_LIMIT_SECONDS)


def gated(gate: asyncio.Event, round_: list) -> list:
    """``round_``, waiting at ``gate`` once it opened: a round still streaming until the test lets it go on."""
    return [round_[0], gate, *round_[1:]]


def answers(events: list[dict]) -> list[str]:
    """The texts of a response's ``final_answer`` items, in order."""
    return [
        event["item"]["content"][0]["text"]
        for event in events
        if event["type"] == "response.output_item.done" and event["item"].get("phase") == "final_answer"
    ]


class HeldAnswerTests(IMCase):
    async def worker_answers_while_deciding(self):
        """W1's worker answers "Black tea." while the conductor decides on D2, which it then steers into W1."""
        round_gate, deciding = asyncio.Event(), asyncio.Event()
        conductor = Model(decisions(start("D1")), [deciding, *decisions(steer("W1", "D2"))])
        self.workers = Model(gated(round_gate, text_round("Black tea.")), text_round("Green tea, as you asked."))
        self.im = self.session(source(conductor, self.workers).new(), hold_answers_while_deciding=True)
        first = self.im.admit(ask("Find teas."), delegation_id="D1")
        await self.turns()
        second = self.im.admit(ask("And a green one?"), delegation_id="D2")
        await self.turns()
        round_gate.set()  # the worker answers while the conductor decides
        await self.turns()
        self.early = answers([event for _, event in self.wire.events])  # what was said before the decision
        deciding.set()
        return await read(first), await read(second)

    async def test_held_and_steered_the_answer_is_progress_and_the_worker_runs_one_more_round(self):
        older, newer = await self.worker_answers_while_deciding()
        self.assertEqual(self.early, [])  # nothing said while the conductor decided
        self.assertEqual((answers(older), answers(newer)), ([], ["Green tea, as you asked."]))
        self.assertEqual(len(self.workers.payloads), 2)  # one more round, which reads the held answer, then D2
        texts = texts_of(self.workers.payloads[-1])
        newer_at = next(n for n, text in enumerate(texts) if "And a green one?" in text)
        self.assertLess(texts.index("Black tea."), newer_at)
        self.assertEqual((self.im.ledger.delegations["D2"].work, list(self.im.ledger.works)), ("W1", ["W1"]))
        self.assertEqual(self.im.ledger.delegations["D1"].handed_to, "D2")

    async def crossed_on_the_backend_llm(self, hold: bool) -> list[list[str]]:
        """D3 waits as steering while the backend LLM's round answers; the timeline of the three responses."""
        self.wire.events.clear()
        first_gate, second_gate = asyncio.Event(), asyncio.Event()
        workers = Model(
            gated(first_gate, text_round("Old.")),
            gated(second_gate, text_round("Black tea.")),
            text_round("Green tea, as you asked."),
        )
        llm = LlmSource(FAST_WORKER, endpoints=lookup, tools=lambda: [], make_provider=lambda endpoint: workers)
        im = self.session(llm.new(), hold_answers_while_deciding=hold)
        first = im.admit(ask("Find teas."), delegation_id="D1")
        await self.turns()
        im.admit(ask("Black ones."), delegation_id="D2")  # redoes the streaming round
        await self.turns()
        third = im.admit(ask("Green ones, actually."), delegation_id="D3")  # waits: the run had its one redo
        await self.turns()
        second_gate.set()
        await read(first)
        self.assertEqual(answers(await read(third)), ["Green tea, as you asked."])
        said = ["Black tea.", "Green tea, as you asked."]  # D2's answer, then D3's follow-up Work's
        self.assertEqual(answers([event for _, event in self.wire.events]), said)
        self.assertEqual((im.ledger.delegations["D3"].work, len(workers.payloads)), ("W2", 3))
        self.assertEqual([s.state for s in im.ledger.steerings.values()], ["applied", "follow_up"])
        follow_ups = [fields for kind, fields in self.traces if kind == "im.follow_up"]
        self.assertEqual(follow_ups[-1], {"work_id": "W1", "delegation_ids": ["D3"]})
        texts = texts_of(workers.payloads[-1])
        self.assertLess(texts.index("Black tea."), next(n for n, t in enumerate(texts) if "Green ones" in t))
        return self.timeline()

    async def test_with_the_backend_llm_an_answer_a_request_crosses_is_the_answer_with_holding_on_or_off(self):
        # The rule conductor places each request at once, so nothing waits for a decision when a round answers: the
        # answer that waiting steering crossed is written, and the steering starts a follow-up Work, on or off.
        self.assertEqual(await self.crossed_on_the_backend_llm(False), await self.crossed_on_the_backend_llm(True))
        self.assertNotIn("im.answer.held", [kind for kind, _ in self.traces])
