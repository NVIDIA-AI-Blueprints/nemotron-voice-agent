# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D103

"""Tests for the harness's conductor on a session double: its prompt, its door, what it leaves and its sessions."""

import asyncio
import json
import unittest

from examples.frontend_backend_live.delegation_core.backend.harness.conductor import HarnessConductor
from examples.frontend_backend_live.delegation_core.backend.port import ConductorInput, Start, Steer
from tests.unit.contracts.backend import FakeDecisionTarget, a_request, an_ended_work, an_open_work
from tests.unit.delegation_core.backend.harness.doubles import FakeHost, plain, settings, until

USAGE = {"totalTokens": 50}
EARLIER = ({"type": "message", "role": "user", "content": "Earlier: book a table."},)


def given(*requests, works=(), conversation=(), new_from=0):
    conversation = (*conversation, *(item for request in requests for item in request.input))
    return ConductorInput(tuple(requests), tuple(works), conversation, new_from)


class ConductorTests(unittest.IsolatedAsyncioTestCase):
    def make(self, harness=None, **block):
        self.host = FakeHost(harness or settings(**block))
        self.conductor = HarnessConductor(self.host)
        self.target = FakeDecisionTarget()
        self.sessions = self.host.runtime.client.sessions

    def prompted(self):
        """The session whose prompt is out: the newest one in use with a prompt that has not ended."""
        return next(
            (session for session in reversed(self.sessions) if session.outstanding and not session.retired), None
        )

    async def deciding(self, given_input):
        """A ``decide()`` running, once its prompt went out: the task and its session."""
        task = asyncio.ensure_future(self.conductor.decide(given_input, self.target.place))
        await until(self.prompted)
        return task, self.prompted()

    async def placed(self, given_input):
        """One ``decide()`` whose turn starts work for its first arrival: the session it ran in."""
        task, session = await self.deciding(given_input)
        await self.conductor.door.call("start_work", {"delegations": [given_input.arrivals[0].delegation_id]})
        session.end(session.last_prompt(), usage=USAGE)
        await task
        return session

    async def test_one_prompt_per_decide_its_calls_placed_through_its_door_and_its_text_ignored(self):
        self.make()
        works = (an_open_work(), an_ended_work())
        task, session = await self.deciding(given(a_request("d_new", "Find teas."), works=works, conversation=EARLIER))
        [prompt] = session.texts("prompt")
        for part in (
            "Earlier: book a table.",  # a fresh session reads the whole conversation
            "New requests to place:\n[d_new] Find teas.",
            "[W1] running, carried by d_old; requests: [d_old] Find teas.",
            "task W1.1: running for 3 s; steps: get_menu: done; newest note: Looking it up.",
            "[W0] answered, carried by d_done",
        ):
            self.assertIn(part, prompt)
        folder = self.host.runtime.client.folders[0]
        self.assertEqual((folder / ".hermes.md").read_text().strip(), self.host.vocabulary.instructions(3))
        self.assertEqual(self.host.runtime.client.servers[0].name, "c1")  # its own door: the control tools only
        tools = [tool["name"] for tool in self.conductor.door.tools()]
        self.assertEqual(tools, ["start_work", "steer_work", "cancel_work", "answer_status"])
        malformed = await self.conductor.door.call("steer_work", {"delegations": ["d_new"]})
        self.assertEqual(json.loads(malformed), {"error": "steer_work: missing work_id"})  # answered and skipped
        outcome = await self.conductor.door.call("steer_work", {"work_id": "W1", "delegations": ["d_new"]})
        self.assertEqual(outcome, "Sent to W1; it applies at the next step.")
        session.say("m1", "Placed ", "it.")  # never used
        session.end(1, usage=USAGE)
        await task
        self.assertEqual(self.target.decisions, [Steer("W1", ("d_new",))])
        self.assertIs(await self.placed(given(a_request("d_2", "Hours?"), conversation=EARLIER, new_from=1)), session)
        self.assertNotIn("Earlier: book a table.", session.texts("prompt")[1])  # the session has read it

    async def test_a_plain_agent_reads_the_conductor_prompt_atop_a_fresh_sessions_first_prompt_only(self):
        self.make(plain())
        task, session = await self.deciding(given(a_request("d_1", "Hi.")))
        instructions = self.host.vocabulary.instructions(3)
        self.assertTrue(session.texts("prompt")[0].startswith(instructions + "\n\n"))
        self.assertEqual(list(self.host.runtime.client.folders[0].iterdir()), [])
        await self.conductor.door.call("start_work", {"delegations": ["d_1"]})
        session.end(1)  # a plain agent reports no usage: its conductor needs none
        await task
        self.assertIs(await self.placed(given(a_request("d_2", "Hours?"))), session)
        self.assertNotIn(instructions, session.texts("prompt")[1])
        self.assertEqual(self.target.decisions, [Start(("d_1",)), Start(("d_2",))])

    async def test_a_turn_that_ended_any_other_way_or_a_lost_runtime_gives_the_next_decide_one_fresh_session(self):
        self.make()
        task, session = await self.deciding(given(a_request("d_1", "Teas?")))
        session.end(1, "refusal")  # e.g. Hermes does not know the session: it is never used again
        await task
        self.assertEqual((session.retired, self.target.decisions), (True, []))  # d_1 is left to the IM
        fresh = await self.placed(given(a_request("d_2", "Hours?")))
        self.assertIsNot(fresh, session)
        self.host.runtime.client.alive = False  # the runtime process ended between two decisions
        await self.placed(given(a_request("d_3", "Teas?")))
        self.assertEqual((len(self.sessions), fresh.retired), (3, True))  # one fresh session, not two racing

    async def test_a_turn_past_its_limit_is_cancelled_and_its_session_kept_only_when_the_turn_ends_cancelled(self):
        # A turn that ends end_turn after its cancel went out may have crossed it: the agent got the cancel idle.
        ends = {
            "cancelled": {"stop": "cancelled"},
            "end_turn": {},
            "lost": {"stop": None, "error": "lost"},
            "never": None,
        }
        for name, end in ends.items():
            with self.subTest(name):
                self.make(conductor_timeout_seconds=0.2)
                task, session = await self.deciding(given(a_request("d_1", "Teas?")))
                await until(lambda session=session: session.cancels == 1)
                self.assertFalse(task.done())  # it reads on until the turn ends, within stop_timeout_seconds
                if end is not None:
                    session.end(1, **end)
                await task
                self.assertEqual((session.retired, self.target.decisions), (name != "cancelled", []))
                if name == "cancelled":
                    self.assertIs(await self.placed(given(a_request("d_2", "Hours?"))), session)

    async def test_a_cancelled_decide_waits_for_its_turn_and_the_next_decide_takes_the_same_session(self):
        self.make()
        task, session = await self.deciding(given(a_request("d_0", "Book.")))
        task.cancel()  # as the turn ends: no cancel goes to a turn that has ended
        session.end(1, usage=USAGE)
        await asyncio.gather(task, return_exceptions=True)
        self.assertEqual((session.cancels, session.retired), (0, False))
        task, _ = await self.deciding(given(a_request("d_1", "Teas?")))
        task.cancel()  # the IM's busy rule: a decide() that placed nothing restarts
        await until(lambda: session.cancels == 1)
        self.assertFalse(task.done())  # the next decide() must not prompt while the turn runs
        session.end(2, "cancelled")
        await asyncio.gather(task, return_exceptions=True)
        self.assertIs(await self.placed(given(a_request("d_1", "Teas?"), a_request("d_2", "Green."))), session)

    async def test_a_cancelled_decide_whose_turn_never_ends_or_that_is_cancelled_again_retires_its_session(self):
        for again in (False, True):  # a second cancel (the IM's close) ends the wait while the turn runs
            with self.subTest(again=again):
                self.make()
                session = await self.placed(given(a_request("d_0", "Book."), conversation=EARLIER))
                newer = given(a_request("d_1", "Teas?"), conversation=EARLIER, new_from=1)
                task, _ = await self.deciding(newer)
                task.cancel()  # the double's turn does not end within stop_timeout_seconds
                if again:
                    await until(lambda session=session: session.cancels == 1)
                    task.cancel()
                    session.end(2, "cancelled")
                await asyncio.gather(task, return_exceptions=True)
                self.assertEqual((session.cancels, session.retired), (1, True))
                fresh = await self.placed(newer)
                self.assertIsNot(fresh, session)
                self.assertIn("Earlier: book a table.", fresh.texts("prompt")[0])  # a fresh session reads it all


if __name__ == "__main__":
    unittest.main()
