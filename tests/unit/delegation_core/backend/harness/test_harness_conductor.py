# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D103

"""Tests for the harness's conductor on a session double: its prompt, its door, what it leaves and its restarts.

Under the real IM, the IM's steer rule places what a turn leaves, against the Works as the turn left them.
"""

import asyncio
import json
import unittest

from delegation_core.backend.harness.conductor import HarnessConductor
from delegation_core.backend.port import ConductorInput, Start, Steer
from tests.unit.contracts.backend import FakeDecisionTarget, a_request, an_open_work
from tests.unit.delegation_core.backend.harness.doubles import FakeHost, plain, settings, until
from tests.unit.delegation_core.helpers import IMCase, ScriptedAdapter, answer, ask, finished_items, read_all, wait

USAGE = {"totalTokens": 50}


class ConductorTests(unittest.IsolatedAsyncioTestCase):
    def make(self, harness=None, **block):
        self.host = FakeHost(harness or settings(**block))
        self.conductor = HarnessConductor(self.host)
        self.target = FakeDecisionTarget()
        return self.conductor

    def given(self, *requests, works=(), conversation=None, new_from=0):
        conversation = (
            tuple(item for request in requests for item in request.input) if conversation is None else conversation
        )
        return ConductorInput(tuple(requests), tuple(works), tuple(conversation), new_from)

    async def deciding(self, given):
        task = asyncio.ensure_future(self.conductor.decide(given, self.target.place))
        await until(lambda: self.host.runtime.client.sessions and self.host.runtime.client.sessions[-1].sent)
        return task, self.host.runtime.client.sessions[-1]

    async def test_one_prompt_per_decide_its_calls_placed_through_the_door_its_text_ignored(self):
        self.make()
        task, session = await self.deciding(self.given(a_request("d_new", "Find teas."), works=(an_open_work(),)))
        [prompt] = session.texts("prompt")
        self.assertIn("[d_new] Find teas.", prompt)
        self.assertIn("[W1] running, carried by d_old", prompt)
        self.assertIn("get_menu: done", prompt)
        door = self.conductor.door
        self.assertEqual(
            [tool["name"] for tool in door.tools()], ["start_work", "steer_work", "cancel_work", "answer_status"]
        )
        outcome = await door.call("steer_work", {"work_id": "W1", "delegations": ["d_new"]})
        self.assertEqual(outcome, "Sent to W1; it applies at the next step.")
        session.say("m1", "Placed ", "it.")  # never used
        session.end(1, usage=USAGE)
        await task
        self.assertEqual(self.target.decisions, [Steer("W1", ("d_new",))])

    async def test_the_instructions_are_the_conductor_prompt_in_its_own_folder_never_the_envelope(self):
        self.make()
        task, session = await self.deciding(self.given(a_request("d_1", "Hi.")))
        folder = self.host.runtime.client.folders[0]
        instructions = (folder / ".hermes.md").read_text().strip()
        self.assertEqual(instructions, self.host.vocabulary.instructions(3))  # the limit filled in: max_running 3
        self.assertIn("3 tasks at once", instructions)
        self.assertEqual(self.host.runtime.client.servers[0].name, "c1")
        await self.conductor.door.call("start_work", {"delegations": ["d_1"]})
        session.end(1, usage=USAGE)
        await task

    async def test_a_plain_agent_reads_the_conductor_prompt_atop_a_fresh_sessions_first_prompt_only(self):
        self.make(plain())
        task, session = await self.deciding(self.given(a_request("d_1", "Hi.")))
        [prompt] = session.texts("prompt")
        self.assertTrue(prompt.startswith(self.host.vocabulary.instructions(3) + "\n\n"))
        self.assertEqual(list(self.host.runtime.client.folders[0].iterdir()), [])
        await self.conductor.door.call("start_work", {"delegations": ["d_1"]})
        session.end(1)  # a plain agent reports no usage: its conductor needs none
        await task
        task = asyncio.ensure_future(self.conductor.decide(self.given(a_request("d_2", "Hours?")), self.target.place))
        await until(lambda: len(session.sent) == 2)
        self.assertNotIn(self.host.vocabulary.instructions(3), session.texts("prompt")[1])  # the session has read it
        await self.conductor.door.call("start_work", {"delegations": ["d_2"]})
        session.end(2)
        await task
        self.assertEqual(self.target.decisions, [Start(("d_1",)), Start(("d_2",))])

    async def test_a_malformed_call_is_answered_with_its_error_and_skipped(self):
        self.make()
        task, session = await self.deciding(self.given(a_request("d_1", "Hi.")))
        answer = await self.conductor.door.call("steer_work", {"delegations": ["d_1"]})
        self.assertEqual(json.loads(answer), {"error": "steer_work: missing work_id"})
        session.end(1, usage=USAGE)
        await task
        self.assertEqual(self.target.decisions, [])  # d_1 is left to the IM

    async def test_a_turn_the_runtime_did_not_end_as_a_turn_retires_the_session(self):
        # A refusal (Hermes does not know the session): never reused.
        self.make()
        task, session = await self.deciding(self.given(a_request("d_1", "Teas?")))
        session.end(1, "refusal")
        await task
        self.assertTrue(session.retired)
        self.assertEqual(self.target.decisions, [])
        task, fresh = await self.deciding(self.given(a_request("d_2", "Hours?")))
        self.assertIsNot(fresh, session)
        fresh.end(1, usage=USAGE)
        await task

    async def test_a_lost_runtime_is_replaced_by_one_fresh_session(self):
        self.make()
        task, session = await self.deciding(self.given(a_request("d_1", "Teas?")))
        await self.conductor.door.call("start_work", {"delegations": ["d_1"]})
        session.end(1, usage=USAGE)
        await task
        client = self.host.runtime.client
        client.alive = False  # the runtime process ended between two decisions
        task = asyncio.ensure_future(self.conductor.decide(self.given(a_request("d_2", "Hours?")), self.target.place))
        await until(lambda: len(client.sessions) >= 2 and client.sessions[-1].sent)
        await asyncio.sleep(0.05)
        self.assertEqual(len(client.sessions), 2)  # one fresh session, not two racing to be the conductor's
        self.assertTrue(session.retired)
        await self.conductor.door.call("start_work", {"delegations": ["d_2"]})
        client.sessions[-1].end(1, usage=USAGE)
        await task
        self.assertEqual(self.target.decisions, [Start(("d_1",)), Start(("d_2",))])

    async def test_a_turn_past_its_limit_that_never_ends_once_cancelled_is_retired_its_arrivals_left_to_the_im(self):
        self.make(conductor_timeout_seconds=0.2)  # the double's turn does not end within stop_timeout_seconds
        task, session = await self.deciding(self.given(a_request("d_1", "Teas?")))
        await task
        self.assertEqual(session.cancels, 1)
        self.assertTrue(session.retired)
        self.assertEqual(self.target.decisions, [])

    async def test_a_turn_past_its_limit_is_cancelled_and_once_it_ends_its_session_takes_the_next_decide(self):
        self.make(conductor_timeout_seconds=0.2)
        task, session = await self.deciding(self.given(a_request("d_1", "Teas?")))
        await until(lambda: session.cancels == 1)
        self.assertFalse(task.done())  # it reads on until the turn ends
        session.end(1, "cancelled")
        await task
        task = asyncio.ensure_future(self.conductor.decide(self.given(a_request("d_2", "Hours?")), self.target.place))
        await until(lambda: len(session.sent) == 2)
        self.assertEqual((session.retired, len(self.host.runtime.client.sessions)), (False, 1))
        await self.conductor.door.call("start_work", {"delegations": ["d_2"]})
        session.end(2, usage=USAGE)
        await task
        self.assertEqual(self.target.decisions, [Start(("d_2",))])

    async def test_a_cancelled_turn_keeps_its_session_only_when_it_ends_cancelled(self):
        # A turn that ends end_turn after its cancel went out may have crossed it: the agent got the cancel idle.
        ends = (("cancelled", None, True), ("end_turn", None, False), ("refusal", None, False), (None, "lost", False))
        for stop, error, kept in ends:
            with self.subTest(stop=stop, error=error):
                self.make(conductor_timeout_seconds=0.2)
                task, session = await self.deciding(self.given(a_request("d_1", "Teas?")))
                await until(lambda session=session: session.cancels == 1)
                session.end(1, stop, None, error)
                await task
                self.assertEqual(session.retired, not kept)

    async def test_a_turn_that_ended_before_its_cancel_went_out_keeps_its_session(self):
        # No cancel goes to a turn that has ended, so its end_turn is the turn's own end.
        self.make()
        task, session = await self.deciding(self.given(a_request("d_1", "Teas?")))
        task.cancel()  # the busy rule, as the turn ends
        session.end(1, usage=USAGE)
        await asyncio.gather(task, return_exceptions=True)
        self.assertEqual((session.cancels, session.retired), (0, False))

    async def test_a_cancelled_decide_waits_for_its_turn_to_end_and_the_next_decide_takes_the_same_session(self):
        self.make()
        task, session = await self.deciding(self.given(a_request("d_1", "Teas?")))
        task.cancel()  # the IM's busy rule: a decide() that placed nothing restarts
        await until(lambda: session.cancels == 1)
        self.assertFalse(task.done())  # the next decide() must not prompt while the turn runs
        session.end(1, "cancelled")
        await asyncio.gather(task, return_exceptions=True)
        self.assertTrue(task.cancelled())
        given = self.given(a_request("d_1", "Teas?"), a_request("d_2", "Green, please."))
        task = asyncio.ensure_future(self.conductor.decide(given, self.target.place))
        await until(lambda: len(session.sent) == 2)
        self.assertEqual((session.retired, len(self.host.runtime.client.sessions)), (False, 1))
        await self.conductor.door.call("start_work", {"delegations": ["d_1", "d_2"]})
        session.end(2, usage=USAGE)
        await task
        self.assertEqual(self.target.decisions, [Start(("d_1", "d_2"))])

    async def test_a_decide_cancelled_again_while_it_waits_retires_its_session(self):
        # A second cancel (the IM's close) ends the wait while the turn runs: the session is retired, though the turn
        # then ends cancelled.
        self.make()
        task, session = await self.deciding(self.given(a_request("d_1", "Teas?")))
        task.cancel()  # the busy rule
        await until(lambda: session.cancels == 1)
        task.cancel()  # again, while it waits for the turn to end
        session.end(1, "cancelled")
        await asyncio.gather(task, return_exceptions=True)
        self.assertTrue(session.retired)
        task, fresh = await self.deciding(self.given(a_request("d_2", "Hours?")))
        self.assertIsNot(fresh, session)
        await self.conductor.door.call("start_work", {"delegations": ["d_2"]})
        fresh.end(1, usage=USAGE)
        await task
        self.assertEqual(self.target.decisions, [Start(("d_2",))])

    async def test_a_cancelled_decide_whose_turn_never_ends_retires_its_session_and_a_fresh_one_reads_it_all(self):
        self.make()
        conversation = (
            {"type": "message", "role": "user", "content": "Earlier: book a table."},
            {"type": "message", "role": "assistant", "content": "Booked."},
            {"type": "message", "role": "user", "content": "Now: teas?"},
        )
        task, session = await self.deciding(self.given(a_request("d_0", "Book."), conversation=conversation[:2]))
        await self.conductor.door.call("start_work", {"delegations": ["d_0"]})
        session.end(1, usage=USAGE)
        await task
        given = self.given(a_request("d_1", "Teas?"), conversation=conversation, new_from=2)
        task = asyncio.ensure_future(self.conductor.decide(given, self.target.place))
        await until(lambda: len(session.sent) == 2)
        second = session.texts("prompt")[1]  # the session read the conversation: it gets only what is new
        self.assertIn("Now: teas?", second)
        self.assertNotIn("Earlier: book a table.", second)
        task.cancel()  # the busy rule; the double's turn does not end within stop_timeout_seconds
        await asyncio.gather(task, return_exceptions=True)
        self.assertEqual(session.cancels, 1)
        self.assertTrue(session.retired)
        task, fresh = await self.deciding(given)
        self.assertIsNot(fresh, session)
        self.assertIn("Earlier: book a table.", fresh.texts("prompt")[0])  # a fresh session reads it all
        await self.conductor.door.call("start_work", {"delegations": ["d_1"]})
        fresh.end(1, usage=USAGE)
        await task
        self.assertEqual(self.target.decisions, [Start(("d_0",)), Start(("d_1",))])


class UnderTheIMTests(IMCase):
    """The conductor under the real IM, its workers scripted."""

    async def test_a_request_the_turn_leaves_joins_the_work_the_same_turn_started(self):
        # Both requests reach one prompt, whose one call starts work for the first; the IM places the second against the
        # Works as the turn left them, so it joins that Work.
        gate, host = asyncio.Event(), FakeHost()
        conductor = HarnessConductor(host)
        im = self.session(ScriptedAdapter([wait(gate), answer("Friday at seven, for two.")], conductor=conductor))
        first = im.admit(ask("Book a table for Friday."), delegation_id="D1")
        second = im.admit(ask("For two people."), delegation_id="D2")
        client = host.runtime.client
        await until(lambda: client.sessions and client.sessions[0].sent)
        session = client.sessions[0]
        [prompt] = session.texts("prompt")
        self.assertIn("[D1] Book a table for Friday.", prompt)
        self.assertIn("[D2] For two people.", prompt)
        await conductor.door.call("start_work", {"delegations": ["D1"]})
        session.end(1, usage=USAGE)  # the turn leaves D2
        await until(lambda: im.ledger.delegations["D2"].work)
        self.assertEqual([im.ledger.delegations[d].work for d in ("D1", "D2")], ["W1", "W1"])
        placed = [(kind, fields["decision"]) for kind, fields in self.traces if kind.startswith("im.placed")]
        self.assertEqual(placed, [("im.placed", "Start"), ("im.placed_by_rule", "Steer")])
        gate.set()
        await asyncio.wait_for(read_all(first.stream), 5)
        items = finished_items(await asyncio.wait_for(read_all(second.stream), 5))
        answers = [item["content"][0]["text"] for item in items if item.get("phase") == "final_answer"]
        self.assertEqual(answers, ["Friday at seven, for two."])  # the newer request carries the one Work's answer
        self.assertEqual(im.ledger.delegations["D1"].handed_to, "D2")
        self.assertEqual((sorted(im.ledger.works), len(session.sent)), (["W1"], 1))


if __name__ == "__main__":
    unittest.main()
