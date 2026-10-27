# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D103

"""Tests for the harness's conductor on a session double: its prompt, its door, its rule and its restarts."""

import asyncio
import unittest

from delegation_core.backend.harness.conductor import HarnessConductor
from delegation_core.backend.port import Answer, Cancel, ConductorInput, Start, Steer
from tests.unit.contracts.backend import FakeDecisionTarget, a_request, an_ended_work, an_open_work
from tests.unit.delegation_core.backend.harness.doubles import FakeHost, plain, settings, until

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
        self.assertEqual((folder / ".hermes.md").read_text().strip(), self.host.vocabulary.prompt)
        self.assertEqual(self.host.runtime.client.servers[0].name, "c1")
        await self.conductor.door.call("start_work", {"delegations": ["d_1"]})
        session.end(1, usage=USAGE)
        await task

    async def test_a_plain_agent_reads_the_conductor_prompt_atop_a_fresh_sessions_first_prompt_only(self):
        self.make(plain())
        task, session = await self.deciding(self.given(a_request("d_1", "Hi.")))
        [prompt] = session.texts("prompt")
        self.assertTrue(prompt.startswith(self.host.vocabulary.prompt + "\n\n"))
        self.assertEqual(list(self.host.runtime.client.folders[0].iterdir()), [])
        await self.conductor.door.call("start_work", {"delegations": ["d_1"]})
        session.end(1)  # a plain agent reports no usage: its conductor needs none
        await task
        task = asyncio.ensure_future(self.conductor.decide(self.given(a_request("d_2", "Hours?")), self.target.place))
        await until(lambda: len(session.sent) == 2)
        self.assertNotIn(self.host.vocabulary.prompt, session.texts("prompt")[1])  # the session has read it
        await self.conductor.door.call("start_work", {"delegations": ["d_2"]})
        session.end(2)
        await task
        self.assertEqual(self.target.decisions, [Start(("d_1",)), Start(("d_2",))])

    async def test_a_malformed_call_is_answered_with_its_error_and_skipped(self):
        self.make()
        task, session = await self.deciding(self.given(a_request("d_1", "Hi.")))
        answer = await self.conductor.door.call("steer_work", {"delegations": ["d_1"]})
        self.assertIn("Not applied: steer_work: missing work_id", answer)
        session.end(1, usage=USAGE)
        await task
        self.assertEqual(self.target.decisions, [Start(("d_1",))])  # the rule placed what was left

    async def test_an_arrival_left_unplaced_joins_the_newest_open_work(self):
        self.make()
        given = self.given(a_request("d_1", "Also a latte."), works=(an_open_work(), an_ended_work()))
        task, session = await self.deciding(given)
        session.end(1, usage=USAGE)  # the model placed nothing
        await task
        self.assertEqual(self.target.decisions, [Steer("W1", ("d_1",))])

    async def test_with_no_open_work_the_rule_starts_the_arrivals_together(self):
        self.make()
        task, session = await self.deciding(self.given(a_request("d_1", "Teas?"), a_request("d_2", "And hours?")))
        session.end(1, None, None, "ConnectionError: lost")  # a failed turn places nothing
        await task
        self.assertEqual(self.target.decisions, [Start(("d_1", "d_2"))])

    async def test_a_work_cancelled_in_the_call_is_never_joined(self):
        self.make()
        task, session = await self.deciding(self.given(a_request("d_stop", "Stop that."), works=(an_open_work(),)))
        await self.conductor.door.call("cancel_work", {"work_id": "W1"})
        session.end(1, usage=USAGE)
        await task
        self.assertEqual(self.target.decisions, [Cancel("W1"), Start(("d_stop",))])

    async def test_a_decision_naming_work_it_was_not_shown_leaves_its_arrival_to_the_rule(self):
        # The IM refuses a Steer to a Work or a task it does not know, so that arrival is still unplaced: the rule joins
        # it to the open Work instead of the IM starting a second Work beside it.
        for name, arguments in (
            ("an unknown work", {"work_id": "W9", "delegations": ["d_1"]}),
            ("an unknown task", {"work_id": "W1", "delegations": ["d_1"], "task_id": "W1.7"}),
        ):
            with self.subTest(name):
                self.make()
                task, session = await self.deciding(self.given(a_request("d_1", "Also tea."), works=(an_open_work(),)))
                await self.conductor.door.call("steer_work", arguments)
                session.end(1, usage=USAGE)
                await task
                self.assertEqual(self.target.decisions[1:], [Steer("W1", ("d_1",))])

    async def test_a_turn_the_runtime_did_not_end_as_a_turn_retires_the_session(self):
        # A refusal (Hermes does not know the session) or a cancel the conductor did not send: never reused.
        self.make()
        task, session = await self.deciding(self.given(a_request("d_1", "Teas?")))
        session.end(1, "refusal")
        await task
        self.assertTrue(session.retired)
        self.assertEqual(self.target.decisions, [Start(("d_1",))])
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

    async def test_an_answer_places_its_arrival(self):
        self.make()
        task, session = await self.deciding(self.given(a_request("d_q", "How is it going?"), works=(an_open_work(),)))
        await self.conductor.door.call("answer_status", {"delegation_id": "d_q", "text": "Still looking it up."})
        session.end(1, usage=USAGE)
        await task
        self.assertEqual(self.target.decisions, [Answer("d_q", "Still looking it up.")])

    async def test_a_turn_past_its_limit_is_cancelled_retired_and_its_arrivals_placed_by_the_rule(self):
        self.make(conductor_timeout_seconds=0.2)
        task, session = await self.deciding(self.given(a_request("d_1", "Teas?")))
        await task
        self.assertEqual(session.cancels, 1)
        self.assertTrue(session.retired)
        self.assertEqual(self.target.decisions, [Start(("d_1",))])

    async def test_a_cancelled_decide_cancels_the_turn_and_the_next_session_reads_the_whole_conversation(self):
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
        self.assertNotIn("Earlier: book a table.", session.texts("prompt")[1])  # it read that before
        task.cancel()  # the IM's busy rule: a decide() that placed nothing restarts
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

    async def test_a_session_that_read_the_conversation_gets_only_what_is_new(self):
        self.make()
        conversation = (
            {"type": "message", "role": "user", "content": "First request."},
            {"type": "message", "role": "assistant", "content": "First answer."},
            {"type": "message", "role": "user", "content": "Second request."},
        )
        task, session = await self.deciding(self.given(a_request("d_1", "One"), conversation=conversation[:2]))
        await self.conductor.door.call("start_work", {"delegations": ["d_1"]})
        session.end(1, usage=USAGE)
        await task
        task = asyncio.ensure_future(
            self.conductor.decide(
                self.given(a_request("d_2", "Two"), conversation=conversation, new_from=2), self.target.place
            )
        )
        await until(lambda: len(session.sent) == 2)
        second = session.texts("prompt")[1]
        self.assertIn("Second request.", second)
        self.assertNotIn("First request.", second)
        session.end(2, usage=USAGE)
        await task


if __name__ == "__main__":
    unittest.main()
