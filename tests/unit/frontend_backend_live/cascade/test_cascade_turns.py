# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D103

"""Caller turns: heard at the stop, decided one at a time, carried out."""

import asyncio

from examples.frontend_backend_live.models.decision import TalkerDecision
from tests.unit.contracts.delegation_port import ScriptedDelegationPort
from tests.unit.frontend_backend_live.cascade.doubles import (
    NO_WORK_IN_PROGRESS,
    Call,
    answered_events,
    settle,
)


def speak(text):
    return TalkerDecision("speak", text)


def delegate(lead_in=""):
    return TalkerDecision("delegate", lead_in)


class RoutingTests(Call):
    async def test_a_spoken_reply_is_said_and_nothing_is_delegated(self):
        await self.call([speak("We open at seven.")])
        await self.turn("When do you open?")
        self.assertEqual(self.spoken, ["We open at seven."])
        self.assertEqual(self.port.requests, [])
        self.assertEqual(self.frontend.calls[0]["history"], [{"role": "user", "content": "When do you open?"}])
        self.assertIn("You are Remy.", self.frontend.calls[0]["instructions"])
        self.assertIn(NO_WORK_IN_PROGRESS, self.frontend.calls[0]["instructions"])  # the port reports events
        self.assertEqual(self.memory()[1:], [("user", "When do you open?"), ("assistant", "We open at seven.")])

    async def test_a_delegation_says_its_lead_in_then_the_worded_answer(self):
        await self.call([delegate("Let me look that up."), speak("A latte is four fifty.")])
        await self.turn("How much is a latte?")
        self.assertEqual(self.spoken, ["Let me look that up."])
        self.emit("item_1", *answered_events("r1", "Latte: $4.50."))
        await self.said(2)
        self.assertEqual(self.spoken, ["Let me look that up.", "A latte is four fifty."])
        wording = self.frontend.calls[1]
        self.assertEqual(wording["actions"], ("speak",))
        self.assertIn("Latte: $4.50.", wording["history"][-1]["content"])
        self.assertIn('you already told the caller: "Let me look that up."', wording["history"][-1]["content"])
        self.assertEqual(
            self.memory()[-2:],
            [("user", "Backend result (reference data): Latte: $4.50."), ("assistant", "A latte is four fifty.")],
        )

    async def test_a_delegation_with_no_lead_in_is_acknowledged_with_the_fixed_line_at_once(self):
        await self.call([delegate()])
        await self.turn("Book a table.")
        self.assertEqual(self.spoken, [self.lines.lead_in])  # acknowledgement_delay_ms 0 asks at once

    async def test_the_talker_reads_the_work_note_while_work_is_open(self):
        await self.call([delegate("One moment."), speak("Still on it.")])
        await self.turn("Book a table.")
        await self.turn("How is it going?")
        line = self.frontend.calls[1]["instructions"]
        self.assertIn('1. waiting to start: "Book a table."', line)
        self.assertNotIn(NO_WORK_IN_PROGRESS, line)

    async def test_a_client_delegation_is_never_open_work_and_the_talker_reads_no_work_line(self):
        await self.call(
            [delegate("One moment."), speak("Let me look.")], port=ScriptedDelegationPort(reports_events=False)
        )
        await self.turn("What is the weather?")
        await self.turn("Any update?")
        self.assertNotIn("Work in progress", self.frontend.calls[1]["instructions"])
        self.assertEqual(self.session.worknote.open(), [])

    async def test_wait_says_and_delegates_nothing(self):
        # The talker's own floor turns `wait` into a delegation unless the prompt set declares it; the cascade's
        # action table is tested with a talker that answers `wait`.
        class Waits:
            async def decide(self, instructions, history, work_in_progress="", *, work_open=False):
                return TalkerDecision("wait")

        await self.call([])
        self.session.agent.talker = Waits()
        await self.turn("Hold on a second.")
        self.assertEqual((self.spoken, self.port.requests), ([], []))
        self.assertIn("talker.waited", self.kinds())


class TurnEndTests(Call):
    async def test_the_turn_is_heard_at_the_stop_and_either_frame_order_decides_it_once(self):
        for order in ("stop first", "context only"):
            with self.subTest(order):
                await self.call([speak("Sure.")])
                self.buffer.turn("A latte, please.")
                if order == "stop first":
                    self.session.caller_stopped()
                    self.assertEqual(self.session.turns.waiting, ["A latte, please."])
                    self.assertTrue(self.session.floor.turns_busy)
                await self.session.on_turn()
                await self.session.on_turn()  # a second run finds nothing waiting
                self.assertEqual((self.spoken, len(self.frontend.calls)), (["Sure."], 1))
                self.assertFalse(self.session.floor.turns_busy)

    async def test_the_seed_is_history_and_pipecats_notes_never_reach_the_talker(self):
        await self.call([speak("Hello.")], seed=[{"role": "user", "content": "Earlier."}])
        self.buffer.items.append({"role": "developer", "content": "Briefly ask them to repeat what they said."})
        await self.session.on_turn()  # Pipecat's run after a turn with no words: nothing to decide
        self.assertEqual(self.frontend.calls, [])
        await self.turn("Hi.")
        call = self.frontend.calls[0]
        self.assertNotIn("Briefly ask them", call["instructions"])
        self.assertEqual(call["history"], [{"role": "user", "content": "Earlier."}, {"role": "user", "content": "Hi."}])

    async def test_a_turn_cut_off_while_the_talker_decides_is_decided_with_the_next_turn(self):
        slow = asyncio.get_running_loop().create_future()
        await self.call([slow, delegate("One moment.")])
        self.buffer.turn("My user ID is sofia kim")
        self.session.caller_stopped()
        deciding = asyncio.create_task(self.session.on_turn())
        await settle(lambda: self.frontend.calls)
        deciding.cancel()  # the caller speaks again: Pipecat cancels the decision
        with self.assertRaises(asyncio.CancelledError):
            await deciding
        await self.turn("seven two eight seven")
        pieces = [
            {"role": "user", "content": "My user ID is sofia kim"},
            {"role": "user", "content": "seven two eight seven"},
        ]
        self.assertEqual(self.frontend.calls[1]["history"], pieces)
        transcript = self.port.requests[0][0]["content"][0]["text"]
        self.assertIn("Caller: My user ID is sofia kim\nCaller: seven two eight seven", transcript)
        self.assertEqual(self.spoken, ["One moment."])

    async def test_a_result_ready_during_a_caller_turn_is_said_after_the_reply(self):
        await self.call([delegate("One moment."), speak("Yes, we have oat milk."), speak("Your latte is added.")])
        await self.turn("Add a latte.")
        self.buffer.turn("Do you have oat milk?")
        self.session.caller_started()
        self.emit("item_1", *answered_events("r1", "Latte added."))  # ready while the caller speaks
        await asyncio.sleep(0.05)
        self.assertEqual(self.spoken, ["One moment."])
        self.session.caller_stopped()
        await asyncio.sleep(0.05)
        self.assertEqual(self.spoken, ["One moment."])  # the turn waits for its decision: still held
        await self.session.on_turn()
        await self.said(3)
        self.assertEqual(self.spoken, ["One moment.", "Yes, we have oat milk.", "Your latte is added."])


class OneDecisionTests(Call):
    async def test_an_instruction_during_a_caller_decision_waits_and_the_turn_is_decided_once(self):
        slow = asyncio.get_running_loop().create_future()
        await self.call([slow, speak("Understood.")])
        self.buffer.turn("Book a table.")
        self.session.caller_stopped()
        deciding = asyncio.create_task(self.session.on_turn())
        await settle(lambda: self.frontend.calls)
        instructing = asyncio.create_task(self.session.on_instruction("Speak only Spanish."))
        await asyncio.sleep(0.05)
        self.assertEqual(len(self.frontend.calls), 1)  # the instruction waits for the caller's decision
        slow.set_result(delegate("Un momento."))
        await deciding
        await instructing
        self.assertEqual(len(self.frontend.calls), 2)
        self.assertEqual(len(self.port.requests), 1)  # the caller's turn was decided once
        self.assertEqual(self.spoken, ["Un momento.", "Understood."])

    async def test_an_instruction_covers_the_caller_turns_waiting_for_it(self):
        await self.call([speak("Hecho.")])
        self.buffer.turn("Book a table.")
        self.session.caller_stopped()  # heard; its context frame has not come yet
        await self.session.on_instruction("Speak only Spanish.")
        await self.session.on_turn()  # the context frame: nothing waits any more
        self.assertEqual((len(self.frontend.calls), self.spoken), (1, ["Hecho."]))
        self.assertIn("Speak only Spanish.", self.frontend.calls[0]["instructions"])

    async def test_a_turn_heard_while_an_instruction_is_decided_is_decided_after_it(self):
        # The instruction's decision runs in the application's task, so the caller's new turn interrupts nothing: the
        # talker never read that turn, so it waits for its own context frame's decision instead of being dropped.
        slow = asyncio.get_running_loop().create_future()
        await self.call([slow, speak("A table for two, booked.")])
        instructing = asyncio.create_task(self.session.on_instruction("Speak only English."))
        await settle(lambda: self.frontend.calls)
        self.buffer.turn("Book a table for two.")
        self.session.caller_started()
        self.session.caller_stopped()
        deciding = asyncio.create_task(self.session.on_turn())
        slow.set_result(speak("Understood."))
        await instructing
        await deciding
        self.assertEqual(self.spoken, ["Understood.", "A table for two, booked."])
        self.assertIn({"role": "user", "content": "Book a table for two."}, self.frontend.calls[1]["history"])
        self.assertEqual(self.session.turns.waiting, [])

    async def test_the_normalizers_rewrite_lands_on_its_own_turn_when_a_turn_is_heard_meanwhile(self):
        rewritten = asyncio.get_running_loop().create_future()

        class Normalizer:
            def wanted(self, text):
                return True

            async def normalize(self, history, text):
                return await rewritten, {}

        await self.call([delegate("One moment.")])
        self.session.agent.normalizer = Normalizer()
        self.buffer.turn("my id is sofia kim")
        self.session.caller_stopped()
        deciding = asyncio.create_task(self.session.on_turn())
        await asyncio.sleep(0.01)
        self.buffer.turn("seven two eight seven")  # heard while the normalizer runs (nothing interrupted it)
        self.session.caller_stopped()
        rewritten.set_result("my id is sofia_kim")
        await deciding
        callers = [m for m in self.frontend.calls[0]["history"] if m["role"] == "user"]
        self.assertEqual(
            callers,
            [{"role": "user", "content": "my id is sofia_kim"}, {"role": "user", "content": "seven two eight seven"}],
        )
        # The request the work note and the display read is both turns, the rewritten one in its place.
        self.assertEqual(self.display.calls[0], ("started", "item_1", "my id is sofia_kim seven two eight seven"))
        self.assertEqual(self.session.turns.waiting, [])
