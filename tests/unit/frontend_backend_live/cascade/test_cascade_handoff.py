# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D103, D107

"""The hand-off survives a barge-in, and a delegation is acknowledged."""

import asyncio

from examples.frontend_backend_live.cascade.heard_log import Heard
from examples.frontend_backend_live.models.decision import TalkerDecision
from tests.unit.contracts.delegation_port import ScriptedDelegationPort
from tests.unit.frontend_backend_live.cascade.doubles import (
    WORK,
    Call,
    answered_events,
    completed,
    created,
    failed,
    in_progress,
    message,
    runtime_data,
    settle,
    work_item,
)

# The talker's lead-in for the teas question, its delegation's acknowledgement.
LEAD_IN = "Let me see what teas we have."


class SuspendingPort(ScriptedDelegationPort):
    """A port whose ``delegate`` suspends a few times before the core accepts, as a network round trip does."""

    def __init__(self, steps=6, **kwargs):
        super().__init__(**kwargs)
        self.steps = steps
        self.accepted: list[str] = []

    async def delegate(self, input):
        for _ in range(self.steps):
            await asyncio.sleep(0)
        delegation_id = await super().delegate(input)
        self.accepted.append(delegation_id)
        return delegation_id


class HandOffTests(Call):
    async def test_twelve_cancel_points_lose_no_delegation(self):
        for point in range(12):
            with self.subTest(point=point):
                await self.call([TalkerDecision("delegate", "One moment.")], port=SuspendingPort())
                self.buffer.turn("Book a table for two.")
                self.session.caller_stopped()
                turn = asyncio.create_task(self.session.on_turn())
                for _ in range(point):
                    await asyncio.sleep(0)
                turn.cancel()  # the caller barges in somewhere in the hand-off
                await asyncio.gather(turn, return_exceptions=True)
                await settle(lambda: not self.session.turns.handoffs)
                for delegation_id in self.port.accepted:
                    self.assertIn(delegation_id, self.session.worknote)  # tracked: its events will be read
                callers = [entry.text for entry in self.session.heard_log.entries if entry.kind == Heard.CALLER]
                if self.port.accepted:
                    self.assertEqual(callers, [])  # its words went with it, once
                else:
                    self.assertEqual(callers, ["Book a table for two."])
                    self.assertEqual(self.session.worknote.open(), [])  # no reservation left behind
                await self.session.close()

    async def test_a_turn_decided_during_the_hand_off_reads_the_entry_waiting_to_start(self):
        port = ScriptedDelegationPort(hold=True)
        await self.call([TalkerDecision("delegate", "One moment."), TalkerDecision("speak", "Still on it.")], port=port)
        first = asyncio.create_task(self.turn("Book a table."))
        await port.arrived.wait()
        first.cancel()  # the caller speaks again: Pipecat cancels the turn while its hand-off is in flight
        await asyncio.gather(first, return_exceptions=True)
        await self.turn("Is it booked?")
        self.assertIn(work_item(1, WORK.waiting, "Book a table."), runtime_data(self.frontend.calls[1]))
        port.accept()
        await settle(lambda: "item_1" in self.session.worknote)
        self.assertEqual(self.spoken, ["Still on it."])  # the cut-off turn says no lead-in

    async def test_a_refused_delegation_says_the_refusal_line_and_keeps_its_words(self):
        port = ScriptedDelegationPort(hold=True)
        await self.call([TalkerDecision("delegate", "One moment.")], port=port)
        turn = asyncio.create_task(self.turn("Book a table."))
        await port.arrived.wait()
        port.refuse(RuntimeError("Delegation queue is full"))
        await turn
        self.assertEqual(self.spoken, [self.lines.refusal])  # no lead-in for a refused delegation
        self.assertEqual(self.session.worknote.open(), [])
        self.assertEqual([entry.text for entry in self.session.heard_log.entries][0], "Book a table.")
        self.assertIn("delegation.refused", self.kinds())

    async def test_a_refusal_while_the_caller_speaks_waits_for_them_to_stop(self):
        port = ScriptedDelegationPort(hold=True)
        await self.call([TalkerDecision("delegate", "One moment.")], port=port)
        turn = asyncio.create_task(self.turn("Book a table."))
        await port.arrived.wait()
        self.session.caller_started()
        port.refuse(RuntimeError("Delegation queue is full"))
        await asyncio.sleep(0.05)
        self.assertEqual(self.spoken, [])
        self.session.caller_stopped()
        await turn
        self.assertEqual(self.spoken, [self.lines.refusal])


class AcknowledgementTests(Call):
    async def slow_ack_call(self, *script, delay_ms=80):
        port = ScriptedDelegationPort(acknowledgement_delay_ms=delay_ms)
        await self.call([TalkerDecision("delegate", LEAD_IN), *script], port=port)
        await self.turn("Teas?")
        self.assertEqual(self.spoken, [])  # the acknowledgement waits for its delay

    async def test_a_delayed_acknowledgement_is_said_after_its_delay(self):
        await self.slow_ack_call()
        await self.said(1)
        self.assertEqual(self.spoken, [LEAD_IN])
        self.assertEqual(self.session.utterances.told_text("item_1"), LEAD_IN)

    async def test_its_result_arriving_first_drops_it(self):
        await self.slow_ack_call(TalkerDecision("speak", "We have Earl Grey."))
        await asyncio.sleep(0.04)
        self.emit("item_1", *answered_events("r1", "Earl Grey."))
        await self.said(1)
        await asyncio.sleep(0.1)
        self.assertEqual(self.spoken, ["We have Earl Grey."])
        self.assertIn("acknowledgement.dropped", self.kinds())

    async def test_its_ending_first_drops_it_so_it_never_follows_the_failure_line(self):
        await self.slow_ack_call()
        self.emit("item_1", created("r1"), in_progress("r1"), failed("r1"))
        await self.said(1)
        await asyncio.sleep(0.15)
        self.assertEqual(self.spoken, [self.lines.apology])
        self.assertIn("acknowledgement.dropped", self.kinds())

    async def test_its_handover_drops_it_the_taker_speaks_for_the_work(self):
        await self.slow_ack_call(TalkerDecision("delegate", "Saturday, then."))
        self.buffer.turn("Make it Saturday.")  # a turn whose start the binding missed: nothing dropped yet
        await self.session.on_turn()
        self.assertIn("item_1", self.session.acknowledgements.pending)
        self.emit("item_1", created("r1"), in_progress("r1"))
        self.emit("item_2", created("r2"), in_progress("r2", previous="r1"))
        self.emit(
            "item_1", message("Handled together with your newer request (item_2).", phase="commentary"), completed("r1")
        )
        await asyncio.sleep(0.15)
        self.assertEqual(self.spoken, ["Saturday, then."])
        self.assertEqual(self.session.acknowledgements.pending, {})

    async def test_an_instruction_keeps_it(self):
        class Waits:
            async def decide(self, instructions, history, work_in_progress="", *, work_open=False):
                return TalkerDecision("wait")

        await self.slow_ack_call()
        self.session.agent.talker = Waits()
        await self.session.on_instruction("Keep it short.")
        await self.said(1)
        self.assertEqual(self.spoken, [LEAD_IN])

    async def test_commentary_or_thinking_drops_every_pending_one(self):
        for append in ("commentary", "thinking"):
            with self.subTest(append):
                port = ScriptedDelegationPort(acknowledgement_delay_ms=80)
                await self.call(
                    [TalkerDecision("delegate", LEAD_IN), TalkerDecision("delegate", "Cakes too.")], port=port
                )
                await self.turn("Teas?")
                self.buffer.turn("And cakes?")  # a turn whose start the binding missed: nothing dropped yet
                await self.session.on_turn()
                self.assertEqual(len(self.session.acknowledgements.pending), 2)
                if append == "commentary":
                    await self.session.on_commentary("The kitchen is busy.")
                else:
                    self.session.on_thinking("The caller is a regular.")
                self.assertEqual(self.session.acknowledgements.pending, {})
                await asyncio.sleep(0.15)
                self.assertFalse({LEAD_IN, "Cakes too."} & set(self.spoken))

    async def test_the_caller_starting_during_the_hand_off_drops_it_and_tells_the_records_nothing(self):
        port = ScriptedDelegationPort(hold=True)
        await self.call([TalkerDecision("delegate", LEAD_IN)], port=port)
        turn = asyncio.create_task(self.turn("Teas?"))
        await port.arrived.wait()
        self.session.caller_started()  # decided and sent: the delegation goes on, its lead-in would talk over them
        port.accept()
        await turn
        self.assertEqual((self.spoken, self.session.acknowledgements.pending), ([], {}))
        self.assertIn("acknowledgement.dropped", self.kinds())
        self.assertIn("item_1", self.session.worknote)  # its events will be read and its result said
        self.assertIsNone(self.session.utterances.told_text("item_1"))  # its result is worded as news
        self.assertNotIn(("assistant", LEAD_IN), self.memory())
        self.assertEqual(self.session.heard_log.entries, [])  # its turn went with the delegation; no line was said

    async def test_an_instructions_lead_in_waits_for_the_caller_after_its_delay(self):
        # The application's task decides while the caller speaks: the delegation is decided after their start.
        port = ScriptedDelegationPort(acknowledgement_delay_ms=20)
        await self.call([TalkerDecision("delegate", LEAD_IN)], port=port)
        self.session.caller_started()
        await self.session.on_instruction("Tell the caller which teas we have.")
        await asyncio.sleep(0.1)  # its delay passed
        self.assertEqual((self.spoken, list(self.session.acknowledgements.pending)), ([], ["item_1"]))
        self.session.caller_stopped()
        await self.said(1)
        self.assertEqual(self.spoken, [LEAD_IN])

    async def test_its_result_arriving_while_it_waits_for_the_caller_drops_it(self):
        await self.call([TalkerDecision("delegate", LEAD_IN), TalkerDecision("speak", "We have Earl Grey.")])
        self.session.caller_started()
        await self.session.on_instruction("Tell the caller which teas we have.")
        self.assertIn("item_1", self.session.acknowledgements.pending)  # due at once, but the caller speaks
        self.emit("item_1", *answered_events("r1", "Earl Grey."))
        self.assertEqual(self.session.acknowledgements.pending, {})
        self.session.caller_stopped()
        await self.said(1)
        await asyncio.sleep(0.05)
        self.assertEqual(self.spoken, ["We have Earl Grey."])

    async def test_a_delegation_with_no_lead_in_says_nothing(self):
        for delay_ms in (0, 80):
            with self.subTest(delay_ms=delay_ms):
                port = ScriptedDelegationPort(acknowledgement_delay_ms=delay_ms)
                await self.call([TalkerDecision("delegate", "")], port=port)
                await self.turn("Teas?")
                await asyncio.sleep(0.15)
                pending = self.session.acknowledgements.pending
                self.assertEqual((self.spoken, pending, len(self.port.requests)), ([], {}, 1))

    async def test_a_delay_update_applies_to_later_delegations(self):
        port = ScriptedDelegationPort(acknowledgement_delay_ms=0, reports_events=False)
        await self.call([TalkerDecision("delegate", LEAD_IN), TalkerDecision("delegate", "Cakes too.")], port=port)
        await self.turn("Teas?")
        self.assertEqual(self.spoken, [LEAD_IN])
        port.acknowledgement_delay_ms = 60_000
        await self.turn("And cakes?")
        self.assertEqual(self.spoken, [LEAD_IN])
        self.assertIn("item_2", self.session.acknowledgements.pending)
