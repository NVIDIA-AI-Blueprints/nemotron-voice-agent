# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D103

"""Caller turns: heard at the stop, decided one at a time, carried out."""

import asyncio

from examples.frontend_backend_live.cascade.heard_log import Heard
from examples.frontend_backend_live.cascade.settings import CascadeSettings
from examples.frontend_backend_live.models.decision import TalkerDecision
from tests.unit.contracts.delegation_port import ScriptedDelegationPort
from tests.unit.frontend_backend_live.cascade.doubles import (
    NO_WORK_IN_PROGRESS,
    PROMPTS,
    WORK,
    Call,
    answered_events,
    conversation,
    runtime_data,
    settle,
    work_item,
)


def speak(text):
    return TalkerDecision("speak", text)


def delegate(lead_in=""):
    return TalkerDecision("delegate", lead_in)


# Results said as written, so the scripts hold the talker's decisions only.
AS_WRITTEN = CascadeSettings.parse({"delegation": {"result_wording": "as_written", "progress_after_seconds": None}})


class RoutingTests(Call):
    async def test_a_spoken_reply_is_said_and_nothing_is_delegated(self):
        await self.call([speak("We open at seven.")])
        await self.turn("When do you open?")
        self.assertEqual(self.spoken, ["We open at seven."])
        self.assertEqual(self.port.requests, [])
        self.assertEqual(conversation(self.frontend.calls[0]), [{"role": "user", "content": "When do you open?"}])
        self.assertIn("You are Remy.", self.frontend.calls[0]["instructions"])
        self.assertIn(NO_WORK_IN_PROGRESS, runtime_data(self.frontend.calls[0]))  # the port reports events
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
        self.assertIn(
            '(the caller already heard: "Let me look that up."; do not say that again)',
            wording["history"][-1]["content"],
        )
        self.assertEqual(
            self.memory()[-2:],
            [("user", PROMPTS.result_label + "Latte: $4.50."), ("assistant", "A latte is four fifty.")],
        )

    async def test_a_lead_in_the_talker_wrote_is_said_while_work_is_open(self):
        await self.call([delegate("Adding a latte."), delegate("And the muffin, too.")])
        await self.turn("Add a latte.")
        await self.turn("And a muffin.")
        self.assertEqual(self.spoken, ["Adding a latte.", "And the muffin, too."])

    async def test_a_delayed_lead_in_the_caller_speaks_over_is_dropped_and_the_next_one_is_said(self):
        # A client-chosen delay: the caller speaks before the first lead-in plays, so it never does; the next one does.
        port = ScriptedDelegationPort(acknowledgement_delay_ms=50)
        await self.call([delegate("Checking Friday."), delegate("For two, then.")], port=port)
        await self.turn("Book a table for Friday.")
        self.assertEqual(list(self.session.acknowledgements.pending), ["item_1"])
        await self.turn("For two people.")
        self.assertEqual(list(self.session.acknowledgements.pending), ["item_2"])
        await settle(lambda: not self.session.acknowledgements.pending)
        self.assertEqual(self.spoken, ["For two, then."])

    async def test_a_delegation_whose_turn_was_cut_off_during_its_hand_off_acknowledges_nothing(self):
        port = ScriptedDelegationPort(hold=True)
        await self.call([delegate("Checking Friday."), delegate("For two, then.")], port=port)
        first = asyncio.create_task(self.turn("Book a table for Friday."))
        await port.arrived.wait()
        first.cancel()  # the caller speaks again while the hand-off runs: the turn says nothing
        await asyncio.gather(first, return_exceptions=True)
        port.hold = False
        port.accept()
        await settle(lambda: "item_1" in self.session.worknote)
        await self.turn("For two people.")
        self.assertEqual((self.spoken, len(port.requests)), (["For two, then."], 2))

    async def test_a_lead_in_nobody_heard_tells_the_caller_nothing(self):
        await self.call([delegate("On it.")], reports_playout=True)
        await self.turn("Book a table for Friday.")
        self.session.interrupted()
        self.session.playout.cut()  # cut before any word played: nobody heard "On it."
        self.assertEqual(self.spoken, ["On it."])
        self.assertIsNone(self.session.utterances.told_text("item_1"))

    async def answer_ready_while_the_caller_asks(self, script):
        """The caller asks for a latte, thanks the voice, then asks for a muffin; the latte's answer comes meanwhile."""
        await self.call([delegate("Adding a latte."), TalkerDecision("wait"), *script], settings=AS_WRITTEN)
        await self.turn("Add a latte.")
        await self.turn("Thanks.")
        self.buffer.turn("And a muffin.")
        self.session.caller_started()
        self.emit("item_1", *answered_events("r1", "Latte added."))  # ready while the caller speaks
        self.session.caller_stopped()
        await self.session.on_turn()

    async def test_while_an_answer_waits_to_be_heard_a_lead_in_the_talker_wrote_is_said(self):
        await self.answer_ready_while_the_caller_asks([delegate("And the muffin, too.")])
        await self.said(3)
        self.assertEqual(self.spoken, ["Adding a latte.", "And the muffin, too.", "Latte added."])

    async def test_in_client_mode_a_lead_in_is_said_while_commentary_waits(self):
        port = ScriptedDelegationPort(reports_events=False)
        await self.call([delegate("Checking the weather."), delegate("Tomorrow, too.")], port=port, settings=AS_WRITTEN)
        await self.turn("What is the weather?")
        self.buffer.turn("And tomorrow?")
        self.session.caller_started()
        await self.session.on_commentary("Sunny today.", "item_1")  # the app's answer waits while the caller speaks
        self.session.caller_stopped()
        await self.session.on_turn()
        await self.said(3)
        self.assertEqual(self.spoken, ["Checking the weather.", "Tomorrow, too.", "Sunny today."])

    async def test_a_failed_talker_call_with_no_open_work_is_recovered_as_a_delegation(self):
        await self.call([RuntimeError("hub down")])
        await self.turn("What teas do you have?")
        self.assertEqual(len(self.port.requests), 1)
        self.assertIn("talker.recovered", self.kinds())

    async def test_the_talker_reads_the_work_note_while_work_is_open(self):
        await self.call([delegate("One moment."), speak("Still on it.")])
        await self.turn("Book a table.")
        await self.turn("How is it going?")
        line = runtime_data(self.frontend.calls[1])
        self.assertIn(work_item(1, WORK.waiting, "Book a table."), line)
        self.assertNotIn(NO_WORK_IN_PROGRESS, line)

    async def test_a_client_delegation_is_never_open_work_and_the_talker_reads_no_work_line(self):
        await self.call(
            [delegate("One moment."), speak("Let me look.")], port=ScriptedDelegationPort(reports_events=False)
        )
        await self.turn("What is the weather?")
        await self.turn("Any update?")
        self.assertNotIn("Work in progress", runtime_data(self.frontend.calls[1]))
        talker = self.session.agent.talker
        self.assertEqual(runtime_data(self.frontend.calls[1]), "\n" + talker.runtime_data(turn="Any update?"))
        self.assertEqual(self.session.worknote.open(), [])

    async def test_after_a_wait_the_next_decision_names_only_the_newer_turn(self):
        # The turn waited on and the newer one reach the talker as one message: the runtime data names the newer one.
        await self.call([TalkerDecision("wait"), delegate("Let me check.")])
        await self.turn("Can you wait?")
        await self.turn("My zip is one nine one two two.")
        newest = PROMPTS.newest_turn.format(turn="My zip is one nine one two two.")
        self.assertIn(newest, runtime_data(self.frontend.calls[1]))
        self.assertNotIn("Can you wait?", runtime_data(self.frontend.calls[1]))

    async def test_wait_says_and_delegates_nothing(self):
        # The prompt set declares `wait`, so the talker's floor keeps it.
        await self.call([TalkerDecision("wait")])
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
        self.assertEqual(
            conversation(call), [{"role": "user", "content": "Earlier."}, {"role": "user", "content": "Hi."}]
        )

    async def test_a_turn_cut_off_while_the_talker_decides_is_decided_with_the_next_turn(self):
        slow = asyncio.get_running_loop().create_future()
        await self.call([slow, delegate("One moment.")])
        self.buffer.turn("My user ID is Q nine one")
        self.session.caller_stopped()
        deciding = asyncio.create_task(self.session.on_turn())
        await settle(lambda: self.frontend.calls)
        deciding.cancel()  # the caller speaks again: Pipecat cancels the decision
        with self.assertRaises(asyncio.CancelledError):
            await deciding
        await self.turn("underscore T")
        pieces = [
            {"role": "user", "content": "My user ID is Q nine one"},
            {"role": "user", "content": "underscore T"},
        ]
        self.assertEqual(conversation(self.frontend.calls[1]), pieces)
        transcript = self.port.requests[0][0]["content"][0]["text"]
        caller = PROMPTS.transcript.caller
        self.assertIn(f"{caller}: My user ID is Q nine one\n{caller}: underscore T", transcript)
        self.assertEqual(self.spoken, ["One moment."])

    async def test_a_result_ready_during_a_caller_turn_is_said_after_the_reply(self):
        # The writer words the result at once, while the caller speaks; the reply moves the stamp, so it words it again.
        script = [delegate("One moment."), speak("A latte, added."), speak("Yes, we have oat milk.")]
        await self.call([*script, speak("Your latte is added.")])
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
        self.assertEqual([call["role"] for call in self.frontend.calls], ["talker", "writer", "talker", "writer"])


class SpokenAgainTests(Call):
    """The caller goes on after a turn ends and before its decision starts: no interruption cuts that decision off."""

    async def spoken_again(self, first, *script):
        """The caller's ``first`` turn ends and the next begins; return the held decision's future and its task."""
        slow = asyncio.get_running_loop().create_future()
        await self.call([slow, *script])
        self.buffer.turn(first)
        self.session.caller_started()
        self.session.caller_stopped()
        self.session.caller_started()  # the caller goes on: its interruption came before the decision
        deciding = asyncio.create_task(self.session.on_turn())
        await settle(lambda: self.frontend.calls)
        return slow, deciding

    async def test_a_decision_the_caller_speaks_over_is_discarded_and_the_next_covers_both_turns(self):
        slow, deciding = await self.spoken_again("My user ID is J, A, M, E, S", delegate("Let me look up that ID."))
        slow.set_result(delegate("Let me check."))
        await deciding
        self.assertEqual((self.spoken, self.port.requests), ([], []))  # no lead-in over the caller, nothing sent
        self.assertIn("talker.discarded", self.kinds())
        self.assertEqual(self.session.turns.waiting, ["My user ID is J, A, M, E, S"])
        self.buffer.turn("underscore, L, E, E")
        self.session.caller_stopped()
        await self.session.on_turn()
        pieces = [
            {"role": "user", "content": "My user ID is J, A, M, E, S"},
            {"role": "user", "content": "underscore, L, E, E"},
        ]
        self.assertEqual(conversation(self.frontend.calls[1]), pieces)
        transcript = self.port.requests[0][0]["content"][0]["text"]
        caller = PROMPTS.transcript.caller
        self.assertIn(f"{caller}: My user ID is J, A, M, E, S\n{caller}: underscore, L, E, E", transcript)
        self.assertEqual((self.spoken, len(self.port.requests)), (["Let me look up that ID."], 1))

    async def test_a_discarded_decision_leaves_every_record_as_the_caller_left_it(self):
        slow, deciding = await self.spoken_again("Do you have oat milk?")
        slow.set_result(speak("Yes, we have oat milk."))
        await deciding
        self.assertEqual(self.memory()[1:], [("user", "Do you have oat milk?")])
        heard = [(entry.kind, entry.text) for entry in self.session.heard_log.entries]
        self.assertEqual(heard, [(Heard.CALLER, "Do you have oat milk?")])
        self.assertEqual((self.session.worknote.open(), self.display.calls), ([], []))
        self.assertEqual(self.session.utterances.told, {})
        self.assertTrue(self.session.floor.turns_busy)  # its turn still waits: nothing said on its own goes first

    async def test_after_a_discarded_decision_a_turn_with_no_words_decides_the_waiting_turn(self):
        # A turn with no words that cut off the voice's work runs the processor again (Pipecat's empty-turn recovery),
        # beside the decision the caller's silence starts: the waiting turn is decided once.
        slow, deciding = await self.spoken_again("Do you have oat milk?", speak("Yes, we have oat milk."))
        slow.set_result(speak("Yes, we have oat milk."))
        await deciding
        self.assertEqual(self.spoken, [])
        self.session.caller_stopped()  # no words
        await self.session.on_turn()
        await settle(lambda: not self.session.turns.redecisions)
        self.assertEqual((self.spoken, self.session.turns.waiting), (["Yes, we have oat milk."], []))
        self.assertEqual((self.kinds().count("talker.discarded"), len(self.frontend.calls)), (1, 2))

    async def test_after_a_discarded_decision_the_callers_silence_decides_the_waiting_turn(self):
        # Speech with no words that cut nothing off brings no turn: no context frame comes, and no new words.
        slow, deciding = await self.spoken_again("Do you have oat milk?", speak("Yes, we have oat milk."))
        slow.set_result(speak("Sure."))
        await deciding
        self.session.caller_stopped()
        await self.said(1)
        self.assertEqual((self.spoken, self.session.turns.waiting), (["Yes, we have oat milk."], []))
        self.assertEqual(conversation(self.frontend.calls[1]), [{"role": "user", "content": "Do you have oat milk?"}])
        remembered = [("user", "Do you have oat milk?"), ("assistant", "Yes, we have oat milk.")]
        self.assertEqual(self.memory()[1:], remembered)
        self.assertFalse(self.session.floor.turns_busy)

    async def test_the_callers_silence_while_a_decision_covers_the_waiting_turn_adds_no_decision(self):
        held = asyncio.get_running_loop().create_future()
        slow, deciding = await self.spoken_again("Book a table.", held)
        slow.set_result(speak("Sure."))
        await deciding
        instructing = asyncio.create_task(self.session.on_instruction("Speak only Spanish."))
        await settle(lambda: len(self.frontend.calls) == 2)  # the instruction's decision covers the waiting turn
        self.session.caller_stopped()  # silent while it runs
        await asyncio.sleep(0.05)
        self.assertEqual((len(self.frontend.calls), self.spoken), (2, []))
        held.set_result(speak("Claro, una mesa."))
        await instructing
        await settle(lambda: not self.session.turns.redecisions)
        self.assertEqual((self.spoken, len(self.frontend.calls)), (["Claro, una mesa."], 2))
        self.assertEqual(self.session.turns.waiting, [])

    async def test_a_turn_heard_while_the_callers_silence_decides_drops_that_decision_and_the_next_covers_both(self):
        held = asyncio.get_running_loop().create_future()
        slow, deciding = await self.spoken_again("Book a table.", held, delegate("A table for two, then."))
        slow.set_result(speak("Sure."))
        await deciding
        self.session.caller_stopped()  # no words: the waiting turn is decided
        await settle(lambda: len(self.frontend.calls) == 2)
        self.buffer.turn("For two.")  # heard while that decision runs, which no interruption cuts off
        self.session.caller_started()
        self.session.caller_stopped()
        turn = asyncio.create_task(self.session.on_turn())
        held.set_result(speak("Sure, a table."))
        await turn
        await settle(lambda: not self.session.turns.redecisions)
        self.assertEqual((self.spoken, len(self.port.requests)), (["A table for two, then."], 1))
        self.assertEqual(self.display.calls[0], ("started", "item_1", "Book a table. For two."))
        self.assertEqual(self.kinds().count("talker.discarded"), 2)
        pieces = [{"role": "user", "content": "Book a table."}, {"role": "user", "content": "For two."}]
        self.assertEqual(conversation(self.frontend.calls[2]), pieces)
        self.assertEqual((len(self.frontend.calls), self.session.turns.waiting), (3, []))

    async def test_an_instructions_decision_covering_a_turn_is_discarded_when_the_caller_speaks_again(self):
        slow = asyncio.get_running_loop().create_future()
        await self.call([slow, speak("Claro, una mesa para dos.")])
        self.buffer.turn("Book a table.")
        self.session.caller_started()
        self.session.caller_stopped()  # heard; its context frame has not come yet
        instructing = asyncio.create_task(self.session.on_instruction("Speak only Spanish."))
        await settle(lambda: self.frontend.calls)
        self.buffer.turn("For two.")  # the application's task decides: no interruption cuts it off
        self.session.caller_started()
        self.session.caller_stopped()
        slow.set_result(speak("Hecho."))
        await instructing
        await self.said(1)  # the caller is silent: both turns are decided again with no new turn
        await self.session.on_turn()  # the context frame of "For two." finds them decided
        self.assertEqual(self.spoken, ["Claro, una mesa para dos."])
        self.assertIn("Speak only Spanish.", self.frontend.calls[1]["instructions"])
        pieces = [{"role": "user", "content": "Book a table."}, {"role": "user", "content": "For two."}]
        self.assertEqual([m for m in conversation(self.frontend.calls[1]) if m["role"] == "user"], pieces)

    async def test_an_instruction_decided_while_the_caller_speaks_is_replied_to_once_they_stop(self):
        await self.call([speak("Understood.")])
        self.session.caller_started()
        instructing = asyncio.create_task(self.session.on_instruction("Keep it short."))
        await settle(lambda: self.frontend.calls)
        await asyncio.sleep(0.05)
        self.assertEqual(self.spoken, [])  # decided, and carried out: the reply waits for the floor
        self.session.caller_stopped()
        await instructing
        self.assertEqual(self.spoken, ["Understood."])
        self.assertNotIn("talker.discarded", self.kinds())


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
            async def normalize(self, history, text):
                return await rewritten, {}

        await self.call([delegate("One moment.")])
        self.session.agent.normalizer = Normalizer()
        self.buffer.turn("my id is j dot wren")
        self.session.caller_stopped()
        deciding = asyncio.create_task(self.session.on_turn())
        await asyncio.sleep(0.01)
        self.buffer.turn("at example dot org")  # heard while the normalizer runs (nothing interrupted it)
        self.session.caller_stopped()
        rewritten.set_result("my id is j.wren")
        await deciding
        callers = [m for m in conversation(self.frontend.calls[0]) if m["role"] == "user"]
        self.assertEqual(
            callers,
            [{"role": "user", "content": "my id is j.wren"}, {"role": "user", "content": "at example dot org"}],
        )
        # The request the work note and the display read is both turns, the rewritten one in its place.
        self.assertEqual(self.display.calls[0], ("started", "item_1", "my id is j.wren at example dot org"))
        self.assertEqual(self.session.turns.waiting, [])
