# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D103, D107

"""The presenter's inbox: everything ready, worded together and said at idle, as one line."""

import asyncio
import unittest

from examples.frontend_backend_live.cascade.ports import ReadyResult, WordingNotes
from examples.frontend_backend_live.cascade.settings import CascadeSettings
from examples.frontend_backend_live.cascade.wording import RESULT_WORDINGS, AsWritten
from examples.frontend_backend_live.models.decision import TalkerDecision
from tests.unit.frontend_backend_live.cascade.doubles import (
    PROMPTS,
    Call,
    ScriptedFrontend,
    agent,
    answered_events,
    message,
    settle,
)


class RecordingWording:
    """A result wording that records each call and joins the speak parts with " + "."""

    def __init__(self, gate=None):
        self.calls: list[tuple[list[ReadyResult], WordingNotes]] = []
        self.gate = gate

    async def words(self, results, notes):
        self.calls.append((list(results), notes))
        if self.gate is not None:
            await self.gate.wait()
        return " + ".join(result.speak for result in results)


class SnapshotWording(RecordingWording):
    """A recording wording that also keeps the talker's memory as each call read it."""

    def __init__(self, memory):
        super().__init__()
        self.memory = memory
        self.read: list[list[str]] = []

    async def words(self, results, notes):
        self.read.append([message["content"] for message in self.memory.messages()])
        return await super().words(results, notes)


class CuttableWording(RecordingWording):
    """A gated recording wording that also records which of its calls a move cut short."""

    def __init__(self, gate):
        super().__init__(gate)
        self.cut: list[int] = []

    async def words(self, results, notes):
        index = len(self.calls)
        try:
            return await super().words(results, notes)
        except asyncio.CancelledError:
            self.cut.append(index)
            raise


class InboxCall(Call):
    async def inbox_call(self, script=(), wording=None, **kwargs):
        await self.call(script, **kwargs)
        self.wording = wording or RecordingWording()
        self.session.presenter.wording = self.wording

    def deliveries(self):
        return [[result.speak for result in results] for results, _ in self.wording.calls]


class TogetherTests(InboxCall):
    async def test_two_results_ready_at_idle_give_one_wording_call_one_line_and_two_memory_entries(self):
        await self.inbox_call()
        self.session.caller_started()  # the caller speaks: the results wait together
        self.session.presenter.ready(ReadyResult("D1", "m1", "Latte added."))
        self.session.presenter.ready(ReadyResult("D2", "m2", "Muffin added."))
        await asyncio.sleep(0.05)
        self.assertEqual(self.spoken, [])
        self.session.caller_stopped()
        await self.said(1)
        self.assertEqual(self.deliveries(), [["Latte added.", "Muffin added."]])
        self.assertEqual(self.spoken, ["Latte added. + Muffin added."])
        self.assertEqual(
            self.memory()[-3:],
            [
                ("user", PROMPTS.result_label + "Latte added."),
                ("user", PROMPTS.result_label + "Muffin added."),
                ("assistant", "Latte added. + Muffin added."),
            ],
        )
        self.assertFalse(self.session.presenter.busy)

    async def test_fixed_lines_are_said_word_for_word_in_their_place(self):
        await self.inbox_call()
        self.session.caller_started()
        self.session.presenter.notice("D1", "Sorry, that failed.")
        self.session.presenter.ready(ReadyResult("D2", "m2", "Tea: yes."))
        self.session.presenter.ready(ReadyResult("D3", "m3", "Cake: no."))
        self.session.caller_stopped()
        await self.said(1)
        self.assertEqual(self.spoken, ["Sorry, that failed. Tea: yes. + Cake: no."])
        self.assertEqual(self.deliveries(), [["Tea: yes.", "Cake: no."]])  # the fixed line is not worded

    async def test_the_applications_commentary_is_said_in_arrival_order(self):
        # Back-to-back updates are worded in one call, so they are said in arrival order.
        await self.inbox_call()
        self.session.caller_started()
        await self.session.on_commentary("The kitchen is busy.")
        await self.session.on_commentary("Your order is next.")
        self.session.caller_stopped()
        await self.said(1)
        self.assertEqual(self.spoken, ["The kitchen is busy. + Your order is next."])
        self.assertEqual(
            self.memory()[-3:-1],
            [
                ("user", PROMPTS.context_label + "The kitchen is busy."),
                ("user", PROMPTS.context_label + "Your order is next."),
            ],
        )


class WordAheadTests(InboxCall):
    """The writer words what is queued at once; at idle its line is said while its stamp holds, else worded again."""

    async def test_a_slow_writer_words_through_the_callers_turn_and_only_a_wait_keeps_its_line(self):
        cases = (
            ("Thanks.", TalkerDecision("wait"), [], 1),
            ("Do you have oat milk?", TalkerDecision("speak", "Yes, we have oat milk."), ["Yes, we have oat milk."], 2),
            ("And a muffin.", TalkerDecision("delegate"), [], 2),  # a new request: the newer requests change
        )
        for turn, decision, said, wordings in cases:
            with self.subTest(decision.action):
                gate = asyncio.Event()
                script = [TalkerDecision("delegate", "One moment."), decision]
                await self.inbox_call(script, wording=RecordingWording(gate))
                await self.turn("Add a latte.")
                self.buffer.turn(turn)
                self.session.caller_started()
                self.emit("item_1", *answered_events("r1", "Latte added."))
                await settle(lambda: self.wording.calls)  # the writer starts while the caller speaks
                self.session.caller_stopped()
                await self.session.on_turn()  # the talker decides while the writer still works
                self.assertEqual(self.spoken, ["One moment.", *said])
                gate.set()
                await self.said(2 + len(said))
                self.assertEqual(self.spoken, ["One moment.", *said, "Latte added."])
                self.assertEqual(self.deliveries(), [["Latte added."]] * wordings)
                newer = ("And a muffin.",) if decision.action == "delegate" else ()
                self.assertEqual(self.wording.calls[-1][1].newer_requests, newer)

    async def test_with_the_writer_model_a_wait_keeps_the_line_worded_before_the_callers_turn_was_heard(self):
        loop = asyncio.get_running_loop()
        writer = loop.create_future()
        await self.call([TalkerDecision("delegate", "One moment."), writer, TalkerDecision("wait")])
        await self.turn("Add a latte.")
        self.buffer.turn("Thanks.")
        self.session.caller_started()
        self.emit("item_1", *answered_events("r1", "Latte added."))
        await settle(lambda: len(self.frontend.calls) == 2)
        self.session.caller_stopped()
        await self.session.on_turn()
        writer.set_result(TalkerDecision("speak", "Your latte is added."))
        await self.said(2)
        self.assertEqual(self.spoken, ["One moment.", "Your latte is added."])
        roles = [call["role"] for call in self.frontend.calls]
        self.assertEqual(roles, ["talker", "writer", "talker"])  # one writer call: the wait kept its line
        history = [message["content"] for message in self.frontend.calls[1]["history"]]
        self.assertEqual(history[:-1], ["Add a latte.", "One moment."])  # the turn was not heard yet
        self.assertIn('the caller already heard: "One moment."', history[-1])

    async def test_a_reply_or_a_lead_in_that_moves_the_stamp_has_the_line_worded_again_while_it_plays(self):
        decisions = (TalkerDecision("speak", "Yes, we have oat milk."), TalkerDecision("delegate", "And a muffin."))
        for decision in decisions:
            with self.subTest(decision.action):
                await self.inbox_call([TalkerDecision("delegate", "One moment."), decision], reports_playout=True)
                self.wording = self.session.presenter.wording = SnapshotWording(self.session.memory)
                playout = self.session.playout
                await self.turn("Add a latte.")
                playout.started()
                playout.played("One moment.")
                playout.finished()
                self.buffer.turn("Do you have oat milk?")
                self.session.caller_started()
                self.emit("item_1", *answered_events("r1", "Latte added."))
                await settle(lambda: self.wording.read)  # worded while the caller speaks
                self.session.caller_stopped()
                await self.session.on_turn()
                await self.said(2)
                # Worded again while the reply or the lead-in plays, so at idle no writer call comes before the line.
                await settle(lambda: len(self.wording.read) == 2)
                self.assertIn(decision.speech, self.wording.read[-1])
                playout.started()
                playout.played(decision.speech)
                playout.finished()
                await self.said(3)
                self.assertEqual((self.spoken[-1], len(self.wording.read)), ("Latte added.", 2))

    async def test_an_item_that_comes_while_the_batch_is_worded_cuts_that_wording_short(self):
        gate = asyncio.Event()
        await self.inbox_call(wording=CuttableWording(gate))
        self.session.presenter.ready(ReadyResult("D1", "m1", "Latte added."))
        await settle(lambda: self.wording.calls)
        self.session.presenter.ready(ReadyResult("D2", "m2", "Muffin added."))
        await settle(lambda: len(self.wording.calls) == 2)  # worded again while the first call is still out
        self.assertEqual(self.wording.cut, [0])
        gate.set()
        await self.said(1)
        self.assertEqual(self.spoken, ["Latte added. + Muffin added."])

    async def test_a_reply_cuts_the_wording_under_way_short_once_its_turn_is_decided(self):
        gate = asyncio.Event()
        await self.inbox_call([TalkerDecision("speak", "Sure.")], wording=CuttableWording(gate))
        self.session.presenter.ready(ReadyResult("D1", "m1", "Latte added."))
        await settle(lambda: self.wording.calls)
        await self.turn("Thanks.")  # the reply moves the memory
        await settle(lambda: len(self.wording.calls) == 2)
        self.assertEqual(self.wording.cut, [0])
        gate.set()
        await self.said(2)
        self.assertEqual(self.spoken, ["Sure.", "Latte added."])

    async def test_a_caller_turn_the_talker_waits_on_keeps_the_wording_under_way(self):
        gate = asyncio.Event()
        await self.inbox_call([TalkerDecision("wait")], wording=CuttableWording(gate))
        self.session.presenter.ready(ReadyResult("D1", "m1", "Latte added."))
        await settle(lambda: self.wording.calls)
        await self.turn("Thanks.")
        await asyncio.sleep(0.05)
        self.assertEqual((len(self.wording.calls), self.wording.cut), (1, []))
        gate.set()
        await self.said(1)
        self.assertEqual(self.spoken, ["Latte added."])

    async def test_a_line_worded_before_an_instruction_or_the_applications_context_is_worded_again_with_it(self):
        rule = "Never mention prices."
        for append in ("instruction", "thinking"):
            with self.subTest(append):
                await self.inbox_call([TalkerDecision("delegate", "One moment."), TalkerDecision("wait")])
                self.wording = self.session.presenter.wording = SnapshotWording(self.session.memory)
                await self.turn("Add a latte.")
                self.session.caller_started()  # the caller speaks: the line waits
                self.emit("item_1", *answered_events("r1", "Latte added, 4 dollars."))
                await settle(lambda: self.wording.read)
                if append == "instruction":
                    await self.session.on_instruction(rule)  # the talker waits on it
                else:
                    self.session.on_thinking(rule)
                self.session.caller_stopped()
                await self.said(2)
                self.assertEqual(len(self.wording.read), 2)
                self.assertTrue(any(rule in content for content in self.wording.read[-1]))

    async def test_a_result_that_comes_while_the_line_waits_is_worded_with_it_into_one_line(self):
        await self.inbox_call()
        self.session.caller_started()
        self.session.presenter.ready(ReadyResult("D1", "m1", "Latte added."))
        await settle(lambda: self.wording.calls)
        self.session.presenter.ready(ReadyResult("D2", "m2", "Muffin added."))
        await asyncio.sleep(0.05)
        self.assertEqual(self.spoken, [])
        self.session.caller_stopped()
        await self.said(1)
        self.assertEqual(self.deliveries(), [["Latte added."], ["Latte added.", "Muffin added."]])
        self.assertEqual(self.spoken, ["Latte added. + Muffin added."])

    async def test_a_result_enters_the_memory_only_with_its_line_and_a_line_nobody_heard_is_worded_again_at_once(self):
        script = [TalkerDecision("delegate", "One moment."), TalkerDecision("wait"), TalkerDecision("wait")]
        await self.inbox_call(script, reports_playout=True)
        playout, result = self.session.playout, ("user", PROMPTS.result_label + "Latte added.")
        await self.turn("Add a latte.")
        playout.started()
        playout.played("One moment.")
        playout.finished()
        self.buffer.turn("Hmm.")
        self.session.caller_started()
        self.emit("item_1", *answered_events("r1", "Latte added."))
        await settle(lambda: self.wording.calls)
        self.assertNotIn(result, self.memory())  # worded, not said: the memory does not hold it
        self.session.caller_stopped()
        await self.session.on_turn()
        await self.said(2)
        self.assertIn(result, self.memory())
        # The caller barges in before a word of it played: nobody heard it, so it leaves the memory, goes back to the
        # inbox and is worded again at once, while the caller speaks.
        self.buffer.turn("Sorry?")
        self.session.caller_started()
        self.session.interrupted()
        playout.cut()
        self.assertNotIn(result, self.memory())
        await settle(lambda: len(self.wording.calls) == 2)
        self.assertTrue(self.session.worknote.answer_ready())
        self.session.caller_stopped()
        await self.session.on_turn()
        await self.said(3)
        playout.started()
        playout.played("Latte added.")
        playout.finished()
        self.assertEqual(self.spoken, ["One moment.", "Latte added.", "Latte added."])
        self.assertEqual((len(self.wording.calls), self.memory().count(result)), (2, 1))
        self.assertFalse(self.session.worknote.answer_ready())


class NotesTests(InboxCall):
    async def test_the_wording_knows_what_the_caller_was_told_and_the_newer_requests(self):
        await self.inbox_call([TalkerDecision("delegate", "Let me check."), TalkerDecision("delegate", "On it.")])
        await self.turn("Teas?")
        await self.turn("And oat milk?")
        self.emit("item_1", *answered_events("r1", "Earl Grey."))
        await self.said(3)
        results, notes = self.wording.calls[0]
        self.assertEqual(notes, WordingNotes({"item_1": "Let me check."}, ("And oat milk?",)))
        self.assertEqual(self.session.utterances.told, {"item_2": [(2, "On it.")]})  # item_1's was used once

    async def test_what_the_caller_was_told_goes_with_the_named_taker(self):
        await self.inbox_call(
            [TalkerDecision("delegate", "Friday, checking."), TalkerDecision("delegate", "Saturday.")]
        )
        await self.turn("Friday?")
        await self.turn("Actually Saturday.")
        self.emit("item_1", *answered_events("r1", start=True)[:3])
        self.emit("item_2", *answered_events("r2", start=True)[:2])
        self.emit("item_2", {"type": "response.in_progress", "response": {"id": "r2", "previous_response_id": "r1"}})
        self.emit("item_1", message("Handled together (item_2).", phase="commentary"), answered_events("r1")[-1])
        self.emit("item_2", message("Saturday is booked.", item_id="m2"), answered_events("r2")[-1])
        await self.said(3)
        _, notes = self.wording.calls[-1]
        self.assertEqual(notes.told, {"item_2": "Friday, checking. Saturday."})


class FailureTests(InboxCall):
    async def test_a_delivery_that_raises_is_requeued_once_then_said(self):
        failures = []

        async def say(text):
            if not failures:
                failures.append(text)
                raise RuntimeError("the speech service failed")
            self.spoken.append(text)

        await self.inbox_call(say=say)
        self.session.presenter.ready(ReadyResult("D1", "m1", "Latte added."))
        await self.said(1)
        self.assertEqual((failures, self.spoken), (["Latte added."], ["Latte added."]))
        self.assertIn("answer.failed", self.kinds())
        self.assertEqual([m for m in self.memory() if "Latte" in m[1]].count(("assistant", "Latte added.")), 1)

    async def test_a_delivery_that_raises_twice_is_dropped(self):
        async def say(text):
            raise RuntimeError("the speech service failed")

        await self.inbox_call(say=say)
        self.session.presenter.ready(ReadyResult("D1", "m1", "Latte added."))
        await settle(lambda: self.kinds().count("answer.failed") == 2)
        await asyncio.sleep(0.05)
        self.assertEqual(self.kinds().count("answer.failed"), 2)
        self.assertFalse(self.session.presenter.busy)


class WordingTests(unittest.IsolatedAsyncioTestCase):
    async def test_as_written_says_every_speak_part_cleaned_for_speech_in_order(self):
        results = [
            ReadyResult("D1", "m1", "Tea: yes. "),
            ReadyResult("D2", "m2", ""),
            ReadyResult("D2", "m3", "**No**"),
            ReadyResult("D3", "m4", "## Teas\n- Earl Grey\n1. Mint"),
        ]
        words = await AsWritten().words(results, WordingNotes({}, ()))
        self.assertEqual(words, "Tea: yes. No. Teas. Earl Grey. Mint.")

    async def test_the_writer_words_every_result_in_one_call_reading_the_talkers_memory(self):
        # WriterWording is the writer entry (models/test_wording.py pins its request and its fallback): one call
        # for every ready result, reading the talker's memory when it words, under the one attempt's deadline.
        frontend = ScriptedFrontend([TalkerDecision("speak", "Earl Grey, and the total is five dollars.")])
        from examples.frontend_backend_live.cascade.memory import TalkerMemory
        from examples.frontend_backend_live.cascade.writer.wording import WriterWording

        memory = TalkerMemory.seeded(
            PROMPTS, [{"role": "system", "content": "Be Remy."}], context_tokens=16000, compact_at=0.8
        )
        memory.caller("Tea and the total?")
        wording = RESULT_WORDINGS["writer"](agent(frontend), memory, lambda kind, **fields: None)
        self.assertIsInstance(wording, WriterWording)
        self.assertEqual(wording.timeout_seconds, frontend.reliability.talker_attempt_timeout_seconds)
        words = await wording.words(
            [ReadyResult("D1", "m1", "Earl Grey."), ReadyResult("D2", "m2", "**Total:** $5")],
            WordingNotes({"D1": "Let me check."}, ("Oat milk?",)),
        )
        self.assertEqual(words, "Earl Grey, and the total is five dollars.")
        self.assertEqual(len(frontend.calls), 1)
        (call,) = frontend.calls
        self.assertIn("Tea and the total?", [message["content"] for message in call["history"]])
        self.assertIn("Earl Grey.\n\n**Total:** $5", call["history"][-1]["content"])


class AsWrittenCall(Call):
    async def test_a_session_on_as_written_says_the_results_unchanged(self):
        await self.call(
            [TalkerDecision("delegate", "One moment.")],
            settings=CascadeSettings.parse(
                {"delegation": {"result_wording": "as_written", "progress_after_seconds": None}}
            ),
        )
        await self.turn("Teas?")
        self.emit("item_1", *answered_events("r1", "Earl Grey and mint.", "Both are in stock."))
        await self.said(2)
        self.assertEqual(self.spoken, ["One moment.", "Earl Grey and mint. Both are in stock."])
        self.assertEqual(len(self.frontend.calls), 1)  # no writer call
