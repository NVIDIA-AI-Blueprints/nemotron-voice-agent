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
                ("user", "Backend result (reference data): Latte added."),
                ("user", "Backend result (reference data): Muffin added."),
                ("assistant", "Latte added. + Muffin added."),
            ],
        )
        self.assertFalse(self.session.presenter.busy)

    async def test_what_arrives_while_the_batch_is_worded_is_worded_again_with_it(self):
        gate = asyncio.Event()
        await self.inbox_call(wording=RecordingWording(gate))
        self.session.presenter.ready(ReadyResult("D1", "m1", "Latte added."))
        await settle(lambda: self.wording.calls)
        self.session.presenter.ready(ReadyResult("D2", "m2", "Muffin added."))
        gate.set()
        await self.said(1)
        self.assertEqual(self.deliveries(), [["Latte added."], ["Latte added.", "Muffin added."]])
        self.assertEqual(self.spoken, ["Latte added. + Muffin added."])

    async def test_a_caller_turn_while_the_batch_is_worded_makes_it_worded_again(self):
        gate = asyncio.Event()
        await self.inbox_call([TalkerDecision("speak", "Sure.")], wording=RecordingWording(gate))
        self.session.presenter.ready(ReadyResult("D1", "m1", "Latte added."))
        await settle(lambda: self.wording.calls)
        await self.turn("Thanks.")  # the memory moved: the reply and the turn
        gate.set()
        await self.said(2)
        self.assertEqual(self.spoken, ["Sure.", "Latte added."])
        self.assertEqual(len(self.wording.calls), 2)

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
                ("user", "Application context (reference data): The kitchen is busy."),
                ("user", "Application context (reference data): Your order is next."),
            ],
        )


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
        self.emit("item_1", message("Handled together.", phase="commentary"), answered_events("r1")[-1])
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
    async def test_as_written_says_every_speak_part_unchanged_in_order(self):
        results = [
            ReadyResult("D1", "m1", "Tea: yes. "),
            ReadyResult("D2", "m2", ""),
            ReadyResult("D2", "m3", "**No**"),
        ]
        self.assertEqual(await AsWritten().words(results, WordingNotes({}, ())), "Tea: yes. **No**")

    async def test_the_writer_words_every_result_in_one_call_reading_the_talkers_memory(self):
        # WriterWording is the writer entry (models/test_wording.py pins its request and its fallback): one call
        # for every ready result, reading the talker's memory when it words, under the one attempt's deadline.
        frontend = ScriptedFrontend([TalkerDecision("speak", "Earl Grey, and the total is five dollars.")])
        from examples.frontend_backend_live.cascade.memory import TalkerMemory
        from examples.frontend_backend_live.cascade.writer.wording import WriterWording

        memory = TalkerMemory.seeded([{"role": "system", "content": "Be Remy."}], context_tokens=16000, compact_at=0.8)
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

    def test_the_wording_is_chosen_by_name(self):
        self.assertEqual(
            CascadeSettings.parse({"delegation": {"result_wording": "as_written"}}).result_wording, "as_written"
        )
        with self.assertRaisesRegex(ValueError, "Unknown result wording 'poetic'; known: as_written, writer"):
            CascadeSettings.parse({"delegation": {"result_wording": "poetic"}})


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
