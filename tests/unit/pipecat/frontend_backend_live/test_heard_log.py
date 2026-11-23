# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D103, D107

"""The heard log carries the backend's input: one user message, a labeled transcript."""

import asyncio
import json
import unittest

from nemotron_voice_agent.pipecat.examples.frontend_backend_live.heard_log import MAX_TRANSCRIPT_CHARS, Heard, HeardLog
from nemotron_voice_agent.pipecat.examples.frontend_backend_live.prompts import TranscriptText, load_prompts
from nemotron_voice_agent.pipecat.examples.frontend_backend_live.spoken import Line, LineKind, Outcome
from nemotron_voice_agent.pipecat.shared.delegation_contract import CALLER_CONTEXT, CALLER_EXCERPT

PROMPTS = load_prompts().cascade
WORDS = PROMPTS.transcript
TURNS = [f"Turn {n}: " + "words " * 10 for n in range(200)]


class Side:
    """A delegation side: it accepts each request at once or, ``held``, as the test decides."""

    def __init__(self, held=False):
        self.requests: list[list[dict]] = []
        self.arrived = asyncio.Event()
        self.decided = asyncio.get_running_loop().create_future() if held else None

    async def delegate(self, input: list[dict]) -> str:
        self.requests.append(input)
        self.arrived.set()
        return await self.decided if self.decided else f"item_{len(self.requests)}"


def text_of(items):
    return items[0]["content"][0]["text"]


def line(label, text):
    return f"{label}: {text}"


def carried(log):
    return [text for _, text in log.lines()]


def playback_only(labeled):
    label, _, text = labeled.partition(": ")
    receipt = json.loads(text)
    del receipt["generated_text"]
    return line(label, json.dumps(receipt, ensure_ascii=False))


def present(log, speech, outcome=Outcome.HANDED, *, prefix=None):
    speech.outcome = outcome
    speech.audio_started = outcome in (Outcome.PLAYING, Outcome.HEARD, Outcome.CUT)
    speech.heard = speech.text if outcome is Outcome.HEARD and prefix is None else prefix or ""
    log.presented(
        speech,
        {
            "complete_conveyance": outcome is Outcome.HEARD and speech.text == speech.heard,
            "context_work_refs": speech.work_refs,
            "results": speech.publication_items,
        },
    )
    return speech


class TranscriptTests(unittest.TestCase):
    def test_cut_lines_are_marked_and_unheard_ones_are_gone(self):
        log = HeardLog(PROMPTS)
        ready = present(log, Line(LineKind.REPLY, "Your latte is ready at the front counter.", number=1))
        present(log, Line(LineKind.REPLY, "Anything else?", number=2))
        present(log, Line(LineKind.RESULT, "Your total is five dollars.", results=True, number=3), Outcome.HEARD)
        order = Line(LineKind.RESULT, "Your order number is BB-42 and it will be ready soon.", results=True, number=4)
        present(log, ready, Outcome.CUT, prefix="Your latte is ready")
        log.unsay(2)
        present(log, order, Outcome.CUT, prefix="Your order number is")
        entries = [json.loads(text.partition(": ")[2]) for text in carried(log)]
        self.assertEqual(
            [entry["recorded_sentence_prefix"] for entry in entries],
            ["Your total is five dollars.", "Your latte is ready", "Your order number is"],
        )
        self.assertEqual([entry["playback_state"] for entry in entries], ["complete", "interrupted", "interrupted"])
        self.assertEqual(entries[1]["generated_text"], "Your latte is ready at the front counter.")

    def test_one_message_of_labeled_lines_in_the_prompts_words(self):
        words = TranscriptText(
            header="HEADER",
            omitted="OMITTED",
            caller="CALLER",
            agent="AGENT",
            context="CONTEXT",
            instruction="NOTE",
            earlier="EARLIER",
            unheard="UNHEARD",
        )
        log = HeardLog(PROMPTS.model_copy(update={"transcript": words, "cut_mark": " <CUT>"}))
        greeting = present(log, Line(LineKind.REPLY, "Welcome.", number=1), Outcome.HEARD)
        log.caller("Teas?")
        log.context("Regular.")
        log.instruction("Be brief.")
        items, _ = log.delegation_input()
        self.assertEqual([item["role"] for item in items], ["user"])
        welcome = log.lines()[0][1]
        self.assertEqual(json.loads(welcome.partition(": ")[2])["recorded_sentence_prefix"], "Welcome.")
        expected = playback_only(welcome)
        self.assertEqual(text_of(items), f"HEADER\n{expected}\nCALLER: Teas?\nCONTEXT: Regular.\nNOTE: Be brief.")
        present(log, greeting, Outcome.CUT, prefix="Wel")
        kinds = [Heard.CALLER, Heard.CONTEXT, Heard.INSTRUCTION, Heard.AGENT]
        self.assertEqual([kind for kind, _ in log.lines()], kinds)
        agent = json.loads(log.lines()[-1][1].partition(": ")[2])
        self.assertEqual((agent["generated_text"], agent["recorded_sentence_prefix"]), ("Welcome.", "Wel"))
        assert [text for _, text in log.lines()[:-1]] == ["CALLER: Teas?", "CONTEXT: Regular.", "NOTE: Be brief."]
        self.assertEqual(text_of(log.delegation_input()[0]).splitlines()[0], "HEADER")
        self.assertEqual(text_of(log.delegation_input(unheard=True)[0]).splitlines()[-1], "UNHEARD")
        for _ in range(200):
            log.caller("words " * 10)
        self.assertEqual(text_of(log.delegation_input()[0]).splitlines()[:2], ["HEADER", "OMITTED"])
        log.sent(log.delegation_input()[1])
        log.caller("Hours?")
        earlier = ["EARLIER", "NOTE: Be brief.", "HEADER", "CALLER: Hours?"]
        self.assertEqual(text_of(log.delegation_input()[0]).splitlines(), earlier)

    def test_the_unheard_line_and_the_cap_on_old_lines(self):
        log = HeardLog(PROMPTS)
        present(log, Line(LineKind.REPLY, "Let me check.", number=1), Outcome.HEARD)
        log.caller("Okay.")
        heard = [WORDS.header, playback_only(log.lines()[0][1]), line(WORDS.caller, "Okay.")]
        self.assertEqual(text_of(log.delegation_input(unheard=True)[0]), "\n".join([*heard, WORDS.unheard]))
        self.assertEqual(text_of(log.delegation_input()[0]), "\n".join(heard))
        for turn in TURNS:
            log.caller(turn)
        self.assertEqual(text_of(log.delegation_input(unheard=True)[0]).splitlines()[-1], WORDS.unheard)
        text = text_of(log.delegation_input()[0])
        self.assertLessEqual(len(text), MAX_TRANSCRIPT_CHARS + len(WORDS.header) + len(WORDS.omitted) + 100)
        self.assertEqual(text.splitlines()[:3], [*heard[:2], WORDS.omitted])
        self.assertTrue(text.endswith(line(WORDS.caller, TURNS[-1])))
        long_wording = "Complete spoken term. " * (MAX_TRANSCRIPT_CHARS // 10)
        present(log, Line(LineKind.RESULT, long_wording, results=True, number=2), Outcome.HEARD)
        log.caller("Yes.")
        sent, _ = log.delegation_input()
        preserved = text_of(sent).splitlines()
        self.assertTrue(sent[0][CALLER_EXCERPT].endswith("Yes."))
        self.assertNotIn("Complete spoken term", sent[0][CALLER_EXCERPT])
        self.assertEqual(preserved[:2], [WORDS.header, WORDS.omitted])
        self.assertEqual(preserved[-1], line(WORDS.caller, "Yes."))
        receipt = json.loads(preserved[-2].partition(": ")[2])
        self.assertEqual(receipt["recorded_sentence_prefix"], long_wording)
        self.assertTrue(receipt["complete_conveyance"])
        disposition = log.not_conveyed(
            [{"delegation_id": "D1", "item_id": "old", "branch_ref": "B1", "kind": "result"}],
            "writer_omitted",
        )
        capped = text_of(log.delegation_input()[0])
        self.assertIn(disposition, capped, "the latest correlated omission survives the soft transcript cap")
        present(log, Line(LineKind.RESULT, "Should I proceed?", results=True, number=3), Outcome.NOT_HEARD)
        self.assertEqual(text_of(log.delegation_input()[0]), capped)

    def test_the_latest_result_playback_survives_a_long_note_and_newer_acknowledgment(self):
        proposal = "The refund is forty dollars. It goes back to your card in five days. Proceed?"
        prefix = "The refund is forty dollars."
        for interrupted in (False, True):
            with self.subTest(interrupted=interrupted):
                log = HeardLog(PROMPTS)
                proposal_line = Line(
                    LineKind.RESULT,
                    proposal,
                    results=True,
                    number=1,
                    delegation_ids=("D1",),
                    work_refs=frozenset(("refund",)),
                    publication_items=({"delegation_id": "D1", "item_id": "proposal", "task_ref": "W1.1"},),
                )
                present(
                    log,
                    proposal_line,
                    Outcome.CUT if interrupted else Outcome.HEARD,
                    prefix=prefix if interrupted else proposal,
                )
                log.context("X" * MAX_TRANSCRIPT_CHARS)
                present(log, Line(LineKind.REPLY, "Take your time.", number=2), Outcome.HEARD)
                log.caller("Yes.")

                payload, _ = log.delegation_input()
                context = payload[0][CALLER_CONTEXT]
                self.assertEqual([turn for _, turn in context["caller_turns"]], ["Yes."])
                actual, _ = context["playback"]
                self.assertEqual(actual["results"][0]["task_ref"], "W1.1")
                self.assertEqual(actual["recorded_sentence_prefix"], prefix if interrupted else proposal)
                text = text_of(payload)
                header, result, omitted, acknowledgment, caller = text.splitlines()
                self.assertEqual((header, omitted, caller), (WORDS.header, WORDS.omitted, line(WORDS.caller, "Yes.")))
                result = json.loads(result.partition(": ")[2])
                self.assertEqual(result["line_id"], 1)
                self.assertEqual(result["work_refs"], ["refund"])
                self.assertEqual(result["results"], [{"delegation_id": "D1", "item_id": "proposal"}])
                self.assertEqual(result["recorded_sentence_prefix"], prefix if interrupted else proposal)
                self.assertEqual(result["complete_conveyance"], not interrupted)
                self.assertNotIn("generated_text", result)
                if interrupted:
                    self.assertNotIn("It goes back to your card in five days.", text)
                self.assertTrue(any(proposal in row for row in carried(log)))
                self.assertEqual(json.loads(acknowledgment.partition(": ")[2])["line_id"], 2)
                self.assertLessEqual(len(text), MAX_TRANSCRIPT_CHARS)


class DelegationTests(unittest.IsolatedAsyncioTestCase):
    async def test_two_requests_in_flight_remove_what_they_carried(self):
        log, first, second = HeardLog(PROMPTS), Side(held=True), Side(held=True)
        log.caller("Add a latte.")
        sending_first = asyncio.create_task(log.delegate(first))
        await first.arrived.wait()
        reply = present(log, Line(LineKind.REPLY, "One moment.", number=1))
        log.caller("And a muffin.")
        sending_second = asyncio.create_task(log.delegate(second))
        await second.arrived.wait()
        # This playback update happened after the second handoff captured its input frontier.
        present(log, reply, Outcome.HEARD)
        present(log, Line(LineKind.REPLY, "Still working.", number=2), Outcome.HEARD)
        second.decided.set_result("item_2")  # the newer one is accepted first: it carried everything before it
        await sending_second
        self.assertEqual(
            [json.loads(text.partition(": ")[2])["recorded_sentence_prefix"] for text in carried(log)],
            ["One moment.", "Still working."],
        )
        self.assertIn('"playback_state": "queued"', text_of(second.requests[0]))
        log.unsay(1)
        log.unsay(2)
        present(log, Line(LineKind.REPLY, "Done.", number=3), Outcome.HEARD)
        first.decided.set_result("item_1")
        await sending_first
        self.assertEqual(json.loads(carried(log)[0].partition(": ")[2])["recorded_sentence_prefix"], "Done.")
        completed = Side()
        exchange = present(
            log,
            Line(
                LineKind.RESULT,
                "Confirm the exchange.",
                results=True,
                number=4,
                delegation_ids=("D1",),
                work_refs=frozenset(("branch1",)),
                publication_items=({"delegation_id": "D1", "item_id": "proposal1"},),
            ),
        )
        await log.delegate(completed)
        self.assertEqual(carried(log), [])
        # The queued receipt was consumed; later physical completion is a new correlated observation.
        present(log, exchange, Outcome.PLAYING)
        playing = json.loads(carried(log)[0].partition(": ")[2])
        self.assertEqual((playing["playback_state"], playing["recorded_sentence_prefix"]), ("playing", ""))
        self.assertTrue(playing["audio_started"])
        present(log, exchange, Outcome.HEARD)
        receipt = json.loads(carried(log)[0].partition(": ")[2])
        self.assertEqual(
            (receipt["playback_state"], receipt["recorded_sentence_prefix"]), ("complete", "Confirm the exchange.")
        )
        self.assertEqual(receipt["results"], [{"delegation_id": "D1", "item_id": "proposal1"}])
        await log.delegate(completed)
        self.assertEqual(carried(log), [], "unchanged receipts are consumed once")
        _, before_omission = log.delegation_input()
        results = [
            {"delegation_id": "D1", "item_id": "proposal1", "task_ref": "W1.1"},
            {"delegation_id": "D2", "item_id": "proposal2", "task_ref": "W2.1"},
        ]
        disposition = log.not_conveyed(results, "writer_omitted")
        log.sent(before_omission)
        self.assertEqual(carried(log), [line(WORDS.context, disposition)])
        [request], _ = log.delegation_input()
        receipts = request[CALLER_CONTEXT]["playback"]
        self.assertEqual(
            [(row["playback_state"], row["results"]) for row in receipts],
            [("not_conveyed", [result]) for result in results],
        )
        self.assertNotIn("task_ref", disposition)
        log.caller("Yes.")
        log.replace_agent(99, "Unrelated speech.")
        self.assertEqual(log.entries[0].text, disposition)
        await log.delegate(completed)
        self.assertEqual(carried(log), [], "omission evidence is consumed only by a handoff that carried it")

    async def test_an_instruction_is_carried_outside_the_size_cap(self):
        log, side = HeardLog(PROMPTS), Side()
        log.caller("A latte, please.")
        log.instruction("Reply in Spanish from now on.")
        await log.delegate(side)
        present(log, Line(LineKind.REPLY, "Un momento.", number=1), Outcome.HEARD)
        log.caller("And a muffin.")
        newer = [
            playback_only(text) if kind is Heard.AGENT else text
            for kind, text in log.lines()
            if kind is not Heard.INSTRUCTION
        ]
        await log.delegate(side)
        instruction = line(WORDS.instruction, "Reply in Spanish from now on.")
        self.assertEqual(text_of(side.requests[1]), "\n".join([WORDS.earlier, instruction, WORDS.header, *newer]))
        plain = HeardLog(PROMPTS)
        for turn in TURNS:
            log.caller(turn)
            plain.caller(turn)
        await log.delegate(side, unheard=True)
        lines = text_of(side.requests[2]).splitlines()
        self.assertEqual(lines[:4], [WORDS.earlier, instruction, WORDS.header, WORDS.omitted])
        self.assertEqual(lines[-1], WORDS.unheard)
        self.assertEqual(lines[4:-1], text_of(plain.delegation_input()[0]).splitlines()[2:])

    async def test_a_refused_request_keeps_everything_noted(self):
        log, side = HeardLog(PROMPTS), Side(held=True)
        log.caller("What teas do you have?")
        log.instruction("Reply in Spanish from now on.")
        sending = asyncio.create_task(log.delegate(side))
        await side.arrived.wait()
        side.decided.set_exception(RuntimeError("Delegation queue is full"))
        with self.assertRaises(RuntimeError):
            await sending
        asked = line(WORDS.caller, "What teas do you have?")
        instruction = line(WORDS.instruction, "Reply in Spanish from now on.")
        self.assertEqual(carried(log), [asked, instruction])
        log.caller("A latte, please.")
        new = [WORDS.header, asked, instruction, line(WORDS.caller, "A latte, please.")]
        self.assertEqual(text_of(log.delegation_input()[0]).splitlines(), new)
