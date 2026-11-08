# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D103, D107

"""The heard log carries the backend's input: one user message, a labeled transcript."""

import asyncio
import unittest

from examples.frontend_backend_live.heard_log import (
    HEARD,
    MAX_TRANSCRIPT_CHARS,
    Heard,
    HeardLog,
    Unheard,
    UnheardHow,
)
from examples.frontend_backend_live.prompts import TranscriptText, load_prompts
from examples.frontend_backend_live.spoken import Line, LineKind, sentences, unheard_rest
from examples.shared.delegation_contract import ReplyUnheard, question_marks

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


class TranscriptTests(unittest.TestCase):
    def test_cut_lines_are_marked_and_unheard_ones_are_gone(self):
        log = HeardLog(PROMPTS)
        log.agent(1, "Your latte is ready at the front counter.")
        log.agent(2, "Anything else?")
        log.agent(3, "Your total is five dollars.", result=True)  # heard in full: the backend holds its own answer
        log.agent(4, "Your order number is BB-42 and it will be ready soon.", result=True)
        log.cut(1, "Your latte is ready")
        log.unsay(2)
        log.cut(4, "Your order number is")  # a cut result is a delivery note
        cut = [line(WORDS.agent, "Your latte is ready" + PROMPTS.cut_mark)]
        self.assertEqual(carried(log), [*cut, line(WORDS.agent, "Your order number is" + PROMPTS.cut_mark)])

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
        log.agent(1, "Welcome.")
        log.caller("Teas?")
        log.context("Regular.")
        log.instruction("Be brief.")
        items, _ = log.delegation_input()
        self.assertEqual([item["role"] for item in items], ["user"])
        self.assertEqual(text_of(items), "HEADER\nAGENT: Welcome.\nCALLER: Teas?\nCONTEXT: Regular.\nNOTE: Be brief.")
        log.cut(1, "Wel")
        kinds = [Heard.AGENT, Heard.CALLER, Heard.CONTEXT, Heard.INSTRUCTION]
        texts = ["AGENT: Wel <CUT>", "CALLER: Teas?", "CONTEXT: Regular.", "NOTE: Be brief."]
        self.assertEqual(log.lines(), list(zip(kinds, texts, strict=True)))
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
        log.agent(1, "Let me check.")
        log.caller("Okay.")
        heard = [WORDS.header, line(WORDS.agent, "Let me check."), line(WORDS.caller, "Okay.")]
        self.assertEqual(text_of(log.delegation_input(unheard=True)[0]), "\n".join([*heard, WORDS.unheard]))
        self.assertEqual(text_of(log.delegation_input()[0]), "\n".join(heard))
        for turn in TURNS:
            log.caller(turn)
        self.assertEqual(text_of(log.delegation_input(unheard=True)[0]).splitlines()[-1], WORDS.unheard)
        text = text_of(log.delegation_input()[0])
        self.assertLessEqual(len(text), MAX_TRANSCRIPT_CHARS + len(WORDS.header) + len(WORDS.omitted) + 100)
        self.assertTrue(text.startswith(f"{WORDS.header}\n{WORDS.omitted}\n"))
        self.assertTrue(text.endswith(line(WORDS.caller, TURNS[-1])))


class SpelledNoteTests(unittest.TestCase):
    def test_a_caller_line_with_a_spelling_ends_with_the_note_and_keeps_its_words(self):
        log = HeardLog(PROMPTS)
        log.caller("last name e L L I S")
        self.assertEqual(carried(log), [line(WORDS.caller, f"last name e L L I S [{WORDS.spelled}: eLLIS]")])

    def test_with_the_switch_off_or_no_label_lines_are_as_heard(self):
        for log in (
            HeardLog(PROMPTS, spelled_notes=False),
            HeardLog(PROMPTS.model_copy(update={"transcript": WORDS.model_copy(update={"spelled": None})})),
        ):
            log.caller("J O H N")
            self.assertEqual(carried(log), [line(WORDS.caller, "J O H N")])

    def test_a_note_counts_toward_the_size_cap(self):
        log = HeardLog(PROMPTS)
        text = "x" * (MAX_TRANSCRIPT_CHARS - 40) + " A B"
        log.caller("old line")
        log.caller(text)
        items, _ = log.delegation_input()
        self.assertNotIn("old line", text_of(items))  # with the note, the newest line alone fills the cap


class NoteTests(unittest.TestCase):
    def test_a_dropped_answer_is_a_note_with_no_label(self):
        log = HeardLog(PROMPTS)
        log.not_said("The gym opens at six.")
        self.assertEqual(carried(log), [WORDS.not_said.format(answer="The gym opens at six.")])

    def test_a_prompt_set_without_the_not_said_text_adds_no_note(self):
        log = HeardLog(PROMPTS.model_copy(update={"transcript": WORDS.model_copy(update={"not_said": None})}))
        log.not_said("The gym opens at six.")
        self.assertEqual(carried(log), [])


class DelegationTests(unittest.IsolatedAsyncioTestCase):
    async def test_two_requests_in_flight_remove_what_they_carried(self):
        log, first, second = HeardLog(PROMPTS), Side(held=True), Side(held=True)
        log.caller("Add a latte.")
        sending_first = asyncio.create_task(log.delegate(first))
        await first.arrived.wait()
        log.agent(1, "One moment.")
        log.caller("And a muffin.")
        sending_second = asyncio.create_task(log.delegate(second))
        await second.arrived.wait()
        log.agent(2, "Still working.")
        second.decided.set_result("item_2")  # the newer one is accepted first: it carried everything before it
        await sending_second
        self.assertEqual(carried(log), [line(WORDS.agent, "Still working.")])
        log.unsay(2)
        log.agent(3, "Done.")
        first.decided.set_result("item_1")
        await sending_first
        self.assertEqual(carried(log), [line(WORDS.agent, "Done.")])

    async def test_an_instruction_is_carried_outside_the_size_cap(self):
        log, side = HeardLog(PROMPTS), Side()
        log.caller("A latte, please.")
        log.instruction("Reply in Spanish from now on.")
        await log.delegate(side)
        log.agent(1, "Un momento.")
        log.caller("And a muffin.")
        await log.delegate(side)
        instruction = line(WORDS.instruction, "Reply in Spanish from now on.")
        newer = [line(WORDS.agent, "Un momento."), line(WORDS.caller, "And a muffin.")]
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


class UnheardQuestionTests(unittest.TestCase):
    def test_a_result_that_lost_a_sentence_to_missing_audio_is_marked_and_names_its_question(self):
        # Replaces the test that kept what played without the cut mark: partly() now marks the line as cut.
        log = HeardLog(PROMPTS)
        log.agent(1, "Your total is five dollars. Anything else?", result=True)
        log.partly(1, "Your total is five dollars.", unheard=["Anything else?"])
        log.result_settled(1, ["Anything else?"], UnheardHow.NO_AUDIO)
        note = WORDS.cut_question.format(questions="Anything else?")
        self.assertEqual(carried(log), [line(WORDS.agent, "Your total is five dollars." + PROMPTS.cut_mark + note)])
        log = HeardLog(PROMPTS)
        log.agent(1, "Your total is five dollars. Anything else?", result=True)
        log.partly(1, "Anything else?")
        log.result_settled(1, ["Your total is five dollars."], UnheardHow.NO_AUDIO)
        # No question lost: no note.
        self.assertEqual(carried(log), [line(WORDS.agent, "Anything else?" + PROMPTS.cut_mark)])

    def test_a_result_heard_in_full_is_still_left_out(self):
        log = HeardLog(PROMPTS)
        log.agent(1, "The gym opens at six.", result=True)
        log.result_settled(1, [], None)
        self.assertEqual(carried(log), [])

    def test_a_cut_result_names_its_unheard_questions(self):
        log = HeardLog(PROMPTS)
        text = "The gym opens at six. Towels are free. Do you want a locker too?"
        log.agent(1, text, result=True)
        log.cut(1, "The gym opens at six.", unheard=["Do you want a locker too?"])
        log.result_settled(1, ["Towels are free.", "Do you want a locker too?"], UnheardHow.CUT)
        note = WORDS.cut_question.format(questions="Do you want a locker too?")
        self.assertEqual(carried(log), [line(WORDS.agent, "The gym opens at six." + PROMPTS.cut_mark + note)])
        log.agent(2, "The gym opens at six. Towels are free.", result=True)
        log.cut(2, "The gym opens at six.", unheard=[])
        log.result_settled(2, ["Towels are free."], UnheardHow.CUT)
        self.assertEqual(carried(log)[-1], line(WORDS.agent, "The gym opens at six." + PROMPTS.cut_mark))

    def test_the_switch_off_keeps_todays_transcript(self):
        log = HeardLog(PROMPTS, unheard_questions=False)
        log.agent(1, "Your total is five dollars. Anything else?", result=True)
        log.cut(1, "Your total is five dollars.", unheard=["Anything else?"])
        log.result_settled(1, ["Anything else?"], UnheardHow.CUT)
        log.agent(2, "Your total is five dollars. Anything else?", result=True)
        log.partly(2, "Anything else?")
        cut = line(WORDS.agent, "Your total is five dollars." + PROMPTS.cut_mark)
        expected = [cut, line(WORDS.agent, "Anything else?")]
        self.assertEqual(carried(log), expected)

    def test_a_rest_heard_after_its_entry_was_sent_is_a_now_heard_note(self):
        log = HeardLog(PROMPTS)
        log.whole(1, "The gym opens at six. Shall I book you?", rest="Shall I book you?")
        self.assertEqual(carried(log), [WORDS.now_heard.format(rest="Shall I book you?")])


READ_BACK = "The note is ready. It goes in the blue folder. Shall I save it?"


def settle(log: HeardLog, number: int, text: str, heard: str, how=UnheardHow.CUT) -> Unheard:
    """A result line ``text`` that settled heard as ``heard``: the voice's call, with its unheard sentences."""
    rest = sentences(unheard_rest(Line(LineKind.RESULT, text, heard=heard)))
    return log.result_settled(number, rest, how)


class MetadataDelegation(Side):
    """A delegation side that also records what the frontend states about each request."""

    def __init__(self):
        super().__init__()
        self.metadata: list[dict] = []

    async def delegate(self, input, *, metadata=None):
        self.metadata.append(dict(metadata or {}))
        return await super().delegate(input)


class AnswerStateTests(unittest.IsolatedAsyncioTestCase):
    def test_a_result_cut_before_its_question_leaves_the_question_unheard(self):
        log = HeardLog(PROMPTS)
        log.agent(1, READ_BACK, result=True)
        self.assertEqual(
            settle(log, 1, READ_BACK, "The note is ready."), Unheard(ReplyUnheard.QUESTION, UnheardHow.CUT)
        )
        self.assertTrue(log.question_cut)
        self.assertEqual(log.entries[0].lost, ReplyUnheard.QUESTION)

    def test_a_result_cut_after_its_question_leaves_only_the_answer_unheard(self):
        log = HeardLog(PROMPTS)
        text = "Shall I save it? It goes in the blue folder."
        self.assertEqual(settle(log, 1, text, "Shall I save it?").reply, ReplyUnheard.ANSWER)
        self.assertFalse(log.question_cut)

    def test_a_result_cut_after_its_last_word_settles_heard(self):
        log = HeardLog(PROMPTS)  # the caller's "yes, please" overlaps the end of a line whose every sentence played
        self.assertEqual(settle(log, 1, READ_BACK, READ_BACK), HEARD)

    def test_a_stitched_cut_heard_whole_settles_heard(self):
        log = HeardLog(PROMPTS)  # a stitched line cut while it played is heard whole
        self.assertEqual(log.result_settled(1, [], UnheardHow.CUT), HEARD)

    def test_a_result_heard_in_full_clears_the_state(self):
        log = HeardLog(PROMPTS)
        settle(log, 1, READ_BACK, "The note is ready.")
        settle(log, 2, "It is saved.", "It is saved.", None)
        self.assertEqual((log.newest, log.question_cut), (HEARD, False))

    def test_a_silent_sentence_that_held_the_question_sets_it_again_after_the_line_settled_whole(self):
        log = HeardLog(PROMPTS)
        log.agent(1, READ_BACK, result=True)
        settle(log, 1, READ_BACK, READ_BACK, None)
        heard = "The note is ready. It goes in the blue folder."
        log.partly(1, heard)
        self.assertEqual(settle(log, 1, READ_BACK, heard, UnheardHow.NO_AUDIO).reply, ReplyUnheard.QUESTION)

    async def test_the_state_holds_after_a_delegation_took_the_lines_entry(self):
        log, port = HeardLog(PROMPTS), Side()
        log.agent(1, READ_BACK, result=True)
        await log.delegate(port)  # the line plays: a delegation made meanwhile carries it whole and removes it
        self.assertEqual(log.entries, [])
        settle(log, 1, READ_BACK, "The note is ready.")
        self.assertEqual(log.since_sent.reply, ReplyUnheard.QUESTION)

    def test_an_unsaid_result_reads_as_dropped(self):
        log = HeardLog(PROMPTS)
        self.assertEqual(log.result_unsaid(READ_BACK), Unheard(ReplyUnheard.QUESTION, UnheardHow.DROPPED))

    def test_the_cut_mark_holds_no_question_mark(self):
        self.assertEqual(question_marks(PROMPTS.cut_mark), 0)

    def test_since_sent_keeps_an_older_cut_question_after_a_newer_line_is_heard_in_full(self):
        log = HeardLog(PROMPTS)
        settle(log, 1, READ_BACK, "The note is ready.")
        settle(log, 2, "The folder is blue.", "The folder is blue.", None)
        self.assertEqual((log.newest, log.since_sent.reply), (HEARD, ReplyUnheard.QUESTION))

    async def test_since_sent_resets_when_a_delegation_is_accepted_and_not_when_it_is_refused(self):
        log, refusing = HeardLog(PROMPTS), Side(held=True)
        settle(log, 1, READ_BACK, "The note is ready.")
        refusing.decided.set_exception(RuntimeError("refused"))
        with self.assertRaises(RuntimeError):
            await log.delegate(refusing)
        self.assertEqual(log.since_sent.reply, ReplyUnheard.QUESTION)
        held = Side(held=True)
        sending = asyncio.create_task(log.delegate(held))
        await held.arrived.wait()
        settle(log, 2, "Shall I tag it?", "")  # settles while the request is in flight: counts for the next one
        held.decided.set_result("item_1")
        await sending
        self.assertEqual(log.since_sent, Unheard(ReplyUnheard.QUESTION, UnheardHow.CUT))
        await log.delegate(Side())
        self.assertEqual(log.since_sent, HEARD)

    async def test_a_port_without_metadata_still_delegates_and_states_no_fact(self):
        observed, old = [], Side()  # delegate(self, input): written before the keyword
        log = HeardLog(PROMPTS, observe=lambda kind, **fields: observed.append(kind))
        for _ in range(2):
            self.assertTrue(await log.delegate(old, metadata={"live_reply_unheard": "none"}))
        self.assertEqual((len(old.requests), observed), (2, ["delegation.metadata_unsupported"]))
        stating = MetadataDelegation()
        await log.delegate(stating, metadata={"live_reply_unheard": "question"})
        self.assertEqual(stating.metadata, [{"live_reply_unheard": "question"}])
