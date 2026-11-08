# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102
"""The write gate: an app call that may change data waits until the caller heard what they would be agreeing to."""

import asyncio
import unittest

from examples.frontend_backend_live.delegation_core.backend.port import Answer, Start
from examples.frontend_backend_live.delegation_core.config import ToolKindSettings, WriteGateSettings
from examples.frontend_backend_live.delegation_core.im.conversation import AskedQuestions
from examples.shared.delegation_contract import ReplyUnheard
from tests.unit.delegation_core.helpers import (
    TEXTS,
    IMCase,
    OwnCall,
    ScriptedAdapter,
    StartEach,
    WorkTool,
    answer,
    ask,
    call_item,
    checkpoint,
    function_calls,
    placing,
    planned,
    read_all,
    wait,
)
from tests.unit.delegation_core.test_im_calls import Round, outputs

NONE, ANSWER, QUESTION = ReplyUnheard.NONE, ReplyUnheard.ANSWER, ReplyUnheard.QUESTION
ON = WriteGateSettings(enabled=True)


def tool(name: str, **hints) -> dict:
    """A function tool named ``name``, with ``hints`` as its MCP annotations when given."""
    made = {"type": "function", "name": name, "parameters": {"type": "object", "properties": {}}}
    return {**made, "annotations": hints} if hints else made


SAVE = tool("save_note", readOnlyHint=False)  # a declared write
READ = tool("read_note", readOnlyHint=True)  # a declared read
TAG = tool("tag_note", readOnlyHint=False, destructiveHint=False)  # a declared, easily undone write
PLAIN, OTHER = tool("plain_tool"), tool("other_plain_tool")  # declared by nobody
LIST, PURGE = tool("list_notes"), tool("purge_notes")  # declared by the operator's lists
APP = {"tools": [SAVE, READ, TAG, PLAIN, OTHER, LIST, PURGE]}


def calls(*names: str, round_number: int = 1) -> Round:
    """One round that calls the app's functions ``names``, call ids ``c<round>_<n>``."""
    return Round(*(call_item(name, {}, f"c{round_number}_{n}") for n, name in enumerate(names)))


def gate_traces(traces: list) -> list[dict]:
    """The fields of each ``im.write_gate`` trace, in order."""
    return [fields for kind, fields in traces if kind == "im.write_gate"]


class WriteGateTests(IMCase):
    async def first(self, *scripts, fact=QUESTION, gate=ON, adapter_options=None, **settings):
        """A session with the gate ``gate`` on ``scripts``, and the events of D1 stating ``fact``."""
        adapter = ScriptedAdapter(*scripts, **(adapter_options or {}))
        im = self.session(adapter, write_gate=gate, **settings)
        stream = im.admit(ask("Save my note."), delegation_id="D1", settings=APP, reply_unheard=fact).stream
        return adapter, im, await read_all(stream)

    async def test_a_write_after_a_delegation_with_an_unheard_question_is_held_and_never_reaches_the_app(self):
        adapter, im, events = await self.first([calls("save_note"), answer("Shall I save the note?")])
        self.assertEqual((adapter.runs[0].routes, function_calls(events)), (["refused"], []))
        self.assertEqual(outputs(adapter.runs[0].histories[-1]), {"c1_0": TEXTS.unheard_write_output})
        self.assertEqual(im.ledger.actions["c1_0"].state, "not_run")

    async def test_the_same_write_after_a_delegation_that_heard_everything_reaches_the_app(self):
        adapter, _, events = await self.first([calls("save_note")], fact=NONE)
        self.assertEqual((adapter.runs[0].routes, function_calls(events)), (["app"], ["c1_0"]))

    async def test_an_unheard_statement_holds_a_write_in_answer_mode_and_not_in_question_mode(self):
        adapter, _, _ = await self.first([calls("save_note"), answer()], fact=ANSWER)
        self.assertEqual(adapter.runs[0].routes, ["refused"])
        narrow = WriteGateSettings(enabled=True, unheard="question")
        adapter, _, events = await self.first([calls("save_note")], fact=ANSWER, gate=narrow)
        self.assertEqual(function_calls(events), ["c1_0"])

    async def test_a_request_that_states_no_fact_runs_its_write(self):
        adapter, im, events = await self.first([calls("save_note")], fact=None)
        self.assertEqual(function_calls(events), ["c1_0"])
        adapter = ScriptedAdapter([calls("save_note")])
        im = self.session(adapter, write_gate=ON)
        typed = await read_all(im.admit_client_request(ask("Save it."), settings=APP).stream)
        self.assertEqual(function_calls(typed), ["c1_0"])

    async def test_a_declared_read_runs_whatever_the_fact(self):
        kinds = ToolKindSettings(read_tools=("list_*",))
        adapter, _, events = await self.first([calls("read_note", "list_notes")], tool_kinds=kinds)
        self.assertEqual(function_calls(events), ["c1_0", "c1_1"])

    async def test_an_undeclared_tool_is_held_once_then_runs_and_runs_at_once_with_undeclared_run(self):
        script = [calls("plain_tool"), calls("plain_tool", round_number=2)]
        adapter, im, events = await self.first(script)
        self.assertEqual((adapter.runs[0].routes, function_calls(events)), (["refused", "app"], ["c2_0"]))
        self.assertEqual(outputs(adapter.runs[0].histories[-1]), {"c1_0": TEXTS.unheard_undeclared_output})
        task = im.works[next(iter(im.works))].tasks[0]
        self.assertFalse(task.tools_closed)
        run = WriteGateSettings(enabled=True, undeclared="run")
        adapter, _, events = await self.first([calls("plain_tool")], gate=run)
        self.assertEqual(function_calls(events), ["c1_0"])

    async def test_a_declared_write_called_again_after_its_hold_is_held_again(self):
        script = [calls("save_note"), calls("save_note", round_number=2), calls("save_note", round_number=3), answer()]
        adapter, im, events = await self.first(script)
        self.assertEqual((adapter.runs[0].routes, function_calls(events)), (["refused"] * 3, []))
        self.assertEqual([f["outcome"] for f in gate_traces(self.traces)], ["held", "held_again"])
        read = outputs(adapter.runs[0].histories[-1])
        self.assertEqual(set(read.values()), {TEXTS.unheard_write_output})

    async def test_two_undeclared_writes_in_one_round_are_both_held(self):
        script = [calls("plain_tool", "other_plain_tool"), calls("plain_tool", "other_plain_tool", round_number=2)]
        adapter, _, events = await self.first(script)
        self.assertEqual(adapter.runs[0].routes, ["refused", "refused", "app", "app"])
        self.assertEqual(function_calls(events), ["c2_0", "c2_1"])
        held = dict.fromkeys(["c1_0", "c1_1"], TEXTS.unheard_undeclared_output)
        self.assertEqual(outputs(adapter.runs[0].histories[1]), held)

    async def test_a_call_whose_own_check_passes_is_held_with_its_rounds_hold(self):
        # An older question never heard holds the round's first change (open_question, once per question); the second
        # change of that round passes its own check, and is held with the round, with the same output.
        scripts = [[answer("Which of the two lists?")], [answer("The lists are A and B.")]]
        adapter, im, _ = await self.first(*scripts, fact=NONE, tool_kinds=ToolKindSettings(write_tools=("purge_*",)))
        await read_all(im.admit(ask("Hm?"), delegation_id="D2", settings=APP, reply_unheard=QUESTION).stream)
        adapter.scripts.append([calls("save_note", "purge_notes"), answer()])
        await read_all(im.admit(ask("Save it."), delegation_id="D3", settings=APP, reply_unheard=NONE).stream)
        traced = [(f["reason"], f["with_round"]) for f in gate_traces(self.traces)]
        self.assertEqual(traced, [("open_question", False), ("open_question", True)])
        named = TEXTS.unheard_question_output.format(question="Which of the two lists?")
        self.assertEqual(set(outputs(adapter.runs[-1].histories[-1]).values()), {named})

    async def test_a_lookup_and_a_change_in_one_round_cost_one_extra_round(self):
        script = [calls("read_note", "plain_tool", "save_note"), calls("plain_tool", "save_note", round_number=2)]
        adapter, _, events = await self.first(script)
        self.assertEqual(adapter.runs[0].routes[:3], ["app", "refused", "refused"])
        self.assertEqual(function_calls(events), ["c1_0"])  # the read reaches the app at once

    async def test_a_held_lookup_does_not_let_a_later_undeclared_write_run(self):
        script = [calls("plain_tool"), calls("plain_tool", "other_plain_tool", round_number=2)]
        adapter, _, _ = await self.first(script)
        self.assertEqual(adapter.runs[0].routes, ["refused", "app", "refused"])

    async def test_held_again_only_for_a_declared_write_that_was_itself_held(self):
        script = [calls("plain_tool"), calls("save_note", round_number=2), answer()]
        adapter, im, _ = await self.first(script)
        self.assertEqual([f["reason"] for f in gate_traces(self.traces)], ["reply_unheard", "reply_unheard"])
        self.assertFalse(any(task.tools_closed for work in im.ended_works for task in work.tasks))

    async def test_an_unlabelled_read_held_once_runs_on_the_next_request(self):
        adapter, im, _ = await self.first([calls("plain_tool"), answer("Which note?")], [calls("plain_tool")])
        d2 = im.admit(ask("The red one."), delegation_id="D2", settings=APP, reply_unheard=NONE)
        events = await read_all(d2.stream)
        self.assertEqual((adapter.runs[1].routes, function_calls(events)), (["app"], ["c1_0"]))

    async def test_a_reversible_write_runs_by_default_and_is_held_only_with_reversible_hold(self):
        _, _, events = await self.first([calls("tag_note")])
        self.assertEqual(function_calls(events), ["c1_0"])
        hold = WriteGateSettings(enabled=True, reversible="hold")
        adapter, _, _ = await self.first([calls("tag_note"), answer()], gate=hold)
        self.assertEqual(adapter.runs[0].routes, ["refused"])

    async def test_a_question_written_after_the_delegation_was_admitted_holds_the_write(self):
        for asked, route in (("Which colour do you want?", "refused"), ("The colour is blue.", "app")):
            with self.subTest(asked=asked):
                first, second = asyncio.Event(), asyncio.Event()
                one = [wait(first), answer(asked)]
                two = [wait(second), calls("save_note"), answer()]
                adapter = ScriptedAdapter(one, two, max_running=2, conductor=StartEach())
                im = self.session(adapter, write_gate=ON)
                d1 = im.admit(ask("What colours are there?"), delegation_id="D1", settings=APP, reply_unheard=NONE)
                d2 = im.admit(ask("Save the note."), delegation_id="D2", settings=APP, reply_unheard=NONE)
                await self.until(lambda runs=adapter.runs: len(runs) == 2, "both tasks")
                first.set()
                await read_all(d1.stream)
                second.set()
                await read_all(d2.stream)
                self.assertEqual(adapter.runs[1].routes, [route])
                held = [f["reason"] for f in gate_traces(self.traces) if f["outcome"] == "held"]
                self.assertEqual(held, ["newer_answer"] if route == "refused" else [])
                self.traces.clear()

    async def test_an_older_unheard_question_holds_one_later_change_and_names_it(self):
        scripts = [
            [answer("Which of the two lists?")],
            [answer("The lists are A and B.")],
            [calls("save_note"), answer()],
        ]
        adapter, im, _ = await self.first(*scripts, fact=NONE)
        await read_all(im.admit(ask("Hm?"), delegation_id="D2", settings=APP, reply_unheard=QUESTION).stream)
        await read_all(im.admit(ask("Save it."), delegation_id="D3", settings=APP, reply_unheard=NONE).stream)
        named = TEXTS.unheard_question_output.format(question="Which of the two lists?")
        self.assertEqual(outputs(adapter.runs[2].histories[-1]), {"c1_0": named})
        self.assertTrue(im.questions.entries["whichofthetwolists"].challenged)
        adapter.scripts.append([calls("save_note")])
        d4 = im.admit(ask("Save it."), delegation_id="D4", settings=APP, reply_unheard=NONE)
        events = await read_all(d4.stream)
        self.assertEqual(function_calls(events), ["c1_0"])

    async def test_a_change_after_a_delegation_whose_options_question_lost_its_sentence_is_held(self):
        # The options question got no audio (the frontend states question); the backend picks a value anyway.
        adapter, im, events = await self.first([calls("save_note"), answer("Which version?")], [calls("save_note")])
        self.assertEqual((adapter.runs[0].routes, function_calls(events)), (["refused"], []))
        self.assertEqual(outputs(adapter.runs[0].histories[-1]), {"c1_0": TEXTS.unheard_write_output})
        d2 = im.admit(ask("The first one."), delegation_id="D2", settings=APP, reply_unheard=NONE)
        self.assertEqual(function_calls(await read_all(d2.stream)), ["c1_0"])

    async def test_a_change_after_an_options_question_cut_two_answers_ago_is_held_once_naming_it(self):
        scripts = [[answer("Which of the two versions?")], [answer("Both were saved today.")]]
        scripts.append([calls("save_note"), answer()])
        adapter, im, _ = await self.first(*scripts, fact=NONE)
        await read_all(im.admit(ask("When?"), delegation_id="D2", settings=APP, reply_unheard=QUESTION).stream)
        await read_all(im.admit(ask("The other one."), delegation_id="D3", settings=APP, reply_unheard=NONE).stream)
        named = TEXTS.unheard_question_output.format(question="Which of the two versions?")
        self.assertEqual(outputs(adapter.runs[2].histories[-1]), {"c1_0": named})

    async def test_a_status_answer_after_the_yes_does_not_hold_its_write(self):
        gate = asyncio.Event()
        conductor = planned(placing(Start(("D1",))), placing(Answer("D2", "Still on it. Anything else?")))
        adapter = ScriptedAdapter([wait(gate), calls("save_note")], conductor=conductor)
        im = self.session(adapter, write_gate=ON)
        d1 = im.admit(ask("Yes, save it."), delegation_id="D1", settings=APP, reply_unheard=NONE)
        await self.until(lambda: adapter.runs, "the task")
        await read_all(im.admit(ask("Are you there?"), delegation_id="D2", settings=APP, reply_unheard=NONE).stream)
        gate.set()
        self.assertEqual(function_calls(await read_all(d1.stream)), ["c1_0"])
        self.assertEqual(im.questions.answers, 0)  # a status answer is not recorded

    async def test_steering_that_lands_brings_its_own_fact_and_reopens_closed_tools(self):
        gate = asyncio.Event()
        script = [
            calls("save_note"),
            calls("save_note", round_number=2),  # held again: the task's tools close
            wait(gate),
            checkpoint(),  # D2 lands
            calls("save_note", round_number=3),
        ]
        adapter = ScriptedAdapter(script)
        im = self.session(adapter, write_gate=ON)
        d1 = im.admit(ask("Save my note."), delegation_id="D1", settings=APP, reply_unheard=QUESTION)
        await self.until(lambda: adapter.runs and len(adapter.runs[0].routes) == 2, "the two holds")
        task = next(iter(im.works.values())).tasks[0]
        self.assertTrue(task.tools_closed)
        d2 = im.admit(ask("Yes, save it."), delegation_id="D2", settings=APP, reply_unheard=NONE)
        await self.turns()
        gate.set()
        await read_all(d1.stream)
        events = await read_all(d2.stream)  # D2 carries the Work once it landed
        self.assertFalse(task.tools_closed)
        self.assertEqual((adapter.runs[0].routes, function_calls(events)), (["refused", "refused", "app"], ["c3_0"]))

    async def test_backend_tools_and_work_tools_are_never_held(self):
        adapter, _, _ = await self.first([OwnCall("lookup"), WorkTool("cancel_work", {}), answer()])
        self.assertEqual(adapter.runs[0].routes[0], "backend")
        self.assertEqual(gate_traces(self.traces), [])

    async def test_a_gate_that_is_off_changes_nothing_but_still_counts(self):
        adapter, im, events = await self.first([calls("save_note")], gate=WriteGateSettings())
        self.assertEqual(function_calls(events), ["c1_0"])
        self.assertEqual(gate_traces(self.traces), [])
        await self.answer_calls(im, events, settings=APP)
        self.assertEqual(im.heard("D1").level, QUESTION)

    async def test_each_checked_write_is_traced_with_its_outcome_reason_and_kind_source(self):
        kinds = ToolKindSettings(write_tools=("purge_notes",))
        await self.first([calls("save_note", "purge_notes", "plain_tool"), answer()], tool_kinds=kinds)
        traced = [(f["tool"], f["outcome"], f["reason"], f["kind_source"]) for f in gate_traces(self.traces)]
        self.assertEqual(
            traced,
            [
                ("save_note", "held", "reply_unheard", "annotations"),
                ("purge_notes", "held", "reply_unheard", "operator"),
                ("plain_tool", "held", "reply_unheard", ""),
            ],
        )


class QuestionRecordTests(unittest.TestCase):
    def test_replied_counts_heard_in_full_only_with_a_fact(self):
        cases = ((NONE, 1, False), (ANSWER, 1, False), (QUESTION, None, True), (None, None, False))
        for level, heard, unheard in cases:
            with self.subTest(level=level):
                record = AskedQuestions()
                record.answer_written("Shall I save it?", ["D1"])
                self.assertEqual(record.replied(level), 1)
                (entry,) = record.entries.values()
                self.assertEqual((entry.heard_in_full, entry.unheard, entry.awaiting), (heard, unheard, False))

    def test_heard_in_full_adds_up_over_the_call(self):
        record = AskedQuestions()
        for _ in range(3):
            record.answer_written("Shall I save it?", ["D1"])
            record.replied(NONE)
        (entry,) = record.entries.values()
        self.assertEqual((entry.asked, entry.heard_in_full), (3, 3))

    def test_two_questions_in_one_answer_are_both_marked_unheard_by_a_question_level(self):
        record = AskedQuestions()
        record.answer_written("Which list? And shall I save it?", ["D1"])
        record.replied(QUESTION)
        self.assertEqual([entry.unheard for entry in record.entries.values()], [True, True])
        self.assertEqual(record.open_question().text, "Which list?")

    def test_a_question_asked_again_and_heard_clears_its_open_state(self):
        record = AskedQuestions()
        record.answer_written("Which list?", ["D1"])
        record.replied(QUESTION)
        record.open_question().challenged = True
        record.answer_written("which list?", ["D2"])  # asked again: the same key
        (entry,) = record.entries.values()
        self.assertEqual((entry.unheard, entry.challenged, entry.asked, entry.text), (False, False, 2, "which list?"))
        record.replied(NONE)
        self.assertIsNone(record.open_question())

    def test_a_question_from_the_work_answering_the_same_request_does_not_hold_its_write(self):
        record = AskedQuestions()
        record.answer_written("Which list?", ["D1", "D2"])
        self.assertFalse(record.asked_since(0, besides="D2"))
        self.assertTrue(record.asked_since(0, besides="D3"))
        self.assertFalse(record.asked_since(1, besides="D3"))

    def test_an_answer_that_asks_nothing_is_counted_and_records_no_question(self):
        record = AskedQuestions()
        self.assertEqual(record.answer_written("The note is saved.", ["D1"]), [])
        self.assertEqual((record.answers, record.written, record.entries), (1, 0, {}))

    def test_newest_lists_the_most_recently_asked_oldest_first(self):
        record = AskedQuestions()
        for text in ("One?", "Two?", "Three?", "One?"):
            record.answer_written(text, ["D1"])
        self.assertEqual([entry.text for entry in record.newest(2)], ["Three?", "One?"])
        self.assertEqual(record.newest(0), [])


if __name__ == "__main__":
    unittest.main()
