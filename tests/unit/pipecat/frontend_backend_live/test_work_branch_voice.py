# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D103, D107
"""Branch-specific Voice composition/turnover using actual records, without a model or playback worker."""

import json
import unittest
from types import SimpleNamespace

from pipecat.processors.aggregators.llm_context import LLMContext

from nemotron_voice_agent.pipecat.examples.frontend_backend_live.config_manager.schema import MemoryConfig
from nemotron_voice_agent.pipecat.examples.frontend_backend_live.delegations import ReadyResult
from nemotron_voice_agent.pipecat.examples.frontend_backend_live.models.decision import TalkerDecision
from nemotron_voice_agent.pipecat.examples.frontend_backend_live.ownership import OwnerView, WorkOwnership
from nemotron_voice_agent.pipecat.examples.frontend_backend_live.spoken import Line, LineKind
from nemotron_voice_agent.pipecat.examples.frontend_backend_live.voice import RESULT, NamedNotice, Voice, message_tokens
from nemotron_voice_agent.pipecat.examples.frontend_backend_live.work_provenance import Attribution, Binding, Obligation
from tests.unit.pipecat.frontend_helpers import CONFIG, PROMPTS, Port


def obligation(ref, branch):
    return Obligation(ref, branch, "active", False, False, None, None, None)


def voice(*, known_origins=True):
    result = Voice(
        config=CONFIG,
        prompts=PROMPTS.cascade,
        lines=PROMPTS.lines,
        json_rules="",
        empty_turn_note="",
        delegation=Port(),
        item_text=lambda item: "",
        observe=lambda *args, **kw: None,
        clock=lambda: 12.0,
    )
    result.context = LLMContext([])
    return result, shared_ownership(result.worknote, known_origins=known_origins)


def shared_ownership(note, *, known_origins=True):
    """Open two independent Work branches and give both to one response."""
    ownership = WorkOwnership()
    if known_origins:
        for branch, did in (("A", "D1"), ("B", "D2")):
            note.confirm(note.reserve("request " + branch), did)
            note.running(did)
            ownership.transports["O" + branch] = did
            ownership.bindings[branch] = Binding(branch, "O" + branch, 0)
            ownership.owners["O" + branch] = OwnerView(0, (obligation("old-" + branch, branch),))
        note.sync_ownership(ownership)
    note.confirm(note.reserve("request A and B together"), "D3")
    note.running("D3")
    ownership.transports["O3"] = "D3"
    ownership.bindings = {branch: Binding(branch, "O3", 1) for branch in ("A", "B")}
    ownership.owners["O3"] = OwnerView(1, tuple(obligation("shared-" + b, b) for b in ("A", "B")))
    ownership.composite.add("O3")
    note.sync_ownership(ownership)
    return ownership


def fact_line(voice, branch, number):
    line = Line(
        LineKind.STATUS,
        "fact " + branch,
        ("D3",),
        tells=("D3",),
        number=number,
        work_refs=frozenset((branch,)),
        work_tells=frozenset((branch,)),
    )
    voice.spoken.handed(line)
    voice.spoken.started(number)
    voice.spoken.ended(number)
    voice.spoken.end_played(number, line.text)
    return line


class BranchVoiceTests(unittest.TestCase):
    def test_mixed_results_keep_their_own_writer_history_policy(self):
        ordinary = ReadyResult("D3", "answer-B", "B result", attribution=Attribution("B", "O3", 1))
        focused = ReadyResult("D3", "answer-A", "A result", attribution=Attribution("A", "O3", 1, focused=True))
        question = ReadyResult("D3", "question", "Which booking?", routing_question=True)
        for first, second in (
            (focused, ordinary),
            (ordinary, focused),
            (question, ordinary),
            (ordinary, question),
            (question, focused),
            (focused, question),
        ):
            with self.subTest(first=first.item_id, second=second.item_id):
                v, _ = voice()
                v.instructions.append("Speak Spanish.")
                original = {"role": "user", "content": "Move the booking."}
                later = {"role": "user", "content": "Cancel the afternoon booking."}
                earlier_answer = {"role": "assistant", "content": "The old balance was thirty dollars."}
                history = [earlier_answer, original]
                v.context.set_messages(list(history))
                v.input_delivery[id(original)] = {"delegation_id": "D3"}
                for result in (first, second):
                    v.worknote.answered("D3", attribution=result.attribution, item_id=result.item_id)
                v.ready(first)
                v.talker = SimpleNamespace(response_format=None)
                v.call = initial = v._relay_call(v._take())
                v.talker = None
                v.ready(second)
                self.assertFalse(initial.cut)
                self.assertFalse(initial.plan.moved(v))
                v.call = None
                v.talker = SimpleNamespace(response_format=None)
                deferred = v._relay_call(v._take())
                for call, result in ((initial, first), (deferred, second)):
                    self.assertEqual([item.said for item in call.plan.batch], [result])
                    messages = call.request.messages(history)
                    self.assertEqual(earlier_answer in messages, result is not question)
                    self.assertIn(result.speak, messages[-1]["content"])
                    if result is focused:
                        reference = {"role": "user", "content": v.prompts.result_label + "Earlier booking confirmed."}
                        v._register(reference, RESULT)
                        v.context.set_messages([*history, later, reference])
                        self.assertTrue(call.plan.moved(v))
                        updated, retained = v._prepare_relay(call.plan.batch)
                        current = updated.request.messages(retained)
                        self.assertIn(original["content"] + "\n" + later["content"], current[2]["content"])
                        newest = v.prompts.newest_turn.format(turn=later["content"])
                        self.assertTrue(
                            current[-1]["content"].endswith(f"{newest}\n\n{v.prompts.update_label}{result.speak}")
                        )
                        self.assertNotIn(ordinary.speak, current[-1]["content"])
                        v.context.set_messages(list(history))
                    if result is question:
                        self.assertIn("Speak Spanish.", messages[0]["content"])
                        self.assertIn(v.prompts.routing_question.strip(), messages[0]["content"])
                        self.assertIn(original["content"], messages[-2]["content"])
                        self.assertNotIn("request A and B together", messages[-2]["content"])
                        v.context.add_message(later)
                        self.assertTrue(call.plan.moved(v))
                        for retained in ([*history, later], [earlier_answer, later]):
                            v.context.set_messages(retained)
                            current = v._relay_call(call.plan.batch).request.messages(retained)
                            self.assertIn(later["content"], current[-2]["content"])
                            self.assertNotIn(earlier_answer, current)

    def test_writer_context_and_delivered_result_consume_only_named_branch(self):
        v, _ = voice()
        a, b = fact_line(v, "A", 1), fact_line(v, "B", 2)
        attr = Attribution("A", "O3", 1, focused=True)
        v.worknote.answered("D3", attribution=attr, item_id="answer-A")
        v.ready(ReadyResult("D3", "answer-A", "A result", attribution=attr))
        batch = v._take()
        content, note = v._update(batch)
        self.assertEqual(content, "A result")
        self.assertIn("fact A", note)
        self.assertNotIn("fact B", note)
        self.assertNotIn("request A and B together", note)
        contexts = {row["branch_ref"]: row["relationship"] for row in v.worknote.newer_requests((), ("A",))}
        self.assertEqual(contexts, {"A": "same_branch", "B": "other_open_source_relevance_unknown"})
        line = v.delivered(batch, "A result")
        self.assertEqual(line.work_refs, frozenset(("A",)))
        self.assertEqual(json.loads(v.spoken.context_text(work_refs=("A",)))[0]["recorded_sentence_prefix"], "fact A")
        self.assertEqual(v.worknote.entries["D3"].ready, 1, "creating a candidate does not settle its result")
        self.assertEqual(v.hand(v.guard_presentation(line, batch)), "A result")
        self.assertEqual(
            [item["recorded_sentence_prefix"] for item in json.loads(v.spoken.context_text(work_refs=("A",)))], [""]
        )
        self.assertEqual(json.loads(v.spoken.context_text(work_refs=("B",)))[0]["recorded_sentence_prefix"], "fact B")
        self.assertEqual((a.heard, b.heard), ("fact A", "fact B"))
        v._done(line.items)
        self.assertEqual(v.worknote.entries["D3"].ready, 0)
        v.config = CONFIG.model_copy(update={"memory": MemoryConfig(context_tokens=2_000)})
        overflow = {"role": "user", "content": "request A and B together " * 1_000}
        latest = {"role": "user", "content": "Skip the lookup."}
        v.context.set_messages([overflow, latest])
        v.decided[id(overflow)] = overflow
        v.talker = SimpleNamespace(response_format=None)
        v.spoken.handed(Line(LineKind.STATUS, "Unplayed wording", ("D3",), work_refs=frozenset(("A",))))
        relay, history = v._prepare_relay(batch)
        wire = relay.request.messages(history)
        self.assertEqual(v.context.get_messages(), [latest])
        self.assertEqual([message["role"] for message in wire], ["system", "user", "user"])
        self.assertIn("A result", wire[-1]["content"])
        self.assertIn(latest["content"], wire[1]["content"])
        self.assertNotIn("request A and B together", wire[1]["content"])
        self.assertNotIn("Unplayed wording", wire[1]["content"])
        compact = v._relay_call(batch).request.messages(history)
        limit = sum(map(message_tokens, compact))
        v.config = CONFIG.model_copy(update={"memory": MemoryConfig(context_tokens=limit)})
        relay, history = v._prepare_relay(batch)
        self.assertEqual(relay.request.messages(history), compact)

    def test_writer_keeps_each_result_bound_to_its_branch_in_a_shared_response(self):
        v, _ = voice()
        v.ready(
            ReadyResult("D3", "result-A", "7 is available.\n\nShall I book it?", attribution=Attribution("A", "O3", 1))
        )
        v.ready(ReadyResult("D3", "result-B", "The movie was stopped.", attribution=Attribution("B", "O3", 1)))
        batch = v._take()
        content, _ = v._update(batch)
        self.assertEqual(
            json.loads(content),
            [
                {
                    "item_id": "result-A",
                    "delegation_id": "D3",
                    "branch_ref": "A",
                    "text": "7 is available.\n\nShall I book it?",
                },
                {"item_id": "result-B", "delegation_id": "D3", "branch_ref": "B", "text": "The movie was stopped."},
            ],
        )
        v.config = CONFIG.model_copy(update={"memory": MemoryConfig(context_tokens=2_000)})
        protected = {"role": "user", "content": "Earlier delivery is uncertain."}
        old = {"role": "user", "content": "old " * 1_450}
        kept = {"role": "user", "content": "Keep any history that fits."}
        latest = {"role": "user", "content": "Look up the reservation."}
        v.context.set_messages([protected, old, kept, latest])
        v.decided.update((id(turn), turn) for turn in v.context.get_messages())
        v.input_delivery[id(protected)] = {"state": "unknown"}
        self.assertTrue(v.bound())
        self.assertIn(old, v.context.get_messages())
        v.talker = SimpleNamespace(response_format=None)
        self.assertIsNotNone(v.relay(batch).request)
        self.assertEqual(v.context.get_messages(), [protected, kept, latest])
        v.talker = None

        prefix = "The movie was stopped."
        words = f"{prefix} Seven is available. Shall I book it?"
        line = v.delivered(batch, words)
        self.assertEqual(v.hand(line), words)
        v.spoken.started(line.number)
        v.spoken.cut(line.number, 101)
        v.spoken.cut_played(101, prefix)

        result_sources = [
            {"item_id": "result-A", "delegation_id": "D3", "branch_ref": "A"},
            {"item_id": "result-B", "delegation_id": "D3", "branch_ref": "B"},
        ]
        [presentation] = json.loads(v.spoken.context_text(work_refs=("A", "B")))
        self.assertEqual(presentation["results"], result_sources)
        self.assertEqual(presentation["recorded_sentence_prefix"], prefix)
        self.assertFalse(presentation["complete_conveyance"])
        transcript, _ = v.heard_log.delegation_input()
        spoken = transcript[0]["content"][0]["text"].splitlines()[-1]
        receipt = json.loads(spoken.partition(": ")[2])
        self.assertEqual(receipt["results"], result_sources)
        self.assertEqual(receipt["recorded_sentence_prefix"], prefix)
        self.assertFalse(receipt["complete_conveyance"])

    def test_branch_turnover_does_not_wait_for_or_consume_sibling_result(self):
        v, _ = voice()
        fact_line(v, "A", 1)
        fact_line(v, "B", 2)
        v.ready(ReadyResult("D3", "answer-B", "B result", attribution=Attribution("B", "O3", 1)))
        v.work_ended("A", "D3")
        batch = v._take()
        self.assertEqual([item.said.item_id for item in batch], ["answer-B"])
        self.assertIsNone(v.spoken.context_text(work_refs=("A",)))
        self.assertEqual(json.loads(v.spoken.context_text(work_refs=("B",)))[0]["recorded_sentence_prefix"], "fact B")

    def test_stale_owner_end_does_not_end_current_branch_or_consume_its_context(self):
        v, ownership = voice()
        fact_line(v, "A", 1)
        v.worknote.confirm(v.worknote.reserve("new A input"), "D4")
        ownership.transports["O4"] = "D4"
        ownership.bindings["A"] = Binding("A", "O4", 2)
        v.worknote.sync_ownership(ownership)
        v.work_ended("A", "D3")
        self.assertFalse(v.worknote.branches["A"].ended)
        self.assertEqual(json.loads(v.spoken.context_text(work_refs=("A",)))[0]["recorded_sentence_prefix"], "fact A")
        self.assertEqual(list(v.owed), [])

    def test_notice_uses_exact_origin_not_composite_parent_or_fixed_when_unknown(self):
        for known in (True, False):
            with self.subTest(known=known):
                v, _ = voice(known_origins=known)
                v.work_notice("D3", "The action is uncertain.", Attribution("A", "O3", 1))
                owed = v.owed[0]
                if known:
                    self.assertIsInstance(owed.said, NamedNotice)
                    self.assertIn("request A", owed.said.speak)
                    self.assertNotIn("request A and B together", owed.said.speak)
                else:
                    self.assertIsNone(owed.message)
                    self.assertEqual(owed.said.speak, "The action is uncertain.")

    def test_joint_progress_charges_each_branch_and_keeps_news_after_supersession(self):
        for superseded in (False, True):
            with self.subTest(superseded=superseded):
                v, _ = voice()
                for branch in ("A", "B"):
                    v.worknote.progress("D3", "verified " + branch, attribution=Attribution(branch, "O3", 1))
                v.progress_due = tuple(v.worknote.progress_key(entry) for entry in v.worknote.quiet_candidates(3))
                v.talker = SimpleNamespace(response_format=None)
                call = v._progress()
                v.talker = None
                self.assertEqual({source.branch_ref for source in call.plan.sources}, {"A", "B"})
                for fact in ("verified A", "verified B"):
                    self.assertIn(fact, str(call.request))
                payload = call.request.messages([])[-1]["content"]
                for branch in ("A", "B"):
                    self.assertIn(f'{{"work_refs": ["{branch}"]}}', payload)

                text = ""
                if superseded:
                    v.worknote.progress("D3", "new A fact", attribution=Attribution("A", "O3", 1))
                    text = "old joint line"
                self.assertEqual(call.plan.finished(v, call, TalkerDecision("speak", text, {})), "")
                self.assertEqual([v.worknote.branches[branch].progress_offers for branch in ("A", "B")], [1, 1])
                self.assertEqual(bool(v.worknote.branches["B"].news), superseded)
                if superseded:
                    eligible = {entry.branch_ref: entry.progress for entry in v.worknote.quiet_candidates(3)}
                    self.assertEqual(eligible.get("A"), "new A fact")
                else:
                    self.assertFalse(v.worknote.branches["A"].news)

        v, _ = voice()
        v.config = v.config.model_copy(update={"memory": MemoryConfig(context_tokens=32)})
        for branch in ("A", "B"):
            v.worknote.progress("D3", "verified " + branch, attribution=Attribution(branch, "O3", 1))
        v.progress_due = tuple(v.worknote.progress_key(entry) for entry in v.worknote.quiet_candidates(1))
        v.talker = SimpleNamespace(response_format=None)
        self.assertIsNone(v._progress())
        self.assertEqual([entry.branch_ref for entry in v.worknote.quiet_candidates(1)], ["A", "B"])

    def test_restored_progress_never_creates_news_or_erases_fresh_live_fact(self):
        v, _ = voice()
        attr = Attribution("A", "O3", 1)
        v.worknote.restore_progress("D3", "retained A fact", attribution=attr)
        self.assertEqual(v.worknote.quiet_candidates(3), ())
        v.worknote.progress("D3", "fresh A fact", attribution=attr)
        revision = v.worknote.branches["A"].revision
        v.worknote.restore_progress("D3", "older retained fact", attribution=attr)
        self.assertEqual(v.worknote.branches["A"].progress, "fresh A fact")
        self.assertEqual(v.worknote.branches["A"].revision, revision)
        self.assertTrue(v.worknote.branches["A"].news)

    def test_same_obligation_followup_reads_actual_heard_and_inflight_predecessor_once(self):
        for settled in (False, True):
            with self.subTest(settled=settled):
                v, ownership = voice()
                old = Line(
                    LineKind.STATUS,
                    "Old A was checked.",
                    ("D3",),
                    tells=("D3",),
                    number=1,
                    work_refs=frozenset(("A",)),
                    work_tells=frozenset(("A",)),
                )
                v.spoken.handed(old)
                v.spoken.started(1)
                if settled:
                    v.spoken.ended(1)
                    v.spoken.end_played(1, old.text)
                before = old.heard, old.outcome, old.delegation_ids, old.work_tells
                v.worknote.branches["A"].progress_offers = 2
                v.worknote.confirm(v.worknote.reserve("same pending input"), "D4")
                ownership.transports["O4"] = "D4"
                ownership.bindings["A2"] = Binding("A2", "O4", 0)
                ownership.owners["O4"] = OwnerView(1, (obligation("shared-A", "A2"),))
                v.worknote.sync_ownership(ownership)
                v.work_binding("A2", "D4")
                v.worknote.sync_ownership(ownership)
                v.work_binding("A2", "D4")
                self.assertNotIn("A2", v.worknote.branches["A"].predecessor_refs)
                self.assertEqual(v.worknote.branches["A2"].progress_offers, 2)
                [prior] = json.loads(v.spoken.context_text(work_refs=("A2",)))
                self.assertEqual(json.loads(v.spoken.context_text(work_refs=("A", "A2"))), [prior])
                self.assertEqual(prior["recorded_sentence_prefix"], old.text if settled else "")
                self.assertEqual((old.heard, old.outcome, old.delegation_ids, old.work_tells), before)
                if not settled:
                    v.spoken.cut(1, 101)
                    v.spoken.cut_played(101, "Old A")
                    self.assertEqual(
                        json.loads(v.spoken.context_text(work_refs=("A2",)))[0]["recorded_sentence_prefix"], "Old A"
                    )
                v.worknote.progress("D4", "New A evidence.", attribution=Attribution("A2", "O4", 0))
                v.progress_due = (v.worknote.progress_key(v.worknote.branches["A2"]),)
                v.talker = SimpleNamespace(response_format=None)
                call = v._progress()
                v.talker = None
                expected = old.heard if not settled else old.text
                self.assertEqual(call.request.tail.count('"line_id": 1'), 1)
                self.assertIn('"recorded_sentence_prefix": ' + json.dumps(expected), call.request.tail)
                self.assertNotIn("request B", str(call.request))
