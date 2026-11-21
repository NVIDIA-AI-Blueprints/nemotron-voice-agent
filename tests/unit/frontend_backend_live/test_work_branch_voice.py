# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D103, D107
"""Branch-specific Voice composition/turnover using actual records, without a model or playback worker."""

import json
import unittest
from types import SimpleNamespace

from pipecat.processors.aggregators.llm_context import LLMContext

from examples.frontend_backend_live.delegations import ReadyResult
from examples.frontend_backend_live.models.decision import TalkerDecision
from examples.frontend_backend_live.ownership import OwnerView, WorkOwnership
from examples.frontend_backend_live.spoken import Line, LineKind
from examples.frontend_backend_live.voice import NamedNotice, Voice
from examples.shared.work_provenance import Attribution, Binding, Obligation
from tests.unit.frontend_helpers import CONFIG, PROMPTS, Port


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
        self.assertEqual([item["text"] for item in json.loads(v.spoken.context_text(work_refs=("A",)))], ["A result"])
        self.assertEqual(json.loads(v.spoken.context_text(work_refs=("B",)))[0]["recorded_sentence_prefix"], "fact B")
        self.assertEqual((a.heard, b.heard), ("fact A", "fact B"))
        v._done(line.items)
        self.assertEqual(v.worknote.entries["D3"].ready, 0)
        v.talker = SimpleNamespace(response_format=None)
        v.spoken.handed(Line(LineKind.STATUS, "Unplayed wording", ("D3",), work_refs=frozenset(("A",))))
        wire = v._relay_call(batch).request.messages([{"role": "user", "content": "request A and B together"}])
        self.assertEqual([message["role"] for message in wire], ["system", "user"])
        self.assertIn("A result", wire[1]["content"])
        self.assertNotIn("request A and B together", wire[1]["content"])
        self.assertNotIn("Unplayed wording", wire[1]["content"])

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

                text = ""
                if superseded:
                    v.worknote.progress("D3", "new A fact", attribution=Attribution("A", "O3", 1))
                    text = "old joint line"
                self.assertEqual(call.plan.finished(v, call, TalkerDecision("speak", text, {})), "")
                self.assertEqual([v.worknote.branches[branch].progress_offers for branch in ("A", "B")], [1, 1])
                self.assertEqual(bool(v.worknote.branches["B"].news), superseded)
                if not superseded:
                    self.assertFalse(v.worknote.branches["A"].news)

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
                self.assertEqual(json.loads(v.spoken.context_text(work_refs=("A2",)))[0]["text"], old.text)
                self.assertEqual(json.loads(v.spoken.context_text(work_refs=("A", "A2")))[0]["text"], old.text)
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
