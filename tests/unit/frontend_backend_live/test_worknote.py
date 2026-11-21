# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D103, D107
"""Evidence provenance and snapshot races of the frontend's combined progress note."""

import unittest

from examples.frontend_backend_live.prompts import load_prompts
from examples.frontend_backend_live.worknote import WorkNote


class WorkNoteTests(unittest.TestCase):
    def setUp(self):
        self.note = WorkNote(load_prompts().cascade, tracks=True, clock=lambda: 12.0, observe=lambda *args, **kw: None)

    def add(self, delegation_id, request):
        self.note.confirm(self.note.reserve(request), delegation_id)
        self.note.running(delegation_id)
        return self.note.entries[delegation_id]

    def test_requests_and_starting_work_do_not_create_progress_evidence(self):
        self.add("D1", "Make this requested change.")
        self.note.typed("D2")
        self.note.call_started("D1", "r1", "step", "apply_change")
        self.assertEqual(self.note.quiet_candidates(3), ())
        self.note.call_ended("D1", "r1", "step", "apply_change", "completed")
        self.assertEqual([entry.id for entry in self.note.quiet_candidates(3)], ["D1"])

    def test_one_batch_contains_only_backend_evidence_and_consumes_each_included_revision(self):
        self.add("D1", "Change private booking details.")
        self.add("D2", "Cancel the insurance charge.")
        self.note.progress("D1", "The reservation record was found; checking available dates.")
        self.note.progress("D2", "Reading the coverage terms.")
        sources, text = self.note.offer_progress(("D1", "D2"), 3)
        self.assertIn("reservation record was found", text)
        self.assertIn("Reading the coverage terms", text)
        self.assertNotIn("Change private booking details", text)
        self.assertNotIn("Cancel the insurance charge", text)
        self.assertIn("Change private booking details", self.note.line())
        self.assertTrue(self.note.progress_current(sources))
        self.note.progress_worded(sources)
        self.assertEqual(self.note.quiet_candidates(3), ())
        self.assertEqual([entry.progress_offers for entry in self.note.open()], [1, 1])

    def test_new_note_during_wording_is_never_consumed_as_old_news(self):
        entry = self.add("D1", "Change my booking.")
        self.note.progress("D1", "Checking available dates.")
        sources, _ = self.note.offer_progress(("D1",), 3)
        self.note.progress("D1", "Thursday is available; checking its fare.")
        self.assertFalse(self.note.progress_current(sources))
        self.note.progress_worded(sources)
        self.assertTrue(entry.news)
        self.assertEqual(entry.progress, "Thursday is available; checking its fare.")

    def test_ready_waiting_ended_reserved_and_capped_jobs_do_not_hide_an_eligible_sibling(self):
        for key in ("ready", "app", "ended", "cap", "eligible"):
            self.add(key, f"request {key}")
            self.note.progress(key, f"Checking {key}.")
        self.note.answered("ready")
        self.note.waits_for_app("app", True)
        self.note.ended("ended")
        self.note.entries["cap"].progress_offers = 3
        self.note.reserve("not sent")
        self.assertEqual([entry.id for entry in self.note.quiet_candidates(3)], ["eligible"])
        self.assertFalse(self.note.entries["ready"].news)

    def test_answer_or_handover_invalidates_the_offered_batch(self):
        for event in ("answer", "handover", "end"):
            with self.subTest(event=event):
                self.setUp()
                self.add("D1", "the first task")
                self.add("D2", "the second task")
                self.note.progress("D1", "Checking the first record.")
                sources, _ = self.note.offer_progress(("D1",), 3)
                if event == "answer":
                    self.note.answered("D1")
                elif event == "handover":
                    self.note.hand_over("D1", "D2")
                else:
                    self.note.ended("D1")
                self.assertFalse(self.note.progress_current(sources))

    def test_logical_retirement_keeps_original_published_ready_until_actual_wire_end(self):
        for reopen in (False, True):
            with self.subTest(reopen=reopen):
                self.setUp()
                entry = self.add("D1", "the original task")
                self.note.progress("D1", "A fact from the original response.")
                sources, _ = self.note.offer_progress(("D1",), 3)
                self.note.retire("D1")
                self.assertIs(self.note.entries["D1"], entry)
                self.assertFalse(entry.ended)
                self.assertEqual(self.note.open(), [])
                self.assertEqual(self.note.line(), self.note.prompts.no_work_in_progress)
                self.assertFalse(self.note.progress_current(sources))

                self.note.answered("D1", item_id="original-answer")
                self.assertEqual(self.note.open(), [entry])
                self.assertIn(self.note.prompts.answer_ready_label, self.note.line())
                if reopen:
                    self.note.reopen("D1")
                    self.assertFalse(entry.retired)
                    self.assertEqual(entry.ready, 1)
                    self.assertEqual(entry.request, self.note.prompts.typed_request_label)
                self.note.said("D1", item_id="original-answer")
                self.assertIs(self.note.entries["D1"], entry)
                self.assertEqual(self.note.open(), [entry] if reopen else [])
                self.note.ended("D1")
                self.assertNotIn("D1", self.note.entries)


class BranchWorkNoteTests(unittest.TestCase):
    def setUp(self):
        from examples.frontend_backend_live.ownership import OwnerView
        from examples.shared.work_provenance import Binding, Obligation
        from tests.unit.frontend_backend_live.test_work_branch_voice import shared_ownership

        self.Binding, self.Obligation, self.OwnerView = Binding, Obligation, OwnerView
        self.note = WorkNote(load_prompts().cascade, tracks=True, clock=lambda: 12.0, observe=lambda *a, **kw: None)
        self.ownership = shared_ownership(self.note)

    def obligation(self, ref, branch):
        return self.Obligation(ref, branch, "active", False, False, None, None, None)

    def attr(self, branch, revision=1, owner="O3"):
        from examples.shared.work_provenance import Attribution

        return Attribution(branch, owner, revision)

    def move_a(self):
        self.note.confirm(self.note.reserve("new A input"), "D4")
        self.note.running("D4")
        self.ownership.transports["O4"] = "D4"
        self.ownership.bindings["A"] = self.Binding("A", "O4", 2)
        self.note.sync_ownership(self.ownership)

    def test_move_invalidates_frozen_scope_and_old_closing_calls_without_stealing_sibling_news(self):
        (a, b), sources = self.offer_both_branches()
        self.move_a()
        self.assertEqual((sources[0].id, a.id, a.progress_offers), ("D3", "D4", 1))
        self.assertFalse(self.note.progress_current(sources))
        self.assertEqual((a.news, b.news), (False, True))
        self.note.call_ended("D3", "old-response", "old-call", "read", "completed", attribution=self.attr("A"))
        self.assertEqual((a.news, a.calls), (False, {}))

    def offer_both_branches(self):
        for branch in ("A", "B"):
            self.note.progress("D3", "fact " + branch, attribution=self.attr(branch))
        entries = self.note.quiet_candidates(3)
        self.assertEqual([entry.id for entry in entries], ["D3", "D3"])
        sources, text = self.note.offer_progress(tuple(self.note.progress_key(entry) for entry in entries), 3)
        self.assertEqual({source.branch_ref for source in sources}, {"A", "B"})
        self.assertIn("fact A", text)
        self.assertIn("fact B", text)
        self.assertEqual([entry.progress_offers for entry in entries], [1, 1])
        return entries, sources

    def test_reopening_transport_does_not_reactivate_retired_historical_sibling(self):
        self.note.retire("D3")
        self.note.reopen("D3")
        self.assertFalse(self.note.entries["D3"].retired)
        self.assertTrue(self.note.branches["A"].retired)
        self.assertTrue(self.note.branches["B"].retired)
        self.assertEqual(self.note.open(), [])
        self.note.progress("D3", "late old A fact", attribution=self.attr("A"))
        self.assertEqual(self.note.quiet_candidates(3), ())

    def test_old_publication_stays_with_its_owner_across_a_move_and_settles_once(self):
        old, current = self.attr("A"), self.attr("A", 2, "O4")
        a, b = self.note.branches["A"], self.note.branches["B"]
        self.note.progress("D3", "fact B", attribution=self.attr("B"))
        for _ in range(2):
            self.note.answered("D3", attribution=old, item_id="answer-A")
        observed = []
        self.note.on_settled = lambda: observed.append(a.ready)
        self.move_a()
        self.assertEqual((a.ready, self.note.entries["D3"].ready, self.note.entries["D4"].ready), (1, 1, 0))
        self.assertEqual(self.note.pending_publications(), {("D3", "answer-A")})
        self.assertTrue(b.news)
        self.assertNotIn(a, self.note.quiet_candidates(3))
        label = self.note.prompts.answer_ready_label
        self.assertNotIn(label, self.note._item(1, a))
        self.assertNotIn(label, self.note._item(1, b))
        self.assertIn("request A and B together", self.note._item(1, self.note.entries["D3"]))
        self.assertIn(self.note.entries["D3"], self.note.open())
        self.assertIn(b, self.note.open())
        self.assertEqual(self.note.line().count(label), 1)

        self.note.answered("D4", attribution=current, item_id="new-A")
        self.assertIn(label, self.note._item(1, a))
        self.assertEqual(self.note.line().count(label), 2)
        self.assertEqual(self.note.pending_publications(), {("D3", "answer-A"), ("D4", "new-A")})
        for _ in range(2):
            self.note.said("D3", attribution=old, item_id="answer-A")
        self.note.answered("D3", attribution=old, item_id="answer-A")
        self.assertEqual((self.note.entries["D3"].ready, a.ready), (0, 1))
        self.assertEqual(self.note.pending_publications(), {("D4", "new-A")})
        self.note.said("D4", attribution=current, item_id="new-A")
        self.assertEqual((a.ready, self.note.pending_publications(), observed), (0, set(), [1, 0]))

    def test_app_wait_is_only_the_named_branch(self):
        for branch in ("A", "B"):
            self.note.progress("D3", "fact " + branch, attribution=self.attr(branch))
        self.note.waits_for_app("D3", True, branch_refs=("A",))
        self.assertEqual([entry.branch_ref for entry in self.note.quiet_candidates(3)], ["B"])
        self.note.waits_for_app("D3", False, branch_refs=("A",))
        self.assertEqual({entry.branch_ref for entry in self.note.quiet_candidates(3)}, {"A", "B"})
        self.note.waits_for_app("D3", True)
        self.assertEqual(self.note.quiet_candidates(3), ())
        self.note.waits_for_app("D3", False)
        self.assertEqual({entry.branch_ref for entry in self.note.quiet_candidates(3)}, {"A", "B"})

    def test_reopen_real_transport_retains_old_ready_and_branch_facts(self):
        self.note.progress("D3", "still working B", attribution=self.attr("B"))
        self.note.answered("D3", attribution=self.attr("A"), item_id="old-A")
        old_branch = self.note.branches["B"]
        self.note.reopen("D3")
        self.assertEqual(self.note.entries["D3"].request, self.note.prompts.typed_request_label)
        self.assertEqual(self.note.entries["D3"].ready, 1)
        self.assertEqual(old_branch.progress, "still working B")
        self.note.said("D3", attribution=self.attr("A"), item_id="old-A")
        self.assertEqual(self.note.entries["D3"].ready, 0)

    def test_closed_owner_snapshots_cannot_recreate_parents_and_stay_inside_existing_window(self):
        from examples.frontend_backend_live.delegations import MAX_KEPT_ENDED
        from examples.frontend_backend_live.ownership import WorkOwnership

        note = WorkNote(load_prompts().cascade, tracks=True, clock=lambda: 12.0, observe=lambda *a, **kw: None)
        ownership = WorkOwnership()
        retained = []
        for number in range(MAX_KEPT_ENDED * 3):
            did, owner, branch = f"D{number}", f"O{number}", f"branch{number}"
            note.typed(did)
            ownership.transports[owner] = did
            ownership.bindings[branch] = self.Binding(branch, owner, 0)
            ownership.owners[owner] = self.OwnerView(0, (self.obligation("input" + str(number), branch),))
            note.sync_ownership(ownership)
            note.ended(did)
            retained = [*retained[-(MAX_KEPT_ENDED - 1) :], did]
            ownership.prune(retained)
            note.sync_ownership(ownership)
            self.assertEqual(note.entries, {})
            self.assertLessEqual(len(note.branches), MAX_KEPT_ENDED)
        self.assertEqual(note.open(), [])

    def test_new_owner_projection_waits_for_real_parent_and_invalidates_old_offer_early(self):
        self.note.progress("D3", "fact A", attribution=self.attr("A"))
        sources, _ = self.note.offer_progress((self.note.progress_key(self.note.branches["A"]),), 3)
        self.ownership.bindings["A"] = self.Binding("A", "not-yet-observed-owner", 2)
        self.note.sync_ownership(self.ownership)
        self.assertFalse(self.note.progress_current(sources))
        self.assertNotIn("not-yet-observed-owner", self.note.entries)
        self.ownership.transports["not-yet-observed-owner"] = "D4"
        self.note.sync_ownership(self.ownership)
        self.assertNotIn("D4", self.note.entries)
        self.assertEqual(self.note.branches["A"].id, "D3")
        self.note.typed("D4")
        self.note.sync_ownership(self.ownership)
        self.assertEqual(self.note.branches["A"].id, "D4")

    def test_single_early_branch_preserves_legacy_note_offer_and_whole_handover(self):
        from examples.frontend_backend_live.ownership import WorkOwnership

        def make():
            return WorkNote(load_prompts().cascade, tracks=True, clock=lambda: 12.0, observe=lambda *a, **kw: None)

        legacy, attributed = make(), make()
        ownership = WorkOwnership()
        ownership.transports["O1"] = "D1"
        ownership.bindings["A"] = self.Binding("A", "O1", 0)
        ownership.owners["O1"] = self.OwnerView(0, (self.obligation("one", "A"),))
        for note in (legacy, attributed):
            note.confirm(note.reserve("first request"), "D1")
            note.running("D1")
        attributed.sync_ownership(ownership)
        legacy.progress("D1", "verified fact")
        attributed.progress("D1", "verified fact", attribution=self.attr("A", 0, "O1"))
        self.assertEqual(attributed.line(), legacy.line())
        self.assertEqual(attributed.offer_progress(("D1",), 3)[1], legacy.offer_progress(("D1",), 3)[1])
        for note in (legacy, attributed):
            note.confirm(note.reserve("second request"), "D2")
            note.running("D2")
        legacy.hand_over("D1", "D2")
        ownership.transports["O2"] = "D2"
        ownership.bindings["A"] = self.Binding("A", "O2", 1)
        ownership.owners["O2"] = self.OwnerView(1, (self.obligation("two", "A"),))
        attributed.sync_ownership(ownership)
        attributed.hand_over("D1", "D2")
        self.assertEqual(attributed.line(), legacy.line())
        self.assertEqual(attributed.branches["A"].progress_offers, legacy.entries["D2"].progress_offers)
