# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D103, D107

"""Real reader ownership across partial moves, stale frames and late typed requests."""

import unittest
from copy import deepcopy
from dataclasses import replace

from nemotron_voice_agent.pipecat.examples.frontend_backend_live.ownership import WorkOwnership
from nemotron_voice_agent.pipecat.examples.frontend_backend_live.work_provenance import (
    Attribution,
    Binding,
    BranchSources,
    Obligation,
    OwnerSettlement,
    Provenance,
    encode_provenance,
)
from nemotron_voice_agent.pipecat.shared.delegation_contract import WORK_PROVENANCE_KEY
from tests.unit.pipecat.frontend_backend_live.test_delegations import (
    Reading,
    completed,
    created,
    failed,
    function_call,
    in_progress,
    mcp_call,
    mcp_started,
    message,
)


def edge(branch, *, owner="D3", state="active", target=None):
    return Obligation(
        f"{owner}-{branch}", branch, state, False, False, "answered" if state == "settled" else None, None, target
    )


def snapshot(
    owner="D3", response="resp3", *, revision=1, bindings=None, obligations=None, attribution=None, settlements=()
):
    return Provenance(
        "conversation",
        response,
        owner,
        revision,
        tuple(bindings if bindings is not None else [Binding("A", owner, 0), Binding("B", owner, 0)]),
        tuple(obligations if obligations is not None else [edge("A", owner=owner), edge("B", owner=owner)]),
        attribution,
        tuple(settlements),
    )


def tagged(event, provenance):
    return {**deepcopy(event), WORK_PROVENANCE_KEY: encode_provenance(provenance)}


def response_opened(proof):
    """Return the start of a response with its private ownership visible to the reader."""
    return tuple(tagged(event(proof.response_id), proof) for event in (created, in_progress))


def observe_header(registry, proof, transport=None):
    """Apply only a header, as can happen when a listener joins an in-progress response."""
    response = proof.response_id
    registry.apply(transport or proof.owner_ref, tagged(in_progress(response), proof), response)


class OwnershipTests(unittest.TestCase):
    def test_a_first_settlement_with_pending_work_is_rejected_atomically(self):
        registry = WorkOwnership()
        invalid = snapshot(settlements=[OwnerSettlement("old", 3, "answered", (edge("A"),))])
        before = deepcopy(registry.__dict__)
        with self.assertRaisesRegex(ValueError, "pending obligations"):
            observe_header(registry, invalid, "D3")
        self.assertEqual(registry.__dict__, before)

    def test_known_disposition_survives_later_plain_headers_and_old_settlements(self):
        for settlement_first in (True, False):
            with self.subTest(settlement_first=settlement_first):
                registry = WorkOwnership()
                rows = (edge("A", state="transferred", target="D4"),)
                plain = snapshot(revision=4, obligations=rows)
                settled = snapshot(
                    "D4",
                    "resp4",
                    bindings=[Binding("A", "D4", 1)],
                    obligations=[edge("A", owner="D4")],
                    settlements=[OwnerSettlement("D3", 3, "handed_over", rows)],
                )
                events = [("D3", plain), ("D4", settled)]
                for did, value in reversed(events) if settlement_first else events:
                    observe_header(registry, value, did)
                self.assertEqual(registry.owners["D3"].disposition, "handed_over")
                self.assertEqual(registry.owners["D3"].revision, 4)
                invalid = (
                    replace(settled, owner_settlements=(OwnerSettlement("D3", 5, "failed", rows),)),
                    snapshot(revision=5, obligations=[edge("A")]),
                )
                for update in invalid:
                    before = deepcopy(registry.__dict__)
                    with self.assertRaises(ValueError):
                        observe_header(registry, update)
                    self.assertEqual(registry.__dict__, before)

    def test_private_core_owner_is_not_a_public_transport_alias(self):
        registry = WorkOwnership()
        value = snapshot("core_new", "fresh", bindings=[Binding("A", "core_new", 0)], obligations=[edge("A")])
        observe_header(registry, value, "old_transport")
        self.assertEqual(registry.current_transport("A"), "old_transport")
        self.assertNotEqual(value.owner_ref, "old_transport")
        self.assertIsNone(registry.transport("unobserved_owner"))

    def test_contradictory_revision_is_wholly_rejected_without_observing_a_new_frame(self):
        registry = WorkOwnership()
        value = snapshot()
        observe_header(registry, value, "D3")
        before = deepcopy(registry.__dict__)
        invalid = snapshot("D4", "resp4", bindings=[Binding("A", "D4", 0)], obligations=[edge("A", owner="D4")])
        with self.assertRaisesRegex(ValueError, "Contradictory work binding"):
            observe_header(registry, invalid, "D4")
        self.assertEqual(registry.__dict__, before)

    def test_foreign_scope_wrong_writer_and_item_spoof_do_not_mutate_registry(self):
        registry = WorkOwnership()
        original = snapshot()
        observe_header(registry, original, "D3")
        for invalid in (
            replace(original, conversation_ref="foreign"),
            replace(original, response_id="another_frame"),
            replace(original, item_attribution=Attribution("A", "other_owner", 0)),
            replace(original, item_attribution=Attribution("unknown_branch", "D3", 0)),
            replace(original, request_sources=(BranchSources("unknown_branch", ("D1",)),)),
        ):
            with self.subTest(invalid=invalid):
                before = deepcopy(registry.__dict__)
                with self.assertRaises(ValueError):
                    registry.apply("D3", tagged(message("Fact.", item_id="item"), invalid), "resp3")
                self.assertEqual(registry.__dict__, before)
        business = message('{"_nva_work": "business text"}')
        with self.assertRaisesRegex(ValueError, "lacks its publication scope"):
            registry.apply("D3", business, "resp3")
        self.assertIsNone(WorkOwnership().apply("legacy", business, "legacy-response"))

    def test_closed_owners_use_the_same_window_on_distinct_and_shared_transports(self):
        for shared in (False, True):
            with self.subTest(shared=shared):
                registry = WorkOwnership()
                for number in range(100):
                    owner, response, branch = f"D{number}", f"r{number}", f"branch{number}"
                    value = snapshot(
                        owner,
                        response,
                        bindings=[Binding(branch, owner, 0)],
                        obligations=[edge(branch, owner=owner, state="settled")],
                    )
                    transport = "one_transport" if shared else owner
                    value = replace(value, request_sources=(BranchSources(branch, (owner,)),))
                    registry.apply(transport, tagged(completed(response), value), response)
                    retained = {transport} if shared else {f"D{i}" for i in range(max(0, number - 31), number + 1)}
                    registry.prune(retained, max_closed=32)
                retained_sizes = (registry.owners, registry.frames, registry.bindings, registry.request_sources)
                self.assertEqual(tuple(map(len, retained_sizes)), (32, 32, 32, 32))

    def test_duplicate_empty_late_responses_do_not_retain_every_closed_frame(self):
        registry = WorkOwnership()
        for number in range(100):
            response = f"late{number}"
            value = snapshot("ended", response, bindings=[], obligations=[])
            registry.apply("transport", tagged(completed(response), value), response)
            registry.prune({"transport"}, max_closed=32)
        self.assertEqual(len(registry.owners), 1)
        self.assertEqual(len(registry.frames), 32)


def later_shared_owner():
    """Return the next owner after both branches move from D3 to D5."""
    branches = ("A", "B")
    moved = tuple(edge(branch, state="transferred", target="D5") for branch in branches)
    return snapshot(
        "D5",
        "resp5",
        bindings=[Binding(branch, "D5", 1) for branch in branches],
        obligations=[edge(branch, owner="D5") for branch in branches],
        settlements=[OwnerSettlement("D3", 3, "handed_over", moved)],
    )


class ReaderOwnershipTests(unittest.IsolatedAsyncioTestCase):
    def test_shared_response_requires_item_scope_even_with_an_owner_snapshot(self):
        reading = Reading("D3")
        original = snapshot()
        reading.hand("D3", *response_opened(original))
        before = deepcopy(reading.reader.ownership.__dict__)

        reading.hand(
            "D3",
            tagged(message("Unknown Work result.", item_id="unscoped"), replace(original, owner_revision=2)),
        )

        self.assertEqual(reading.calls("ready"), [])
        self.assertEqual(reading.reader.ownership.__dict__, before)

        notice = message("Pause requested.", item_id="control")
        wrong_kind = function_call("book_dinner")
        wrong_kind["item"]["id"] = "control"
        reading.hand(
            "D3",
            tagged({**notice, "type": "response.output_item.added"}, replace(original, source_item=True)),
            wrong_kind,
            notice,
        )
        self.assertEqual(reading.calls("ready"), [("D3", "Pause requested.")])
        self.assertEqual(reading.reader.open["D3"].response.calls, 0)

        result = message("Dinner result.", item_id="dinner")
        reading.hand(
            "D3",
            tagged(
                {**result, "type": "response.output_item.added"},
                replace(original, item_attribution=Attribution("A", "D3", 0)),
            ),
            tagged(result, original),
        )
        self.assertEqual(reading.calls("ready"), [("D3", "Pause requested."), ("D3", "Dinner result.")])
        self.assertEqual(reading.reader.worknote.branches["A"].ready, 1)

    def test_shared_request_and_later_correction_follow_exact_core_sources(self):
        reading = Reading("D1", "D2", "D3")
        reader, note = reading.reader, reading.reader.worknote
        shared = replace(
            snapshot(),
            request_sources=(BranchSources("A", ("D1", "D3")), BranchSources("B", ("D2", "D3"))),
        )
        reading.hand("D3", *response_opened(shared))
        correction = "Move the flight to 8 p.m."
        reservation = note.reserve(correction)
        updated = replace(
            snapshot("D4", "resp4", bindings=[Binding("A", "D4", 1)], obligations=[edge("A", owner="D4")]),
            request_sources=(BranchSources("A", ("D1", "D3", "D4")),),
        )
        reading.hand("D4", *response_opened(updated))
        self.assertEqual(note.branches["A"].request, "request D1 / request D3")
        note.confirm(reservation, "D4")
        # Older replay can report the old binding and a shorter source list after the correction.
        reader.restore_work_provenance("D3", tagged(in_progress("resp3"), shared))

        carrier = replace(
            snapshot("D5", "resp5", bindings=[Binding("A", "D5", 2)], obligations=[edge("A", owner="D5")]),
            request_sources=updated.request_sources,
        )
        note.confirm(note.reserve("Playback carrier only."), "D5")
        reading.hand("D5", *response_opened(carrier))
        context = {row["branch_ref"]: row for row in note.newer_requests(("D5",), ("A",)) if row["branch_ref"]}
        self.assertEqual(context["A"]["delegation_id"], "D5")
        self.assertEqual(context["A"]["relationship"], "same_branch")
        self.assertEqual(context["A"]["caller_requests"], ["request D1", "request D3", correction])
        self.assertEqual(context["B"]["caller_requests"], ["request D2", "request D3"])

    def test_acceptance_replaces_only_the_branches_exact_early_request_placeholder(self):
        reading = Reading("D2")
        note = reading.reader.worknote
        request = "Check whether a flight leaves after 8 p.m."
        reservation = note.reserve(request)
        first = snapshot("D1", "resp1", bindings=[Binding("A", "D1", 0)], obligations=[edge("A", owner="D1")])
        other = snapshot("D2", "resp2", bindings=[Binding("B", "D2", 0)], obligations=[edge("B", owner="D2")])
        reading.hand("D1", *response_opened(first))
        reading.hand("D2", *response_opened(other))
        branch = note.branches["A"]
        self.assertEqual(branch.request, note.prompts.typed_request_label)
        note.progress("D1", "Flight times retrieved.", attribution=Attribution("A", "D1", 0))
        offered, _ = note.prepare_progress((branch.progress_key,), 3)
        note.progress_offered(offered)
        self.assertTrue(note.progress_current(offered))
        other_request = note.branches["B"].request

        note.confirm(reservation, "D1")

        context = {row["branch_ref"]: row["caller_requests"] for row in note.newer_requests(("D1",), ("A",))}
        self.assertEqual(context, {"A": [request], "B": [other_request]})
        self.assertIn(request, note.line())
        self.assertIs(note.branches["A"], branch)
        self.assertFalse(note.progress_current(offered))
        self.assertEqual(branch.progress_offers, 1)
        self.assertEqual(branch.progress, "Flight times retrieved.")

    def test_unheard_original_publication_remains_guarded_after_later_closed_sources(self):
        reading = Reading("D3", "D5")
        original = snapshot()
        old_item = replace(original, item_attribution=Attribution("A", "D3", 0))
        reading.hand(
            "D3",
            *response_opened(original),
            tagged(message("Earlier A.", item_id="oldA"), old_item),
        )
        newer = later_shared_owner()
        reading.hand("D5", *response_opened(newer))
        reading.hand("D3", tagged(completed("resp3"), original))
        for number in range(40):
            did, response = f"X{number}", f"xresp{number}"
            value = snapshot(did, response, bindings=[], obligations=[])
            reading.hand(
                did,
                *response_opened(value),
                tagged(completed(response), value),
            )
        reader, note = reading.reader, reading.reader.worknote
        self.assertNotIn("D3", reader.ended)
        self.assertIn("D3", reader.ownership.owners)
        self.assertIn("resp3", reader.ownership.frames)
        reading.hand(
            "D3",
            *response_opened(original),
            tagged(message("Earlier A.", item_id="oldA"), old_item),
            tagged(completed("resp3"), original),
        )
        self.assertEqual(reading.calls("ready"), [("D3", "Earlier A.")])
        self.assertEqual(note.entries["D3"].ready, 1)
        self.assertEqual(reader.ownership.current_transport("A"), "D5")
        note.said("D3", attribution=Attribution("A", "D3", 0), item_id="oldA")
        self.assertNotIn("D3", reader.ownership.owners)
        self.assertNotIn("resp3", reader.ownership.frames)

    def test_already_published_a_and_b_keep_independent_readiness_across_partial_move(self):
        reading = Reading("D3", "D4")
        reader, note = reading.reader, reading.reader.worknote
        initial = snapshot()
        reading.hand("D3", *response_opened(initial))
        moved = snapshot("D4", "resp4", bindings=[Binding("A", "D4", 1)], obligations=[edge("A", owner="D4")])
        reading.hand("D4", *response_opened(moved))
        reading.hand(
            "D3",
            tagged(
                message("Earlier A progress.", phase="commentary", item_id="progress-a"),
                replace(initial, item_attribution=Attribution("A", "D3", 0)),
            ),
            tagged(
                message("Current B progress.", phase="commentary", item_id="progress-b"),
                replace(initial, item_attribution=Attribution("B", "D3", 0)),
            ),
        )
        reading.hand(
            "D3",
            tagged(
                message("Earlier A result.", item_id="a"), replace(initial, item_attribution=Attribution("A", "D3", 0))
            ),
        )
        reading.hand(
            "D3",
            tagged(message("B result.", item_id="b"), replace(initial, item_attribution=Attribution("B", "D3", 0))),
        )
        self.assertEqual(note.entries["D3"].ready, 2)
        note.said("D3", attribution=Attribution("A", "D3", 0), item_id="a")
        self.assertEqual(note.entries["D3"].ready, 1)
        self.assertEqual(note.branches["B"].ready, 1)
        self.assertEqual(reader.ownership.current_transport("A"), "D4")
        self.assertEqual(reader.ownership.current_transport("B"), "D3")
        self.assertEqual(reader.ownership.attribution("resp3", "a"), Attribution("A", "D3", 0))
        self.assertEqual(reading.calls("ready"), [("D3", "Earlier A result."), ("D3", "B result.")])
        self.assertEqual(reading.calls("progress"), [("D3", "Current B progress.")])

    def test_old_closing_app_frame_does_not_close_its_newer_response(self):
        for restored in (False, True):
            with self.subTest(restored=restored):
                reading = Reading("D3")
                original = snapshot()
                reading.hand("D3", *response_opened(original))
                if not restored:
                    app = replace(original, item_attribution=Attribution("A", "D3", 0))
                    backend = replace(original, item_attribution=Attribution("B", "D3", 0))
                    reading.hand(
                        "D3",
                        tagged(function_call("read", "call-a"), app),
                        tagged(mcp_started("lookup", "read", "old-backend"), backend),
                    )
                reading.hand("D3", *response_opened(replace(original, response_id="resp4")))
                terminal = tagged(completed("resp3"), original)
                if restored:
                    reading.reader.restore_terminal("D3", terminal)
                else:
                    result = tagged(mcp_call("lookup", "read", "old-backend"), backend)
                    reading.hand("D3", result, terminal)
                self.assertEqual(reading.reader.open["D3"].response.id, "resp4")
                self.assertFalse(reading.reader.open["D3"].waits)
                self.assertTrue(reading.reader.responses["resp3"].ended)
                self.assertFalse(reading.reader.responses["resp4"].ended)
                self.assertEqual(reading.notices(), [])

    def test_single_branch_completion_distinguishes_transferred_work_from_a_missing_answer(self):
        settled = edge("A", owner="D1", state="settled")
        transferred = edge("A", owner="D1", state="transferred", target="D2")
        cases = {
            "empty": (),
            "still active": (replace(settled, state="active"),),
            "delivery pending": (replace(settled, state="pending", terminal_outcome=None, delivery_pending=True),),
            "publication pending": (replace(settled, state="active", publication_pending=True),),
            "failed": (replace(settled, terminal_outcome="failed"),),
            "cancelled": (replace(settled, terminal_outcome="cancelled"),),
            "transfer delivery pending": (replace(transferred, delivery_pending=True),),
            "transfer publication pending": (replace(transferred, publication_pending=True),),
            "transferred": (transferred,),
        }
        opening = snapshot("D1", "r1", bindings=[Binding("A", "D1", 0)], obligations=[edge("A", owner="D1")])
        for case, obligations in cases.items():
            for replay in (False, True):
                with self.subTest(case=case, replay=replay):
                    reading = Reading("D1")
                    self.addCleanup(reading.reader.close)
                    bindings = (Binding("A", "D2", 1),) if case.startswith("transfer") else opening.bindings
                    value = replace(opening, owner_revision=2, bindings=bindings, obligations=obligations)
                    terminal = tagged(completed("r1"), value)
                    reading.hand("D1", *response_opened(opening))
                    if replay:
                        reading.reader.restore_work_provenance("D1", terminal)
                        reading.reader.restore_terminal("D1", terminal)
                    else:
                        reading.hand("D1", terminal)
                    self.assertEqual(reading.endings(), {"D1": "handed_over" if case == "transferred" else "late"})
                    self.assertEqual(len(reading.notices()), int(not replay and case != "transferred"))
                    self.assertEqual(reading.calls("ready"), [])
                    self.assertEqual(reading.calls("handed_over"), [])

    def test_a_failed_control_does_not_report_a_failure_on_transferred_work(self):
        first = snapshot("D1", "r1", bindings=[Binding("A", "D1", 0)], obligations=[edge("A", owner="D1")])
        transfer = replace(
            first,
            owner_revision=2,
            bindings=(Binding("A", "D2", 1),),
            obligations=(edge("A", owner="D1", state="transferred", target="D2"),),
        )
        second = snapshot("D2", "r2", bindings=[Binding("A", "D2", 1), Binding("B", "D2", 0)])
        finished = replace(
            second,
            owner_revision=2,
            obligations=(
                replace(edge("A", owner="D2", state="settled"), terminal_outcome="failed"),
                edge("B", owner="D2", state="settled"),
            ),
        )
        for reported in (False, True):
            with self.subTest(reported=reported):
                reading = Reading("D1", "D2")
                self.addCleanup(reading.reader.close)
                reading.hand("D1", *response_opened(first))
                if reported:
                    reading.hand("D1", tagged(message("Stop unconfirmed."), replace(first, source_item=True)))
                reading.hand("D1", tagged(failed("r1"), transfer))
                reading.hand("D2", *response_opened(second), tagged(failed("r2"), finished))

                self.assertEqual(reading.endings(), {"D1": "failed", "D2": "failed"})
                expected = ["D2"] if reported else ["D1", "D2"]
                self.assertEqual([source for source, _ in reading.calls("notice")], expected)

    def test_an_old_or_duplicate_created_event_cannot_reopen_a_finished_source(self):
        reading = Reading("D3")
        settled = snapshot(obligations=[edge("A", state="settled"), edge("B", state="settled")])
        reading.hand(
            "D3",
            *response_opened(settled),
            tagged(completed("resp3"), settled),
        )
        reading.hand("D3", tagged(created("resp3"), settled))
        self.assertNotIn("D3", reading.reader.open)
        self.assertIn("D3", reading.reader.ended)
