# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102

"""Private host provenance survives JSON while business data and historical attribution stay distinct."""

import json
import unittest
from copy import deepcopy
from dataclasses import FrozenInstanceError

from examples.shared.work_provenance import (
    WORK_PROVENANCE_KEY,
    Attribution,
    Binding,
    Obligation,
    OwnerSettlement,
    Provenance,
    decode_provenance,
    encode_provenance,
    strip_provenance,
)


def pending(ref="edge_A", branch="branch_A"):
    """An actual placement edge, independent from stable output branch identity."""
    return Obligation(ref, branch, "active", False, False, None, None, None)


class ProvenanceTests(unittest.TestCase):
    def setUp(self):
        self.value = Provenance(
            "conversation",
            "response_3",
            "D3",
            3,
            (Binding("branch_A", "D4", 1), Binding("branch_B", "D3", 0)),
            (
                Obligation("edge_A", "branch_A", "transferred", False, False, None, None, "D4"),
                pending("edge_B", "branch_B"),
            ),
            Attribution("branch_A", "D3", 0),
            (OwnerSettlement("D1", 1, "handed_over", ()),),
        )

    def test_json_roundtrip_preserves_old_item_owner_separate_from_new_binding(self):
        event = {"type": "response.output_item.done", WORK_PROVENANCE_KEY: encode_provenance(self.value)}
        decoded = decode_provenance(json.loads(json.dumps(event)))
        self.assertEqual(decoded, self.value)
        self.assertEqual((decoded.item_attribution.owner_ref, decoded.bindings[0].owner_ref), ("D3", "D4"))
        with self.assertRaises(FrozenInstanceError):
            decoded.owner_ref = "D5"

    def test_public_strip_is_top_level_only_and_never_mutates_the_journal(self):
        business = {WORK_PROVENANCE_KEY: {"version": 1, "claim": "not host authority"}}
        event = {"type": "response.output_item.done", "item": {"output": json.dumps(business)}}
        original = deepcopy(event)
        self.assertIsNone(decode_provenance(event))
        event[WORK_PROVENANCE_KEY] = encode_provenance(self.value)
        self.assertEqual(strip_provenance(event), original)
        self.assertIn(WORK_PROVENANCE_KEY, event)
        self.assertEqual(json.loads(event["item"]["output"]), business)

    def test_strict_primitives_and_unknown_fields_cannot_be_ownership_facts(self):
        raw = encode_provenance(self.value)
        mutations = [
            ("version", True),
            ("version", 1.0),
            ("version", 2),
            ("owner_revision", True),
            ("owner_revision", 1.0),
            ("owner_revision", -1),
            ("owner_ref", ""),
            ("response_id", None),
            ("bindings", {}),
            ("unknown", "foreign"),
        ]
        for key, value in mutations:
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                decode_provenance({WORK_PROVENANCE_KEY: {**raw, key: value}})
        for flag in ("delivery_pending", "publication_pending"):
            for value in (0, 1, "false", None):
                altered = deepcopy(raw)
                altered["obligations"][0][flag] = value
                with self.subTest(flag=flag, value=value), self.assertRaises(ValueError):
                    decode_provenance({WORK_PROVENANCE_KEY: altered})

    def test_distinct_placement_edges_can_share_a_stable_output_branch(self):
        value = Provenance(
            "conversation",
            "response",
            "canonical_Core_owner",
            0,
            (Binding("branch_A", "canonical_Core_owner", 0),),
            (pending("source_one"), pending("source_two")),
        )
        event = {WORK_PROVENANCE_KEY: encode_provenance(value)}
        self.assertEqual(len(decode_provenance(event).obligations), 2)
        # Transport identity is a separate consumer namespace, not a structural equality assertion.
        self.assertEqual(decode_provenance(event).owner_ref, "canonical_Core_owner")

    def test_duplicate_identities_and_false_settlement_are_rejected(self):
        raw = encode_provenance(self.value)
        for field in ("bindings", "obligations", "owner_settlements"):
            altered = deepcopy(raw)
            altered[field].append(deepcopy(altered[field][0]))
            with self.subTest(field=field), self.assertRaises(ValueError):
                decode_provenance({WORK_PROVENANCE_KEY: altered})
        for changes in (
            {"state": "settled", "terminal_outcome": "answered", "publication_pending": True},
            {"state": "settled", "terminal_outcome": None},
            {"state": "transferred", "to_owner_ref": None},
            {"stop_outcome": "confirmed"},
        ):
            altered = deepcopy(raw)
            altered["obligations"][1].update(changes)
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                decode_provenance({WORK_PROVENANCE_KEY: altered})
