# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

# ruff: noqa: D101, D102

"""Tests for the Frontend/Backend schema-driven identifier normalization."""

from __future__ import annotations

import json
import unittest
from typing import Any

from examples.frontend_backend_agent.src.normalization import (
    EMPTY_SCHEMA_RULES,
    ToolCall,
    build_schema_rules,
    canonical_json,
    canonicalize,
    ends_mid_spelling,
    is_incomplete,
    join_across_turns,
    matches_complete_pattern,
    normalize_transcript,
    other_phone_form,
    screen_call,
    spelled_runs,
    ten_digits,
    tool_kind,
    tools_sha256,
)


def _tool(name: str, description: str, properties: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "function",
        "name": name,
        "description": description,
        "parameters": {"type": "object", "properties": properties, "required": list(properties)},
    }


def _string(description: str, **extra: Any) -> dict[str, Any]:
    return {"type": "string", "description": description, **extra}


TOOLS: list[dict[str, Any]] = [
    _tool(
        "lookup_member",
        "Look up a member profile.",
        {
            "member_id": _string("The member ID, such as 'jordan_lee_82'.", title="Member ID"),
            "phone_number": _string("The member's phone number."),
        },
    ),
    _tool("get_order", "Get one order.", {"order_id": _string("The order ID, like '#Q4821937'.")}),
    _tool(
        "find_ticket",
        "Find a support ticket.",
        {"ticket_id": _string('The ticket ID, e.g. "XB7TP2".'), "note": _string("Free text.")},
    ),
    _tool(
        "search_members",
        "Search members by name.",
        {"first_name": _string("A first name such as 'Quentin'."), "region_code": _string("Such as 'QUENTIN'.")},
    ),
    _tool(
        "list_tickets",
        "List tickets.",
        {"ticket_ids": {"type": "array", "items": {"type": "string", "description": "e.g. 'MN4RS8'"}}},
    ),
    _tool("update_member", "Update a member's details.", {"member_id": _string("Such as 'jordan_lee_82'.")}),
    _tool("check_in_member", "Check a member in.", {"member_id": _string("Such as 'jordan_lee_82'.")}),
]


class ToolKindTests(unittest.TestCase):
    def test_read_needs_prefix_verb_and_no_mutating_token(self) -> None:
        self.assertEqual(tool_kind("get_order_items", "Get the items in an order."), "read")
        self.assertEqual(tool_kind("lookup_member", "  look up a member"), "read")
        self.assertEqual(tool_kind("search_members", "Search members."), "read")

    def test_everything_else_is_a_write(self) -> None:
        self.assertEqual(tool_kind("check_in_member", "Check a member in."), "write")
        self.assertEqual(tool_kind("lookup_member", "Updates the member record."), "write")
        self.assertEqual(tool_kind("fetch_x", "Get x."), "write")
        self.assertEqual(tool_kind("get_order", "Getaway order."), "write")
        self.assertEqual(tool_kind("get_cancel_status", "Get the status."), "write")
        self.assertEqual(tool_kind("send_receipt", "Send a receipt."), "write")


class SchemaRulesTests(unittest.TestCase):
    def setUp(self) -> None:
        self.rules = build_schema_rules(TOOLS)

    def test_read_tools(self) -> None:
        self.assertEqual(
            self.rules.read_tools,
            frozenset({"lookup_member", "get_order", "find_ticket", "search_members", "list_tickets"}),
        )

    def test_lower_underscore_example(self) -> None:
        rule = self.rules.rule_for("lookup_member", "member_id")
        assert rule is not None
        self.assertEqual((rule.separator, rule.case, rule.prefix), ("_", "lower", ""))
        self.assertEqual(rule.max_length, 0)
        self.assertEqual(canonicalize(rule, "Jordan underscore Lee underscore eight two"), "jordan_lee_82")
        self.assertEqual(canonicalize(rule, "jordan_lee_eight_two"), "jordan_lee_82")

    def test_hash_prefixed_example(self) -> None:
        rule = self.rules.rule_for("get_order", "order_id")
        assert rule is not None
        self.assertEqual((rule.separator, rule.case, rule.prefix, rule.min_length), (None, "upper", "#", 8))
        self.assertEqual(canonicalize(rule, "hash q four eight two one nine three seven"), "#Q4821937")
        self.assertEqual(canonicalize(rule, "Q4821937"), "#Q4821937")

    def test_upper_case_code_example(self) -> None:
        rule = self.rules.rule_for("find_ticket", "ticket_id")
        assert rule is not None
        self.assertEqual((rule.separator, rule.case, rule.min_length, rule.max_length), (None, "upper", 6, 6))
        self.assertEqual(canonicalize(rule, "x b seven t p two"), "XB7TP2")
        self.assertEqual(canonicalize(rule, "X, B, double seven, P, two"), "XB77P2")

    def test_no_rule_without_identifier_example(self) -> None:
        self.assertIsNone(self.rules.rule_for("find_ticket", "note"))
        self.assertIsNone(self.rules.rule_for("search_members", "first_name"))
        self.assertIsNone(self.rules.rule_for("lookup_member", "phone_number"))

    def test_write_tools_get_no_rules(self) -> None:
        self.assertIsNone(self.rules.rule_for("update_member", "member_id"))
        self.assertIsNone(self.rules.rule_for("check_in_member", "member_id"))

    def test_non_identifier_examples_are_skipped(self) -> None:
        for example in ("Quentin", "2024-05-01", "555-123-4567", "a@b.co", "12.50", "two words", "ab"):
            tools = [_tool("get_x", "Get x.", {"x_id": _string(f"such as '{example}'")})]
            built = build_schema_rules(tools)
            self.assertIsNone(built.rule_for("get_x", "x_id"), example)
            self.assertEqual(built.complete_patterns, (), example)

    def test_complete_patterns_only_from_fixed_identifier_arguments(self) -> None:
        sources = sorted(pattern.pattern for pattern in self.rules.complete_patterns)
        # order_id, plus one shared shape for ticket_id and ticket_ids; never the name-like member_id
        # or the non-id region_code.
        self.assertEqual(len(sources), 2)
        self.assertTrue(any(pattern.fullmatch("#Q4821937") for pattern in self.rules.complete_patterns))
        self.assertTrue(any(pattern.fullmatch("Q4821937") for pattern in self.rules.complete_patterns))
        self.assertTrue(any(pattern.fullmatch("MN4RS8") for pattern in self.rules.complete_patterns))
        self.assertFalse(any(pattern.fullmatch("jordan_lee_82") for pattern in self.rules.complete_patterns))
        self.assertFalse(any(pattern.fullmatch("QUENTIN") for pattern in self.rules.complete_patterns))

    def test_phone_arguments_and_labels(self) -> None:
        self.assertEqual(dict(self.rules.phone_arguments), {"lookup_member": ("phone_number",)})
        self.assertEqual(self.rules.labels[("lookup_member", "member_id")], "member id")
        self.assertEqual(self.rules.labels[("lookup_member", "phone_number")], "phone number")
        titled = build_schema_rules([_tool("get_x", "Get x.", {"x_ref": _string("", title="  Account  Code ")})])
        self.assertEqual(titled.labels[("get_x", "x_ref")], "account code")

    def test_snapshot_is_json_and_carries_no_example_values(self) -> None:
        dumped = json.dumps(self.rules.snapshot())
        self.assertEqual(self.rules.snapshot()["counts"]["identifier_rules"], 4)
        for example in ("jordan_lee_82", "Q4821937", "XB7TP2", "MN4RS8"):
            self.assertNotIn(example, dumped)

    def test_malformed_schemas_never_raise(self) -> None:
        malformed: list[Any] = [
            None,
            "get_x",
            {"name": 3},
            {"name": ""},
            {"name": "get_a", "description": None, "parameters": "bad"},
            {"name": "get_b", "description": "Get b.", "parameters": {"properties": ["x"]}},
            {"name": "get_c", "description": "Get c.", "parameters": {"properties": {"c_id": 5, 7: {}}}},
            {"name": "get_d", "description": "Get d.", "parameters": {"properties": {"d_id": {"type": "string"}}}},
            {"name": "get_e", "description": "Get e.", "parameters": {"properties": {"e_id": {"description": 9}}}},
            {"function": {"name": "get_f", "description": "Get f.", "parameters": None}},
        ]
        built = build_schema_rules(malformed)
        self.assertEqual(built.identifier_rules, {})
        self.assertIn("get_f", built.read_tools)
        self.assertEqual(build_schema_rules([]).snapshot(), EMPTY_SCHEMA_RULES.snapshot())

    def test_tools_sha256_is_stable_and_order_sensitive(self) -> None:
        digest = tools_sha256(TOOLS)
        self.assertEqual(len(digest), 64)
        self.assertEqual(digest, tools_sha256(json.loads(json.dumps(TOOLS))))
        self.assertNotEqual(digest, tools_sha256(list(reversed(TOOLS))))


class ScreenCallTests(unittest.TestCase):
    def setUp(self) -> None:
        self.rules = build_schema_rules(TOOLS)

    def test_read_identifier_is_canonicalized(self) -> None:
        result = screen_call(
            self.rules, ToolCall("lookup_member", {"member_id": "Jordan underscore Lee underscore eight two"})
        )
        self.assertEqual(result.kind, "read")
        self.assertEqual(result.arguments, {"member_id": "jordan_lee_82"})
        self.assertEqual(result.changed, ("member_id",))
        self.assertEqual(result.incomplete, ())

    def test_write_arguments_are_never_rewritten(self) -> None:
        for name in ("update_member", "check_in_member", "unknown_tool"):
            arguments = {"member_id": "Jordan Lee eight two", "extra": {"nested": [1, 2]}}
            result = screen_call(self.rules, ToolCall(name, arguments))
            self.assertEqual(result.kind, "write")
            self.assertEqual(result.arguments, arguments)
            self.assertIsNot(result.arguments["extra"], arguments["extra"])
            self.assertEqual((result.changed, result.incomplete), ((), ()))

    def test_value_matching_keep_pattern_is_never_changed(self) -> None:
        for call in (
            ToolCall("lookup_member", {"member_id": "kim_ng_7"}),
            ToolCall("get_order", {"order_id": "#Q4821937"}),
            ToolCall("find_ticket", {"ticket_id": "XB7TP2"}),
        ):
            result = screen_call(self.rules, call)
            self.assertEqual(result.arguments, dict(call.arguments))
            self.assertEqual((result.changed, result.incomplete), ((), ()))

    def test_incomplete_value_is_flagged_and_left_alone(self) -> None:
        result = screen_call(self.rules, ToolCall("get_order", {"order_id": "q 4 8 2"}))
        self.assertEqual(result.arguments, {"order_id": "q 4 8 2"})
        self.assertEqual(result.incomplete, ("order_id",))
        snake = screen_call(self.rules, ToolCall("lookup_member", {"member_id": "jordan underscore lee"}))
        self.assertEqual(snake.incomplete, ("member_id",))

    def test_unfixable_value_is_neither_changed_nor_incomplete(self) -> None:
        result = screen_call(self.rules, ToolCall("lookup_member", {"member_id": "jordan lee 82"}))
        self.assertEqual(result.arguments, {"member_id": "jordan lee 82"})
        self.assertEqual((result.changed, result.incomplete), ((), ()))

    def test_non_string_and_unruled_arguments_are_untouched(self) -> None:
        call = ToolCall("find_ticket", {"ticket_id": 42, "note": "x b seven t p two"})
        result = screen_call(self.rules, call)
        self.assertEqual(result.arguments, dict(call.arguments))
        self.assertEqual(result.changed, ())

    def test_screen_is_idempotent(self) -> None:
        values = ["x b seven t p two", "q 4 8 2", "Q4821937", "Jordan_Lee_82", "jordan lee", "", "x"]
        for name, argument in (("find_ticket", "ticket_id"), ("get_order", "order_id"), ("lookup_member", "member_id")):
            for value in values:
                once = screen_call(self.rules, ToolCall(name, {argument: value}))
                twice = screen_call(self.rules, ToolCall(name, once.arguments))
                self.assertEqual(twice.arguments, once.arguments, value)
                self.assertEqual(twice.changed, ())

    def test_canonicalize_is_idempotent_and_incomplete_is_conservative(self) -> None:
        rule = self.rules.rule_for("find_ticket", "ticket_id")
        assert rule is not None
        for value in ("x b seven t p two", "t w o", "X-B-7", "triple x", "S as in Sam b 7", "double", "hash 4"):
            once = canonicalize(rule, value)
            self.assertEqual(canonicalize(rule, once), once, value)
        self.assertTrue(is_incomplete(rule, "x b seven t"))
        self.assertTrue(is_incomplete(rule, ""))
        self.assertFalse(is_incomplete(rule, "XB7TP2Q"))

    def test_empty_rules_treat_every_call_as_a_write(self) -> None:
        result = screen_call(EMPTY_SCHEMA_RULES, ToolCall("get_order", {"order_id": "q 4"}))
        self.assertEqual((result.kind, result.arguments), ("write", {"order_id": "q 4"}))

    def test_canonical_json(self) -> None:
        self.assertEqual(canonical_json({"b": "é", "a": [1, 2]}), '{"a":[1,2],"b":"é"}')
        with self.assertRaises(ValueError):
            canonical_json({"a": float("nan")})


class TranscriptTests(unittest.TestCase):
    def test_spelled_code_is_joined(self) -> None:
        self.assertEqual(normalize_transcript("my code is x b seven t p two."), "my code is XB7TP2.")
        self.assertEqual(normalize_transcript("Q, U, E, N, T, I, N"), "QUENTIN")
        self.assertEqual(normalize_transcript("S as in Sam, four, double eight"), "S488")
        self.assertEqual(normalize_transcript("hash q four eight two"), "#Q482")

    def test_snake_run_is_lower_case(self) -> None:
        text = "it's Jordan underscore Lee underscore eighty two, thanks"
        self.assertEqual(normalize_transcript(text), "it's jordan_lee_82, thanks")
        self.assertEqual(spelled_runs(text), [(5, text.index(","), "jordan_lee_82")])

    def test_ordinary_speech_is_untouched(self) -> None:
        for text in (
            "I need a car for two days",
            "I a",
            "a i a",
            "oh I see",
            "ten twenty five works",
            "the pound is weak",
            "five people",
        ):
            self.assertEqual(normalize_transcript(text), text)
            self.assertEqual(spelled_runs(text), [])

    def test_normalize_is_idempotent(self) -> None:
        for text in (
            "my code is x b seven t p two",
            "jordan underscore lee underscore eight two and X B 7",
            "one two three, a b c",
            "1 2 3 hash 4 5 6",
            "four oh seven, uh, nine",
            "X uh B 7 T",
        ):
            once = normalize_transcript(text)
            self.assertEqual(normalize_transcript(once), once, text)

    def test_join_across_turns(self) -> None:
        self.assertEqual(join_across_turns("my code is X B 7", "T P 2 please"), "XB7TP2")
        self.assertEqual(join_across_turns("it's jordan underscore", "lee underscore eight two"), "jordan_lee_82")
        self.assertIsNone(join_across_turns("my code is X B 7. Thanks", "T P 2"))
        self.assertIsNone(join_across_turns("X B 7", "thanks a lot"))
        self.assertIsNone(join_across_turns("", "X B 7"))

    def test_ends_mid_spelling(self) -> None:
        self.assertTrue(ends_mid_spelling("my code is x b seven t"))
        self.assertTrue(ends_mid_spelling("jordan underscore lee underscore"))
        self.assertFalse(ends_mid_spelling("x t"))
        self.assertFalse(ends_mid_spelling("I want plan A"))
        self.assertFalse(ends_mid_spelling("x b seven t please"))
        self.assertFalse(ends_mid_spelling(""))

    def test_matches_complete_pattern(self) -> None:
        patterns = build_schema_rules(TOOLS).complete_patterns
        self.assertTrue(matches_complete_pattern("it is x b seven t p two", patterns))
        self.assertTrue(matches_complete_pattern("q four eight two one nine three seven.", patterns))
        self.assertFalse(matches_complete_pattern("it is x b seven", patterns))
        self.assertFalse(matches_complete_pattern("it is x b seven t p two", ()))


class PhoneTests(unittest.TestCase):
    def test_ten_digits(self) -> None:
        self.assertEqual(ten_digits("(555) 123-4567"), "5551234567")
        self.assertEqual(ten_digits("555.123.4567"), "5551234567")
        self.assertIsNone(ten_digits("1-555-123-4567"))
        self.assertIsNone(ten_digits("+1 555 123 4567"))
        self.assertIsNone(ten_digits("555-1234"))
        self.assertIsNone(ten_digits("555x123x4567"))
        self.assertIsNone(ten_digits(5551234567))

    def test_other_phone_form(self) -> None:
        self.assertEqual(other_phone_form("555-123-4567"), "5551234567")
        self.assertEqual(other_phone_form("5551234567"), "555-123-4567")
        self.assertEqual(other_phone_form("(555) 123 4567"), "555-123-4567")
        self.assertIsNone(other_phone_form("15551234567"))


if __name__ == "__main__":
    unittest.main()
