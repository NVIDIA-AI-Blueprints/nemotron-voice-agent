# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D100, D101, D102, D103, D107

"""Schema-derived rules (tau3-voice-domain-specific-changes-genericization.md sections 4.5-4.9, 5).

Invented schemas prove the rules are not keyed on benchmark names; the snapshot over the five tau2
domains' schemas is the reviewed list of where each behaviour fires. Regenerate the snapshot only
after reviewing the change::

    PYTHONPATH=src uv run python tests/unit/frontend_delegation_hermes/core/test_fdh_schema_rules.py --write
"""

from __future__ import annotations

import json
import re
import sys
import unittest
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from _fdh_voice_fakes import (  # noqa: E402
    PACKAGE,
    PROFILES,
    ROOT,
    DelegationHarness,
    profile_config,
    tau2_session_update,
)

from examples.frontend_delegation_hermes.core.config import load_delegation_config  # noqa: E402
from examples.frontend_delegation_hermes.core.engine.spelling_hold import COMPLETE, SpellingHoldPredicate  # noqa: E402
from examples.frontend_delegation_hermes.core.normalization.arguments import (  # noqa: E402
    ArgumentNormalizer,
    ToolArgumentSettings,
)
from examples.frontend_delegation_hermes.core.normalization.schema_rules import (  # noqa: E402
    build_schema_rules,
    first_example,
    is_read_tool,
    tools_sha256,
)
from examples.frontend_delegation_hermes.core.normalization.transcript import TranscriptSettings  # noqa: E402
from examples.frontend_delegation_hermes.core.text.messages import ToolCall  # noqa: E402
from examples.frontend_delegation_hermes.core.tools.result_hints import ResultHints  # noqa: E402

SNAPSHOT = HERE / "fixtures" / "schema_rules_snapshot.json"
TAU2_DOMAINS = ("airline", "retail", "telecom", "banking_knowledge", "mock")
EVAL = load_delegation_config(PROFILES / "realtime_eval_control.yaml")  # without the geval fixes
EVAL_ARGS = EVAL.voice.normalization.tool_arguments


def _tool(name: str, description: str = "", **arguments: tuple[str, str]) -> dict[str, Any]:
    """An invented Realtime tool; ``arguments`` maps a name to ``(type, description)``."""
    return {
        "type": "function",
        "name": name,
        "description": description,
        "parameters": {
            "type": "object",
            "properties": {arg: {"type": kind, "description": text} for arg, (kind, text) in arguments.items()},
            "required": list(arguments),
        },
    }


#: An invented product: nothing here appears in tau2.
INVENTED = [
    _tool("fetch_member", "Look up a member by their member number.",
          member_id=("string", "The member ID, such as 'ab_cd_12'.")),
    _tool("show_ticket", "Show one support ticket.", ticket_id=("string", "The ticket, e.g. 'Q7XK2P'.")),
    _tool("get_parcel", "Get a parcel.", parcel_id=("string", "The parcel ID, such as '$P00000'.")),
    _tool("get_badge", "Get a badge.", badge_id=("string", "The badge ID, such as 'Ab3dE9'.")),
    _tool("get_seat", "Get a seat.", seat_id=("string", "The seat, such as 'row-12'."),
          state=("string", "The state, such as 'CA'."), zip=("string", "A zip code, such as '12345'.")),
    _tool("reassign_member", "Reassign a member to a team.",
          member_id=("string", "The member ID, such as 'ab_cd_12'."), team_ids=("array", "Each such as 'T0000'.")),
    _tool("phone_directory_lookup_x", "Find a contact by phone.", phone=("string", "The phone number.")),
    _tool("lookup_by_phone", "Search contacts.", contact_phone=("string", "The phone number to search for.")),
    _tool("store_phone", "Store a phone number.", member_phone=("string", "The phone number.")),
]  # fmt: skip


def _by_tool(rules) -> dict[str, Any]:
    return {f"{rule.tool}.{rule.argument}": rule for rule in rules}


def _normalizer(tools: list[dict[str, Any]]) -> ArgumentNormalizer:
    schema = build_schema_rules(tools)
    settings = ToolArgumentSettings(enabled=True, rules=schema.rules, schema_rules=True)
    return ArgumentNormalizer(
        settings,
        transcript=TranscriptSettings(enabled=True),
        invalid_template="invalid {label} {value}",
        already_failed_template="failed {value}",
    )


def snapshot() -> dict[str, Any]:
    out = {}
    for domain in TAU2_DOMAINS:
        tools = tau2_session_update(domain)["session"]["tools"]
        out[domain] = build_schema_rules(tools).snapshot()
    return out


# -- invented schemas ----------------------------------------------------------------------------


class InventedSchemaTests(unittest.TestCase):
    def test_read_tools_by_name_or_description_and_unknown_is_write(self) -> None:
        self.assertTrue(is_read_tool("fetch_member", "Look up a member."))
        self.assertFalse(is_read_tool("show_ticket", "Show a ticket."))
        self.assertTrue(is_read_tool("get_anything", ""))
        self.assertTrue(is_read_tool("thing", "  Search things."))
        self.assertFalse(is_read_tool("reassign_member", "Reassign a member."))
        self.assertFalse(is_read_tool("transfer", ""))

    def test_examples_are_single_quoted_tokens(self) -> None:
        self.assertEqual(first_example("The ID, such as 'ab_cd_12'."), "ab_cd_12")
        self.assertEqual(first_example('Code (e.g., "Q7XK2P")'), "Q7XK2P")
        self.assertEqual(first_example("An expression such as '2 + 2'."), "")
        self.assertEqual(first_example("The ID of the user"), "")

    def test_rules_only_for_read_tools_with_an_identifier_example(self) -> None:
        rules = _by_tool(build_schema_rules(INVENTED).rules)
        self.assertEqual(
            sorted(rules), ["fetch_member.member_id", "get_badge.badge_id", "get_parcel.parcel_id", "get_seat.seat_id"]
        )  # show_ticket is not a read tool by name or description; reassign_member is a write tool
        member = rules["fetch_member.member_id"]
        self.assertEqual(
            (member.label, member.format_hint, member.collapse_separators, member.case, member.strip),
            ("member ID", "name_name_digits", "_", "lower", " .-"),
        )
        self.assertEqual(rules["get_badge.badge_id"].case, "keep")  # mixed-case example: case kept
        self.assertEqual(rules["get_parcel.parcel_id"].restore_prefix, "$")
        self.assertEqual(rules["get_seat.seat_id"].strip, " .")  # '-' is in the example, so it is kept

    def test_canonicalization_is_idempotent_and_never_adds_letters_or_digits(self) -> None:
        normalizer = _normalizer(INVENTED)
        rules = _by_tool(normalizer.settings.rules)
        cases = {
            ("fetch_member.member_id", "Ab_Cd_12"): "ab_cd_12",
            ("fetch_member.member_id", "ab cd_12"): "abcd_12",
            ("fetch_member.member_id", "Ab underscore Cd underscore one two"): "ab_cd_12",
            ("get_parcel.parcel_id", "P12345"): "$P12345",  # the symbol is restored; the letter is the user's
            ("get_parcel.parcel_id", "p 1234 5"): "$P12345",
            ("get_badge.badge_id", "aB3De9"): "aB3De9",  # mixed case: kept
        }
        for (key, value), expected in cases.items():
            with self.subTest(key=key, value=value):
                rule = rules[key]
                once = normalizer.canonical(rule, value)
                self.assertEqual(once, expected)
                self.assertEqual(normalizer.canonical(rule, once), once)
                if "underscore" not in value:  # spoken digit words become digits; nothing else is added
                    self.assertEqual(
                        sorted(c.lower() for c in once if c.isalnum()), sorted(c.lower() for c in value if c.isalnum())
                    )
        for key, well_formed in (("fetch_member.member_id", "zz_yy_9"), ("get_parcel.parcel_id", "$Q1"),
                                 ("get_badge.badge_id", "Zz9"), ("get_seat.seat_id", "row-3")):  # fmt: skip
            with self.subTest(well_formed=well_formed):
                self.assertEqual(normalizer.canonical(rules[key], well_formed), well_formed)

    def test_only_clearly_incomplete_values_are_answered_locally(self) -> None:
        normalizer = _normalizer(INVENTED)

        def local(tool: str, arg: str, value: str) -> bool:
            call = ToolCall(id="c1", name=tool, arguments_json=json.dumps({arg: value}))
            return bool(normalizer.screen([call], frozenset()).local)

        for value in ("ab_cd_1", "ab_cd_123456", "abc_defgh_7", "a_b_c_9"):  # any lengths, more segments
            self.assertFalse(local("fetch_member", "member_id", value), value)
        for value in ("ab_cd", "ab_", "ab__12", "ab_cd_xy", "abcd12"):  # missing segment, empty, no final digits
            self.assertTrue(local("fetch_member", "member_id", value), value)
        self.assertFalse(local("get_parcel", "parcel_id", "P1"))  # never over a missing prefix or length
        self.assertFalse(local("get_badge", "badge_id", "x"))
        self.assertTrue(local("get_badge", "badge_id", ""))

    def test_write_tools_are_never_rewritten(self) -> None:
        normalizer = _normalizer(INVENTED)
        arguments = {"member_id": "AB CD 12", "team_ids": ["t 1"]}
        call = ToolCall(id="c1", name="reassign_member", arguments_json=json.dumps(arguments))
        screening = normalizer.screen([call], frozenset())
        self.assertEqual((screening.rewrites, screening.local), ((), ()))
        self.assertEqual(json.loads(screening.sent[0].arguments_json), arguments)

    def test_hold_patterns_from_identifier_arguments_only(self) -> None:
        patterns = build_schema_rules(INVENTED).complete_patterns
        self.assertEqual(
            set(patterns),
            {r"^[A-Z0-9]{6}$", r"^\$?[A-Z]\d{5}$", r"^[A-Za-z0-9]{6}$", r"^[A-Z]\d{4}$"},
        )  # 'ab_cd_12' and 'row-12' have a name-like part: their length is not fixed, so no pattern
        self.assertFalse(any(re.fullmatch(p, value) for p in patterns for value in ("CA", "12345")))

    def test_phone_arguments_of_read_tools(self) -> None:
        self.assertEqual(
            build_schema_rules(INVENTED).phone_arguments,
            ("phone_directory_lookup_x.phone", "lookup_by_phone.contact_phone"),
        )

    def test_tools_fingerprint_ignores_order_and_sees_schema_changes(self) -> None:
        self.assertEqual(tools_sha256(INVENTED), tools_sha256(list(reversed(INVENTED))))
        changed = [dict(INVENTED[0], description="Other."), *INVENTED[1:]]
        self.assertNotEqual(tools_sha256(INVENTED), tools_sha256(changed))
        self.assertRegex(tools_sha256(INVENTED), r"^[0-9a-f]{16}$")


# -- the five tau2 domains -----------------------------------------------------------------------


class Tau2SnapshotTests(unittest.TestCase):
    def test_snapshot_matches_the_reviewed_file(self) -> None:
        self.assertEqual(snapshot(), json.loads(SNAPSHOT.read_text(encoding="utf-8")))

    def test_rules_match_the_design_tables(self) -> None:  # section 4.5
        expected = {
            "airline": {"get_user_details.user_id", "get_reservation_details.reservation_id"},
            "retail": {"get_user_details.user_id", "get_order_details.order_id",
                       "get_product_details.product_id", "get_item_details.item_id"},
            "telecom": set(), "banking_knowledge": set(), "mock": set(),
        }  # fmt: skip
        snap = snapshot()
        for domain, keys in expected.items():
            with self.subTest(domain=domain):
                self.assertEqual({f"{r['tool']}.{r['argument']}" for r in snap[domain]["rules"]}, keys)
        rules = {(d, f"{r['tool']}.{r['argument']}"): r for d in snap for r in snap[d]["rules"]}
        user = rules["airline", "get_user_details.user_id"]
        self.assertEqual((user["case"], user["collapse_separators"]), ("lower", "_"))
        self.assertEqual(rules["airline", "get_reservation_details.reservation_id"]["case"], "upper")
        order = rules["retail", "get_order_details.order_id"]
        self.assertEqual((order["restore_prefix"], order["case"], order["strip"]), ("#", "upper", " .-"))
        self.assertEqual(rules["retail", "get_item_details.item_id"]["keep_pattern"], r"^\d+$")

    def test_hold_patterns_match_the_design_table(self) -> None:  # section 4.9
        snap = snapshot()
        self.assertIn(r"^[A-Z0-9]{6}$", snap["airline"]["complete_patterns"])
        self.assertIn(r"^#?[A-Z]\d{7}$", snap["retail"]["complete_patterns"])
        self.assertIn(r"^\d{10}$", snap["retail"]["complete_patterns"])
        for domain in ("telecom", "banking_knowledge", "mock"):
            self.assertEqual(snap[domain]["complete_patterns"], [])
        patterns = [p for d in snap for p in snap[d]["complete_patterns"]]
        excluded = ("CA", "SFO", "JFK", "John", "Doe", "12345", "2 + 2", "San Francisco", "123 Main St",
                    "2024-01-01", "basic_economy", "one_way")  # fmt: skip
        for value in excluded:
            with self.subTest(value=value):
                self.assertFalse(any(re.fullmatch(p, value, re.IGNORECASE) for p in patterns))
        kept = ("ZFA04Y", "8JX2WO", "#W0000000", "W0000000", "6086499569", "1008292230")
        for value in kept:
            with self.subTest(value=value):
                self.assertTrue(any(re.fullmatch(p, value, re.IGNORECASE) for p in patterns))
        # Name-like identifiers vary in length: no digit count from one example ever marks them complete.
        for value in ("sara_doe_496", "zq_xv_123", "zq_xv_1234", "gift_card_0000000", "credit_card_7815826"):
            with self.subTest(value=value):
                self.assertFalse(any(re.fullmatch(p, value, re.IGNORECASE) for p in patterns))

    def test_no_rule_on_any_write_tool(self) -> None:  # G6
        writes = re.compile(r"^(book_|cancel_|modify_|update_|exchange_|return_|send_|log_|change_|create_|"
                            r"suspend_|resume_|enable_|disable_|refuel_|transfer_|give_|unlock_|call_)")  # fmt: skip
        for domain in TAU2_DOMAINS:
            tools = tau2_session_update(domain)["session"]["tools"]
            schema = build_schema_rules(tools)
            with self.subTest(domain=domain):
                self.assertTrue(all(is_read_tool(r.tool, "") or r.tool in schema.read_tools for r in schema.rules))
                self.assertFalse([r.tool for r in schema.rules if writes.match(r.tool)])
                self.assertFalse([t for t in schema.read_tools if writes.match(t)])

    def test_values_returned_by_tools_pass_unchanged(self) -> None:  # G6 idempotence
        normalizer = _normalizer(
            tau2_session_update("retail")["session"]["tools"] + tau2_session_update("airline")["session"]["tools"]
        )
        rules = _by_tool(normalizer.settings.rules)
        returned = (
            ("get_user_details.user_id", "sara_doe_496"),
            ("get_order_details.order_id", "#W0000000"),
            ("get_item_details.item_id", "6086499569"),
            ("get_reservation_details.reservation_id", "8JX2WO"),
            ("get_user_details.user_id", "zq_xv_1"),
            ("get_order_details.order_id", "#W12"),
        )
        for key, value in returned:
            with self.subTest(value=value):
                self.assertEqual(normalizer.canonical(rules[key], value), value)
        self.assertEqual(normalizer.canonical(rules["get_order_details.order_id"], "w 1234567"), "#W1234567")

    def test_phone_arguments(self) -> None:
        snap = snapshot()
        self.assertEqual(snap["telecom"]["phone_arguments"], ["get_customer_by_phone.phone_number"])
        self.assertEqual(snap["banking_knowledge"]["phone_arguments"], [])  # log_verification is a write tool


# -- result hints and the retry guard over tau2's not-found texts ---------------------------------

#: Every "not found" / "no records found" text of the five domains (tools.py, banking db_query.py).
HINT = {
    "Error: User zq_xv_1 not found",  # airline, mock
    "Error: User not found",  # retail (all three user lookups)
    "Error: Customer with ID C0 not found",
    "Error: Customer with phone number 3034971268 not found",
    "Error: Customer C0 not found",
    "Error: User 'U0' not found.",
    "Error: User with ID 'U0' not found.",
    "No records found in 'users'.",
}
NO_HINT = {
    "Error: Flight QV101 not found", "Error: Flight QV101 not found on date 2024-01-01",
    "Error: Payment method not found", "Error: Payment method credit_card_1 not found",
    "Error: Reservation XB7TP2 not found",
    "Error: Item not found", "Error: Item 1 not found", "Error: New item 1 not found or available",
    "Error: Order not found", "Error: Product not found", "Error: Variant not found", "Error: Some item not found",
    "Error: Bill with ID B0 not found", "Error: Bill B0 not found for customer C0",
    "Error: Device with ID D0 not found", "Error: Line with ID L0 not found",
    "Error: Line with phone number 3034971268 not found", "Error: Line L0 not found for customer C0",
    "Error: Overdue bill B0 not found", "Error: Plan with ID P0 not found", "Error: Plan not found for this line",
    "Error: Account 'A0' not found.", "Error: Checking account 'A0' not found.",
    "Error: Source account 'A0' not found.", "Error: Destination account 'A0' not found.",
    "Error: Credit card account 'A0' not found.", "Error: Debit card 'D0' not found.",
    "Error: Transaction 'T0' not found.", "No records found in 'transactions'.",
    "Error: Task T0 not found",
    '{"user_id": "zq_xv_1", "name": "Q"}', "Error: upstream timeout",
}  # fmt: skip


class ResultTextTests(unittest.TestCase):
    def test_hint_pattern_names_a_user_or_customer(self) -> None:  # section 4.7
        pattern = re.compile(EVAL_ARGS.result_hints.failure_pattern)
        for text in sorted(HINT):
            with self.subTest(text=text):
                self.assertIsNotNone(pattern.search(text))
        for text in sorted(NO_HINT):
            with self.subTest(text=text):
                self.assertIsNone(pattern.search(text))
        self.assertIsNotNone(pattern.search("Error: Database 'users' not found."))  # the known false positive

    def test_retry_guard_pattern_covers_every_not_found(self) -> None:  # section 4.6
        self.assertEqual(EVAL_ARGS.retry_guard.scope, "all")
        pattern = re.compile(EVAL_ARGS.retry_guard.permanent_failure_pattern)
        for text in sorted((HINT | NO_HINT) - {'{"user_id": "zq_xv_1", "name": "Q"}', "Error: upstream timeout"}):
            if text.startswith("No records found"):
                continue  # an empty query result, not a failure text
            with self.subTest(text=text):
                self.assertIsNotNone(pattern.search(text))
        self.assertIsNone(pattern.search("Error: upstream timeout"))
        self.assertIsNotNone(pattern.search("ERROR: NO SUCH widget"))

    def test_auto_hints_watch_read_tools_and_count_person_arguments_only(self) -> None:
        schema = build_schema_rules(INVENTED)
        hints = ResultHints(
            EVAL_ARGS.result_hints, message="[I3]", escalate_message="[I4]", read_tools=schema.read_tools
        )
        self.assertIsNone(hints.on_output("reassign_member", "Error: Member m1 not found"))  # write tool
        self.assertIsNone(hints.on_output("get_seat", "Error: Seat not found"))  # no person entity
        hints.on_local_invalid("get_seat", "seat_id")  # not a person argument: not counted
        first = hints.on_output("fetch_member", "Error: Member m1 not found")
        assert first is not None
        self.assertEqual((first.message_key, first.miss), ("lookup_not_found_hint", 1))
        hints.on_local_invalid("fetch_member", "member_id")  # a person argument: counted
        third = hints.on_output("get_parcel", "Error: Parcel for customer c1 not found")
        assert third is not None
        self.assertEqual((third.message_key, third.miss), ("lookup_not_found_hint_words", 3))


# -- the spelling hold ---------------------------------------------------------------------------


class SchemaHoldTests(unittest.TestCase):
    def _predicate(self, domain: str) -> SpellingHoldPredicate:
        patterns = build_schema_rules(tau2_session_update(domain)["session"]["tools"]).complete_patterns
        return SpellingHoldPredicate(TranscriptSettings(enabled=True), schema_patterns=patterns)

    def test_a_complete_code_is_not_held_and_a_spelled_name_is(self) -> None:
        airline = self._predicate("airline")
        self.assertFalse(airline.check("it's X B seven T P two").hold)  # complete, any case
        self.assertTrue(airline.check("it's X B seven T").hold)
        self.assertTrue(airline.check("my name is C A").hold)  # 'CA' examples never become patterns
        self.assertTrue(self._predicate("telecom").check("it's X B seven T P two").hold)  # no patterns there

    def test_a_spelled_user_id_is_never_cut_short_by_the_example_digit_count(self) -> None:
        for domain in ("airline", "retail"):
            predicate = self._predicate(domain)
            with self.subTest(domain=domain):
                # 'sara_doe_496' has three digits; a user ID that has more must not end the turn after three.
                check = predicate.check("my user ID is zq underscore xv underscore one two three")
                self.assertTrue(check.hold)
                self.assertEqual(check.value, "zq_xv_123")
                for digits in ("one two", "one two three", "one two three four", "one two three four five"):
                    check = predicate.check(f"zq underscore xv underscore {digits}")
                    self.assertNotEqual(check.reason, COMPLETE, digits)  # no pattern rules a user ID complete


# -- the per-session wiring ----------------------------------------------------------------------


class SessionWiringTests(unittest.IsolatedAsyncioTestCase):
    async def _session(self, domain: str) -> DelegationHarness:
        harness = DelegationHarness(profile_config("realtime_eval_control.yaml"), transcripts=["I need help"])
        await harness.start(tau2_session_update(domain))
        await harness.speak()
        await harness.wait_sent("delegate")
        harness.link.deliver("state", state="WORKING", epoch=1, run_id="run_1")
        return harness

    async def _call(self, harness: DelegationHarness, index: int, name: str, arguments: dict, output: str):
        call_id = f"c{index}"
        results = len(harness.link.of_type("tool.result"))
        harness.link.deliver(
            "tool.call", call_id=call_id, epoch=1, run_id="run_1", name=name, arguments=json.dumps(arguments)
        )
        done = await harness.wait_for("response.function_call_arguments.done", count=index)
        await harness.send(
            {"type": "conversation.item.create",
             "item": {"type": "function_call_output", "call_id": call_id, "output": output}}
        )  # fmt: skip
        await harness.send({"type": "response.create"})
        result = await harness.wait_sent("tool.result", count=results + 1)
        return json.loads(done["arguments"]), result["output"]

    def test_before_the_tools_arrive_there_are_no_schema_rules(self) -> None:
        harness = DelegationHarness(profile_config("realtime_eval_control.yaml"))
        parts = harness.runtime.parts(None)  # type: ignore[arg-type]
        assert parts.argument_normalizer is not None and parts.rebuild is not None
        self.assertEqual(parts.argument_normalizer.settings.rules, ())

    async def test_retail_rules_apply_to_reads_only_and_are_logged_with_the_tools_hash(self) -> None:
        harness = await self._session("retail")
        tools = tau2_session_update("retail")["session"]["tools"]
        sent, _ = await self._call(harness, 1, "get_order_details", {"order_id": "w 1234567"}, "Error: Order not found")
        self.assertEqual(sent, {"order_id": "#W1234567"})
        write = {"order_id": "w 7654321", "reason": "ordered by mistake"}
        sent, _ = await self._call(harness, 2, "cancel_pending_order", write, "Error: Order not found")
        self.assertEqual(sent, write)  # G6
        (rules,) = harness.logged("session_rules")
        self.assertEqual(rules["session_tools_sha256"], tools_sha256(tools))
        self.assertIn("get_order_details", [r["tool"] for r in rules["rules"]])
        self.assertEqual(harness.logged("backend_configured")[0]["session_tools_sha256"], tools_sha256(tools))
        start = harness.logged("fdh_session_start")[0]
        self.assertTrue(start["features"]["schema_rules"])
        await harness.close()

    async def test_telecom_customer_miss_gets_a_hint_and_a_bill_miss_does_not(self) -> None:
        harness = await self._session("telecom")
        miss = "Error: Customer with phone number 3034971268 not found"
        _, output = await self._call(harness, 1, "get_customer_by_phone", {"phone_number": "3034971268"}, miss)
        self.assertTrue(output.startswith(miss + "\n\n[Not found. At least one value"))
        bill = "Error: Bill B0 not found for customer C0"
        _, output = await self._call(harness, 2, "get_bills_for_customer", {"customer_id": "C0", "limit": 1}, bill)
        self.assertEqual(output, bill)
        await harness.close()


# -- section 5.2: the submission configuration names no tau2 tool or domain ------------------------

SUBMISSION_FILES = (
    PACKAGE / "config" / "profiles" / "realtime_eval.yaml",
    PACKAGE / "config" / "delegation_agent.yaml",
    PACKAGE / "config" / "voice" / "realtime_eval.yaml",
    PACKAGE / "config" / "voice" / "base.yaml",
    PACKAGE / "config" / "gateway.eval.yaml",
    PACKAGE / "config" / "gateway.yaml",
    PACKAGE / "config" / "prompts.yaml",
    PACKAGE / "config" / "prompts.backend.yaml",
    ROOT / "src" / "examples" / "frontend_delegation_hermes" / "config" / "voice_agent.yaml",
    ROOT / "src" / "examples" / "frontend_delegation_hermes" / "config" / "prompts.voice.yaml",
)
TAU2_NAMES = re.compile(
    r"get_user_details|find_user_id|get_customer_by_phone|get_reservation_details|telecom|airline|retail"
)
BENCHMARK_VALUES = re.compile(r"ROSSI|IFOYYZ|NQNU5R|K1NW8N|mia_kim|mia_fancy|555-123|\\d\{4\}")
#: The one exception, disclosed word for word with the submission (analysis/2026-10-11_fdh-telecom-gpt55-deep):
#: the telecom detection rule (gateway.yaml, gateway.eval.yaml) and the key of the telecom note
#: (prompts.backend.yaml). Comments may name domains; they are never sent to a model.
DISCLOSED_DOMAIN_LINES = {
    "gateway.yaml": "telecom: {tools_any: [get_customer_by_phone]}",
    "gateway.eval.yaml": "telecom: {tools_any: [get_customer_by_phone]}",
    "prompts.backend.yaml": "telecom: |-",
}


def _code_lines(text: str) -> list[str]:
    """The stripped lines of a YAML file without full-line and trailing comments."""
    lines = []
    for line in text.splitlines():
        if line.lstrip().startswith("#"):
            continue
        lines.append(re.sub(r"\s+#\s.*$", "", line).strip())
    return lines


def _tau2_hits(path: Path) -> list[str]:
    """Lines naming a tau2 tool or domain, except the disclosed telecom lines."""
    allowed = DISCLOSED_DOMAIN_LINES.get(path.name)
    return [
        line for line in _code_lines(path.read_text(encoding="utf-8")) if TAU2_NAMES.search(line) and line != allowed
    ]


class SubmissionConfigTests(unittest.TestCase):
    def test_no_tau2_tool_or_domain_name_and_no_benchmark_value(self) -> None:
        for path in SUBMISSION_FILES:
            text = path.read_text(encoding="utf-8")
            with self.subTest(path=path.name):
                self.assertEqual(_tau2_hits(path), [])
                self.assertEqual(BENCHMARK_VALUES.findall(text), [])

    def test_the_disclosed_telecom_lines_are_present_once(self) -> None:
        for name, line in DISCLOSED_DOMAIN_LINES.items():
            with self.subTest(file=name):
                self.assertEqual(_code_lines((PACKAGE / "config" / name).read_text(encoding="utf-8")).count(line), 1)

    def test_other_hits_are_historical_tau3_files(self) -> None:
        config = PACKAGE / "config"
        hits = {path.relative_to(config).as_posix() for path in config.rglob("*.yaml") if _tau2_hits(path)}
        self.assertTrue(all(Path(hit).name.startswith("tau3") for hit in hits), hits)


def main() -> None:
    if "--write" in sys.argv:
        SNAPSHOT.write_text(json.dumps(snapshot(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"wrote {SNAPSHOT}")
        return
    unittest.main()


if __name__ == "__main__":
    main()
