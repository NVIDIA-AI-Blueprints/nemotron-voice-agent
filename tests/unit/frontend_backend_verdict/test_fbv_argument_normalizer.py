# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D100, D101, D102, D103, D107

"""The tool-argument hook: canonical values, pattern checks, local answers, retry-guard keys."""

from __future__ import annotations

import json
import unittest
from dataclasses import replace

from examples.frontend_backend_verdict.text.messages import ToolCall, canonical_json
from examples.frontend_backend_verdict.voice.normalization.arguments import (
    REASON_ALREADY_FAILED,
    REASON_INVALID,
    ArgumentNormalizer,
    ArgumentRule,
    RetryGuardSettings,
    ToolArgumentSettings,
    render_message,
)
from examples.frontend_backend_verdict.voice.normalization.transcript import TranscriptSettings

RULE = ArgumentRule(
    tool="get_user_details",
    argument="user_id",
    label="user ID",
    format_hint="firstname_lastname_1234",
    spoken_form=True,
    strip=" .-",
    collapse_separators="_",
    case="lower",
    pattern=r"^[a-z]+_[a-z]+_\d{4}$",
)
GUARD = RetryGuardSettings(enabled=True, permanent_failure_pattern=r"^Error: .*\bnot found\b")
SETTINGS = ToolArgumentSettings(enabled=True, rules=(RULE,), retry_guard=GUARD)


def normalizer(settings: ToolArgumentSettings = SETTINGS) -> ArgumentNormalizer:
    return ArgumentNormalizer(
        settings,
        transcript=TranscriptSettings(),
        invalid_template="bad {label} {value} ({format_hint})",
        already_failed_template="failed {tool} {label} {value}: {spelled}",
    )


def call(user_id: object, *, name: str = "get_user_details", call_id: str = "c1") -> ToolCall:
    return ToolCall(id=call_id, name=name, arguments_json=canonical_json({"user_id": user_id}))


class CanonicalTests(unittest.TestCase):
    def test_case_separators_and_spoken_form(self) -> None:
        gate = normalizer()
        cases = {
            "Jordan_Lee_8264": "jordan_lee_8264",
            "JORDAN_LEE_8264": "jordan_lee_8264",
            "jordan__lee_8264_": "jordan_lee_8264",
            "jordan.lee_8264": "jordanlee_8264",
            "Jordan underscore Lee underscore eight two six four": "jordan_lee_8264",
            "A_A_L_T_O_B_E_L_L_O_3170": "a_a_l_t_o_b_e_l_l_o_3170",
        }
        for raw, expected in cases.items():
            with self.subTest(raw=raw):
                self.assertEqual(gate.canonical(RULE, raw), expected)

    def test_pattern(self) -> None:
        self.assertTrue(ArgumentNormalizer.valid(RULE, "jordan_lee_8264"))
        for value in ("ya", "699", "_7340", "a_a_l_t_o_b_e_l_l_o_3170", "jordanlee_8264"):
            self.assertFalse(ArgumentNormalizer.valid(RULE, value), value)
        self.assertTrue(ArgumentNormalizer.valid(replace(RULE, pattern=""), "anything"))


class ScreenTests(unittest.TestCase):
    def test_rewrite_is_sent_with_canonical_arguments(self) -> None:
        screening = normalizer().screen([call("Jordan_Lee_8264")], frozenset())
        (sent,) = screening.sent
        self.assertEqual(json.loads(sent.arguments_json), {"user_id": "jordan_lee_8264"})
        (rewrite,) = screening.rewrites
        self.assertEqual((rewrite.before, rewrite.after), ("Jordan_Lee_8264", "jordan_lee_8264"))
        self.assertEqual(screening.keys["c1"], ("get_user_details", canonical_json({"user_id": "jordan_lee_8264"})))

    def test_invalid_is_answered_locally_and_asks_for_the_whole_value(self) -> None:
        screening = normalizer().screen([call("YA")], frozenset())
        self.assertEqual(screening.sent, ())
        (answer,) = screening.local
        self.assertEqual((answer.reason, answer.value), (REASON_INVALID, "ya"))
        self.assertEqual(answer.message, "bad user ID ya (firstname_lastname_1234)")

    def test_escalated_wording_for_the_second_invalid_answer_of_a_tool(self) -> None:
        settings = replace(SETTINGS, escalate_invalid_message_key="spell_all")
        gate = ArgumentNormalizer(
            settings,
            transcript=TranscriptSettings(),
            invalid_template="read back {spelled}",
            already_failed_template="failed",
            escalate_invalid_template="spell the whole {label}",
        )
        first = gate.screen([call("YA")], frozenset())
        second = gate.screen([call("YA", call_id="c2")], frozenset(), {"get_user_details": 1})
        both = gate.screen([call("YA"), call("YB", call_id="c2")], frozenset())
        self.assertEqual(
            (first.local[0].message, first.local[0].message_key), ("read back y, a", "tool_argument_invalid")
        )
        self.assertEqual(
            (second.local[0].message, second.local[0].message_key), ("spell the whole user ID", "spell_all")
        )
        self.assertEqual([a.message_key for a in both.local], ["tool_argument_invalid", "spell_all"])
        self.assertEqual(second.sent, ())
        # Without an escalation key the counts change nothing (the baseline).
        baseline = normalizer().screen([call("YA")], frozenset(), {"get_user_details": 3})
        self.assertEqual(baseline.local[0].message, "bad user ID ya (firstname_lastname_1234)")

    def test_on_invalid_send_only_rewrites(self) -> None:
        settings = replace(SETTINGS, rules=(replace(RULE, on_invalid="send"),))
        screening = normalizer(settings).screen([call("YA")], frozenset())
        self.assertEqual(len(screening.sent), 1)
        self.assertEqual(screening.local, ())

    def test_already_failed_is_answered_locally_with_a_read_back(self) -> None:
        key = ("get_user_details", canonical_json({"user_id": "jordan_lee_8264"}))
        screening = normalizer().screen([call("JORDAN_LEE_8264")], frozenset({key}))
        (answer,) = screening.local
        self.assertEqual(answer.reason, REASON_ALREADY_FAILED)
        self.assertIn("j, o, r, d, a, n, underscore, l, e, e, underscore, 8, 2, 6, 4", answer.message)

    def test_other_tools_and_malformed_arguments_pass_through(self) -> None:
        other = ToolCall(id="c2", name="get_reservation_details", arguments_json='{"reservation_id": "h9z1c"}')
        broken = ToolCall(id="c3", name="get_user_details", arguments_json="{not json")
        not_a_string = call(4397, call_id="c4")
        screening = normalizer().screen([other, broken, not_a_string], frozenset())
        self.assertEqual(screening.sent, (other, broken, not_a_string))
        self.assertEqual(screening.rewrites, ())
        self.assertNotIn("c2", screening.keys)  # scope: rules

    def test_scope_all_guards_every_tool(self) -> None:
        settings = replace(SETTINGS, retry_guard=replace(GUARD, scope="all"))
        other = ToolCall(id="c2", name="get_reservation_details", arguments_json='{"reservation_id": "h9z1c"}')
        key = ("get_reservation_details", canonical_json({"reservation_id": "h9z1c"}))
        screening = normalizer(settings).screen([other], frozenset({key}))
        self.assertEqual(screening.local[0].reason, REASON_ALREADY_FAILED)

    def test_only_permanent_failures_count(self) -> None:
        gate = normalizer()
        self.assertTrue(gate.is_permanent_failure("Error: User jordan_lee_8264 not found"))
        self.assertFalse(gate.is_permanent_failure("Error: service unavailable, try again"))
        self.assertFalse(gate.is_permanent_failure('{"user_id": "jordan_lee_8264"}'))
        disabled = normalizer(replace(SETTINGS, retry_guard=RetryGuardSettings()))
        self.assertFalse(disabled.is_permanent_failure("Error: User x not found"))


class MessageTests(unittest.TestCase):
    def test_literal_substitution(self) -> None:
        self.assertEqual(render_message(" {a} and {b} {c} ", a="1", b="{a}"), "1 and {a} {c}")
