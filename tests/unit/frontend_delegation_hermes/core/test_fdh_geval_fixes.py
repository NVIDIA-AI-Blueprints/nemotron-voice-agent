# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D100, D101, D102, D103, D107

"""G0 of tau3-geval-failure-fixes-plan.md: A1 / A1-lite, A2, A3, B1-B3, C1, C2 and the P0.3 trace.

Invented tool schemas and values only (no tau-bench names in triggers); the tau2 fixtures are used
only to show where a switch fires and that the control stays byte-identical.
"""

from __future__ import annotations

import asyncio
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from typing import Any

from _fdh_variants import variant  # noqa: E402
from _fdh_voice_fakes import PACKAGE, DelegationHarness, profile_config, tau2_session_update
from test_fdh_delegation_config import GEVAL_BACKEND_FIXES

from examples.frontend_delegation_hermes.core.engine.output_scheduler import CallsItem
from examples.frontend_delegation_hermes.core.errors import VoiceConfigError
from examples.frontend_delegation_hermes.core.normalization.arguments import (
    ArgumentNormalizer,
    FormatRetrySettings,
    ResultHintSettings,
    RetryGuardSettings,
    ToolArgumentSettings,
)
from examples.frontend_delegation_hermes.core.normalization.schema_rules import build_schema_rules
from examples.frontend_delegation_hermes.core.normalization.transcript import (
    SpelledRunSettings,
    TranscriptNormalizer,
    TranscriptSettings,
)
from examples.frontend_delegation_hermes.core.sidecar import gateway_config as gc
from examples.frontend_delegation_hermes.core.sidecar.session_runtime import controller_settings
from examples.frontend_delegation_hermes.core.sidecar.templates import BackendTemplates
from examples.frontend_delegation_hermes.core.text.messages import canonical_json
from examples.frontend_delegation_hermes.core.tools.field_provenance import (
    HEARD,
    SPELLED,
    UNKNOWN,
    FieldProvenance,
    SpelledSpan,
    field_labels,
)
from examples.frontend_delegation_hermes.core.tools.format_retry import FormatRetry, other_format
from examples.frontend_delegation_hermes.core.tools.relay import FormatRetryMessages, ToolRelay
from examples.frontend_delegation_hermes.core.tools.result_hints import ResultHints
from examples.frontend_delegation_hermes.core.voice_config import load_voice_config
from examples.frontend_delegation_hermes.core.worker.hermes_adapter import BackendTrace, register_trace_hooks

CONFIG = PACKAGE / "config"
NOT_FOUND = "Error: Account with phone number {} not found"
PERSON_MISS = "Error: Customer not found"
FOUND = '{"account_id": "AC-1", "name": "Pat Example"}'
MESSAGES = FormatRetryMessages(
    hint="[Same digits in the other common format: {value}. Look that up once before asking the user.]",
    found="[Looked up as {value}: same digits, other format.] [Found only after a format change.]",
    not_found="[Also not found as {value}.]",
)
#: Words of the benchmark domains that must not appear in the new prompt text (plan section 8).
TAU_WORDS = ("refuel", "basic economy", "apn", "roaming", "airline", "retail", "telecom", "reservation")


def _tool(name: str, description: str, properties: dict[str, Any], *, read_only: bool | None = None) -> dict:
    tool: dict[str, Any] = {
        "type": "function",
        "name": name,
        "description": description,
        "parameters": {"type": "object", "properties": properties},
    }
    if read_only is not None:
        tool["annotations"] = {"readOnlyHint": read_only, "idempotentHint": read_only}
    return tool


def _string(description: str = "") -> dict[str, str]:
    return {"type": "string", "description": description}


#: Invented schemas: what must and must not get a format retry.
INVENTED_TOOLS = [
    _tool("find_account", "Find an account by phone.", {"phone": _string()}, read_only=True),
    _tool("lookup_member", "Look up a member.", {"mobile_number": _string()}, read_only=True),
    _tool("get_status", "Get the status of a line.", {"phone": _string()}),  # no declaration: hint only
    _tool("get_contact", "Get a contact.", {"contact": _string("The contact's phone number.")}),
    _tool("get_item", "Get an item.", {"item_id": _string("The item id, such as '1234567890'.")}),
    _tool("get_order", "Get an order.", {"order_number": _string()}),
    _tool("update_contact", "Update a contact's phone.", {"phone": _string()}),  # a write tool
    _tool("send_code", "Send a code.", {"phone": _string()}),  # a write tool, never a read tool
]


# -- backend prompt variants (A2, B1-B3, C1, C2) ------------------------------------------------------


class BackendPromptTests(unittest.TestCase):
    def _render(self, **features: bool) -> str:
        on = {"spelling_v2": True, "spoken_output": True, "write_consent": True, "phone_format": True, **features}
        return BackendTemplates(CONFIG / "prompts" / "customer_service" / "backend.yaml", prompt_features=on).render(
            "backend_system", instructions="POLICY"
        )

    def test_each_variant_adds_only_its_own_text_and_off_is_the_control(self) -> None:
        control = self._render()
        blocks = {
            "format_correction": "is a format correction and not a variant",
            "write_consent_v2": "A request, even one that names the change, is not that yes.",
            "policy_check": "Refuse only what\na rule actually forbids for this case.",
            "request_scope": 'uses words like "all", "both", "each" or "every"',
            "transfer_check": "Call\nthe transfer tool before you tell the user you are transferring them.",
            "capability_check": "check your tool list and the tool descriptions",
        }
        self.assertEqual(set(blocks), set(GEVAL_BACKEND_FIXES))
        for name, text in blocks.items():
            with self.subTest(variant=name):
                rendered = self._render(**{name: True})
                self.assertIn(text, rendered)
                self.assertNotIn(text, control)
                for other, other_text in blocks.items():
                    if other != name:
                        self.assertNotIn(other_text, rendered)
                self.assertEqual(rendered.count("<consent>"), 1)

    def test_blocks_wrap_only_when_on_and_b1_stands_alone_without_write_consent(self) -> None:
        control = self._render()
        for tag in ("<before_change>", "<before_transfer>"):
            self.assertNotIn(tag, control)
        both = self._render(policy_check=True, request_scope=True, transfer_check=True, capability_check=True)
        self.assertEqual((both.count("<before_change>"), both.count("<before_transfer>")), (1, 1))
        alone = self._render(write_consent=False, write_consent_v2=True)
        self.assertIn("<consent>\nYour policy decides", alone)
        self.assertNotIn('Treat a "yes"', alone)
        self.assertNotIn("<consent>", self._render(write_consent=False))

    def test_new_prompt_text_uses_no_benchmark_words(self) -> None:
        added = self._render(**dict.fromkeys(GEVAL_BACKEND_FIXES, True))
        control = self._render()
        new_lines = [line for line in added.splitlines() if line not in control.splitlines()]
        self.assertTrue(new_lines)
        for word in TAU_WORDS:
            self.assertFalse(any(word in line.lower() for line in new_lines), word)

    def test_the_eval_gateway_turns_on_every_variant_and_the_control_none(self) -> None:
        default = gc.load_gateway_config(CONFIG / "gateway.eval.yaml")
        control = gc.load_gateway_config(variant("gateway.eval_control.yaml"))
        self.assertTrue(all(default.prompt_features[name] for name in GEVAL_BACKEND_FIXES))
        self.assertFalse(any(control.prompt_features[name] for name in GEVAL_BACKEND_FIXES))
        self.assertTrue(default.hermes.backend_trace and control.hermes.backend_trace)


class CatalogVariantTests(unittest.TestCase):
    def test_a2_hint_variant_is_the_i3_note_plus_one_sentence(self) -> None:
        import yaml

        catalog = yaml.safe_load((PACKAGE / "config/prompts/customer_service/voice.yaml").read_text())
        base = " ".join(catalog["lookup_not_found_hint"]["content"].split())
        variant = " ".join(catalog["lookup_not_found_hint_format"]["content"].split())
        sentence = (
            "If a value may only be in the wrong format (the same characters with different separators or case), "
            "look it up once in the other format before asking the user. "
        )
        self.assertIn(sentence, variant)
        self.assertEqual(variant.replace(sentence, ""), base)


# -- A1 / A1-lite ---------------------------------------------------------------------------------------


class SchemaFieldTests(unittest.TestCase):
    def test_phone_fields_by_name_or_description_and_declared_read_only_tools(self) -> None:
        schema = build_schema_rules(INVENTED_TOOLS)
        self.assertEqual(
            set(schema.format_retry_arguments),
            {"find_account.phone", "lookup_member.mobile_number", "get_status.phone", "get_contact.contact"},
        )
        self.assertEqual(set(schema.read_only_tools), {"find_account", "lookup_member"})

    def test_a_declaration_must_say_read_only_and_idempotent(self) -> None:
        half = _tool("find_line", "Find a line.", {"phone": _string()})
        half["annotations"] = {"readOnlyHint": True}
        nested = {"type": "function", "function": {**_tool("find_box", "Find a box.", {"phone": _string()})}}
        nested["function"]["annotations"] = {"readOnlyHint": True, "idempotentHint": True}
        self.assertEqual(build_schema_rules([half, nested]).read_only_tools, ("find_box",))


class OtherFormatTests(unittest.TestCase):
    def test_both_directions_and_only_ten_digits(self) -> None:
        self.assertEqual(other_format("5558675309"), "555-867-5309")
        self.assertEqual(other_format("555-867-5309"), "5558675309")
        self.assertEqual(other_format("(555) 867.5309"), "5558675309")
        for value in ("555867530", "55586753091", "555-867-530A", "", "+1 555 867 5309"):
            with self.subTest(value=value):
                self.assertIsNone(other_format(value))


class CandidateTests(unittest.TestCase):
    def _retry(self, mode: str = "call") -> FormatRetry:
        schema = build_schema_rules(INVENTED_TOOLS)
        return FormatRetry(
            FormatRetrySettings(enabled=True, mode=mode),
            fields=schema.format_retry_fields(),
            read_only_tools=schema.read_only_tools,
        )

    def _candidate(self, retry: FormatRetry, tool: str, args: dict, *, failed=frozenset(), rule: bool = False):
        return retry.candidate(tool, json.dumps(args), has_rule=lambda *_: rule, failed=failed)

    def test_declared_read_only_tools_are_called_others_hinted(self) -> None:
        retry = self._retry()
        found = self._candidate(retry, "find_account", {"phone": "5558675309"})
        self.assertEqual((found.after, found.call), ("555-867-5309", True))
        self.assertEqual(found.key, ("find_account", canonical_json({"phone": "555-867-5309"})))
        hint = self._candidate(retry, "get_status", {"phone": "555-867-5309"})
        self.assertEqual((hint.after, hint.call), ("5558675309", False))
        self.assertFalse(self._candidate(self._retry("hint"), "find_account", {"phone": "5558675309"}).call)

    def test_never_on_non_phone_ids_writes_or_other_digit_counts(self) -> None:
        retry = self._retry()
        for tool, args in (
            ("get_item", {"item_id": "1234567890"}),
            ("get_order", {"order_number": "5551234567"}),
            ("update_contact", {"phone": "5558675309"}),
            ("send_code", {"phone": "5558675309"}),
            ("find_account", {"phone": "555867530"}),
            ("find_account", {"phone": "55586753091"}),
        ):
            with self.subTest(tool=tool, args=args):
                self.assertIsNone(self._candidate(retry, tool, args))

    def test_once_per_value_and_never_a_known_failure_or_a_ruled_argument(self) -> None:
        retry = self._retry()
        self.assertIsNotNone(self._candidate(retry, "find_account", {"phone": "5558675309"}))
        self.assertIsNone(self._candidate(retry, "find_account", {"phone": "555-867-5309"}))  # same digits
        failed = frozenset({("lookup_member", canonical_json({"mobile_number": "555-867-5309"}))})
        self.assertIsNone(self._candidate(retry, "lookup_member", {"mobile_number": "5558675309"}, failed=failed))
        self.assertIsNone(self._candidate(retry, "get_status", {"phone": "5551234567"}, rule=True))


class _RelayRig:
    """A wire relay with the retry guard, I3 notes and a format retry over the invented tools."""

    def __init__(
        self, *, mode: str | None = "call", provenance: FieldProvenance | None = None, timeout_s: float = 5.0
    ) -> None:
        self.results: list[tuple[str, int, str, bool]] = []
        self.emitted: list[CallsItem] = []
        self.logs: list[tuple[str, dict[str, Any]]] = []
        schema = build_schema_rules(INVENTED_TOOLS)
        guard = RetryGuardSettings(enabled=True, scope="all", permanent_failure_pattern=r"(?i)\bnot found\b")
        hint_settings = ResultHintSettings(
            enabled=True,
            auto_tools=True,
            failure_pattern=r"(?i)\b(account|customer|member)s?\b[^.]*\bnot found\b",
            field_provenance=provenance is not None,
        )
        self.normalizer = ArgumentNormalizer(
            ToolArgumentSettings(enabled=True, retry_guard=guard),
            transcript=TranscriptSettings(),
            invalid_template="",
            already_failed_template="failed {value}",
        )
        hints = ResultHints(
            hint_settings, message="[I3 note]", escalate_message="[I4 note]", read_tools=schema.read_tools
        )
        retry = (
            FormatRetry(
                FormatRetrySettings(enabled=True, mode=mode),
                fields=schema.format_retry_fields(),
                read_only_tools=schema.read_only_tools,
            )
            if mode
            else None
        )
        self.relay = ToolRelay(
            executor="wire",
            send_result=lambda *result: self.results.append(result),
            emit_calls=self.emitted.append,
            log=lambda kind, **data: self.logs.append((kind, data)),
            normalizer=self.normalizer,
            result_hints=hints,
            batch_window_ms=1,
            format_retry=retry,
            format_messages=MESSAGES,
            provenance=provenance,
            result_timeout_s=timeout_s,
        )

    async def call(self, call_id: str, name: str, args: dict[str, Any], epoch: int = 1) -> None:
        self.relay.on_call({"call_id": call_id, "epoch": epoch, "name": name, "arguments": json.dumps(args)})
        await asyncio.sleep(0.02)

    def logged(self, kind: str) -> list[dict[str, Any]]:
        return [data for k, data in self.logs if k == kind]

    @property
    def failed(self) -> frozenset:
        return self.relay._failed  # noqa: SLF001 - the retry guard's set


class RelayFormatRetryTests(unittest.IsolatedAsyncioTestCase):
    async def test_hint_on_an_undeclared_tool_then_the_i3_note(self) -> None:
        rig = _RelayRig()
        await rig.call("c1", "get_status", {"phone": "5558675309"})
        rig.relay.on_function_output("c1", NOT_FOUND.format("5558675309"))
        (_, _, output, local) = rig.results[-1]
        self.assertFalse(local)
        self.assertEqual(
            output,
            NOT_FOUND.format("5558675309")
            + "\n\n[Same digits in the other common format: 555-867-5309. Look that up once before asking the user.]"
            + "\n\n[I3 note]",
        )
        self.assertEqual(rig.logged("format_retry")[0]["outcome"], "hint")
        self.assertEqual(len(rig.emitted), 1)  # no call of its own

    async def test_found_after_retry_is_emitted_after_the_ack_and_only_the_original_form_fails(self) -> None:
        rig = _RelayRig()
        await rig.call("c1", "find_account", {"phone": "5558675309"})
        rig.relay.on_function_output("c1", NOT_FOUND.format("5558675309"))
        self.assertEqual(rig.results, [])  # withheld
        self.assertEqual(len(rig.emitted), 1)  # the retry waits for the original batch's acknowledgement
        self.assertTrue(rig.relay.take_ack())
        self.assertEqual(len(rig.emitted), 2)
        retry = rig.emitted[-1].calls[0]
        self.assertEqual(retry, ("c1.fmt", "find_account", '{"phone": "555-867-5309"}'))
        rig.relay.on_function_output("c1.fmt", FOUND)
        (call_id, epoch, output, _) = rig.results[-1]
        self.assertEqual((call_id, epoch), ("c1", 1))
        self.assertTrue(output.startswith("[Looked up as 555-867-5309: same digits, other format.]"))
        self.assertTrue(output.endswith(FOUND))
        self.assertEqual(rig.failed, {("find_account", canonical_json({"phone": "5558675309"}))})
        self.assertTrue(rig.relay.take_ack())  # the synthetic batch's own acknowledgement
        self.assertFalse(rig.relay.take_ack())
        self.assertEqual(rig.logged("format_retry")[-1]["outcome"], "found")
        self.assertTrue(rig.logged("tool_calls_out")[-1]["format_retry"])

    async def test_not_found_marks_both_forms_and_appends_the_note_then_the_hint(self) -> None:
        rig = _RelayRig()
        await rig.call("c1", "find_account", {"phone": "555-867-5309"})
        rig.relay.on_function_output("c1", NOT_FOUND.format("555-867-5309"))
        rig.relay.take_ack()
        rig.relay.on_function_output("c1.fmt", NOT_FOUND.format("5558675309"))
        output = rig.results[-1][2]
        self.assertEqual(output, NOT_FOUND.format("555-867-5309") + "\n\n[Also not found as 5558675309.]\n\n[I3 note]")
        self.assertEqual(len(rig.failed), 2)
        self.assertEqual(len(rig.logged("result_hint")), 1)  # one miss for the pair

    async def test_transient_error_returns_the_original_and_leaves_the_other_form_unmarked(self) -> None:
        rig = _RelayRig()
        await rig.call("c1", "find_account", {"phone": "5558675309"})
        rig.relay.on_function_output("c1", NOT_FOUND.format("5558675309"))
        rig.relay.take_ack()
        rig.relay.on_function_output("c1.fmt", "Error: service unavailable")
        self.assertEqual(rig.results[-1][2], NOT_FOUND.format("5558675309") + "\n\n[I3 note]")
        self.assertEqual(len(rig.failed), 1)
        self.assertEqual(rig.logged("format_retry")[-1]["outcome"], "transient")

    async def test_timeout_returns_the_original_and_ignores_the_late_output(self) -> None:
        rig = _RelayRig(timeout_s=0.05)
        await rig.call("c1", "find_account", {"phone": "5558675309"})
        rig.relay.on_function_output("c1", NOT_FOUND.format("5558675309"))
        rig.relay.take_ack()
        await asyncio.sleep(0.1)
        self.assertEqual(rig.results[-1][:3], ("c1", 1, NOT_FOUND.format("5558675309") + "\n\n[I3 note]"))
        rig.relay.on_function_output("c1.fmt", FOUND)
        self.assertEqual(len(rig.results), 1)
        self.assertEqual(rig.logged("tool_result_ignored")[-1]["call_id"], "c1.fmt")
        self.assertEqual(rig.logged("format_retry")[-1]["outcome"], "timeout")
        self.assertTrue(rig.relay.take_ack())  # the synthetic batch is still acknowledged once

    async def test_cancelling_the_original_cancels_the_retry_and_sends_nothing(self) -> None:
        rig = _RelayRig()
        await rig.call("c1", "find_account", {"phone": "5558675309"})
        rig.relay.on_function_output("c1", NOT_FOUND.format("5558675309"))
        rig.relay.take_ack()
        rig.relay.on_cancel("c1", "interrupt")
        rig.relay.on_function_output("c1.fmt", FOUND)
        self.assertEqual(rig.results, [])
        self.assertEqual(rig.logged("format_retry")[-1]["outcome"], "cancelled")

    async def test_cancel_before_the_ack_never_emits_the_retry(self) -> None:
        rig = _RelayRig()
        await rig.call("c1", "find_account", {"phone": "5558675309"})
        rig.relay.on_function_output("c1", NOT_FOUND.format("5558675309"))
        rig.relay.on_cancel("c1", "new_epoch")
        self.assertTrue(rig.relay.take_ack())
        self.assertEqual(len(rig.emitted), 1)
        self.assertEqual(rig.results, [])

    async def test_off_and_non_phone_outputs_are_byte_identical(self) -> None:
        for mode, tool, args in (
            (None, "find_account", {"phone": "5558675309"}),
            ("call", "get_item", {"item_id": "1234567890"}),
            ("call", "get_order", {"order_number": "5551234567"}),
        ):
            with self.subTest(mode=mode, tool=tool):
                rig = _RelayRig(mode=mode)
                await rig.call("c1", tool, args)
                miss = "Error: Item 1234567890 not found"
                rig.relay.on_function_output("c1", miss)
                self.assertEqual(rig.results[-1][2], miss)
                self.assertEqual(rig.logged("format_retry"), [])


# -- A3 -------------------------------------------------------------------------------------------------


def _spelled(text: str) -> tuple[SpelledSpan, ...]:
    normalizer = TranscriptNormalizer(
        TranscriptSettings(enabled=True, spelled_runs=SpelledRunSettings(enabled=True, min_tokens=3))
    )
    return tuple(
        SpelledSpan(span.written, span.start, span.end)
        for span in normalizer.normalize(text).spans
        if span.kind == "spelled"
    )


class FieldProvenanceTests(unittest.TestCase):
    ARGS = {"first_name": "Leigh", "last_name": "Wilson", "zip": "32255"}

    def _classes(self, provenance: FieldProvenance, args: dict[str, str] | None = None) -> dict[str, tuple]:
        return {name: (kind, other) for name, _, kind, other in provenance.attribute(args or self.ARGS)}

    def _turn(self, provenance: FieldProvenance, text: str, question: str = "") -> None:
        provenance.add_turn(text, _spelled(text), question)

    def test_labels(self) -> None:
        self.assertEqual(field_labels("first_name"), ("first name",))
        self.assertEqual(field_labels("zip_code"), ("zip code", "zip"))
        self.assertEqual(field_labels("phone_number"), ("phone number", "phone"))

    def test_spelled_heard_and_unknown(self) -> None:
        provenance = FieldProvenance()
        self._turn(provenance, "My first name is Leigh.")
        self._turn(provenance, "L, I, it's W, I, L, S, O, N", "Could you spell your last name?")
        classes = self._classes(provenance)
        self.assertEqual(classes["first_name"], (HEARD, ""))
        self.assertEqual(classes["last_name"], (SPELLED, ""))
        self.assertEqual(classes["zip"], (UNKNOWN, ""))  # never mentioned

    def test_the_spelling_the_lookup_did_not_use_is_reported(self) -> None:
        provenance = FieldProvenance()
        self._turn(provenance, "S, O, F, I, A", "And your first name, letter by letter?")
        args = {"first_name": "Sophia", "last_name": "Lee"}
        self.assertEqual(self._classes(provenance, args)["first_name"], (UNKNOWN, "SOFIA"))
        self.assertIn(
            'first_name "Sophia" (unknown) (the user spelled "SOFIA" for the first name)',
            provenance.describe(json.dumps(args)),
        )

    def test_a_span_answering_another_fields_request_is_unknown(self) -> None:
        provenance = FieldProvenance()
        self._turn(provenance, "W, I, L, S, O, N", "Could you spell your last name?")
        args = {"first_name": "WILSON", "last_name": "Grey"}
        self.assertEqual(self._classes(provenance, args)["first_name"], (UNKNOWN, ""))
        self.assertEqual(self._classes(provenance, args)["last_name"], (UNKNOWN, "WILSON"))

    def test_a_multi_value_or_two_person_turn_is_unknown(self) -> None:
        provenance = FieldProvenance()
        self._turn(provenance, "My last name is K, I, M and hers is P, A, R, K", "")
        self.assertEqual(self._classes(provenance, {"last_name": "KIM"})["last_name"], (UNKNOWN, ""))

    def test_separators_must_match_and_digit_only_values_are_never_attributed(self) -> None:
        provenance = FieldProvenance()
        self._turn(provenance, "my code is A, B, one, two", "")
        self.assertEqual(self._classes(provenance, {"code": "AB-12"})["code"][0], UNKNOWN)
        self._turn(provenance, "my zip is 32255", "")
        self.assertEqual(self._classes(provenance, {"zip": "32255"})["zip"], (UNKNOWN, ""))

    def test_a_later_answer_for_the_field_supersedes_an_earlier_one(self) -> None:
        provenance = FieldProvenance()
        self._turn(provenance, "L, E, I, G, H", "Can you spell your first name?")
        self._turn(provenance, "Sorry, my first name is actually Lee", "")
        self.assertEqual(self._classes(provenance, {"first_name": "LEIGH"})["first_name"], (UNKNOWN, ""))


class RelayProvenanceTests(unittest.IsolatedAsyncioTestCase):
    async def test_only_the_second_and_later_miss_gets_the_line(self) -> None:
        provenance = FieldProvenance()
        provenance.add_turn("My first name is Leigh", (), "")
        rig = _RelayRig(mode=None, provenance=provenance)
        args = {"first_name": "Leigh", "zip": "32255"}
        for index in (1, 2):
            await rig.call(f"c{index}", "find_account", {**args, "n": index})
            rig.relay.on_function_output(f"c{index}", PERSON_MISS)
            rig.relay.take_ack()
        first, second = rig.results[0][2], rig.results[1][2]
        self.assertEqual(first, PERSON_MISS + "\n\n[I3 note]")
        self.assertEqual(
            second,
            PERSON_MISS
            + '\n\n[I4 note]\n[Fields in this lookup: first_name "Leigh" (heard as a word), zip "32255" (unknown), '
            + 'n "2" (unknown).]',
        )


# -- end to end through the voice server ---------------------------------------------------------------


class VoiceArmTests(unittest.IsolatedAsyncioTestCase):
    async def _round_trip(self, profile: str | Path, domain: str, name: str, args: str, output: str, transcript: str):
        harness = DelegationHarness(profile_config(profile), transcripts=[transcript])
        await harness.start(tau2_session_update(domain))
        await harness.speak()
        await harness.wait_sent("delegate")
        harness.link.deliver("state", state="WORKING", epoch=1, run_id="run_1")
        harness.link.deliver("tool.call", call_id="c1", epoch=1, run_id="run_1", name=name, arguments=args)
        await harness.wait_for("response.function_call_arguments.done")
        await harness.send(
            {
                "type": "conversation.item.create",
                "item": {"type": "function_call_output", "call_id": "c1", "output": output},
            }
        )
        await harness.send({"type": "response.create"})
        result = (await harness.wait_sent("tool.result"))["output"]
        return harness, result

    async def test_a1_on_a_tau2_phone_lookup_is_the_hint_and_the_control_is_unchanged(self) -> None:
        miss = "Error: Customer with phone number 5551232002 not found"
        args = '{"phone_number": "5551232002"}'
        control, out = await self._round_trip(
            variant("profiles/realtime_eval_control.yaml"),
            "telecom",
            "get_customer_by_phone",
            args,
            miss,
            "my number please",
        )
        self.assertTrue(out.startswith(miss + "\n\n[Not found."))
        self.assertNotIn("other common format", out)
        self.assertFalse(control.logged("fdh_session_start")[0]["features"]["format_retry"])
        await control.close()
        arm, out = await self._round_trip(
            variant("profiles/realtime_eval_arm_a1_format_retry.yaml"),
            "telecom",
            "get_customer_by_phone",
            args,
            miss,
            "my number",
        )
        self.assertTrue(
            out.startswith(miss + "\n\n[Same digits in the other common format: 555-123-2002. Look that up once")
        )
        self.assertIn("\n\n[Not found.", out)  # the I3 note still follows
        logged = arm.logged("format_retry")
        self.assertEqual((logged[0]["mode"], logged[0]["outcome"]), ("hint", "hint"))  # no declaration in tau2
        rules = arm.logged("session_rules")[-1]
        self.assertEqual(rules["format_retry_arguments"], ["get_customer_by_phone.phone_number"])
        self.assertEqual(rules["read_only_tools"], [])
        self.assertEqual(arm.logged("tool_calls_out")[0]["arguments"], [args])
        await arm.close()

    async def test_a3_reports_a_spelling_the_lookup_did_not_use(self) -> None:
        harness = DelegationHarness(
            profile_config(variant("profiles/realtime_eval_arm_a3_field_provenance.yaml")),
            transcripts=["My first name is spelled S, O, F, I, A"],
        )
        await harness.start(tau2_session_update("retail"))
        await harness.speak()
        await harness.wait_sent("delegate")
        harness.link.deliver("state", state="WORKING", epoch=1, run_id="run_1")
        outputs = []
        for index, last_name in ((1, "Lee"), (2, "Li")):  # a repeated identical call is answered locally
            args = json.dumps({"first_name": "Sophia", "last_name": last_name, "zip": "10001"})
            harness.link.deliver(
                "tool.call",
                call_id=f"c{index}",
                epoch=1,
                run_id="run_1",
                name="find_user_id_by_name_zip",
                arguments=args,
            )
            await harness.wait_for("response.function_call_arguments.done", count=index)
            await harness.send(
                {
                    "type": "conversation.item.create",
                    "item": {"type": "function_call_output", "call_id": f"c{index}", "output": "Error: User not found"},
                }
            )
            await harness.send({"type": "response.create"})
            outputs.append((await harness.wait_sent("tool.result", count=index))["output"])
        self.assertNotIn("[Fields in this lookup", outputs[0])
        self.assertIn(
            '[Fields in this lookup: first_name "Sophia" (unknown) (the user spelled "SOFIA" for the first name), '
            'last_name "Li" (unknown), zip "10001" (unknown).]',
            outputs[1],
        )
        self.assertTrue(harness.logged("fdh_session_start")[0]["features"]["field_provenance"])
        await harness.close()


class FormatRetryConfigTests(unittest.TestCase):
    def _load(self, text: str):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "voice.yaml"
            path.write_text(f"extends: {variant('voice/realtime_eval_control.yaml')}\n{text}")
            return load_voice_config(path)

    def test_defaults_are_off_and_invalid_settings_fail(self) -> None:
        settings = self._load("").normalization.tool_arguments
        self.assertFalse(settings.format_retry.enabled)
        self.assertFalse(settings.result_hints.field_provenance)
        base = "normalization:\n  tool_arguments:\n    format_retry:\n      enabled: true\n"
        self.assertEqual(self._load(base).normalization.tool_arguments.format_retry.mode, "call")
        for extra in (
            "      mode: always\n",
            "      fields: email\n",
            "      hint_message_key: no_such_key\n",
        ):
            with self.subTest(extra=extra), self.assertRaises(VoiceConfigError):
                self._load(base + extra)
        with self.assertRaises(VoiceConfigError):
            self._load(base + "    retry_guard: {enabled: false}\n")

    def test_the_submission_profile_adds_its_fixes_over_the_control(self) -> None:
        control = profile_config(variant("profiles/realtime_eval_control.yaml")).features
        default = profile_config("realtime_eval.yaml")
        changed = {k: v for k, v in default.features.items() if control.get(k) != v}
        # Since 2026-10-09 the default also runs fix-plan-f1-f2-f4-f8.md (F1 + F2, F8; the ledger replaces A3).
        self.assertEqual(
            changed,
            {
                "format_retry": "call",
                "schema_format_retry_fields": True,
                "liveness_reply": True,
                "question_never_silent": True,
                "ping_as_status": True,
                "replay_unheard_answer": True,
                "replay_triggers": ["backchannel", "ping", "repeat"],
                "identity_ledger": True,
            },
        )
        self.assertEqual(
            default.voice.normalization.tool_arguments.result_hints.message_key, "lookup_not_found_hint_format"
        )


# -- P0.3 backend trace -----------------------------------------------------------------------------


class BackendTraceTests(unittest.TestCase):
    def test_records_and_hook_registration(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            trace = BackendTrace(Path(tmp) / "sess_1.trace.jsonl", session_id="sess_1")
            hooks: dict[str, list] = {}
            manager = types.SimpleNamespace(_hooks=hooks, discover_and_load=lambda: None)
            plugins = types.ModuleType("hermes_cli.plugins")
            plugins.get_plugin_manager = lambda: manager  # type: ignore[attr-defined]
            package = types.ModuleType("hermes_cli")
            saved = {name: sys.modules.get(name) for name in ("hermes_cli", "hermes_cli.plugins")}
            sys.modules.update({"hermes_cli": package, "hermes_cli.plugins": plugins})
            try:
                self.assertTrue(register_trace_hooks(trace))
            finally:
                for name, module in saved.items():
                    if module is None:
                        sys.modules.pop(name, None)
                    else:
                        sys.modules[name] = module
            trace.session(tools=[{"type": "function"}], model="m", instructions="POLICY", request_overrides={})
            hooks["pre_api_request"][0](
                session_id="sess_1", api_request_id="r1", request_messages=[{"role": "user", "content": "hi"}]
            )
            hooks["post_api_request"][0](
                session_id="sess_1",
                api_request_id="r1",
                finish_reason="tool_calls",
                response={"assistant_message": {"content": "", "tool_calls": [{"id": "t1"}]}},
            )
            hooks["pre_api_request"][0](session_id="another", api_request_id="r2")
            records = [json.loads(line) for line in trace.path.read_text().splitlines()]
        self.assertEqual([r["kind"] for r in records], ["session", "request", "response"])
        self.assertEqual(records[1]["messages"][0]["content"], "hi")
        self.assertEqual(records[2]["assistant_message"]["tool_calls"], [{"id": "t1"}])
        self.assertEqual(len(records[0]["tools_sha256"]), 16)

    def test_gateway_passes_the_log_dir_only_when_on(self) -> None:
        on = gc.load_gateway_config(CONFIG / "gateway.eval.yaml")
        off = gc.load_gateway_config(CONFIG / "gateway.yaml")
        self.assertEqual(controller_settings(on).worker_hermes["backend_trace_dir"], on.workers.log_dir)
        self.assertEqual(controller_settings(off).worker_hermes["backend_trace_dir"], "")
        self.assertTrue(on.summary()["hermes"]["backend_trace"])


if __name__ == "__main__":
    unittest.main()
