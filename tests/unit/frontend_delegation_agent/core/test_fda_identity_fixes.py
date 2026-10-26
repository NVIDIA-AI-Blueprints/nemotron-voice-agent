# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D100, D101, D102, D103, D107

"""I1 (the domain-agnostic <phone_numbers> block), the one shipped domain note (telecom) and I3/I4 (recovery notes)."""

from __future__ import annotations

import re
import tempfile
import unittest
from pathlib import Path

from _fda_backend_fakes import Harness as ControllerHarness
from _fda_variants import variant  # noqa: E402
from _fda_voice_fakes import PACKAGE, DelegationHarness, benchmark_session_update, profile_config

from examples.frontend_delegation_agent.core.errors import VoiceConfigError
from examples.frontend_delegation_agent.core.normalization.arguments import ResultHintSettings
from examples.frontend_delegation_agent.core.prompt_features import sha256_text
from examples.frontend_delegation_agent.core.sidecar import gateway_config as gc
from examples.frontend_delegation_agent.core.sidecar.templates import BackendTemplates
from examples.frontend_delegation_agent.core.tools.result_hints import ResultHints
from examples.frontend_delegation_agent.core.voice_config import load_voice_config

CONFIG = PACKAGE / "config"
BACKEND_PROMPTS = CONFIG / "prompts" / "customer_service" / "backend" / "hermes.yaml"
ALL_ON = {"spelling_v2": True, "spoken_output": True, "write_consent": True, "phone_format": True}
ALL_ON_BUT_PHONE = {**ALL_ON, "phone_format": False}
NOT_FOUND = "Error: User not found"
FOUND = '"sara_doe_496"'
DOMAINS = ("airline", "retail", "telecom", "mock")
#: The telecom note, exactly as disclosed with the submission (analysis/2026-10-11_fdh-telecom-gpt55-deep). Changing it
#: needs a new held-out validation and a new disclosure (telecom-domain-note-plan.md sections 4 and 5).
TELECOM_NOTE = (
    "Customers' location affects mobile service. For any problem with service, mobile data or messaging,\n"
    "ask early, right after you have identified the customer, whether they are at home or travelling,\n"
    "unless they have already said or your policy requires an immediate transfer. Ask once, and use the\n"
    "answer only for troubleshooting."
)
TELECOM_RULE = (("telecom", ("get_customer_by_phone",)),)
#: Benchmark words, fault names and digits that the note must not contain (submission doc section 0.2).
NOTE_FORBIDDEN = re.compile(r"roaming|apn|refuel|telecom|airline|retail|user_abroad|break_|_issue|mms|\d", re.I)


def _instructions(domain: str) -> str:
    return benchmark_session_update(domain)["session"]["instructions"]


def _phone_block(text: str) -> str:
    return text[text.index("<phone_numbers>") : text.index("</phone_numbers>") + len("</phone_numbers>")]


def _catalog_with_note(tmp: str) -> Path:
    """The shipped backend catalog plus one invented domain note (the mechanism, not a shipped note)."""
    text = BACKEND_PROMPTS.read_text(encoding="utf-8")
    assert "backend_domain_notes:\n  telecom: |-\n" in text  # noqa: S101 - the one shipped note
    path = Path(tmp) / "prompts.backend.yaml"
    path.write_text(
        text.replace(
            "backend_domain_notes:\n",
            "backend_domain_notes:\n  widgets: |-\n"
            "    {% if features.domain_notes %}Widget codes are case-sensitive.{% endif %}\n",
        ),
        encoding="utf-8",
    )
    return path


# -- the telecom domain note (analysis/2026-10-11_fdh-telecom-gpt55-deep) ---------------------------


class TelecomNoteTests(unittest.TestCase):
    def test_only_telecom_is_detected_and_only_it_has_a_note(self) -> None:
        for path in (CONFIG / "gateway.yaml", CONFIG / "gateway.eval.yaml", variant("gateway.eval_no_note.yaml")):
            with self.subTest(gateway=path.name):
                self.assertEqual(gc.load_gateway_config(path).domains, TELECOM_RULE)
        self.assertEqual(BackendTemplates(BACKEND_PROMPTS).domains_with_notes, ("telecom",))

    def test_the_note_is_on_by_default_and_off_in_the_controls_and_the_ablation(self) -> None:
        for name in ("gateway.yaml", "gateway.eval.yaml"):
            with self.subTest(gateway=name):
                self.assertTrue(gc.load_gateway_config(CONFIG / name).prompt_features["domain_notes"])
        off = ("gateway.eval_no_note.yaml", "gateway.eval_control.yaml", "gateway.eval_124acf7.yaml",
               "gateway.eval_situational.yaml", "gateway.baseline.yaml", "gateway.identity_control.yaml")  # fmt: skip
        for name in off:
            with self.subTest(gateway=name):
                self.assertFalse(gc.load_gateway_config(variant(name)).prompt_features["domain_notes"])
        on = gc.load_gateway_config(CONFIG / "gateway.eval.yaml")
        ablation = gc.load_gateway_config(variant("gateway.eval_no_note.yaml"))
        changed = {k for k, v in on.prompt_features.items() if v != ablation.prompt_features[k]}
        self.assertEqual(changed, {"domain_notes"})
        self.assertEqual(on.backend, ablation.backend)

    def test_the_note_renders_word_for_word_only_with_its_switch(self) -> None:
        self.assertEqual(BackendTemplates(BACKEND_PROMPTS, prompt_features={"domain_notes": True}).domain_note(
            "telecom"), TELECOM_NOTE)  # fmt: skip
        self.assertEqual(BackendTemplates(BACKEND_PROMPTS).domain_note("telecom"), "")
        for domain in ("airline", "retail", "mock", ""):
            with self.subTest(domain=domain):
                note = BackendTemplates(BACKEND_PROMPTS, prompt_features={"domain_notes": True}).domain_note(domain)
                self.assertEqual(note, "")

    def test_the_note_names_no_benchmark_term_fault_or_digit(self) -> None:
        self.assertEqual(NOTE_FORBIDDEN.findall(TELECOM_NOTE), [])
        self.assertLessEqual(TELECOM_NOTE.count(". ") + 1, 4)  # a few sentences at most


# -- I1: the phone-number block ----------------------------------------------------------------------


class PhoneFormatTests(unittest.TestCase):
    def test_block_renders_only_with_its_feature_and_identically_in_every_domain(self) -> None:
        on = BackendTemplates(BACKEND_PROMPTS, prompt_features=ALL_ON)
        off = BackendTemplates(BACKEND_PROMPTS, prompt_features=ALL_ON_BUT_PHONE)
        blocks = set()
        for domain in DOMAINS:
            with self.subTest(domain=domain):
                with_block = on.render("backend_system", instructions=_instructions(domain))
                without = off.render("backend_system", instructions=_instructions(domain))
                block = _phone_block(with_block)
                blocks.add(block)
                self.assertEqual(with_block.replace(f"\n\n{block}", "", 1), without)
                self.assertLess(with_block.index("</policy>"), with_block.index("<phone_numbers>"))
                self.assertNotIn("<phone_numbers>", without)
        self.assertEqual(len(blocks), 1)

    def test_block_formats_lookups_only_and_states_no_data_fact(self) -> None:
        block = _phone_block(
            BackendTemplates(BACKEND_PROMPTS, prompt_features=ALL_ON).render("backend_system", instructions="")
        )
        self.assertIn("look something up by phone number", block)
        self.assertIn("XXX-XXX-XXXX", block)
        self.assertIn("retry it once with the\nsame digits in the other common form", block)
        self.assertIn("Never add, remove or change digits", block)
        self.assertIn("pass it exactly as the user confirmed it or\nas a tool returned it; do not reformat it", block)
        self.assertNotIn("stored", block)  # a preference, not a statement about the data
        self.assertNotRegex(block, r"\d{3}")  # no example number at all

    def test_gateway_domains_are_validated(self) -> None:
        for bad in ({"widgets": ["lookup_widget"]}, {"widgets": {"tools_any": []}},
                    {"widgets": {"tools_any": ["x"], "tools_all": ["y"]}}, ["widgets"]):  # fmt: skip
            with self.subTest(bad=bad), self.assertRaises(gc.GatewayConfigError):
                gc.load_gateway_config(None, overrides={"domains": bad})
        self.assertEqual(gc.load_gateway_config(None).domains, ())

    def test_the_note_mechanism_still_works_with_an_invented_note(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = _catalog_with_note(tmp)
            on = BackendTemplates(path, prompt_features={"domain_notes": True})
            self.assertEqual(on.domain_note("widgets"), "Widget codes are case-sensitive.")
            self.assertEqual(on.domain_note(""), "")
            self.assertEqual(BackendTemplates(path).domain_note("widgets"), "")


class ControllerDomainTests(unittest.IsolatedAsyncioTestCase):
    async def _configure(
        self, tools: list[dict], instructions: str, features: dict[str, bool], **settings: object
    ) -> ControllerHarness:
        h = ControllerHarness(domains=gc.load_gateway_config(CONFIG / "gateway.yaml").domains, **settings)
        h.controller._templates = BackendTemplates(BACKEND_PROMPTS, prompt_features=features)  # noqa: SLF001
        await h.controller.open({"steer_mode": "auto"})
        await h.controller.configure(tools, instructions)
        return h

    async def test_without_the_note_every_domain_gets_the_same_prompt_shape(self) -> None:
        for domain in DOMAINS:
            for features in (ALL_ON, ALL_ON_BUT_PHONE):
                with self.subTest(domain=domain, features=features):
                    instructions = _instructions(domain)
                    h = await self._configure(
                        benchmark_session_update(domain)["session"]["tools"], instructions, features
                    )
                    system = h.worker.of_type("configure")[0]["instructions"]
                    self.assertNotIn("<domain_notes>", system)
                    self.assertEqual(features["phone_format"], "<phone_numbers>" in system)
                    plain = BackendTemplates(BACKEND_PROMPTS, prompt_features=features).render(
                        "backend_system", instructions=instructions
                    )
                    self.assertEqual(system, plain)
                    configured = h.of_type("session.configured")[0]
                    self.assertEqual(configured["backend_domain"], "telecom" if domain == "telecom" else "")
                    self.assertEqual(configured["backend_system_sha256"], sha256_text(system))

    async def test_the_note_changes_the_telecom_prompt_only(self) -> None:
        eval_features = dict(gc.load_gateway_config(CONFIG / "gateway.eval.yaml").prompt_features)
        for domain in DOMAINS:
            with self.subTest(domain=domain):
                tools, instructions = benchmark_session_update(domain)["session"]["tools"], _instructions(domain)
                # Read each harness before building the next one (the emulator worker belongs to the latest harness).
                on = await self._configure(tools, instructions, eval_features)
                with_note = on.worker.of_type("configure")[0]["instructions"]
                configured = on.of_type("session.configured")[0]
                off = await self._configure(tools, instructions, {**eval_features, "domain_notes": False})
                without = off.worker.of_type("configure")[0]["instructions"]
                if domain != "telecom":
                    self.assertEqual(with_note, without)  # byte-identical: the fingerprint is unchanged
                    self.assertEqual(configured["backend_domain"], "")
                    continue
                block = f"\n\n<domain_notes>\n{TELECOM_NOTE}\n</domain_notes>"
                self.assertEqual(with_note.count("<domain_notes>"), 1)
                self.assertIn(block, with_note)
                self.assertEqual(with_note.replace(block, "", 1), without)
                self.assertLess(with_note.index("</policy>"), with_note.index("<domain_notes>"))
                self.assertLess(with_note.index("</domain_notes>"), with_note.index("<situational_facts>"))
                self.assertEqual(configured["backend_domain"], "telecom")
                self.assertEqual(configured["backend_system_sha256"], sha256_text(with_note))

    async def test_an_invented_domain_gets_its_note_and_the_fingerprint_hashes_it(self) -> None:
        tools = [{"type": "function", "name": "lookup_widget", "description": "Look up a widget.", "parameters": {}}]
        with tempfile.TemporaryDirectory() as tmp:
            h = ControllerHarness(domains=(("widgets", ("lookup_widget",)),))
            h.controller._templates = BackendTemplates(  # noqa: SLF001
                _catalog_with_note(tmp), prompt_features={"domain_notes": True}
            )
            await h.controller.open({"steer_mode": "auto"})
            await h.controller.configure(tools, "POLICY")
        system = h.worker.of_type("configure")[0]["instructions"]
        self.assertIn("<domain_notes>\nWidget codes are case-sensitive.\n</domain_notes>", system)
        configured = h.of_type("session.configured")[0]
        self.assertEqual(configured["backend_domain"], "widgets")
        self.assertEqual(configured["backend_system_sha256"], sha256_text(system))


# -- I3 / I4: the per-session miss counter ---------------------------------------------------------


def _hints(**overrides: object) -> ResultHints:
    settings = ResultHintSettings(
        **{
            "enabled": True,
            "tools": ("find_user_id_by_name_zip", "find_user_id_by_email", "get_user_details"),
            "failure_pattern": r"^Error: .*\bnot found\b",
            "message_key": "i3",
            "escalate_message_key": "i4",
            **overrides,
        }
    )
    return ResultHints(
        settings, message="[I3 note]", escalate_message="[I4 note]" if settings.escalate_message_key else ""
    )


class ResultHintsTests(unittest.TestCase):
    def test_first_miss_gets_i3_later_misses_i4_and_the_output_is_kept(self) -> None:
        hints = _hints()
        first = hints.on_output("find_user_id_by_name_zip", NOT_FOUND)
        assert first is not None
        self.assertEqual((first.output, first.message_key, first.miss), (f"{NOT_FOUND}\n\n[I3 note]", "i3", 1))
        second = hints.on_output("find_user_id_by_email", NOT_FOUND)
        assert second is not None
        self.assertEqual((second.message_key, second.miss), ("i4", 2))  # one counter for every watched lookup

    def test_success_resets_and_is_never_annotated(self) -> None:
        hints = _hints()
        hints.on_output("find_user_id_by_name_zip", NOT_FOUND)
        self.assertIsNone(hints.on_output("find_user_id_by_name_zip", FOUND))
        again = hints.on_output("find_user_id_by_name_zip", NOT_FOUND)
        assert again is not None
        self.assertEqual(again.message_key, "i3")

    def test_local_invalid_counts_toward_the_next_miss(self) -> None:
        hints = _hints()
        hints.on_local_invalid("get_user_details")
        hint = hints.on_output("get_user_details", "Error: User aalto_ahlen_4417 not found")
        assert hint is not None
        self.assertEqual((hint.message_key, hint.miss), ("i4", 2))
        uncounted = _hints(count_local_invalid=False)
        uncounted.on_local_invalid("get_user_details")
        hint = uncounted.on_output("get_user_details", "Error: User aalto_ahlen_4417 not found")
        assert hint is not None
        self.assertEqual(hint.message_key, "i3")

    def test_bounded_by_max_hints(self) -> None:
        hints = _hints(max_hints=3)
        got = [hints.on_output("find_user_id_by_name_zip", NOT_FOUND) for _ in range(5)]
        self.assertEqual([h is not None for h in got], [True, True, True, False, False])

    def test_unwatched_tools_and_other_errors_are_untouched(self) -> None:
        hints = _hints()
        self.assertIsNone(
            hints.on_output("get_customer_by_phone", "Error: Customer with phone number 3034971268 not found")
        )
        self.assertIsNone(hints.on_output("get_order_details", "Error: Order not found"))
        hints.on_output("find_user_id_by_name_zip", NOT_FOUND)
        self.assertIsNone(hints.on_output("find_user_id_by_name_zip", "Error: upstream timeout"))  # transient
        later = hints.on_output("find_user_id_by_name_zip", NOT_FOUND)
        assert later is not None
        self.assertEqual(later.message_key, "i4")  # a transient error does not reset the count

    def test_without_escalation_every_miss_gets_the_first_note(self) -> None:
        hints = _hints(escalate_message_key="")
        keys = [hints.on_output("find_user_id_by_name_zip", NOT_FOUND) for _ in range(2)]
        self.assertEqual([h.message_key for h in keys if h], ["i3", "i3"])
        self.assertTrue(all(h and h.output.endswith("[I3 note]") for h in keys))


# -- I3 / I4 through the voice server ---------------------------------------------------------------


class RelayHintTests(unittest.IsolatedAsyncioTestCase):
    async def _session(self, profile: str | Path, domain: str) -> DelegationHarness:
        harness = DelegationHarness(profile_config(profile), transcripts=["I need help with my order"])
        await harness.start(benchmark_session_update(domain))
        await harness.speak()
        await harness.wait_sent("delegate")
        harness.link.deliver("state", state="WORKING", epoch=1, run_id="run_1")
        return harness

    async def _round_trip(self, harness: DelegationHarness, index: int, name: str, args: str, output: str) -> str:
        call_id = f"c{index}"
        results = len(harness.link.of_type("tool.result"))
        harness.link.deliver("tool.call", call_id=call_id, epoch=1, run_id="run_1", name=name, arguments=args)
        await harness.wait_for("response.function_call_arguments.done", count=index)
        await harness.send(
            {
                "type": "conversation.item.create",
                "item": {"type": "function_call_output", "call_id": call_id, "output": output},
            }
        )
        await harness.send({"type": "response.create"})
        return (await harness.wait_sent("tool.result", count=results + 1))["output"]

    async def test_default_profile_annotates_retail_misses_and_leaves_successes(self) -> None:
        harness = await self._session(variant("profiles/bench_eval.yaml"), "retail")
        args = '{"first_name": "Maren", "last_name": "Oakhurst", "zip": "40917"}'
        first = await self._round_trip(harness, 1, "find_user_id_by_name_zip", args, NOT_FOUND)
        second = await self._round_trip(harness, 2, "find_user_id_by_name_zip", args, NOT_FOUND)
        found = await self._round_trip(harness, 3, "find_user_id_by_name_zip", args, FOUND)
        self.assertTrue(first.startswith(NOT_FOUND + "\n\n[Not found. At least one value"))
        self.assertTrue(second.startswith(NOT_FOUND + "\n\n[Not found again."))
        self.assertIn('"S as in Sam', second)
        self.assertEqual(found, FOUND)
        logged = [(e["message_key"], e["n"]) for e in harness.logged("result_hint")]
        self.assertEqual(logged, [("lookup_not_found_hint", 1), ("lookup_not_found_hint_words", 2)])
        start = harness.logged("fda_session_start")[0]
        self.assertTrue(start["features"]["result_hints"])
        self.assertEqual(start["invalid_message_keys"]["result_hint_escalate"], "lookup_not_found_hint_words")
        await harness.close()

    async def test_control_and_telecom_outputs_are_byte_identical(self) -> None:
        email = '{"email": "a@b.com"}'
        for profile, domain, name, args, output in (
            (variant("profiles/bench_eval_baseline.yaml"), "retail", "find_user_id_by_email", email, NOT_FOUND),
            (variant("profiles/bench_identity_control.yaml"), "retail", "find_user_id_by_email", email, NOT_FOUND),
            (variant("profiles/bench_eval.yaml"), "telecom", "get_customer_by_phone", '{"phone_number": "3034971268"}',
             "Error: Customer with phone number 3034971268 not found"),
        ):  # fmt: skip
            with self.subTest(profile=profile.name, domain=domain):
                harness = await self._session(profile, domain)
                self.assertEqual(await self._round_trip(harness, 1, name, args, output), output)
                self.assertEqual(harness.logged("result_hint"), [])
                await harness.close()

    async def test_airline_local_invalid_then_external_miss_gets_i4(self) -> None:
        harness = await self._session(variant("profiles/bench_eval.yaml"), "airline")
        harness.link.deliver(
            "tool.call",
            call_id="c0",
            epoch=1,
            run_id="run_1",
            name="get_user_details",
            arguments='{"user_id": "aalto_ah"}',
        )
        local = await harness.wait_sent("tool.result")
        self.assertTrue(local["local"])
        self.assertNotIn("[Not found", local["output"])  # a local answer is never annotated
        output = await self._round_trip(
            harness, 1, "get_user_details", '{"user_id": "aalto_ahlen_4417"}', "Error: User aalto_ahlen_4417 not found"
        )
        self.assertIn("[Not found again.", output)
        await harness.close()

    async def test_i3_arm_repeats_the_first_note(self) -> None:
        harness = await self._session(variant("profiles/bench_arm_i3_recovery_hint.yaml"), "retail")
        outputs = [
            await self._round_trip(harness, i, "find_user_id_by_email", '{"email": "a@b.com"}', NOT_FOUND)
            for i in (1, 2)
        ]
        self.assertTrue(all("[Not found. At least one value" in o for o in outputs))
        await harness.close()


class ResultHintConfigTests(unittest.TestCase):
    def _load(self, text: str):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "voice.yaml"
            path.write_text(f"extends: {variant('voice/bench.yaml')}\n{text}")
            return load_voice_config(path)

    def test_defaults_are_off(self) -> None:
        hints = self._load("").normalization.tool_arguments.result_hints
        self.assertFalse(hints.enabled)
        self.assertEqual((hints.max_hints, hints.count_local_invalid), (3, True))

    def test_invalid_settings_fail_at_load(self) -> None:
        base = "normalization:\n  tool_arguments:\n    result_hints:\n      enabled: true\n"
        good = "      tools: [find_user_id_by_email]\n      failure_pattern: 'not found'\n"
        for extra in (
            "      failure_pattern: 'not found'\n",  # no tools
            "      tools: [find_user_id_by_email]\n",  # no pattern
            good + "      failure_pattern: '('\n",  # bad regex
            good + "      message_key: no_such_key\n",  # not in the catalog
            good + "      escalate_message_key: no_such_key\n",
            good + "      max_hints: 0\n",
        ):
            with self.subTest(extra=extra), self.assertRaises(VoiceConfigError):
                self._load(base + extra)
        off = "normalization:\n  tool_arguments:\n    enabled: false\n"
        with self.assertRaises(VoiceConfigError):
            self._load(off + "    result_hints:\n      enabled: true\n" + good)
        self.assertTrue(self._load(base + good).normalization.tool_arguments.result_hints.enabled)

    def test_shipped_voice_profiles(self) -> None:
        voice = CONFIG / "voice"
        on = {
            p.name
            for p in voice.glob("*.yaml")
            if load_voice_config(p).normalization.tool_arguments.result_hints.enabled
        }
        self.assertEqual(
            on,
            {
                "pipecat_realtime_eval.yaml",  # the Pipecat adapter's eval voice profile extends realtime_eval.yaml
                "realtime_eval.yaml",
            },
        )


if __name__ == "__main__":
    unittest.main()
