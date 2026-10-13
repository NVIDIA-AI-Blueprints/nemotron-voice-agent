# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D100, D101, D102, D103, D107

"""fix-plan-f1-f2-f4-f8.md: deployed goldens, F1/F2 (no dead air), F4 (situational facts), F8 (identity ledger)."""

from __future__ import annotations

import re
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from _fdh_golden_eval import GATEWAY, load_golden, render_all
from _fdh_voice_fakes import PACKAGE, PROFILES, DelegationHarness, profile_config, tau2_session_update

from examples.frontend_delegation_hermes.cli import backend_decision_replay as g2
from examples.frontend_delegation_hermes.core.config import (
    DelegationConfigError,
    ReplaySettings,
    load_delegation_config,
)
from examples.frontend_delegation_hermes.core.engine.output_scheduler import CallsItem
from examples.frontend_delegation_hermes.core.engine.transcript import NOT_HEARD, PARTIAL
from examples.frontend_delegation_hermes.core.frontend.decider import DecisionContext, RuleDecider, apply_guards
from examples.frontend_delegation_hermes.core.frontend.delegate_tool import Decision
from examples.frontend_delegation_hermes.core.frontend.turn_class import (
    BACKCHANNEL,
    CONTENT,
    HOLD,
    LIVENESS,
    UNCLASSIFIED,
    classify_turn,
    lexicon_for,
)
from examples.frontend_delegation_hermes.core.normalization.arguments import ResultHintSettings
from examples.frontend_delegation_hermes.core.prompt_features import render_backend_system
from examples.frontend_delegation_hermes.core.sidecar import gateway_config as gc
from examples.frontend_delegation_hermes.core.sidecar.templates import BackendTemplates
from examples.frontend_delegation_hermes.core.tools.identity_ledger import (
    AMBIGUOUS,
    ERROR,
    FOUND,
    NOT_FOUND,
    UNKNOWN,
    IdentityLedger,
    LedgerSettings,
    classify_lookup_output,
    identification_tools,
)
from examples.frontend_delegation_hermes.core.tools.relay import ToolRelay
from examples.frontend_delegation_hermes.core.tools.result_hints import ResultHints

LEXICONS = load_delegation_config(PROFILES / "realtime_eval.yaml").delegation.liveness.lexicons
EN = LEXICONS["en"]
BACKEND_PROMPTS = PACKAGE / "config" / "prompts.backend.yaml"
FAILURE = r"(?i)\b(users?|customers?|members?|clients?|profiles?)\b[^.]*\b(not found|no such|does not exist)\b"


# -- step 0: the deployed eval profile is unchanged with every new switch off ------------------------------------


class DeployedGoldenTests(unittest.TestCase):
    def test_eval_profile_renders_and_decides_exactly_as_deployed_124acf7(self) -> None:
        renders, golden = render_all(), load_golden()
        self.assertEqual(sorted(renders), sorted(golden))
        for name, text in renders.items():
            with self.subTest(name=name):
                self.assertEqual(text, golden[name])

    def test_the_golden_telecom_prompt_has_the_hash_the_deployed_runs_logged(self) -> None:
        from examples.frontend_delegation_hermes.core.prompt_features import sha256_text  # noqa: PLC0415

        # backend_configured.backend_system_sha256 of every telecom session of the 2026-10-08 gpt-5.5 run.
        self.assertEqual(sha256_text(load_golden()["backend_system_telecom.txt"]), "e69f6c719285ddae")


# -- F1/F2: the turn classifier ---------------------------------------------------------------------------------


class TurnClassTests(unittest.TestCase):
    def test_table(self) -> None:
        cases = {
            "Hello? Are you still there?": (LIVENESS, False, True),
            "Okay. Hello, any update?": (LIVENESS, False, True),
            "Hold on. Second. Okay. Any update? Are you still there?": (LIVENESS, False, True),
            "Sorry, could you repeat that?": (LIVENESS, True, True),
            "could you repeat that": (LIVENESS, True, False),  # has_question is about content words
            "I can't hear you": (LIVENESS, True, False),
            "Okay.": (BACKCHANNEL, False, False),
            "Mm-hmm.": (BACKCHANNEL, False, False),
            "okay sure": (BACKCHANNEL, False, False),
            "Okay?": (BACKCHANNEL, False, True),
            "Give me a moment.": (HOLD, False, False),
            "Second.": (HOLD, False, False),
            "No.": (CONTENT, False, False),
            "Yes.": (CONTENT, False, False),
            "Not okay.": (CONTENT, False, False),
            "That's not right.": (CONTENT, False, False),
            "No, I'm still here": (CONTENT, False, False),
            "Hello, I want to change my order": (CONTENT, False, False),
            "Any update on order 1234?": (CONTENT, False, True),
            "what do you need from me": (CONTENT, False, True),
            "can you check my balance": (CONTENT, False, True),
            "hellooo": (CONTENT, False, False),
            "um": (UNCLASSIFIED, False, False),
            "": (UNCLASSIFIED, False, False),
        }
        for text, (kind, repeat, question) in cases.items():
            with self.subTest(text=text):
                got = classify_turn(text, EN)
                self.assertEqual((got.kind, got.repeat, got.has_question), (kind, repeat, question))

    def test_unicode_apostrophes_and_no_partial_tokens(self) -> None:
        self.assertEqual(classify_turn("I didn’t catch that", EN).kind, LIVENESS)
        self.assertEqual(classify_turn("helloworld", EN).kind, CONTENT)

    def test_language_fallback_and_missing_lexicon(self) -> None:
        self.assertEqual(lexicon_for("en-US", LEXICONS)[0], "en")
        self.assertEqual(lexicon_for("en-GB", LEXICONS)[0], "en")
        self.assertEqual(lexicon_for("de-DE", LEXICONS), ("", None))
        self.assertEqual(classify_turn("Hallo? Sind Sie noch da?", None).kind, UNCLASSIFIED)


# -- F2: the guards -------------------------------------------------------------------------------------------


SILENT = Decision(delegate=False, filler_text="")
SPEAKS = Decision(delegate=False, filler_text="You're welcome!")
TASK = Decision(delegate=True, filler_text="One moment.", request="task")
ALL_ON = {"liveness_reply": True, "question_never_silent": True, "ping_as_status": True}


def guard(decision: Decision, text: str, state: str = "IDLE", *, asked: bool = False, **flags: bool) -> Decision:
    context = DecisionContext(system_prompt="", messages=[], user_text=text, state=state, backend_asked_question=asked)
    return apply_guards(
        decision,
        context,
        backend_question_needs_delegate=True,
        backchannel_words=tuple(EN.backchannel),
        turn_class=classify_turn(text, EN),
        **flags,
    )


class GuardTests(unittest.TestCase):
    def test_r1_delegates_a_silent_ping_status_in_working_task_otherwise(self) -> None:
        for state, request in (("IDLE", "task"), ("NO_SESSION", "task"), ("WORKING", "status")):
            with self.subTest(state=state):
                out = guard(SILENT, "Are you still there?", state, liveness_reply=True)
                self.assertEqual(
                    (out.delegate, out.request, out.repair, out.filler_text), (True, request, "liveness_reply", "")
                )

    def test_r2_delegates_a_silent_question_but_not_hold_or_backchannel(self) -> None:
        out = guard(SILENT, "What do you need from me?", question_never_silent=True)
        self.assertEqual((out.delegate, out.repair), (True, "question_never_silent"))
        for text in ("Give me a moment.", "Okay?", "Okay."):
            with self.subTest(text=text):
                self.assertFalse(guard(SILENT, text, question_never_silent=True).delegate)

    def test_r3_relabels_a_working_ping_and_keeps_the_filler(self) -> None:
        out = guard(TASK, "Any update?", "WORKING", ping_as_status=True)
        self.assertEqual((out.request, out.filler_text, out.repair), ("status", "One moment.", "ping_as_status"))
        self.assertEqual(guard(TASK, "And my other booking", "WORKING", ping_as_status=True), TASK)

    def test_a_decision_that_speaks_or_delegates_content_is_never_changed(self) -> None:
        for decision, text, state in (
            (SPEAKS, "Are you still there?", "IDLE"),
            (TASK, "Are you still there?", "IDLE"),
            (TASK, "What do you need?", "WORKING"),
        ):
            with self.subTest(text=text, state=state):
                self.assertEqual(guard(decision, text, state, **ALL_ON), decision)

    def test_switches_off_or_no_class_is_today(self) -> None:
        self.assertEqual(guard(SILENT, "Are you still there?"), SILENT)
        context = DecisionContext(system_prompt="", messages=[], user_text="Hello?", state="IDLE")
        out = apply_guards(SILENT, context, backend_question_needs_delegate=True, turn_class=None, **ALL_ON)
        self.assertEqual(out, SILENT)


# -- F1/F2 config -----------------------------------------------------------------------------------------------


def _profile(text: str):
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "p.yaml"
        path.write_text(f"extends: {PROFILES / 'realtime_eval_124acf7.yaml'}\n{text}")
        return load_delegation_config(path)


class LivenessConfigTests(unittest.TestCase):
    def test_on_in_the_submission_profile_off_in_every_control_and_in_the_code_defaults(self) -> None:
        names = ("liveness_reply", "question_never_silent", "ping_as_status", "replay_unheard_answer")
        on = load_delegation_config(PROFILES / "realtime_eval.yaml")
        self.assertTrue(all(on.features[name] for name in names))
        self.assertEqual(on.features["replay_triggers"], ["backchannel", "ping", "repeat"])
        self.assertEqual((on.delegation.replay.ttl_s, on.delegation.replay.whole_if_heard_below), (60.0, 0.2))
        for control in ("realtime_eval_124acf7.yaml", "realtime_eval_control.yaml", "tau3_eval.yaml"):
            with self.subTest(control=control):
                features = load_delegation_config(PROFILES / control).features
                self.assertFalse(any(features[name] for name in names))
                self.assertEqual(features["replay_triggers"], [])
        self.assertEqual(ReplaySettings().triggers, ("status_intent",))  # Python defaults stay off

    def test_new_triggers_need_no_frontend_variant_but_status_intent_does(self) -> None:
        config = _profile("delegation: {replay_unheard_answer: {enabled: true, triggers: [backchannel, ping]}}\n")
        self.assertEqual(config.delegation.replay.triggers, ("backchannel", "ping"))
        with self.assertRaisesRegex(DelegationConfigError, "replay_intent"):
            _profile("delegation: {replay_unheard_answer: {enabled: true}}\n")

    def test_invalid_settings_fail(self) -> None:
        for text, message in (
            ("delegation: {replay_unheard_answer: {triggers: [shout]}}\n", "triggers"),
            ("delegation: {replay_unheard_answer: {whole_if_heard_below: 1.5}}\n", "above 1"),
            ("delegation: {guards: {lexicon: {en: {pings: [hi]}}}}\n", "unknown lexicon lists"),
        ):
            with self.subTest(text=text), self.assertRaisesRegex(DelegationConfigError, message):
                _profile(text)

    def test_the_liveness_arm_turns_on_only_its_own_switches(self) -> None:
        control = load_delegation_config(PROFILES / "realtime_eval_124acf7.yaml").features
        arm = load_delegation_config(PROFILES / "realtime_eval_liveness.yaml").features
        changed = {k for k in arm if arm[k] != control[k]}
        self.assertEqual(
            changed,
            {"liveness_reply", "question_never_silent", "ping_as_status", "replay_unheard_answer", "replay_triggers"},
        )
        replay = load_delegation_config(PROFILES / "realtime_eval_liveness.yaml").delegation.replay
        self.assertEqual((replay.ttl_s, replay.whole_if_heard_below), (60.0, 0.2))


# -- F1/F2 through the voice server -----------------------------------------------------------------------------

ANSWER = "Your order ships on Monday. It will arrive by Friday. The tracking number is in your email."
TASK_SCRIPT = {"delegate": True, "filler_text": "Sure, one moment.", "request": "task"}
SILENT_SCRIPT = {"delegate": False, "filler_text": ""}


def liveness_config(*, triggers=("backchannel", "ping", "repeat"), whole: float = 0.0, **flags: bool):
    config = profile_config("realtime_eval.yaml")
    liveness = replace(config.delegation.liveness, **{**dict.fromkeys(ALL_ON, False), **flags})
    replay = ReplaySettings(enabled=bool(triggers), ttl_s=60.0, triggers=tuple(triggers), whole_if_heard_below=whole)
    return replace(config, delegation=replace(config.delegation, liveness=liveness, replay=replay))


class LivenessTurnTests(unittest.IsolatedAsyncioTestCase):
    async def _answered(self, harness: DelegationHarness, *, outcome: str = NOT_HEARD, heard: str = "") -> None:
        await harness.start(tau2_session_update("retail"))
        await harness.speak()
        await harness.wait_sent("delegate")
        harness.link.deliver("state", state="WORKING", epoch=1, run_id="run_1")
        await harness.idle(1500)
        harness.link.deliver(
            "answer", answer_id="a1", run_id="run_1", epoch=1, kind="hermes", text=ANSWER, turn_ids=[1]
        )
        await harness.wait_for("response.done", count=2)
        manager = harness.manager
        assert manager is not None
        entry = manager.transcript.last_backend_answer()
        assert entry is not None and entry.released_mono is not None
        manager._unheard.clear()
        entry.outcome, entry.heard_text = outcome, heard
        harness.link.deliver(
            "backend_run_done", run_id="run_1", epoch=1, status="ok", history="committed", turn_ids=[1]
        )
        harness.link.deliver("state", state="IDLE", epoch=1, run_id=None)

    async def test_cut_answer_then_okay_is_replayed_not_silent(self) -> None:
        harness = DelegationHarness(
            liveness_config(), decider=RuleDecider([TASK_SCRIPT, SILENT_SCRIPT]), transcripts=["where is it", "Okay."]
        )
        await self._answered(harness)
        await harness.speak()
        await harness.wait_for("response.done", count=3)
        replayed = harness.logged("answer_replayed")
        self.assertEqual([r["trigger"] for r in replayed], ["backchannel"])
        self.assertEqual(harness.transcripts()[-1], ANSWER)
        self.assertEqual(len(harness.link.of_type("delegate")), 1)
        await harness.close()

    async def test_heard_answer_then_okay_stays_silent(self) -> None:
        harness = DelegationHarness(
            liveness_config(), decider=RuleDecider([TASK_SCRIPT, SILENT_SCRIPT]), transcripts=["where is it", "Okay."]
        )
        await self._answered(harness, outcome="heard", heard=ANSWER)
        await harness.speak()
        await harness.settle()
        await harness.idle(500)
        self.assertEqual(harness.logged("answer_replayed"), [])
        self.assertEqual(harness.logged("answer_replay_skipped")[-1]["reason"], "answer_heard")
        self.assertEqual(len(harness.link.of_type("delegate")), 1)
        await harness.close()

    async def test_ping_replays_without_r1_and_a_second_ping_is_delegated_by_r1(self) -> None:
        harness = DelegationHarness(
            liveness_config(liveness_reply=True),
            decider=RuleDecider([TASK_SCRIPT, SILENT_SCRIPT, SILENT_SCRIPT]),
            transcripts=["where is it", "Are you still there?", "Hello?"],
        )
        await self._answered(harness)
        await harness.speak()
        await harness.wait_for("response.done", count=3)
        self.assertEqual([r["trigger"] for r in harness.logged("answer_replayed")], ["ping"])
        await harness.idle(3000)
        await harness.speak()
        sent = await harness.wait_sent("delegate", count=2)
        self.assertEqual(sent["request"], "task")
        guards = [g for g in harness.logged("delegation_guard") if g["guard"] == "liveness_reply"]
        self.assertEqual(len(guards), 2)
        self.assertEqual(harness.logged("answer_replay_skipped")[-1]["reason"], "already_replayed")
        await harness.close()

    async def test_ping_trigger_alone_without_r1_stays_silent_when_the_replay_check_fails(self) -> None:
        harness = DelegationHarness(
            liveness_config(triggers=("ping",)),
            decider=RuleDecider([TASK_SCRIPT, SILENT_SCRIPT]),
            transcripts=["where is it", "Hello?"],
        )
        await self._answered(harness, outcome="heard", heard=ANSWER)
        await harness.speak()
        await harness.idle(500)
        self.assertEqual(harness.logged("answer_replay_skipped")[-1]["trigger"], "ping")
        self.assertEqual(len(harness.link.of_type("delegate")), 1)
        await harness.close()

    async def test_partial_answer_replays_the_cut_sentence_or_the_whole_below_the_threshold(self) -> None:
        for whole, heard, expected in (
            (0.0, "Your", ANSWER),
            (
                0.0,
                "Your order ships on Monday. It will",
                "It will arrive by Friday. The tracking number is in your email.",
            ),
            (0.2, "Your", ANSWER),
            (
                0.2,
                "Your order ships on Monday. It will",
                "It will arrive by Friday. The tracking number is in your email.",
            ),
        ):
            with self.subTest(whole=whole, heard=heard):
                harness = DelegationHarness(
                    liveness_config(whole=whole),
                    decider=RuleDecider([TASK_SCRIPT, SILENT_SCRIPT]),
                    transcripts=["where is it", "Okay."],
                )
                await self._answered(harness, outcome=PARTIAL, heard=heard)
                await harness.speak()
                await harness.wait_for("response.done", count=3)
                self.assertEqual(harness.transcripts()[-1], expected)
                await harness.close()

    async def test_working_ping_labelled_task_goes_as_status_with_a_filler(self) -> None:
        harness = DelegationHarness(
            liveness_config(triggers=(), ping_as_status=True),
            decider=RuleDecider([TASK_SCRIPT, TASK_SCRIPT]),
            transcripts=["where is it", "Are you still there?"],
        )
        await harness.start(tau2_session_update("retail"))
        await harness.speak()
        await harness.wait_sent("delegate")
        harness.link.deliver("state", state="WORKING", epoch=1, run_id="run_1")
        await harness.speak()
        second = await harness.wait_sent("delegate", count=2)
        self.assertEqual(second["request"], "status")
        self.assertEqual(harness.logged("filler_deduped")[-1]["action"], "replaced")
        await harness.close()

    async def test_unclassified_language_keeps_todays_behaviour(self) -> None:
        config = liveness_config(liveness_reply=True)
        voice = config.voice
        asr = replace(voice.asr, endpoint=replace(voice.asr.endpoint, language_code="de-DE"))
        config = replace(config, voice=replace(voice, asr=asr))
        harness = DelegationHarness(config, decider=RuleDecider([SILENT_SCRIPT]), transcripts=["Are you still there?"])
        await harness.start(tau2_session_update("retail"))
        await harness.speak()
        await harness.idle(500)
        self.assertEqual(harness.link.of_type("delegate"), [])
        self.assertEqual(len(harness.logged("lexicon_missing")), 1)
        await harness.close()


# -- F4 ---------------------------------------------------------------------------------------------------------


class SituationalFactsTests(unittest.TestCase):
    def _render(self, **features: bool) -> str:
        base = dict(gc.load_gateway_config(GATEWAY).prompt_features)
        templates = BackendTemplates(BACKEND_PROMPTS, prompt_features={**base, **features})
        return render_backend_system(templates, tau2_session_update("telecom")["session"]["instructions"])

    def test_off_is_the_deployed_prompt_and_on_adds_only_the_block_right_after_the_policy(self) -> None:
        off, on = self._render(), self._render(situational_facts=True)
        self.assertEqual(off, load_golden()["backend_system_telecom.txt"])
        start = on.index("\n\n<situational_facts>")
        end = on.index("</situational_facts>") + len("</situational_facts>")
        self.assertEqual(on[:start] + on[end:], off)
        self.assertTrue(on[:start].endswith("</policy>\n"))
        added = on[start:end]
        # analysis/2026-10-10_fdh-roaming-gpt55: ask early, only when not told, and before any transfer.
        self.assertIn("where the user is right now", added)
        self.assertIn("If the user has\nalready told you, do not ask", added)
        self.assertIn("unless your policy requires the transfer right away", added)

    def test_the_block_renders_only_with_its_switch_in_the_controls(self) -> None:
        for name in ("gateway.eval_124acf7.yaml", "gateway.eval_control.yaml", "gateway.baseline.yaml"):
            with self.subTest(gateway=name):
                features = gc.load_gateway_config(PACKAGE / "config" / name).prompt_features
                templates = BackendTemplates(BACKEND_PROMPTS, prompt_features=features)
                text = render_backend_system(templates, tau2_session_update("telecom")["session"]["instructions"])
                self.assertNotIn("<situational_facts>", text)
        on = gc.load_gateway_config(PACKAGE / "config" / "gateway.yaml").prompt_features
        self.assertTrue(on["situational_facts"])

    def test_the_block_renders_in_every_domain_and_uses_no_benchmark_words(self) -> None:
        for domain in ("airline", "retail", "telecom"):
            base = dict(gc.load_gateway_config(GATEWAY).prompt_features)
            templates = BackendTemplates(BACKEND_PROMPTS, prompt_features={**base, "situational_facts": True})
            text = render_backend_system(templates, tau2_session_update(domain)["session"]["instructions"])
            block = text[text.index("<situational_facts>") :]
            for word in ("roaming", "abroad", "apn", "airline", "flight", "telecom", "retail", "order"):
                with self.subTest(domain=domain, word=word):
                    self.assertIsNone(re.search(rf"\b{word}\b", block.lower()))

    def test_the_replay_renders_exactly_what_the_gateway_renders(self) -> None:
        control = load_golden()["backend_system_telecom.txt"]
        self.assertEqual(g2.render_arm_prompt(str(GATEWAY), control), control)
        with tempfile.TemporaryDirectory() as tmp:
            arm = Path(tmp) / "gateway.arm.yaml"
            arm.write_text(f"extends: {GATEWAY}\nprompt_features: {{situational_facts: true}}\n")
            self.assertEqual(g2.render_arm_prompt(str(arm), control), self._render(situational_facts=True))

    def test_the_situational_arm_turns_on_only_its_variant(self) -> None:
        eval_features = gc.load_gateway_config(GATEWAY).prompt_features
        arm = gc.load_gateway_config(PACKAGE / "config" / "gateway.eval_situational.yaml").prompt_features
        self.assertEqual({k for k in arm if arm[k] != eval_features[k]}, {"situational_facts"})


# -- F8 ---------------------------------------------------------------------------------------------------------


def tool(name: str, description: str, required: list[str], extra: tuple[str, ...] = ()) -> dict:
    properties = {arg: {"type": "string"} for arg in (*required, *extra)}
    return {"type": "function", "name": name, "description": description,
            "parameters": {"type": "object", "properties": properties, "required": required}}  # fmt: skip


INVENTED = [
    tool("find_member_id_by_email", "Find a member id by email.", ["email"]),
    tool(
        "find_member_id_by_name_postal",
        "Find a member id by name and postal code.",
        ["first_name", "last_name", "postal_code"],
    ),
    tool("get_member_details", "Get the details of a member.", ["member_id"]),
    tool("get_bills_for_member", "Get the bills of a member.", ["member_id"]),
    tool("get_data_usage", "Get the data usage of a line.", ["member_id", "line_id"]),
    tool("get_details_by_id", "Retrieves the details for an id.", ["id"]),
    tool("get_order_details", "Get the details of an order.", ["order_id"]),
    tool("update_member_address", "Update the address of a member.", ["member_id", "address"]),
]  # fmt: skip


def ledger(tools=INVENTED, **settings) -> IdentityLedger:
    return IdentityLedger(
        identification_tools(tools),
        tools=tools,
        failure_pattern=FAILURE,
        template="[Ledger:\n{lines}\nend]",
        settings=LedgerSettings(**settings),
    )


class IdentificationToolTests(unittest.TestCase):
    def test_invented_schemas(self) -> None:
        self.assertEqual(
            identification_tools(INVENTED),
            ("find_member_id_by_email", "find_member_id_by_name_postal", "get_member_details"),
        )

    def test_tau2_fixtures(self) -> None:
        expected = {
            "airline": {"get_user_details"},
            "retail": {"find_user_id_by_email", "find_user_id_by_name_zip", "get_user_details"},
            "telecom": {"get_customer_by_phone", "get_customer_by_id", "get_customer_by_name"},
        }
        for domain, names in expected.items():
            with self.subTest(domain=domain):
                tools = tau2_session_update(domain)["session"]["tools"]
                self.assertEqual(set(identification_tools(tools)), names)


class OutputClassTests(unittest.TestCase):
    def test_classes_in_order(self) -> None:
        failure = re.compile(FAILURE)
        cases = {
            ("Error: User not found", True): NOT_FOUND,
            ("", True): NOT_FOUND,
            ("[]", False): NOT_FOUND,
            ("{}", False): NOT_FOUND,
            ("null", False): NOT_FOUND,
            ("Error: timeout", True): ERROR,
            ('{"error": "bad", "request_id": "r1"}', False): ERROR,
            ('[{"customer_id": "C1"}, {"customer_id": "C2"}]', False): AMBIGUOUS,
            ('[{"customer_id": "C1"}]', False): FOUND,
            ('{"user_id": "jane_doe_12", "name": {}}', False): FOUND,
            ("jane_doe_12", True): FOUND,
            ("jane_doe_12", False): UNKNOWN,
            ('{"user_id": null}', False): UNKNOWN,
            ('{"id": "x"}', False): UNKNOWN,
            ('{"request_id": "r1"}', False): UNKNOWN,
            ("Your account is ready", True): UNKNOWN,
        }
        for (output, returns_an_id), expected in cases.items():
            with self.subTest(output=output):
                self.assertEqual(classify_lookup_output(output, failure=failure, returns_an_id=returns_an_id), expected)


MISS = "Error: Member not found"


class LedgerTests(unittest.TestCase):
    def test_spelling_status_never_hides_a_spelling(self) -> None:
        book = ledger()
        book.add_turn(1, "C H E N", ["CHEN"], "Could you spell your last name?")
        book.add_turn(2, "K O V A C S", ["KOVACS"], "And your first name?")
        book.add_turn(3, "john dot doe", ["JOHN.DOE"], "Your email?")
        book.add_turn(4, "D A I K I", ["DAIKI"], "Your first name?")
        book.on_issue(
            "c1",
            "find_member_id_by_name_postal",
            '{"first_name": "kovacs", "last_name": "Hen", "postal_code": "19122"}',
        )
        book.on_issue("c2", "find_member_id_by_email", '{"email": "johndoe"}')
        book.on_output("c1", MISS)
        _, note = book.on_output("c2", MISS)
        self.assertIn('"CHEN" (asked: "Could you spell your last name?") - not sent', note)
        self.assertIn('"KOVACS" (asked: "And your first name?") - sent as first_name in lookup 1', note)
        self.assertIn('"JOHN.DOE" (asked: "Your email?") - sent as email in lookup 2, with different separators', note)
        self.assertIn('"DAIKI" (asked: "Your first name?") - not sent', note)

    def test_name_inside_a_sent_email_is_not_sent(self) -> None:
        book = ledger()
        book.add_turn(1, "D A I K I", ["DAIKI"], "Your first name?")
        book.on_issue("c1", "find_member_id_by_email", '{"email": "daiki.sanchez@example.com"}')
        _, note = book.on_output("c1", MISS)
        self.assertIn('"DAIKI" (asked: "Your first name?") - not sent', note)

    def test_lookups_side_by_side_and_other_lookups_from_the_second_miss(self) -> None:
        book = ledger()
        book.on_issue(
            "c1",
            "find_member_id_by_name_postal",
            '{"first_name": "Yusuf", "last_name": "Rossi", "postal_code": "19121"}',
        )
        _, first = book.on_output("c1", MISS)
        self.assertEqual(first, "")  # nothing to state yet: no spelling, no correction, first miss
        book.on_issue(
            "c2",
            "find_member_id_by_name_postal",
            '{"first_name": "Yusuf", "last_name": "Rosi", "postal_code": "19122"}',
        )
        _, second = book.on_output("c2", MISS)
        self.assertIn(
            '1) find_member_id_by_name_postal first_name="Yusuf" last_name="Rossi" postal_code="19121"', second
        )
        self.assertIn(
            '2) find_member_id_by_name_postal first_name="Yusuf" last_name="Rosi" postal_code="19122"', second
        )
        self.assertIn("find_member_id_by_email(email), get_member_details(member_id)", second)

    def test_corrections_are_verbatim(self) -> None:
        book = ledger()
        book.add_turn(1, "It's Chen, not Ken. C, H, E, N", ["CHEN"], "")
        book.add_turn(2, "my postal code is 19122", [], "")
        book.on_issue(
            "c1", "find_member_id_by_name_postal", '{"first_name": "a", "last_name": "Ken", "postal_code": "1"}'
        )
        _, note = book.on_output("c1", MISS)
        self.assertIn('User corrections: "It\'s Chen, not Ken. C, H, E, N".', note)
        self.assertNotIn("19122", note)

    def test_episodes_found_closes_and_only_found_closes(self) -> None:
        book = ledger()
        book.add_turn(1, "C H E N", ["CHEN"], "")  # before any lookup: joins episode 1
        book.on_issue("c1", "find_member_id_by_email", '{"email": "a@b.c"}')
        self.assertIn("CHEN", book.on_output("c1", MISS)[1])
        for output in ("[]", "Error: timeout", '[{"member_id": "M1"}, {"member_id": "M2"}]', "free text"):
            book.on_issue("cx", "find_member_id_by_email", '{"email": "x@y.z"}')
            book.on_output("cx", output)
            self.assertEqual(book.episode, 1, output)
        book.on_issue("c2", "get_member_details", '{"member_id": "M1"}')
        self.assertEqual(book.on_output("c2", '{"member_id": "M1"}'), (FOUND, ""))
        self.assertEqual(book.episode, 0)
        book.add_turn(9, "S M I T H", ["SMITH"], "")
        book.on_issue("c3", "find_member_id_by_email", '{"email": "p@q.r"}')
        note = book.on_output("c3", MISS)[1]
        self.assertIn("SMITH", note)
        self.assertNotIn("CHEN", note)  # a closed episode is never shown again

    def test_unrelated_reads_cancel_and_late_results(self) -> None:
        book = ledger()
        book.on_issue("o1", "get_order_details", '{"order_id": "O1"}')
        self.assertEqual(book.on_output("o1", '{"order_id": "O1"}'), ("", ""))
        book.on_issue("c1", "find_member_id_by_email", '{"email": "a@b.c"}')
        book.on_issue("c2", "find_member_id_by_email", '{"email": "d@e.f"}')
        book.on_issue("c3", "find_member_id_by_email", '{"email": "g@h.i"}')
        book.on_cancel("c3")
        book.on_output("c2", '"M1"')  # not JSON with a person id: unknown
        self.assertEqual(book.on_output("c1", "M1")[0], FOUND)
        self.assertEqual(book.on_output("c2", MISS), (NOT_FOUND, ""))  # late: its episode is closed

    def test_age_cap_and_note_cap(self) -> None:
        book = ledger(max_age_turns=2, max_notes=2)
        book.add_turn(1, "O L D", ["OLD"], "")
        book.add_turn(5, "N E W", ["NEW"], "")
        for index in range(3):
            book.on_issue(f"c{index}", "find_member_id_by_email", '{"email": "a@b.c"}')
            note = book.on_output(f"c{index}", MISS)[1]
            if index < 2:
                self.assertIn('"NEW"', note)
                self.assertNotIn('"OLD"', note)
            else:
                self.assertEqual(note, "")

    def test_digits_only_and_one_letter_spellings_are_not_listed(self) -> None:
        book = ledger()
        book.add_turn(1, "1 9 1 2 2", ["19122", "A"], "")
        book.on_issue("c1", "find_member_id_by_email", '{"email": "a@b.c"}')
        self.assertEqual(book.on_output("c1", MISS)[1], "")


class LedgerRelayTests(unittest.IsolatedAsyncioTestCase):
    def _relay(self, *, ledger_on: bool = True):
        settings = ResultHintSettings(enabled=True, auto_tools=True, failure_pattern=FAILURE, max_hints=6)
        book = ledger() if ledger_on else None
        reads = [t["name"] for t in INVENTED if t["name"].startswith("get_") or t["name"].startswith("find_")]
        hints = ResultHints(settings, message="[I3]", escalate_message="[I4]", read_tools=reads, ledger=book)
        results: list[tuple[str, int, str, bool]] = []
        logs: list[tuple[str, dict]] = []
        relay = ToolRelay(
            executor="wire",
            send_result=lambda *r: results.append(r),
            emit_calls=lambda item: None if isinstance(item, CallsItem) else None,
            log=lambda kind, **data: logs.append((kind, data)),
            result_hints=hints,
            batch_window_ms=1,
        )
        return relay, results, logs, book

    async def _call(self, relay: ToolRelay, call_id: str, name: str, arguments: str) -> None:
        import asyncio  # noqa: PLC0415

        relay.on_call({"call_id": call_id, "epoch": 1, "name": name, "arguments": arguments})
        await asyncio.sleep(0.01)

    async def test_the_note_follows_the_i3_note_and_an_unrelated_read_never_resets(self) -> None:
        relay, results, logs, book = self._relay()
        assert book is not None
        book.add_turn(1, "C H E N", ["CHEN"], "Spell your last name?")
        await self._call(
            relay, "c1", "find_member_id_by_name_postal", '{"first_name": "a", "last_name": "Hen", "postal_code": "1"}'
        )
        relay.on_function_output("c1", MISS)
        relay.take_ack()
        await self._call(relay, "o1", "get_order_details", '{"order_id": "O1"}')
        relay.on_function_output("o1", '{"order_id": "O1"}')
        relay.take_ack()
        await self._call(relay, "c2", "find_member_id_by_email", '{"email": "a@b.c"}')
        relay.on_function_output("c2", MISS)
        first, order, second = (r[2] for r in results)
        self.assertTrue(first.startswith(MISS + "\n\n[I3]\n[Ledger:"))
        self.assertEqual(order, '{"order_id": "O1"}')
        self.assertTrue(second.startswith(MISS + "\n\n[I4]\n[Ledger:"))  # the count did not reset
        self.assertIn("1) find_member_id_by_name_postal", second)
        classes = [d["output_class"] for k, d in logs if k == "identity_ledger"]
        self.assertEqual(classes, [NOT_FOUND, NOT_FOUND])

    async def test_off_is_todays_output(self) -> None:
        relay, results, logs, _ = self._relay(ledger_on=False)
        await self._call(relay, "c1", "find_member_id_by_email", '{"email": "a@b.c"}')
        relay.on_function_output("c1", MISS)
        self.assertEqual(results[0][2], MISS + "\n\n[I3]")
        self.assertEqual([k for k, _ in logs if k == "identity_ledger"], [])


class IdentityProfileTests(unittest.TestCase):
    def test_the_identity_arm_turns_on_the_ledger_and_keeps_collection(self) -> None:
        arm = load_delegation_config(PROFILES / "realtime_eval_identity.yaml")
        hints = arm.voice.normalization.tool_arguments.result_hints
        self.assertEqual((hints.ledger, hints.field_provenance, hints.max_hints), (True, False, 6))
        self.assertTrue(hints.auto_identification_tools)
        control = load_delegation_config(PROFILES / "realtime_eval_124acf7.yaml")
        self.assertFalse(control.voice.normalization.tool_arguments.result_hints.ledger)
        identity = load_delegation_config(PROFILES / "realtime_eval_identity.yaml").features
        self.assertEqual(
            {k for k, v in identity.items() if control.features[k] != v}, {"identity_ledger", "field_provenance"}
        )
        default = load_delegation_config(PROFILES / "realtime_eval.yaml").voice.normalization.tool_arguments
        self.assertEqual((default.result_hints.ledger, default.result_hints.max_hints), (True, 6))

    def test_identity_name_line_only_adds_its_own_sentence(self) -> None:
        base = dict(gc.load_gateway_config(GATEWAY).prompt_features)
        off = BackendTemplates(BACKEND_PROMPTS, prompt_features=base).render("backend_system", instructions="P")
        on = BackendTemplates(BACKEND_PROMPTS, prompt_features={**base, "identity_from_identifier": True}).render(
            "backend_system", instructions="P"
        )
        sentence = (
            "If the user gives an identifier that visibly contains a name (for example jane_doe_1234) and your\n"
            "instructions require a lookup by name, read that name back to the user and, once they confirm it,\n"
            "use it for the lookup.\n"
        )
        self.assertEqual(on.replace(sentence, ""), off)
        arm = gc.load_gateway_config(PACKAGE / "config" / "gateway.eval_identity_name.yaml").prompt_features
        self.assertEqual({k for k in arm if arm[k] != base[k]}, {"identity_from_identifier"})


if __name__ == "__main__":
    unittest.main()


# -- step 0 joins and the G2 comparison ----------------------------------------------------------------------


class JoinAndReplayTests(unittest.TestCase):
    def test_canonical_tools_hash_is_the_same_for_both_wire_shapes(self) -> None:
        from examples.frontend_delegation_hermes.core.prompt_features import tools_canonical_sha256  # noqa: PLC0415

        flat = tau2_session_update("telecom")["session"]["tools"]
        wrapped = [
            {"type": "function", "function": {k: v for k, v in tool.items() if k != "type"}} for tool in reversed(flat)
        ]
        self.assertEqual(tools_canonical_sha256(flat), tools_canonical_sha256(wrapped))
        self.assertNotEqual(tools_canonical_sha256(flat), tools_canonical_sha256(flat[1:]))

    def test_u_needs_the_complete_ordered_call_list_and_old_decisions_keep_the_old_rule(self) -> None:
        a, b = ("cancel", '{"id": "1"}'), ("transfer", '{"summary": "x"}')
        new = g2.Decision("d", "s", "r", *a, calls=[list(a), list(b)])
        self.assertEqual(g2.verdict(new, [[a, b], [a, b]], []), "pass")
        self.assertEqual(g2.verdict(new, [[a, b], [a]], []), "fail")
        self.assertEqual(g2.verdict(new, [[b, a]], []), "fail")
        old = g2.Decision("d", "s", "r", *a)
        self.assertEqual(g2.verdict(old, [[a, b], [a]], []), "pass")


class IdentityVoiceTests(unittest.IsolatedAsyncioTestCase):
    async def test_retail_miss_gets_the_ledger_after_the_recovery_note_through_the_voice_server(self) -> None:
        harness = DelegationHarness(profile_config("realtime_eval_identity.yaml"), transcripts=["my email please"])
        await harness.start(tau2_session_update("retail"))
        rules = harness.logged("session_rules")[-1]
        self.assertEqual(
            set(rules["identification_tools"]),
            {"find_user_id_by_email", "find_user_id_by_name_zip", "get_user_details"},
        )
        await harness.speak()
        await harness.wait_sent("delegate")
        harness.link.deliver("state", state="WORKING", epoch=1, run_id="run_1")
        for index, email in enumerate(("jo@example.com", "joe@example.com"), 1):
            harness.link.deliver(
                "tool.call", call_id=f"c{index}", epoch=1, run_id="run_1", name="find_user_id_by_email",
                arguments=f'{{"email": "{email}"}}',
            )  # fmt: skip
            await harness.wait_for("response.function_call_arguments.done", count=index)
            await harness.send(
                {
                    "type": "conversation.item.create",
                    "item": {"type": "function_call_output", "call_id": f"c{index}", "output": "Error: User not found"},
                }
            )
            await harness.send({"type": "response.create"})
        second = (await harness.wait_sent("tool.result", count=2))["output"]
        self.assertIn("[Not found again.", second)
        self.assertIn("[Identification so far in this call (not found yet):", second)
        self.assertIn("find_user_id_by_name_zip(first_name, last_name, zip)", second)
        self.assertEqual([d["output_class"] for d in harness.logged("identity_ledger")], ["not_found", "not_found"])
        await harness.close()


class DefaultsTests(unittest.TestCase):
    def test_the_submission_pair_runs_every_fix_and_the_124acf7_pair_none(self) -> None:
        config = PACKAGE / "config"
        new = {"situational_facts", "identity_from_identifier"}
        on = gc.load_gateway_config(config / "gateway.eval.yaml").prompt_features
        off = gc.load_gateway_config(GATEWAY).prompt_features
        # The telecom domain note (analysis/2026-10-11_fdh-telecom-gpt55-deep) came after 124acf7 too.
        self.assertEqual({k for k in on if on[k] != off[k]}, {*new, "domain_notes"})
        self.assertTrue(all(gc.load_gateway_config(config / "gateway.yaml").prompt_features[k] for k in new))
        for control in ("gateway.eval_control.yaml", "gateway.baseline.yaml", "gateway.identity_control.yaml"):
            with self.subTest(control=control):
                features = gc.load_gateway_config(config / control).prompt_features
                self.assertFalse(any(features[k] for k in new))
