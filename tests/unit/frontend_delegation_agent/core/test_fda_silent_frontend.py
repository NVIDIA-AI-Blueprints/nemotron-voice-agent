# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D100, D101, D102, D103, D107

"""``delegation.silent_frontend`` (single-turn evaluations): fillers logged, silent requests delegated."""

from __future__ import annotations

import asyncio
import os
import unittest
from dataclasses import replace
from unittest import mock

from _fda_voice_fakes import PROFILES, DelegationHarness, bench_config, benchmark_session_update

from examples.frontend_delegation_agent.core.config import ProactiveStatusSettings, load_delegation_config
from examples.frontend_delegation_agent.core.frontend.decider import (
    SILENT_DELEGATED,
    DecisionContext,
    RuleDecider,
    apply_guards,
)
from examples.frontend_delegation_agent.core.frontend.delegate_tool import Decision
from examples.frontend_delegation_agent.core.frontend.turn_class import classify_turn

TASK = {"delegate": True, "filler_text": "Sure, one moment.", "request": "task"}
DIRECT = {"delegate": False, "filler_text": "Paris is the capital of France.", "request": "task"}
SILENT = {"delegate": False, "filler_text": "", "request": "task"}
REQUEST = "Describe the winning play in the final seconds of a championship game as a sports commentator"
EVAL = load_delegation_config(PROFILES / "realtime_eval.yaml").delegation
EN = EVAL.liveness.lexicons["en"]
ANSWER = "Your order 1 shipped on May third."


def silent(config=None, *, proactive_after_s: float = 0.0):
    config = config or bench_config()
    config = replace(config, delegation=replace(config.delegation, silent_frontend=True))
    if proactive_after_s:
        output = replace(config.output, proactive_status=ProactiveStatusSettings(True, proactive_after_s, 1))
        config = replace(config, output=output)
    return config


def load_eval_profile(value: str | None):
    env = {key: val for key, val in os.environ.items() if key != "FDA_SILENT_FRONTEND"}
    if value is not None:
        env["FDA_SILENT_FRONTEND"] = value
    with mock.patch.dict(os.environ, env, clear=True):
        return load_delegation_config(PROFILES / "realtime_eval.yaml")


class SilentFrontendConfigTests(unittest.TestCase):
    def test_off_by_default_and_absent_from_the_effective_settings(self) -> None:
        config = load_eval_profile(None)
        self.assertFalse(config.delegation.silent_frontend)
        self.assertTrue(config.delegation.speak_when_delegating)
        self.assertNotIn("silent_frontend", config.effective["delegation"])
        self.assertEqual(config.config_hash, load_eval_profile("false").config_hash)

    def test_the_environment_turns_it_on_and_changes_the_config_hash(self) -> None:
        on = load_eval_profile("true")
        self.assertTrue(on.delegation.silent_frontend)
        self.assertTrue(on.effective["delegation"]["silent_frontend"])
        self.assertNotEqual(on.config_hash, load_eval_profile(None).config_hash)


def guard(decision: Decision, text: str, state: str = "IDLE", *, classify: bool = True, **flags: bool) -> Decision:
    context = DecisionContext(system_prompt="", messages=[], user_text=text, state=state)
    return apply_guards(
        decision,
        context,
        backend_question_needs_delegate=True,
        backchannel_words=tuple(EVAL.backchannel_words),
        turn_class=classify_turn(text, EN) if classify else None,
        liveness_reply=True,
        question_never_silent=True,
        ping_as_status=True,
        **flags,
    )


class SilentTurnGuardTests(unittest.TestCase):
    """prompt-profiles-plan.md section 5.1: with the silent frontend, a silent request is delegated."""

    def test_a_silent_request_is_delegated_with_or_without_a_turn_class(self) -> None:
        for text in (REQUEST, "I have a hard time falling asleep . Is there any type of music that can help"):
            for classify in (True, False):
                with self.subTest(text=text, classify=classify):
                    out = guard(Decision(False, ""), text, classify=classify, delegate_silent=True)
                    self.assertEqual(
                        (out.delegate, out.filler_text, out.request, out.repair), (True, "", "task", SILENT_DELEGATED)
                    )

    def test_a_silent_request_after_a_contract_repair_is_delegated(self) -> None:
        out = guard(Decision(False, "", repair="repaired"), REQUEST, delegate_silent=True)
        self.assertEqual((out.delegate, out.repair), (True, SILENT_DELEGATED))

    def test_backchannels_pauses_and_empty_turns_stay_silent(self) -> None:
        for text in ("okay", "Okay.", "thanks", "hold on", "Give me a moment.", "", "   "):
            for classify in (True, False):
                with self.subTest(text=text, classify=classify):
                    if not classify and text in ("hold on", "Give me a moment."):
                        continue  # without a lexicon a pause is not recognised
                    out = guard(Decision(False, ""), text, classify=classify, delegate_silent=True)
                    self.assertFalse(out.delegate)

    def test_a_backchannel_the_model_spoke_is_silenced_not_delegated(self) -> None:
        out = guard(Decision(True, "One moment.", "task"), "okay", delegate_silent=True)
        self.assertEqual((out.delegate, out.filler_text, out.repair), (False, "", "guard_backchannel"))

    def test_a_ping_goes_through_r1_as_today(self) -> None:
        out = guard(Decision(False, ""), "Are you still there?", delegate_silent=True)
        self.assertEqual(out.repair, "liveness_reply")

    def test_nothing_fires_while_the_backend_works(self) -> None:
        out = guard(Decision(False, ""), REQUEST, "WORKING", delegate_silent=True)
        self.assertFalse(out.delegate)

    def test_a_decision_that_speaks_or_delegates_is_unchanged(self) -> None:
        for decision in (Decision(False, "Paris."), Decision(True, "One moment.", "task")):
            with self.subTest(decision=decision):
                self.assertEqual(guard(decision, REQUEST, delegate_silent=True), decision)

    def test_switch_off_changes_nothing(self) -> None:
        for classify in (True, False):
            with self.subTest(classify=classify):
                self.assertEqual(guard(Decision(False, ""), REQUEST, classify=classify), Decision(False, ""))


class SilentFrontendTurnTests(unittest.IsolatedAsyncioTestCase):
    async def test_off_speaks_the_delegated_filler(self) -> None:
        harness = DelegationHarness(decider=RuleDecider([TASK]), transcripts=["check order 1"])
        await harness.start(benchmark_session_update("mock"))
        await harness.speak()
        await harness.wait_sent("delegate")
        await harness.wait_for("response.done")
        self.assertEqual(harness.transcripts(), ["Sure, one moment."])
        self.assertEqual(harness.logged("filler_silenced"), [])
        await harness.close()

    async def test_on_the_first_response_is_the_backend_answer(self) -> None:
        harness = DelegationHarness(silent(), decider=RuleDecider([TASK]), transcripts=["check order 1"])
        await harness.start(benchmark_session_update("mock"))
        await harness.speak()
        await harness.wait_sent("delegate")
        harness.link.deliver("state", state="WORKING", epoch=1, run_id="run_1")
        await harness.idle(300)
        self.assertEqual(harness.of_type("response.created"), [])
        silenced = harness.logged("filler_silenced")
        self.assertEqual([(e["text"], e["reason"]) for e in silenced], [("Sure, one moment.", "silent_frontend")])
        harness.link.deliver(
            "answer", answer_id="a1", run_id="run_1", epoch=1, kind="hermes", text=ANSWER, turn_ids=[1]
        )
        await harness.wait_for("response.done")
        self.assertEqual(len(harness.of_type("response.done")), 1)
        self.assertEqual(harness.transcripts(), [ANSWER])
        await harness.close()

    async def test_on_a_direct_answer_is_still_spoken(self) -> None:
        harness = DelegationHarness(silent(), decider=RuleDecider([DIRECT]), transcripts=["capital of france"])
        await harness.start(benchmark_session_update("mock"))
        await harness.speak()
        await harness.wait_for("response.done")
        self.assertEqual(harness.transcripts(), ["Paris is the capital of France."])
        self.assertEqual(harness.link.of_type("delegate"), [])
        self.assertEqual(harness.logged("filler_silenced"), [])
        await harness.close()

    async def test_on_a_silent_request_is_delegated_and_the_backend_answer_is_first(self) -> None:
        harness = DelegationHarness(silent(), decider=RuleDecider([SILENT]), transcripts=[REQUEST])
        await harness.start(benchmark_session_update("mock"))
        await harness.speak()
        await harness.wait_sent("delegate")
        harness.link.deliver("state", state="WORKING", epoch=1, run_id="run_1")
        await harness.idle(300)
        self.assertEqual(harness.of_type("response.created"), [])
        harness.link.deliver(
            "answer", answer_id="a1", run_id="run_1", epoch=1, kind="hermes", text=ANSWER, turn_ids=[1]
        )
        await harness.wait_for("response.done")
        self.assertEqual(len(harness.of_type("response.done")), 1)
        self.assertEqual(harness.transcripts(), [ANSWER])
        guards = harness.logged("delegation_guard")
        self.assertEqual([event["guard"] for event in guards], [SILENT_DELEGATED])
        await harness.close()

    async def test_off_a_silent_request_stays_silent(self) -> None:
        harness = DelegationHarness(decider=RuleDecider([SILENT]), transcripts=[REQUEST])
        await harness.start(benchmark_session_update("mock"))
        await harness.speak()
        await harness.idle(300)
        self.assertEqual(harness.link.of_type("delegate"), [])
        self.assertEqual(harness.of_type("response.created"), [])
        await harness.close()

    async def test_on_no_proactive_status_is_spoken_while_working(self) -> None:
        harness = DelegationHarness(
            silent(proactive_after_s=0.3), decider=RuleDecider([TASK]), transcripts=["check order 1"]
        )
        await harness.start(benchmark_session_update("mock"))
        await harness.speak()
        await harness.wait_sent("delegate")
        harness.link.deliver("state", state="WORKING", epoch=1, run_id="run_1")
        await asyncio.sleep(1.0)
        await harness.idle(300)
        self.assertEqual(harness.of_type("response.created"), [])
        self.assertEqual(len(harness.logged("proactive_status_silenced")), 1)
        await harness.close()


if __name__ == "__main__":
    unittest.main()
