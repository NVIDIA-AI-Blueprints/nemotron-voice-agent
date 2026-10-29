# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D100, D101, D102, D103, D107

"""R4, ``delegation.guards.acknowledge_caller_while_working``: an acknowledgement line, not silence.

A backchannel ("okay") while the backend works, with nothing else pending, gets a short line from
``acknowledgement_lines``; a repeated delegated filler while WORKING is replaced from the same pool instead
of dropped. Off by default, capped per backend run, spaced in time, and only for a language with a lexicon.
"""

from __future__ import annotations

import os
import unittest
from dataclasses import replace
from typing import Any
from unittest import mock

from _fda_voice_fakes import PACKAGE, PROFILES, DelegationHarness, benchmark_session_update, profile_config

from examples.frontend_delegation_agent.core.config import (
    ACKNOWLEDGE_KEYS,
    AcknowledgeSettings,
    DelegationConfigError,
    load_delegation_config,
)
from examples.frontend_delegation_agent.core.frontend.decider import DecisionContext, RuleDecider
from examples.frontend_delegation_agent.core.frontend.delegate_tool import Decision
from examples.frontend_delegation_agent.core.frontend.prompts import load_prompts

TASK = {"delegate": True, "filler_text": "Sure, one moment.", "request": "task"}
SILENT = {"delegate": False, "filler_text": ""}
LINES = ("Thanks for your patience.", "Just a moment longer.", "Bear with me.")


def config(*, enabled: bool = True, **settings: Any) -> Any:
    base = profile_config("realtime_eval.yaml")
    acknowledge = AcknowledgeSettings(enabled=enabled, **{"min_interval_s": 0.0, **settings})
    return replace(base, delegation=replace(base.delegation, acknowledge=acknowledge))


def german(cfg: Any) -> Any:
    voice = cfg.voice
    asr = replace(voice.asr, endpoint=replace(voice.asr.endpoint, language_code="de-DE"))
    return replace(cfg, voice=replace(voice, asr=asr))


class AcknowledgeTurnTests(unittest.IsolatedAsyncioTestCase):
    async def _working(self, harness: DelegationHarness, *, state: str = "WORKING") -> None:
        """A first turn is delegated, the backend works, and its filler (if spoken) has been played."""
        await harness.start(benchmark_session_update("retail"))
        await harness.speak()
        await harness.wait_sent("delegate")
        if state == "WORKING":
            harness.link.deliver("state", state="WORKING", epoch=1, run_id="run_1")
        if not harness.config.delegation.silent_frontend:
            await harness.wait_for("response.done")
        await harness.idle(1500)

    def acknowledged(self, harness: DelegationHarness) -> list[dict[str, Any]]:
        return [g for g in harness.logged("delegation_guard") if g["guard"] == "acknowledge_caller"]

    async def test_okay_while_working_gets_an_acknowledgement_line(self) -> None:
        harness = DelegationHarness(config(), decider=RuleDecider([TASK, SILENT]), transcripts=["where is it", "Okay."])
        await self._working(harness)
        await harness.speak()
        await harness.wait_for("response.done", count=2)
        self.assertEqual(harness.transcripts()[-1], LINES[0])
        self.assertEqual(len(self.acknowledged(harness)), 1)
        self.assertEqual(len(harness.link.of_type("delegate")), 1)  # never sent to the backend
        await harness.close()

    async def test_off_by_default_stays_silent(self) -> None:
        harness = DelegationHarness(
            config(enabled=False), decider=RuleDecider([TASK, SILENT]), transcripts=["where is it", "Okay."]
        )
        await self._working(harness)
        await harness.speak()
        await harness.idle(800)
        self.assertEqual(len(harness.of_type("response.done")), 1)
        self.assertEqual(self.acknowledged(harness), [])
        await harness.close()

    async def test_not_while_idle_not_on_hold_not_without_a_lexicon_not_with_the_silent_frontend(self) -> None:
        cases = {
            "idle": (config(), "Okay.", "IDLE"),
            "hold": (config(), "Hold on.", "WORKING"),
            "no_lexicon": (german(config()), "Okay.", "WORKING"),
            "silent_frontend": (
                replace(config(), delegation=replace(config().delegation, silent_frontend=True)),
                "Okay.",
                "WORKING",
            ),
        }
        for name, (cfg, text, state) in cases.items():
            with self.subTest(case=name):
                harness = DelegationHarness(cfg, decider=RuleDecider([TASK, SILENT]), transcripts=["where is it", text])
                await self._working(harness, state=state)
                await harness.speak()
                await harness.idle(800)
                self.assertEqual(self.acknowledged(harness), [])
                await harness.close()

    async def test_capped_per_backend_run_and_rotated(self) -> None:
        harness = DelegationHarness(
            config(max_per_run=2),
            decider=RuleDecider([TASK, SILENT, SILENT, SILENT]),
            transcripts=["where is it", "Okay.", "Mm-hmm.", "Sure."],
        )
        await self._working(harness)
        for count in (2, 3):
            await harness.speak()
            await harness.wait_for("response.done", count=count)
            await harness.idle(1500)
        await harness.speak()
        await harness.idle(800)
        self.assertEqual(harness.transcripts()[-2:], list(LINES[:2]))
        self.assertEqual(len(self.acknowledged(harness)), 2)
        await harness.close()

    async def test_minimum_interval_between_lines(self) -> None:
        harness = DelegationHarness(
            config(min_interval_s=3600.0),
            decider=RuleDecider([TASK, SILENT, SILENT]),
            transcripts=["where is it", "Okay.", "Mm-hmm."],
        )
        await self._working(harness)
        await harness.speak()
        await harness.wait_for("response.done", count=2)
        await harness.idle(1500)
        await harness.speak()
        await harness.idle(800)
        self.assertEqual(len(self.acknowledged(harness)), 1)
        await harness.close()

    async def test_minimum_gap_after_agent_speech(self) -> None:
        harness = DelegationHarness(
            config(min_since_agent_s=3600.0), decider=RuleDecider([TASK, SILENT]), transcripts=["where is it", "Okay."]
        )
        await self._working(harness)
        await harness.speak()
        await harness.idle(800)
        self.assertEqual(self.acknowledged(harness), [])
        await harness.close()

    async def test_repeated_filler_while_working_is_replaced_not_dropped(self) -> None:
        for enabled, expected in ((True, "replaced"), (False, "dropped")):
            with self.subTest(enabled=enabled):
                harness = DelegationHarness(
                    config(enabled=enabled), decider=RuleDecider([TASK, TASK]), transcripts=["where is it", "and now?"]
                )
                await self._working(harness)
                await harness.speak()
                await harness.wait_sent("delegate", count=2)
                await harness.idle(800)
                deduped = harness.logged("filler_deduped")[-1]
                self.assertEqual(deduped["action"], expected)
                if enabled:
                    self.assertEqual((deduped["new"], deduped["reason"]), (LINES[0], "acknowledgement"))
                await harness.close()


class AcknowledgeRuleTests(unittest.IsolatedAsyncioTestCase):
    async def test_never_for_a_backend_question_a_spoken_decision_or_with_something_pending(self) -> None:
        harness = DelegationHarness(config(max_per_run=10), decider=RuleDecider([TASK]), transcripts=["where is it"])
        await AcknowledgeTurnTests._working(AcknowledgeTurnTests(), harness)
        manager = harness.manager
        assert manager is not None
        context = DecisionContext(system_prompt="", messages=[], user_text="okay", state="WORKING")
        silent = Decision(delegate=False, filler_text="")
        self.assertEqual(manager._acknowledge(silent, context, None).filler_text, LINES[0])
        self.assertIsNone(manager._acknowledge(silent, replace(context, backend_asked_question=True), None))
        self.assertIsNone(manager._acknowledge(Decision(delegate=False, filler_text="Sure."), context, None))
        self.assertIsNone(manager._acknowledge(replace(silent, delegate=True), context, None))
        self.assertIsNone(manager._acknowledge(silent, replace(context, user_text="what about my refund"), None))
        manager._awaiting = 1  # another utterance is on its way
        self.assertIsNone(manager._acknowledge(silent, context, None))
        manager._awaiting = 0
        await harness.close()


class AcknowledgeConfigTests(unittest.TestCase):
    def test_off_in_every_profile_and_absent_from_the_effective_config(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("FDA_ACKNOWLEDGE_CALLER", None)
            for path in sorted(PROFILES.glob("*.yaml")):
                with self.subTest(profile=path.name):
                    loaded = load_delegation_config(path)
                    self.assertFalse(loaded.delegation.acknowledge.enabled)
                    guards = loaded.effective["delegation"]["guards"]
                    self.assertFalse(set(ACKNOWLEDGE_KEYS) & set(guards))

    def test_the_variable_turns_it_on_in_every_profile(self) -> None:
        with mock.patch.dict(os.environ, {"FDA_ACKNOWLEDGE_CALLER": "true"}):
            for path in sorted(PROFILES.glob("*.yaml")):
                with self.subTest(profile=path.name):
                    loaded = load_delegation_config(path)
                    self.assertTrue(loaded.delegation.acknowledge.enabled)
                    self.assertEqual(loaded.delegation.acknowledge.max_per_run, 2)
                    self.assertIn("acknowledge_caller_while_working", loaded.effective["delegation"]["guards"])

    def test_invalid_values_fail(self) -> None:
        with mock.patch.dict(os.environ, {"FDA_ACKNOWLEDGE_CALLER": "maybe"}), self.assertRaises(DelegationConfigError):
            load_delegation_config(PROFILES / "realtime_eval.yaml")

    def test_every_shipped_frontend_profile_has_the_lines(self) -> None:
        for path in sorted((PACKAGE / "config" / "prompts").glob("*/frontend.yaml")):
            with self.subTest(profile=path.parent.name):
                lines = load_prompts(path).lines("acknowledgement_lines")
                self.assertEqual(lines, LINES)
                others = {
                    line
                    for key in ("status_proactive_lines", "filler_alternatives")
                    for line in load_prompts(path).lines(key)
                }
                self.assertFalse(set(lines) & others)


if __name__ == "__main__":
    unittest.main()
