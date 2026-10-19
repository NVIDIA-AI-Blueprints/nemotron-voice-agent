# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D100, D101, D102, D103, D107

"""``delegation.silent_frontend`` (VoiceBench): delegated fillers and proactive status are logged, never spoken."""

from __future__ import annotations

import asyncio
import os
import unittest
from dataclasses import replace
from unittest import mock

from _fdh_voice_fakes import PROFILES, DelegationHarness, tau2_session_update, tau3_config

from examples.frontend_delegation_hermes.core.config import ProactiveStatusSettings, load_delegation_config
from examples.frontend_delegation_hermes.core.frontend.decider import RuleDecider

TASK = {"delegate": True, "filler_text": "Sure, one moment.", "request": "task"}
DIRECT = {"delegate": False, "filler_text": "Paris is the capital of France.", "request": "task"}
ANSWER = "Your order 1 shipped on May third."


def silent(config=None, *, proactive_after_s: float = 0.0):
    config = config or tau3_config()
    config = replace(config, delegation=replace(config.delegation, silent_frontend=True))
    if proactive_after_s:
        output = replace(config.output, proactive_status=ProactiveStatusSettings(True, proactive_after_s, 1))
        config = replace(config, output=output)
    return config


def load_eval_profile(value: str | None):
    env = {key: val for key, val in os.environ.items() if key != "FDH_SILENT_FRONTEND"}
    if value is not None:
        env["FDH_SILENT_FRONTEND"] = value
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


class SilentFrontendTurnTests(unittest.IsolatedAsyncioTestCase):
    async def test_off_speaks_the_delegated_filler(self) -> None:
        harness = DelegationHarness(decider=RuleDecider([TASK]), transcripts=["check order 1"])
        await harness.start(tau2_session_update("mock"))
        await harness.speak()
        await harness.wait_sent("delegate")
        await harness.wait_for("response.done")
        self.assertEqual(harness.transcripts(), ["Sure, one moment."])
        self.assertEqual(harness.logged("filler_silenced"), [])
        await harness.close()

    async def test_on_the_first_response_is_the_backend_answer(self) -> None:
        harness = DelegationHarness(silent(), decider=RuleDecider([TASK]), transcripts=["check order 1"])
        await harness.start(tau2_session_update("mock"))
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
        await harness.start(tau2_session_update("mock"))
        await harness.speak()
        await harness.wait_for("response.done")
        self.assertEqual(harness.transcripts(), ["Paris is the capital of France."])
        self.assertEqual(harness.link.of_type("delegate"), [])
        self.assertEqual(harness.logged("filler_silenced"), [])
        await harness.close()

    async def test_on_no_proactive_status_is_spoken_while_working(self) -> None:
        harness = DelegationHarness(
            silent(proactive_after_s=0.3), decider=RuleDecider([TASK]), transcripts=["check order 1"]
        )
        await harness.start(tau2_session_update("mock"))
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
