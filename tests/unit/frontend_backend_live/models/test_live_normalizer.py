# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D103

"""The turn normalizer: which turns it rewrites, what it keeps, and what the talker and the backend see."""

import asyncio
import unittest

from examples.frontend_backend_live.cascade.normalizer import TurnNormalizer
from examples.frontend_backend_live.config_manager.loader import ConfigError
from examples.frontend_backend_live.config_manager.schema import LiveConfig, NormalizerConfig
from examples.frontend_backend_live.models.decision import TalkerDecision
from examples.frontend_backend_live.prompts import get_prompt_set
from tests.unit.frontend_backend_live.cascade.doubles import Call, ScriptedFrontend, conversation

ON = NormalizerConfig(enabled=True)
PROMPTS = get_prompt_set("default")
NEWEST_TURN = PROMPTS.normalizer_prompt.newest_turn


def speak(text):
    return TalkerDecision("speak", text, {})


class NormalizerTests(unittest.IsolatedAsyncioTestCase):
    async def test_the_newest_turn_is_framed_once_after_the_history(self):
        frontend = ScriptedFrontend([speak("my user id is mia_li_3668")])
        history = [
            {"role": "assistant", "content": "What is your user ID?"},
            {"role": "user", "content": "m i a underscore l i three six six eight"},
        ]
        text, diagnostics = await TurnNormalizer(frontend, PROMPTS, ON).normalize(history, history[-1]["content"])
        self.assertEqual(text, "my user id is mia_li_3668")
        self.assertEqual(diagnostics["outcome"], "changed")
        sent = frontend.calls[0]["history"]
        self.assertEqual(sent[0], history[0])
        self.assertEqual(sent[-1]["content"], NEWEST_TURN + "m i a underscore l i three six six eight")
        self.assertEqual(len(sent), 2)
        self.assertEqual(frontend.calls[0]["actions"], ("speak",))
        self.assertEqual(frontend.calls[0]["instructions"], PROMPTS.normalizer_prompt.instructions)

    async def test_failures_and_empty_or_echoed_answers_keep_the_turn_as_heard(self):
        heard = "two bags"
        for script, outcome in (
            ([RuntimeError("down"), RuntimeError("down")], "failed"),
            ([speak("")], "empty"),
            ([TalkerDecision("delegate", "", {"recovery": "invalid_json"})], "empty"),
            ([speak("two bags")], "unchanged"),
        ):
            text, diagnostics = await TurnNormalizer(ScriptedFrontend(script), PROMPTS, ON).normalize([], heard)
            self.assertEqual((text, diagnostics["outcome"]), (heard, outcome))

    async def test_a_slow_normalization_keeps_the_turn_as_heard(self):
        class Slow(ScriptedFrontend):
            async def complete_once(self, *args):
                await asyncio.sleep(5)

        config = NormalizerConfig(enabled=True, timeout_seconds=0.05)
        text, diagnostics = await TurnNormalizer(Slow([]), PROMPTS, config).normalize([], "two bags")
        self.assertEqual((text, diagnostics["outcome"]), ("two bags", "failed"))


class SessionTests(Call):
    """The normalizer in a call, on the cascade's own doubles (a scripted delegation port, no core)."""

    def enable(self, config=ON):
        self.session.agent.normalizer = TurnNormalizer(self.frontend, self.session.agent.prompts, config)

    async def test_the_talker_and_the_delegation_see_the_normalized_turn(self):
        heard = "it's m i a underscore l i three six six eight"
        await self.call(
            [speak("it's Q91_T"), TalkerDecision("delegate", "One moment.", {})],
            seed=({"role": "assistant", "content": "What is your user ID?"},),
        )
        self.enable()
        await self.turn(heard)
        await self.said(1)
        self.assertEqual(conversation(self.frontend.calls[1])[-1]["content"], "it's Q91_T")  # the talker's read
        self.assertIn(("user", "it's Q91_T"), self.memory())
        delegated = str(self.port.requests[0])
        self.assertIn("Q91_T", delegated)
        self.assertNotIn("underscore", delegated)
        completed = [fields for kind, fields in self.events if kind == "normalizer.completed"]
        self.assertEqual((completed[0]["heard"], completed[0]["outcome"]), (heard, "changed"))

    async def test_an_enabled_normalizer_reads_every_turn_and_an_echo_keeps_it_as_heard(self):
        await self.call([speak("I want to change my flight."), speak("Sure, what would you like to change?")])
        self.enable()
        await self.turn("I want to change my flight.")
        self.assertEqual([call["role"] for call in self.frontend.calls], ["normalizer", "talker"])
        completed = [fields for kind, fields in self.events if kind == "normalizer.completed"]
        self.assertEqual([fields["outcome"] for fields in completed], ["unchanged"])
        self.assertEqual(conversation(self.frontend.calls[1])[-1]["content"], "I want to change my flight.")
        self.assertEqual(self.spoken, ["Sure, what would you like to change?"])


class ConfigTests(unittest.TestCase):
    def test_the_normalizer_section_is_optional_and_validated(self):
        bare = {"provider": "p", "slot": "llm"}  # the frontend.cascade block
        self.assertFalse(LiveConfig.parse(bare).normalizer.enabled)
        config = LiveConfig.parse({**bare, "normalizer": {"enabled": True, "timeout_seconds": 2}})
        self.assertEqual((config.normalizer.enabled, config.normalizer.timeout_seconds), (True, 2.0))
        with self.assertRaises(ConfigError):
            LiveConfig.parse({**bare, "normalizer": {"enabled": "yes"}})
        with self.assertRaises(ConfigError):
            LiveConfig.parse({**bare, "normalizer": {"model": "x"}})
