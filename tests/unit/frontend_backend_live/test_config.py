# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D103, D105, D107
"""The example's settings in its registry entry: the frontend adapter and its block, the core, the UI bot."""

import unittest

from examples.frontend_backend_live.config_manager.schema import ExampleConfig, LiveConfig, SpeechConfig
from examples.frontend_backend_live.delegation_core.config import CoreSettings
from tests.unit.core_helpers import CORE_TEXTS, EXAMPLE_SETTINGS


def example(**cascade) -> dict:
    """Return minimal settings: the cascade adapter with ``cascade`` as its block, a core, and the UI bot's channel."""
    return {
        "frontend": {"adapter": "cascade", "cascade": {"slot": "llm", **cascade}},
        "core": {},
        "ui_bot": {"channel": "cafe"},
    }


def cascade_of(config: ExampleConfig, label: str = "delegation_core") -> LiveConfig:
    """Parse the cascade's settings as the root does, with the core's display label handed in."""
    return LiveConfig.model_validate({**config.frontend.block, "display_label": label})


class ConfigTests(unittest.TestCase):
    def test_the_shipped_settings_agree_with_the_core_and_the_schema(self):
        config = ExampleConfig.model_validate(EXAMPLE_SETTINGS)
        self.assertEqual((config.frontend.adapter, list(config.frontend.blocks)), ("cascade", ["cascade"]))
        cascade = cascade_of(config)
        self.assertEqual(cascade.style_of("a-model-not-listed"), "json")
        deadline = CoreSettings.parse(config.core, CORE_TEXTS).delegation.client_tool_timeout_seconds
        self.assertLess(cascade.delegation.app_wait_seconds, deadline)
        self.assertEqual(cascade.speech, SpeechConfig())

    def test_a_bad_setting_is_refused_with_its_place(self):
        for bad, message in (
            ({**example(), "oops": 1}, "oops\n.*Extra inputs are not permitted"),
            ({"frontend": {"adapter": "cascade"}, "core": {}}, "no frontend.cascade block"),
            ({**example(), "ui_bot": {"core": {}}}, "ui_bot.channel\n.*Field required"),
            (example(display_label="screen"), "display_label comes from the core's delegation.display_label"),
            (example(client_tool_timeout_seconds=30), "comes from the core's delegation.client_tool_timeout_seconds"),
        ):
            with self.subTest(bad=bad), self.assertRaisesRegex(ValueError, message):
                ExampleConfig.model_validate(bad)
        for cascade, message in (
            ({"oops": 1}, "frontend.cascade\noops\n.*Extra inputs are not permitted"),
            ({"slot": ""}, "slot\n.*at least 1 character"),
            ({"talker_style": "tools"}, "talker_style\n.*valid dictionary"),
            ({"reliability": {"provider_retries": "two"}}, "reliability.provider_retries\n.*valid integer"),
            ({"reliability": {"talker_attempt_timeout_seconds": 0}}, "greater than 0"),
            ({"memory": {"compact_at": 1.5}}, "memory.compact_at\n.*less than or equal to 1"),
            ({"speech": {"interrupt_min_words": 0}}, "speech.interrupt_min_words\n.*greater than or equal to 1"),
            ({"tts_preroll_ms": 2001}, "tts_preroll_ms\n.*less than or equal to 2000"),
            ({"delegation": {"answers": {"max_tries": 0}}}, "answers.max_tries\n.*greater than or equal to 1"),
            ({"delegation": {"answers": {"max_tries": 1.5}}}, "answers.max_tries\n.*valid integer"),
            ({"delegation": {"answers": {"requeued_as_said": "yes"}}}, "answers.requeued_as_said\n.*valid boolean"),
            ({"delegation": {"answers": {"lead_in_skip": 1}}}, "answers.lead_in_skip\n.*valid boolean"),
            ({"delegation": {"answers": {"oops": True}}}, "answers.oops\n.*Extra inputs are not permitted"),
            ({"delegation": {"answers": {"clear_ended_on_cut": "on"}}}, "answers.clear_ended_on_cut\n.*valid boolean"),
            ({"delegation": {"answers": {"worded_carried": True}}}, "answers.worded_carried\n.*Extra inputs are not"),
        ):
            with self.subTest(cascade=cascade), self.assertRaisesRegex(ValueError, message):
                cascade_of(ExampleConfig.model_validate(example(**cascade)))

    def test_the_answers_block_has_its_defaults_and_a_null_cap_means_no_limit(self):
        answers = cascade_of(ExampleConfig.model_validate(example())).delegation.answers
        self.assertEqual(answers.max_tries, 3)
        self.assertTrue(answers.requeued_as_said and answers.unheard_rest_once and answers.lead_in_skip)
        self.assertTrue(answers.unheard_questions)
        self.assertFalse(answers.clear_ended_on_cut)
        config = ExampleConfig.model_validate(example(delegation={"answers": {"max_tries": None}}))
        self.assertIsNone(cascade_of(config).delegation.answers.max_tries)

    def test_values_override_the_defaults_and_null_keeps_a_parts(self):
        config = ExampleConfig.model_validate(
            example(
                reliability={"provider_retries": 0, "talker_attempt_timeout_seconds": 3},
                delegation={"app_wait_seconds": None, "progress_after_seconds": None, "progress_max_per_delegation": 1},
                memory=None,
                talker_style=None,
            )
        )
        cascade = cascade_of(config)
        self.assertEqual(cascade.reliability.provider_retries, 0)
        self.assertEqual(cascade.reliability.talker_attempt_timeout_seconds, 3.0)
        self.assertIsNone(cascade.delegation.app_wait_seconds)
        self.assertIsNone(cascade.delegation.progress_after_seconds)
        self.assertEqual(cascade.delegation.progress_max_per_delegation, 1)
        self.assertEqual(cascade.memory, LiveConfig(slot="s").memory)
        self.assertEqual(cascade.talker_style, {})
        self.assertEqual(cascade_of(config, label="screen").display_label, "screen")
