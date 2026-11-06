# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D100, D101, D102, D103, D105, D107
"""The example's settings in its registry entry: the frontend adapter and its block, the core, the UI bot."""

import os
import re
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from examples.frontend_backend_live.config_manager.schema import (
    DelegationConfig,
    ExampleConfig,
    LiveConfig,
    SpeechConfig,
)
from examples.frontend_backend_live.delegation_core.config import CoreSettings
from examples.frontend_backend_live.prompts import load_prompts
from tests.unit.core_helpers import CORE_TEXTS, EXAMPLE_SETTINGS, merged

CASCADE = {"slot": "llm"}
UI_BOT = {"channel": "cafe"}


def example(**cascade) -> dict:
    """Minimal settings: the cascade adapter with ``cascade`` as its block, a core, and the UI bot's channel."""
    return {"frontend": {"adapter": "cascade", "cascade": {**CASCADE, **cascade}}, "core": {}, "ui_bot": UI_BOT}


def cascade_of(config: ExampleConfig, label: str = "delegation_core") -> LiveConfig:
    """The cascade's settings as the root parses them, with the core's display label handed in."""
    return LiveConfig.model_validate({**config.frontend.block, "display_label": label})


class ShippedSettingsTests(unittest.TestCase):
    def test_the_shipped_settings_name_the_cascade_and_its_settings(self):
        config = ExampleConfig.model_validate(EXAMPLE_SETTINGS)
        self.assertEqual((config.frontend.adapter, list(config.frontend.blocks)), ("cascade", ["cascade"]))
        self.assertEqual(config.ui_bot.channel, "cafe")
        cascade = cascade_of(config)
        self.assertEqual((cascade.slot, cascade.prompt), ("llm", "generic"))
        self.assertEqual(LiveConfig(slot="s").prompt, cascade.prompt)  # the schema's default agrees
        self.assertEqual(cascade.style_of("nemotron-lightning"), "json")  # Lightning makes the JSON decision
        self.assertEqual(cascade.style_of("a-model-not-listed"), "json")  # as does any model the block leaves out
        delegation = cascade.delegation
        self.assertEqual(delegation.app_wait_seconds, 50.0)
        self.assertEqual((delegation.progress_after_seconds, delegation.progress_max_per_delegation), (8.0, 3))
        self.assertEqual((cascade.memory.context_tokens, cascade.memory.compact_at), (16000, 0.8))
        turns = (0.2, 1.2, 2)
        speech, default = cascade.speech, SpeechConfig()
        self.assertEqual((speech.vad_stop_seconds, speech.turn_silence_seconds, speech.interrupt_min_words), turns)
        self.assertEqual((default.vad_stop_seconds, default.turn_silence_seconds, default.interrupt_min_words), turns)
        self.assertEqual((speech.vad_start_seconds, default.vad_start_seconds), (0.12, 0.12))
        reliability = cascade.reliability
        self.assertEqual((reliability.provider_retries, reliability.talker_attempt_timeout_seconds), (2, 10.0))
        self.assertEqual((cascade.tts_preroll_ms, cascade.display_label), (300, "delegation_core"))
        prompts = load_prompts()  # the entries the settings name are the prompt file's
        self.assertTrue(prompts.text("Talker", cascade.prompt))
        self.assertTrue(prompts.text("Channel", config.ui_bot.channel))

    def test_the_app_wait_line_comes_before_the_cores_deadline_for_the_apps_outputs(self):
        config = ExampleConfig.model_validate(EXAMPLE_SETTINGS)
        deadline = CoreSettings.parse(config.core, CORE_TEXTS).delegation.client_tool_timeout_seconds
        shipped = cascade_of(config).delegation.app_wait_seconds
        self.assertLess(shipped, deadline)
        self.assertEqual(DelegationConfig().app_wait_seconds, shipped)  # the schema's default is the shipped value

    def test_settings_with_no_app_wait_line_or_progress_lines(self):
        quiet = {"frontend": {"cascade": {"delegation": {"app_wait_seconds": None, "progress_after_seconds": None}}}}
        shipped = cascade_of(ExampleConfig.model_validate(EXAMPLE_SETTINGS))
        bench = cascade_of(ExampleConfig.model_validate(merged(EXAMPLE_SETTINGS, quiet)))
        self.assertIsNone(bench.delegation.app_wait_seconds)
        self.assertIsNone(bench.delegation.progress_after_seconds)
        self.assertEqual(bench.delegation.progress_max_per_delegation, shipped.delegation.progress_max_per_delegation)
        self.assertEqual((bench.slot, bench.prompt, bench.memory), (shipped.slot, "generic", shipped.memory))

    def test_the_ui_bots_core_is_the_cores_settings_with_its_own_merged_over_them(self):
        config = ExampleConfig.model_validate(EXAMPLE_SETTINGS)
        self.assertEqual(config.ui_bot_core, merged(config.core, EXAMPLE_SETTINGS["ui_bot"]["core"]))
        self.assertEqual(config.ui_bot_core["backend"]["adapter"], "llm")  # whatever the core's adapter is
        self.assertEqual(config.core["backend"]["slot"], config.ui_bot_core["backend"]["slot"])

    def test_a_server_prompt_file_replaces_the_examples_whole_prompt_file(self):
        with TemporaryDirectory() as folder:
            other = Path(folder) / "other.yaml"
            other.write_text("custom:\n  content: not the talker prompt\n", encoding="utf-8")
            lacks = f"agent_prompts.Cascade was not found in {re.escape(str(other))}"
            with patch.dict(os.environ, {"PROMPT_FILE_PATH": str(other)}), self.assertRaisesRegex(KeyError, lacks):
                load_prompts()


class DisplayLabelTests(unittest.TestCase):
    """The display label comes from the core's settings only."""

    def test_the_display_label_comes_from_the_core_only(self):
        with self.assertRaisesRegex(ValueError, "display_label comes from the core's delegation.display_label"):
            ExampleConfig.model_validate(example(display_label="screen"))
        config = ExampleConfig.model_validate(example())
        self.assertEqual(cascade_of(config, label="screen").display_label, "screen")


class CheckTests(unittest.TestCase):
    def test_unknown_keys_and_bad_values_are_errors(self):
        for bad, message in (
            ({**example(), "oops": 1}, "oops\n.*Extra inputs are not permitted"),
            ({"frontend": {"cascade": CASCADE}, "core": {}}, "frontend.adapter\n.*Field required"),
            ({"frontend": {"adapter": "cascade"}, "core": {}}, "no frontend.cascade block"),
            ({"frontend": {"adapter": "cascade", "cascade": "fast"}, "core": {}}, "cascade\n.*valid dictionary"),
            ({**example(), "ui_bot": {"tools": ["cafe"]}}, "ui_bot.tools\n.*Extra inputs are not permitted"),
            ({**example(), "ui_bot": {"core": {}}}, "ui_bot.channel\n.*Field required"),
            ({"frontend": example()["frontend"], "core": {}}, "ui_bot\n.*Field required"),
            ({"frontend": example()["frontend"]}, "core\n.*Field required"),
        ):
            with self.subTest(bad=bad), self.assertRaisesRegex(ValueError, message):
                ExampleConfig.model_validate(bad)
        for cascade, message in (
            ({"oops": 1}, "oops\n.*Extra inputs are not permitted"),
            ({"slot": ""}, "slot\n.*at least 1 character"),
            ({"provider": "chat-completions"}, "provider\n.*Extra inputs are not permitted"),
            ({"talker_style": "tools"}, "talker_style\n.*valid dictionary"),
            ({"talker_style": {"nemotron-lightning": ""}}, "talker_style.nemotron-lightning\n.*at least 1 character"),
            ({"prompt": ""}, "prompt\n.*at least 1 character"),
            ({"reliability": {"provider_retries": "two"}}, "reliability.provider_retries\n.*valid integer"),
            ({"reliability": {"nope": 1}}, "reliability.nope\n.*Extra inputs are not permitted"),
            ({"reliability": {"client_timeout_seconds": 45}}, "client_timeout_seconds\n.*Extra inputs are not"),
            ({"memory": {"context_tokens": 0.5}}, "memory.context_tokens\n.*valid integer"),
            ({"memory": {"compact_at": None}}, "memory.compact_at\n.*valid number"),
            ({"memory": {"compact_at": 1.5}}, "memory.compact_at\n.*less than or equal to 1"),
            ({"delegation": {"progress_max_per_delegation": 1.5}}, "progress_max_per_delegation\n.*valid integer"),
            ({"delegation": {"app_wait_seconds": -1}}, "app_wait_seconds\n.*greater than 0"),
            ({"speech": {"interrupt_min_words": 1.5}}, "interrupt_min_words\n.*valid integer"),
            ({"speech": {"vad_start_seconds": 0}}, "vad_start_seconds\n.*greater than 0"),
            ({"speech": {"vad_stop_seconds": 0}}, "vad_stop_seconds\n.*greater than 0"),
            ({"speech": {"turn_silence_seconds": -1}}, "turn_silence_seconds\n.*greater than 0"),
            ({"tts_preroll_ms": 2001}, "tts_preroll_ms\n.*less than or equal to 2000"),
        ):
            with self.subTest(cascade=cascade), self.assertRaisesRegex(ValueError, message):
                cascade_of(ExampleConfig.model_validate(example(**cascade)))

    def test_a_zero_is_only_allowed_for_the_retry_count(self):
        config = ExampleConfig.model_validate(example(reliability={"provider_retries": 0}))
        self.assertEqual(cascade_of(config).reliability.provider_retries, 0)
        with self.assertRaisesRegex(ValueError, "greater than 0"):
            cascade_of(ExampleConfig.model_validate(example(reliability={"talker_attempt_timeout_seconds": 0})))

    def test_values_override_the_defaults_and_null_keeps_a_parts_defaults(self):
        config = ExampleConfig.model_validate(
            example(
                reliability={"provider_retries": 0, "talker_attempt_timeout_seconds": 3},
                delegation={"app_wait_seconds": None, "progress_max_per_delegation": 1},
                memory=None,
                talker_style=None,
            )
        )
        cascade = cascade_of(config)
        self.assertEqual(cascade.reliability.talker_attempt_timeout_seconds, 3.0)
        self.assertIsNone(cascade.delegation.app_wait_seconds)
        self.assertEqual(cascade.delegation.progress_max_per_delegation, 1)
        self.assertEqual(cascade.memory, LiveConfig(slot="s").memory)
        self.assertEqual(cascade.talker_style, {})


if __name__ == "__main__":
    unittest.main()
