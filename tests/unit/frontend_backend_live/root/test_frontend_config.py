# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D100, D101, D102, D103, D105, D107
"""The frontend's ``config.yaml``: the frontend adapter, its block, the UI bot's profile."""

import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from delegation_core.config import CoreSettings
from examples.frontend_backend_live.config_manager import schema
from examples.frontend_backend_live.config_manager.loader import CONFIG_PATH, ConfigError, load_live_config
from examples.frontend_backend_live.config_manager.schema import (
    DelegationConfig,
    ExampleConfig,
    LiveConfig,
    SpeechConfig,
)
from examples.frontend_backend_live.prompts.catalog import catalog_prompt, role_instructions

BENCH_PATH = CONFIG_PATH.with_name("config.bench.yaml")
CASCADE = {"provider": "chat-completions", "slot": "llm"}


def example(**cascade) -> dict:
    """A minimal file: the cascade adapter with ``cascade`` as its block."""
    return {"frontend": {"adapter": "cascade", "cascade": {**CASCADE, **cascade}}}


def cascade_of(config: ExampleConfig, label: str = "delegation_core") -> LiveConfig:
    """The cascade's settings as the root parses them, with the core's display label handed in."""
    return LiveConfig.parse({**config.blocks[config.adapter], "display_label": label})


def write(folder: str, name: str, text: str) -> Path:
    path = Path(folder) / name
    path.write_text(text, encoding="utf-8")
    return path


class ShippedFileTests(unittest.TestCase):
    def test_the_shipped_file_names_the_cascade_and_its_settings(self):
        config = load_live_config(CONFIG_PATH)
        self.assertEqual((config.adapter, list(config.blocks)), ("cascade", ["cascade"]))
        self.assertEqual(config.ui_bot.core_config, "core.ui-bot.yaml")
        cascade = cascade_of(config)
        self.assertEqual((cascade.provider, cascade.slot, cascade.prompt), ("chat-completions", "llm", "voice"))
        self.assertEqual(LiveConfig(provider="p", slot="s").prompt, cascade.prompt)  # the schema's default agrees
        self.assertEqual(cascade.prompt_version, "default")
        delegation = cascade.delegation
        self.assertEqual((delegation.app_wait_seconds, delegation.result_wording), (50.0, "writer"))
        self.assertEqual((delegation.progress_after_seconds, delegation.progress_max_per_delegation), (8.0, 3))
        self.assertEqual((cascade.memory.context_tokens, cascade.memory.compact_at), (16000, 0.8))
        self.assertEqual(cascade.speech.tts_mode, "per_sentence")
        turn_end = ("smart_turn", 0.2, 1.2)
        speech, default = cascade.speech, SpeechConfig()
        self.assertEqual((speech.turn_end, speech.vad_stop_seconds, speech.turn_silence_seconds), turn_end)
        self.assertEqual((default.turn_end, default.vad_stop_seconds, default.turn_silence_seconds), turn_end)
        self.assertEqual((speech.vad_start_seconds, default.vad_start_seconds), (0.12, 0.12))
        self.assertEqual((cascade.speech.heard_words, cascade.speech.estimate_chars_per_second), ("as_recorded", 15.0))
        self.assertEqual(cascade.reliability.client_timeout_seconds, 45.0)
        self.assertEqual((cascade.tts_preroll_ms, cascade.display_label), (300, "delegation_core"))
        self.assertTrue(catalog_prompt(cascade.prompt))

    def test_the_app_wait_line_comes_before_the_cores_deadline_for_the_apps_outputs(self):
        deadline = CoreSettings.load(CONFIG_PATH.with_name("core.yaml")).delegation.client_tool_timeout_seconds
        shipped = cascade_of(load_live_config(CONFIG_PATH)).delegation.app_wait_seconds
        self.assertLess(shipped, deadline)
        self.assertEqual(DelegationConfig().app_wait_seconds, shipped)  # the schema's default is the shipped value

    def test_the_bench_profile_extends_the_file_without_the_app_wait_line_or_progress_lines(self):
        shipped, bench = cascade_of(load_live_config(CONFIG_PATH)), cascade_of(load_live_config(BENCH_PATH))
        self.assertIsNone(bench.delegation.app_wait_seconds)
        self.assertIsNone(bench.delegation.progress_after_seconds)
        self.assertEqual(bench.delegation.result_wording, shipped.delegation.result_wording)
        self.assertEqual(
            (bench.provider, bench.prompt_version, bench.memory), (shipped.provider, "default", shipped.memory)
        )

    def test_live_config_path_points_the_server_at_another_file(self):
        with TemporaryDirectory() as folder:
            other = write(folder, "other.yaml", "frontend: {adapter: cascade, cascade: {provider: p, slot: s}}\n")
            with patch.dict(os.environ, {"LIVE_CONFIG_PATH": str(other)}):
                self.assertEqual(cascade_of(load_live_config()).slot, "s")

    def test_a_server_prompt_file_does_not_replace_the_examples_own_prompts(self):
        with TemporaryDirectory() as tmp:
            other = Path(tmp) / "other.yaml"
            other.write_text("custom:\n  content: not the talker prompt\n")
            with patch.dict(os.environ, {"PROMPT_FILE_PATH": str(other)}):
                self.assertTrue(catalog_prompt("talker"))
                self.assertTrue(role_instructions("thinker").startswith(catalog_prompt("thinker")))

    def test_role_instructions_append_the_channel_context(self):
        text = role_instructions("thinker")
        self.assertTrue(text.startswith(catalog_prompt("thinker")))
        self.assertTrue(text.endswith(catalog_prompt("channel")))
        with self.assertRaises(KeyError):
            catalog_prompt("no-such-prompt")


class DisplayLabelTests(unittest.TestCase):
    """The display label comes from the core's file only."""

    def test_the_display_label_comes_from_the_core_only(self):
        with self.assertRaisesRegex(ConfigError, "display_label comes from core.yaml delegation.display_label"):
            ExampleConfig.parse(example(display_label="screen"))
        config = ExampleConfig.parse(example())
        self.assertEqual(cascade_of(config, label="screen").display_label, "screen")


class CheckTests(unittest.TestCase):
    def test_unknown_keys_and_bad_values_are_errors(self):
        for bad, message in (
            ({**example(), "oops": 1}, "config.yaml has unknown keys: oops"),
            ({"frontend": {"cascade": CASCADE}}, "frontend.adapter must name"),
            ({"frontend": {"adapter": "cascade"}}, "no frontend.cascade block"),
            ({"frontend": {"adapter": "cascade", "cascade": "fast"}}, "frontend.cascade must be a mapping"),
            ({**example(), "ui_bot": {"tools": ["cafe"]}}, "ui_bot has unknown keys: tools"),
        ):
            with self.subTest(bad=bad), self.assertRaisesRegex(ConfigError, message):
                ExampleConfig.parse(bad)
        for cascade, message in (
            ({"oops": 1}, "frontend.cascade has unknown keys: oops"),
            ({"provider": ""}, "frontend.cascade.provider must name"),
            ({"prompt": ""}, "frontend.cascade.prompt must be the name"),
            ({"prompt_version": "v9"}, "prompt_version: Unknown prompt set 'v9'"),
            ({"reliability": {"provider_retries": "two"}}, "must be a non-negative whole number"),
            ({"reliability": {"nope": 1}}, "frontend.cascade.reliability has unknown keys: nope"),
            ({"normalizer": {"enabled": "yes"}}, "true or false"),
            ({"memory": {"context_tokens": 0.5}}, "memory.context_tokens must be a positive whole number"),
            ({"delegation": {"progress_max_per_delegation": 1.5}}, "must be a positive whole number"),
            ({"delegation": {"app_wait_seconds": -1}}, "app_wait_seconds must be a positive number"),
            ({"memory": {"compact_at": None}}, "compact_at must be a positive number"),
            ({"speech": {"tts_mode": ""}}, "tts_mode must be a name"),
            ({"speech": {"turn_end": ""}}, "turn_end must be a name"),
            ({"speech": {"vad_start_seconds": 0}}, "vad_start_seconds must be a positive number"),
            ({"speech": {"vad_stop_seconds": 0}}, "vad_stop_seconds must be a positive number"),
            ({"speech": {"turn_silence_seconds": -1}}, "turn_silence_seconds must be a positive number"),
            ({"speech": {"estimate_chars_per_second": 0}}, "estimate_chars_per_second must be a positive number"),
            ({"tts_preroll_ms": 2001}, "tts_preroll_ms must be a whole number from 0 to 2000"),
        ):
            with self.subTest(cascade=cascade), self.assertRaisesRegex(ConfigError, message):
                cascade_of(ExampleConfig.parse(example(**cascade)))

    def test_a_zero_is_only_allowed_for_the_retry_count(self):
        config = ExampleConfig.parse(example(reliability={"provider_retries": 0}))
        self.assertEqual(cascade_of(config).reliability.provider_retries, 0)
        for field in ("talker_attempt_timeout_seconds", "client_timeout_seconds"):
            with self.subTest(field=field), self.assertRaisesRegex(ConfigError, "positive"):
                cascade_of(ExampleConfig.parse(example(reliability={field: 0})))

    def test_a_prompt_set_the_table_registers_loads_by_its_name(self):
        with patch.dict(schema.PROMPT_SETS, {"custom": object()}):
            self.assertEqual(cascade_of(ExampleConfig.parse(example(prompt_version="custom"))).prompt_version, "custom")
        # Left out, the prompt sets' own default applies.
        self.assertEqual(cascade_of(ExampleConfig.parse(example())).prompt_version, "")

    def test_values_override_the_defaults(self):
        config = ExampleConfig.parse(
            example(
                reliability={"provider_retries": 0, "talker_attempt_timeout_seconds": 3},
                delegation={"app_wait_seconds": None, "result_wording": "as_written"},
            )
        )
        cascade = cascade_of(config)
        self.assertEqual(cascade.reliability.talker_attempt_timeout_seconds, 3.0)
        self.assertIsNone(cascade.delegation.app_wait_seconds)
        self.assertEqual(cascade.delegation.result_wording, "as_written")


class ProfileTests(unittest.TestCase):
    def test_a_profile_extends_another_and_null_replaces(self):
        with TemporaryDirectory() as folder:
            write(
                folder, "base.yaml", "frontend: {adapter: cascade, cascade: {provider: p, slot: s, prompt: talker}}\n"
            )
            profile = write(
                folder,
                "profile.yaml",
                "extends: base.yaml\nfrontend: {cascade: {delegation: {app_wait_seconds: null}}}\n",
            )
            cascade = cascade_of(load_live_config(profile))
            self.assertEqual((cascade.provider, cascade.slot, cascade.prompt), ("p", "s", "talker"))
            self.assertIsNone(cascade.delegation.app_wait_seconds)

    def test_a_profile_that_extends_itself_or_a_missing_file_is_refused(self):
        with TemporaryDirectory() as folder:
            write(folder, "a.yaml", "extends: b.yaml\n")
            write(folder, "b.yaml", "extends: a.yaml\n")
            with self.assertRaisesRegex(ConfigError, "extends itself"):
                load_live_config(Path(folder) / "a.yaml")
            missing = write(folder, "c.yaml", "extends: nowhere.yaml\n")
            with self.assertRaisesRegex(ConfigError, "Cannot read .*nowhere.yaml"):
                load_live_config(missing)


if __name__ == "__main__":
    unittest.main()
