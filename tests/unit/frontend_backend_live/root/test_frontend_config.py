# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D100, D101, D102, D103, D105, D107
"""The frontend's ``config.yaml``: the frontend adapter, its block, the UI bot's profile."""

import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from examples.frontend_backend_live.config_manager import schema
from examples.frontend_backend_live.config_manager.loader import CONFIG_PATH, ConfigError, load_live_config
from examples.frontend_backend_live.config_manager.schema import ExampleConfig, LiveConfig
from examples.frontend_backend_live.prompts.catalog import catalog_prompt, role_instructions

BENCH_PATH = CONFIG_PATH.with_name("config.bench.yaml")
CASCADE = {"provider": "chat-completions", "slot": "llm"}
LICENSE_HEADER = [
    "# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.",
    "# SPDX-License-Identifier: Apache-2.0",
]


def example(**cascade) -> dict:
    """A minimal file: the cascade adapter with ``cascade`` as its block."""
    return {"frontend": {"adapter": "cascade", "cascade": {**CASCADE, **cascade}}}


def cascade_of(config: ExampleConfig, label: str = "voiceclaw") -> LiveConfig:
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
        self.assertEqual((cascade.provider, cascade.slot, cascade.prompt), ("chat-completions", "llm", "talker"))
        self.assertEqual(cascade.prompt_version, "v2")
        self.assertFalse(cascade.guards.repeat_guard)
        delegation = cascade.delegation
        self.assertEqual((delegation.app_wait_seconds, delegation.result_wording), (62.0, "writer"))
        self.assertEqual((delegation.progress_after_seconds, delegation.progress_max_per_delegation), (8.0, 1))
        self.assertEqual((cascade.memory.context_tokens, cascade.memory.compact_at), (16000, 0.8))
        self.assertEqual(cascade.speech.tts_mode, "per_sentence")
        self.assertEqual(cascade.reliability.client_timeout_seconds, 45.0)
        self.assertEqual((cascade.tts_preroll_ms, cascade.display_label), (300, "voiceclaw"))
        self.assertTrue(catalog_prompt(cascade.prompt))

    def test_the_bench_profile_extends_the_file_without_the_app_wait_line_or_progress_lines(self):
        shipped, bench = cascade_of(load_live_config(CONFIG_PATH)), cascade_of(load_live_config(BENCH_PATH))
        self.assertIsNone(bench.delegation.app_wait_seconds)
        self.assertIsNone(bench.delegation.progress_after_seconds)
        self.assertEqual(bench.delegation.result_wording, shipped.delegation.result_wording)
        self.assertEqual((bench.provider, bench.prompt_version, bench.memory), (shipped.provider, "v2", shipped.memory))

    def test_the_shipped_files_carry_the_license_header(self):
        for name in ("config.yaml", "config.bench.yaml", "core.yaml", "core.ui-bot.yaml", "core.bench.yaml"):
            with self.subTest(name):
                lines = CONFIG_PATH.with_name(name).read_text(encoding="utf-8").splitlines()
                self.assertEqual(lines[:2], LICENSE_HEADER)

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


class MovedKeyTests(unittest.TestCase):
    """A key that moved, or that a split replaced, names where its setting lives."""

    def test_the_cores_keys_name_the_cores_file(self):
        cases = {
            "backend": ({"backend": {"provider": "p", "slot": "t"}}, "core.yaml backend"),
            "voice_note": ({"voice_note": "v2"}, "core.yaml delegation.voice_note"),
            "backend_context_tokens": ({"backend_context_tokens": 1000}, "core.yaml delegation.context_tokens"),
        }
        for name, (extra, successor) in cases.items():
            with self.subTest(name), self.assertRaisesRegex(ConfigError, f"^{name} moved: set .*{successor}"):
                ExampleConfig.parse({**example(), **extra})

    def test_todays_layout_names_the_cascade_block(self):
        today = {"frontend": {"provider": "chat-completions", "slot": "llm", "prompt": "talker"}}
        with self.assertRaisesRegex(ConfigError, "^frontend.provider moved: set frontend.cascade.provider"):
            ExampleConfig.parse(today)
        for key in ("prompt_version", "reliability", "guards", "delegation", "normalizer", "chat_history_recent_turns"):
            with self.subTest(key), self.assertRaisesRegex(ConfigError, f"^{key} moved: set frontend.cascade"):
                ExampleConfig.parse({**example(), key: {}})

    def test_each_split_key_names_its_successors(self):
        cases = {
            ("reliability", "tool_timeout_seconds"): (
                "client_tool_timeout_seconds.*server_tool_timeout_seconds.*app_wait_seconds"
            ),
            ("reliability", "api_timeout_seconds"): "client_timeout_seconds.*first_event_timeout_seconds",
            ("reliability", "recover_failures"): "provider_retries; the voice says its own lines",
            ("reliability", "max_tool_rounds"): "core.yaml backend.llm.max_tool_rounds",
            ("reliability", "max_pending_delegations"): "core.yaml delegation.max_pending_delegations",
            ("delegation", "merge_pending"): "core.yaml backend.llm.conductor",
            ("delegation", "fold_before_acting"): "core.yaml backend.llm.conductor",
            ("delegation", "status_from_ledger"): "nothing: the talker answers a status question",
            ("delegation", "progress_speech"): "progress_after_seconds \\(null: off\\)",
            ("delegation", "progress_interval_seconds"): "nothing: a progress line comes after",
        }
        for (section, key), successor in cases.items():
            config = ExampleConfig.parse(example(**{section: {key: 1}}))
            with (
                self.subTest(key=f"{section}.{key}"),
                self.assertRaisesRegex(ConfigError, f"^frontend.cascade.{section}.{key} moved: set .*{successor}"),
            ):
                cascade_of(config)
        with self.assertRaisesRegex(ConfigError, "chat_history_recent_turns moved: set frontend.cascade.memory"):
            cascade_of(ExampleConfig.parse(example(chat_history_recent_turns=20)))

    def test_the_display_label_comes_from_the_core_only(self):
        with self.assertRaisesRegex(ConfigError, "display_label moved: set core.yaml delegation.display_label"):
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
            ({"guards": {"repeat_guard": "yes"}}, "true or false"),
            ({"guards": {"continuers": "okay"}}, "list of strings"),
            ({"memory": {"context_tokens": 0.5}}, "memory.context_tokens must be a positive whole number"),
            ({"delegation": {"progress_max_per_delegation": 1.5}}, "must be a positive whole number"),
            ({"delegation": {"app_wait_seconds": -1}}, "app_wait_seconds must be a positive number"),
            ({"memory": {"compact_at": None}}, "compact_at must be a positive number"),
            ({"speech": {"tts_mode": ""}}, "tts_mode must be a name"),
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
        with patch.dict(schema.PROMPT_SETS, {"v3": object()}):
            self.assertEqual(cascade_of(ExampleConfig.parse(example(prompt_version="v3"))).prompt_version, "v3")
        # Left out, the prompt sets' own default applies.
        self.assertEqual(cascade_of(ExampleConfig.parse(example())).prompt_version, "")

    def test_values_override_the_defaults(self):
        config = ExampleConfig.parse(
            example(
                reliability={"provider_retries": 0, "talker_attempt_timeout_seconds": 3},
                guards={"turn_router_shadow": True, "continuers": ["okay"]},
                delegation={"app_wait_seconds": None, "result_wording": "as_written"},
            )
        )
        cascade = cascade_of(config)
        self.assertEqual(cascade.reliability.talker_attempt_timeout_seconds, 3.0)
        self.assertTrue(cascade.guards.turn_router_shadow)
        self.assertEqual(cascade.guards.continuers, ("okay",))
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

    def test_an_unreadable_file_is_a_config_error(self):
        with TemporaryDirectory() as tmp, self.assertRaises(ConfigError):
            load_live_config(Path(tmp) / "missing.yaml")


if __name__ == "__main__":
    unittest.main()
