# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D103

"""The cascade's settings: the ``frontend.cascade`` keys it reads, with their defaults, validated."""

import unittest
from dataclasses import replace

from examples.frontend_backend_live.cascade.settings import CascadeSettings
from examples.frontend_backend_live.config_manager.schema import DelegationConfig, MemoryConfig, SpeechConfig
from tests.unit.frontend_backend_live.cascade.doubles import CONFIG


class SettingsTests(unittest.TestCase):
    def test_the_defaults_are_the_documented_ones(self):
        settings = CascadeSettings.parse({})
        self.assertEqual(
            (
                settings.app_wait_seconds,
                settings.result_wording,
                settings.progress_after_seconds,
                settings.progress_max_per_delegation,
                settings.context_tokens,
                settings.compact_at,
                settings.start_timeout_seconds,
                settings.resume_after_noise,
                settings.retry_silent_lines,
                settings.heard_words,
                settings.estimate_chars_per_second,
                settings.display_label,
            ),
            (50.0, "writer", 8.0, 3, 16_000, 0.8, 3.0, True, True, "as_recorded", 15.0, "delegation_core"),
        )

    def test_unknown_keys_and_wrong_values_are_refused(self):
        for block, message in (
            ({"timers": {}}, "unknown cascade settings: timers"),
            ({"memory": {"window": 3}}, "memory has unknown keys: window"),
            ({"memory": {"context_tokens": 1.5}}, "memory.context_tokens must be a positive whole number"),
            ({"memory": {"compact_at": 1.5}}, "memory.compact_at must be a number above 0 and at most 1"),
            ({"speech": {"start_timeout_seconds": None}}, "speech.start_timeout_seconds must be a positive number"),
            ({"speech": {"resume_after_noise": "yes"}}, "speech.resume_after_noise must be true or false"),
            ({"speech": {"heard_words": True}}, "speech.heard_words must be a name"),
            ({"speech": {"heard_words": "guessed"}}, "Unknown speech.heard_words 'guessed'"),
            ({"speech": {"estimate_chars_per_second": 0}}, "must be a positive number of characters per second"),
            ({"delegation": {"progress_max_per_delegation": True}}, "must be a positive whole number"),
            (
                {"delegation": {"result_wording": "poetic"}},
                "Unknown result wording 'poetic'; known: as_written, writer",
            ),
        ):
            with self.subTest(block=block), self.assertRaisesRegex(ValueError, message):
                CascadeSettings.parse(block)

    def test_the_frontend_block_maps_onto_the_cascades_settings_one_to_one(self):
        # The root reads the whole frontend.cascade block as LiveConfig; the cascade takes its delegation, memory and
        # speech sections and the core's display label, and checks them by its own rules.
        self.assertEqual(CascadeSettings.from_live_config(CONFIG), CascadeSettings.parse({}))
        config = replace(
            CONFIG,
            delegation=DelegationConfig(
                app_wait_seconds=None,
                result_wording="as_written",
                progress_after_seconds=None,
                progress_max_per_delegation=2,
            ),
            memory=MemoryConfig(context_tokens=8000, compact_at=0.5),
            speech=SpeechConfig(
                start_timeout_seconds=2.0,
                resume_after_noise=False,
                retry_silent_lines=False,
                heard_words="estimated",
                estimate_chars_per_second=12.0,
            ),
            display_label="ours",
        )
        settings = CascadeSettings.from_live_config(config)
        self.assertEqual(
            (
                settings.app_wait_seconds,
                settings.result_wording,
                settings.progress_after_seconds,
                settings.progress_max_per_delegation,
                settings.context_tokens,
                settings.compact_at,
                settings.start_timeout_seconds,
                settings.resume_after_noise,
                settings.retry_silent_lines,
                settings.heard_words,
                settings.estimate_chars_per_second,
                settings.display_label,
            ),
            (None, "as_written", None, 2, 8000, 0.5, 2.0, False, False, "estimated", 12.0, "ours"),
        )
        with self.assertRaisesRegex(ValueError, "Unknown result wording 'poetic'"):
            CascadeSettings.from_live_config(replace(CONFIG, delegation=DelegationConfig(result_wording="poetic")))
