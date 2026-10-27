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
    def test_the_defaults_are_the_frozen_ones(self):
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
                settings.display_label,
            ),
            (62.0, "writer", 8.0, 1, 16_000, 0.8, 3.0, True, True, "voiceclaw"),
        )

    def test_a_block_names_only_what_differs_and_null_turns_a_clock_off(self):
        settings = CascadeSettings.parse(
            {"delegation": {"app_wait_seconds": None, "progress_after_seconds": None}, "memory": {"compact_at": 0.5}},
            display_label="ours",
        )
        self.assertEqual(
            (settings.app_wait_seconds, settings.progress_after_seconds, settings.compact_at, settings.display_label),
            (None, None, 0.5, "ours"),
        )

    def test_unknown_keys_and_wrong_values_are_refused(self):
        for block, message in (
            ({"timers": {}}, "unknown cascade settings: timers"),
            ({"memory": {"window": 3}}, "memory has unknown keys: window"),
            ({"memory": {"context_tokens": 1.5}}, "memory.context_tokens must be a positive whole number"),
            ({"memory": {"compact_at": 1.5}}, "memory.compact_at must be a number above 0 and at most 1"),
            ({"speech": {"start_timeout_seconds": None}}, "speech.start_timeout_seconds must be a positive number"),
            ({"speech": {"resume_after_noise": "yes"}}, "speech.resume_after_noise must be true or false"),
            ({"delegation": {"progress_max_per_delegation": True}}, "must be a positive whole number"),
            ({"delegation": {"result_wording": "poetic"}}, "Unknown result wording 'poetic'"),
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
            speech=SpeechConfig(start_timeout_seconds=2.0, resume_after_noise=False, retry_silent_lines=False),
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
                settings.display_label,
            ),
            (None, "as_written", None, 2, 8000, 0.5, 2.0, False, False, "ours"),
        )
        with self.assertRaisesRegex(ValueError, "Unknown result wording 'poetic'"):
            CascadeSettings.from_live_config(replace(CONFIG, delegation=DelegationConfig(result_wording="poetic")))
