# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D100, D101, D102

import unittest

from nemotron_voice_agent.core.settings import apply_service_settings


class ApplyServiceSettingsTest(unittest.TestCase):
    def test_non_finite_and_out_of_range_numbers_fall_back_or_clamp(self) -> None:
        schema = {"max_tokens": {"type": "int", "min": 1, "max": 4096, "default": 512, "path": "max_tokens"}}
        for raw, expected in ((float("inf"), 512), (float("nan"), 512), ("1e999", 512), (True, 512), (99999, 4096)):
            with self.subTest(raw=raw):
                self.assertEqual(apply_service_settings({}, schema, {"max_tokens": raw})["max_tokens"], expected)

    def test_reasoning_budget_stays_under_max_tokens(self) -> None:
        schema = {
            "max_tokens": {"type": "int", "min": 1, "max": 16384, "path": "max_tokens"},
            "enable_thinking": {"type": "bool", "default": False, "path": "enable_thinking"},
            "reasoning_budget": {
                "type": "int",
                "default": 16384,
                "min": 128,
                "max": 32768,
                "path": "budget",
                "requires": "enable_thinking",
                "below": "max_tokens",
                "reserve": 64,
            },
        }
        cases = [
            ({"enable_thinking": True}, 16384),
            ({"enable_thinking": True, "max_tokens": 4096, "reasoning_budget": 1024}, 1024),
            ({"enable_thinking": True, "max_tokens": 1024}, 960),
            ({"enable_thinking": True, "max_tokens": 192}, 128),
        ]
        for values, budget in cases:
            with self.subTest(values=values):
                applied = apply_service_settings({}, schema, values)
                self.assertEqual(applied["budget"], budget)
                self.assertIs(applied["enable_thinking"], True)
        for max_tokens in (64, 191):
            with self.subTest(max_tokens=max_tokens):
                applied = apply_service_settings({}, schema, {"enable_thinking": True, "max_tokens": max_tokens})
                self.assertIs(applied["enable_thinking"], False)
                self.assertNotIn("budget", applied)
        unconditional = {
            **schema,
            "reasoning_budget": {k: v for k, v in schema["reasoning_budget"].items() if k != "requires"},
        }
        self.assertNotIn("budget", apply_service_settings({}, unconditional, {"max_tokens": 64}))
        self.assertEqual(apply_service_settings({}, unconditional, {"max_tokens": 1024})["budget"], 960)
