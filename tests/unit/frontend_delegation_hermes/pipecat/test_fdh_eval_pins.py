# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D100, D101, D102, D103, D107

"""Today's FDH evaluation behaviour is unchanged (``shared-stack-plan.md``, step 0; see ``_fdh_eval_pins.py``)."""

from __future__ import annotations

import unittest

import _fdh_eval_pins as pins


class EvalPinTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.stored = pins.load_pins()

    def test_profiles_and_endpoints(self) -> None:
        for name, profile in (("eval_profile", pins.EVAL_PROFILE), ("demo_profile", pins.DEMO_PROFILE)):
            with self.subTest(profile=name):
                self.assertEqual(pins.config_pins(profile), self.stored[name])

    def test_gateway_prompts_and_settings(self) -> None:
        for name, path in (("eval_gateway", pins.EVAL_GATEWAY), ("demo_gateway", pins.DEMO_GATEWAY)):
            with self.subTest(gateway=name):
                self.assertEqual(pins.gateway_pins(path), self.stored[name])

    def test_pipecat_settings_and_processors(self) -> None:
        self.assertEqual(pins.pipeline_pins(), self.stored["pipeline"])

    def test_scripted_eval_sessions(self) -> None:
        for domain in pins.DOMAINS:
            with self.subTest(domain=domain):
                self.assertEqual(pins.session_pins(domain), self.stored["sessions"][domain])


if __name__ == "__main__":
    unittest.main()
