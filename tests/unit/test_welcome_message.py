# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

# ruff: noqa: D100, D101, D102

import os
import unittest
from unittest.mock import patch

import examples_registry

_EXAMPLE_KEY = "generic-assistant"


class WelcomeMessageResolutionTests(unittest.TestCase):
    def _example_with_welcome(self, value: bool) -> dict:
        return {**examples_registry.EXAMPLES[_EXAMPLE_KEY], "welcome_message": value}

    def test_defaults_to_enabled(self) -> None:
        # Empty string is treated as unset, so the registry value (default True) applies.
        with patch.dict(os.environ, {"ENABLE_WELCOME_MESSAGE": ""}):
            self.assertTrue(examples_registry.welcome_message_enabled(_EXAMPLE_KEY))

    def test_registry_value_disables_per_example(self) -> None:
        with (
            patch.dict(os.environ, {"ENABLE_WELCOME_MESSAGE": ""}),
            patch.dict(examples_registry.EXAMPLES, {_EXAMPLE_KEY: self._example_with_welcome(False)}),
        ):
            self.assertFalse(examples_registry.welcome_message_enabled(_EXAMPLE_KEY))

    def test_env_override_wins_over_registry(self) -> None:
        with (
            patch.dict(os.environ, {"ENABLE_WELCOME_MESSAGE": "false"}),
            patch.dict(examples_registry.EXAMPLES, {_EXAMPLE_KEY: self._example_with_welcome(True)}),
        ):
            self.assertFalse(examples_registry.welcome_message_enabled(_EXAMPLE_KEY))

    def test_env_override_can_force_enabled(self) -> None:
        with (
            patch.dict(os.environ, {"ENABLE_WELCOME_MESSAGE": "true"}),
            patch.dict(examples_registry.EXAMPLES, {_EXAMPLE_KEY: self._example_with_welcome(False)}),
        ):
            self.assertTrue(examples_registry.welcome_message_enabled(_EXAMPLE_KEY))


if __name__ == "__main__":
    unittest.main()
