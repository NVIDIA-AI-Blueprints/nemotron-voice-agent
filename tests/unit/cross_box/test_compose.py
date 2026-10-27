# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102

"""The Live example's Compose recipes serve the Live routes by default.

A recipe's ``EXAMPLE_SELECTION`` is only a default, and ``all`` leaves the Live routes off, so the three Live recipes
default ``LIVE_ENABLED`` to ``true``. No other recipe sets it, because the routes are open to anyone who can reach the
server unless ``LIVE_API_TOKEN`` is set.
"""

import re
import unittest
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[3]
LIVE_EXAMPLE = "frontend-backend-live"
LIVE_RECIPES = {"frontend-backend-live", "frontend-backend-live-server", "frontend-backend-live-single-gpu"}
DEFAULT_ON = "${LIVE_ENABLED:-true}"
INTERPOLATION = re.compile(r"^\$\{\w+:-(?P<default>[^}]*)\}$")


def compose_default(value: object) -> str:
    """The value Compose gives an ``environment`` entry whose variable is unset or empty in the shell and ``.env``."""
    match = INTERPOLATION.match(str(value))
    return match.group("default") if match else str(value)


class LiveRecipeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        compose = yaml.safe_load((REPO / "docker-compose.yml").read_text(encoding="utf-8"))
        cls.environments = {name: service.get("environment") or {} for name, service in compose["services"].items()}
        cls.live = {
            name
            for name, environment in cls.environments.items()
            if compose_default(environment.get("EXAMPLE_SELECTION", "")) == LIVE_EXAMPLE
        }

    def test_the_live_recipes_are_the_three_profiles(self):
        self.assertEqual(self.live, LIVE_RECIPES)

    def test_each_live_recipe_defaults_live_enabled_to_true(self):
        for name in sorted(self.live):
            with self.subTest(service=name):
                self.assertEqual(self.environments[name].get("LIVE_ENABLED"), DEFAULT_ON)
                self.assertEqual(compose_default(self.environments[name]["LIVE_ENABLED"]), "true")

    def test_no_other_recipe_sets_live_enabled(self):
        others = {name for name, environment in self.environments.items() if "LIVE_ENABLED" in environment}
        self.assertEqual(others - self.live, set())


if __name__ == "__main__":
    unittest.main()
