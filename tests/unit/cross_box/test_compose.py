# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102

"""No Compose service sets the Live settings.

The Live routes are open to anyone who can reach the server unless ``LIVE_API_TOKEN`` is set, so no recipe serves them
by default: a deployment sets the token, or ``LIVE_ENABLED=true``, in its ``.env``.
"""

import unittest
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[3]
LIVE_SETTINGS = {"LIVE_ENABLED", "LIVE_API_TOKEN"}


class ComposeTests(unittest.TestCase):
    def test_no_service_sets_the_live_settings(self):
        compose = yaml.safe_load((REPO / "docker-compose.yml").read_text(encoding="utf-8"))
        services = dict(compose["services"])
        for include in compose.get("include", []):
            services.update(yaml.safe_load((REPO / include["path"]).read_text(encoding="utf-8")).get("services") or {})
        for name, service in services.items():
            # A mapping's keys, or a list's NAME=value and NAME items.
            names = {str(item).split("=", 1)[0] for item in service.get("environment") or ()}
            with self.subTest(service=name):
                self.assertEqual(names & LIVE_SETTINGS, set())


if __name__ == "__main__":
    unittest.main()
