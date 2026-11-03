# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D103, D107
"""``GET /v1/live/info``'s example half, composed by the root."""

import asyncio
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from examples.frontend_backend_live import factory
from tests.unit.frontend_backend_live.root.root_helpers import StubAdapter, clear_root_caches, stub_root

# What each backend adapter's source says of itself (its describe()), as the core adds it to its own.
CORE_HALVES = {
    "llm": {"adapter": "llm", "one_shot_adapter": "llm", "provider": "chat-completions", "conductor": "start_each"},
    "llm-agent": {"adapter": "llm-agent", "one_shot_adapter": "llm", "framework": "agents-sdk"},
    "harness": {"adapter": "harness", "one_shot_adapter": "llm", "protocol": "acp"},
}
# The keys the example's half always holds, by path.
INFO_KEYS = (
    ("warnings",),
    ("backend_models",),
    ("backend", "model"),
    ("backend", "prompt"),
    ("frontend", "model"),
    ("frontend", "prompt"),
    ("frontend", "provider"),
    ("prompt_version",),
)


class DescribeCase(unittest.TestCase):
    def setUp(self):
        clear_root_caches(self)
        self.state = stub_root()
        self.enterContext(patch.object(factory, "root", return_value=self.state))

    def describe(self, core_half: dict) -> dict:
        with patch.object(factory, "delegation_core", return_value=SimpleNamespace(describe=lambda: dict(core_half))):
            return asyncio.run(factory.describe())


class DescribeTests(DescribeCase):
    def test_the_keys_are_there_under_each_backend_adapter(self):
        for name, core_half in CORE_HALVES.items():
            with self.subTest(adapter=name):
                info = self.describe(core_half)
                for path in INFO_KEYS:
                    value = info
                    for key in path:
                        self.assertIn(key, value, f"{'.'.join(path)} is missing")
                        value = value[key]
                self.assertEqual((info["backend"]["adapter"], info["backend"]["one_shot_adapter"]), (name, "llm"))

    def test_the_adapters_half_and_the_cores_half_compose(self):
        info = self.describe(CORE_HALVES["llm"])
        adapter_half = StubAdapter.describe(self.state.settings, self.state.catalog)
        self.assertEqual(info["frontend"], adapter_half["frontend"])
        self.assertEqual(info["prompt_version"], adapter_half["prompt_version"])
        self.assertEqual(
            info["backend"],
            {
                **CORE_HALVES["llm"],
                "slot": "thinker-llm",
                "model": "org/back",
                "base_url": "http://back/v1",
                "prompt": "generic",  # the core's own default profile, for a request with no instructions
            },
        )
        self.assertEqual(info["backend_models"], ["org/back", "org/other"])  # the default first
        self.assertEqual(info["warnings"], [])

    def test_the_adapters_warnings_are_reported(self):
        warned = {**StubAdapter.describe(None, self.state.catalog), "warnings": ["the talker is degraded"]}
        with patch.object(StubAdapter, "describe", return_value=warned):
            self.assertEqual(self.describe(CORE_HALVES["llm"])["warnings"], ["the talker is degraded"])

    def test_describe_reads_off_the_event_loop(self):
        threads = []

        def describe_in_thread():
            threads.append(threading.current_thread())
            return {}

        with patch.object(factory, "_describe", describe_in_thread):
            asyncio.run(factory.describe())
        self.assertIsNot(threads[0], threading.main_thread())


if __name__ == "__main__":
    unittest.main()
