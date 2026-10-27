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
from examples.frontend_backend_live.prompts.catalog import role_instructions
from tests.unit.contracts.base import ContractCase
from tests.unit.frontend_backend_live.root.root_helpers import StubAdapter, clear_root_caches, stub_root

# What each backend adapter's source says of itself (its describe()), as the core adds it to its own.
CORE_HALVES = {
    "llm": {"adapter": "llm", "one_shot_adapter": "llm", "provider": "chat-completions", "conductor": "start_each"},
    "llm-agent": {"adapter": "llm-agent", "one_shot_adapter": "llm", "framework": "agents-sdk"},
    "harness": {"adapter": "harness", "one_shot_adapter": "llm", "protocol": "acp"},
}
# The keys applyInfo() reads, by path.
CONSOLE_KEYS = (
    ("warnings",),
    ("backend_models",),
    ("backend", "model"),
    ("instructions", "frontend"),
    ("instructions", "backend"),
    ("sample_tools",),
    ("frontend", "model"),
    ("frontend", "provider"),
    ("prompt_version",),
)


class Source:
    """A tool source by the port: its tools only."""

    def __init__(self, *names):
        self.names = names

    def tools(self):
        return [{"type": "function", "name": name, "parameters": {"type": "object"}} for name in self.names]


class DescribeCase(unittest.TestCase):
    def setUp(self):
        clear_root_caches(self)
        self.state, self.blocks = stub_root(), []

        def tool_sources(blocks):
            self.blocks.append(dict(blocks))
            return lambda: (Source("get_menu", "manage_cart"), Source("checkout"))

        self.enterContext(patch.object(factory, "root", return_value=self.state))
        self.enterContext(patch.object(factory, "tool_sources", tool_sources))

    def describe(self, core_half: dict) -> dict:
        with patch.object(factory, "delegation_core", return_value=SimpleNamespace(describe=lambda: dict(core_half))):
            return asyncio.run(factory.describe())


class DescribeTests(DescribeCase):
    def test_the_console_keys_are_there_under_each_backend_adapter(self):
        for name, core_half in CORE_HALVES.items():
            with self.subTest(adapter=name):
                factory.sample_tools.cache_clear()
                info = self.describe(core_half)
                for path in CONSOLE_KEYS:
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
                "prompt": "thinker",
            },
        )
        self.assertEqual(info["backend"]["provider"], "chat-completions")  # the console's backend badge
        self.assertEqual(info["backend_models"], ["org/back", "org/other"])  # the default first
        self.assertEqual(
            info["instructions"], {"frontend": "the talker's instructions", "backend": role_instructions("thinker")}
        )
        self.assertEqual(info["warnings"], [])

    def test_the_adapters_warnings_reach_the_console(self):
        warned = {**StubAdapter.describe(None, self.state.catalog), "warnings": ["the talker is degraded"]}
        with patch.object(StubAdapter, "describe", return_value=warned):
            self.assertEqual(self.describe(CORE_HALVES["llm"])["warnings"], ["the talker is degraded"])

    def test_the_sample_tools_are_the_ui_bot_profiles_tool_sources_built_by_name(self):
        info = self.describe(CORE_HALVES["llm"])
        self.assertEqual(self.blocks, [{"cafe": {}}])  # core.ui-bot.yaml's backend.tools
        self.assertEqual([tool["name"] for tool in info["sample_tools"]], ["get_menu", "manage_cart", "checkout"])
        self.describe(CORE_HALVES["llm"])
        self.assertEqual(len(self.blocks), 1)  # built once

    def test_describe_reads_off_the_event_loop(self):
        threads = []

        def describe_in_thread():
            threads.append(threading.current_thread())
            return {}

        with patch.object(factory, "_describe", describe_in_thread):
            asyncio.run(factory.describe())
        self.assertIsNot(threads[0], threading.main_thread())


class ShippedSampleToolsTests(ContractCase, unittest.TestCase):
    def setUp(self):
        clear_root_caches(self)
        self.enterContext(patch.object(factory, "root", return_value=stub_root()))

    def test_the_shipped_profiles_sample_tools_are_the_cafe_demos(self):
        from delegation_core.backend.tools.cafe.schemas import CAFE_TOOLS

        self.assertEqual(factory.sample_tools(), CAFE_TOOLS)


if __name__ == "__main__":
    unittest.main()
