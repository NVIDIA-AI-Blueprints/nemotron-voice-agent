# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D103, D107
"""``GET /v1/live/info``'s example half, composed by the root."""

import asyncio
import threading
import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

from examples.frontend_backend_live import factory
from examples.frontend_backend_live.cascade.lines import catalog_lines
from examples.frontend_backend_live.prompts.catalog import get_profile
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
    # and the lines that client-mode.js, runToolCalls and the append presets send
    ("lines", "apology"),
    ("lines", "unclear"),
    ("lines", "still_working"),
    ("lines", "intro"),
    ("lines", "app_started"),
    # and the other append presets, by append type
    ("presets",),
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
                factory.console_sample.cache_clear()
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
                "prompt": "generic",  # the core's own default profile, for a request with no instructions
            },
        )
        self.assertEqual(info["backend"]["provider"], "chat-completions")  # the console's backend badge
        self.assertEqual(info["backend_models"], ["org/back", "org/other"])  # the default first
        # Both prefills are the default profile's (the cafe demo's), which the console runs with its sample tools.
        cafe = get_profile("cafe")
        self.assertEqual(
            info["instructions"], {"frontend": cafe.instructions("talker"), "backend": cafe.instructions("thinker")}
        )
        self.assertEqual(info["warnings"], [])

    def test_the_console_gets_the_lines_the_frontend_adapter_says_from_the_catalog(self):
        lines = catalog_lines()
        self.assertEqual(
            self.describe(CORE_HALVES["llm"])["lines"],
            {
                "apology": lines.apology,
                "unclear": lines.unclear,
                "still_working": lines.still_working,
                "intro": lines.intro,
                "app_started": lines.app_started,
            },
        )

    def test_the_console_gets_the_default_profiles_append_presets_as_written(self):
        block = get_profile("cafe").console_presets
        self.assertTrue(block)
        self.assertEqual(self.describe(CORE_HALVES["llm"])["presets"], block)

    def test_the_adapters_warnings_reach_the_console(self):
        warned = {**StubAdapter.describe(None, self.state.catalog), "warnings": ["the talker is degraded"]}
        with patch.object(StubAdapter, "describe", return_value=warned):
            self.assertEqual(self.describe(CORE_HALVES["llm"])["warnings"], ["the talker is degraded"])

    def test_the_sample_tools_are_the_default_profiles_tool_sources_built_by_name(self):
        info = self.describe(CORE_HALVES["llm"])
        self.assertEqual(self.blocks, [{"cafe": {}}])  # prompts/cafe/profile.yaml's tools
        self.assertEqual([tool["name"] for tool in info["sample_tools"]], ["get_menu", "manage_cart", "checkout"])
        self.describe(CORE_HALVES["llm"])
        self.assertEqual(len(self.blocks), 1)  # built once

    def test_the_prefills_and_the_sample_tools_come_from_one_read_of_the_default_profile(self):
        cafe = get_profile("cafe")
        roles = {**cafe.roles, "thinker": {"content": "Run the kiosk.\n"}}
        profile = replace(cafe, roles=roles, tools={"kiosk": {}})
        with patch.object(factory, "default_profile", return_value=profile) as read:
            info = self.describe(CORE_HALVES["llm"])
        backend = f"Run the kiosk.\n\n{cafe.prompt('channel')}"  # as a UI bot session of the profile starts its backend
        self.assertEqual(info["instructions"], {"frontend": cafe.instructions("talker"), "backend": backend})
        self.assertEqual(self.blocks, [{"kiosk": {}}])
        self.assertEqual(read.call_count, 1)  # one read of the profile for the prefills, the tools and the presets
        self.assertEqual(info["backend"]["prompt"], self.state.core.backend.prompt)  # the core keeps its own default

    def test_describe_reads_off_the_event_loop(self):
        threads = []

        def describe_in_thread():
            threads.append(threading.current_thread())
            return {}

        with patch.object(factory, "_describe", describe_in_thread):
            asyncio.run(factory.describe())
        self.assertIsNot(threads[0], threading.main_thread())


class ConsolePresetsTests(unittest.TestCase):
    def profile(self, presets):
        """The cafe profile with ``presets`` as its ``console_presets`` setting."""
        return replace(get_profile("cafe"), console_presets=presets)

    def test_the_presets_are_the_settings_pairs_and_none_without_it(self):
        presets = {"session.thinking.append": [["Store state", "The machine is busy."]]}
        self.assertEqual(factory.console_presets(self.profile(presets)), presets)
        self.assertEqual(factory.console_presets(self.profile({})), {})
        self.assertEqual(factory.console_presets(get_profile("generic")), {})  # a profile without the setting

    def test_a_malformed_block_is_refused(self):
        malformed = {
            "a text": "Say hi.",
            "a preset that is a text": {"session.thinking.append": ["Say hi."]},
            "a preset that is no text": {"session.thinking.append": [["Count", 3]]},
        }
        for name, block in malformed.items():
            with (
                self.subTest(malformed=name),
                self.assertRaisesRegex(ValueError, "prompts/cafe/profile.yaml console_presets must map each"),
            ):
                factory.console_presets(self.profile(block))


class ShippedSampleTests(ContractCase, unittest.TestCase):
    def setUp(self):
        clear_root_caches(self)
        self.enterContext(patch.object(factory, "root", return_value=stub_root()))

    def test_the_shipped_profiles_sample_tools_are_the_cafe_demos(self):
        from delegation_core.backend.tools.cafe.schemas import CAFE_TOOLS

        self.assertEqual(factory.console_sample().tools, CAFE_TOOLS)


if __name__ == "__main__":
    unittest.main()
