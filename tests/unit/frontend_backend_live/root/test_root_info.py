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
from examples.frontend_backend_live.prompts import load_prompts
from tests.unit.contracts.base import ContractCase
from tests.unit.frontend_backend_live.root.root_helpers import StubAdapter, clear_root_caches, stub_root
from utils import load_prompt_catalog

# What each backend adapter's source says of itself (its describe()), as the core adds it to its own.
CORE_HALVES = {
    "llm": {"adapter": "llm", "one_shot_adapter": "llm", "provider": "chat-completions", "conductor": "start_each"},
    "llm-agent": {"adapter": "llm-agent", "one_shot_adapter": "llm", "framework": "agents-sdk"},
    "harness": {"adapter": "harness", "one_shot_adapter": "llm", "protocol": "acp"},
}
# The keys the example's half always holds, by path, among them the keys applyInfo() reads.
INFO_KEYS = (
    ("warnings",),
    ("backend_models",),
    ("backend", "model"),
    ("backend", "prompt"),
    ("frontend", "model"),
    ("frontend", "prompt"),
    ("frontend", "provider"),
    ("instructions", "frontend"),
    ("instructions", "backend"),
    ("sample_tools",),
    # and the lines that client-mode.js, runToolCalls and the append presets send
    ("lines", "apology"),
    ("lines", "unclear"),
    ("lines", "still_working"),
    ("lines", "intro"),
    ("lines", "app_started"),
    # and the other append presets, by append type
    ("presets",),
)
# The talker prompt that the UI selects by default (defaults.prompt in examples_registry.yaml), as the file holds it.
CAFE_TALKER = load_prompt_catalog(factory.__file__)["cafe"]["content"]


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
    def test_the_keys_are_there_under_each_backend_adapter(self):
        for name, core_half in CORE_HALVES.items():
            with self.subTest(adapter=name):
                factory.console_sample.cache_clear()
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
        self.assertEqual(
            info["backend"],
            {
                **CORE_HALVES["llm"],
                "slot": "thinker-llm",
                "model": "org/back",
                "base_url": "http://back/v1",
                "prompt": "generic",  # the core's own default backend prompt, for a request with no instructions
            },
        )
        self.assertEqual(info["backend"]["provider"], "chat-completions")  # the console's backend badge
        self.assertEqual(info["backend_models"], ["org/back", "org/other"])  # the default first
        # Both prefills are the cafe demo's: the UI's default talker prompt and the UI bot's backend prompt, each with
        # the channel context after it, which the console runs with its sample tools.
        prompts = load_prompts()
        channel = prompts.text("Channel", "cafe")
        self.assertEqual(
            info["instructions"],
            {
                "frontend": f"{CAFE_TALKER.strip()}\n\n{channel}",
                "backend": f"{prompts.text('Backend', 'cafe')}\n\n{channel}",
            },
        )
        self.assertEqual(info["warnings"], [])

    def test_the_console_gets_the_lines_the_frontend_adapter_says_from_the_prompt_file(self):
        lines = load_prompts().lines
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

    def test_the_console_gets_the_append_presets_of_the_prompt_file(self):
        presets = load_prompts().console_presets()
        self.assertEqual(
            list(presets), ["session.instructions.append", "session.thinking.append", "session.commentary.append"]
        )
        self.assertEqual(self.describe(CORE_HALVES["llm"])["presets"], presets)

    def test_the_adapters_warnings_reach_the_console(self):
        warned = {**StubAdapter.describe(None, self.state.catalog), "warnings": ["the talker is degraded"]}
        with patch.object(StubAdapter, "describe", return_value=warned):
            self.assertEqual(self.describe(CORE_HALVES["llm"])["warnings"], ["the talker is degraded"])

    def test_the_sample_tools_are_the_ui_bots_tool_sources_built_by_name(self):
        info = self.describe(CORE_HALVES["llm"])
        self.assertEqual(self.blocks, [{"cafe": {}}])  # core.ui-bot.yaml's backend.tools
        self.assertEqual([tool["name"] for tool in info["sample_tools"]], ["get_menu", "manage_cart", "checkout"])
        self.describe(CORE_HALVES["llm"])
        self.assertEqual(len(self.blocks), 1)  # built once

    def test_the_prefills_and_the_sample_tools_come_from_one_read_of_the_prompt_file_and_the_ui_bots_core(self):
        prompts = load_prompts()
        backend = {**prompts.agent_prompts["Backend"], "cafe": {"content": "Run the kiosk.\n"}}
        edited = replace(prompts, agent_prompts={**prompts.agent_prompts, "Backend": backend})
        core = factory.ui_bot_core(self.state.config)
        kiosk = replace(core, backend=replace(core.backend, tools={"kiosk": {}}))
        with (
            patch.object(factory, "load_prompts", return_value=edited) as read,
            patch.object(factory, "ui_bot_core", return_value=kiosk),
        ):
            info = self.describe(CORE_HALVES["llm"])
        channel = prompts.text("Channel", "cafe")
        backend = f"Run the kiosk.\n\n{channel}"  # as a UI bot session starts its backend
        self.assertEqual(info["instructions"], {"frontend": f"{CAFE_TALKER.strip()}\n\n{channel}", "backend": backend})
        self.assertEqual(self.blocks, [{"kiosk": {}}])
        self.assertEqual(read.call_count, 1)  # one read for the channel, the backend prompt, the lines and the presets
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
    def prompts(self, group):
        """The prompt file's read with ``group`` as its ``agent_prompts.ConsolePresets``."""
        prompts = load_prompts()
        return replace(prompts, agent_prompts={**prompts.agent_prompts, "ConsolePresets": group})

    def test_the_presets_are_the_entries_by_append_type_and_none_without_the_group(self):
        store = {"description": "Store state", "append": "session.thinking.append", "content": "The machine is busy."}
        presets = {"session.thinking.append": [["Store state", "The machine is busy."]]}
        self.assertEqual(self.prompts({"store_state": store}).console_presets(), presets)
        self.assertEqual(self.prompts({}).console_presets(), {})
        prompts = load_prompts()
        groups = {group: entries for group, entries in prompts.agent_prompts.items() if group != "ConsolePresets"}
        self.assertEqual(replace(prompts, agent_prompts=groups).console_presets(), {})  # a file without the group

    def test_a_malformed_group_is_refused(self):
        malformed = {
            "a text": "Say hi.",
            "a preset that is a text": {"greet": "Say hi."},
            "a preset with no append": {"greet": {"description": "Greet", "content": "Say hi."}},
            "a preset that is no text": {
                "count": {"description": "Count", "append": "session.thinking.append", "content": 3}
            },
        }
        for name, group in malformed.items():
            with (
                self.subTest(malformed=name),
                self.assertRaisesRegex(ValueError, "agent_prompts.ConsolePresets"),
            ):
                self.prompts(group).console_presets()


class ShippedSampleTests(ContractCase, unittest.TestCase):
    def setUp(self):
        clear_root_caches(self)
        self.enterContext(patch.object(factory, "root", return_value=stub_root()))

    def test_the_shipped_profiles_sample_tools_are_the_cafe_demos(self):
        from examples.frontend_backend_live.delegation_core.backend.tools.cafe.schemas import CAFE_TOOLS

        self.assertEqual(factory.console_sample().tools, CAFE_TOOLS)


if __name__ == "__main__":
    unittest.main()
