# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D103, D107

"""Tests for ``backend.tools``: each named source gets its block, one set per session."""

import json
import unittest
from collections.abc import Mapping
from typing import Any, ClassVar

from delegation_core.backend.port import ToolSource
from delegation_core.backend.tools import TOOL_SOURCES, CafeTools
from delegation_core.backend.tools.cafe.schemas import CAFE_TOOLS
from delegation_core.tables import pick


def session_tools(table: Mapping[str, type], blocks: Mapping[str, Mapping[str, Any]]):
    """What a composition root does with ``backend.tools``: a fresh set of the named sources for each IM session."""
    return lambda: [pick(table, name, "tool source").from_settings(block) for name, block in blocks.items()]


class LookupTools:
    """A source that reads its endpoint from its block, as a real one would (an environment variable's name)."""

    label: ClassVar[str] = "lookup"

    def __init__(self, url_env: str):
        self.url_env = url_env

    @classmethod
    def from_settings(cls, block):
        return cls(block["url_env"])

    def tools(self):
        return [{"type": "function", "name": "find", "parameters": {"type": "object", "properties": {}}}]

    async def execute(self, call_id, name, arguments):
        return json.dumps({"from": self.url_env})


class ToolSourceTests(unittest.IsolatedAsyncioTestCase):
    async def test_two_sessions_get_two_carts(self):
        tools = session_tools(TOOL_SOURCES, {"cafe": {}})
        first, second = tools(), tools()
        self.assertIsNot(first[0], second[0])
        await first[0].execute("c1", "manage_cart", json.dumps({"action": "add", "item_id": "lat"}))
        self.assertEqual(first[0].snapshot()["item_count"], 1)
        self.assertEqual(second[0].snapshot()["item_count"], 0)

    async def test_a_block_reaches_its_source(self):
        table = {**TOOL_SOURCES, LookupTools.label: LookupTools}
        sources = session_tools(table, {"cafe": {}, "lookup": {"url_env": "MENU_URL"}})()
        self.assertEqual([source.label for source in sources], ["cafe", "lookup"])
        self.assertEqual(json.loads(await sources[1].execute("c1", "find", "{}")), {"from": "MENU_URL"})

    def test_the_cafe_takes_no_settings_and_an_unknown_source_is_named(self):
        with self.assertRaisesRegex(ValueError, "takes no settings"):
            CafeTools.from_settings({"url": "http://example"})
        with self.assertRaisesRegex(ValueError, "Unknown tool source 'menu'; known: cafe"):
            session_tools(TOOL_SOURCES, {"menu": {}})()

    def test_the_cafe_is_a_tool_source_named_by_its_label(self):
        self.assertEqual(dict(TOOL_SOURCES), {"cafe": CafeTools})
        cafe: ToolSource = CafeTools.from_settings({})
        self.assertEqual(cafe.label, "cafe")
        self.assertEqual([tool["name"] for tool in cafe.tools()], [tool["name"] for tool in CAFE_TOOLS])

    def test_its_schemas_are_a_copy(self):
        cafe = CafeTools()
        cafe.tools()[0]["name"] = "changed"
        self.assertEqual(cafe.tools()[0]["name"], CAFE_TOOLS[0]["name"])


if __name__ == "__main__":
    unittest.main()
