# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D103

"""Tests for ``backend.tools``: the tool sources by name, and the cafe's."""

import unittest

from delegation_core.backend.port import ToolSource
from delegation_core.backend.tools import TOOL_SOURCES, CafeTools
from delegation_core.backend.tools.cafe.schemas import CAFE_TOOLS


class ToolSourceTests(unittest.TestCase):
    def test_the_cafe_is_a_tool_source_named_by_its_label_that_takes_no_settings(self):
        self.assertEqual(dict(TOOL_SOURCES), {"cafe": CafeTools})
        cafe: ToolSource = CafeTools.from_settings({})
        self.assertEqual(cafe.label, "cafe")
        self.assertEqual([tool["name"] for tool in cafe.tools()], [tool["name"] for tool in CAFE_TOOLS])
        with self.assertRaisesRegex(ValueError, "takes no settings"):
            CafeTools.from_settings({"url": "http://example"})

    def test_its_schemas_are_a_copy(self):
        cafe = CafeTools()
        cafe.tools()[0]["name"] = "changed"
        self.assertEqual(cafe.tools()[0]["name"], CAFE_TOOLS[0]["name"])


if __name__ == "__main__":
    unittest.main()
