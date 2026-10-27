# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D107

"""The cafe demo tools that the delegation core runs as server tools."""

import json
import unittest

from delegation_core.backend.tools.cafe.schemas import CAFE_TOOLS
from delegation_core.backend.tools.cafe.tools import CafeTools


# ------------------------------------------------------------------------------- cafe tools
class CafeToolTests(unittest.IsolatedAsyncioTestCase):
    async def run_tool(self, tools, call_id, name, **arguments):
        return json.loads(await tools.execute(call_id, name, json.dumps(arguments)))

    async def test_a_bad_call_cannot_poison_the_cart(self):
        tools = CafeTools()
        bad = [
            ("manage_cart", {"action": "add", "item_id": "nope"}),
            ("manage_cart", {"action": "add", "item_id": None}),
            ("manage_cart", {"action": "remove", "line_id": "line_99"}),
            ("manage_cart", {"action": "update_quantity"}),
        ]
        for i, (name, args) in enumerate(bad):
            result = await self.run_tool(tools, f"bad{i}", name, **args)
            self.assertEqual(result["status"], "error")
        added = await self.run_tool(tools, "a", "manage_cart", action="add", item_id="lat", size="huge", quantity="x")
        line = added["cart"]["lines"][0]
        self.assertEqual((line["size"], line["quantity"]), ("medium", 1))  # defaults, not a crash later
        self.assertEqual((await self.run_tool(tools, "v", "manage_cart", action="view"))["cart"]["item_count"], 1)
        self.assertEqual(
            (await self.run_tool(tools, "m", "get_menu", query=None, category=None))["items"][0]["id"], "esp"
        )

    async def test_cart_total_includes_size_modifiers_and_tax(self):
        tools = CafeTools()
        cart = (
            await self.run_tool(
                tools, "1", "manage_cart", action="add", item_id="lat", size="large", modifiers=["oat_milk"]
            )
        )["cart"]
        self.assertEqual((cart["subtotal"], cart["tax"], cart["total"]), (6.25, 0.53, 6.78))

    async def test_a_repeated_call_id_neither_adds_a_second_item_nor_places_a_second_order(self):
        tools = CafeTools()
        await self.run_tool(tools, "a", "manage_cart", action="add", item_id="esp")
        await self.run_tool(tools, "a", "manage_cart", action="add", item_id="esp")
        self.assertEqual(tools.snapshot()["item_count"], 1)
        first = await self.run_tool(tools, "p", "place_order", customer_name="Sam", confirmed=True)
        again = await self.run_tool(tools, "p", "place_order", customer_name="Sam", confirmed=True)
        self.assertEqual((first["status"], first["order"]["order_id"]), ("confirmed", again["order"]["order_id"]))

    async def test_an_order_needs_explicit_confirmation(self):
        tools = CafeTools()
        await self.run_tool(tools, "a", "manage_cart", action="add", item_id="crs", quantity=2)
        pending = await self.run_tool(tools, "b", "place_order", customer_name="Sam", confirmed=False)
        self.assertEqual(pending["status"], "needs_confirmation")
        self.assertEqual(tools.last_order, None)
        empty = await self.run_tool(CafeTools(), "c", "place_order", customer_name="Sam", confirmed=True)
        self.assertEqual(empty["status"], "error")

    async def test_bad_input_returns_an_error_result_not_an_exception(self):
        tools = CafeTools()
        self.assertEqual((await self.run_tool(tools, "x", "nope"))["status"], "error")
        self.assertEqual(
            (await self.run_tool(tools, "y", "manage_cart", action="add", item_id="zzz"))["status"], "error"
        )
        self.assertEqual(json.loads(await tools.execute("z", "manage_cart", "not json"))["status"], "error")

    async def test_menu_filters_by_category(self):
        menu = await self.run_tool(CafeTools(), "m", "get_menu", category="tea")
        self.assertEqual([item["name"] for item in menu["items"]], ["Matcha Latte", "Earl Grey"])

    def test_tool_schemas_match_the_executor(self):
        self.assertEqual({tool["name"] for tool in CAFE_TOOLS}, {"get_menu", "manage_cart", "place_order"})


if __name__ == "__main__":
    unittest.main()
