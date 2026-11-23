# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102

"""The cafe demo tools that the delegation core runs as server tools."""

import json
import unittest

from nemotron_voice_agent.pipecat.examples.frontend_backend_live.delegation_core.backend.tools.cafe.tools import (
    CafeTools,
)


class CafeToolTests(unittest.IsolatedAsyncioTestCase):
    async def run_tool(self, tools, call_id, name, **arguments):
        return json.loads(await tools.execute(call_id, name, json.dumps(arguments)))

    async def test_a_bad_call_is_an_error_and_cannot_poison_the_cart(self):
        tools = CafeTools()
        for call_id, name, arguments in (
            ("b0", "manage_cart", {"action": "add", "item_id": "nope"}),
            ("b1", "manage_cart", {"action": "add", "item_id": None}),
            ("b2", "manage_cart", {"action": "remove", "line_id": "line_99"}),
            ("b3", "manage_cart", {"action": "update_quantity"}),
            ("b4", "nope", {}),
        ):
            with self.subTest(name=name, arguments=arguments):
                self.assertEqual((await self.run_tool(tools, call_id, name, **arguments))["status"], "error")
        self.assertEqual(json.loads(await tools.execute("b5", "manage_cart", "not json"))["status"], "error")
        added = await self.run_tool(tools, "a", "manage_cart", action="add", item_id="lat", size="huge", quantity="x")
        line = added["cart"]["lines"][0]
        self.assertEqual((line["size"], line["quantity"]), ("medium", 1))
        self.assertEqual((await self.run_tool(tools, "v", "manage_cart", action="view"))["cart"]["item_count"], 1)
        for call_id, options in (
            ("u1", {"size": "large", "modifiers": ["unknown"]}),
            ("u2", {"size": "unknown", "modifiers": ["oat_milk"]}),
            ("u3", {}),
        ):
            with self.subTest(options=options):
                failed = await self.run_tool(
                    tools, call_id, "manage_cart", action="update_options", line_id=line["line_id"], **options
                )
                self.assertEqual(failed["status"], "error")
                self.assertEqual(tools.snapshot(), added["cart"])

    async def test_the_menu_filters_and_the_total_adds_size_and_tax(self):
        tools = CafeTools()
        assert (await self.run_tool(tools, "m", "get_menu", query=None, category=None))["items"][0]["id"] == "esp"
        tea = await self.run_tool(tools, "t", "get_menu", category="tea")
        self.assertEqual([item["name"] for item in tea["items"]], ["Matcha Latte", "Earl Grey"])
        added = await self.run_tool(
            tools, "1", "manage_cart", action="add", item_id="lat", quantity=2, size="large", modifiers=["oat_milk"]
        )
        cart = added["cart"]
        self.assertEqual((cart["subtotal"], cart["tax"], cart["total"]), (12.5, 1.06, 13.56))
        original = cart["lines"][0]
        line_id = original["line_id"]
        for call_id, options, expected in (
            ("2", {"size": "small"}, ("small", ["oat_milk"], 11.39)),
            ("3", {"modifiers": ["extra_shot"]}, ("small", ["extra_shot"], 11.94)),
            ("4", {"size": "medium", "modifiers": []}, ("medium", [], 10.85)),
        ):
            with self.subTest(options=options):
                updated = await self.run_tool(
                    tools, call_id, "manage_cart", action="update_options", line_id=line_id, **options
                )
                self.assertEqual(updated["status"], "updated")
                [line] = updated["cart"]["lines"]
                self.assertEqual((line["line_id"], line["item_id"], line["quantity"]), (line_id, "lat", 2))
                self.assertEqual((line["size"], line["modifiers"], updated["cart"]["total"]), expected)

    async def test_an_order_needs_confirmation_and_runs_once(self):
        tools = CafeTools()
        await self.run_tool(tools, "a", "manage_cart", action="add", item_id="esp")
        await self.run_tool(tools, "a", "manage_cart", action="add", item_id="esp")
        self.assertEqual(tools.snapshot()["item_count"], 1)
        pending = await self.run_tool(tools, "b", "place_order", customer_name="Sam", confirmed=False)
        self.assertEqual((pending["status"], tools.last_order), ("needs_confirmation", None))
        first = await self.run_tool(tools, "p", "place_order", customer_name="Sam", confirmed=True)
        again = await self.run_tool(tools, "p", "place_order", customer_name="Sam", confirmed=True)
        self.assertEqual((first["status"], first["order"]["order_id"]), ("confirmed", again["order"]["order_id"]))
        empty = await self.run_tool(CafeTools(), "c", "place_order", customer_name="Sam", confirmed=True)
        self.assertEqual(empty["status"], "error")
