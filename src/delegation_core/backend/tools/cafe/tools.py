# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""In-memory cafe tools: menu, cart and checkout, as a server tool source. No external order is ever placed.

The Live Console's browser cafe (``client/public/live/cafe.js``) gives the same results, rule for rule:
``client/test/live/cafe-parity.json`` holds them, and ``tests/unit/cross_box/test_console_cafe_parity.py`` checks this
module against it, so a change here goes into both files.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from copy import deepcopy
from datetime import UTC, datetime
from decimal import ROUND_HALF_UP, Decimal
from typing import Any, ClassVar

from .schemas import CAFE_TOOLS

MENU = [
    ("esp", "Espresso", "coffee", 3.0, True, "Double shot, pulled to order."),
    ("lat", "Latte", "coffee", 4.5, True, "Espresso with steamed milk."),
    ("cap", "Cappuccino", "coffee", 4.25, True, "Equal parts espresso, milk, foam."),
    ("cbr", "Cold Brew", "coffee", 4.75, True, "18-hour steep, served over ice."),
    ("mch", "Matcha Latte", "tea", 5.0, True, "Ceremonial grade matcha."),
    ("erl", "Earl Grey", "tea", 3.25, True, "Bergamot black tea."),
    ("crs", "Butter Croissant", "pastry", 3.75, False, "Baked this morning."),
    ("bmf", "Blueberry Muffin", "pastry", 3.5, False, "Made with Maine blueberries."),
    ("avt", "Avocado Toast", "food", 8.5, False, "Sourdough, chili flake, lemon."),
]
# The tools the schemas offer, by name: the schemas are the one list (each name is a method below).
TOOL_NAMES = frozenset(tool["name"] for tool in CAFE_TOOLS)
SIZES = {"small": 0, "medium": 0.5, "large": 1}
MODIFIERS = {
    "oat_milk": {"label": "Oat milk", "price": 0.75},
    "extra_shot": {"label": "Extra shot", "price": 1},
    "decaf": {"label": "Decaf", "price": 0},
    "iced": {"label": "Iced", "price": 0},
}


def _whole_number(value, default):
    """Return ``value`` as an int, or ``default`` when it is missing or not a number."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def money(value):
    """Round a price to cents."""
    return float(Decimal(str(value)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


class CafeTools:
    """One IM session's cart and orders, in memory: a server tool source (``backend/port.py``, ``ToolSource``).

    ``backend.tools: {cafe: {}}`` names it; the composition root makes one for each IM session, so two calls never
    share a cart. No external order is ever placed.
    """

    label: ClassVar[str] = "cafe"  # the server label of its calls' mcp_call items, and its name in backend.tools

    def __init__(self):
        """Start with an empty cart."""
        self.lines = []
        self.line_seq, self.order_seq = 0, 41
        self.last_order = None
        self.by_call = {}

    @classmethod
    def from_settings(cls, block: Mapping[str, Any]) -> CafeTools:
        """A fresh cart from its ``backend.tools.cafe`` block, which takes no settings.

        Raises:
            ValueError: The block sets anything.
        """
        if not isinstance(block, Mapping) or block:
            raise ValueError(f"backend.tools.{cls.label} takes no settings; got {block!r}")
        return cls()

    def tools(self) -> list[dict]:
        """The cafe's function tools, in the Responses shape (a copy, so a caller cannot change the schemas)."""
        return deepcopy(CAFE_TOOLS)

    def snapshot(self):
        """Return the cart with line totals, subtotal, tax and total."""
        lines = []
        for line in self.lines:
            item = next(x for x in MENU if x[0] == line["item_id"])
            price = item[3] + (SIZES[line["size"]] if item[4] else 0)
            price += sum(MODIFIERS[m]["price"] for m in line["modifiers"])
            lines.append({**line, "name": item[1], "line_total": money(price * line["quantity"])})
        subtotal = money(sum(x["line_total"] for x in lines))
        tax = money(subtotal * 0.085)
        return {
            "lines": lines,
            "item_count": sum(x["quantity"] for x in lines),
            "subtotal": subtotal,
            "tax": tax,
            "total": money(subtotal + tax),
            "currency": "USD",
        }

    async def execute(self, call_id: str, name: str, arguments: str) -> str:
        """Run one tool call and return its result as JSON; a repeated ``call_id`` returns the first result."""
        # A repeated call_id must not add a second item or place a second order.
        if call_id in self.by_call:
            return self.by_call[call_id]
        if name not in TOOL_NAMES:
            result = {"status": "error", "error": f"Unknown tool: {name}"}
        else:
            try:
                result = getattr(self, name)(json.loads(arguments))
            except (ValueError, TypeError, KeyError, StopIteration, AttributeError) as exc:
                result = {"status": "error", "error": str(exc)}
        self.by_call[call_id] = json.dumps(result)
        return self.by_call[call_id]

    def get_menu(self, args):
        """Return menu items matching ``category`` and ``query``, with modifiers and the tax rate."""
        query = str(args.get("query") or "").lower().strip()
        category = args.get("category") or "all"
        items = [
            {
                "id": x[0],
                "name": x[1],
                "category": x[2],
                "price": x[3],
                "sizes": SIZES if x[4] else None,
                "notes": x[5],
            }
            for x in MENU
            if (category == "all" or category == x[2]) and (not query or query in x[1].lower() or query in x[5].lower())
        ]
        return {"items": items, "modifiers": MODIFIERS, "tax_rate": 0.085}

    def manage_cart(self, args):
        """View or change the cart; returns the full cart afterwards."""
        action, status = args["action"], None
        if action == "add":
            item = next((x for x in MENU if x[0] == args.get("item_id")), None)
            if item is None:
                raise ValueError(f"Unknown item_id {args.get('item_id')!r}. Call get_menu for valid ids.")
            size = args.get("size")
            self.line_seq += 1
            self.lines.append(
                {
                    "line_id": f"line_{self.line_seq}",
                    "item_id": item[0],
                    "quantity": max(1, min(20, _whole_number(args.get("quantity"), 1))),
                    "size": (size if size in SIZES else "medium") if item[4] else None,
                    "modifiers": [m for m in args.get("modifiers") or [] if m in MODIFIERS],
                }
            )
            status = "added"
        elif action == "clear":
            self.lines = []
            status = "cleared"
        elif action in {"remove", "update_quantity"}:
            line = next((x for x in self.lines if x["line_id"] == args.get("line_id")), None)
            if line is None:
                raise ValueError(f"Unknown line_id {args.get('line_id')!r}. View the cart for its lines.")
            quantity = max(0, min(20, _whole_number(args.get("quantity"), 1)))
            if action == "remove" or quantity == 0:
                self.lines.remove(line)
                status = "removed"
            else:
                line["quantity"] = quantity
                status = "updated"
        elif action != "view":
            raise ValueError("Unsupported cart action")
        result = {"action": action, "cart": self.snapshot()}
        if status:
            result["status"] = status
        return result

    def place_order(self, args):
        """Return a read-back summary, or place the order when ``confirmed`` is true."""
        cart = self.snapshot()
        if not cart["lines"]:
            return {"status": "error", "error": "The cart is empty.", "cart": cart}
        if args.get("confirmed") is not True:
            return {
                "status": "needs_confirmation",
                "cart": cart,
                "total": cart["total"],
                "confirmation_summary": ", ".join(f"{x['quantity']} {x['name']}" for x in cart["lines"]),
                "next_step": "Read back the order and ask the user to confirm.",
            }
        self.order_seq += 1
        self.last_order = {
            "order_id": f"BB-{self.order_seq}",
            "customer_name": args.get("customer_name", "guest"),
            "ready_in_minutes": 4,
            "total": cart["total"],
            "lines": cart["lines"],
            "placed_at": datetime.now(UTC).isoformat(),
        }
        self.lines = []
        return {"status": "confirmed", "order": self.last_order, "cart": self.snapshot()}
