# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Function schemas for the cafe tools the thinker may call (Responses function-tool format)."""

CAFE_TOOLS = [
    {
        "type": "function",
        "name": "get_menu",
        "description": (
            "Look up Bluebird Cafe menu items with prices, available sizes, and modifiers. "
            "Call this before quoting any item or price."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "category": {
                    "type": "string",
                    "enum": ["all", "coffee", "tea", "pastry", "food"],
                    "description": "Limit results to one category. Defaults to all.",
                },
                "query": {
                    "type": "string",
                    "description": "Optional free-text filter matched against item names.",
                },
            },
            "required": [],
            "additionalProperties": False,
        },
        "strict": False,
    },
    {
        "type": "function",
        "name": "manage_cart",
        "description": (
            "Read or change the user's cart. Returns the full cart with line items, "
            "subtotal, tax, and total after the change. This is the source of truth for "
            "the order."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["view", "add", "update_quantity", "remove", "clear"],
                    "description": "The cart operation to perform.",
                },
                "item_id": {
                    "type": "string",
                    "description": "Menu item id, required for add. Example: 'lat'.",
                },
                "line_id": {
                    "type": "string",
                    "description": "Cart line id, required for update_quantity and remove.",
                },
                "quantity": {
                    "type": "integer",
                    "description": "Quantity for add or update_quantity. Defaults to 1.",
                },
                "size": {
                    "type": "string",
                    "enum": ["small", "medium", "large"],
                    "description": "Drink size for add. Ignored for food and pastries.",
                },
                "modifiers": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "enum": ["oat_milk", "extra_shot", "decaf", "iced"],
                    },
                    "description": "Optional modifiers for add.",
                },
            },
            "required": ["action"],
            "additionalProperties": False,
        },
        "strict": False,
    },
    {
        "type": "function",
        "name": "place_order",
        "description": (
            "Place the current cart as an order. Call with confirmed=false to get a "
            "read-back summary to confirm with the user, and confirmed=true only after "
            "the user has said yes. Takes a few seconds to complete."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "customer_name": {
                    "type": "string",
                    "description": "Name to call out when the order is ready.",
                },
                "confirmed": {
                    "type": "boolean",
                    "description": "True only if the user explicitly confirmed the order.",
                },
            },
            "required": ["customer_name", "confirmed"],
            "additionalProperties": False,
        },
        "strict": False,
    },
]
