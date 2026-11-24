# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Function schemas for the cafe tools the backend may call (Responses function-tool format)."""

CAFE_TOOLS = [
    {
        "type": "function",
        "name": "get_menu",
        "description": (
            "Look up Bluebird Cafe menu items with base prices, drink sizes and surcharges, and item notes. "
            "Also returns the modifier catalog and tax rate."
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
                    "description": "Optional case-insensitive substring match in item names or notes.",
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
            "Read or change the current cart. Returns the cart after the operation, including line items, "
            "subtotal, tax, and total."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["view", "add", "update_quantity", "update_options", "remove", "clear"],
                    "description": (
                        "update_quantity changes only the line's quantity. Use update_options with a size, "
                        "modifiers, or both to edit an existing line in one call, keeping its ID and quantity."
                    ),
                },
                "item_id": {
                    "type": "string",
                    "description": "Menu item id, required for add. Example: 'lat'.",
                },
                "line_id": {
                    "type": "string",
                    "description": "Cart line id, required for update_quantity, update_options, and remove.",
                },
                "quantity": {
                    "type": "integer",
                    "description": "Defaults to 1. Use 1 to 20 for add and 0 to 20 for update_quantity. "
                    "Updating to zero removes the line.",
                },
                "size": {
                    "type": "string",
                    "enum": ["small", "medium", "large"],
                    "description": (
                        "Drink size for add or update_options. Added drinks default to medium. "
                        "Omit for food and pastries or to keep the current size when editing."
                    ),
                },
                "modifiers": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "enum": ["oat_milk", "extra_shot", "decaf", "iced"],
                    },
                    "description": (
                        "Optional modifiers for add or update_options. On edit, supply the full replacement list, "
                        "use an empty list to clear it, or omit to keep current modifiers."
                    ),
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
            "Preview the current cart or submit it as a pickup order. A preview returns the full current cart. "
            "A successful submission returns the order receipt and clears the cart."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "customer_name": {
                    "type": "string",
                    "description": "Pickup name supplied by the caller, or 'guest' if they gave no name.",
                },
                "confirmed": {
                    "type": "boolean",
                    "description": "False previews the cart. Set true to order after the caller approves the readback.",
                },
            },
            "required": ["customer_name", "confirmed"],
            "additionalProperties": False,
        },
        "strict": False,
    },
]
