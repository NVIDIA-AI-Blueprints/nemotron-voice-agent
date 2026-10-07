# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tiny in-memory tool set used by the REPL demo and by manual smoke runs.

Demo only: the data lives in a module-level dict and resets with the process.
Real deployments inject their own :class:`ToolSpec` list.
"""

from __future__ import annotations

from typing import Any

from examples.frontend_backend_verdict.text.tools import ToolSpec

_ORDERS: dict[str, dict[str, Any]] = {
    "5512": {"order_id": "5512", "status": "shipped", "shipped_on": "2026-09-19", "carrier": "Parcelflow"},
    "5513": {"order_id": "5513", "status": "processing", "placed_on": "2026-09-21"},
}


def get_order(order_id: str) -> dict[str, Any]:
    """Return one order record, or a not-found marker."""
    record = _ORDERS.get(str(order_id).strip())
    if record is None:
        return {"found": False, "order_id": order_id}
    return {"found": True, **record}


def cancel_order(order_id: str) -> dict[str, Any]:
    """Cancel an order that has not shipped yet."""
    key = str(order_id).strip()
    record = _ORDERS.get(key)
    if record is None:
        return {"cancelled": False, "reason": "not_found", "order_id": order_id}
    if record["status"] == "shipped":
        return {"cancelled": False, "reason": "already_shipped", "order_id": order_id}
    record["status"] = "cancelled"
    return {"cancelled": True, "order_id": order_id}


TOOLS: list[ToolSpec] = [
    ToolSpec(
        name="get_order",
        description="Look up one order by its id.",
        parameters={
            "type": "object",
            "properties": {"order_id": {"type": "string", "description": "The order id."}},
            "required": ["order_id"],
        },
        callable=get_order,
    ),
    ToolSpec(
        name="cancel_order",
        description="Cancel an order that has not shipped yet.",
        parameters={
            "type": "object",
            "properties": {"order_id": {"type": "string", "description": "The order id."}},
            "required": ["order_id"],
        },
        callable=cancel_order,
    ),
]
