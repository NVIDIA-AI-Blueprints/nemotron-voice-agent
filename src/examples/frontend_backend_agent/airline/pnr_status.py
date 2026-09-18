# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Thinker PNR-status tool implementation."""

from __future__ import annotations

from typing import Any

from loguru import logger

from examples.frontend_backend_agent.airline.backend import BookingBackend
from examples.frontend_backend_agent.airline.branding import user_facing_record
from examples.frontend_backend_agent.airline.slot_parsing import canonical_pnr, slot
from examples.frontend_backend_agent.src.protocol import response_hint, tool_result


async def pnr_status(*, backend: BookingBackend, slots: dict[str, Any]) -> dict[str, Any]:
    """Look up a PNR and return a Thinker payload."""
    pnr = slot(slots, "pnr_code")
    if pnr is None:
        return response_hint(
            reason="params_missing",
            action="req_params",
            params_needed=["pnr_code"],
            response_text="Sure, could you tell me your PNR number?",
            context="pnr_status",
        )
    pnr = canonical_pnr(pnr)
    try:
        record = await backend.get_pnr(pnr)
    except Exception as exc:
        logger.warning(f"pnr_status backend failed: {exc}")
        return response_hint(
            reason="tool_error",
            action="retry",
            params_needed=["pnr_code"],
            error=str(exc),
            response_text="I could not check that PNR right now. Please try again.",
            context="pnr_status",
        )
    if record is None:
        return response_hint(
            reason="tool_error",
            action="req_params",
            params_needed=["pnr_code"],
            error=f"PNR {pnr.upper()} was not found.",
            response_text=f"I could not find PNR {pnr.upper()}. Could you check the code?",
            context="pnr_status",
        )
    return tool_result(
        tool="pnr_status",
        status="success",
        data={"booking": user_facing_record(record)},
        response_text=(
            f"PNR {record['pnr']} is {record['status']} for {record['origin_city']} to "
            f"{record['dest_city']} on {record['date']}."
        ),
        context="pnr_status",
    )
