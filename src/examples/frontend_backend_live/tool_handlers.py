# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pipecat function handlers for the tool-calling talker's routing tools."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING

from pipecat.frames.frames import FunctionCallResultProperties

if TYPE_CHECKING:
    from pipecat.services.llm_service import FunctionCallParams

    from examples.frontend_backend_live.talker import Calls

DELEGATED = {True: "delegated", False: "refused", None: "nothing_to_delegate"}
WAITED = "waited"


def build_handlers(calls: Calls) -> dict[str, Callable[[FunctionCallParams], Awaitable[None]]]:
    """Return the routing tools' handlers, bound to the session's voice ``calls``."""

    async def handle_delegate(params: FunctionCallParams) -> None:
        """Hand over the turn the call decided, and settle once the delegation part has answered."""
        await _settle(params, DELEGATED[await calls.hand_off(params.tool_call_id)])

    async def handle_wait(params: FunctionCallParams) -> None:
        """Settle at once: nothing is said and nothing is handed over."""
        await _settle(params, WAITED)

    return {"delegate": handle_delegate, "wait": handle_wait}


async def _settle(params: FunctionCallParams, status: str) -> None:
    """Settle the call with ``status``, and no run of the talker for it."""
    await params.result_callback({"status": status}, properties=FunctionCallResultProperties(run_llm=False))
