# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pipecat function handlers for the tool-calling talker's routing tools.

Built as the Frontend/Backend Agent's (``examples.frontend_backend_agent.src.tool_handlers``), one per name of
``examples.frontend_backend_live.models.decision.TOOL_DECISIONS``, and registered with ``cancel_on_interruption=False``
(``examples.frontend_backend_live.talker.TalkerTools``), so the caller's words never cancel one. The voice has acted on
the routing decision by the time Pipecat runs a handler, so each only settles its call, with ``run_llm=False``: the
voice says whatever follows, and Pipecat writes the call and its result into the talker's context.

* ``delegate``: the voice handed the turn over as it acted on the decision
  (``examples.frontend_backend_live.handoffs``): its lead-in is said once the delegation part accepts it, after the
  part's delay, or the refusal line when the part refuses it. The call settles once the part has answered.
* ``wait``: nothing is said; the call settles at once.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING

from pipecat.frames.frames import FunctionCallResultProperties

if TYPE_CHECKING:
    from pipecat.services.llm_service import FunctionCallParams

    from examples.frontend_backend_live.talker import Calls

# What a delegate call's result tells the talker, by whether the delegation part accepted the turn; None: nothing was
# handed over.
DELEGATED = {True: "delegated", False: "refused", None: "nothing_to_delegate"}
WAITED = "waited"


def build_handlers(calls: Calls) -> dict[str, Callable[[FunctionCallParams], Awaitable[None]]]:
    """Return the routing tools' handlers, bound to the session's voice ``calls``."""

    async def handle_delegate(params: FunctionCallParams) -> None:
        """Settle once the delegation part has answered the hand-off the call's decision made."""
        await _settle(params, DELEGATED[await calls.hand_off(params.tool_call_id)])

    async def handle_wait(params: FunctionCallParams) -> None:
        """Settle at once: nothing is said and nothing is handed over."""
        await _settle(params, WAITED)

    return {"delegate": handle_delegate, "wait": handle_wait}


async def _settle(params: FunctionCallParams, status: str) -> None:
    """Settle the call with ``status``, and no run of the talker for it."""
    await params.result_callback({"status": status}, properties=FunctionCallResultProperties(run_llm=False))
