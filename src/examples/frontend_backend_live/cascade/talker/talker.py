# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The talker: one routing decision per caller turn.

Joins the client's instructions with the prompt set's routing rules, asks the frontend model for a
``{action, speech}`` decision, and applies the no-silence floor: a turn always gets a reply or a
delegation (``never_silent``).
"""

from examples.frontend_backend_live.models.base import Frontend
from examples.frontend_backend_live.models.decision import TalkerDecision
from examples.frontend_backend_live.prompts import PromptSet


def never_silent(decision: TalkerDecision) -> TalkerDecision:
    """Return ``decision``, or a delegation when it would leave the caller with nothing.

    Routing has no silent outcome: a turn that reached the talker gets a reply or a delegation.
    ``listen`` and a ``speak`` with no words both leave the caller with nothing, and a prompt rule
    against them measurably does not hold. The backend answer is always voiced, so delegating is the
    safe default; a wasted delegation costs seconds, dead air costs the call. Commentary keeps its own
    optional silence and does not use this.
    """
    if decision.action == "delegate" or (decision.action == "speak" and decision.speech.strip()):
        return decision
    reason = "listen_removed" if decision.action == "listen" else "empty_speak"
    return TalkerDecision("delegate", "", {**decision.diagnostics, "recovery": reason})


class Talker:
    """Routes each caller turn to a spoken reply or a delegation."""

    def __init__(self, frontend: Frontend, prompts: PromptSet):
        """Bind the frontend model and the prompt set."""
        self.frontend, self.prompts = frontend, prompts

    async def decide(self, instructions: str, history: list[dict], work_in_progress: str = "") -> TalkerDecision:
        """Return the routing decision for ``history`` under the client's ``instructions``.

        ``work_in_progress`` describes the delegations still open, when there are any.
        """
        routing = instructions + "\n" + self.prompts.routing
        if work_in_progress:
            routing += "\n" + work_in_progress
        decision = await self.frontend.complete(routing, history)
        return never_silent(decision)
