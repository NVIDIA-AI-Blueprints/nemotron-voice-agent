# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""How a newer request reaches a running worker: the IM decides when, the agent class decides how."""

from __future__ import annotations

from examples.frontend_backend_live.delegation_core.backend.port import Steering

from .protocol import Agent, Steerable


class NativeCommand:
    """The agent's own mid-turn steer, for an agent class that has one (``Agent.native_steer``)."""

    def deliver(self, run: Steerable, steering: Steering) -> None:
        """Send the batch as the agent's steer while the turn runs, else as the next prompt.

        The steer joins at the turn's next model boundary; a batch not taken by the turn's end goes as the next prompt.
        """
        run.steer_in_turn(steering)


class CancelReprompt:
    """Cancel and re-prompt, for an agent with no mid-turn steer."""

    def deliver(self, run: Steerable, steering: Steering) -> None:
        """Cancel the turn once no bridged call runs, then send the batch as the session's next prompt.

        The session holds the history, so the cancelled text is never sent again. A turn that did not end ``cancelled``
        may not have taken the cancel, so a fresh session gets the history with the batch instead.
        """
        run.reprompt(steering)


def agent_steering(agent: Agent) -> NativeCommand | CancelReprompt:
    """Return the strategy of ``agent``'s class: its own mid-turn steer when it has one, else cancel and re-prompt.

    Either way, a turn that answers first ends the task; each batch it did not take goes back to the IM.
    """
    return NativeCommand() if agent.native_steer else CancelReprompt()
