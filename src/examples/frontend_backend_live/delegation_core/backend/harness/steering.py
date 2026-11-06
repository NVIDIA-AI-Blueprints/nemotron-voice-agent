# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""How a newer request reaches a running worker: the IM decides when, the agent class how (:func:`agent_steering`).

An agent class that takes a steer in the middle of a turn (``Agent.native_steer``, such as Hermes's ``/steer``) gets
:class:`NativeCommand`: the agent's own steer, and a batch the turn did not take goes as the next prompt. Every other
agent gets :class:`CancelReprompt`: the turn is cancelled once no bridged call runs, and when it has ended, the batch
goes as the session's next prompt, never re-sending the cancelled text, since the session holds the history. A turn
that did not end ``cancelled`` after the cancel went out may not have taken it, so the task's history with the batch
then goes to a fresh session. Each strategy hands the batch to the run's matching method (double dispatch). A turn that
answers first ends the task, and every batch it did not take goes back to the IM, which starts it as the next request.
"""

from __future__ import annotations

from examples.frontend_backend_live.delegation_core.backend.port import Steering

from .protocol import Agent, Steerable


class NativeCommand:
    """The agent's own mid-turn steer."""

    def deliver(self, run: Steerable, steering: Steering) -> None:
        """While the turn runs, the agent's steer; otherwise the next prompt."""
        run.steer_in_turn(steering)


class CancelReprompt:
    """Cancel the turn, then the batch as the session's next prompt."""

    def deliver(self, run: Steerable, steering: Steering) -> None:
        """Cancel and re-prompt, once no bridged call runs."""
        run.reprompt(steering)


def agent_steering(agent: Agent) -> NativeCommand | CancelReprompt:
    """How a newer request reaches ``agent``'s running worker: its own mid-turn steer when its class has one."""
    return NativeCommand() if agent.native_steer else CancelReprompt()
