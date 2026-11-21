# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""How a newer request reaches a running worker: the IM decides when, the agent class decides how."""

from __future__ import annotations

from examples.frontend_backend_live.delegation_core.backend.port import Steering

from .protocol import Agent, Steerable, SteerStrategy


class NextPrompt:
    """Retain steering until the current prompt ends, preserving the native conversation."""

    cooperative_end = True

    def deliver(self, run: Steerable, steering: Steering) -> None:
        """Queue input without cancelling the current invocation or inferring application from streamed output."""
        run.prompt_next(steering)


class CancelReprompt(NextPrompt):
    """Queue the correction and let the run cancel after active calls settle."""

    cooperative_end = False


def agent_steering(agent: Agent) -> SteerStrategy:
    """Select the adapter's declared same-session continuation policy."""
    return {"next_prompt": NextPrompt, "cancel_reprompt": CancelReprompt}[agent.capabilities.delivery]()
