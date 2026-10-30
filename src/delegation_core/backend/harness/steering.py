# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""How a newer request reaches a running worker (``backend.harness.steer``): the IM decides when, the adapter how.

``native_command`` sends the agent's own mid-turn steer, only for an agent that takes one, and a batch the turn did not
take goes as the next prompt; ``next_prompt`` (the default) waits for the turn's end; ``cancel_reprompt`` cancels the
turn once no bridged call runs and sends the task's history with the batch to a fresh session, never re-sending the
cancelled text. Each strategy hands the batch to the run's matching method (double dispatch). A turn that answers
first ends the task, and every batch it did not take goes back to the IM, which starts it as the next request.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, ClassVar

from delegation_core.backend.port import Steering

from .protocol import Steerable


class NativeCommand:
    """``native_command``: the agent's own mid-turn steer."""

    native: ClassVar[bool] = True

    @classmethod
    def from_settings(cls, block: Mapping[str, Any]) -> NativeCommand:
        """It reads no key: the agent class words the steer."""
        return cls()

    def deliver(self, run: Steerable, steering: Steering) -> None:
        """While the turn runs, the agent's steer; otherwise the next prompt."""
        run.steer_in_turn(steering)


class NextPrompt:
    """``next_prompt``: the batch goes as the next prompt, once the turn ends."""

    native: ClassVar[bool] = False

    @classmethod
    def from_settings(cls, block: Mapping[str, Any]) -> NextPrompt:
        """It reads no key."""
        return cls()

    def deliver(self, run: Steerable, steering: Steering) -> None:
        """The batch waits for the turn's end."""
        run.prompt_next(steering)


class CancelReprompt:
    """``cancel_reprompt``: cancel the turn, then a fresh session whose first prompt carries the batch."""

    native: ClassVar[bool] = False

    @classmethod
    def from_settings(cls, block: Mapping[str, Any]) -> CancelReprompt:
        """It reads no key."""
        return cls()

    def deliver(self, run: Steerable, steering: Steering) -> None:
        """Cancel and re-prompt, once no bridged call runs."""
        run.reprompt(steering)
