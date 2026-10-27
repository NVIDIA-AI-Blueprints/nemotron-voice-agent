# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Apply the client's commentary, thinking and instruction appends to the cascade.

* Commentary is a fact to say: queued in arrival order, worded with whatever else is ready, said at idle, and kept
  in the talker's memory once said. In client mode it may answer a client delegation.
* Thinking is silent context for the talker's later turns and, through the heard log, for the next delegation.
* An instruction is pinned in the talker's memory and noted for the backend; it cuts the line in progress for good
  and is decided in turn with any waiting caller turn.

Commentary and thinking drop every pending acknowledgement; an instruction does not. The protocol layer validates and
acknowledges the commands.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable

from examples.frontend_backend_live.cascade.heard_log import HeardLog
from examples.frontend_backend_live.cascade.memory import TalkerMemory
from examples.frontend_backend_live.cascade.presenter import Presenter
from examples.frontend_backend_live.cascade.turns import Acknowledgements, CallerTurns
from examples.frontend_backend_live.cascade.utterances import Utterances


class ApplicationInput:
    """Applies the application's commentary, thinking and instructions to one call."""

    def __init__(
        self,
        memory: TalkerMemory,
        heard_log: HeardLog,
        presenter: Presenter,
        turns: CallerTurns,
        acknowledgements: Acknowledgements,
        utterances: Utterances,
        *,
        interrupt: Callable[[], Awaitable[None]] | None = None,
    ):
        """Wire the records an append writes and the parts it reaches.

        Args:
            memory: The talker's memory.
            heard_log: Keeps the context the backend should see with the next delegation.
            presenter: Says the commentary, in turn.
            turns: Decides an instruction, in turn with the caller's turns.
            acknowledgements: The pending acknowledgements that commentary and thinking drop.
            utterances: The lines said, whose cut by an instruction is never resumed.
            interrupt: Awaited to cut off the line in progress.
        """
        self.memory, self.heard_log, self.presenter, self.turns = memory, heard_log, presenter, turns
        self.acknowledgements, self.utterances, self.interrupt = acknowledgements, utterances, interrupt

    def commentary(self, text: str, delegation_id: str | None = None) -> None:
        """Queue a fact from the application to be said; ``delegation_id`` names the client delegation it answers."""
        self.acknowledgements.drop_all()
        self.presenter.commentary(text, delegation_id)

    def thinking(self, text: str) -> None:
        """Keep silent context for later decisions and for the next delegation; nothing is spoken."""
        self.acknowledgements.drop_all()
        self.memory.app_context(text)
        self.heard_log.context(text)

    async def instruction(self, text: str) -> None:
        """Steer behavior: pin the direction, cut the line in progress, and decide it in turn."""
        self.memory.instruction(text)
        self.heard_log.instruction(text)
        if self.interrupt:
            await self.interrupt()
        # Before any other await: the tracker's cut of the line comes later, and a turn with no words never resumes it.
        self.utterances.cut_on_purpose()
        await self.turns.decide_instruction(text)
