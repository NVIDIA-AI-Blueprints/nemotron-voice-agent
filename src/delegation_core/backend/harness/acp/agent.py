# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The agent interface the ACP client talks to (``backend.harness.agent``), and a plain ACP agent by default.

ACP leaves agent-specific behavior to each agent: where a session's instructions go, mid-turn steering, tool call
titles, which text its model wrote, and turns per process. On its own, :class:`AcpAgent` keeps to the protocol
(``agent: acp``): instructions open the first prompt, a newer request goes as the next prompt or by a cancel and a fresh
session, a chunk with no message id belongs to the message being streamed (message ids are not stable ACP), a tool
call's title is its name, and a turn that ended ``end_turn`` answered. A subclass lives in its own package, is named in
``acp/agents.py``, and sets ``name`` and ``keys``, which :meth:`AcpAgent.check_keys` reads.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

if TYPE_CHECKING:
    from ..protocol import MessageFacts
    from .updates import SessionUpdates


class AcpAgent:
    """A plain, standard ACP agent (``agent: acp``), and the interface every agent class of the ACP client keeps."""

    # Its key in the agents table (acp/agents.py), which its block's errors name; a subclass sets its own.
    name: ClassVar[str] = "acp"
    # How many turns one process runs at once; None: no limit is known.
    turns_per_process: ClassVar[int | None] = None
    # The keys its agents.<name> block may hold (check_keys refuses any other); a subclass with a block sets its own.
    keys: ClassVar[frozenset[str]] = frozenset()
    # The text that opens a mid-turn steer; None: it has none.
    steer_command: str | None = None

    @classmethod
    def from_settings(cls, block: Mapping[str, Any]) -> AcpAgent:
        """Build it from its ``backend.harness.agents.<name>`` block (none for a plain agent).

        Raises:
            ValueError: The block is not a mapping or has a key the class does not take.
        """
        cls.check_keys(block)
        return cls()

    @classmethod
    def check_keys(cls, block: Mapping[str, Any]) -> None:
        """Refuse a block that is not a mapping or names a key the class does not take.

        Raises:
            ValueError: The block is not a mapping or has an unknown key.
        """
        if not isinstance(block, Mapping):
            raise ValueError(f"backend.harness.agents.{cls.name} must be a mapping")
        unknown = set(block) - cls.keys
        if unknown:
            raise ValueError(
                f"backend.harness.agents.{cls.name} has unknown keys: {', '.join(sorted(map(str, unknown)))}"
            )

    @property
    def native_steer(self) -> bool:
        """Whether it takes a steer in the middle of a turn."""
        return self.steer_command is not None

    def environment(self) -> Mapping[str, str]:
        """What its process needs in its environment when this side starts it: nothing."""
        return {}

    def prepare(self, folder: Path, instructions: str) -> tuple[str, ...]:
        """Write nothing into the session's folder: the instructions open the session's first prompt."""
        return (instructions.strip(),) if instructions.strip() else ()

    def prompt_text(self, text: str) -> str:
        """A prompt's text as the agent gets it: as written."""
        return text

    def steer_text(self, text: str) -> str:
        """A mid-turn steer's text as the agent gets it: its steer command, then ``text``.

        Raises:
            RuntimeError: The agent has no mid-turn steer (the settings refuse ``native_command`` for it).
        """
        if self.steer_command is None:
            raise RuntimeError(f"The {self.name} agent takes no steer in the middle of a turn")
        return self.steer_command + text

    def tool_name(self, title: Any) -> str:
        """A tool card's tool name, from its title: the title as given."""
        return str(title or "").strip()

    def bridged(self, name: str, server: str) -> bool:
        """Whether a tool card is a bridged call: every card is, so only the bridge reports calls."""
        return True

    def unnamed_message(self, updates: SessionUpdates) -> str | None:
        """The id of a message chunk that came with none: the message being streamed (ids are not stable ACP)."""
        return updates.continued()

    def model_text(self, message: MessageFacts) -> bool:
        """Whether a message was written by a model call: every message of a plain agent is."""
        return True

    def no_answer(self, total: int | None, total_before: int | None) -> str | None:
        """Why a turn that ended ``end_turn`` gave no answer: a plain agent's always gave one."""
        return None
