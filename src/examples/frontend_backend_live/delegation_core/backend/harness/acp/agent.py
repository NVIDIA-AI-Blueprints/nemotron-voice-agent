# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The agent interface the ACP client talks to (``backend.harness.agent``), and the plain ACP agent by default."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

from pydantic import ValidationError

from examples.shared.strict_settings import StrictSettings

if TYPE_CHECKING:
    from .updates import SessionUpdates


class AcpAgent:
    """A plain ACP agent (``agent: acp``), and the base of every agent class with ways of its own."""

    # Its key in AGENTS, which names its agents.<name> block; a subclass sets its own.
    name: ClassVar[str] = "acp"
    # None: no limit is known on the turns one process runs at once.
    turns_per_process: ClassVar[int | None] = None
    # The model of its agents.<name> block, whose values build it; a plain agent's block holds no key.
    Settings: ClassVar[type[StrictSettings]] = StrictSettings
    # The text that opens a mid-turn steer; None: it takes no mid-turn steer.
    steer_command: str | None = None

    @classmethod
    def from_settings(cls, block: Mapping[str, Any]) -> AcpAgent:
        """Build it from its ``backend.harness.agents.<name>`` block; a refused block raises ``ValueError``."""
        try:
            settings = cls.Settings.model_validate(block)
        except ValidationError as exc:
            # Raised again with the block's place: pydantic would merge the error into the harness block's without it.
            raise ValueError(f"backend.harness.agents.{cls.name}: {exc}") from None
        return cls(**settings.model_dump())

    @property
    def native_steer(self) -> bool:
        """Whether it takes a steer in the middle of a turn: its runs steer with it, or else cancel and re-prompt."""
        return self.steer_command is not None

    def environment(self) -> Mapping[str, str]:
        """Return what its process needs in its environment when this side starts it: nothing."""
        return {}

    def prepare(self, folder: Path, instructions: str) -> tuple[str, ...]:
        """Write nothing into the session's folder: the instructions open the session's first prompt."""
        return (instructions.strip(),) if instructions.strip() else ()

    def prompt_text(self, text: str) -> str:
        """Return a prompt's text as the agent gets it: as written."""
        return text

    def steer_text(self, text: str) -> str:
        """Return a mid-turn steer's text as the agent gets it: its steer command, then ``text``."""
        if self.steer_command is None:
            raise RuntimeError(f"The {self.name} agent takes no steer in the middle of a turn")
        return self.steer_command + text

    def tool_name(self, title: Any) -> str:
        """Return a tool card's tool name from its title: the title as given."""
        return str(title or "").strip()

    def bridged(self, name: str, server: str) -> bool:
        """Return whether a tool card is a bridged call: every card is, so only the bridge reports calls."""
        return True

    def unnamed_message(self, updates: SessionUpdates) -> str | None:
        """Return the id for a message chunk with none: the message being streamed, as ids are not stable ACP."""
        return updates.continued()

    def no_answer(self, total: int | None, total_before: int | None) -> str | None:
        """Return why an ``end_turn`` turn gave no answer: ``None``, since a plain agent's always gave one."""
        return None
