# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Hermes over ACP (``backend.harness.agent: hermes``): the ways of ``hermes acp`` that ACP leaves to the agent."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, ClassVar

from pydantic import AfterValidator

from examples.frontend_backend_live.delegation_core.config import Text
from examples.shared.strict_settings import StrictSettings

from ..agent import AcpAgent

if TYPE_CHECKING:
    from ..updates import SessionUpdates

TOOL_PREFIX = "mcp__{server}__"
# The files Hermes reads instructions from in a session's folder; with any other, a session runs without them.
INSTRUCTIONS_FILES = (
    ".hermes.md",
    "HERMES.md",
    "AGENTS.override.md",
    "AGENTS.md",
    "agents.md",
    "CLAUDE.md",
    "claude.md",
    ".cursorrules",
)


def _read_by_hermes(name: str) -> str:
    if name not in INSTRUCTIONS_FILES:
        raise ValueError(f"must be a file name Hermes reads in a session's folder: {', '.join(INSTRUCTIONS_FILES)}")
    return name


def _ends_with_space(command: str) -> str:
    if not command[-1].isspace():
        raise ValueError(
            "must end with a space: Hermes reads a command's name up to the first space (quote it in YAML)"
        )
    return command


def _not_a_command(guard: str) -> str:
    if guard.lstrip().startswith("/"):
        raise ValueError('must not start with "/": Hermes would run it as a command')
    return guard


class HermesSettings(StrictSettings):
    """The ``backend.harness.agents.hermes`` block: Hermes's file of instructions, its steer command, its guard."""

    instructions_file: Annotated[Text, AfterValidator(_read_by_hermes)] = ".hermes.md"
    steer_command: Annotated[Text, AfterValidator(_ends_with_space)] = "/steer "
    command_guard: Annotated[Text, AfterValidator(_not_a_command)]


class HermesAgent(AcpAgent):
    """Hermes's own ways over ACP (``agent: hermes``), with the values its ``agents.hermes`` block may set."""

    name: ClassVar[str] = "hermes"
    turns_per_process: ClassVar[int | None] = 4
    Settings: ClassVar[type[HermesSettings]] = HermesSettings

    def __init__(self, *, instructions_file: str, steer_command: str, command_guard: str):
        """Take Hermes's instructions file, its steer command, and the guard put before a prompt starting with ``/``."""
        self.instructions_file, self.steer_command, self.command_guard = instructions_file, steer_command, command_guard

    def environment(self) -> Mapping[str, str]:
        """Return the environment that skips Hermes's startup connect to the MCP servers of its own configuration."""
        return {"HERMES_ACP_SKIP_CONFIGURED_MCP": "1"}

    def prepare(self, folder: Path, instructions: str) -> tuple[str, ...]:
        """Write the instructions as the session folder's ``instructions_file``, pinned by an empty ``.git``.

        Hermes reads them from its ``cwd``'s project files, as ``session/new`` has no instructions field, and the
        ``.git`` keeps parent files out. Hermes blocks a file failing its threat scan, so the texts avoid its patterns.
        """
        (folder / ".git").mkdir(exist_ok=True)
        if instructions.strip():
            (folder / self.instructions_file).write_text(instructions.strip() + "\n", encoding="utf-8")
        return ()

    def prompt_text(self, text: str) -> str:
        """Put the guard before a text that starts with ``/``, which Hermes would run as one of its commands."""
        stripped = text.lstrip()
        return self.command_guard + stripped if stripped.startswith("/") else text

    def steer_text(self, text: str) -> str:
        """Return the steer command and the guarded text, since an idle Hermes runs a steer's text as a prompt.

        Hermes puts a steer after the running tool batch, or before the next model call, and its reply is the ack.
        """
        return self.steer_command + self.prompt_text(text)

    def tool_name(self, title: Any) -> str:
        """Return the tool's name from the card's title, ``<tool>: <preview>``, whose preview may hold raw arguments."""
        return str(title or "").split(": ", 1)[0].strip()

    def bridged(self, name: str, server: str) -> bool:
        """Return whether a call is one of the session's bridged tools, named ``mcp__<server>__<tool>``."""
        return name.startswith(TOOL_PREFIX.format(server=server))

    def unnamed_message(self, updates: SessionUpdates) -> str | None:
        """Return ``None``: a chunk with no message id is Hermes's own reply, such as an ack, never model text."""
        return None

    def no_answer(self, total: int | None, total_before: int | None) -> str | None:
        """Return why an ``end_turn`` turn gave no answer: no model call completed unless its usage moved.

        Hermes also ends provider errors and its own exits ``end_turn``, with its own text, and reports running usage.
        """
        if total is None:
            return "the agent reported no usage: no model call completed (its own text)"
        if total_before is not None and total == total_before:
            return "the agent's usage did not move: no model call completed (its own text)"
        return None
