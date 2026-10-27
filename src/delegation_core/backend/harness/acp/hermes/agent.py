# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Hermes over ACP (``backend.harness.agent: hermes``): the connection facts specific to Hermes.

Checked against ``hermes acp`` (stdio) at commit ``af26acab737db4382815aadcd73c06755471ab52``; from ``eb75627faf`` on,
``hermes acp`` is a viewer of Hermes's gateway daemon and these facts need checking again. Paths are Hermes's.

* **Instructions.** ``session/new`` has no instructions field. Hermes builds the system prompt once from ``SOUL.md`` and
  the project context of the session's ``cwd``: the nearest ``.hermes.md``, else an ``AGENTS.md`` chain from the git
  root (``agent/prompt_builder.py``). So instructions go into the session's folder, pinned with an empty ``.git`` so no
  parent file reaches it. A file that fails Hermes's threat scan is replaced by a ``[BLOCKED: ...]`` line (``:83-111``),
  so the texts avoid its patterns ("you are now a", "do not tell the user", ...).
* **Commands and the steer.** A prompt starting with ``/`` runs as a command (``acp_adapter/server.py:834-844``), so it
  gets ``command_guard``. ``/steer <text>`` (``commands.py:287-301``) appends the text as a user row after the running
  tool batch, or before the next model call, and never aborts a model call; its reply is the ack. A command's name ends
  at the first space (``commands.py:94-95``). An idle session runs a steer's text as a prompt (``server.py:719-727``).
* **Messages.** Each model call's message has its own UUID ``messageId``, shared with its reasoning; Hermes's own
  replies have none (``acp_adapter/events.py:185-236``). A tool card's title is ``<tool>: <preview>`` and the preview
  may hold raw arguments (``acp_adapter/tools.py:193-198``). MCP tools are named ``mcp__<server>__<tool>``.
* **Turn endings.** Provider errors, Hermes's own exits and an idle cancel's poisoned prompt also end in ``end_turn``,
  with one chunk of Hermes's own text, and usage is the session's running total (``turn_finalizer.py``). So a turn is
  answered only when its usage moved and its message came from a model call; a one-chunk reply that is a turn's only
  text cannot be told apart from Hermes's own on the wire, so it fails the task.
* **Cancel.** A cancel on an idle session makes the next prompt end unprocessed (``agent/turn_facade_lease.py:374-400``)
  and a cancelled prompt's text is put before the next prompt (``server.py:201-211``), so a cancel goes only to a
  running prompt and a cancelled session is never used again.
* **Turns.** One process runs at most 4 turns at once (``server.py:46``); a turn in a client tool holds its slot.
* **Only our tools.** ``HERMES_ACP_SKIP_CONFIGURED_MCP=1`` skips only the startup connect to the configured MCP servers
  (``acp_adapter/entry.py:210-216``); each ``session/new`` still waits for them (``acp_adapter/session.py:557-566``).
  The operator's ``platform_toolsets.acp: [no_mcp]`` keeps their tools out (``hermes_cli/tools_config.py:692-704``).
* **The bridge.** Hermes keys MCP servers by name per process, pauses a name for a minute after three error results in
  a row (``tools/mcp_tool.py:476``), fails an MCP connect silently, and probes an MCP URL with HEAD and GET. So the
  bridge gives each session a fresh name, never sets ``isError``, takes ``tools/list`` as readiness, and answers any
  method but POST with 405.
* **Permissions.** Hermes asks only before dangerous terminal commands and file edits
  (``acp_adapter/permissions.py:128-153``); the session's MCP tools never ask.
* **Close.** Hermes has no session close; on end-of-file on stdin it ends its sessions and exits 0
  (``acp_adapter/entry.py:226-231``).
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

from ..agent import AcpAgent

if TYPE_CHECKING:
    from ...protocol import MessageFacts
    from ..updates import SessionUpdates

# The prefix of a session's MCP tools in Hermes's tool cards.
TOOL_PREFIX = "mcp__{server}__"
# The instruction files Hermes reads in a session's folder (``agent/prompt_builder.py``); ``instructions_file`` must
# be one of them, or the session runs without its instructions.
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


class HermesAgent(AcpAgent):
    """Hermes's own ways over ACP (``agent: hermes``), with the values its ``agents.hermes`` block may set."""

    name: ClassVar[str] = "hermes"
    turns_per_process: ClassVar[int | None] = 4
    keys: ClassVar[frozenset[str]] = frozenset({"instructions_file", "steer_command", "command_guard"})

    def __init__(
        self, *, instructions_file: str = ".hermes.md", steer_command: str = "/steer ", command_guard: str = "Request: "
    ):
        """Hermes's file of instructions, its steer command, and the guard put before a prompt starting with ``/``."""
        self.instructions_file, self.steer_command, self.command_guard = instructions_file, steer_command, command_guard

    @classmethod
    def from_settings(cls, block: Mapping[str, Any]) -> HermesAgent:
        """Build it from ``backend.harness.agents.hermes``; every key has its default.

        Raises:
            ValueError: An unknown key, a value that is empty or not text, an instructions file Hermes does not read, a
                steer command with no space at its end, or a guard that would start a command itself.
        """
        cls.check_keys(block)
        for key, value in block.items():
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"backend.harness.agents.hermes.{key} must be non-empty text")
        if block.get("instructions_file", INSTRUCTIONS_FILES[0]) not in INSTRUCTIONS_FILES:
            raise ValueError(
                "backend.harness.agents.hermes.instructions_file must be a file name Hermes reads in a session's "
                f"folder: {', '.join(INSTRUCTIONS_FILES)}"
            )
        if not block.get("steer_command", " ")[-1].isspace():
            raise ValueError(
                "backend.harness.agents.hermes.steer_command must end with a space: Hermes reads a command's name up "
                "to the first space (quote it in YAML)"
            )
        if block.get("command_guard", "").lstrip().startswith("/"):
            raise ValueError(
                'backend.harness.agents.hermes.command_guard must not start with "/": Hermes would run it as a command'
            )
        return cls(**block)

    def environment(self) -> Mapping[str, str]:
        """A started ``hermes acp`` skips the connect to the MCP servers of its own configuration at startup."""
        return {"HERMES_ACP_SKIP_CONFIGURED_MCP": "1"}

    def prepare(self, folder: Path, instructions: str) -> tuple[str, ...]:
        """Pin the session's folder with an empty ``.git``, write the instructions as its ``.hermes.md``; no blocks."""
        (folder / ".git").mkdir(exist_ok=True)
        if instructions.strip():
            (folder / self.instructions_file).write_text(instructions.strip() + "\n", encoding="utf-8")
        return ()

    def prompt_text(self, text: str) -> str:
        """A text that would start with ``/`` gets the guard first, so it never runs as one of Hermes's commands."""
        stripped = text.lstrip()
        return self.command_guard + stripped if stripped.startswith("/") else text

    def steer_text(self, text: str) -> str:
        """Its steer command, then ``text`` guarded as a prompt's: an idle Hermes runs a steer's text as a prompt."""
        return self.steer_command + self.prompt_text(text)

    def tool_name(self, title: Any) -> str:
        """The tool's name: the card's title up to the preview, which may hold raw arguments."""
        return str(title or "").split(": ", 1)[0].strip()

    def bridged(self, name: str, server: str) -> bool:
        """A call of the session's bridged tools: its name has the session's ``mcp__<server>__`` prefix."""
        return name.startswith(TOOL_PREFIX.format(server=server))

    def unnamed_message(self, updates: SessionUpdates) -> str | None:
        """A chunk with no message id is Hermes's own reply (an ack, a command's reply), never model text."""
        return None

    def model_text(self, message: MessageFacts) -> bool:
        """Streamed by a model call: two chunks or more, or one after reasoning once its turn had streamed text."""
        return message.chunks >= 2 or (message.thought and message.after_text)

    def no_answer(self, total: int | None, total_before: int | None) -> str | None:
        """No answer when the turn reported no usage, or when its usage did not move: no model call completed."""
        if total is None:
            return "the agent reported no usage: no model call completed (its own text)"
        if total_before is not None and total == total_before:
            return "the agent's usage did not move: no model call completed (its own text)"
        return None
