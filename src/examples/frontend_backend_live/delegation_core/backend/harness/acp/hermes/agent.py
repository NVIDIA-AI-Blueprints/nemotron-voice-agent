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

from ...protocol import HarnessCapabilities, RuntimeLaunch
from ..agent import AcpAgent
from .evidence import PRIVATE_ORIGIN, OriginStore, origin_reference
from .observer import _assigned_tool, observer_installed
from .steer import HermesSteering

if TYPE_CHECKING:
    from acp.schema import NewSessionResponse

    from ..updates import SessionUpdates

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


def _not_a_command(guard: str) -> str:
    if guard.lstrip().startswith("/"):
        raise ValueError('must not start with "/": Hermes would run it as a command')
    return guard


class HermesSettings(StrictSettings):
    """The ``backend.harness.agents.hermes`` block: Hermes's file of instructions and command guard."""

    instructions_file: Annotated[Text, AfterValidator(_read_by_hermes)] = ".hermes.md"
    command_guard: Annotated[Text, AfterValidator(_not_a_command)]


class HermesAgent(AcpAgent):
    """Hermes's own ways over ACP (``agent: hermes``), with the values its ``agents.hermes`` block may set."""

    name: ClassVar[str] = "hermes"
    turns_per_process: ClassVar[int | None] = 4
    Settings: ClassVar[type[HermesSettings]] = HermesSettings
    # Hermes's ACP gateway projects a native session snapshot; field names do not prove per-prompt scope.
    capabilities: ClassVar[HarnessCapabilities] = HarnessCapabilities(
        delivery="cancel_reprompt", counter_scope="unscoped"
    )
    runtime_ownership = "external"

    def __init__(self, *, instructions_file: str, command_guard: str):
        """Take Hermes's instructions file and the guard put before a prompt starting with ``/``."""
        self.instructions_file, self.command_guard = instructions_file, command_guard

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

    def bind_steering(
        self, launch: RuntimeLaunch, workspace: Path, response: NewSessionResponse, server: str, timeout: float
    ) -> HermesSteering | None:
        """Use private evidence only for an explicitly prepared local Hermes profile."""
        selected = launch.env.get("HERMES_HOME")
        if launch.url is not None or not selected:
            return None
        home = Path(selected).expanduser()
        native_id = _gateway_session_id(response)
        if native_id is None or not home.is_absolute() or not observer_installed(home):
            return None
        return HermesSteering(OriginStore.create(workspace, server), native_id, home, launch.env, timeout)

    def tool_name(self, title: Any) -> str:
        """Return the tool's name from the card's title, ``<tool>: <preview>``, whose preview may hold raw arguments."""
        return str(title or "").split(": ", 1)[0].strip()

    def bridged(self, name: str, server: str) -> bool:
        """Recognize this session's legacy or editor-namespaced MCP display name, without granting execution."""
        return _assigned_tool(name, server) is not None

    def bridged_call(self, name: str, arguments: Mapping[str, Any] | None, server: str) -> bool:
        """Let the MCP door own pure structured MCP batches; never hide mixed, foreign or malformed wrappers."""
        if self.bridged(name, server):
            return True
        if name != "tool_call" or not isinstance(arguments, Mapping) or set(arguments) != {"calls"}:
            return False
        calls = arguments["calls"]
        return (
            isinstance(calls, list)
            and bool(calls)
            and all(
                isinstance(call, Mapping)
                and set(call) == {"name", "arguments"}
                and isinstance(call["name"], str)
                and isinstance(call["arguments"], Mapping)
                and self.bridged(call["name"], server)
                for call in calls
            )
        )

    def observed_arguments(self, name: str, arguments: dict[str, Any] | None, server: str) -> dict[str, Any] | None:
        """Keep private origin references out of observations for this session's MCP tools."""
        if arguments is None:
            return None
        if self.bridged(name, server):
            return _without_origin(arguments)
        if name == "tool_call" and self.bridged_call(name, arguments, server):
            return {"calls": [{**call, "arguments": _without_origin(call["arguments"])} for call in arguments["calls"]]}
        return arguments

    def unnamed_message(self, updates: SessionUpdates) -> str | None:
        """Return ``None``: a chunk with no message id is Hermes's own reply, such as an ack, never model text."""
        return None


def _gateway_session_id(response: NewSessionResponse) -> str | None:
    """Use Hermes's structured lineage only when it belongs to the created ACP session."""
    hermes = (response.field_meta or {}).get("hermes")
    provenance = hermes.get("sessionProvenance") if isinstance(hermes, Mapping) else None
    if not isinstance(provenance, Mapping) or provenance.get("acpSessionId") != response.session_id:
        return None
    identity = provenance.get("rootHermesSessionId")
    return identity if isinstance(identity, str) and identity else None


def _without_origin(arguments: dict[str, Any]) -> dict[str, Any]:
    if not origin_reference(arguments.get(PRIVATE_ORIGIN)):
        return arguments
    return {key: value for key, value in arguments.items() if key != PRIVATE_ORIGIN}
