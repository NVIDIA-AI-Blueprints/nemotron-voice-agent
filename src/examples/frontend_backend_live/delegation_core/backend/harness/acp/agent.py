# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The agent interface the ACP client talks to (``backend.harness.agent``), and the plain ACP agent by default."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Literal, Protocol

from pydantic import ValidationError

from examples.shared.strict_settings import StrictSettings

from ..protocol import HarnessCapabilities, MessageChunk, NativeCompletion, NativeSteering, PromptEnded, RuntimeLaunch

if TYPE_CHECKING:
    from acp.client.connection import ClientSideConnection
    from acp.schema import AgentCapabilities, NewSessionResponse

    from .updates import SessionUpdates

CloseRequest = Callable[["ClientSideConnection", str], Awaitable[Any]]


class AcpSteering(NativeSteering, Protocol):
    """Adapter-specific producer and verifier attached to a single ACP worker."""

    def prompt(self, number: int, invocation: str, text: str) -> str:
        """Bind the original prompt to this verifier."""

    def capture(self, event: MessageChunk) -> bool:
        """Return whether the verifier took custody of this message chunk."""

    def finish(self, ended: PromptEnded) -> MessageChunk | None:
        """Settle the verifier and return any authorized final text."""

    async def close(self) -> None:
        """Retire the verifier and join its control requests."""


async def close_session(conn: ClientSideConnection, session_id: str) -> None:
    """End one session with ACP's ``session/close``."""
    await conn.close_session(session_id=session_id)


class AcpAgent:
    """A plain ACP agent (``agent: acp``), and the base of every agent class with ways of its own."""

    # Its key in AGENTS, which names its agents.<name> block; a subclass sets its own.
    name: ClassVar[str] = "acp"
    # None: no limit is known on the turns one process runs at once.
    turns_per_process: ClassVar[int | None] = None
    # The model of its agents.<name> block, whose values build it; a plain agent's block holds no key.
    Settings: ClassVar[type[StrictSettings]] = StrictSettings
    # ACP usage field descriptions do not establish a shared counter epoch or trustworthy per-invocation scope.
    capabilities: ClassVar[HarnessCapabilities] = HarnessCapabilities(delivery="cancel_reprompt")
    runtime_ownership: ClassVar[Literal["process_group", "external"]] = "process_group"

    @classmethod
    def from_settings(cls, block: Mapping[str, Any]) -> AcpAgent:
        """Build it from its ``backend.harness.agents.<name>`` block; a refused block raises ``ValueError``."""
        try:
            settings = cls.Settings.model_validate(block)
        except ValidationError as exc:
            # Raised again with the block's place: pydantic would merge the error into the harness block's without it.
            raise ValueError(f"backend.harness.agents.{cls.name}: {exc}") from None
        return cls(**settings.model_dump())

    def environment(self) -> Mapping[str, str]:
        """Return what its process needs in its environment when this side starts it: nothing."""
        return {}

    def prepare(self, folder: Path, instructions: str) -> tuple[str, ...]:
        """Write nothing into the session's folder: the instructions open the session's first prompt."""
        return (instructions.strip(),) if instructions.strip() else ()

    def prompt_text(self, text: str) -> str:
        """Return a prompt's text as the agent gets it: as written."""
        return text

    def bind_steering(
        self, launch: RuntimeLaunch, workspace: Path, response: NewSessionResponse, server: str, timeout: float
    ) -> AcpSteering | None:
        """Bind optional verified steering from this worker's actual ACP session reply."""
        return None

    def tool_name(self, title: Any) -> str:
        """Return a tool card's tool name from its title: the title as given."""
        return str(title or "").strip()

    def observed_arguments(self, name: str, arguments: dict[str, Any] | None, server: str) -> dict[str, Any] | None:
        """Return the arguments safe to retain in a tool observation."""
        return arguments

    def bridged(self, name: str, server: str) -> bool:
        """Return whether a tool card is a bridged call: every card is, so only the bridge reports calls."""
        return True

    def bridged_call(self, name: str, arguments: Mapping[str, Any] | None, server: str) -> bool:
        """Classify a card using this agent's structured metadata; plain ACP keeps its existing name policy."""
        return self.bridged(name, server)

    def unnamed_message(self, updates: SessionUpdates) -> str | None:
        """Return the id for a message chunk with none: the message being streamed, as ids are not stable ACP."""
        return updates.continued()

    def completion(self, event: PromptEnded) -> NativeCompletion:
        """Interpret ACP's terminal independently of output validation, inference, and usage accounting."""
        invoked = None
        if event.dispatch == "not_dispatched":
            invoked = False
        elif type(event.completion_calls) is int and event.completion_calls >= 0:
            invoked = event.completion_calls > 0
        if event.error is not None:
            outcome = "failed" if event.terminal_confirmed or event.dispatch == "not_dispatched" else "unknown"
            return NativeCompletion(outcome, invoked, event.error)
        if event.stop_reason == "cancelled":
            return NativeCompletion("cancelled", invoked, "the agent cancelled the foreground turn")
        if event.stop_reason != "end_turn":
            return NativeCompletion("unknown", invoked, f"the agent ended the turn {event.stop_reason}")
        return NativeCompletion("succeeded", invoked)

    def close_request(self, advertised: AgentCapabilities) -> CloseRequest | None:
        """Return what ends a session in the agent: ``session/close`` when its ``initialize`` answer advertised it.

        ``None`` sends nothing, and the agent keeps each session until its process ends.
        """
        sessions = advertised.session_capabilities
        return close_session if sessions is not None and sessions.close is not None else None
