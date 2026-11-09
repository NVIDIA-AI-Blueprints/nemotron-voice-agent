# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The harness's interfaces, the events a session reports to its run, and the tables of the classes it loads."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from importlib import import_module
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from examples.frontend_backend_live.delegation_core.backend.port import Steering

if TYPE_CHECKING:
    from .sessions import KeptSession


class EventReader(Protocol):
    """A reader of a session's events, a worker's run or a conductor's turn; each event's ``apply`` calls its method."""

    def message_chunk(self, event: MessageChunk) -> None:
        """Read model text, or, with no message id, the agent's own reply to a command or an absorbed prompt."""

    def thought_chunk(self, event: ThoughtChunk) -> None:
        """Read reasoning of a model call; only its message id counts."""

    def tool_card(self, event: ToolCard) -> None:
        """Read a tool call the agent announces."""

    def tool_card_end(self, event: ToolCardEnd) -> None:
        """Read the end of a tool call the agent announced."""

    def plan(self, event: PlanUpdate) -> None:
        """Read the agent's changed plan."""

    def user_text(self, event: UserText) -> None:
        """Read a prompt the agent replays or drains itself: a model call boundary."""

    def prompt_ended(self, event: PromptEnded) -> None:
        """Read the end of one prompt the client sent."""

    def app_call(self, event: AppCall) -> None:
        """Read a call of the app's functions that the bridge holds."""

    def call_running(self, event: CallRunning) -> None:
        """Read a call the bridge runs itself (a tool source's or a work tool's), or refused."""

    def call_settled(self, event: CallSettled) -> None:
        """Read the end of a call the bridge ran."""

    def wake(self, event: Wake) -> None:
        """Read the run's own wake-up: steering waits, or a stop."""


@dataclass(frozen=True)
class MessageChunk:
    """Text of an assistant message; ``message_id`` is ``None`` for the agent's own reply (an ack, a command's)."""

    message_id: str | None
    text: str

    def apply(self, reader: EventReader) -> None:
        """Hand the event to ``reader``."""
        reader.message_chunk(self)


@dataclass(frozen=True)
class ThoughtChunk:
    """Reasoning of a model call; only its id is kept, naming the message the reasoning begins."""

    message_id: str | None

    def apply(self, reader: EventReader) -> None:
        """Hand the event to ``reader``."""
        reader.thought_chunk(self)


@dataclass(frozen=True)
class ToolCard:
    """A tool call the agent announces; ``bridged`` when the bridge reports it, ``ended`` when it is already over."""

    call_id: str
    name: str
    arguments: Mapping[str, Any] | None = None
    bridged: bool = True
    ended: ToolCardEnd | None = None

    def apply(self, reader: EventReader) -> None:
        """Hand the event to ``reader``."""
        reader.tool_card(self)


@dataclass(frozen=True)
class ToolCardEnd:
    """The end of a tool call the agent announced, with its output text."""

    call_id: str
    failed: bool
    output: str = ""

    def apply(self, reader: EventReader) -> None:
        """Hand the event to ``reader``."""
        reader.tool_card_end(self)


@dataclass(frozen=True)
class PlanUpdate:
    """The agent's plan: one line per entry."""

    entries: tuple[str, ...]

    def apply(self, reader: EventReader) -> None:
        """Hand the event to ``reader``."""
        reader.plan(self)


@dataclass(frozen=True)
class UserText:
    """A prompt the agent replays or drains itself (not one the client sent)."""

    text: str

    def apply(self, reader: EventReader) -> None:
        """Hand the event to ``reader``."""
        reader.user_text(self)


@dataclass(frozen=True)
class PromptEnded:
    """The end of one prompt: its stop reason and the session's running usage, or the error that ended it."""

    prompt: int
    stop_reason: str | None
    usage: Mapping[str, Any] | None = None
    error: str | None = None

    def apply(self, reader: EventReader) -> None:
        """Hand the event to ``reader``."""
        reader.prompt_ended(self)


@dataclass(eq=False)
class HeldCall:
    """A call the bridge holds for the worker: its function call item, and the answer the runtime waits for."""

    call: dict
    answer: asyncio.Future = field(repr=False)

    @property
    def call_id(self) -> str:
        """The call's id (the bridge's own: the agent's protocol carries none)."""
        return self.call["call_id"]

    def reply(self, text: str) -> None:
        """Answer the agent's call with ``text`` (a plain result: the bridge never flags an error)."""
        if not self.answer.done():
            self.answer.set_result(text)


@dataclass(frozen=True)
class AppCall:
    """A call of the app's functions that the bridge holds until the run forwards it and the app answers."""

    held: HeldCall

    def apply(self, reader: EventReader) -> None:
        """Hand the event to ``reader``."""
        reader.app_call(self)


@dataclass(frozen=True)
class CallRunning:
    """A call the bridge took: run by the bridge itself, or ``refused`` by the IM's route and not run."""

    call: dict
    refused: bool = False

    def apply(self, reader: EventReader) -> None:
        """Hand the event to ``reader``."""
        reader.call_running(self)


@dataclass(frozen=True)
class CallSettled:
    """The end of a call the bridge ran, once reported."""

    call_id: str

    def apply(self, reader: EventReader) -> None:
        """Hand the event to ``reader``."""
        reader.call_settled(self)


@dataclass(frozen=True)
class Wake:
    """The run's own wake-up (steering waits, or a stop): the reader looks at its state again."""

    def apply(self, reader: EventReader) -> None:
        """Hand the event to ``reader``."""
        reader.wake(self)


SessionEvent = (
    MessageChunk
    | ThoughtChunk
    | ToolCard
    | ToolCardEnd
    | PlanUpdate
    | UserText
    | PromptEnded
    | AppCall
    | CallRunning
    | CallSettled
    | Wake
)
# One per session, of SessionEvent in wire order; every producer runs on the core's event loop.
Mailbox = asyncio.Queue


@dataclass(frozen=True)
class McpServer:
    """One session's door on the bridge, as the agent is told it: a server name, its URL and its token."""

    name: str
    url: str
    token: str


@dataclass(frozen=True)
class RuntimeLaunch:
    """What one runtime process is started with or reached at: its command or URL, its environment, its work folder."""

    command: tuple[str, ...]
    env: Mapping[str, str]
    url: str | None
    folder: Path


class HarnessSession(Protocol):
    """One session of the agent: a worker's or a conductor's."""

    id: str

    @property
    def outstanding(self) -> frozenset[int]:
        """The prompts sent and not ended yet."""

    def prompt(self, text: str) -> int:
        """Send ``text`` as a prompt (never as a command of the agent); return its number. Its end is a PromptEnded."""

    def steer(self, text: str) -> int:
        """Send ``text`` as the agent's own mid-turn steer (``Agent.native_steer``); return its number."""

    async def cancel(self) -> bool:
        """Cancel the running prompt, if any, and return whether a cancel went out; it still ends with a PromptEnded."""

    async def ended(self, prompt: int) -> None:
        """Return once ``prompt`` has ended (its PromptEnded is in the mailbox)."""

    async def idle(self) -> None:
        """Return once no prompt is outstanding."""

    def retire(self) -> None:
        """Stop using the session; an agent with no session close keeps it until its process ends."""


class HarnessClient(Protocol):
    """A connection to one runtime process over its protocol."""

    @property
    def alive(self) -> bool:
        """Whether the connection is open and the process runs."""

    async def open(self) -> None:
        """Connect and initialize; raise if the agent does not answer or speaks another protocol version."""

    async def new_session(self, cwd: Path, server: McpServer, mailbox: Mailbox) -> HarnessSession:
        """Open a session working in ``cwd`` that reaches the bridge through ``server``; events go to ``mailbox``."""

    async def close(self) -> None:
        """Close the connection and end the process; every prompt still outstanding ends with an error."""


class Agent(Protocol):
    """The agent behind the protocol (``backend.harness.agent``), as the harness reads it."""

    name: str
    turns_per_process: int | None
    native_steer: bool

    def environment(self) -> Mapping[str, str]:
        """Return what its process needs in its environment when this side starts it (``stdio``)."""

    def prepare(self, folder: Path, instructions: str) -> tuple[str, ...]:
        """Ready a session's folder (its ``cwd``) with its instructions; return what opens its first prompt."""

    def no_answer(self, total: int | None, total_before: int | None) -> str | None:
        """Return why an ``end_turn`` turn gave no answer, given the token totals after and before it; else ``None``."""


class HarnessProtocol(Protocol):
    """The agent's protocol (``backend.harness.protocol``), built once from the block by its ``from_settings``."""

    agent: Agent

    def client(
        self,
        launch: RuntimeLaunch,
        *,
        permission: PermissionAnswer,
        start_timeout_seconds: float,
        stop_grace_seconds: float,
    ) -> HarnessClient:
        """Return a client for the runtime process of ``launch``; it connects nothing until ``open()``."""


@dataclass(frozen=True)
class Role:
    """What a runtime process serves: its name and its sessions' MCP server names' prefix."""

    name: str
    prefix: str


class Runtime(Protocol):
    """One runtime process: its client, its work folder, the names its sessions' MCP servers take, its places taken."""

    client: HarnessClient
    places: int

    @property
    def folder(self) -> Path:
        """The process's work folder."""

    def server_name(self, prefix: str) -> str:
        """Return a new MCP server name, unique in this process (``w1``, ``w2``, ...)."""

    def session_folder(self, name: str) -> Path:
        """Make a new folder for one session inside the process's work folder."""

    def leave(self) -> None:
        """Give back the place a session took (``Runtimes.shared``), once its prompts have ended."""

    async def close(self) -> None:
        """End the process and delete its work folder."""


class Runtimes(Protocol):
    """Where a session's runtime comes from (the adapter): a process its role shares, or the session it keeps."""

    async def shared(self, role: Role) -> Runtime:
        """Return a process of ``role`` with a turn free, its place taken until ``Runtime.leave``."""

    def take_place(self, runtime: Runtime) -> bool:
        """Take a place in ``runtime`` for the session kept there, while it has a turn free and its client is alive."""

    def take_kept(self) -> KeptSession | None:
        """Take the kept session out of the slot, so two tasks never share it; ``None`` when none is kept."""

    def keep(self, kept: KeptSession) -> bool:
        """Keep ``kept``, the session of a task that answered, while none is kept; ``False``: the run retires it."""


class Steerable(Protocol):
    """What a steering strategy asks of a worker's run (double dispatch: the strategy picks, the run carries out)."""

    def steer_in_turn(self, steering: Steering) -> None:
        """Send the batch as the agent's own mid-turn steer while the turn runs; else as the next prompt."""

    def reprompt(self, steering: Steering) -> None:
        """Cancel the turn once no bridged call runs; then the batch goes as the session's next prompt."""


class SteerStrategy(Protocol):
    """How a newer request reaches a running worker: the agent class picks it (``steering.agent_steering``)."""

    def deliver(self, run: Steerable, steering: Steering) -> None:
        """Hand ``steering`` to ``run`` the way this strategy does."""


class PermissionAnswer(Protocol):
    """The answer to an agent's permission request (``backend.harness.permission``); no approval is surfaced."""

    def choose(self, options: list[tuple[str, str]]) -> str | None:
        """Return the id of the option to answer with, from ``(option_id, kind)`` pairs; ``None`` answers cancelled."""


# Each name in backend.harness to a loader, so a class and its packages load only when a core is built with it.
Loader = Callable[[], Any]

PROTOCOLS: Mapping[str, Loader] = {
    "acp": lambda: import_module("examples.frontend_backend_live.delegation_core.backend.harness.acp").AcpProtocol,
}
PERMISSIONS: Mapping[str, Loader] = {
    "reject": lambda: (
        import_module("examples.frontend_backend_live.delegation_core.backend.harness.permissions").Reject
    ),
}
