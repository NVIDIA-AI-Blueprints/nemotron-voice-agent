# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The harness's own seams: its interfaces, the events a session reports, and their tables.

Each seam is an interface with implementations named in ``backend.harness`` and resolved by a table of loaders, so a
class loads only when a core is built with it: :class:`HarnessProtocol` (``protocol``, such as ``acp``), which opens a
:class:`HarnessClient` per runtime process; :class:`SteerStrategy` (``steer``); and :class:`PermissionAnswer`
(``permission``). Everything a session reports reaches its run through one mailbox, in wire order: the agent's updates,
each prompt's :class:`PromptEnded`, the bridge's calls and the run's own :class:`Wake`. Each event applies itself to an
:class:`EventReader` (double dispatch), so no reader switches on a type.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from importlib import import_module
from pathlib import Path
from typing import Any, ClassVar, Protocol

from delegation_core.backend.port import Steering


# --------------------------------------------------------------------------------------------- events
class EventReader(Protocol):
    """What reads a session's events: a worker's run or a conductor's turn."""

    def message_chunk(self, event: MessageChunk) -> None:
        """Model text, or (no message id) the agent's own reply to an absorbed prompt or a command."""

    def thought_chunk(self, event: ThoughtChunk) -> None:
        """Reasoning text of a model call; only its message id is kept."""

    def tool_card(self, event: ToolCard) -> None:
        """The agent announces a tool call."""

    def tool_card_end(self, event: ToolCardEnd) -> None:
        """A tool call the agent announced has ended."""

    def plan(self, event: PlanUpdate) -> None:
        """The agent's plan changed."""

    def user_text(self, event: UserText) -> None:
        """The agent replays or drains a prompt of its own: a model call boundary."""

    def prompt_ended(self, event: PromptEnded) -> None:
        """One prompt the client sent has ended."""

    def app_call(self, event: AppCall) -> None:
        """The bridge holds a call of the app's functions."""

    def call_running(self, event: CallRunning) -> None:
        """The bridge runs a call itself (a tool source's or a work tool's), or refused one."""

    def call_settled(self, event: CallSettled) -> None:
        """A call the bridge ran has ended."""

    def wake(self, event: Wake) -> None:
        """The run's own wake-up: steering waits, or a stop."""


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
    """Reasoning of a model call. The text is not kept: the id proves a model call streamed that message."""

    message_id: str | None

    def apply(self, reader: EventReader) -> None:
        """Hand the event to ``reader``."""
        reader.thought_chunk(self)


@dataclass(frozen=True)
class ToolCard:
    """A tool call the agent announces: its id, the tool's name, its arguments, and whose call it is.

    ``bridged``: one of the session's bridged tools, which the bridge reports itself; else a tool the agent runs itself.
    """

    call_id: str
    name: str
    arguments: Mapping[str, Any] | None = None
    bridged: bool = True

    def apply(self, reader: EventReader) -> None:
        """Hand the event to ``reader``."""
        reader.tool_card(self)


@dataclass(frozen=True)
class ToolCardEnd:
    """A tool call the agent announced has ended, with its output text."""

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
    """One prompt has ended: its stop reason and usage, or the error that ended it (a lost connection included).

    ``usage`` is the agent's report as it gave it: ACP counts it across all of the session's turns.
    """

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
    """The bridge holds a call of one of the app's functions until the run forwards it and the app answers."""

    held: HeldCall

    def apply(self, reader: EventReader) -> None:
        """Hand the event to ``reader``."""
        reader.app_call(self)


@dataclass(frozen=True)
class CallRunning:
    """A call the bridge took: ``refused`` (the IM's route said so; it is not run) or run by the bridge itself."""

    call: dict
    refused: bool = False

    def apply(self, reader: EventReader) -> None:
        """Hand the event to ``reader``."""
        reader.call_running(self)


@dataclass(frozen=True)
class CallSettled:
    """A call the bridge ran has ended and was reported."""

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


# A session's events, as its mailbox carries them.
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
Mailbox = asyncio.Queue  # of SessionEvent, one per session; every producer runs on the core's event loop


# --------------------------------------------------------------------------------------------- the agent
@dataclass(frozen=True)
class McpServer:
    """One session's door on the bridge, as the agent is told it: a server name, its URL and its token."""

    name: str  # unique within one runtime process (an agent may key MCP servers by name per process)
    url: str
    token: str


@dataclass(frozen=True)
class RuntimeLaunch:
    """What one runtime process is started with or reached at: its command or URL, its environment, its work folder."""

    command: tuple[str, ...]
    env: Mapping[str, str]  # a started process's whole environment
    url: str | None
    folder: Path  # the process's work folder: a started process runs in it


class HarnessSession(Protocol):
    """One session of the agent: a worker's or a conductor's."""

    id: str

    @property
    def outstanding(self) -> frozenset[int]:
        """The prompts sent and not ended yet."""

    def prompt(self, text: str) -> int:
        """Send ``text`` as a prompt (never as a command of the agent); return its number. Its end is a PromptEnded."""

    def steer(self, text: str) -> int:
        """Send ``text`` as the agent's own mid-turn steer (``Agent.native_steer``); return its number.

        Its end is a PromptEnded: the agent took it for the running turn, or ran it as a turn of its own.
        """

    async def cancel(self) -> bool:
        """Cancel the running prompt, only while one is outstanding: an idle session has no turn to cancel.

        Returns whether a cancel was sent. A cancelled session is never used again: an agent may carry a cancelled
        turn into the session's next prompt.
        """

    async def ended(self, prompt: int) -> None:
        """Return once ``prompt`` has ended (its PromptEnded is in the mailbox)."""

    async def idle(self) -> None:
        """Return once no prompt is outstanding."""

    def retire(self) -> None:
        """Never use the session again (an agent with no session close keeps it until its process ends)."""


class HarnessClient(Protocol):
    """A connection to one runtime process over its protocol."""

    @property
    def alive(self) -> bool:
        """Whether the connection is open and the process runs."""

    async def open(self) -> None:
        """Connect and initialize; raise if the agent does not answer or speaks another protocol version."""

    async def new_session(self, cwd: Path, server: McpServer, mailbox: Mailbox) -> HarnessSession:
        """A new session working in ``cwd`` that reaches the bridge through ``server``; events go to ``mailbox``."""

    async def close(self) -> None:
        """Close the connection and end the process; every prompt still outstanding ends with an error."""


class MessageFacts(Protocol):
    """How one assistant message of a turn arrived, as the run saw it."""

    chunks: int  # the text chunks it came in
    thought: bool  # a reasoning chunk carried its id
    after_text: bool  # its turn had streamed text before it began


class Agent(Protocol):
    """The agent behind the protocol (``backend.harness.agent``), as the harness reads it."""

    name: str
    turns_per_process: int | None  # how many turns one of its processes runs at once; None: no limit is known
    native_steer: bool  # whether it takes a steer in the middle of a turn (``HarnessSession.steer``)

    def environment(self) -> Mapping[str, str]:
        """What its process needs in its environment when this side starts it (the ``stdio`` transport)."""

    def prepare(self, folder: Path, instructions: str) -> tuple[str, ...]:
        """Ready a session's folder (its ``cwd``) with its instructions; return what opens its first prompt."""

    def model_text(self, message: MessageFacts) -> bool:
        """Whether a message was written by a model call, rather than by the agent itself."""

    def no_answer(self, total: int | None, total_before: int | None) -> str | None:
        """Why a turn that ended ``end_turn`` gave no answer the run may pass on (``None``: it gave one).

        ``total`` is the session's token total the turn's end reported (``None``: none), ``total_before`` the one
        reported before the turn began.
        """


class HarnessProtocol(Protocol):
    """The agent's protocol (``backend.harness.protocol``), built once from the block: it reaches one process at a time.

    Its builder, ``from_settings(block)``, reads what its own tables name (for ACP: ``transport``, ``agent`` and its
    ``agents`` block) and refuses what it cannot run with.
    """

    agent: Agent

    def client(
        self,
        launch: RuntimeLaunch,
        *,
        permission: PermissionAnswer,
        start_timeout_seconds: float,
        stop_grace_seconds: float,
    ) -> HarnessClient:
        """A client for one runtime process, started by ``launch.command`` or reached at ``launch.url``.

        It connects nothing until ``open()``.
        """


@dataclass(frozen=True)
class Role:
    """What a runtime process serves: its name and its sessions' MCP server names' prefix."""

    name: str  # conductor | workers
    prefix: str  # the conductor's servers are c1, c2, ...; the workers' w1, w2, ...


class Runtime(Protocol):
    """One runtime process: its client, its work folder, the names its sessions' MCP servers take, its places taken."""

    client: HarnessClient
    places: int  # sessions that hold one of its turns

    @property
    def folder(self) -> Path:
        """The process's work folder."""

    def server_name(self, prefix: str) -> str:
        """A new MCP server name, unique in this process (``w1``, ``w2``, ...)."""

    def session_folder(self, name: str) -> Path:
        """A new folder for one session, inside the process's work folder."""

    def leave(self) -> None:
        """A session gives back the place it took (``Runtimes.shared``), once its prompts have ended."""

    async def close(self) -> None:
        """End the process and delete its work folder."""


class Runtimes(Protocol):
    """Where a session's runtime comes from (the adapter): a process its role shares."""

    async def shared(self, role: Role) -> Runtime:
        """A process of ``role`` with a turn free, its place taken for the caller's session.

        The session gives the place back with ``Runtime.leave``. Another process opens when every one of the role's
        is full or lost.
        """


class Steerable(Protocol):
    """What a steering strategy asks of a worker's run (double dispatch: the strategy picks, the run carries out)."""

    def steer_in_turn(self, steering: Steering) -> None:
        """Send the batch as the agent's own mid-turn steer while the turn runs; else as the next prompt."""

    def prompt_next(self, steering: Steering) -> None:
        """Send the batch as the next prompt once the turn ends."""

    def reprompt(self, steering: Steering) -> None:
        """Cancel the turn once no bridged call runs, then open a fresh session whose first prompt has the batch."""


class SteerStrategy(Protocol):
    """How a newer request reaches a running worker (``backend.harness.steer``)."""

    native: ClassVar[bool]  # it needs the agent's own mid-turn steer

    def deliver(self, run: Steerable, steering: Steering) -> None:
        """Hand ``steering`` to ``run`` the way this strategy does."""


class PermissionAnswer(Protocol):
    """The answer to an agent's permission request (``backend.harness.permission``); no approval is surfaced."""

    def choose(self, options: list[tuple[str, str]]) -> str | None:
        """The id of the option to answer with, from ``(option_id, kind)`` pairs; ``None`` answers "cancelled"."""


# A conductor's ``place``, as the bridge's conductor door calls it.
Placer = Callable[[Any], Awaitable[str]]

# --------------------------------------------------------------------------------------------- the tables
# Each name in backend.harness to its class, loaded only when the core is built with it (the protocol's packages too).
Loader = Callable[[], Any]

PROTOCOLS: Mapping[str, Loader] = {
    "acp": lambda: import_module("delegation_core.backend.harness.acp").AcpProtocol,
}
STEERING: Mapping[str, Loader] = {
    "native_command": lambda: import_module("delegation_core.backend.harness.steering").NativeCommand,
    "next_prompt": lambda: import_module("delegation_core.backend.harness.steering").NextPrompt,
    "cancel_reprompt": lambda: import_module("delegation_core.backend.harness.steering").CancelReprompt,
}
PERMISSIONS: Mapping[str, Loader] = {
    "reject": lambda: import_module("delegation_core.backend.harness.permissions").Reject,
}
