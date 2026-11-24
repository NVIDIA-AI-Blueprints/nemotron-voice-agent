# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The harness's interfaces, the events a session reports to its run, and the tables of the classes it loads."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from importlib import import_module
from pathlib import Path
from typing import Any, Literal, Protocol

from nemotron_voice_agent.pipecat.examples.frontend_backend_live.delegation_core.backend.port import (
    RefusalDetails,
    TaskContext,
)


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

    def prompt_dispatched(self, event: PromptDispatched) -> None:
        """Read local submission of an exact prompt; this does not prove semantic input consumption."""

    def app_call(self, event: AppCall) -> None:
        """Read a call of the app's functions that the bridge holds."""

    def admission_ready(self, event: AdmissionReady) -> None:
        """Recheck a call held before admission after placement or its lifecycle changed."""

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
    prompt: int | None = None

    def apply(self, reader: EventReader) -> None:
        """Hand the event to ``reader``."""
        reader.message_chunk(self)


@dataclass(frozen=True)
class ThoughtChunk:
    """Reasoning of a model call; only its id is kept, naming the message the reasoning begins."""

    message_id: str | None
    prompt: int | None = None

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
    prompt: int | None = None

    def apply(self, reader: EventReader) -> None:
        """Hand the event to ``reader``."""
        reader.tool_card(self)


@dataclass(frozen=True)
class ToolCardEnd:
    """The end of a tool call the agent announced, with its output text."""

    call_id: str
    failed: bool
    output: str = ""
    prompt: int | None = None

    def apply(self, reader: EventReader) -> None:
        """Hand the event to ``reader``."""
        reader.tool_card_end(self)


@dataclass(frozen=True)
class PlanUpdate:
    """The agent's plan: one line per entry."""

    entries: tuple[str, ...]
    prompt: int | None = None

    def apply(self, reader: EventReader) -> None:
        """Hand the event to ``reader``."""
        reader.plan(self)


@dataclass(frozen=True)
class UserText:
    """A prompt the agent replays or drains itself (not one the client sent)."""

    text: str
    prompt: int | None = None

    def apply(self, reader: EventReader) -> None:
        """Hand the event to ``reader``."""
        reader.user_text(self)


@dataclass(frozen=True)
class PromptEnded:
    """The end of one prompt: its stop reason, agent-reported counters, or the error that ended it.

    Counter scope comes from the native contract. Exact invocation counters override the session snapshot
    when present. An empty invocation mapping records unavailable usage. Completion calls are evidence of
    model invocation independent of token usage.
    """

    prompt: int
    stop_reason: str | None
    usage: Mapping[str, Any] | None = None
    error: str | None = None
    completion_calls: int | None = None
    invocation_usage: Mapping[str, Any] | None = None
    invocation_id: str | None = None
    dispatch: Literal["not_dispatched", "dispatched", "unknown"] = "unknown"
    terminal_confirmed: bool = False

    def apply(self, reader: EventReader) -> None:
        """Hand the event to ``reader``."""
        reader.prompt_ended(self)


@dataclass(frozen=True)
class PromptDispatched:
    """One prompt passed its final local fence and was submitted to the native connection."""

    prompt: int
    invocation_id: str

    def apply(self, reader: EventReader) -> None:
        """Hand the exact local submission receipt to ``reader``."""
        reader.prompt_dispatched(self)


@dataclass(frozen=True)
class NativeCompletion:
    """Typed invocation outcome, separately from a valid task result, inference, usage, or external effects."""

    outcome: Literal["succeeded", "failed", "cancelled", "unknown"]
    model_invoked: bool | None = None
    error: str | None = None


CancelDispatch = Literal["not_dispatched", "dispatched", "unknown"]


@dataclass(frozen=True)
class HarnessCapabilities:
    """Native guarantees, bound to an initialized connection and refreshed for every created session."""

    continuation: bool = True
    stop_scope: Literal["foreground"] = "foreground"
    stop_confirmation: Literal["terminal"] = "terminal"
    tool_mediation: Literal["unknown", "bridge_only"] = "unknown"
    event_attribution: Literal["serialized_session"] = "serialized_session"
    counter_scope: Literal["invocation", "cumulative_epoch", "unscoped"] = "unscoped"


@dataclass(frozen=True)
class SessionBinding:
    """The connection, native session, role, and peer for which the capabilities were selected."""

    connection_id: str
    session_id: str
    protocol: str
    protocol_version: int
    transport: str
    role: str
    peer_name: str | None = None
    peer_version: str | None = None


@dataclass
class PromptState:
    """Local lifecycle of one native invocation; dispatched is not a remote consumption receipt."""

    invocation_id: str
    phase: Literal["scheduled", "dispatched", "ended"] = "scheduled"
    dispatched: bool = False
    terminal: PromptEnded | None = None


@dataclass(frozen=True)
class NativeCleanup:
    """Cleanup evidence for its stated scope; a closed connection does not establish native runtime termination."""

    scope: Literal["connection", "session", "runtime"]
    confirmed: bool
    workspace_safe: bool


@dataclass(frozen=True)
class RefusedReply:
    """A bridge-owned refusal result; application strings never become this control record."""

    text: str
    details: RefusalDetails


def refused_reply(task: TaskContext, call_id: str, *, fallback: str | None = None) -> str | RefusedReply:
    """Keep legacy text when a context supplies no typed, exactly bound refusal snapshot."""
    read = getattr(task, "refusal_details", None)
    try:
        details = read(call_id) if read is not None else None
    except KeyError:
        details = None
    text = task.refusal(call_id) if fallback is None else fallback
    if isinstance(details, RefusalDetails) and (details.work_id, details.task_id, details.call_id) == (
        task.work_id,
        task.task_id,
        call_id,
    ):
        return RefusedReply(text, details)
    return text


@dataclass(eq=False)
class HeldCall:
    """A call the bridge holds for the worker: its function call item, and the answer the runtime waits for."""

    call: dict
    answer: asyncio.Future = field(repr=False)

    @property
    def call_id(self) -> str:
        """The call's id (the bridge's own: the agent's protocol carries none)."""
        return self.call["call_id"]

    def reply(self, text: str | RefusedReply) -> None:
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
class AdmissionReady:
    """Placement waiting changed; recheck queued calls and the turn's remaining reading budget."""

    def apply(self, reader: EventReader) -> None:
        """Hand the event to ``reader``."""
        reader.admission_ready(self)


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
    | PromptDispatched
    | AppCall
    | AdmissionReady
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
    capabilities: HarnessCapabilities
    binding: SessionBinding

    @property
    def outstanding(self) -> frozenset[int]:
        """The prompts sent and not ended yet."""

    @property
    def alive(self) -> bool:
        """Whether it can still serve: not retired, and not ended or lost as far as its protocol can tell."""

    @property
    def quiescent(self) -> bool:
        """Whether no submitted invocation remains active or has an uncertain native terminal."""

    def prompt(self, text: str, *, may_dispatch: Callable[[], bool] | None = None) -> int:
        """Send ``text`` as a prompt (never as a command of the agent); return its number. Its end is a PromptEnded."""

    def prompt_state(self, prompt: int) -> PromptState:
        """Return the lifecycle of the exact scheduled invocation; an unknown number raises ``KeyError``."""

    def on_prompt_terminal(self, prompt: int, callback: Callable[[PromptEnded], None]) -> Callable[[], None]:
        """Subscribe to that exact recorded terminal; the subscriber decides which guarantee it establishes."""

    async def cancel(self) -> CancelDispatch:
        """Request cancellation of the active prompt and report local submission independently of its terminal."""

    async def ended(self, prompt: int) -> None:
        """Return once ``prompt`` has ended (its PromptEnded is in the mailbox)."""

    async def idle(self) -> None:
        """Return once local prompt waiters settle; inspect ``quiescent`` for a native-idle guarantee."""

    def retire(self) -> bool:
        """Stop using it, and close it in the agent if the agent has a session close; return whether a close went out.

        An agent with no session close keeps the session until its process ends.
        """


class HarnessClient(Protocol):
    """A connection to one runtime process over its protocol."""

    @property
    def alive(self) -> bool:
        """Whether the connection is open and the process runs."""

    def on_lost(self, callback: Callable[[], None]) -> None:
        """Notify the runtime owner once the connection is lost, including loss before registration."""

    async def open(self) -> None:
        """Connect and initialize; raise if the agent does not answer or speaks another protocol version."""

    async def new_session(
        self, cwd: Path, server: McpServer, mailbox: Mailbox, *, role: str = "worker"
    ) -> HarnessSession:
        """Open a session working in ``cwd`` that reaches the bridge through ``server``; events go to ``mailbox``."""

    async def close(self) -> None:
        """Close the connection; ``wait_closed`` reports what native termination scope was actually confirmed."""

    async def wait_closed(self) -> NativeCleanup:
        """Join owned connection cleanup and return what native scope was actually confirmed."""


class Agent(Protocol):
    """The agent behind the protocol (``backend.harness.agent``), as the harness reads it."""

    name: str
    turns_per_process: int | None
    capabilities: HarnessCapabilities

    def environment(self) -> Mapping[str, str]:
        """Return what its process needs in its environment when this side starts it (``stdio``)."""

    def prepare(self, folder: Path, instructions: str) -> tuple[str, ...]:
        """Ready a session's folder (its ``cwd``) with its instructions; return what opens its first prompt."""

    def completion(self, event: PromptEnded) -> NativeCompletion:
        """Interpret typed terminal evidence without inferring model execution or task success from token counts."""


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
    cleanup_receipt: NativeCleanup

    @property
    def folder(self) -> Path:
        """The process's work folder."""

    def server_name(self, prefix: str) -> str:
        """Return a new MCP server name, unique in this process (``w1``, ``w2``, ...)."""

    def session_folder(self, name: str) -> Path:
        """Make a new folder for one session inside the process's work folder."""

    def leave(self) -> None:
        """Give back the place a session took, once its prompts have ended."""

    async def close(self) -> None:
        """End the process and delete its work folder."""

    async def wait_closed(self) -> NativeCleanup:
        """Return cleanup evidence for this owner, without equating connection closure with physical shutdown."""


class PermissionAnswer(Protocol):
    """The answer to an agent's permission request (``backend.harness.permission``); no approval is surfaced."""

    def choose(self, options: list[tuple[str, str]]) -> str | None:
        """Return the id of the option to answer with, from ``(option_id, kind)`` pairs; ``None`` answers cancelled."""


# Each name in backend.harness to a loader, so a class and its packages load only when a core is built with it.
Loader = Callable[[], Any]

PROTOCOLS: Mapping[str, Loader] = {
    "acp": lambda: (
        import_module(
            "nemotron_voice_agent.pipecat.examples.frontend_backend_live.delegation_core.backend.harness.acp"
        ).AcpProtocol
    ),
}
PERMISSIONS: Mapping[str, Loader] = {
    "reject": lambda: (
        import_module(
            "nemotron_voice_agent.pipecat.examples.frontend_backend_live.delegation_core.backend.harness.permissions"
        ).Reject
    ),
}
