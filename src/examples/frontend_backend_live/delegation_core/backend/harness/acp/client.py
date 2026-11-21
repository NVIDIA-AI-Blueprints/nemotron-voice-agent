# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The ACP client (``backend.harness.protocol: acp``) on the ACP SDK's client connection."""

from __future__ import annotations

import asyncio
import contextlib
import itertools
import json
import uuid
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

from acp import PROTOCOL_VERSION, RequestError, connect_to_agent
from acp.client.connection import ClientSideConnection
from acp.connection import StreamDirection, StreamEvent
from acp.meta import CLIENT_METHODS
from acp.schema import (
    AgentCapabilities,
    AllowedOutcome,
    ClientCapabilities,
    DeniedOutcome,
    FileSystemCapabilities,
    HttpHeader,
    HttpMcpServer,
    Implementation,
    RequestPermissionResponse,
    TextContentBlock,
)
from acp.task import InMemoryMessageStateStore
from loguru import logger

from ..protocol import (
    CancelDispatch,
    Mailbox,
    McpServer,
    MessageChunk,
    NativeCleanup,
    PermissionAnswer,
    PromptDispatched,
    PromptEnded,
    PromptState,
    RuntimeLaunch,
    SessionBinding,
    ToolCard,
)
from ..usage import InvocationUsage
from .agent import AcpAgent, AcpSteering, CloseRequest
from .transports import Transport
from .updates import SessionUpdates

CLIENT_INFO = Implementation(name="delegation-core-harness", version="1")
SESSION_UPDATE = CLIENT_METHODS["session_update"]
SESSION_CANCEL = "session/cancel"
LOST = "the connection to the agent is lost"
CLOSED = "the client closed"
# A cap on an agent's error data in a reason, which goes on the wire in a task's error text.
ERROR_DATA_CHARS = 300
NOT_FOUND = -32002


class PromptNotDispatched(RequestError):
    """A local dispatch fence rejected an operation before any bytes were written."""

    def __init__(self):
        """Identify a known-unstarted local rejection without manufacturing a native RPC failure."""
        super().__init__(-32001, "the prompt was fenced before native dispatch")


class _CancelNotDispatched(Exception):
    """The exact target stopped being active before the cancel notification could be written."""


class _Sender:
    """ACP sender injection: serialize writes and recheck prompt eligibility at the actual write boundary."""

    def __init__(self, owner: AcpClient, writer: asyncio.StreamWriter):
        self.owner, self.writer = owner, writer
        self._writing = asyncio.Lock()
        self._closed = False

    async def send(self, payload: dict[str, Any]) -> None:
        async with self._writing:
            if self._closed:
                raise ConnectionError(CLOSED)
            if not self.owner._may_write(payload):
                if payload.get("method") == SESSION_CANCEL:
                    raise _CancelNotDispatched
                self.owner._rpc.reject_outgoing(payload["id"], PromptNotDispatched())
                return
            await self.owner.transport.send_record(
                self.writer,
                (json.dumps(payload, separators=(",", ":")) + "\n").encode("utf-8"),
                lambda: self.owner._wrote(payload),
            )

    async def close(self) -> None:
        self._closed = True


def reason(exc: BaseException) -> str:
    """Return why a request failed: the error, then the agent's own reason from its JSON-RPC error's data, cut short."""
    text = f"{type(exc).__name__}: {exc}"
    data = exc.data if isinstance(exc, RequestError) else None
    if data is None:
        return text
    detail = data if isinstance(data, str) else json.dumps(data, ensure_ascii=False, default=str)
    return f"{text}: {detail[:ERROR_DATA_CHARS]}"


class _Client:
    """What the SDK routes the agent's requests to: permission answers; updates are read by the observer."""

    def __init__(self, owner: AcpClient):
        self.owner = owner

    async def request_permission(self, options: list, session_id: str, tool_call: Any, **kwargs: Any) -> Any:
        """Answer by the settings' permission answer; ``cancelled`` while its cancelled turn runs, or once retired."""
        session = self.owner.sessions.get(session_id)
        if session is None or session.cancelled:
            return RequestPermissionResponse(outcome=DeniedOutcome(outcome="cancelled"))
        choice = self.owner.permission.choose([(option.option_id, str(option.kind)) for option in options])
        logger.info(f"harness.permission session={session_id} answer={choice or 'cancelled'}")
        if choice is None:
            return RequestPermissionResponse(outcome=DeniedOutcome(outcome="cancelled"))
        return RequestPermissionResponse(outcome=AllowedOutcome(option_id=choice, outcome="selected"))

    async def session_update(self, session_id: str, update: Any, **kwargs: Any) -> None:
        """Do nothing: the stream observer reads every update in wire order."""


class AcpSession:
    """One ACP session: prompts and steers sent, each one's end put in the session's mailbox."""

    def __init__(
        self,
        client: AcpClient,
        session_id: str,
        mailbox: Mailbox,
        updates: SessionUpdates,
        *,
        role: str = "worker",
        steering: AcpSteering | None = None,
    ):
        """Hold a session of ``client`` whose events, read by ``updates``, go to ``mailbox``."""
        self.client, self.id, self.mailbox, self.updates = client, session_id, mailbox, updates
        self.cancelled = False
        self.retired = False
        self._numbers = itertools.count(1)
        self._outstanding: set[int] = set()
        self._ended: dict[int, asyncio.Event] = {}
        self._terminal_callbacks: dict[int, dict[object, Callable[[PromptEnded], None]]] = {}
        self._idle = asyncio.Event()
        self._idle.set()
        self.capabilities = client.agent.capabilities
        self.native_steering = steering
        self.binding = SessionBinding(
            client.connection_id,
            session_id,
            "acp",
            PROTOCOL_VERSION,
            client.transport_name,
            role,
            client.peer_name,
            client.peer_version,
        )
        self._states: dict[int, PromptState] = {}
        self._usage: dict[int, InvocationUsage] = {}
        self._active_prompt: int | None = None
        self._event_owners: dict[tuple[str, str], int | None] = {}
        self._uncertain: set[int] = set()
        self.unattributed: list[Any] = []

    @property
    def outstanding(self) -> frozenset[int]:
        """The prompts sent and not ended yet."""
        return frozenset(self._outstanding)

    @property
    def alive(self) -> bool:
        """Whether it can still serve: its connection is open and it is not retired (ACP reports no session's end)."""
        return self.client.alive and not self.retired

    @property
    def quiescent(self) -> bool:
        """Local waiter completion cannot erase an uncertain native invocation."""
        return not self._outstanding and not self._uncertain

    async def ended(self, prompt: int) -> None:
        """Return once ``prompt`` has ended."""
        event = self._ended.get(prompt)
        if event is not None:
            await event.wait()

    async def idle(self) -> None:
        """Return once no prompt is outstanding."""
        await self._idle.wait()

    def read(self, update: Any) -> None:
        """Attribute a wire update before letting it change the active prompt's message grouping."""
        if not isinstance(update, dict) or not isinstance(kind := update.get("sessionUpdate"), str):
            return
        field = None
        if kind in ("tool_call", "tool_call_update"):
            field = "toolCallId"
        elif kind in ("agent_message_chunk", "agent_thought_chunk"):
            field = "messageId"
        identity = update.get(field) if field is not None else None
        key = (field, identity) if field is not None and isinstance(identity, str) and identity else None
        active = self._active_prompt
        # A tool update reports an existing call. Its first sighting cannot establish a prompt's ownership.
        prompt = active if kind != "tool_call_update" else None
        if key is not None:
            prompt = self._event_owners.get(key, prompt)
        reader = (
            self.updates
            if prompt is not None and prompt == active
            else SessionUpdates(self.updates.agent, self.updates.server)
        )
        event = reader.read(update)
        if event is None:
            return
        if key is not None and kind != "tool_call_update":
            self._event_owners.setdefault(key, prompt)
        if prompt is None:
            self.unattributed.append(event)
            return
        event = replace(event, prompt=prompt)
        if isinstance(event, MessageChunk) and self.native_steering is not None and self.native_steering.capture(event):
            return
        if isinstance(event, ToolCard) and event.ended is not None:
            event = replace(event, ended=replace(event.ended, prompt=prompt))
        self.mailbox.put_nowait(event)

    def prompt(self, text: str, *, may_dispatch: Callable[[], bool] | None = None) -> int:
        """Send ``text`` as a prompt, as the agent class gives it (never as a command of the agent)."""
        return self._send(self.client.agent.prompt_text(text), may_dispatch)

    def prompt_state(self, prompt: int) -> PromptState:
        """Return the exact local prompt lifecycle, without inferring remote acceptance."""
        return self._states[prompt]

    def on_prompt_terminal(self, prompt: int, callback: Callable[[PromptEnded], None]) -> Callable[[], None]:
        """Observe the exact invocation without a polling task or inference from a cancel-send receipt."""
        state = self._states[prompt]
        token = object()
        callbacks = self._terminal_callbacks.setdefault(prompt, {})
        callbacks[token] = callback

        def notify(terminal: PromptEnded) -> None:
            selected = callbacks.pop(token, None)
            if selected is not None:
                selected(terminal)

        if state.terminal is not None:
            asyncio.get_running_loop().call_soon(notify, state.terminal)

        def unsubscribe() -> None:
            callbacks.pop(token, None)
            if not callbacks and self._terminal_callbacks.get(prompt) is callbacks:
                self._terminal_callbacks.pop(prompt, None)

        return unsubscribe

    def _send(self, text: str, may_dispatch: Callable[[], bool] | None) -> int:
        number = next(self._numbers)
        busy = not self.quiescent
        self._outstanding.add(number)
        self._ended[number] = asyncio.Event()
        self._idle.clear()
        invocation_id = f"native:{self.client.connection_id}:{self.id}:{number}"
        self._states[number] = PromptState(invocation_id)
        self._usage[number] = InvocationUsage(self.binding, number, invocation_id)
        # An ended connection answers nothing, so the prompt ends at once, unsent.
        if self.retired or self.cancelled or busy or not self.client.alive:
            error = (
                self.client.ended or LOST
                if not self.client.alive
                else "the native session is not available for a new prompt"
            )
            self._end(PromptEnded(number, None, error=error))
            return number
        task = asyncio.create_task(self._run(number, text, may_dispatch), name=f"acp-prompt-{number}")
        self.client.prompts.add(task)
        task.add_done_callback(self.client.prompts.discard)
        return number

    async def _run(self, number: int, text: str, may_dispatch: Callable[[], bool] | None) -> None:
        ended = PromptEnded(number, None, None, "the prompt did not end")
        try:
            if self.native_steering is not None:
                text = self.native_steering.prompt(number, self._states[number].invocation_id, text)
            self.client._dispatching[self.id] = (self, number, may_dispatch)
            response = await self.client.conn.prompt(
                prompt=[TextContentBlock(type="text", text=text)], session_id=self.id
            )
            usage = response.usage.model_dump(by_alias=True, exclude_none=True) if response.usage else None
            ended = PromptEnded(number, str(response.stop_reason), usage, terminal_confirmed=True)
        except asyncio.CancelledError:
            ended = PromptEnded(number, None, None, self.client.ended or CLOSED)
            raise
        except RequestError as exc:
            # The ACP handler can fail while its gateway still owns a dispatched turn.
            # Only a typed prompt result supplies foreground terminal evidence here.
            ended = PromptEnded(number, None, None, reason(exc))
        except Exception as exc:
            # The SDK could not send the prompt or read its answer: the connection is lost.
            self.client.lose(f"a prompt failed without the agent's answer: {reason(exc)}")
            ended = PromptEnded(number, None, None, self.client.ended or LOST)
        finally:
            self.client._dispatching.pop(self.id, None)
            self._end(ended)

    def _dispatched(self, number: int) -> None:
        state = self._states[number]
        state.phase, state.dispatched = "dispatched", True
        self._active_prompt = number
        self.updates.begin_prompt()
        self.mailbox.put_nowait(PromptDispatched(number, state.invocation_id))
        self._usage[number].record(self.client.agent, dispatched=True)

    def _end(self, ended: PromptEnded) -> None:
        """Put a prompt's end in the mailbox (after every update it carried) and mark it ended."""
        state = self._states[ended.prompt]
        if state.terminal is not None:
            return
        ended = replace(
            ended,
            invocation_id=state.invocation_id,
            dispatch="dispatched" if state.dispatched else "not_dispatched",
        )
        state.phase, state.terminal = "ended", ended
        if state.dispatched and not ended.terminal_confirmed:
            self._uncertain.add(ended.prompt)
        if self._active_prompt == ended.prompt:
            self._active_prompt = None
        self._usage[ended.prompt].record(self.client.agent, terminal=ended)
        self._outstanding.discard(ended.prompt)
        if self.native_steering is not None and (message := self.native_steering.finish(ended)) is not None:
            self.mailbox.put_nowait(message)
        self.mailbox.put_nowait(ended)
        self._ended.pop(ended.prompt).set()
        if not self._outstanding:
            self.cancelled = False
            self._idle.set()
        callbacks = self._terminal_callbacks.pop(ended.prompt, {})
        for token in tuple(callbacks):

            def notify(selected=token):
                callback = callbacks.pop(selected, None)
                if callback is not None:
                    callback(ended)

            asyncio.get_running_loop().call_soon(notify)

    async def cancel(self) -> CancelDispatch:
        """Fence the scheduled prompt or send a cancel only while the exact target remains active.

        A retired session can still own a running prompt. If the transport fails, submission is uncertain.
        """
        if not self._outstanding or not self.client.alive:
            return "not_dispatched"
        if self.cancelled or self.id in self.client._cancelling:
            return "unknown"
        self.cancelled = True
        prompt = self._active_prompt
        if prompt is None:
            return "not_dispatched"
        self.client._cancelling[self.id] = self, prompt
        try:
            await self.client.conn.cancel(session_id=self.id)
            return "dispatched"
        except _CancelNotDispatched:
            return "not_dispatched"
        except Exception:
            return "unknown"
        finally:
            self.client._cancelling.pop(self.id, None)

    def retire(self) -> bool:
        """Stop reading updates for the session, and close it in the agent as its class's close request says.

        Return whether a close went out (in the background); with none, the agent may keep the session until it exits.
        """
        if self.retired:
            return False
        self.retired = True
        self.client.sessions.pop(self.id, None)
        if self.native_steering is not None:
            cleanup = asyncio.create_task(self.native_steering.close(), name="acp-steering-close")
            self.client.closes.add(cleanup)
            cleanup.add_done_callback(self.client.closes.discard)
        return self.client.send_close(self.id)


class AcpClient:
    """A connection to one ACP runtime process (the harness's ``HarnessClient``), to the agent ``agent`` describes."""

    def __init__(
        self,
        transport: Transport,
        agent: AcpAgent,
        *,
        permission: PermissionAnswer,
        start_timeout_seconds: float,
        stop_grace_seconds: float,
        transport_name: str = "unknown",
        launch: RuntimeLaunch | None = None,
    ):
        """Speak ACP over ``transport`` to the agent of ``agent``'s class; nothing connects until :meth:`open`.

        Its close waits up to ``stop_grace_seconds`` for the sessions' closes before it closes the transport.
        """
        self.transport, self.agent, self.permission = transport, agent, permission
        self.transport_name = transport_name
        self.launch = launch
        self.connection_id = str(uuid.uuid4())
        self.peer_name: str | None = None
        self.peer_version: str | None = None
        self.start_timeout, self.stop_grace = start_timeout_seconds, stop_grace_seconds
        self.conn: ClientSideConnection | None = None
        self.advertised = AgentCapabilities()
        self.sessions: dict[str, AcpSession] = {}
        self.prompts: set[asyncio.Task] = set()
        self.closes: set[asyncio.Task] = set()
        self.ended: str | None = None
        self._watch: asyncio.Task | None = None
        self._ending: asyncio.Task | None = None
        self._rpc = InMemoryMessageStateStore()
        self._dispatching: dict[str, tuple[AcpSession, int, Callable[[], bool] | None]] = {}
        self._cancelling: dict[str, tuple[AcpSession, int]] = {}
        self._requests: dict[int | str, tuple[AcpSession, int]] = {}
        self._owned_sessions: set[str] = set()
        self._closed_sessions: set[str] = set()
        self._closing: asyncio.Task | None = None

    @property
    def alive(self) -> bool:
        """Whether the connection is open and the process runs."""
        return self.conn is not None and self.ended is None

    async def open(self) -> None:
        """Start or reach the agent, then ``initialize`` protocol version 1, within ``start_timeout_seconds``."""
        if self.conn is not None or self.ended is not None:
            raise RuntimeError("An ACP client owns one connection; create a new client to reconnect")
        try:
            async with asyncio.timeout(self.start_timeout):
                reader, writer = await self.transport.open()
                self.conn = connect_to_agent(
                    _Client(self),
                    writer,
                    reader,
                    observers=[self._observe],
                    state_store=self._rpc,
                    sender_factory=lambda writer, supervisor: _Sender(self, writer),
                )
                self._watch = asyncio.create_task(self._watch_close(), name="acp-watch")
                response = await self.conn.initialize(
                    protocol_version=PROTOCOL_VERSION,
                    client_capabilities=ClientCapabilities(
                        fs=FileSystemCapabilities(read_text_file=False, write_text_file=False), terminal=False
                    ),
                    client_info=CLIENT_INFO,
                )
        except TimeoutError as exc:
            raise TimeoutError(
                f"The agent was not reached, or did not answer initialize, within start_timeout_seconds "
                f"({self.start_timeout:g} s)"
            ) from exc
        except RequestError as exc:
            raise RuntimeError(f"The agent answered initialize with an error: {reason(exc)}") from exc
        if response.protocol_version != PROTOCOL_VERSION:
            raise RuntimeError(f"The agent speaks ACP version {response.protocol_version}, not {PROTOCOL_VERSION}")
        self.advertised = response.agent_capabilities or AgentCapabilities()
        if response.agent_info is not None:
            self.peer_name, self.peer_version = response.agent_info.name, response.agent_info.version

    async def new_session(self, cwd: Path, server: McpServer, mailbox: Mailbox, *, role: str = "worker") -> AcpSession:
        """Open a session in ``cwd`` whose one MCP server is ``server``, its bearer token in a header."""
        if not self.alive:
            raise ConnectionError(self.ended or LOST)
        entry = HttpMcpServer(
            type="http",
            name=server.name,
            url=server.url,
            headers=[HttpHeader(name="Authorization", value=f"Bearer {server.token}")],
        )
        try:
            response = await asyncio.wait_for(
                self.conn.new_session(cwd=str(cwd), mcp_servers=[entry]), self.start_timeout
            )
        except TimeoutError as exc:
            raise TimeoutError(
                f"The agent did not answer session/new within start_timeout_seconds ({self.start_timeout:g} s)"
            ) from exc
        except RequestError as exc:
            raise RuntimeError(f"The agent answered session/new with an error: {reason(exc)}") from exc
        if response.session_id in self._owned_sessions:
            raise RuntimeError("The agent reused an existing native identity for a new session")
        steering = None
        if role == "worker" and self.launch is not None:
            steering = self.agent.bind_steering(self.launch, cwd, response, server.name, self.start_timeout)
        session = AcpSession(
            self,
            response.session_id,
            mailbox,
            SessionUpdates(self.agent, server.name),
            role=role,
            steering=steering,
        )
        self.sessions[session.id] = session
        self._owned_sessions.add(session.id)
        return session

    def _may_write(self, payload: dict[str, Any]) -> bool:
        """Recheck the exact prompt or cancel after sender serialization, before writing its bytes."""
        method = payload.get("method")
        if method not in ("session/prompt", SESSION_CANCEL):
            return True
        session_id = (payload.get("params") or {}).get("sessionId")
        if method == SESSION_CANCEL:
            target = self._cancelling.get(session_id)
            return self.alive and target is not None and target[0]._active_prompt == target[1]
        given = self._dispatching.get(session_id)
        if given is None:
            return False
        session, number, guard = given
        if not self.alive or session.retired or session.cancelled or session._active_prompt is not None:
            return False
        if session.prompt_state(number).phase != "scheduled":
            return False
        try:
            return guard is None or bool(guard())
        except Exception as error:
            logger.warning(f"harness.dispatch_guard_failed error_type={type(error).__name__}")
            return False

    def _wrote(self, payload: dict[str, Any]) -> None:
        """Publish local submission; asynchronous transport acceptance and remote consumption remain separate."""
        if payload.get("method") != "session/prompt":
            return
        session_id = (payload.get("params") or {}).get("sessionId")
        session, number, _ = self._dispatching[session_id]
        self._requests[payload["id"]] = session, number
        session._dispatched(number)

    def send_close(self, session_id: str) -> bool:
        """Send the agent class's close request for one session in the background; return whether it went out."""
        request = self.agent.close_request(self.advertised)
        if request is None or not self.alive:
            return False
        sending = asyncio.create_task(self._close(request, session_id), name="acp-close")
        self.closes.add(sending)
        sending.add_done_callback(self.closes.discard)
        return True

    async def _close(self, request: CloseRequest, session_id: str) -> None:
        """Send one session's close; an error is only logged, and a session the agent does not have counts as closed."""
        try:
            await request(self.conn, session_id)
        except RequestError as exc:
            if exc.code != NOT_FOUND:
                logger.warning(f"harness.session_close_failed session={session_id} {reason(exc)}")
                return
        except Exception as exc:
            logger.warning(f"harness.session_close_failed session={session_id} {reason(exc)}")
            return
        logger.debug(f"harness.session_closed session={session_id}")
        self._closed_sessions.add(session_id)

    def _observe(self, event: StreamEvent) -> None:
        """Route each incoming ``session/update`` to its session's mailbox, in wire order.

        The SDK runs each notification callback in a task of its own, which may run after the response that followed it.
        A message that is not a JSON object stops the SDK's receive loop for good: the connection is lost.
        """
        if event.direction is not StreamDirection.INCOMING:
            return
        if not isinstance(event.message, dict):
            sent = json.dumps(event.message, ensure_ascii=False)[:ERROR_DATA_CHARS]
            self.lose(f"the agent sent a message that is not a JSON object: {sent}")
            return
        if "method" not in event.message and (given := self._requests.pop(event.message.get("id"), None)):
            session, number = given
            if session._active_prompt == number:
                session._active_prompt = None
        if event.message.get("method") != SESSION_UPDATE:
            return
        params = event.message.get("params") or {}
        session = self.sessions.get(params.get("sessionId"))
        if session is not None:
            session.read(params.get("update"))

    def lose(self, why: str) -> None:
        """Count the connection as lost, warn of ``why`` and close the transport, which ends every waiting prompt."""
        if self.ended is not None:
            return
        self.ended = LOST
        logger.warning(f"harness.connection_lost {why}")
        self._ending = asyncio.create_task(self.transport.close(), name="acp-lost")

    async def _watch_close(self) -> None:
        await self.transport.closed.wait()
        self.ended = self.ended or LOST
        await self._end_prompts()

    async def _end_prompts(self) -> None:
        """Close the SDK connection, which fails its waiting requests, then cancel each prompt it left waiting.

        Its close fails none once its sender has died. Every prompt's task runs a step first, so one just answered ends
        with its own outcome, and one not started yet still puts its end.
        """
        if self.conn is not None:
            with contextlib.suppress(Exception):
                await self.conn.close()
        await asyncio.sleep(0)
        waiting = [task for task in self.prompts if not task.done()]
        for task in waiting:
            task.cancel()
        await asyncio.gather(*waiting, return_exceptions=True)

    async def close(self) -> None:
        """Join one owned connection shutdown; cancellation of this waiter does not abandon it."""
        if self._closing is None:
            self._closing = asyncio.create_task(self._shutdown(), name="acp-client-close")
            self._closing.add_done_callback(lambda task: task.cancelled() or task.exception())
        await asyncio.shield(self._closing)

    async def wait_closed(self) -> NativeCleanup:
        """Return actual cleanup scope, independently of foreground cancellation and connection shutdown."""
        await self.close()
        await self.transport.wait_closed()
        if self.agent.runtime_ownership == "process_group" and self.transport.runtime_stopped:
            return NativeCleanup("runtime", True, True)
        if self._owned_sessions and self._owned_sessions <= self._closed_sessions:
            return NativeCleanup("session", True, True)
        safe = not self._owned_sessions and (self.transport_name != "stdio" or self.transport.runtime_stopped)
        return NativeCleanup("connection", True, safe)

    async def _shutdown(self) -> None:
        for session in list(self.sessions.values()):
            session.retire()
        self.ended = self.ended or CLOSED
        if self.closes:
            await asyncio.wait(set(self.closes), timeout=self.stop_grace)
        waiting = [sending for sending in self.closes if not sending.done()]
        for sending in waiting:
            sending.cancel()
        await asyncio.gather(*waiting, return_exceptions=True)
        await self.transport.close()
        await self._end_prompts()
        if self._watch is not None:
            self._watch.cancel()
            await asyncio.gather(self._watch, return_exceptions=True)
        if self._ending is not None:
            await asyncio.gather(self._ending, return_exceptions=True)
