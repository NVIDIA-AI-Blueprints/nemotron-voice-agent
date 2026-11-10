# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The ACP client (``backend.harness.protocol: acp``) on the ACP SDK's client connection."""

from __future__ import annotations

import asyncio
import contextlib
import itertools
import json
from pathlib import Path
from typing import Any

from acp import PROTOCOL_VERSION, RequestError, connect_to_agent
from acp.client.connection import ClientSideConnection
from acp.connection import StreamDirection, StreamEvent
from acp.meta import CLIENT_METHODS
from acp.schema import (
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
from loguru import logger

from ..protocol import Mailbox, McpServer, PermissionAnswer, PromptEnded
from .agent import AcpAgent
from .transports import Transport
from .updates import SessionUpdates

CLIENT_INFO = Implementation(name="delegation-core-harness", version="1")
SESSION_UPDATE = CLIENT_METHODS["session_update"]
LOST = "the connection to the agent is lost"
CLOSED = "the client closed"
# A cap on an agent's error data in a reason, which goes on the wire in a task's error text.
ERROR_DATA_CHARS = 300


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

    def __init__(self, client: AcpClient, session_id: str, mailbox: Mailbox, updates: SessionUpdates):
        """Hold a session of ``client`` whose events, read by ``updates``, go to ``mailbox``."""
        self.client, self.id, self.mailbox, self.updates = client, session_id, mailbox, updates
        self.cancelled = False
        self.retired = False
        self._numbers = itertools.count(1)
        self._outstanding: set[int] = set()
        self._ended: dict[int, asyncio.Event] = {}
        self._idle = asyncio.Event()
        self._idle.set()

    @property
    def outstanding(self) -> frozenset[int]:
        """The prompts sent and not ended yet."""
        return frozenset(self._outstanding)

    async def ended(self, prompt: int) -> None:
        """Return once ``prompt`` has ended."""
        event = self._ended.get(prompt)
        if event is not None:
            await event.wait()

    async def idle(self) -> None:
        """Return once no prompt is outstanding."""
        await self._idle.wait()

    def read(self, update: Any) -> None:
        """Put the event one ``session/update`` carries in the mailbox (called in wire order)."""
        event = self.updates.read(update)
        if event is not None:
            self.mailbox.put_nowait(event)

    def prompt(self, text: str) -> int:
        """Send ``text`` as a prompt, as the agent class gives it (never as a command of the agent)."""
        return self._send(self.client.agent.prompt_text(text))

    def steer(self, text: str) -> int:
        """Send ``text`` as the agent's own mid-turn steer, as the agent class gives it."""
        return self._send(self.client.agent.steer_text(text))

    def _send(self, text: str) -> int:
        number = next(self._numbers)
        self._outstanding.add(number)
        self._ended[number] = asyncio.Event()
        self._idle.clear()
        # An ended connection answers nothing, so the prompt ends at once, unsent.
        if not self.client.alive:
            self._end(PromptEnded(number, None, None, self.client.ended or LOST))
            return number
        task = asyncio.create_task(self._run(number, text), name=f"acp-prompt-{number}")
        self.client.prompts.add(task)
        task.add_done_callback(self.client.prompts.discard)
        return number

    async def _run(self, number: int, text: str) -> None:
        ended = PromptEnded(number, None, None, "the prompt did not end")
        try:
            response = await self.client.conn.prompt(
                prompt=[TextContentBlock(type="text", text=text)], session_id=self.id
            )
            usage = response.usage.model_dump(by_alias=True, exclude_none=True) if response.usage else None
            ended = PromptEnded(number, str(response.stop_reason), usage)
        except asyncio.CancelledError:
            ended = PromptEnded(number, None, None, self.client.ended or CLOSED)
            raise
        except RequestError as exc:
            ended = PromptEnded(number, None, None, reason(exc))
        except Exception as exc:
            # The SDK could not send the prompt or read its answer: the connection is lost.
            self.client.lose(f"a prompt failed without the agent's answer: {reason(exc)}")
            ended = PromptEnded(number, None, None, self.client.ended or LOST)
        finally:
            self._end(ended)

    def _end(self, ended: PromptEnded) -> None:
        """Put a prompt's end in the mailbox (after every update it carried) and mark it ended."""
        self._outstanding.discard(ended.prompt)
        self.mailbox.put_nowait(ended)
        self._ended.pop(ended.prompt).set()
        if not self._outstanding:
            self.cancelled = False
            self._idle.set()

    async def cancel(self) -> bool:
        """Send ``session/cancel`` while a prompt is outstanding; return whether it was sent.

        An agent may take a stray cancel for its next prompt; a retired session's running turn is still cancelled.
        """
        if not self._outstanding or not self.client.alive:
            return False
        self.cancelled = True
        with contextlib.suppress(Exception):
            await self.client.conn.cancel(session_id=self.id)
            return True
        return False

    def retire(self) -> None:
        """Stop reading updates for the session; the agent may keep it until it exits."""
        self.retired = True
        self.client.sessions.pop(self.id, None)


class AcpClient:
    """A connection to one ACP runtime process (the harness's ``HarnessClient``), to the agent ``agent`` describes."""

    def __init__(
        self,
        transport: Transport,
        agent: AcpAgent,
        *,
        permission: PermissionAnswer,
        start_timeout_seconds: float,
    ):
        """Speak ACP over ``transport`` to the agent of ``agent``'s class; nothing connects until :meth:`open`."""
        self.transport, self.agent, self.permission = transport, agent, permission
        self.start_timeout = start_timeout_seconds
        self.conn: ClientSideConnection | None = None
        self.sessions: dict[str, AcpSession] = {}
        self.prompts: set[asyncio.Task] = set()
        self.ended: str | None = None
        self._watch: asyncio.Task | None = None
        self._ending: asyncio.Task | None = None

    @property
    def alive(self) -> bool:
        """Whether the connection is open and the process runs."""
        return self.conn is not None and self.ended is None

    async def open(self) -> None:
        """Start or reach the agent, then ``initialize`` protocol version 1, within ``start_timeout_seconds``."""
        try:
            async with asyncio.timeout(self.start_timeout):
                reader, writer = await self.transport.open()
                self.conn = connect_to_agent(_Client(self), writer, reader, observers=[self._observe])
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

    async def new_session(self, cwd: Path, server: McpServer, mailbox: Mailbox) -> AcpSession:
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
        session = AcpSession(self, response.session_id, mailbox, SessionUpdates(self.agent, server.name))
        self.sessions[session.id] = session
        return session

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
        """Close the transport (the agent ends its sessions and exits), then the SDK connection; every prompt ends."""
        self.ended = self.ended or CLOSED
        for session in list(self.sessions.values()):
            session.retire()
        await self.transport.close()
        await self._end_prompts()
        if self._watch is not None:
            self._watch.cancel()
            await asyncio.gather(self._watch, return_exceptions=True)
        if self._ending is not None:
            await asyncio.gather(self._ending, return_exceptions=True)
