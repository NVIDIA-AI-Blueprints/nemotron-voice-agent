# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""One live session: its state, and the dispatch of client commands to the module that handles each."""

from __future__ import annotations

import asyncio
import base64
import time
from collections import deque
from collections.abc import Awaitable, Callable
from contextvars import ContextVar
from typing import TYPE_CHECKING, Any

from loguru import logger

from examples.shared.ids import uid
from gateways.live.client_delegation import ClientDelegation
from gateways.live.events import Fanout, Timeline, client_event_allowed, error_event, event
from gateways.live.events.fanout import Close, Send
from gateways.live.ports import Inspectable
from gateways.live.protocol import BackendConfig, ProtocolError, SessionConfig
from gateways.live.server_delegation import ServerDelegation
from gateways.live.session import appends, audio_input, lifecycle, tool_calling, updates

if TYPE_CHECKING:
    from gateways.live.ports import (
        AdapterCloseReason,
        CoreHook,
        DelegationEvents,
        DelegationPart,
        PrimarySocket,
        SessionParts,
    )

COMMAND_WAIT_FOR_START_SECONDS = 30
ADAPTER_EVENTS = frozenset({"session.input_transcript.delta", "session.output_transcript.delta"})
DELEGATION_EVENTS = frozenset({"session.delegation.created", "response.event"})
RETAINED_INPUTS = frozenset(
    {
        "response.item.create",
        "response.create",
        "session.instructions.append",
        "session.thinking.append",
        "session.commentary.append",
    }
)

# Only a responses part calls the core hook, so a client-mode session never builds a core.
DELEGATION_PARTS: dict[str, Callable[[CoreHook], DelegationPart]] = {
    "client": lambda core: ClientDelegation(),
    "responses": lambda core: ServerDelegation(core()),
}


def no_core():
    """Refuse a ``responses`` session: the core hook of a server that runs no delegation core."""
    raise ProtocolError("Responses delegation is not available on this server", param="session.delegation.type")


Command = Callable[["LiveProtocolSession", dict, str, "str | None"], Awaitable[None]]
# An error a command causes goes back only to the connection that sent it, also when reported later.
COMMAND_CONNECTION: ContextVar[str | None] = ContextVar("command_connection", default=None)


async def refuse_start(session: LiveProtocolSession, message: dict, source: str, client_event_id: str | None) -> None:
    """Refuse ``session.start`` on a running session, which starts once."""
    raise ProtocolError("The session already started", "session_already_started")


async def close_requested(
    session: LiveProtocolSession, message: dict, source: str, client_event_id: str | None
) -> None:
    """Close the session for the client's ``session.close``."""
    await session.close("close_requested", client_event_id, asked_by=f"the client ({source} connection)")


def unsupported(feature: str) -> Command:
    """Return the handler of an official command this server does not offer: a refusal that names it."""

    async def refuse(session: LiveProtocolSession, message: dict, source: str, client_event_id: str | None) -> None:
        raise ProtocolError(f"{feature} is not supported by this server", param="type")

    return refuse


COMMANDS: dict[str, Command] = {
    "session.start": refuse_start,
    "session.update": updates.update,
    "session.input_audio.append": audio_input.append,
    "session.input_audio.mute": audio_input.mute,
    "session.input_audio.unmute": audio_input.unmute,
    "session.instructions.append": appends.instructions,
    "session.thinking.append": appends.thinking,
    "session.commentary.append": appends.commentary,
    "response.item.create": tool_calling.item_create,
    "response.create": tool_calling.response_create,
    "session.close": close_requested,
    "session.output_audio.fixture.play": unsupported("Fixture playback (session.output_audio.fixture.play)"),
    "decision.create": unsupported("The Decisions API (decision.create)"),
    "decision.item.create": unsupported("The Decisions API (decision.item.create)"),
    "decision.item.delete": unsupported("The Decisions API (decision.item.delete)"),
}


class LiveProtocolSession:
    """The protocol state of one live session."""

    def __init__(
        self,
        config: SessionConfig,
        transport: str,
        parts: SessionParts,
        *,
        core: CoreHook = no_core,
        socket: PrimarySocket | None = None,
        max_session_seconds: int = 1800,
        max_appends_per_minute: int = 64,
        on_closed: Callable[[LiveProtocolSession], Awaitable[None]] | None = None,
    ):
        """Create a session in the ``created`` state."""
        self.id = uid("live")
        self.config, self.transport, self.parts = config, transport, parts
        self.frontend = parts.frontend
        self.delegation: DelegationPart = DELEGATION_PARTS[config.mode](core)
        self.max_session_seconds, self.on_closed = max_session_seconds, on_closed
        self.max_appends_per_minute = max_appends_per_minute
        self.recent_appends: deque[float] = deque()
        self.socket = socket
        self.attachment_generation = 1
        self.attachment_state = "active"
        self.reconnect = None
        self.status = "created"
        self.created_at = time.time()
        self.expires_at = int(self.created_at + max_session_seconds)
        self.started_at: float | None = None
        self.fanout = Fanout(
            lambda: self.config,
            transport,
            self.primary_lost,
        )
        self.timeline = Timeline()
        self.tasks: set[asyncio.Task] = set()
        self.command_lock = asyncio.Lock()
        self.close_task: asyncio.Task | None = None
        self.ready, self.closed = asyncio.Event(), asyncio.Event()
        self.adapter_view, self.delegation_view = AdapterView(self), DelegationSideView(self)
        self.delegation_port = DelegationPortView(self)

    def new_adapter_view(self) -> None:
        """Give fresh media a generation-bound port; old callbacks never acquire the replacement's authority."""
        self.adapter_view = AdapterView(self)
        self.delegation_port = DelegationPortView(self)

    async def primary_lost(self, socket: PrimarySocket | None = None) -> None:
        """Classify a failed primary as accidental loss, independently of deliberate logical close."""
        from gateways.live.session.reconnect import TRANSIENT_CLOSE_CODES

        if socket is not None and socket is not self.socket:
            return
        source = self.socket if socket is None else socket
        code = getattr(source, "code", None)
        if code is not None and code not in TRANSIENT_CLOSE_CODES:
            reason = "remote_hangup" if code in {1000, 1001} else "connection_lost"
            await self.close(reason, asked_by="the primary connection closed")
        elif self.reconnect is not None:
            # Identity is checked again inside the lease lock, after any concurrent attachment setup.
            await self.reconnect.lost(source)
        else:
            await self.close("connection_lost", asked_by="the primary connection failed")

    @property
    def input_ms(self) -> int:
        """Milliseconds of caller audio seen so far (the input media clock)."""
        return self.timeline.input_ms

    def clock_ms(self) -> int:
        """Return milliseconds since the session became active."""
        return 0 if self.started_at is None else int((time.monotonic() - self.started_at) * 1000)

    def snapshot(self) -> dict:
        """Return the session resource of lifecycle events: always ``active``, and without MCP credentials.

        A ``store: true`` session in Responses mode names its core conversation in ``delegation.responses``, while
        the delegation part holds it: from the start, and at the close only when the core kept the conversation.
        """
        snapshot = {
            **self.config.public_dump(),
            "id": self.id,
            "status": "active",
            "expires_at": self.expires_at,
        }
        responses = snapshot["delegation"].get("responses")
        if responses is not None and self.config.store:
            responses.pop("conversation", None)
            if self.delegation.conversation is not None:
                responses["conversation"] = self.delegation.conversation
        return snapshot

    async def history(self) -> dict:
        """Return what the session's models hold now: the adapter's half and the delegation part's ``backend``."""
        adapter = (await self.frontend.history() if isinstance(self.frontend, Inspectable) else None) or {}
        part = await self.delegation.history()
        backend = None if part is None else {**part, **(adapter.get("backend") or {})}
        return {**adapter, "frontend": adapter.get("frontend"), "backend": backend}

    def spawn(self, coro, name: str) -> asyncio.Task:
        """Run ``coro`` as a session-owned task that is cancelled on close; its log lines name the session."""
        with logger.contextualize(stream_id=self.id):
            task = asyncio.create_task(coro, name=f"{self.id}:{name}")
        self.tasks.add(task)
        task.add_done_callback(self.tasks.discard)
        return task

    def subscribe(self, send: Send, source: str, close: Close | None = None) -> str:
        """Attach a connection in its role (``primary`` or ``sideband``) and return its key."""
        return self.fanout.subscribe(send, source, close)

    async def unsubscribe(self, key: str) -> None:
        """Detach a connection."""
        await self.fanout.unsubscribe(key)

    async def publish(self, message: dict, audience: str = "all") -> None:
        """Send ``message`` to every permitted connection of ``audience`` (``all``, ``primary`` or ``sideband``)."""
        await self.fanout.publish(message, audience)

    async def emit(self, kind: str, **fields: Any) -> None:
        """Build and publish a server event."""
        await self.publish(event(kind, **fields))

    async def reply_error(self, exc: ProtocolError, client_event_id: str | None) -> None:
        """Send the ``error`` event of ``exc`` to the connection whose command runs in this context, else to all."""
        connection = COMMAND_CONNECTION.get()
        if connection is None:
            await self.publish(error_event(exc, client_event_id))
        else:
            await self.fanout.send_to(connection, error_event(exc, client_event_id))

    async def report_error(self, message: str, client_event_id: str | None = None) -> None:
        """Report a fatal failure, or a client command's, as an ``error`` event with code ``pipeline_error``."""
        logger.warning(f"session={self.id} pipeline_error={message}")
        await self.reply_error(ProtocolError(message, "pipeline_error"), client_event_id)

    async def send_output_audio(self, audio: bytes) -> None:
        """Send reply audio, in the session's wire format, to the primary connection only, with no ``event_id``."""
        if self.socket is None:
            raise RuntimeError("Reply audio events are sent only on a primary WebSocket")
        if audio:
            await self.publish(
                {"type": "session.output_audio.delta", "delta": base64.b64encode(audio).decode()}, audience="primary"
            )

    async def advance_input_clock(self, ms: int) -> None:
        """Record that the caller's audio reached ``ms``."""
        self.timeline.advance(ms)

    async def start(self, client_event_id: str | None = None) -> None:
        """Open the delegation part, start the frontend adapter, then announce ``session.started``."""
        with logger.contextualize(stream_id=self.id):
            await lifecycle.start(self, client_event_id)

    async def close(
        self, reason: str = "close_requested", client_event_id: str | None = None, *, asked_by: str = "the gateway"
    ) -> None:
        """Close the session: close the adapter and the delegation part, emit ``session.closed``, then clean up."""
        if self.reconnect is not None:
            self.reconnect.revoke()
        with logger.contextualize(stream_id=self.id):
            await lifecycle.close(self, reason, client_event_id, asked_by)

    async def handle(self, message: Any, connection: str) -> None:
        """Run one client command from ``connection``; a rejection is an ``error`` event for that connection only."""
        source = self.fanout.role(connection)
        if source is None:
            logger.debug(f"session={self.id} drops a command from a detached connection")
            return
        client_event_id = message.get("event_id") if isinstance(message, dict) else None
        token = COMMAND_CONNECTION.set(connection)
        with logger.contextualize(stream_id=self.id):
            try:
                async with self.command_lock:
                    if self.fanout.role(connection) != source:
                        return
                    if self.status in {"created", "starting"}:
                        await self._wait_until_ready()
                    if self.fanout.role(connection) != source:
                        return
                    if self.attachment_state != "active" and (
                        not isinstance(message, dict) or message.get("type") != "session.close"
                    ):
                        raise ProtocolError(
                            "The session is waiting for its caller to reconnect", "session_reconnecting"
                        )
                    retained = (
                        self.reconnect is not None
                        and isinstance(message, dict)
                        and message.get("type") in RETAINED_INPUTS
                    )
                    if retained and source != "primary":
                        raise ProtocolError(
                            "Reconnectable call input requires its current primary attachment", "permission_denied"
                        )
                    prepared = None

                    def prepare() -> None:
                        nonlocal prepared
                        self._command(command, source)
                        kind = command["type"]
                        if kind == "response.item.create":
                            prepared = tool_calling.prepare_item_create(self, command, client_event_id)
                        elif kind == "response.create":
                            prepared = tool_calling.prepare_response_create(self, command, client_event_id)
                        else:
                            prepared = appends.prepare(self, command, client_event_id)

                    # The lease owns operation metadata. Command handlers receive only their command's
                    # fields, while the unchanged envelope remains available for its admission receipt.
                    command = (
                        {key: value for key, value in message.items() if key != "operation_id"} if retained else message
                    )
                    if retained:
                        if not await self.reconnect.operation(message, connection, prepare):
                            return
                        # New reservations and the first prepared mutation have no intervening wait.
                        await prepared()
                        await self.reconnect.operation_recorded(message, connection)
                    else:
                        await self._dispatch(command, source)
            except ProtocolError as exc:
                await self.reply_error(exc, client_event_id)
            except Exception as exc:
                # A fault in a part must not end the session or the connection: report it and keep serving.
                logger.exception(f"session={self.id} command {client_event_id!r} failed: {exc}")
                await self.reply_error(
                    ProtocolError("The command could not be processed.", "server_error"), client_event_id
                )
            finally:
                COMMAND_CONNECTION.reset(token)

    async def _wait_until_ready(self) -> None:
        """Wait, within a limit, until a starting session is active: a connection may send before it is."""
        try:
            async with asyncio.timeout(COMMAND_WAIT_FOR_START_SECONDS):
                await self.ready.wait()
        except TimeoutError as exc:
            raise ProtocolError("The session did not become active in time", "session_closed") from exc

    async def _dispatch(self, message: Any, source: str) -> None:
        command = self._command(message, source)
        await command(self, message, source, message.get("event_id"))

    def _command(self, message: Any, source: str) -> Command:
        """Check dispatch authority synchronously, without admitting or invoking a command."""
        if not isinstance(message, dict) or not isinstance(message.get("type"), str):
            raise ProtocolError("Expected a JSON event object")
        if self.status != "active":
            raise ProtocolError("Session is not active", "session_closed")
        kind = message["type"]
        if not client_event_allowed(self.config, self.transport, kind, source):
            raise ProtocolError("Client event is not permitted", "permission_denied")
        command = COMMANDS.get(kind)
        if command is None:
            raise ProtocolError("Unknown client event", param="type")
        return command


def _refuse_unless_allowed(kind: str, allowed: frozenset[str], part: str) -> None:
    if kind not in allowed:
        raise ValueError(f"The {part} may not send {kind!r}; it may send only {', '.join(sorted(allowed))}")


class AdapterView:
    """The frontend adapter's narrow view of its session (:class:`gateways.live.ports.SessionPort`)."""

    def __init__(self, session: LiveProtocolSession):
        """Give the frontend adapter its view of ``session``."""
        self._session = session
        self.id = session.id
        self.generation = session.attachment_generation

    def current(self) -> bool:
        """Whether this attachment still owns the session's media and callback authority."""
        return self.generation == self._session.attachment_generation

    @property
    def input_ms(self) -> int:
        """Return the elapsed caller audio in milliseconds."""
        return self._session.input_ms

    @property
    def socket(self) -> PrimarySocket | None:
        """Return the primary WebSocket if this connection uses one."""
        return self._session.socket

    def is_active(self) -> bool:
        """Return whether the session and its current attachment are active."""
        return self.current() and self._session.status == "active" and self._session.attachment_state == "active"

    async def emit(self, kind: str, **fields: Any) -> None:
        """Send a transcript event; refuse every other event type with ``ValueError``."""
        _refuse_unless_allowed(kind, ADAPTER_EVENTS, "frontend adapter")
        if not self.is_active():
            return
        await self._session.emit(kind, **fields)

    async def advance_input_clock(self, ms: int) -> None:
        """Record that the caller's audio reached ``ms``."""
        if self.is_active():
            await self._session.advance_input_clock(ms)

    async def send_output_audio(self, audio: bytes) -> None:
        """Send reply audio, in the session's wire format, to the primary connection only; no ``event_id``."""
        if self.is_active():
            await self._session.send_output_audio(audio)

    async def report_error(self, message: str, client_event_id: str | None = None, *, fatal: bool = False) -> None:
        """Report a failure: an ``error`` for a command's or a fatal one, else a logged ``info`` notice."""
        if not self.current():
            return
        if fatal or client_event_id is not None:
            await self._session.report_error(message, client_event_id)
            return
        logger.warning(f"session={self.id} pipeline_error={message}")
        await self._session.emit("info", code="pipeline_error", message=message)

    def request_close(self, reason: AdapterCloseReason) -> None:
        """Ask the session to close. The close is scheduled; this returns at once."""
        if not self.current() or self._session.attachment_state != "active":
            return
        if reason == "connection_lost":
            socket = self._session.socket
            self._session.spawn(self._session.primary_lost(socket), "adapter-connection-lost")
        else:
            self._session.spawn(self._session.close(reason, asked_by="the frontend adapter"), "adapter-close")


class DelegationSideView:
    """The delegation part's narrow view of its session (:class:`gateways.live.ports.DelegationView`)."""

    def __init__(self, session: LiveProtocolSession):
        """Give the delegation part its view of ``session``."""
        self._session = session
        self.id = session.id

    @property
    def input_ms(self) -> int:
        """Milliseconds of caller audio seen so far (the input media clock)."""
        return self._session.input_ms

    @property
    def responses(self) -> BackendConfig | None:
        """The session's ``delegation.responses`` settings, as last updated; ``None`` in client mode."""
        return self._session.config.delegation.responses

    @property
    def acknowledgement_delay_ms(self) -> int:
        """The session's ``delegation.acknowledgement_delay_ms``, as ``session.update`` last changed it."""
        return self._session.config.delegation.acknowledgement_delay_ms

    @property
    def store(self) -> bool:
        """The session's ``store``, its opt-in to keep backend work that outlives the call."""
        return self._session.config.store

    async def emit(self, kind: str, **fields: Any) -> None:
        """Send a delegation event; refuse every other event type with ``ValueError``."""
        _refuse_unless_allowed(kind, DELEGATION_EVENTS, "delegation part")
        if self._session.reconnect is not None and self._session.attachment_state != "active":
            return
        await self._session.emit(kind, **fields)

    async def emit_paced(self, kind: str, *, before_publish: Callable[[], bool] | None = None, **fields: Any) -> None:
        """Check the current attachment after pacing, then admit the event immediately before publishing."""
        _refuse_unless_allowed(kind, DELEGATION_EVENTS, "delegation part")
        generation = self._session.attachment_generation
        await self._session.fanout.wait_for_room()
        if self._session.reconnect is not None and (
            generation != self._session.attachment_generation or self._session.attachment_state != "active"
        ):
            return
        if before_publish is not None and not before_publish():
            return
        await self._session.emit(kind, **fields)

    def report_refusal(self, message: str, code: str, client_event_id: str | None, param: str | None = None) -> None:
        """Schedule the ``error`` event of the core's refusal of a client command's request or settings."""
        self._session.spawn(self._session.reply_error(ProtocolError(message, code, param), client_event_id), "refusal")


class DelegationPortView:
    """The frontend adapter's narrow view of the delegation part (:class:`gateways.live.ports.DelegationPort`)."""

    def __init__(self, session: LiveProtocolSession):
        """Bind the frontend's delegation authority to the same attachment as its media port."""
        self._session, self._part = session, session.delegation
        self.generation = session.attachment_generation

    def _check(self, *, listening: bool = False) -> None:
        session = self._session
        if self.generation != session.attachment_generation:
            raise ProtocolError("The caller attachment is no longer current", "session_reconnecting")
        if not listening and session.attachment_state != "active":
            raise ProtocolError("The caller attachment is unavailable", "session_reconnecting")

    @property
    def target(self) -> str:
        """Where the work goes, in the words of ``session.delegation.created``: data only, never compared."""
        return self._part.target

    @property
    def reports_events(self) -> bool:
        """Whether each delegation's events follow."""
        return self._part.reports_events

    @property
    def acknowledgement_delay_ms(self) -> int:
        """Milliseconds before the automatic spoken acknowledgement of a delegation, read when asked."""
        return self._part.acknowledgement_delay_ms

    def begin_caller_turn(self) -> Callable[[], None] | None:
        """Register routing only for this active frontend attachment."""
        session = self._session
        if (
            self.generation == session.attachment_generation
            and session.attachment_state == "active"
            and session.status not in {"closing", "closed"}
            and callable(begin := getattr(self._part, "begin_caller_turn", None))
        ):
            return begin()
        return None

    async def delegate(self, input: list[dict]) -> str:
        """Delegate a request and return its ``delegation_id``."""
        if self._session.status in {"closing", "closed"}:
            raise RuntimeError("Session is closing")
        self._check()
        return await self._part.delegate(input)

    def listen(self, listener: DelegationEvents) -> None:
        """Hand every event of every delegation of the session to ``listener``."""
        # A frontend start can finish after logical close; it acquires no listener authority then.
        if self._session.status in {"closing", "closed"}:
            return
        self._check(listening=True)
        self._part.listen(listener)
