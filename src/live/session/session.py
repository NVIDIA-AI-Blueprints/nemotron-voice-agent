# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""One live session: its state, and the dispatch of client commands to the module that handles each.

The session owns what the protocol defines about a connection's life: the status machine, who receives which events,
and the ordering of commands. :data:`COMMANDS` maps each command kind to its handler (modules ``audio_input``,
``updates``, ``appends`` and ``tool_calling``), and ``lifecycle`` runs the session's life. Voice commands go to the
frontend adapter, delegation commands to the delegation part that :data:`DELEGATION_PARTS` gives the resolved
``delegation.type``. The adapter and the delegation part reach the session only through their own views
(:class:`AdapterView`, :class:`DelegationSideView`), which send only the events each may send, and the adapter reaches
the delegation part only through :class:`DelegationPortView`.
"""

from __future__ import annotations

import asyncio
import base64
import time
from collections.abc import Awaitable, Callable
from contextvars import ContextVar
from typing import TYPE_CHECKING, Any, Protocol

from loguru import logger

from live.client_delegation import ClientDelegation
from live.events import Fanout, Timeline, client_event_allowed, error_event, event
from live.events.fanout import Close, Send
from live.ports import Inspectable
from live.protocol import BackendConfig, ProtocolError, SessionConfig, uid
from live.server_delegation import ServerDelegation
from live.session import appends, audio_input, lifecycle, tool_calling, updates

if TYPE_CHECKING:
    from live.ports import AdapterCloseReason, CoreHook, DelegationEvents, DelegationPart, SessionParts

COMMAND_WAIT_FOR_START_SECONDS = 30
# The events the adapter and the delegation part may send through their views (live.ports); any other is refused.
ADAPTER_EVENTS = frozenset({"session.input_transcript.delta", "session.output_transcript.delta"})
DELEGATION_EVENTS = frozenset({"session.delegation.created", "response.event"})

# The delegation part each resolved delegation type gets (live.ports.DelegationPart), given the gateway's core hook.
# Only a responses part asks for the core, so a client-mode session never builds one.
DELEGATION_PARTS: dict[str, Callable[[CoreHook], DelegationPart]] = {
    "client": lambda core: ClientDelegation(),
    "responses": lambda core: ServerDelegation(core()),
}


def no_core():
    """The core hook of a server that runs no delegation core: a ``responses`` session is refused."""
    raise ProtocolError("Responses delegation is not available on this server", param="session.delegation.type")


Command = Callable[["LiveProtocolSession", dict, str, "str | None"], Awaitable[None]]
# The connection whose command runs (its subscription key). An error the command causes goes back to that connection
# only, a choice GPT-Live leaves open (it names only client_event_id), also when a task or a callback that runs in the
# command's context reports it later.
COMMAND_CONNECTION: ContextVar[str | None] = ContextVar("command_connection", default=None)


async def refuse_start(session: LiveProtocolSession, message: dict, source: str, client_event_id: str | None) -> None:
    """``session.start`` on a running session: it starts once."""
    raise ProtocolError("The session already started", "session_already_started")


async def close_requested(
    session: LiveProtocolSession, message: dict, source: str, client_event_id: str | None
) -> None:
    """``session.close``: close the session for the client."""
    await session.close("close_requested", client_event_id, asked_by=f"the client ({source} connection)")


def unsupported(feature: str) -> Command:
    """The handler of an official command this server does not offer: one refusal that names it."""

    async def refuse(session: LiveProtocolSession, message: dict, source: str, client_event_id: str | None) -> None:
        raise ProtocolError(f"{feature} is not supported by this server", param="type")

    return refuse


# Each client command of the protocol and its handler: (session, message, source, client_event_id).
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
    # Official commands of features this server does not run.
    "session.output_audio.fixture.play": unsupported("Fixture playback (session.output_audio.fixture.play)"),
    "decision.create": unsupported("The Decisions API (decision.create)"),
    "decision.item.create": unsupported("The Decisions API (decision.item.create)"),
    "decision.item.delete": unsupported("The Decisions API (decision.item.delete)"),
}


class AudioCodec(Protocol):
    """A primary WebSocket's wire audio codec, which the WebSocket edge (``live.gateway.websocket``) hands in."""

    exchange_rate: int  # the rate of the PCM16 that decode() returns and encode() takes

    def decode(self, raw: bytes) -> bytes:
        """Return PCM16 at ``exchange_rate`` for a chunk of wire audio."""

    def encode(self, pcm: bytes) -> bytes:
        """Return wire audio for a chunk of PCM16 at ``exchange_rate``."""


class LiveProtocolSession:
    """The protocol state of one live session."""

    def __init__(
        self,
        config: SessionConfig,
        transport: str,
        parts: SessionParts,
        *,
        core: CoreHook = no_core,
        codec: AudioCodec | None = None,
        max_session_seconds: int = 1800,
        on_closed: Callable[[LiveProtocolSession], Awaitable[None]] | None = None,
    ):
        """Create a session in the ``created`` state.

        Args:
            config: The validated startup configuration.
            transport: ``webrtc`` or ``websocket``.
            parts: What serves this session: its frontend adapter.
            core: The gateway's hook to its one delegation core client. A ``responses`` session builds its server
                delegation on that client; a client-mode session builds a client delegation and never asks for it.
            codec: A primary WebSocket's wire audio codec, from the WebSocket edge; ``None`` on WebRTC.
            max_session_seconds: Lifetime after which the session expires.
            on_closed: Awaited once after the session closed, for registry cleanup.

        Raises:
            ProtocolError: A ``responses`` session on a server with no delegation core.
        """
        self.id = uid("live")
        self.config, self.transport, self.parts = config, transport, parts
        self.frontend = parts.frontend
        # One delegation part, by the resolved delegation.type, which never changes during a session.
        self.delegation: DelegationPart = DELEGATION_PARTS[config.mode](core)
        self.max_session_seconds, self.on_closed = max_session_seconds, on_closed
        self.codec = codec
        self.status = "created"
        self.created_at = time.time()
        self.expires_at = int(self.created_at + max_session_seconds)
        self.started_at: float | None = None
        self.fanout = Fanout(
            lambda: self.config,
            transport,
            lambda: self.close("connection_lost", asked_by="the gateway: the primary connection fell behind or failed"),
        )
        self.timeline = Timeline()
        self.tasks: set[asyncio.Task] = set()
        self.command_lock = asyncio.Lock()
        self.close_task: asyncio.Task | None = None
        self.ready, self.closed = asyncio.Event(), asyncio.Event()
        # What each gets in place of the session: the adapter at start, the delegation part at open; and what the
        # adapter gets in place of the delegation part.
        self.adapter_view, self.delegation_view = AdapterView(self), DelegationSideView(self)
        self.delegation_port = DelegationPortView(self.delegation)

    # ----------------------------------------------------------------------------- state
    @property
    def input_ms(self) -> int:
        """Milliseconds of caller audio seen so far (the input media clock)."""
        return self.timeline.input_ms

    def clock_ms(self) -> int:
        """Return milliseconds since the session became active."""
        return 0 if self.started_at is None else int((time.monotonic() - self.started_at) * 1000)

    def snapshot(self) -> dict:
        """Return the session resource sent in lifecycle events (always ``active``, as the protocol specifies).

        It carries the session's settings without an MCP tool's credentials.
        """
        return {
            **self.config.public_dump(),
            "id": self.id,
            "status": "active",
            "expires_at": self.expires_at,
        }

    async def history(self) -> dict:
        """What the session's models hold now, for ``GET /v1/live/sessions/{id}/history``: two halves composed.

        The delegation part's half is ``backend``: the settings each request carries and the conversation's items in
        the core, or ``None`` in client mode, where the server runs no backend. The frontend adapter's half, when it is
        :class:`live.ports.Inspectable`, gives ``frontend`` and adds to ``backend`` what only it holds (its pending
        input, whether work is open) or knows better (the core's defaults, which the gateway does not know). An adapter
        that holds nothing yet (a WebRTC session's adapter starts after its offer is answered) gives no half.
        """
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

    # -------------------------------------------------------------------------- publishing
    def subscribe(self, send: Send, source: str, close: Close | None = None) -> str:
        """Attach a connection in its role (``primary`` or ``sideband``) and return its key.

        ``close(reason)`` closes a sideband that falls behind or fails, once it is detached.
        """
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
        """Send the ``error`` event of ``exc`` to the connection whose command caused it, or to every connection.

        The connection is the one whose command runs in this context (:data:`COMMAND_CONNECTION`); an error that no
        command caused goes to every connection.
        """
        connection = COMMAND_CONNECTION.get()
        if connection is None:
            await self.publish(error_event(exc, client_event_id))
        else:
            await self.fanout.send_to(connection, error_event(exc, client_event_id))

    async def report_error(self, message: str, client_event_id: str | None = None) -> None:
        """Report a recoverable pipeline failure to the client: an ``error`` event with code ``pipeline_error``."""
        logger.warning(f"session={self.id} pipeline_error={message}")
        await self.reply_error(ProtocolError(message, "pipeline_error"), client_event_id)

    async def send_output_audio(self, pcm: bytes, sample_rate: int) -> None:
        """Send reply audio to the primary connection only, in its wire format.

        The session's codec encodes the PCM16, which must be at the codec's exchange rate. The event carries no
        ``event_id``, as a primary WebSocket client expects; a chunk that the codec turns into no bytes sends nothing.
        """
        if self.codec is None:
            raise RuntimeError("Reply audio events are sent only on a primary WebSocket, which hands in the codec")
        if sample_rate != self.codec.exchange_rate:
            raise ValueError(f"Reply audio must be PCM16 at {self.codec.exchange_rate} Hz, not {sample_rate} Hz")
        wire = self.codec.encode(pcm)
        if wire:
            await self.publish(
                {"type": "session.output_audio.delta", "delta": base64.b64encode(wire).decode()}, audience="primary"
            )

    async def advance_input_clock(self, ms: int) -> None:
        """Record that the caller's audio reached ``ms``."""
        self.timeline.advance(ms)

    # ------------------------------------------------------------------------ life cycle
    async def start(self, client_event_id: str | None = None) -> None:
        """Open the delegation part, start the frontend adapter, then announce ``session.started``.

        Every log line of the start, and of the tasks the parts start in it, names the session (``stream_id``).
        """
        with logger.contextualize(stream_id=self.id):
            await lifecycle.start(self, client_event_id)

    async def close(
        self, reason: str = "close_requested", client_event_id: str | None = None, *, asked_by: str = "the gateway"
    ) -> None:
        """Close the session: close the adapter and the delegation part, emit ``session.closed``, then clean up.

        ``asked_by`` names who asked, for the one log line that records each close.
        """
        with logger.contextualize(stream_id=self.id):
            await lifecycle.close(self, reason, client_event_id, asked_by)

    # -------------------------------------------------------------------------- commands
    async def handle(self, message: Any, connection: str) -> None:
        """Run one client command from the connection ``connection`` (its subscription key).

        A rejection is an ``error`` event for that connection only. A client's mistake is a ``ProtocolError``; any other
        exception is a fault on the server's side, which is logged and reported as ``server_error`` with no detail.
        A detached connection's commands are dropped: nothing could reach it.
        """
        source = self.fanout.role(connection)
        if source is None:
            logger.debug(f"session={self.id} drops a command from a detached connection")
            return
        client_event_id = message.get("event_id") if isinstance(message, dict) else None
        token = COMMAND_CONNECTION.set(connection)
        with logger.contextualize(stream_id=self.id):
            try:
                async with self.command_lock:
                    if source == "sideband" and self.status in {"created", "starting"}:
                        await self._wait_until_ready()
                    await self._dispatch(message, source)
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
        """A sideband's command waits for a session that is still starting, within a limit."""
        try:
            async with asyncio.timeout(COMMAND_WAIT_FOR_START_SECONDS):
                await self.ready.wait()
        except TimeoutError as exc:
            raise ProtocolError("The session did not become active in time", "session_closed") from exc

    async def _dispatch(self, message: Any, source: str) -> None:
        if not isinstance(message, dict) or not isinstance(message.get("type"), str):
            raise ProtocolError("Expected a JSON event object")
        if self.status != "active":
            raise ProtocolError("Session is not active", "session_closed")
        kind, client_event_id = message["type"], message.get("event_id")
        if not client_event_allowed(self.config, self.transport, kind, source):
            raise ProtocolError("Client event is not permitted", "permission_denied")
        command = COMMANDS.get(kind)
        if command is None:
            raise ProtocolError("Unknown client event", param="type")
        await command(self, message, source, client_event_id)


def _refuse_unless_allowed(kind: str, allowed: frozenset[str], part: str) -> None:
    if kind not in allowed:
        raise ValueError(f"The {part} may not send {kind!r}; it may send only {', '.join(sorted(allowed))}")


class AdapterView:
    """The frontend adapter's narrow view of its session (:class:`live.ports.SessionPort`).

    The adapter sends its transcripts, its reply audio and its errors, advances the input clock and asks for a close.
    It may send no other event type, and the rest of the session stays out of its reach.
    """

    def __init__(self, session: LiveProtocolSession):
        """Give the frontend adapter its view of ``session``."""
        self._session = session
        self.id = session.id

    @property
    def input_ms(self) -> int:
        """Milliseconds of caller audio seen so far (the input media clock)."""
        return self._session.input_ms

    def is_active(self) -> bool:
        """Return whether the session is active."""
        return self._session.status == "active"

    async def emit(self, kind: str, **fields: Any) -> None:
        """Send a transcript event; refuse every other event type with ``ValueError``."""
        _refuse_unless_allowed(kind, ADAPTER_EVENTS, "frontend adapter")
        await self._session.emit(kind, **fields)

    async def advance_input_clock(self, ms: int) -> None:
        """Record that the caller's audio reached ``ms``."""
        await self._session.advance_input_clock(ms)

    async def send_output_audio(self, pcm: bytes, sample_rate: int) -> None:
        """Send reply audio (mono PCM16) to the primary connection only; the event carries no ``event_id``."""
        await self._session.send_output_audio(pcm, sample_rate)

    async def report_error(self, message: str, client_event_id: str | None = None) -> None:
        """Report a recoverable failure as an ``error`` event with code ``pipeline_error``."""
        await self._session.report_error(message, client_event_id)

    def request_close(self, reason: AdapterCloseReason) -> None:
        """Ask the session to close. The close is scheduled; this returns at once."""
        self._session.spawn(self._session.close(reason, asked_by="the frontend adapter"), "adapter-close")


class DelegationSideView:
    """The delegation part's narrow view of its session (:class:`live.ports.DelegationView`).

    The server delegation relays the delegation events, ``session.delegation.created`` and ``response.event``, paced
    (:meth:`emit_paced`), reports the core's refusals of the client's own requests, reads the input clock for
    ``offset_ms``, and reads the session's id and its ``delegation.responses`` settings for each request it sends. The
    client delegation sends only ``session.delegation.created``. Both read the acknowledgement delay the session's
    settings hold. Neither may send any other event type, nor read the rest of the session's settings.
    """

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
        """The session's ``delegation.responses`` settings as ``session.update`` last changed them: read, never copied.

        ``None`` in client mode.
        """
        return self._session.config.delegation.responses

    @property
    def acknowledgement_delay_ms(self) -> int:
        """The session's ``delegation.acknowledgement_delay_ms``, as ``session.update`` last changed it."""
        return self._session.config.delegation.acknowledgement_delay_ms

    async def emit(self, kind: str, **fields: Any) -> None:
        """Send a delegation event; refuse every other event type with ``ValueError``."""
        _refuse_unless_allowed(kind, DELEGATION_EVENTS, "delegation part")
        await self._session.emit(kind, **fields)

    async def emit_paced(self, kind: str, **fields: Any) -> None:
        """Send a delegation event once the primary connection's queue is at most half full.

        The relay of a delegation's events uses it, so a burst waits for the client instead of overflowing its queue
        and closing the session.
        """
        _refuse_unless_allowed(kind, DELEGATION_EVENTS, "delegation part")
        await self._session.fanout.wait_for_room()
        await self._session.emit(kind, **fields)

    def report_refusal(self, message: str, code: str, client_event_id: str | None, param: str | None = None) -> None:
        """Report the core's refusal of a request or of settings a client command made, after the command returned.

        A plain call: the ``error`` event is scheduled. Called in the command's context, it reaches the connection that
        sent the command, as the command's own errors do; the start's settings, which no command sent, are reported to
        every connection.
        """
        self._session.spawn(self._session.reply_error(ProtocolError(message, code, param), client_event_id), "refusal")


class DelegationPortView:
    """The frontend adapter's narrow view of the session's delegation part (:class:`live.ports.DelegationPort`).

    The adapter delegates and listens through it, and reads the part's data; the session's own commands, the core's
    client and the part's state stay out of its reach.
    """

    def __init__(self, part: DelegationPart):
        """Give the frontend adapter its view of the delegation ``part``."""
        self._part = part

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

    async def delegate(self, input: list[dict]) -> str:
        """Delegate a request and return its ``delegation_id``."""
        return await self._part.delegate(input)

    def listen(self, listener: DelegationEvents) -> None:
        """Hand every event of every delegation of the session to ``listener``."""
        self._part.listen(listener)
