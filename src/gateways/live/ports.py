# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The ports between the live gateway and the parts that serve a session: types and their rules.

A composition root mounts the routes with ``gateways.live.mount.mount_live``. Its :data:`AdapterFactory` builds each
session's parts (:class:`SessionParts`): the frontend adapter and, for a WebRTC session, the peer connection
(:class:`MediaConnection`). Its ``core`` hook builds, on first use, the one delegation core client the
gateway holds (``gateways.live.server_delegation.responses_client.ResponsesClient``); a client-mode session never asks
for it. The frontend adapter (:class:`FrontendAdapter`) hears and speaks to the caller and reports back through
:class:`SessionPort`; a WebSocket session's adapter reads the caller from the primary socket the gateway accepted
(:class:`PrimarySocket`). The session builds its delegation part (:class:`DelegationPart`) from ``DELEGATION_PARTS``
by its resolved delegation type: the server delegation in ``responses`` mode, the client delegation in client mode. The
adapter hands each delegation to the part through :class:`DelegationPort` and learns about it only from the official
events handed back (:class:`DelegationEvents`). The session opens the delegation part before the frontend adapter and
closes them in reverse order. Nothing here knows about models, prompts or Pipecat.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, Protocol, runtime_checkable

from gateways.live.protocol import AudioFormat, BackendConfig, SessionConfig

if TYPE_CHECKING:
    from starlette.websockets import WebSocketState

# Why an adapter asks to close its session: only facts the adapter can know.
AdapterCloseReason = Literal["connection_lost", "remote_hangup"]


class SessionPort(Protocol):
    """The frontend adapter's view of its session (adapter → gateway)."""

    id: str
    input_ms: int  # the input clock: milliseconds of caller audio seen so far
    socket: PrimarySocket | None  # a WebSocket session's primary socket, which the adapter reads; None on WebRTC

    def is_active(self) -> bool:
        """Return whether the session is active."""

    async def emit(self, kind: str, **fields: Any) -> None:
        """Send a server event: only ``session.input_transcript.delta`` and ``session.output_transcript.delta``."""

    async def advance_input_clock(self, ms: int) -> None:
        """Record that the caller's audio reached ``ms``."""

    async def send_output_audio(self, audio: bytes) -> None:
        """Send reply audio, in the session's wire format, to the primary WebSocket only; no ``event_id``."""

    async def report_error(self, message: str, client_event_id: str | None = None, *, fatal: bool = False) -> None:
        """Report a failure, with code ``pipeline_error``.

        The failure of the client command ``client_event_id`` is an ``error`` event for that command, and so is a
        ``fatal`` failure, after which the adapter asks to close. Any other failure is recoverable and no client
        command caused it: it is logged as a warning and sent as an ``info`` notice, never as an ``error``, which a
        client may take as the end of the session.
        """

    def request_close(self, reason: AdapterCloseReason) -> None:
        """Ask the session to close. The close is scheduled; this returns at once."""


class FrontendAdapter(Protocol):
    """Hears the caller and speaks to them for one session (gateway → adapter): the six voice calls.

    The rules every adapter keeps:

    * every call runs under the session's command lock, so it returns quickly and never waits on a model or on speech
      (a 1 s append delays the session's audio by 1 s); slow work runs in the adapter's own tasks;
    * the adapter advances the input clock (:meth:`SessionPort.advance_input_clock`) as it consumes caller audio, which
      places the delegation announcements and the appends' acknowledgments on the session's timeline;
    * :meth:`start` runs once, before any other call; :meth:`close` runs once after the last, and is safe twice;
    * the prompt messages (the client's instructions and every appended instruction) survive every trim of the
      frontend's memory.
    """

    async def start(self, session: SessionPort, delegation: DelegationPort) -> None:
        """Build and start the frontend, which delegates to ``delegation``. Raise if it cannot; the session closes."""

    async def close(self) -> None:
        """Stop the frontend and release its connections. Safe to call twice."""

    async def set_muted(self, muted: bool) -> None:
        """Stop or resume consuming caller audio. The session and its delegation part stay active."""

    async def append_instructions(
        self, text: str, client_event_id: str | None = None, delegation_id: str | None = None
    ) -> None:
        """Steer behavior: interrupt current speech and evaluate the new direction.

        An empty (blank) ``text`` gives no direction, so it does nothing.
        """

    async def append_thinking(
        self, text: str, client_event_id: str | None = None, delegation_id: str | None = None
    ) -> None:
        """Retain silent context for later decisions; an empty (blank) ``text`` retains nothing.

        Empty or not, it cancels every pending acknowledgement, as :attr:`DelegationPort.acknowledgement_delay_ms` says.
        """

    async def append_commentary(
        self, text: str, client_event_id: str | None = None, delegation_id: str | None = None
    ) -> None:
        """Retain the facts and phrase them for the caller; an empty (blank) ``text`` has none to retain or phrase.

        Empty or not, it cancels every pending acknowledgement, as :attr:`DelegationPort.acknowledgement_delay_ms` says.
        ``delegation_id`` names the client delegation the facts answer (client mode; the session has checked that it
        announced it), or is ``None`` for session-wide context. The other appends take it the same way.
        """


class DelegationEvents(Protocol):
    """Hears every event of every delegation of a session (delegation part → frontend adapter).

    **The delegation event rules**, which a frontend reads by and the delegation core writes by: each delegation has
    one response per request (its first, then one per continuation), and each response:

    * opens with ``response.created`` (status ``queued`` once it is accepted, with ``response.queued``) and goes
      ``response.in_progress`` when work runs for it;
    * carries items, read from ``response.output_item.done``: messages with ``phase`` (``commentary`` is progress, each
      ``final_answer`` is one ready result, several allowed), ``mcp_call`` items for tools the backend ran (``show``
      with server label ``delegation_core`` carries display, never spoken), and ``function_call`` items for the app;
    * ends with exactly one terminal: ``response.completed``, ``response.failed`` (an ``error`` object) or
      ``response.incomplete`` (status ``cancelled`` and no reason at close; a reason such as ``max_output_tokens``
      otherwise). A ``completed`` whose output has function calls waits for the app; one with no item answers a late
      continuation. A handover is the older delegation's one ``commentary`` line then ``completed``, after the
      taker's ``in_progress`` named the older response in ``previous_response_id``. A Work cancelled for a newer
      request answers its delegation with a ``final_answer`` line (stopped, or not confirmed), then ``completed``.
    """

    def on_delegation_event(self, delegation_id: str, event: dict) -> None:
        """Take one official Responses streaming event of the delegation ``delegation_id``, as the core wrote it.

        Typed runs are included; the client gets the same event, redacted. A plain call: it only records or queues, so
        the reader that hands it never waits on the voice.
        """

    def on_delegation_lost(self, delegation_id: str) -> None:
        """A transport fact: the delegation's stream ended with no terminal, so no event will end it. A plain call."""


class DelegationPort(Protocol):
    """Where the frontend adapter hands its delegations: its narrow view of the session's delegation part.

    In ``responses`` mode it is the server delegation, whose events follow; in client mode it is the client
    delegation: the client app does the work and answers with appends, and no event follows. The adapter decides by
    :attr:`reports_events`, never by comparing :attr:`target`.
    """

    target: str  # where the work goes, in the words of session.delegation.created: data only, never compared
    reports_events: bool  # whether each delegation's events follow (responses: True; client: False)
    # Milliseconds to wait before the automatic spoken acknowledgement of a delegation: 0 asks at once.
    # Appended commentary or thinking cancels every pending acknowledgement, other delegations' too; an instructions
    # append does not. An update applies to later delegations.
    acknowledgement_delay_ms: int

    async def delegate(self, input: list[dict]) -> str:
        """Delegate a request and return its ``delegation_id``.

        The server delegation returns once the core accepted the request. The client delegation returns at once: it
        announces the delegation to the client app (from close step 2 on, it announces nothing) and sends ``input``
        nowhere. A request the core refuses raises before any event, so a refused delegation never gets its lead-in
        spoken. No event of the delegation reaches the listener before this returns. A part that reports events
        refuses a delegation while no listener was given (:meth:`listen` comes first).
        """

    def listen(self, listener: DelegationEvents) -> None:
        """Hand every event of every delegation of the session to ``listener`` (a client delegation has none)."""


class DelegationPart(DelegationPort, Protocol):
    """The session's delegation part: one per session, ``DELEGATION_PARTS[config.mode]``.

    ``config.mode`` is the resolved mode, so an omitted, ``null`` or ``{}`` delegation builds the client part.
    Both parts have every member; a member that does not apply refuses the call.
    """

    conversation: str | None  # the session's conversation with its delegation core (client mode: None)
    close_seconds: float  # how long :meth:`close` may take before the session stops waiting for it (close step 4)

    async def open(self, view: DelegationView | None) -> None:
        """Open the part: the server delegation creates the session's conversation. Safe to call twice."""

    def stop_handing(self) -> None:
        """Close step 2: send no new request and hand the frontend nothing more; the client part announces nothing more.

        The server part still relays to the client the ends its core writes until its close (step 4), so they arrive
        before ``session.closed``.
        """

    async def close(self) -> None:
        """Close step 4: release the part. Safe twice, and before, during or after :meth:`open`."""

    def knows(self, delegation_id: object) -> bool:
        """Whether an append may name ``delegation_id``: a client delegation this session announced."""

    async def update(self, config: SessionConfig, patch: dict, client_event_id: str | None) -> SessionConfig:
        """Apply a ``session.update`` patch and return the new settings; validation of the backend runs afterwards.

        The client part accepts ``{delegation: {type: "client", acknowledgement_delay_ms}}`` and ``delegation: null``
        (a no-op). Either part refuses a type change (``immutable_field_update``); the client part refuses
        ``responses`` settings. The responses part returns at once and then asks its core, in the background, whether
        it can honor the settings; a refused setting reaches the client as a nonfatal ``responses_validation_failed``.
        The session checks the settings it started with as an update that changes nothing (``{}``).
        """

    async def item_create(self, item: dict, client_event_id: str | None) -> None:
        """``response.item.create``: take a function output or a typed message, or refuse it."""

    async def response_create(self, client_event_id: str | None) -> None:
        """``response.create``: send what the client left pending, or refuse it."""

    async def history(self) -> dict | None:
        """The part's half of ``/history`` (the backend's items and settings), or ``None`` in client mode."""


class DelegationView(Protocol):
    """The delegation part's view of its session (delegation part → gateway session).

    It gives the part only what the part reads of the session's settings, each read when asked (never copied): the
    ``delegation.responses`` block each request carries, and the acknowledgement delay the part reports on its port.
    The client delegation uses only ``input_ms``, ``acknowledgement_delay_ms`` and ``emit``, for
    ``session.delegation.created``. Both sides of this port are in the gateway.
    """

    id: str  # the session's id, for each request's metadata
    input_ms: int  # the input clock, for ``offset_ms``
    responses: BackendConfig | None  # the session's delegation.responses settings now (client mode: None)
    acknowledgement_delay_ms: int  # the session's delegation.acknowledgement_delay_ms now

    async def emit(self, kind: str, **fields: Any) -> None:
        """Send a server event: only ``session.delegation.created`` and ``response.event``."""

    async def emit_paced(self, kind: str, **fields: Any) -> None:
        """Send a delegation event as :meth:`emit` does, once the primary connection's queue is at most half full."""

    def report_refusal(self, message: str, code: str, client_event_id: str | None, param: str | None = None) -> None:
        """Report the core's refusal of a request or of settings a client command made, once the command has returned.

        A plain call that schedules the ``error`` event. Called in the command's context, the error reaches the
        connection that sent the command.
        """


@runtime_checkable
class ContextMeter(Protocol):
    """Optional on a frontend adapter: how much its model's context holds, for the session's usage report."""

    def context_tokens(self) -> int | None:
        """Return the tokens the frontend's conversation holds now, or ``None`` when it keeps no count."""

    def context_window_tokens(self) -> int | None:
        """Return the size of the frontend's context window, or ``None`` for the gateway's default."""


@runtime_checkable
class TypedReader(Protocol):
    """Optional on a frontend adapter: the application's typed questions, which its frontend's memory reads."""

    def typed(self, text: str) -> None:
        """The application typed ``text`` as a user message (``response.item.create``), which the backend will run.

        The session calls it once the delegation part has taken the message, so the voice knows what the application
        asked when it words the answer; it never makes the message a caller turn.
        """


@runtime_checkable
class Inspectable(Protocol):
    """Optional on a frontend adapter: its half of ``GET /v1/live/sessions/{id}/history``."""

    async def history(self) -> dict:
        """What the adapter holds now, as plain JSON: ``frontend``, and what it holds of the backend's side.

        The session adds the delegation part's half under ``backend`` (:meth:`DelegationPart.history`); in client mode
        the part has none, and ``backend`` is ``None``.
        """


class PrimarySocket(Protocol):
    """A WebSocket session's primary socket, which the gateway accepted, as its frontend adapter reads it.

    The adapter's transport reads the client's frames with the calls a socket offers (:meth:`receive`, ``client_state``,
    :meth:`close`) and hands each to :meth:`handle`. It sends nothing on the socket: every server event, reply audio
    included (:meth:`SessionPort.send_output_audio`), goes out through the session, the socket's one writer. The
    socket's ``close`` leaves it open: the gateway closes it once ``session.closed`` has gone out.
    """

    format: AudioFormat  # the session's wire audio format
    client_state: WebSocketState  # whether the client is still connected

    async def receive(self) -> dict:
        """The client's next frame, as Starlette's ``WebSocket.receive()`` gives it."""

    async def close(self, code: int = 1000, reason: str | None = None) -> None:
        """Leave the socket open: the gateway closes it once ``session.closed`` has gone out."""

    async def handle(self, data: str | bytes) -> bytes | None:
        """Take one frame the client sent: the wire audio of ``session.input_audio.append``, or ``None``.

        The session runs every other event as the primary connection's command, and answers a frame it refuses with an
        ``error`` event. It never raises.
        """


class MediaConnection(Protocol):
    """A WebRTC session's peer connection: the factory makes it, and the frontend adapter's transport uses it.

    The gateway answers the client's SDP offer with it, carries the session's events on its data channel and closes it,
    with these calls only.
    """

    async def initialize(self, sdp: str, type: str) -> None:
        """Take the client's SDP offer (``type`` is ``offer``) and make the answer."""

    def get_answer(self) -> dict | None:
        """The SDP answer, whose ``sdp`` the gateway returns; ``None`` before :meth:`initialize`."""

    def event_handler(self, event_name: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """A decorator that adds a handler of ``event_name``, called with the connection.

        The gateway handles ``app-message`` (a client event on the data channel, the handler's second argument),
        ``closed``, ``data-channel-closed`` and ``failed``.
        """

    def send_app_message(self, message: dict) -> None:
        """Send one event on the data channel, or queue it until the channel opens."""

    def is_connected(self) -> bool:
        """Return whether the peer is connected."""

    async def close_after_drain(self) -> None:
        """Let the data channel flush the events it holds, then close the peer connection."""


@dataclass(frozen=True)
class SessionParts:
    """What a composition root returns for one session: its frontend adapter and, on WebRTC, its peer connection."""

    frontend: FrontendAdapter
    connection: MediaConnection | None = None  # a WebRTC session's peer, on the ICE servers the factory got


# Builds the parts of a new session from its configuration, its transport (``webrtc`` or ``websocket``) and the ICE
# servers of a WebRTC session's peer (``None`` on a WebSocket): plain ``{urls}`` objects, with ``username`` and
# ``credential`` when they have them, on which the factory makes the peer connection that the parts carry. A factory
# raises ``ProtocolError`` to refuse a session.
AdapterFactory = Callable[[SessionConfig, str, list[dict] | None], Awaitable[SessionParts]]
# Builds the gateway's one delegation core client (``ResponsesClient``), or an awaitable of it. The gateway runs it
# on first use (again after a failed build), so a client-mode session never builds a core, and closes the client
# (``aclose``) at shutdown.
CoreHook = Callable[[], Any]
