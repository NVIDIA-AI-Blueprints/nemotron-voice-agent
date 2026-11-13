# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The ports between the live gateway and the parts that serve a session: types and their rules."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, Protocol, runtime_checkable

from gateways.live.protocol import AudioFormat, BackendConfig, SessionConfig

if TYPE_CHECKING:
    from starlette.websockets import WebSocketState

AdapterCloseReason = Literal["connection_lost", "remote_hangup"]


class SessionPort(Protocol):
    """The frontend adapter's view of its session (adapter → gateway)."""

    id: str
    input_ms: int
    # A WebSocket session's primary socket, which the adapter reads; None on WebRTC.
    socket: PrimarySocket | None

    def is_active(self) -> bool:
        """Return whether the session is active."""

    async def emit(self, kind: str, **fields: Any) -> None:
        """Send a server event: only ``session.input_transcript.delta`` and ``session.output_transcript.delta``."""

    async def advance_input_clock(self, ms: int) -> None:
        """Record that the caller's audio reached ``ms``; the adapter calls it as it consumes caller audio."""

    async def send_output_audio(self, audio: bytes) -> None:
        """Send reply audio, in the session's wire format, to the primary WebSocket only; no ``event_id``."""

    async def report_error(self, message: str, client_event_id: str | None = None, *, fatal: bool = False) -> None:
        """Report a failure with code ``pipeline_error``.

        A command's or a ``fatal`` failure is an ``error`` event; any other is logged and sent as an ``info`` notice.
        """

    def request_close(self, reason: AdapterCloseReason) -> None:
        """Ask the session to close; the close is scheduled and this returns at once."""


class FrontendAdapter(Protocol):
    """Hears the caller and speaks to them for one session (gateway → adapter).

    Mute and the appends run under the session's command lock, so they return quickly; slow work runs in tasks.
    """

    async def start(self, session: SessionPort, delegation: DelegationPort) -> None:
        """Build and start the frontend, which delegates to ``delegation``; raise if it cannot, and the session closes.

        It runs once, before any other call. The client's instructions and appended ones survive every memory trim.
        """

    async def close(self) -> None:
        """Stop the frontend and release its connections; it runs once, after the last call, and is safe twice."""

    async def set_muted(self, muted: bool) -> None:
        """Stop or resume consuming caller audio. The session and its delegation part stay active."""

    async def append_instructions(
        self, text: str, client_event_id: str | None = None, delegation_id: str | None = None
    ) -> None:
        """Steer behavior: interrupt current speech and evaluate the new direction; never cancel an acknowledgement."""

    async def append_thinking(
        self, text: str, client_event_id: str | None = None, delegation_id: str | None = None
    ) -> None:
        """Retain silent context for later decisions; blank or not, it cancels every pending acknowledgement."""

    async def append_commentary(
        self, text: str, client_event_id: str | None = None, delegation_id: str | None = None
    ) -> None:
        """Retain the facts and phrase them for the caller; blank or not, it cancels every pending acknowledgement.

        ``delegation_id``, in every append, is a client delegation the session announced, or ``None``.
        """


class DelegationEvents(Protocol):
    """Hears every event of every delegation of a session (delegation part → frontend adapter)."""

    def on_delegation_event(self, delegation_id: str, event: dict) -> None:
        """Take one official Responses streaming event of a delegation, typed runs included; it only records or queues.

        Each request gets one response: ``response.created``, items in ``response.output_item.done`` (messages whose
        ``phase`` is ``commentary`` or ``final_answer``, ``mcp_call`` items the backend ran, ``function_call`` items for
        the app), then exactly one terminal. A takeover names the response it takes over in ``previous_response_id``.
        """

    def on_delegation_lost(self, delegation_id: str) -> None:
        """Take the loss of the delegation's stream: it ended with no terminal, so no event will end the delegation."""


class DelegationPort(Protocol):
    """Where the frontend adapter hands its delegations: its narrow view of the session's delegation part.

    The adapter tells the parts apart by :attr:`reports_events` only, never by :attr:`target`, which is data.
    """

    target: str
    reports_events: bool
    # Milliseconds before a delegation's spoken acknowledgement, read at each hand-off; 0: at once.
    acknowledgement_delay_ms: int

    async def delegate(self, input: list[dict]) -> str:
        """Delegate a request and return its ``delegation_id``; no event of it reaches the listener before this returns.

        The server delegation returns once the core accepts the request and raises on a refusal, so a refused delegation
        never gets its lead-in spoken; the client delegation announces it to the client app and returns at once.
        """

    def listen(self, listener: DelegationEvents) -> None:
        """Hand every event of every delegation of the session to ``listener`` (a client delegation has none)."""


class DelegationPart(DelegationPort, Protocol):
    """The session's delegation part, by the resolved mode; a member that does not apply refuses the call."""

    # The session's conversation with its delegation core; None in client mode, and once a close deleted it.
    conversation: str | None
    # How long close step 4 waits for :meth:`close` before it goes on.
    close_seconds: float

    async def open(self, view: DelegationView | None) -> None:
        """Open the part: the server delegation creates the session's conversation. Safe to call twice."""

    def stop_handing(self) -> None:
        """Close step 2: send no new request and hand nothing more to the frontend, nor announce anything more."""

    async def close(self) -> None:
        """Close step 4: release the part. Safe twice, and before, during or after :meth:`open`."""

    def knows(self, delegation_id: object) -> bool:
        """Return whether an append may name ``delegation_id``: a client delegation this session announced."""

    async def update(self, config: SessionConfig, patch: dict, client_event_id: str | None) -> SessionConfig:
        """Apply a ``session.update`` patch and return the new settings; the backend's check runs afterwards.

        Either part refuses a type change; the responses part's core reports a refused setting as nonfatal.
        """

    async def item_create(self, item: dict, client_event_id: str | None) -> None:
        """Take a function output or a typed message from ``response.item.create``, or refuse it."""

    async def response_create(self, client_event_id: str | None) -> None:
        """Send what the client left pending, for ``response.create``, or refuse it."""

    async def history(self) -> dict | None:
        """Return the part's half of ``/history`` (the backend's items and settings), or ``None`` in client mode."""


class DelegationView(Protocol):
    """The delegation part's view of its session, whose settings it reads when asked, never copied."""

    id: str
    input_ms: int
    responses: BackendConfig | None
    acknowledgement_delay_ms: int
    # The session's store: true, its opt-in to keep backend work that outlives the call.
    store: bool

    async def emit(self, kind: str, **fields: Any) -> None:
        """Send a server event: only ``session.delegation.created`` and ``response.event``."""

    async def emit_paced(self, kind: str, **fields: Any) -> None:
        """Send a delegation event as :meth:`emit` does, once the primary connection's queue is at most half full."""

    def report_refusal(self, message: str, code: str, client_event_id: str | None, param: str | None = None) -> None:
        """Report the core's refusal of a request or settings that a client command made; a plain call.

        Called in the command's context, the scheduled ``error`` reaches the connection that sent the command.
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
        """Take ``text`` the application typed as a user message, which the backend runs; it is never a caller turn."""


@runtime_checkable
class Inspectable(Protocol):
    """Optional on a frontend adapter: its half of ``GET /v1/live/sessions/{id}/history``."""

    async def history(self) -> dict:
        """Return what the adapter holds now as plain JSON: ``frontend``, and what it holds of the backend's side."""


class PrimarySocket(Protocol):
    """A WebSocket session's primary socket as its adapter's transport reads it; the session is its one writer."""

    format: AudioFormat
    client_state: WebSocketState

    async def receive(self) -> dict:
        """Return the client's next frame, as Starlette's ``WebSocket.receive()`` gives it."""

    async def close(self, code: int = 1000, reason: str | None = None) -> None:
        """Leave the socket open: the gateway closes it once ``session.closed`` has gone out."""

    async def handle(self, data: str | bytes) -> bytes | None:
        """Take a client frame; return a ``session.input_audio.append``'s wire audio, else ``None``; never raise."""


class MediaConnection(Protocol):
    """A WebRTC session's peer connection, which the factory makes; the gateway uses it through these calls only."""

    async def initialize(self, sdp: str, type: str) -> None:
        """Take the client's SDP offer (``type`` is ``offer``) and make the answer."""

    def get_answer(self) -> dict | None:
        """Return the SDP answer, whose ``sdp`` the gateway returns; ``None`` before :meth:`initialize`."""

    def event_handler(self, event_name: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """Return a decorator that adds a handler of ``event_name``, called with the connection.

        The gateway handles ``client-event`` (the event is the second argument, and may come before the peer connects),
        ``closed`` (also when the data channel closes) and ``failed``.
        """

    def send_app_message(self, message: dict) -> None:
        """Send one event on the data channel, or queue it until the channel opens."""

    def is_connected(self) -> bool:
        """Return whether the peer is connected; it is not until the frontend adapter's transport connects it."""

    async def close_after_drain(self) -> None:
        """Let the data channel flush the events it holds, then close the peer connection."""


@dataclass(frozen=True)
class SessionParts:
    """What a composition root returns for one session: its frontend adapter and, on WebRTC, its peer connection."""

    frontend: FrontendAdapter
    connection: MediaConnection | None = None


# Builds a session's parts from its config, its transport (``webrtc`` or ``websocket``) and, on WebRTC, the ICE servers
# for its peer; it raises ``ProtocolError`` to refuse the session.
AdapterFactory = Callable[[SessionConfig, str, list[dict] | None], Awaitable[SessionParts]]
# Builds the gateway's one delegation core client (``ResponsesClient``), or an awaitable of it, on first use.
CoreHook = Callable[[], Any]
