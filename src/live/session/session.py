# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""One live session: its state, and the dispatch of client commands to the module that handles each.

The session owns what the protocol defines about a connection's life: the status machine, who receives which events,
and the ordering of commands. Each kind of command lives in its own module (``audio_input``, ``updates``, ``appends``,
``tool_calling``) and the life cycle in ``lifecycle``. Everything model- or audio-specific is the :class:`LiveEngine`'s.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from typing import Any

from loguru import logger
from pydantic import ValidationError

from live.engine_contract import LiveEngine
from live.events import Fanout, Timeline, client_event_allowed, error_event, event
from live.events.fanout import Send
from live.media.audio import WireCodec
from live.protocol import ProtocolError, SessionConfig, uid
from live.session import appends, audio_input, lifecycle, tool_calling, updates

COMMAND_WAIT_FOR_START_SECONDS = 30


class LiveProtocolSession:
    """The protocol state of one live session."""

    def __init__(
        self,
        config: SessionConfig,
        transport: str,
        engine: LiveEngine,
        *,
        max_session_seconds: int = 1800,
        on_closed: Callable[[LiveProtocolSession], Awaitable[None]] | None = None,
    ):
        """Create a session in the ``created`` state.

        Args:
            config: The validated startup configuration.
            transport: ``webrtc`` or ``websocket``.
            engine: The voice pipeline serving this session.
            max_session_seconds: Lifetime after which the session expires.
            on_closed: Awaited once after the session closed, for registry cleanup.
        """
        self.id = uid("live")
        self.config, self.transport, self.engine = config, transport, engine
        self.max_session_seconds, self.on_closed = max_session_seconds, on_closed
        self.codec = WireCodec(config.audio.format) if config.audio.format else WireCodec(None)
        self.status = "created"
        self.created_at = time.time()
        self.expires_at = int(self.created_at + max_session_seconds)
        self.started_at: float | None = None
        self.fanout = Fanout(lambda: self.config, transport, lambda: self.close("connection_lost"))
        self.timeline = Timeline()
        self.tasks: set[asyncio.Task] = set()
        self.command_lock = asyncio.Lock()
        self.close_lock = asyncio.Lock()
        self.ready, self.closed = asyncio.Event(), asyncio.Event()

    # ----------------------------------------------------------------------------- state
    @property
    def input_ms(self) -> int:
        """Milliseconds of caller audio seen so far (the input media clock)."""
        return self.timeline.input_ms

    def clock_ms(self) -> int:
        """Return milliseconds since the session became active."""
        return 0 if self.started_at is None else int((time.monotonic() - self.started_at) * 1000)

    def snapshot(self) -> dict:
        """Return the session resource sent in lifecycle events (always ``active``, as the protocol specifies)."""
        return {
            **self.config.model_dump(exclude_none=True),
            "id": self.id,
            "status": "active",
            "expires_at": self.expires_at,
        }

    def spawn(self, coro, name: str) -> asyncio.Task:
        """Run ``coro`` as a session-owned task that is cancelled on close."""
        task = asyncio.create_task(coro, name=f"{self.id}:{name}")
        self.tasks.add(task)
        task.add_done_callback(self.tasks.discard)
        return task

    # -------------------------------------------------------------------------- publishing
    def subscribe(self, send: Send, source: str = "sideband") -> str:
        """Attach a connection (``primary`` or ``sideband``) and return its key."""
        return self.fanout.subscribe(send, source)

    async def unsubscribe(self, key: str) -> None:
        """Detach a connection."""
        await self.fanout.unsubscribe(key)

    async def publish(self, message: dict, audience: str = "all") -> None:
        """Send ``message`` to every permitted connection of ``audience`` (``all``, ``primary`` or ``sideband``)."""
        await self.fanout.publish(message, audience)

    async def emit(self, kind: str, **fields: Any) -> None:
        """Build and publish a server event."""
        await self.publish(event(kind, **fields))

    async def fail(self, message: str) -> None:
        """Report a recoverable pipeline failure to the client."""
        logger.warning(f"session={self.id} pipeline_error={message}")
        await self.emit(
            "error",
            error={
                "type": "server_error",
                "code": "pipeline_error",
                "message": message,
                "param": None,
                "client_event_id": None,
            },
        )

    async def advance_input_clock(self, ms: int) -> None:
        """Record that the caller's audio reached ``ms``; release acknowledgments that were waiting for it."""
        for ack in self.timeline.advance(ms):
            await self.publish(ack)

    # ------------------------------------------------------------------------ life cycle
    async def start(self, client_event_id: str | None = None) -> None:
        """Start the engine, then announce ``session.started``."""
        await lifecycle.start(self, client_event_id)

    async def close(self, reason: str = "close_requested", client_event_id: str | None = None) -> None:
        """Close the session: stop the engine, emit ``session.closed``, drain it to each peer, then clean up."""
        await lifecycle.close(self, reason, client_event_id)

    # -------------------------------------------------------------------------- commands
    async def handle(self, message: Any, source: str = "primary") -> None:
        """Run one client command; any rejection is reported as an ``error`` event."""
        client_event_id = message.get("event_id") if isinstance(message, dict) else None
        try:
            async with self.command_lock:
                if source == "sideband" and self.status in {"created", "starting"}:
                    async with asyncio.timeout(COMMAND_WAIT_FOR_START_SECONDS):
                        await self.ready.wait()
                await self._dispatch(message, source)
        except (ProtocolError, ValidationError, ValueError, TypeError, TimeoutError) as exc:
            if not isinstance(exc, ProtocolError):
                exc = ProtocolError(str(exc)[:800])
            await self.publish(error_event(exc, client_event_id))
        except Exception as exc:
            # A fault in the engine must not end the session or the connection: report it and keep serving.
            logger.exception(f"session={self.id} command {client_event_id!r} failed: {exc}")
            await self.publish(
                error_event(ProtocolError("The command could not be processed.", "server_error"), client_event_id)
            )

    async def _dispatch(self, message: Any, source: str) -> None:
        if not isinstance(message, dict) or not isinstance(message.get("type"), str):
            raise ProtocolError("Expected a JSON event object")
        if self.status != "active":
            raise ProtocolError("Session is not active", "session_closed")
        kind, client_event_id = message["type"], message.get("event_id")
        if not client_event_allowed(self.config, self.transport, kind, source):
            raise ProtocolError("Client event is not permitted", "permission_denied")
        if kind == "session.start":
            raise ProtocolError("The session already started", "session_already_started")
        if kind == "session.input_audio.append":
            await audio_input.handle_audio(self, message, source)
        elif kind in audio_input.MUTE_COMMANDS:
            await audio_input.handle_mute(self, kind, client_event_id)
        elif kind == "session.update":
            await updates.handle_update(self, message, client_event_id)
        elif kind in appends.APPEND_KINDS:
            await appends.handle_append(self, kind, message, client_event_id)
        elif kind in tool_calling.RESPONSE_COMMANDS:
            await tool_calling.handle_response_command(self, kind, message)
        elif kind == "session.close":
            await self.close("close_requested", client_event_id)
        else:
            raise ProtocolError("Unknown client event", param="type")
