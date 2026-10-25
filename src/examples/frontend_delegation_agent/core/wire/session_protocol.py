# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""OpenAI Realtime client-event routing shared by every head.

Client events are handled strictly in arrival order. ``session.update`` is
transactional: it is validated on a copy, handed to the turn manager, and
committed only if the turn manager accepts it. Everything that touches input
audio goes through the head's :class:`HeadInput`, so the routing itself has no
audio, VAD or transport logic.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any, Protocol

from loguru import logger

from examples.frontend_delegation_agent.core.errors import WireProtocolError
from examples.frontend_delegation_agent.core.head_api import Emit, SessionChange, TurnManagerLike, UserInput
from examples.frontend_delegation_agent.core.session_log import SessionLog
from examples.frontend_delegation_agent.core.voice_config import VoiceConfig
from examples.frontend_delegation_agent.core.wire import client_events as ce
from examples.frontend_delegation_agent.core.wire import server_events as ev
from examples.frontend_delegation_agent.core.wire.ids import new_item_id
from examples.frontend_delegation_agent.core.wire.response_writer import ConversationOrder
from examples.frontend_delegation_agent.core.wire.session_view import RealtimeSessionView, SessionSettings

Receive = Callable[[], Awaitable[str | bytes | None]]


class HeadInput(Protocol):
    """The input side a head gives the routing: audio, commit, clear and format changes."""

    async def append(self, audio: bytes) -> None:
        """``input_audio_buffer.append`` (decoded bytes)."""
        ...

    async def commit(self) -> None:
        """``input_audio_buffer.commit``."""
        ...

    async def clear(self) -> str | None:
        """``input_audio_buffer.clear``: drop the open utterance and buffered audio; the dropped item id."""
        ...

    def configure(self, settings: SessionSettings) -> None:
        """Apply an accepted update's input format and turn detection."""
        ...


class RealtimeProtocol:
    """Routes one session's client events to the turn manager and the head's input side."""

    def __init__(
        self,
        *,
        config: VoiceConfig,
        emit: Emit,
        view: RealtimeSessionView,
        conversation: ConversationOrder,
        head: HeadInput,
        log: SessionLog,
        session_id: str,
    ) -> None:
        """Bind the session's collaborators; :meth:`bind` adds the turn manager."""
        self.config = config
        self._emit = emit
        self.view = view
        self.conversation = conversation
        self._head = head
        self._log = log
        self.session_id = session_id
        self.turns: TurnManagerLike | None = None
        self.configured = False
        self._warned: set[str] = set()

    def bind(self, turns: TurnManagerLike) -> None:
        """Attach the session's turn manager (built after this object, it needs :meth:`resolve_voice`)."""
        self.turns = turns

    async def serve(self, receive: Receive) -> None:
        """Handle client frames until ``receive`` returns ``None``."""
        timeout = self.config.server.session_update_timeout_s
        while True:
            try:
                if not self.configured:
                    message = await asyncio.wait_for(receive(), timeout=timeout)
                else:
                    message = await receive()
            except TimeoutError:
                logger.warning(f"[{self.session_id}] no session.update within {timeout}s; using defaults")
                self.configured = True
                continue
            if message is None:
                break
            await self.handle_message(message)

    async def handle_message(self, message: str | bytes) -> None:
        """Handle one client frame; protocol errors become non-fatal ``error`` events."""
        event_id: str | None = None
        try:
            data = ce.decode_message(message)
            event_id = data.get("event_id") if isinstance(data.get("event_id"), str) else None
            command = ce.parse_client_event(data)
            await self._dispatch(command)
        except WireProtocolError as exc:
            self._emit(ev.error(str(exc), code=exc.code, param=exc.param, client_event_id=event_id))

    async def _dispatch(self, command: ce.ClientCommand) -> None:
        turns = self.turns
        assert turns is not None  # noqa: S101 - bound before serving
        if isinstance(command, ce.AudioAppend):
            await self._head.append(command.audio)
        elif isinstance(command, ce.SessionUpdate):
            await self._on_session_update(command.session)
        elif isinstance(command, ce.ItemCreate):
            self._on_item_create(command)
        elif isinstance(command, ce.ResponseCreate):
            overrides = {key for key in ("instructions", "tools", "tool_choice") if key in command.overrides}
            if overrides:
                raise WireProtocolError(
                    f"per-response overrides {sorted(overrides)} are not supported",
                    code="unsupported_response_override",
                    param="response",
                )
            turns.on_response_create()
        elif isinstance(command, ce.AudioCommit):
            await self._head.commit()
        elif isinstance(command, ce.AudioClear):
            cleared = await self._head.clear()
            if cleared is not None:
                turns.on_utterance_dropped(cleared, reason="audio_cleared")
            self._emit(ev.audio_cleared())
        elif isinstance(command, ce.ItemTruncate):
            turns.on_truncate(command.item_id, command.content_index, command.audio_end_ms)
        elif isinstance(command, ce.ItemDelete):
            if not turns.delete_pending_input(command.item_id):
                raise WireProtocolError(
                    f"item {command.item_id!r} cannot be deleted (only unconsumed user text items can)",
                    code="invalid_value",
                    param="item_id",
                )
            self._emit(ev.item_deleted(command.item_id))
        elif isinstance(command, ce.ResponseCancel):
            await turns.on_response_cancel()
        elif isinstance(command, ce.OutputAudioBufferClear):
            await turns.on_output_audio_clear()

    async def _on_session_update(self, patch: dict[str, Any]) -> None:
        """Validate on a copy, let the turn manager accept it, then commit (nothing changes on rejection)."""
        turns = self.turns
        assert turns is not None  # noqa: S101 - bound before serving
        preview = self.view.preview(patch)
        result = preview.result
        settings = result.settings
        first = not self.configured
        change = SessionChange(
            first=first, tools_changed=result.tools_changed, instructions_changed=result.instructions_changed
        )
        try:
            await turns.on_session_update(settings, change)
        except WireProtocolError:
            raise
        except Exception as exc:  # noqa: BLE001 - a rejected update must not kill the session
            raise WireProtocolError(f"session.update rejected: {exc}", param="session") from exc
        preview.commit()
        for warning in result.warnings:
            if warning not in self._warned:
                self._warned.add(warning)
                logger.warning(f"[{self.session_id}] {warning}")
        self._head.configure(settings)
        self.configured = True
        self._log.session_updated(settings)
        self._emit(ev.session_updated(self.view.public()))
        if first and self.config.protocol.greeting_enabled:
            turns.speak_greeting()

    def _on_item_create(self, command: ce.ItemCreate) -> None:
        turns = self.turns
        assert turns is not None  # noqa: S101 - bound before serving
        item = command.item
        if isinstance(item, ce.FunctionCallOutput):
            turns.on_function_output(item.call_id, item.output)
            item_id = item.item_id or new_item_id()
            wire_item = ev.function_call_output_item(item_id, call_id=item.call_id, output=item.output)
        else:
            item_id = item.item_id or new_item_id()
            wire_item = ev.user_text_item(item_id, item.text)
        previous = self.conversation.append(item_id)
        self._emit(ev.item_added(wire_item, previous))
        self._emit(ev.item_done(wire_item, previous))
        if isinstance(item, ce.UserText):
            turns.on_user_input(UserInput(text=item.text, item_id=item_id), from_audio=False)

    def resolve_voice(self) -> str | None:
        """The TTS voice for the client's ``voice`` (``tts.voice_map``), or ``None`` for the catalog default."""
        requested = self.view.settings.voice
        voice_map = self.config.tts.voice_map
        if requested in voice_map:
            return voice_map[requested] or None
        if requested and requested not in self._warned:
            self._warned.add(requested)
            logger.warning(f"[{self.session_id}] voice {requested!r} not in tts.voice_map; using the catalog default")
        return None
