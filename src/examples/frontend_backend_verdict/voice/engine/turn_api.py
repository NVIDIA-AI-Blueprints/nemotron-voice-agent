# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The turn-manager seam of :class:`RealtimeSession` (additive hook for sibling prototypes).

The default session builds :class:`TurnManager` around an :class:`AgentPort`. A
sibling prototype can pass a ``turn_manager_factory`` instead: the session then
never builds an agent, and ``session.update`` becomes transactional (validated on a
copy, handed to :meth:`TurnManagerLike.on_session_update`, committed only if that
succeeds). See ``misc/prototypes/frontend-delegation-hermes/prototype-plan.md`` section 4.3.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

from examples.frontend_backend_verdict.voice.agent.filler import FillerLog, Stamp
from examples.frontend_backend_verdict.voice.agent.sinks import EventLog
from examples.frontend_backend_verdict.voice.clock import WallClock
from examples.frontend_backend_verdict.voice.config import VoiceConfig
from examples.frontend_backend_verdict.voice.engine.output_path import OutputPath
from examples.frontend_backend_verdict.voice.wire.response_writer import ConversationOrder
from examples.frontend_backend_verdict.voice.wire.session_view import SessionSettings

Emit = Callable[[dict[str, Any]], None]


@dataclass(frozen=True, slots=True)
class SessionChange:
    """What one accepted ``session.update`` changed."""

    first: bool
    tools_changed: bool
    instructions_changed: bool


@dataclass(frozen=True, slots=True)
class TurnContext:
    """Everything the session hands a turn manager (the same collaborators :class:`TurnManager` gets)."""

    output: OutputPath
    emit: Emit
    conversation: ConversationOrder
    settings: Callable[[], SessionSettings]
    voice: Callable[[], str | None]
    config: VoiceConfig
    clock: WallClock
    audio_now: Callable[[], float]
    session_id: str
    session_start: Stamp
    filler_log: FillerLog
    event_log: EventLog
    model: str = ""
    show_silent_filler: bool = False


class TurnManagerLike(Protocol):
    """The calls :class:`RealtimeSession` makes on its turn manager."""

    @property
    def state(self) -> str:
        """A short state name for logs."""
        ...

    def on_audio_advance(self, step_ms: float) -> None:
        """Input audio advanced (moves the playout cursor)."""
        ...

    async def on_speech_started(self) -> None:
        """Confirmed user speech."""
        ...

    def on_user_input(self, user_input: Any, *, from_audio: bool) -> None:
        """A committed utterance or user text item."""
        ...

    def on_utterance_dropped(self, item_id: str, *, reason: str) -> None:
        """An utterance ended without a transcript."""
        ...

    def on_function_output(self, call_id: str, output: str) -> None:
        """A ``function_call_output`` item."""
        ...

    def on_response_create(self) -> None:
        """Client ``response.create``."""
        ...

    def delete_pending_input(self, item_id: str) -> bool:
        """Drop an unconsumed user item."""
        ...

    def on_truncate(self, item_id: str, content_index: int, audio_end_ms: int) -> None:
        """Client ``conversation.item.truncate``."""
        ...

    def speak_greeting(self) -> None:
        """Speak the configured greeting."""
        ...

    async def on_response_cancel(self) -> None:
        """Client ``response.cancel``."""
        ...

    async def on_output_audio_clear(self) -> None:
        """Client ``output_audio_buffer.clear``."""
        ...

    def on_filler(self, text: str | None, stamp: Stamp) -> None:
        """Filler events from the text agent's sink (unused by managers without one)."""
        ...

    async def on_session_update(self, settings: SessionSettings, change: SessionChange) -> None:
        """Apply a validated update; raising rejects it (nothing is committed)."""
        ...

    async def start(self) -> None:
        """Called once when the session starts serving."""
        ...

    async def close(self) -> None:
        """Release everything."""
        ...


TurnManagerFactory = Callable[[TurnContext], TurnManagerLike]
