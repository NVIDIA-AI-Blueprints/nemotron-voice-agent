# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The one head-neutral interface between an adapter (server head) and the shared core.

An adapter owns the transport, speech and timing. For each session it builds a
:class:`TurnContext` (with a :class:`SpeechOutput`, an ``emit`` sink, the audio clock and
the playout feed) and gets a :class:`TurnManagerLike` from
``DelegationRuntime.turn_manager``. ``session.update`` is transactional: validated on a
copy, handed to :meth:`TurnManagerLike.on_session_update`, committed only if that
succeeds (``core/wire/session_protocol.py``). Nothing here depends on an adapter.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

from examples.frontend_delegation_agent.core.clock import WallClock
from examples.frontend_delegation_agent.core.engine.filler import FillerLog, Stamp
from examples.frontend_delegation_agent.core.engine.sinks import EventLog
from examples.frontend_delegation_agent.core.voice_config import VoiceConfig
from examples.frontend_delegation_agent.core.wire.response_writer import ConversationOrder, MessageItemWriter
from examples.frontend_delegation_agent.core.wire.session_view import SessionSettings

if TYPE_CHECKING:
    from examples.frontend_delegation_agent.core.audio.formats import AudioFormat
    from examples.frontend_delegation_agent.core.engine.playback import ResponseProgress

Emit = Callable[[dict[str, Any]], None]


@dataclass(slots=True)
class UserInput:
    """A committed user utterance or text item, with its timing anchors."""

    text: str
    turn_start: Stamp | None = None
    turn_end: Stamp | None = None
    asr_final: Stamp | None = None
    item_id: str | None = None


class PreparedSpeech(Protocol):
    """One item's text, split and possibly already synthesizing (a head's own type)."""

    async def aclose(self) -> None:
        """Stop any synthesis still running."""
        ...


class SpeechOutput(Protocol):
    """How a head turns agent text into transcript and audio deltas for one item at a time."""

    def prepare(
        self, text: str, *, kind: str, modality: str, voice: str | None, out_format: AudioFormat
    ) -> PreparedSpeech:
        """Split ``text`` and, for audio, start synthesizing as early as the head can."""
        ...

    async def speak(
        self,
        prepared: Any,
        item: MessageItemWriter,
        progress: ResponseProgress,
        *,
        out_format: AudioFormat,
        on_first_audio: Callable[[], None] | None = None,
    ) -> None:
        """Send one item's transcript and audio, feeding ``progress``; cancellation leaves the item open."""
        ...


@dataclass(frozen=True, slots=True)
class SessionChange:
    """What one accepted ``session.update`` changed."""

    first: bool
    tools_changed: bool
    instructions_changed: bool


@dataclass(frozen=True, slots=True)
class TurnContext:
    """Everything the session hands a turn manager (the same collaborators :class:`TurnManager` gets)."""

    output: SpeechOutput
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
