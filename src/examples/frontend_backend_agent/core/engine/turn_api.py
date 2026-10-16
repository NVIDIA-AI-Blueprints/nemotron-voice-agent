# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The turn-manager interface the voice adapter drives.

:class:`TurnManager` implements :class:`TurnManagerLike`; the Pipecat adapter's turn bridge
is typed against it, so the adapter depends on these calls only. ``Emit`` is how the turn
manager hands server events to the adapter.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol

from examples.frontend_backend_agent.core.engine.filler import Stamp

Emit = Callable[[dict[str, Any]], None]


class TurnManagerLike(Protocol):
    """The calls the voice adapter makes on its turn manager (:class:`TurnManager` implements them)."""

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
        """Filler events from the agent's event sink."""
        ...

    async def close(self) -> None:
        """Release everything."""
        ...
