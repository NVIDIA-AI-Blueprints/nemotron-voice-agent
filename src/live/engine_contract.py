# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""What a live session needs from the voice pipeline behind it.

The protocol layer (``session``, ``gateway``) knows nothing about models, prompts or Pipecat. An example
supplies a :class:`LiveEngine`: the protocol session forwards client commands to it, and the engine reports
events back through the session it was started with.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from live.protocol import SessionConfig
    from live.session import LiveProtocolSession


class LiveEngine(Protocol):
    """The voice pipeline and cascade serving one live session."""

    async def start(self, session: LiveProtocolSession) -> None:
        """Build and start the pipeline. Raise if it cannot start; the session then closes."""

    async def close(self) -> None:
        """Stop the pipeline and release every connection."""

    def validate_backend_model(self, model: str) -> None:
        """Raise ``ValueError`` if the backend cannot serve ``model``."""

    def update_backend(self, config: SessionConfig) -> None:
        """Apply the resolved ``delegation.responses`` settings after a ``session.update``."""

    def knows_delegation(self, delegation_id: str) -> bool:
        """Return True for a client-mode delegation id this engine issued."""

    async def set_muted(self, muted: bool) -> None:
        """Stop or resume consuming caller audio. The session and backend stay active."""

    async def feed_audio(self, pcm24k: bytes) -> None:
        """Deliver caller audio (24 kHz mono PCM16) from a primary WebSocket."""

    async def append_instructions(self, text: str) -> None:
        """Steer behavior: interrupt current speech and evaluate the new direction."""

    async def append_thinking(self, text: str) -> None:
        """Retain silent context for later decisions."""

    async def append_commentary(self, text: str, delegation_id: str | None) -> None:
        """Retain the facts and ask the talker to phrase them for the caller."""

    def item_create(self, item: dict) -> None:
        """Queue a user message or a pending function result. Raise ``ProtocolError`` if it is invalid."""

    async def response_create(self) -> None:
        """Run queued input, or continue a complete function-result batch. Raise ``ProtocolError``."""
