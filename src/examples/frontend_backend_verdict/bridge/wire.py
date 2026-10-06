# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pipecat frames and serializer that carry the prototype's Realtime wire unchanged.

The prototype session parses every client event itself (``voice/wire/client_events.py``)
and writes every server event through one FIFO writer (``voice/wire/writer.py``). The
serializer therefore does no protocol work: each WebSocket message becomes one
:class:`PrototypeClientMessageFrame`, and each server event the session writes leaves as
one :class:`~pipecat.frames.frames.OutputTransportMessageUrgentFrame` whose text is sent
as is. Pipecat's websocket transport keeps both directions in order.
"""

from __future__ import annotations

from dataclasses import dataclass

from pipecat.frames.frames import DataFrame, Frame, OutputTransportMessageUrgentFrame
from pipecat.serializers.base_serializer import FrameSerializer


@dataclass
class PrototypeClientMessageFrame(DataFrame):
    """One client WebSocket message, text or bytes, exactly as received."""

    message: str | bytes

    def __str__(self) -> str:
        """Log the size only; audio appends are large."""
        return f"{self.name}({len(self.message)} chars)"


class PrototypeWireSerializer(FrameSerializer):
    """Pass Realtime messages through: parsing and event building belong to the prototype session."""

    async def serialize(self, frame: Frame) -> str | bytes | None:
        """Send a server event the session wrote; ignore every other frame."""
        if isinstance(frame, OutputTransportMessageUrgentFrame) and isinstance(frame.message, str):
            return frame.message
        return None

    async def deserialize(self, data: str | bytes) -> Frame | None:
        """Wrap one client message for the session processor."""
        return PrototypeClientMessageFrame(message=data)
