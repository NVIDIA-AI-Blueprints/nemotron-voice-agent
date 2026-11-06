# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Caller audio over a primary WebSocket, and muting it."""

from __future__ import annotations

import base64
from typing import TYPE_CHECKING

from gateways.live.protocol import ProtocolError

if TYPE_CHECKING:
    from gateways.live.protocol import AudioFormat
    from gateways.live.session.session import LiveProtocolSession

MAX_AUDIO_CHARS = 128000
# The bytes of one sample of each wire format: PCM16, or one byte of G.711.
SAMPLE_BYTES = {"audio/pcm": 2, "audio/pcmu": 1, "audio/pcma": 1}


def wire_audio(message: dict, audio_format: AudioFormat) -> bytes:
    """The wire audio of one ``session.input_audio.append`` on a primary WebSocket: its base64, decoded and checked.

    Raises:
        ProtocolError: The chunk is missing, oversized or not base64, or it ends in part of a sample of the wire format.
    """
    audio = message.get("audio")
    if not isinstance(audio, str) or len(audio) > MAX_AUDIO_CHARS:
        raise ProtocolError(f"Invalid or oversized audio chunk: at most {MAX_AUDIO_CHARS} characters", param="audio")
    try:
        wire = base64.b64decode(audio, validate=True)
    except ValueError as exc:
        raise ProtocolError("Invalid base64 audio or incomplete PCM sample", param="audio") from exc
    if len(wire) % SAMPLE_BYTES[audio_format.type]:
        raise ProtocolError("Invalid base64 audio or incomplete PCM sample", param="audio")
    return wire


async def append(session: LiveProtocolSession, message: dict, source: str, client_event_id: str | None) -> None:
    """``session.input_audio.append`` as a command: refused.

    Only a primary WebSocket carries audio, and its frontend adapter's transport reads that audio from the socket
    (``gateways.live.ports.PrimarySocket``): no append of it comes here.
    """
    raise ProtocolError("Audio append is only allowed on a primary WebSocket")


async def mute(session: LiveProtocolSession, message: dict, source: str, client_event_id: str | None) -> None:
    """``session.input_audio.mute``: stop consuming caller audio, and say so."""
    await session.frontend.set_muted(True)
    await session.emit("session.input_audio.muted", client_event_id=client_event_id)


async def unmute(session: LiveProtocolSession, message: dict, source: str, client_event_id: str | None) -> None:
    """``session.input_audio.unmute``: consume caller audio again, and say so."""
    await session.frontend.set_muted(False)
    await session.emit("session.input_audio.unmuted", client_event_id=client_event_id)
