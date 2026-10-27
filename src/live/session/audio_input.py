# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Caller audio over a primary WebSocket, and muting it."""

from __future__ import annotations

import base64
from typing import TYPE_CHECKING

from live.protocol import ProtocolError

if TYPE_CHECKING:
    from live.session.session import LiveProtocolSession

MAX_AUDIO_CHARS = 128000


async def append(session: LiveProtocolSession, message: dict, source: str, client_event_id: str | None) -> None:
    """Decode one ``session.input_audio.append`` chunk and feed it to the frontend adapter at the codec's rate.

    Only the primary connection of a session whose edge handed in a wire codec carries audio: the primary WebSocket.
    """
    if source != "primary" or session.codec is None:
        raise ProtocolError("Audio append is only allowed on a primary WebSocket")
    audio = message.get("audio")
    if not isinstance(audio, str) or len(audio) > MAX_AUDIO_CHARS:
        raise ProtocolError(f"Invalid or oversized audio chunk: at most {MAX_AUDIO_CHARS} characters", param="audio")
    try:
        pcm = session.codec.decode(base64.b64decode(audio, validate=True))
    except Exception as exc:
        raise ProtocolError("Invalid base64 audio or incomplete PCM sample", param="audio") from exc
    await session.frontend.feed_audio(pcm, session.codec.exchange_rate)


async def mute(session: LiveProtocolSession, message: dict, source: str, client_event_id: str | None) -> None:
    """``session.input_audio.mute``: stop consuming caller audio, and say so."""
    await session.frontend.set_muted(True)
    await session.emit("session.input_audio.muted", client_event_id=client_event_id)


async def unmute(session: LiveProtocolSession, message: dict, source: str, client_event_id: str | None) -> None:
    """``session.input_audio.unmute``: consume caller audio again, and say so."""
    await session.frontend.set_muted(False)
    await session.emit("session.input_audio.unmuted", client_event_id=client_event_id)
