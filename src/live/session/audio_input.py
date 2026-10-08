# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Caller audio over a primary WebSocket, and muting it."""

from __future__ import annotations

import base64
from typing import TYPE_CHECKING

from live.protocol import ProtocolError

if TYPE_CHECKING:
    from live.session.session import LiveProtocolSession

MUTE_COMMANDS = {"session.input_audio.mute", "session.input_audio.unmute"}
MAX_AUDIO_CHARS = 128000


async def handle_audio(session: LiveProtocolSession, message: dict, source: str) -> None:
    """Decode one ``session.input_audio.append`` chunk and feed it to the engine."""
    if source != "primary" or session.transport != "websocket":
        raise ProtocolError("Audio append is only allowed on a primary WebSocket")
    audio = message.get("audio")
    if not isinstance(audio, str) or len(audio) > MAX_AUDIO_CHARS:
        raise ProtocolError("Invalid or oversized audio chunk", param="audio")
    try:
        pcm = session.codec.decode(base64.b64decode(audio, validate=True))
    except Exception as exc:
        raise ProtocolError("Invalid base64 audio or incomplete PCM sample", param="audio") from exc
    await session.engine.feed_audio(pcm)


async def handle_mute(session: LiveProtocolSession, kind: str, client_event_id: str | None) -> None:
    """Stop or resume consuming caller audio and announce the new state."""
    muted = kind.endswith(".mute")
    await session.engine.set_muted(muted)
    await session.emit(
        "session.input_audio.muted" if muted else "session.input_audio.unmuted",
        client_event_id=client_event_id,
    )
