# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pipecat FrameSerializer for a live session's primary WebSocket, on Pipecat's ``FastAPIWebsocketTransport``.

The transport reads the gateway's socket (``gateways.live.ports.PrimarySocket``), which takes every client event:
it runs each one as the primary connection's command and gives back the wire audio of ``session.input_audio.append``.
This serializer converts that audio, and the reply audio, between the session's wire format and PCM16 with Pipecat's
audio helpers (:data:`CODECS`).
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, NamedTuple

from loguru import logger
from pipecat.audio.resamplers.base_audio_resampler import BaseAudioResampler
from pipecat.audio.utils import alaw_to_pcm, create_stream_resampler, pcm_to_alaw, pcm_to_ulaw, ulaw_to_pcm
from pipecat.frames.frames import Frame, InputAudioRawFrame, OutputAudioRawFrame
from pipecat.serializers.base_serializer import FrameSerializer

if TYPE_CHECKING:
    from gateways.live.ports import PrimarySocket, SessionPort

# A conversion of mono audio: (audio, rate in, rate out, the stream resampler it keeps its state in) to audio.
Convert = Callable[[bytes, int, int, BaseAudioResampler], Awaitable[bytes]]


async def resample(audio: bytes, in_rate: int, out_rate: int, resampler: BaseAudioResampler) -> bytes:
    """PCM16 at another rate (``audio/pcm`` carries PCM16 as it is); the same rate passes unchanged."""
    return await resampler.resample(audio, in_rate, out_rate)


class Codec(NamedTuple):
    """How one wire format's audio becomes PCM16, and PCM16 becomes it."""

    decode: Convert
    encode: Convert


# How each wire format converts: the one place the format's type picks a conversion.
CODECS: dict[str, Codec] = {
    "audio/pcm": Codec(decode=resample, encode=resample),
    "audio/pcmu": Codec(decode=ulaw_to_pcm, encode=pcm_to_ulaw),
    "audio/pcma": Codec(decode=alaw_to_pcm, encode=pcm_to_alaw),
}


class LiveFrameSerializer(FrameSerializer):
    """Convert a primary WebSocket's audio events and Pipecat audio frames on ``FastAPIWebsocketTransport``.

    Caller audio becomes an ``InputAudioRawFrame`` at the rate the frontend hears; every other client event the
    gateway's socket runs, and it becomes no frame. Reply audio goes out through the session as
    ``session.output_audio.delta`` (:meth:`send`), so the session stays the socket's one writer: no frame becomes a
    payload of the transport's own. It never raises, since Pipecat's receive loop ends on an exception.
    """

    def __init__(self, socket: PrimarySocket, session: SessionPort, audio_in_rate: int):
        """Read ``socket`` for ``session``, whose frontend hears the caller at ``audio_in_rate``.

        The caller's resampler keeps what it holds back across a pause in the caller's stream, so no caller audio is
        dropped and the input clock never falls further behind.
        """
        super().__init__(FrameSerializer.InputParams(resampler_clear_after_secs=None))
        self.socket, self.session = socket, session
        self.rate = socket.format.rate  # the wire rate, at which the output transport hands over reply audio
        self.audio_in_rate = audio_in_rate
        self._codec = CODECS[socket.format.type]
        self._input_resampler = create_stream_resampler(clear_after_secs=self._params.resampler_clear_after_secs)
        self._output_resampler = create_stream_resampler(clear_after_secs=self._params.resampler_clear_after_secs)

    async def deserialize(self, data: str | bytes) -> Frame | None:
        """The caller's audio, at the frontend's rate; ``None`` for every other frame, which the socket took."""
        try:
            audio = await self.socket.handle(data)
            if not audio:
                return None
            pcm = await self._codec.decode(audio, self.rate, self.audio_in_rate, self._input_resampler)
        except Exception:
            logger.exception("Caller audio from the primary WebSocket could not be read")
            return None
        if not pcm:
            return None
        return InputAudioRawFrame(audio=pcm, sample_rate=self.audio_in_rate, num_channels=1)

    async def serialize(self, frame: Frame) -> str | bytes | None:
        """Send reply audio through the session (:meth:`send`); the transport itself sends nothing."""
        if isinstance(frame, OutputAudioRawFrame):
            await self.send(frame.audio, frame.sample_rate)
        return None

    async def send(self, pcm: bytes, sample_rate: int) -> None:
        """Send mono PCM16 at ``sample_rate`` to the caller in the wire format, as ``session.output_audio.delta``."""
        audio = await self._codec.encode(pcm, sample_rate, self.rate, self._output_resampler)
        await self.session.send_output_audio(audio)
