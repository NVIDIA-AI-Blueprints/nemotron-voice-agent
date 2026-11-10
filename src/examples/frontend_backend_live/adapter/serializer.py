# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pipecat frame serializer for a live session's primary WebSocket, on Pipecat's ``FastAPIWebsocketTransport``."""

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

# (audio, rate in, rate out, the resampler that keeps the stream's state) -> audio
Convert = Callable[[bytes, int, int, BaseAudioResampler], Awaitable[bytes]]


async def resample(audio: bytes, in_rate: int, out_rate: int, resampler: BaseAudioResampler) -> bytes:
    """Resample PCM16 ``audio`` from ``in_rate`` to ``out_rate``; the same rate passes it unchanged."""
    return await resampler.resample(audio, in_rate, out_rate)


class Codec(NamedTuple):
    """How one wire format's audio becomes PCM16, and PCM16 becomes it."""

    decode: Convert
    encode: Convert


CODECS: dict[str, Codec] = {
    "audio/pcm": Codec(decode=resample, encode=resample),
    "audio/pcmu": Codec(decode=ulaw_to_pcm, encode=pcm_to_ulaw),
    "audio/pcma": Codec(decode=alaw_to_pcm, encode=pcm_to_alaw),
}


class LiveFrameSerializer(FrameSerializer):
    """Convert a primary WebSocket's audio events to Pipecat audio frames, and reply audio back."""

    def __init__(self, socket: PrimarySocket, session: SessionPort, audio_in_rate: int):
        """Read ``socket`` for ``session``, whose frontend hears the caller at ``audio_in_rate``."""
        # The resamplers keep their state across a pause, so no caller audio is dropped.
        super().__init__(FrameSerializer.InputParams(resampler_clear_after_secs=None))
        self.socket, self.session = socket, session
        self.rate = socket.format.rate
        self.audio_in_rate = audio_in_rate
        self._codec = CODECS[socket.format.type]
        self._input_resampler = create_stream_resampler(clear_after_secs=self._params.resampler_clear_after_secs)
        self._output_resampler = create_stream_resampler(clear_after_secs=self._params.resampler_clear_after_secs)

    async def deserialize(self, data: str | bytes) -> Frame | None:
        """Return the caller's audio at the frontend's rate; ``None`` for any other event, which the socket ran."""
        try:
            audio = await self.socket.handle(data)
            if not audio:
                return None
            pcm = await self._codec.decode(audio, self.rate, self.audio_in_rate, self._input_resampler)
        except Exception:
            # Pipecat's receive loop ends on an exception.
            logger.exception("Caller audio from the primary WebSocket could not be read")
            return None
        if not pcm:
            return None
        return InputAudioRawFrame(audio=pcm, sample_rate=self.audio_in_rate, num_channels=1)

    async def serialize(self, frame: Frame) -> str | bytes | None:
        """Send reply audio through the session, the socket's one writer; the transport itself sends nothing."""
        if isinstance(frame, OutputAudioRawFrame):
            await self.send(frame.audio, frame.sample_rate)
        return None

    async def send(self, pcm: bytes, sample_rate: int) -> None:
        """Send mono PCM16 at ``sample_rate`` to the caller as ``session.output_audio.delta``."""
        audio = await self._codec.encode(pcm, sample_rate, self.rate, self._output_resampler)
        await self.session.send_output_audio(audio)
