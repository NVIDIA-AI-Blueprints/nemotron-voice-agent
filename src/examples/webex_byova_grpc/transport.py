# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""Pipecat WebSocket transport that writes adapter audio without playback pacing."""

from __future__ import annotations

from pipecat.frames.frames import OutputAudioRawFrame
from pipecat.serializers.base_serializer import FrameSerializer
from pipecat.serializers.protobuf import ProtobufFrameSerializer
from pipecat.transports.websocket.fastapi import (
    FastAPIWebsocketOutputTransport,
    FastAPIWebsocketParams,
    FastAPIWebsocketTransport,
)

from examples.shared.pipeline_utils import (
    PIPELINE_AUDIO_IN_SAMPLE_RATE,
    PIPELINE_AUDIO_OUT_SAMPLE_RATE,
)
from utils import parse_env_int


class WebexUnpacedOutputTransport(FastAPIWebsocketOutputTransport):
    """Serialize generated audio immediately and let Webex pace playback."""

    async def write_audio_frame(self, frame: OutputAudioRawFrame) -> bool:
        """Write one normalized audio frame without simulated playback sleep."""
        if self._client.is_closing or not self._client.is_connected:
            return False
        normalized = OutputAudioRawFrame(
            audio=frame.audio,
            sample_rate=self.sample_rate,
            num_channels=self._params.audio_out_channels,
        )
        await self._write_frame(normalized)
        return True


class WebexUnpacedTransport(FastAPIWebsocketTransport):
    """FastAPI transport with an owned unpaced output implementation."""

    def __init__(self, websocket, params: FastAPIWebsocketParams) -> None:
        """Create the standard transport and replace its supported output type."""
        if params.add_wav_header:
            raise ValueError("Webex adapter transport requires raw audio")
        super().__init__(websocket=websocket, params=params)
        self._output = WebexUnpacedOutputTransport(
            self,
            self._client,
            self._params,
            name=self._output_name,
        )


def create_webex_transport(runner_args) -> WebexUnpacedTransport:
    """Create the adapter-facing protobuf WebSocket transport."""
    websocket = getattr(runner_args, "websocket", None)
    if websocket is None:
        raise TypeError(f"Webex BYOVA requires websocket runner args, got {type(runner_args)}")
    return WebexUnpacedTransport(
        websocket,
        FastAPIWebsocketParams(
            audio_in_enabled=True,
            audio_in_sample_rate=PIPELINE_AUDIO_IN_SAMPLE_RATE,
            audio_out_enabled=True,
            audio_out_sample_rate=PIPELINE_AUDIO_OUT_SAMPLE_RATE,
            audio_out_10ms_chunks=parse_env_int("AUDIO_OUT_10MS_CHUNKS", 10),
            add_wav_header=False,
            serializer=ProtobufFrameSerializer(params=FrameSerializer.InputParams(ignore_rtvi_messages=False)),
        ),
    )
