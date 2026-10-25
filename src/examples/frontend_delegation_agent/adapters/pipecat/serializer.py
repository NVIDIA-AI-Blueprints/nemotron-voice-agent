# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""OpenAI Realtime (GA) over Pipecat's FastAPI WebSocket transport.

Inbound, every client event goes through the shared routing
(:class:`~examples.frontend_delegation_agent.core.wire.session_protocol.RealtimeProtocol`);
only ``input_audio_buffer.append`` produces a frame (the decoded, resampled audio).
Outbound, Realtime event frames are sent as JSON, and the transport's audio becomes
``response.output_audio.delta`` events in the client's format, stamped with the item the
last :class:`ItemAudioFrame` named. Every written chunk feeds the playout estimate.
"""

from __future__ import annotations

import base64
import json
from typing import TYPE_CHECKING

from pipecat.frames.frames import (
    Frame,
    InputAudioRawFrame,
    InterruptionFrame,
    OutputAudioRawFrame,
    OutputTransportMessageFrame,
    OutputTransportMessageUrgentFrame,
)
from pipecat.serializers.base_serializer import FrameSerializer

from examples.frontend_delegation_agent.adapters.pipecat.frames import ItemAudioFrame
from examples.frontend_delegation_agent.core.audio.resample import StreamResampler
from examples.frontend_delegation_agent.core.wire import server_events as ev

if TYPE_CHECKING:
    from examples.frontend_delegation_agent.adapters.pipecat.session import PipecatSession


class RealtimeSerializer(FrameSerializer):
    """One session's wire codec."""

    def __init__(self, session: PipecatSession) -> None:
        """Bind the session (routing, output format, playout estimate)."""
        super().__init__()
        self._session = session
        self._item: ItemAudioFrame | None = None
        self._resampler: StreamResampler | None = None
        self._resampler_rates: tuple[int, int] = (0, 0)

    async def serialize(self, frame: Frame) -> str | bytes | None:
        """Realtime JSON for event and audio frames; nothing for markers and other frames."""
        if isinstance(frame, ItemAudioFrame):
            self._item = frame if frame.begin else None
            self._resampler = None
            return None
        if isinstance(frame, (OutputTransportMessageFrame, OutputTransportMessageUrgentFrame)):
            message = frame.message
            return json.dumps(message) if isinstance(message, dict) else None
        if isinstance(frame, OutputAudioRawFrame):
            return self._audio(frame)
        if isinstance(frame, InterruptionFrame):
            self._item = None
            self._resampler = None
            self._session.playout.on_interruption()
            return None
        return None

    def _audio(self, frame: OutputAudioRawFrame) -> str | None:
        item = self._item
        if item is None or not frame.audio:
            return None
        out_format = self._session.output_format()
        rates = (frame.sample_rate, out_format.rate)
        if self._resampler is None or self._resampler_rates != rates:
            self._resampler = StreamResampler(frame.sample_rate, out_format.rate)
            self._resampler_rates = rates
        pcm = self._resampler.process(frame.audio)
        if not pcm:
            return None
        payload = base64.b64encode(out_format.encode(pcm)).decode("ascii")
        written_ms = len(frame.audio) * 1000.0 / (frame.sample_rate * 2 * max(1, frame.num_channels))
        self._session.playout.on_written(written_ms)
        return json.dumps(ev.output_audio_delta(item.response_id, item.item_id, item.output_index, payload))

    async def deserialize(self, data: str | bytes) -> Frame | None:
        """Route one client event; the decoded input audio for an append, else nothing."""
        audio = await self._session.on_client_message(data)
        if not audio:
            return None
        return InputAudioRawFrame(audio=audio, sample_rate=self._session.input_rate, num_channels=1)
