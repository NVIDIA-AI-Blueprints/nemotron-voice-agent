# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""OpenAI Realtime (GA) over Pipecat's FastAPI WebSocket transport.

Ported from FDH ``adapters/pipecat/serializer.py`` (``a0ef50c``) without its wall-clock
playout estimate (B4: the playout cursor moves on the input-audio clock instead).

Inbound, every client message goes through the session's router; only
``input_audio_buffer.append`` produces a frame (the decoded, resampled audio, carrying v1's
audio-clock step). Outbound, Realtime event frames are sent as JSON, and the transport's
audio becomes ``response.output_audio.delta`` events in the client's format, stamped with
the item the last :class:`ItemAudioFrame` named (B10: Pipecat's 10 ms chunks).
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

from examples.frontend_backend_agent.adapters.pipecat.frames import ItemAudioFrame
from examples.frontend_backend_agent.adapters.pipecat.timing import STEP_KEY
from examples.frontend_backend_agent.core.audio.resample import StreamResampler
from examples.frontend_backend_agent.core.engine import server_events as ev

if TYPE_CHECKING:
    from examples.frontend_backend_agent.adapters.pipecat.session import PipecatSession


class RealtimeSerializer(FrameSerializer):
    """One session's wire codec."""

    def __init__(self, session: PipecatSession) -> None:
        """Bind the session (routing and output format)."""
        super().__init__()
        self._session = session
        self._item: ItemAudioFrame | None = None
        self._resampler: StreamResampler | None = None
        self._resampler_rates: tuple[int, int] = (0, 0)
        #: Audio written to the client (milliseconds of synthesized audio); a test checks heard <= written (B5).
        self.written_ms = 0.0

    async def serialize(self, frame: Frame) -> str | bytes | None:
        """Realtime JSON for event and audio frames; nothing for markers and other frames."""
        if isinstance(frame, ItemAudioFrame):
            # The item's audio ends (or the next begins): its resampler tail is the last delta.
            tail = self._flush()
            self._item = frame if frame.begin else None
            return tail
        if isinstance(frame, (OutputTransportMessageFrame, OutputTransportMessageUrgentFrame)):
            message = frame.message
            return json.dumps(message) if isinstance(message, dict) else None
        if isinstance(frame, OutputAudioRawFrame):
            return self._audio(frame)
        if isinstance(frame, InterruptionFrame):
            self._item = None
            self._resampler = None
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
        self.written_ms += len(frame.audio) * 1000.0 / (frame.sample_rate * 2 * max(1, frame.num_channels))
        return self._delta(item, pcm)

    def _flush(self) -> str | None:
        """The current item's resampler tail as a last delta; the resampler is dropped."""
        resampler, self._resampler = self._resampler, None
        if resampler is None or self._item is None:
            return None
        return self._delta(self._item, resampler.flush())

    def _delta(self, item: ItemAudioFrame, pcm: bytes) -> str | None:
        if not pcm:
            return None
        payload = base64.b64encode(self._session.output_format().encode(pcm)).decode("ascii")
        return json.dumps(ev.output_audio_delta(item.response_id, item.item_id, item.output_index, payload))

    async def deserialize(self, data: str | bytes) -> Frame | None:
        """Route one client message; the decoded input audio for an append, else nothing."""
        decoded = await self._session.on_client_message(data)
        if decoded is None:
            return None
        audio, step_ms = decoded
        frame = InputAudioRawFrame(audio=audio, sample_rate=self._session.input_rate, num_channels=1)
        frame.metadata[STEP_KEY] = step_ms
        return frame
