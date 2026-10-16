# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The adapter's input-audio clock: audio now, speech onset and offset, and the playout cursor.

Ported from FDH ``adapters/pipecat/timing.py`` (``a0ef50c``), ``InputCursor`` only (plan §5,
B4). FDH's wall-clock ``PlayoutEstimator`` is not used: as in v1, the turn manager's playout
cursor advances on the **input** audio clock (S1), so a client that pauses its input (Tau)
pauses what counts as heard.

* **audio now**: input audio received, counted in pipeline order right after the VAD, so a
  VAD event arrives after exactly the audio it analyzed.
* **step**: each input frame carries v1's step for its append (client samples at the client
  rate, ``fbv_step_ms`` in the frame metadata), so the clock advances exactly as v1's did;
  ``on_advance`` (the turn manager's ``on_audio_advance``) gets every step in pipeline order.
* **onset / offset**: the cursor when Pipecat's VAD confirms a start or stop, minus the VAD's
  ``start_secs`` / ``stop_secs`` (the confirmation delay), so ``audio_start_ms`` points at the
  speech itself (B1).
"""

from __future__ import annotations

from collections.abc import Callable

from pipecat.frames.frames import (
    Frame,
    InputAudioRawFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

#: Frame-metadata key carrying v1's audio-clock step for one client append (milliseconds).
STEP_KEY = "fbv_step_ms"


class InputCursor(FrameProcessor):
    """Counts input audio in pipeline order, advances the playout cursor, and records VAD onsets and offsets."""

    def __init__(self, **kwargs: object) -> None:
        """Start at audio time zero; :attr:`on_advance` is bound by the session."""
        super().__init__(**kwargs)
        self.now_ms = 0.0
        self.onset_ms: float | None = None
        self.offset_ms: float | None = None
        self.on_advance: Callable[[float], None] | None = None

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        """Advance on input audio; note the VAD events that follow it downstream."""
        await super().process_frame(frame, direction)
        if direction != FrameDirection.DOWNSTREAM:
            await self.push_frame(frame, direction)
            return
        if isinstance(frame, InputAudioRawFrame):
            step = self._step(frame)
            if step > 0:
                self.now_ms += step
                if self.on_advance is not None:
                    self.on_advance(step)
        elif isinstance(frame, VADUserStartedSpeakingFrame):
            self.onset_ms = max(0.0, self.now_ms - frame.start_secs * 1000.0)
        elif isinstance(frame, VADUserStoppedSpeakingFrame):
            self.offset_ms = max(0.0, self.now_ms - frame.stop_secs * 1000.0)
        await self.push_frame(frame, direction)

    @staticmethod
    def _step(frame: InputAudioRawFrame) -> float:
        step = (frame.metadata or {}).get(STEP_KEY)
        if isinstance(step, (int, float)):
            return float(step)
        if frame.sample_rate <= 0:
            return 0.0
        samples = len(frame.audio) // (2 * max(1, frame.num_channels))
        return samples * 1000.0 / frame.sample_rate

    def take_onset(self) -> tuple[float, str]:
        """The onset of the turn that just started, and where it came from (``vad`` or ``transcription``)."""
        onset, self.onset_ms = self.onset_ms, None
        if onset is None:
            return self.now_ms, "transcription"
        return onset, "vad"

    def take_offset(self) -> float:
        """The offset of the turn that just stopped (the cursor when there was no VAD stop)."""
        offset, self.offset_ms = self.offset_ms, None
        return self.now_ms if offset is None else offset
