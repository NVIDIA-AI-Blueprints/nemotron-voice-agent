# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""What each line's audio did, read around the audio output where the assistant aggregator cannot tell."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from fractions import Fraction
from typing import TYPE_CHECKING

from pipecat.frames.frames import (
    DataFrame,
    Frame,
    InterruptionFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    OutputAudioRawFrame,
    TTSStoppedFrame,
    TTSTextFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from nemotron_voice_agent.pipecat.examples.frontend_backend_live.talker import RESPONSE_KEY

if TYPE_CHECKING:
    from nemotron_voice_agent.pipecat.examples.frontend_backend_live.spoken import SpokenLines

SILENT_KEY = "silent_sentence"
STITCHED_FAILURE_KEY = "stitched_synthesis_failed"
_PCM_POSITION = "frontend_pcm_position"


@dataclass
class SentenceSynthesisFailed(DataFrame):
    """A failed sentence in the same audio queue as its text."""

    context_id: str


class SilentSentences(FrameProcessor):
    """Mark text whose sentence failed or received no audio in per-sentence TTS."""

    def __init__(self, **kwargs):
        """Start with no audio yet."""
        super().__init__(**kwargs)
        self._audio = False
        self._failed_context: str | None = None

    async def process_frame(self, frame, direction: FrameDirection) -> None:
        """Use the sentence's ordered synthesis and audio evidence before passing its text on."""
        await super().process_frame(frame, direction)
        if direction is FrameDirection.DOWNSTREAM:
            if isinstance(frame, OutputAudioRawFrame) and frame.audio:
                self._audio = True
            elif isinstance(frame, SentenceSynthesisFailed):
                self._failed_context = frame.context_id
                return
            elif isinstance(frame, TTSTextFrame):
                failed = self._failed_context is not None and self._failed_context == frame.context_id
                if failed or not self._audio:
                    frame.metadata[SILENT_KEY] = True
                if failed:
                    self._failed_context = None
                self._audio = False
            elif isinstance(frame, (LLMFullResponseStartFrame, InterruptionFrame)):
                self._audio = False
                self._failed_context = None
        await self.push_frame(frame, direction)


class SentenceAudioPositions(FrameProcessor):
    """Attach each text boundary to the sample duration sent into the output transport."""

    def __init__(self):
        """Begin with an empty audio stream."""
        super().__init__()
        self._position = Fraction()

    async def process_frame(self, frame, direction: FrameDirection) -> None:
        """Mark text and stop boundaries before Pipecat resamples or buffers audio."""
        await super().process_frame(frame, direction)
        if direction is FrameDirection.DOWNSTREAM:
            if isinstance(frame, OutputAudioRawFrame):
                self._position += Fraction(frame.num_frames, frame.sample_rate)
            elif isinstance(frame, (TTSTextFrame, TTSStoppedFrame)):
                frame.metadata[_PCM_POSITION] = self._position
            elif isinstance(frame, (LLMFullResponseStartFrame, InterruptionFrame)):
                self._position = Fraction()
        await self.push_frame(frame, direction)


class SilentLines(FrameProcessor):
    """Reports each response's playout to the voice's lines, and holds its text until its audio plays."""

    def __init__(self, lines: SpokenLines, **kwargs):
        """Report to ``lines``."""
        super().__init__(**kwargs)
        self.lines = lines
        self._response: int | None = None
        self._playing = False
        self._sent = Fraction()
        self._held: deque[tuple[Fraction, TTSTextFrame]] = deque()

    async def process_frame(self, frame, direction: FrameDirection) -> None:
        """Read the playout of each response as its frames leave the output; pass every frame a caller heard."""
        await super().process_frame(frame, direction)
        frames = self._read(frame) if direction is FrameDirection.DOWNSTREAM else [frame]
        for each in frames:
            await self.push_frame(each, direction)

    def _read(self, frame) -> list[Frame]:
        """Note ``frame``'s fact; return the frames that go on now, in order."""
        if isinstance(frame, LLMFullResponseStartFrame) and RESPONSE_KEY in frame.metadata:
            self._reset(frame.metadata[RESPONSE_KEY])
        elif isinstance(frame, OutputAudioRawFrame) and self._response is not None and frame.audio:
            self._sent += Fraction(frame.num_frames, frame.sample_rate)
            if not self._playing:
                self._playing = True
                self.lines.started(self._response)
            return [frame, *self._spoken_text()]
        elif isinstance(frame, TTSTextFrame):
            position = frame.metadata.pop(_PCM_POSITION, Fraction())
            if SILENT_KEY in frame.metadata:
                return []
            if self._response is not None:
                self._held.append((position, frame))
                return self._spoken_text()
        elif isinstance(frame, TTSStoppedFrame):
            # Pipecat may pad its final packet. That padding cannot cover the next speech segment.
            self._sent = min(self._sent, frame.metadata.pop(_PCM_POSITION, self._sent))
        elif isinstance(frame, LLMFullResponseEndFrame) and RESPONSE_KEY in frame.metadata:
            if self.lines.ended(
                frame.metadata[RESPONSE_KEY], synthesis_failed=frame.metadata.get(STITCHED_FAILURE_KEY, False)
            ):
                frame.interruptible = False
            self._reset(None)
        elif isinstance(frame, InterruptionFrame):
            self.lines.cut(self._response, frame.id, newest=frame.metadata.get(RESPONSE_KEY))
            self._reset(None)
        return [frame]

    def _spoken_text(self) -> list[Frame]:
        frames: list[Frame] = []
        while self._playing and self._held and self._held[0][0] <= self._sent:
            frames.append(self._held.popleft()[1])
        return frames

    def _reset(self, response: int | None) -> None:
        self._response = response
        self._playing = False
        self._sent = Fraction()
        self._held.clear()
