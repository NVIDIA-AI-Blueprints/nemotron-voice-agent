# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""What each line's audio did, read around the audio output where the assistant aggregator cannot tell."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pipecat.frames.frames import (
    Frame,
    InterruptionFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    OutputAudioRawFrame,
    TTSTextFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from examples.frontend_backend_live.talker import RESPONSE_KEY

if TYPE_CHECKING:
    from examples.frontend_backend_live.spoken import SpokenLines

SILENT_KEY = "silent_sentence"


class SilentSentences(FrameProcessor):
    """Marks each sentence with no audio, right after a per-sentence TTS: only there does text follow its own audio."""

    def __init__(self, **kwargs):
        """Start with no audio yet."""
        super().__init__(**kwargs)
        self._audio = False

    async def process_frame(self, frame, direction: FrameDirection) -> None:
        """Note each sentence's audio, and mark its text when none came; pass every frame on."""
        await super().process_frame(frame, direction)
        if direction is FrameDirection.DOWNSTREAM:
            if isinstance(frame, OutputAudioRawFrame) and frame.audio:
                self._audio = True
            elif isinstance(frame, TTSTextFrame):
                if not self._audio:
                    frame.metadata[SILENT_KEY] = True
                self._audio = False
            elif isinstance(frame, LLMFullResponseStartFrame):
                self._audio = False
        await self.push_frame(frame, direction)


class SilentLines(FrameProcessor):
    """Reports each response's playout to the voice's lines, and holds its text until its audio plays."""

    def __init__(self, lines: SpokenLines, **kwargs):
        """Report to ``lines``."""
        super().__init__(**kwargs)
        self.lines = lines
        self._response: int | None = None
        self._playing = False
        self._held: list[TTSTextFrame] = []

    async def process_frame(self, frame, direction: FrameDirection) -> None:
        """Read the playout of each response as its frames leave the output; pass every frame a caller heard."""
        await super().process_frame(frame, direction)
        frames = self._read(frame) if direction is FrameDirection.DOWNSTREAM else [frame]
        for each in frames:
            await self.push_frame(each, direction)

    def _read(self, frame) -> list[Frame]:
        """Note ``frame``'s fact; return the frames that go on now, in order."""
        if isinstance(frame, LLMFullResponseStartFrame) and RESPONSE_KEY in frame.metadata:
            self._response, self._playing, self._held = frame.metadata[RESPONSE_KEY], False, []
            self.lines.reached(self._response)
        elif isinstance(frame, OutputAudioRawFrame) and self._response is not None and frame.audio:
            if not self._playing:
                self._playing = True
                self.lines.started(self._response)
                held, self._held = self._held, []
                return [frame, *held]
        elif isinstance(frame, TTSTextFrame) and SILENT_KEY in frame.metadata:
            if self._response is not None:
                self.lines.dropped(self._response)
            return []
        elif isinstance(frame, TTSTextFrame) and self._response is not None and not self._playing:
            # The output sends audio in whole chunks: a short first sentence's text can come out before its audio.
            self._held.append(frame)
            return []
        elif isinstance(frame, LLMFullResponseEndFrame) and RESPONSE_KEY in frame.metadata:
            self.lines.ended(frame.metadata[RESPONSE_KEY])
            self._response, self._playing, self._held = None, False, []
        elif isinstance(frame, InterruptionFrame):
            self.lines.cut(self._response)
            self._response, self._playing, self._held = None, False, []
        return [frame]
