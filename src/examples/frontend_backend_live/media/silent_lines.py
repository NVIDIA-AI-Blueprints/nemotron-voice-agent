# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""What each line played, read right after the audio output, before the assistant aggregator records it.

:class:`SilentLines` sits after a session's audio output: Pipecat's output transport, or the primary WebSocket's
``examples.frontend_backend_live.adapter.bridge.LiveWebSocketAudioOutput``. Both push a frame downstream only once the
audio queued before it has gone to the caller, so it sees each response of the talker as it plays, by the number the
talker gives the response's start and end frames (``examples.frontend_backend_live.talker.RESPONSE_KEY``):

* the start frame heads the playout, and the response's first audio starts it;
* each sentence's text that follows audio played; a text that arrives before any audio of its response was not heard
  (the speech service gave it none, say a rate-limited request), so it goes no further: the assistant aggregator never
  writes it into the talker's context, and no transcript shows it;
* the end frame ends the response: heard when it played, silent when it did not, and a silent line is said once more;
* an interruption cuts it after what played, with the milliseconds of its audio since its last sentence and the text
  the speech service announced for that audio (each ``AggregatedTextFrame`` it will speak, ahead of the audio).

The facts go to the voice's lines (``examples.frontend_backend_live.spoken.SpokenLines``); every frame but a text
nobody heard passes on.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pipecat.frames.frames import (
    AggregatedTextFrame,
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


class SilentLines(FrameProcessor):
    """Reports each response's playout to the voice's lines, and stops the text of a sentence nobody heard."""

    def __init__(self, lines: SpokenLines, **kwargs):
        """Report to ``lines``."""
        super().__init__(**kwargs)
        self.lines = lines
        self._response: int | None = None  # the response at the head of the playout
        self._playing = False  # whether its audio has started
        self._audio_ms = 0.0  # its audio since its last sentence (since its start when none played)
        self._speaking: list[str] = []  # the text the speech service took since its last sentence

    async def process_frame(self, frame, direction: FrameDirection) -> None:
        """Read the playout of each response as its frames leave the output; pass every frame a caller heard."""
        await super().process_frame(frame, direction)
        if direction is FrameDirection.DOWNSTREAM and not self._heard(frame):
            return
        await self.push_frame(frame, direction)

    def _heard(self, frame) -> bool:
        """Note ``frame``'s fact; ``False`` for the text of a sentence nobody heard."""
        if isinstance(frame, LLMFullResponseStartFrame) and RESPONSE_KEY in frame.metadata:
            self._response, self._playing, self._audio_ms, self._speaking = frame.metadata[RESPONSE_KEY], False, 0.0, []
            self.lines.reached(self._response)
        elif isinstance(frame, OutputAudioRawFrame) and self._response is not None and frame.audio:
            if not self._playing:
                self._playing = True
                self.lines.started(self._response)
            self._audio_ms += frame.num_frames * 1000 / frame.sample_rate
        elif isinstance(frame, TTSTextFrame) and self._response is not None:
            if not self._playing:
                return False
            if frame.text.strip():
                audio_ms, self._audio_ms, self._speaking = round(self._audio_ms), 0.0, []
                self.lines.played(self._response, frame.text, audio_ms)
        elif isinstance(frame, AggregatedTextFrame) and not isinstance(frame, TTSTextFrame) and frame.will_be_spoken:
            self._speaking.append(frame.text)  # ahead of its audio
        elif isinstance(frame, LLMFullResponseEndFrame) and RESPONSE_KEY in frame.metadata:
            self.lines.ended(frame.metadata[RESPONSE_KEY])
            self._response, self._playing, self._audio_ms, self._speaking = None, False, 0.0, []
        elif isinstance(frame, InterruptionFrame):
            self.lines.cut(self._response, round(self._audio_ms), " ".join(self._speaking))
            self._response, self._playing, self._audio_ms, self._speaking = None, False, 0.0, []
        return True
