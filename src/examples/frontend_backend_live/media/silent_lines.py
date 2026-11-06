# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""What each line's audio did, read right after the audio output: what the assistant aggregator after it cannot tell.

:class:`SilentLines` sits after a session's audio output: Pipecat's output transport, on WebRTC or a primary WebSocket.
It pushes a frame downstream only once the audio queued before it has gone to the caller, so this sees each response
of the talker as it plays, by the number the talker gives the response's start and end frames
(``examples.frontend_backend_live.talker.RESPONSE_KEY``):

* the start frame heads the playout, and the response's first audio starts it;
* a text that arrives before any audio of its response is held, and goes on in order right after that first audio:
  the output sends audio in whole chunks, so a short first sentence's text can come out just before its audio;
* a sentence's text marked as having had no audio (:data:`SILENT_KEY`) goes no further, and its line is heard as the
  assistant aggregator records it;
* the end frame ends the response: heard when it played, silent when it did not, and a silent line is said once more.
  A text still held then was not heard (the speech service gave it no audio, say a rate-limited request), so it goes
  no further;
* an interruption cuts it, and drops what is held, which nobody heard either.

A text that goes no further never reaches the assistant aggregator, which writes the talker's context, nor a live
session's transcript after this edge; the UI bot's RTVI observer reads a line's text before this edge, at the talker
and at the output, so its transcript can show a text nobody heard.

Which sentence got no audio shows only before the output: after it, a sentence with none and a short one whose audio
waits in the output's open chunk come out alike. So :class:`SilentSentences` reads it right after a per-sentence
text-to-speech, where each sentence's text follows its own audio, and marks the text of a sentence that got none. A
stitched line's text comes unmarked at its end, after all its audio.

The facts go to the voice's lines (``examples.frontend_backend_live.spoken.SpokenLines``); the words a cut line played
are the ones the assistant aggregator records and reports as the interruption ends its turn. Every frame but a text
nobody heard passes on: with per-sentence speech, a sentence's text goes on only if audio came for it, and a stitched
line's text goes on whole once its audio played. A line that ends after some of its audio played is heard: it is never
said once more.
"""

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

# The metadata key under which SilentSentences marks the text of a sentence that got no audio, so that SilentLines,
# after the output, lets it go no further.
SILENT_KEY = "silent_sentence"


class SilentSentences(FrameProcessor):
    """Marks the text of each sentence that got no audio, right after a per-sentence text-to-speech.

    There each sentence's text follows its own audio, so a text with no audio since the text before it, or since its
    line's start, got none: say the speech service rate-limited its request. Every frame passes on.
    """

    def __init__(self, **kwargs):
        """Start with no audio yet."""
        super().__init__(**kwargs)
        self._audio = False  # whether audio came since the last text, or since the line's start

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
    """Reports each response's audio to the voice's lines, and holds its text until its audio plays."""

    def __init__(self, lines: SpokenLines, **kwargs):
        """Report to ``lines``."""
        super().__init__(**kwargs)
        self.lines = lines
        self._response: int | None = None  # the response at the head of the playout
        self._playing = False  # whether its audio has started
        self._held: list[TTSTextFrame] = []  # its text that came before its first audio

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
            self._held.append(frame)
            return []
        elif isinstance(frame, LLMFullResponseEndFrame) and RESPONSE_KEY in frame.metadata:
            self.lines.ended(frame.metadata[RESPONSE_KEY])
            self._response, self._playing, self._held = None, False, []
        elif isinstance(frame, InterruptionFrame):
            self.lines.cut(self._response)
            self._response, self._playing, self._held = None, False, []
        return [frame]
