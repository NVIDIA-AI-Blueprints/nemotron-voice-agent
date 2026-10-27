# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""What the caller really heard: the playout facts of each line, read after the audio output that played it.

A :class:`PlayoutTracker` sits right after a session's audio output (Pipecat's WebRTC output, or the primary
WebSocket's :class:`~examples.frontend_backend_live.adapter.bridge.LiveWebSocketAudioOutput`). Both outputs push a frame
downstream only once the audio queued before it has gone to the caller, so the tracker sees each line as it plays:

* a ``TTSStartedFrame`` opens a line, and the line's first audio frame is :meth:`Playout.started`;
* each ``TTSTextFrame`` of a line whose audio has started is :meth:`Playout.played`: the sentence's audio has gone
  out. The tracker sends the client's ``session.output_transcript.delta`` for it here, timed on the session's input
  clock, so the transcript holds played sentences only;
* a ``TTSStoppedFrame`` is :meth:`Playout.finished` when the line played, or :meth:`Playout.silent` when the speech
  service gave it no audio;
* an ``InterruptionFrame`` is :meth:`Playout.cut`: a barge-in or an instruction dropped what was in flight.

The facts go to one listener (:meth:`PlayoutTracker.listen`): the cascade, which corrects its records of each line by
them. Nothing crosses to the delegation core: GPT-Live has no playback receipt.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from pipecat.frames.frames import (
    InterruptionFrame,
    OutputAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
    TTSTextFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

if TYPE_CHECKING:
    from live.ports import SessionPort


class Playout(Protocol):
    """Hears what each line really played, in playout order, one line at a time (the cascade implements it).

    A line is what one hand-off to speech became: ``started`` when its first audio reached the caller, ``played`` for
    each sentence whose audio has gone out, then ``finished``, ``silent`` or ``cut``. A line handed to speech but
    dropped before it reached the output is never reported: the ``cut`` that dropped it is. Plain calls that only
    record: the pipeline never waits on them.
    """

    def started(self) -> None:
        """The line in progress began to play: its first audio reached the caller."""

    def played(self, text: str) -> None:
        """A sentence of the line in progress played: its audio, then its text, reached the caller."""

    def finished(self) -> None:
        """The line in progress played to its end."""

    def silent(self) -> None:
        """The line in progress ended with no audio: the speech service gave it none."""

    def cut(self) -> None:
        """A cut dropped whatever was in flight, the line in progress included, after what it had played."""


class _NoListener:
    """Hears nothing: the facts before a listener is attached."""

    def started(self) -> None:
        pass

    def played(self, text: str) -> None:
        pass

    def finished(self) -> None:
        pass

    def silent(self) -> None:
        pass

    def cut(self) -> None:
        pass


class PlayoutTracker(FrameProcessor):
    """Reports what each line really played, and sends the client the transcript of the sentences that played."""

    def __init__(self, session: SessionPort, **kwargs):
        """Bind the session whose client gets the transcript; the facts go nowhere until :meth:`listen`."""
        super().__init__(**kwargs)
        self.session = session
        self._listener: Playout = _NoListener()
        self._line_playing: bool | None = None  # the line in progress: whether its audio has started; None: no line
        self._sentence_start_ms: int | None = None  # the input clock when the current sentence's audio began

    def listen(self, playout: Playout) -> None:
        """Hand every later fact to ``playout``."""
        self._listener = playout

    async def process_frame(self, frame, direction: FrameDirection) -> None:
        """Read the facts of each line as its frames leave the output; pass every frame on."""
        await super().process_frame(frame, direction)
        if direction is FrameDirection.DOWNSTREAM:
            await self._observe(frame)
        await self.push_frame(frame, direction)

    async def _observe(self, frame) -> None:
        if isinstance(frame, TTSStartedFrame):
            self._line_playing, self._sentence_start_ms = False, None
        elif isinstance(frame, OutputAudioRawFrame) and self._line_playing is not None and frame.audio:
            if not self._line_playing:
                self._line_playing = True
                self._listener.started()
            if self._sentence_start_ms is None:
                self._sentence_start_ms = self.session.input_ms
        elif isinstance(frame, TTSTextFrame) and frame.text.strip() and self._line_playing:
            await self._sentence_played(frame.text)
        elif isinstance(frame, TTSStoppedFrame) and self._line_playing is not None:
            if self._line_playing:
                self._listener.finished()
            else:
                self._listener.silent()
            self._line_playing = None
        elif isinstance(frame, InterruptionFrame):
            self._listener.cut()
            self._line_playing = None

    async def _sentence_played(self, text: str) -> None:
        """A sentence's audio has gone out: tell the listener, and send the client its transcript."""
        end_ms = self.session.input_ms
        start_ms = end_ms if self._sentence_start_ms is None else self._sentence_start_ms
        self._sentence_start_ms = None
        self._listener.played(text)
        await self.session.emit(
            "session.output_transcript.delta",
            delta=text if text.endswith(" ") else text + " ",
            start_ms=start_ms,
            end_ms=end_ms,
        )
