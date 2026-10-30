# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""What the caller really heard: the playout facts of each line, read after the audio output that played it.

A :class:`PlayoutTracker` sits right after a session's audio output (Pipecat's WebRTC output, or the primary
WebSocket's :class:`~examples.frontend_backend_live.adapter.bridge.LiveWebSocketAudioOutput`). Both outputs push a frame
downstream only once the audio queued before it has gone to the caller, so the tracker sees each line as it plays:

* a ``TTSStartedFrame`` opens a line, and the line's first audio frame is :meth:`Playout.started`;
* each ``TTSTextFrame`` of a line whose audio has started is :meth:`Playout.played`: the sentence's audio has gone
  out. The tracker hands it to the host's :class:`Transcript` here, timed on the host's clock, so a transcript holds
  played sentences only;
* a ``TTSStoppedFrame`` is :meth:`Playout.finished` when the line played, or :meth:`Playout.silent` when the speech
  service gave it no audio;
* an ``InterruptionFrame`` is :meth:`Playout.cut`: a barge-in or an instruction dropped what was in flight.

``played`` and ``cut`` carry the milliseconds of the line's audio that went out since the sentence before (since the
line's first audio when none played), measured from the audio frames' own length. A cut also names the text that
audio belongs to: what the speech service announced since the sentence before (each ``AggregatedTextFrame`` it will
speak, which it pushes ahead of that text's audio).

The facts go to one listener (:meth:`PlayoutTracker.listen`): the cascade, which corrects its records of each line by
them. Nothing crosses to the delegation core: GPT-Live has no playback receipt.

Every host of the cascade wires the tracker, its text-to-speech's mode and the cascade together through
:class:`SpeechPlayout`: the cascade adapter in a live session, and the repository UI bot in its own RTVI pipeline.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

from pipecat.frames.frames import (
    AggregatedTextFrame,
    InterruptionFrame,
    OutputAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
    TTSTextFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from examples.frontend_backend_live.cascade.session import CascadeSession

if TYPE_CHECKING:
    from examples.frontend_backend_live.cascade.agent import LiveAgent
    from examples.frontend_backend_live.cascade.memory import TurnBuffer
    from examples.frontend_backend_live.config_manager.schema import LiveConfig
    from examples.frontend_backend_live.media.services import SpeechMode


class Playout(Protocol):
    """Hears what each line really played, in playout order, one line at a time (the cascade implements it).

    A line is what one hand-off to speech became: ``started`` when its first audio reached the caller, ``played`` for
    each sentence whose audio has gone out, then ``finished``, ``silent`` or ``cut``. A line handed to speech but
    dropped before it reached the output is never reported: the ``cut`` that dropped it is. Plain calls that only
    record: the pipeline never waits on them.
    """

    def started(self) -> None:
        """The line in progress began to play: its first audio reached the caller."""

    def played(self, text: str, audio_ms: int = 0) -> None:
        """A sentence of the line in progress played: its audio, then its text, reached the caller.

        ``audio_ms`` of the line's audio went out since the sentence before (since its first audio, for its first).
        """

    def finished(self) -> None:
        """The line in progress played to its end."""

    def silent(self) -> None:
        """The line in progress ended with no audio: the speech service gave it none."""

    def cut(self, heard_ms: int = 0, speaking: str = "") -> None:
        """A cut dropped whatever was in flight, the line in progress included, after what it had played.

        ``heard_ms`` of the line's audio went out after its last sentence that played: 0 when no line was playing.
        ``speaking`` is the text the speech service took after that sentence, which that audio belongs to: the sentence
        in progress with per-sentence speech, the whole line with stitched speech; empty when the service names none.
        """


class _NoListener:
    """Hears nothing: the facts before a listener is attached."""

    def started(self) -> None:
        pass

    def played(self, text: str, audio_ms: int = 0) -> None:
        pass

    def finished(self) -> None:
        pass

    def silent(self) -> None:
        pass

    def cut(self, heard_ms: int = 0, speaking: str = "") -> None:
        pass


class Transcript(Protocol):
    """Where the sentences that played go, timed on the host's clock: a live client's output transcript."""

    @property
    def now_ms(self) -> int:
        """The host's clock, in milliseconds."""

    async def played(self, text: str, start_ms: int, end_ms: int) -> None:
        """A sentence played from ``start_ms`` to ``end_ms``."""


class _NoTranscript:
    """The transcript of a host whose client has one of its own: no clock, and nothing is sent."""

    now_ms = 0

    async def played(self, text: str, start_ms: int, end_ms: int) -> None:
        pass


class PlayoutTracker(FrameProcessor):
    """Reports what each line really played, and hands the sentences that played to the host's transcript."""

    def __init__(self, transcript: Transcript | None = None, **kwargs):
        """Bind the transcript the sentences that played go to; the facts go nowhere until :meth:`listen`.

        With no transcript, only the facts go out: the repository UI bot's RTVI client gets Pipecat's own transcript.
        """
        super().__init__(**kwargs)
        self.transcript = transcript if transcript is not None else _NoTranscript()
        self._listener: Playout = _NoListener()
        self._line_playing: bool | None = None  # the line in progress: whether its audio has started; None: no line
        self._sentence_start_ms: int | None = None  # the input clock when the current sentence's audio began
        self._audio_ms = 0.0  # the line's audio that went out since the sentence before (since the line began)
        self._speaking: list[str] = []  # the text the speech service took since the sentence before

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
            self._line_playing, self._sentence_start_ms, self._audio_ms = False, None, 0.0
        elif isinstance(frame, OutputAudioRawFrame) and self._line_playing is not None and frame.audio:
            if not self._line_playing:
                self._line_playing = True
                self._listener.started()
            if self._sentence_start_ms is None:
                self._sentence_start_ms = self.transcript.now_ms
            self._audio_ms += frame.num_frames * 1000 / frame.sample_rate
        elif isinstance(frame, TTSTextFrame) and frame.text.strip() and self._line_playing:
            await self._sentence_played(frame.text)
        elif isinstance(frame, AggregatedTextFrame) and not isinstance(frame, TTSTextFrame) and frame.will_be_spoken:
            self._speaking.append(frame.text)  # ahead of its audio; a line's first sentence ahead of TTSStartedFrame
        elif isinstance(frame, TTSStoppedFrame) and self._line_playing is not None:
            if self._line_playing:
                self._listener.finished()
            else:
                self._listener.silent()
            self._line_playing, self._audio_ms, self._speaking = None, 0.0, []
        elif isinstance(frame, InterruptionFrame):
            self._listener.cut(round(self._audio_ms), " ".join(self._speaking))
            self._line_playing, self._audio_ms, self._speaking = None, 0.0, []

    async def _sentence_played(self, text: str) -> None:
        """A sentence's audio has gone out: tell the listener, and hand it to the transcript."""
        end_ms = self.transcript.now_ms
        start_ms = end_ms if self._sentence_start_ms is None else self._sentence_start_ms
        self._sentence_start_ms = None
        audio_ms, self._audio_ms, self._speaking = round(self._audio_ms), 0.0, []
        self._listener.played(text, audio_ms)
        await self.transcript.played(text, start_ms, end_ms)


class SpeechPlayout:
    """How a host keeps its cascade's records to what the caller heard, by the mode its text-to-speech runs in.

    The host places :meth:`speech_out` where its text-to-speech goes and a :class:`PlayoutTracker` right after its audio
    output, then builds its cascade with :meth:`cascade`, which hears that tracker. A cut line then keeps only what
    played, sentence by sentence with per-sentence speech, and a line nobody heard leaves the records.
    """

    def __init__(self, speech: SpeechMode):
        """Take how the host's text-to-speech takes a line (``services.SPEECH_MODES``)."""
        self.speech = speech

    def speech_out(self, tts: FrameProcessor) -> list[FrameProcessor]:
        """The text-to-speech, then what keeps each line's text behind its audio on the way to the tracker."""
        return [tts, *self.speech.text_hold()]

    def cascade(
        self, tracker: PlayoutTracker, agent: LiveAgent, config: LiveConfig, conversation: TurnBuffer, **host: Any
    ) -> CascadeSession:
        """The call's cascade, which corrects its records by what ``tracker`` reports.

        ``host`` holds the cascade's other keyword arguments: its delegation part, ``say``, and the host's own.
        """
        cascade = CascadeSession(
            agent, config, conversation, reports_playout=True, reports_sentences=self.speech.reports_sentences, **host
        )
        tracker.listen(cascade.playout)
        return cascade
