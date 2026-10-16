# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Frames the Pipecat adapter adds: Realtime events in three classes, and the item markers.

Ported from FDH ``adapters/pipecat/frames.py`` (``a0ef50c``). Every server event belongs to
exactly one class:

* :class:`StreamEventFrame`: transcript and text deltas, queued behind their audio and
  dropped by an interruption (that audio is never played).
* :class:`ResponseEventFrame`: every other event of a response (created, items, done),
  queued in order behind the audio already sent but kept on an interruption
  (``UninterruptibleFrame``), so a client always sees each response finish and never
  loses a function call.
* :class:`UrgentEventFrame`: session, input-buffer, user-item and error events, sent at
  once and never behind agent audio.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from pipecat.frames.frames import (
    ControlFrame,
    DataFrame,
    OutputTransportMessageFrame,
    OutputTransportMessageUrgentFrame,
    UninterruptibleFrame,
)

if TYPE_CHECKING:
    from examples.frontend_backend_agent.adapters.pipecat.speech_output import SpeechJob


@dataclass
class StreamEventFrame(OutputTransportMessageFrame):
    """A transcript or text delta: ordered with the audio, dropped on interruption."""


@dataclass
class ResponseEventFrame(OutputTransportMessageFrame, UninterruptibleFrame):
    """A response lifecycle or function-call event: ordered with the audio, kept on interruption."""


@dataclass
class UrgentEventFrame(OutputTransportMessageUrgentFrame):
    """A session, input-buffer, user-item or error event: sent immediately."""


@dataclass
class ItemAudioFrame(OutputTransportMessageFrame):
    """Marks where one assistant item's audio begins (``begin``) or ends in the output stream.

    It travels the audio queue like any message and the serializer consumes it (nothing is
    sent): ``response.output_audio.delta`` events are stamped with the item named by the last
    ``begin`` marker, because the transport re-chunks audio frames and drops any tag on them.
    """

    response_id: str = ""
    item_id: str = ""
    output_index: int = 0
    begin: bool = True


@dataclass
class SpeechStartFrame(DataFrame):
    """Pushed before one item's sentences; tells the output tagger which job the audio belongs to."""

    job: SpeechJob


@dataclass
class EndOfSpeechFrame(DataFrame):
    """Pushed after one item's last sentence; the TTS service orders it after that item's audio."""

    job: SpeechJob


@dataclass
class UserTurnFinalFrame(ControlFrame, UninterruptibleFrame):
    """Queued by the turn bridge on itself when a user turn stops; consumed there.

    ``UserStoppedSpeakingFrame`` is a system frame and overtakes the turn's final
    ``TranscriptionFrame`` (a data frame) in the bridge's queue. This marker joins the data
    queue behind the transcripts already pushed, so the bridge finalizes the turn after them.
    """

    item_id: str = ""


def event_type(event: dict[str, Any]) -> str:
    """The Realtime ``type`` of a server event."""
    return str(event.get("type", ""))
