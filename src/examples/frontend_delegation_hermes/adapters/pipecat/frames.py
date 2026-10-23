# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Frames the Pipecat head adds: Realtime events in three classes, and the item markers.

Event classes (every server event belongs to exactly one):

* :class:`StreamEventFrame`: transcript and audio deltas, queued behind their audio and
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
    DataFrame,
    OutputTransportMessageFrame,
    OutputTransportMessageUrgentFrame,
    UninterruptibleFrame,
)

if TYPE_CHECKING:
    from examples.frontend_delegation_hermes.adapters.pipecat.speech_output import SpeechJob


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


def event_type(event: dict[str, Any]) -> str:
    """The Realtime ``type`` of a server event."""
    return str(event.get("type", ""))
