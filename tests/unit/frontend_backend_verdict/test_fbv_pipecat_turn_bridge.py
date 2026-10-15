# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D100, D101, D102, D103, D107

"""The turn bridge tells the turn manager about each user turn once, and in v1's order (S2–S5, B9).

Ported from FDH's ``test_turn_bridge`` (``a0ef50c``, fix ``1a9f37b``): the turn manager holds
every reply while an utterance is open (an ``on_speech_started`` not yet matched by
``on_user_input``), so a turn the bridge never closes silences the agent for the rest of the
session. Pipecat's ``UserStoppedSpeakingFrame`` is a system frame and can overtake the turn's
final ``TranscriptionFrame``; the bridge finalizes the turn on a marker queued behind it.
"""

from __future__ import annotations

import asyncio
import unittest
from typing import Any
from unittest import mock

from _fbv_core_fakes import base_config
from pipecat.frames.frames import (
    ErrorFrame,
    InterruptionFrame,
    TranscriptionFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from examples.frontend_backend_verdict.adapters.pipecat import turn_bridge
from examples.frontend_backend_verdict.adapters.pipecat.frames import UserTurnFinalFrame
from examples.frontend_backend_verdict.adapters.pipecat.session_log import SessionLog
from examples.frontend_backend_verdict.adapters.pipecat.speech_output import OutputTagger
from examples.frontend_backend_verdict.adapters.pipecat.timing import InputCursor
from examples.frontend_backend_verdict.core.clock import SystemClock
from examples.frontend_backend_verdict.core.engine.event_log import EventLog
from examples.frontend_backend_verdict.core.engine.response_writer import ConversationOrder


class _Turns:
    """Counts open utterances the way the turn manager does."""

    def __init__(self) -> None:
        self.calls: list[str] = []
        self.awaiting = 0

    async def on_speech_started(self) -> None:
        self.calls.append("started")
        self.awaiting += 1

    def on_user_input(self, user_input: Any, *, from_audio: bool) -> None:
        self.calls.append(f"input:{user_input.text}")
        if from_audio:
            self.awaiting = max(0, self.awaiting - 1)

    def on_utterance_dropped(self, item_id: str, *, reason: str) -> None:
        self.calls.append(f"dropped:{reason}")
        self.awaiting = max(0, self.awaiting - 1)


async def _noop(*_: Any, **__: Any) -> None:
    pass


def _text(text: str) -> TranscriptionFrame:
    return TranscriptionFrame(text=text, user_id="", timestamp="", finalized=True)


class TurnBridgeOrderTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        # Only the bridge's own handling is under test, not Pipecat's frame bookkeeping.
        patcher = mock.patch.object(FrameProcessor, "process_frame", _noop)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.events: list[dict[str, Any]] = []
        self.tagger = OutputTagger()
        self.tts = FrameProcessor()
        self.bridge = turn_bridge.FbvTurnBridge(
            config=base_config(),
            emit_urgent=self.events.append,
            conversation=ConversationOrder(),
            cursor=InputCursor(),
            tagger=self.tagger,
            log=SessionLog(EventLog(), "sess_test", lambda: 0.0),
            clock=SystemClock(),
            session_id="sess_test",
        )
        self.bridge.push_frame = _noop  # type: ignore[method-assign]
        self.queued: list[Any] = []  # the bridge's own data queue (the turn-final markers)

        async def queue_frame(frame: Any, direction: FrameDirection = FrameDirection.DOWNSTREAM) -> None:
            self.queued.append(frame)

        self.bridge.queue_frame = queue_frame  # type: ignore[method-assign]
        self.turns = _Turns()
        self.bridge.bind(self.turns, tts=self.tts)  # type: ignore[arg-type]
        self.addAsyncCleanup(self.bridge.aclose)

    async def _feed(self, *frames: Any) -> None:
        for frame in frames:
            await self.bridge.process_frame(frame, FrameDirection.DOWNSTREAM)

    async def _drain(self) -> None:
        """Process the queued markers, in order, after what was fed before them."""
        while self.queued:
            await self._feed(self.queued.pop(0))

    def types(self) -> list[str]:
        return [event["type"] for event in self.events]

    async def test_turn_is_started_at_once_and_closed_once(self) -> None:  # B9, S2, S4
        await self._feed(UserStartedSpeakingFrame(), InterruptionFrame(), _text("um"), UserStoppedSpeakingFrame())
        await self._drain()
        self.assertEqual(self.turns.calls, ["started", "input:um"])
        self.assertEqual(self.turns.awaiting, 0)
        self.assertEqual(
            self.types(),
            [
                "input_audio_buffer.speech_started",
                "input_audio_buffer.speech_stopped",
                "input_audio_buffer.committed",
                "conversation.item.added",
                "conversation.item.done",
                "conversation.item.input_audio_transcription.completed",
            ],
        )

    async def test_stop_that_overtakes_the_transcript_still_carries_it(self) -> None:  # S4
        await self._feed(UserStartedSpeakingFrame(), UserStoppedSpeakingFrame())
        self.assertIn("input_audio_buffer.committed", self.types())  # committed at the stop, as v1
        self.assertEqual(self.turns.calls, [])
        await self._feed(_text("hello"))  # the data frame that was queued before the marker
        await self._drain()
        self.assertEqual(self.turns.calls, ["started", "input:hello"])
        self.assertEqual(self.events[-1]["transcript"], "hello")

    async def test_marker_without_a_stopped_turn_is_ignored(self) -> None:
        await self._feed(UserTurnFinalFrame(item_id="item_x"))
        self.assertEqual(self.turns.calls, [])

    async def test_asr_error_drops_the_turn(self) -> None:  # S5
        await self._feed(UserStartedSpeakingFrame())
        self.bridge.on_asr_error("ASR endpoint unavailable")
        await self._feed(UserStoppedSpeakingFrame())
        await self._drain()
        self.assertEqual(self.turns.calls, ["started", "dropped:asr_failed"])
        self.assertEqual(self.events[-1]["type"], "conversation.item.input_audio_transcription.failed")

    async def test_cleared_turn_is_not_finalized(self) -> None:  # S6
        await self._feed(UserStartedSpeakingFrame())
        await asyncio.sleep(0)  # the turn manager has been told
        self.assertIsNotNone(self.bridge.clear())  # the session reports it dropped (audio_cleared)
        await self._feed(_text("ignored"), UserStoppedSpeakingFrame())
        await self._drain()
        self.assertEqual(self.turns.calls, ["started"])
        self.assertNotIn("input_audio_buffer.committed", self.types())

    async def test_clear_before_the_start_was_delivered_cancels_it(self) -> None:  # S6, fix 1a9f37b
        await self._feed(UserStartedSpeakingFrame())
        self.assertIsNone(self.bridge.clear())  # nothing to report: the turn manager never counted it
        await asyncio.sleep(0.01)
        self.assertEqual(self.turns.calls, [])
        self.assertEqual(self.turns.awaiting, 0)

    async def test_tts_error_fails_the_item_being_spoken(self) -> None:
        failed: list[str] = []
        self.tagger.fail_current = failed.append  # type: ignore[method-assign]
        error = ErrorFrame(error="TTS endpoint unavailable")
        error.processor = self.tts
        await self.bridge.process_frame(error, FrameDirection.UPSTREAM)
        other = ErrorFrame(error="something else")
        other.processor = FrameProcessor()
        await self.bridge.process_frame(other, FrameDirection.UPSTREAM)
        self.assertEqual(failed, ["TTS endpoint unavailable"])


if __name__ == "__main__":
    unittest.main()
