# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D100, D101, D102, D103, D107

"""The turn bridge tells the turn manager about each user turn once, and in order.

The turn manager holds every reply while an utterance is open (one ``on_speech_started``
not yet matched by ``on_user_input``). A barge-in's acknowledgement that arrived after the
turn had already stopped left an utterance open for the rest of the session: the agent
produced replies and never spoke them.
"""

from __future__ import annotations

import asyncio
import unittest
from typing import Any
from unittest import mock

from _pipecat_fakes import stub_config
from pipecat.frames.frames import (
    InterruptionFrame,
    TranscriptionFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from examples.frontend_delegation_hermes.adapters.pipecat import turn_bridge
from examples.frontend_delegation_hermes.adapters.pipecat.speech_output import OutputTagger
from examples.frontend_delegation_hermes.adapters.pipecat.timing import InputCursor
from examples.frontend_delegation_hermes.core.clock import SystemClock
from examples.frontend_delegation_hermes.core.session_log import SessionLog
from examples.frontend_delegation_hermes.core.wire.response_writer import ConversationOrder

AFTER_ACK_WINDOW_S = turn_bridge.INTERRUPTION_ACK_TIMEOUT_S + 0.2


class _NullLog:
    def write(self, *_: Any, **__: Any) -> None:
        pass


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


async def _noop(*_: Any, **__: Any) -> None:
    pass


class TurnBridgeOrderTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        # Only the bridge's own handling is under test, not Pipecat's frame bookkeeping.
        patcher = mock.patch.object(FrameProcessor, "process_frame", _noop)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.tagger = OutputTagger()
        self.bridge = turn_bridge.FdhTurnBridge(
            config=stub_config().voice,
            emit_urgent=lambda _event: None,
            conversation=ConversationOrder(),
            cursor=InputCursor(),
            tagger=self.tagger,
            log=SessionLog(_NullLog(), "sess_test", lambda: 0.0),
            clock=SystemClock(),
            session_id="sess_test",
        )
        self.bridge.push_frame = _noop  # type: ignore[method-assign]
        self.turns = _Turns()
        self.bridge.bind(self.turns)  # type: ignore[arg-type]
        self.addAsyncCleanup(self.bridge.aclose)

    async def _feed(self, *frames: Any) -> None:
        for frame in frames:
            await self.bridge.process_frame(frame, FrameDirection.DOWNSTREAM)

    async def test_turn_that_stops_before_the_ack_leaves_no_utterance_open(self) -> None:
        await self._feed(
            UserStartedSpeakingFrame(),
            InterruptionFrame(),
            TranscriptionFrame(text="um", user_id="", timestamp=""),
            UserStoppedSpeakingFrame(),
        )
        await asyncio.sleep(AFTER_ACK_WINDOW_S)
        self.assertEqual(self.turns.calls, ["started", "input:um"])
        self.assertEqual(self.turns.awaiting, 0)

    async def test_acknowledged_interruption_tells_the_turn_manager_once(self) -> None:
        await self._feed(UserStartedSpeakingFrame(), InterruptionFrame())
        self.tagger.interrupted.set()
        await asyncio.sleep(0.05)
        self.assertEqual(self.turns.calls, ["started"])
        await self._feed(TranscriptionFrame(text="hello", user_id="", timestamp=""), UserStoppedSpeakingFrame())
        await asyncio.sleep(AFTER_ACK_WINDOW_S)
        self.assertEqual(self.turns.calls, ["started", "input:hello"])
        self.assertEqual(self.turns.awaiting, 0)

    async def test_cleared_turn_drops_its_pending_ack(self) -> None:
        await self._feed(UserStartedSpeakingFrame(), InterruptionFrame())
        self.assertIsNotNone(self.bridge.clear())
        await asyncio.sleep(AFTER_ACK_WINDOW_S)
        self.assertEqual(self.turns.calls, [])
        self.assertEqual(self.turns.awaiting, 0)


if __name__ == "__main__":
    unittest.main()
