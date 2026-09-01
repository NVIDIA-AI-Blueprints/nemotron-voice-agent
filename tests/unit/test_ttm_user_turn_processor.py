# SPDX-FileCopyrightText: Copyright (c) 2024–2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

# ruff: noqa: D100, D101, D102

import json
import unittest

import numpy as np
from pipecat.frames.frames import (
    InputAudioRawFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.tests.utils import SleepFrame, run_test
from websockets.asyncio.server import serve

from examples.generic.ttm_user_turn_processor import TTMUserTurnProcessor


class TTMUserTurnProcessorTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.sou_only = False

        async def handler(websocket):
            sequence = 0
            async for _ in websocket:
                if sequence == 0:
                    await websocket.send(json.dumps(self._event("SOU")))
                elif sequence == 1 and not self.sou_only:
                    await websocket.send(json.dumps(self._event("EOU")))
                sequence += 1

        self.server = await serve(handler, "127.0.0.1", 0)
        port = self.server.sockets[0].getsockname()[1]
        self.url = f"ws://127.0.0.1:{port}"

    async def asyncTearDown(self):
        self.server.close()
        await self.server.wait_closed()

    @staticmethod
    def _event(name: str) -> dict:
        return {
            "type": "turn.event",
            "event": name,
            "segment_id": "turn-1",
            "audio_offset_ms": 160,
            "confidence": 0.9,
        }

    @staticmethod
    def _audio() -> InputAudioRawFrame:
        return InputAudioRawFrame(
            audio=np.zeros(320, dtype="<i2").tobytes(),
            sample_rate=16_000,
            num_channels=1,
        )

    async def test_native_ttm_events_own_turn_boundaries(self):
        processor = TTMUserTurnProcessor(url=self.url, enable_interruptions=False)

        down, _ = await run_test(
            processor,
            frames_to_send=[self._audio(), self._audio(), SleepFrame(0.05)],
        )

        self.assertEqual(
            sum(isinstance(frame, UserStartedSpeakingFrame) for frame in down),
            1,
        )
        self.assertEqual(
            sum(isinstance(frame, UserStoppedSpeakingFrame) for frame in down),
            1,
        )
        self.assertFalse(processor.speaking)

    async def test_silero_fallback_closes_stuck_sou_once(self):
        self.sou_only = True
        processor = TTMUserTurnProcessor(
            url=self.url,
            enable_interruptions=False,
            silence_fallback_secs=0.05,
        )

        down, _ = await run_test(
            processor,
            frames_to_send=[self._audio(), SleepFrame(0.1)],
        )

        self.assertEqual(
            sum(isinstance(frame, UserStartedSpeakingFrame) for frame in down),
            1,
        )
        self.assertEqual(
            sum(isinstance(frame, UserStoppedSpeakingFrame) for frame in down),
            1,
        )
        self.assertFalse(processor.speaking)


if __name__ == "__main__":
    unittest.main()
