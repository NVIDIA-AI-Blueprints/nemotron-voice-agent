# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D100, D101, D102, D103

"""The reply-audio pre-roll in front of the live cascade's audio output."""

import unittest

from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    InterruptionFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
    TTSTextFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.tests.utils import SleepFrame, run_test

from examples.frontend_backend_live.adapter.preroll import SpeechPreroll
from examples.frontend_backend_live.config_manager.loader import ConfigError
from examples.frontend_backend_live.config_manager.schema import LiveConfig

RATE = 24000


def audio(ms: int) -> TTSAudioRawFrame:
    return TTSAudioRawFrame(audio=b"\0\0" * (RATE * ms // 1000), sample_rate=RATE, num_channels=1)


def word(text: str) -> TTSTextFrame:
    return TTSTextFrame(text=text, aggregated_by="word")


class SpeechPrerollTests(unittest.IsolatedAsyncioTestCase):
    def preroll(self, ms: int = 300) -> tuple[SpeechPreroll, list]:
        processor = SpeechPreroll(ms)
        sent = []

        async def record(frame, direction=FrameDirection.DOWNSTREAM):
            sent.append(frame)

        processor.push_frame = record
        return processor, sent

    async def feed(self, processor, *frames, direction=FrameDirection.DOWNSTREAM):
        for frame in frames:
            await processor.process_frame(frame, direction)

    async def test_a_reply_waits_until_enough_audio_is_ready(self):
        processor, sent = self.preroll(300)
        first = [TTSStartedFrame(), audio(100), word("Hello"), audio(100)]
        await self.feed(processor, *first)
        self.assertEqual(sent, [])
        await self.feed(processor, audio(100))
        self.assertEqual(sent, [*first, sent[-1]])
        self.assertIsInstance(sent[-1], TTSAudioRawFrame)
        # Once playing, the rest of the reply passes straight through.
        later = audio(40)
        await self.feed(processor, later)
        self.assertIs(sent[-1], later)

    async def test_a_reply_shorter_than_the_preroll_plays_when_it_ends(self):
        processor, sent = self.preroll(300)
        frames = [TTSStartedFrame(), audio(120), word("Okay."), TTSStoppedFrame()]
        await self.feed(processor, *frames)
        self.assertEqual(sent, frames)

    async def test_an_interruption_drops_the_held_audio(self):
        dropped, kept = audio(100), audio(60)
        down, _ = await run_test(
            SpeechPreroll(300),
            # System frames overtake queued ones, so let the held audio arrive first. The next reply is held afresh;
            # the end of the pipeline releases it.
            frames_to_send=[TTSStartedFrame(), dropped, SleepFrame(0.05), InterruptionFrame(), TTSStartedFrame(), kept],
            expected_down_frames=[InterruptionFrame, TTSStartedFrame, TTSAudioRawFrame],
        )
        self.assertIs(down[-1], kept)

    async def test_system_and_upstream_frames_are_never_held(self):
        processor, sent = self.preroll(300)
        await self.feed(processor, TTSStartedFrame(), audio(100))
        system = BotStartedSpeakingFrame()
        await self.feed(processor, system)
        upstream = TTSTextFrame(text="up", aggregated_by="word")
        await self.feed(processor, upstream, direction=FrameDirection.UPSTREAM)
        self.assertEqual(sent, [system, upstream])

    async def test_zero_passes_everything_through(self):
        processor, sent = self.preroll(0)
        frames = [TTSStartedFrame(), audio(20), TTSStoppedFrame()]
        await self.feed(processor, *frames)
        self.assertEqual(sent, frames)

    async def test_frames_keep_their_order_in_a_pipeline(self):
        frames = [TTSStartedFrame(), audio(100), word("One"), audio(250), word("two"), audio(50), TTSStoppedFrame()]
        down, _ = await run_test(
            SpeechPreroll(300),
            frames_to_send=frames,
            expected_down_frames=[type(f) for f in frames],
        )
        self.assertEqual([getattr(f, "text", None) for f in down if isinstance(f, TTSTextFrame)], ["One", "two"])


class PrerollConfigTests(unittest.TestCase):
    BLOCK = {"provider": "p", "slot": "llm"}  # the frontend.cascade block

    def test_defaults_to_300_ms_and_accepts_zero(self):
        self.assertEqual(LiveConfig.parse(self.BLOCK).tts_preroll_ms, 300)
        self.assertEqual(LiveConfig.parse({**self.BLOCK, "tts_preroll_ms": 0}).tts_preroll_ms, 0)

    def test_rejects_values_outside_the_range(self):
        for bad in (-1, 2001, "300", True, 1.5):
            with self.subTest(bad=bad), self.assertRaisesRegex(ConfigError, "tts_preroll_ms"):
                LiveConfig.parse({**self.BLOCK, "tts_preroll_ms": bad})


if __name__ == "__main__":
    unittest.main()
