# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

# ruff: noqa: D100, D101, D102

import asyncio
import unittest

from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    LLMConfigureOutputFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
)
from pipecat.processors.frame_processor import FrameDirection

from examples.omni_assistant_subagents.media_dispatch_processor import PostAckMediaDispatchProcessor


class _RecordingHandler:
    def __init__(self) -> None:
        self.events: list[str] = []

    async def start_pending_media_analysis(self) -> None:
        self.events.append("start_pending_media_analysis")

    async def start_pending_thinking(self) -> None:
        self.events.append("start_pending_thinking")

    async def on_user_voice_turn_started(self) -> None:
        self.events.append("on_user_voice_turn_started")

    async def on_user_voice_turn_stopped(self) -> None:
        self.events.append("on_user_voice_turn_stopped")

    async def on_user_interrupted_assistant(self) -> None:
        self.events.append("on_user_interrupted_assistant")

    async def on_assistant_speaking_started(self) -> None:
        self.events.append("on_assistant_speaking_started")

    async def on_assistant_speaking_stopped(self) -> None:
        self.events.append("on_assistant_speaking_stopped")


async def _process(processor: PostAckMediaDispatchProcessor, frame, direction: FrameDirection) -> None:
    async def _push_frame(_frame, _direction=FrameDirection.DOWNSTREAM) -> None:
        return None

    processor.push_frame = _push_frame
    await processor.process_frame(frame, direction)


class OmniMediaDispatchTests(unittest.TestCase):
    def test_text_mode_eval_dispatches_after_llm_response_end_when_tts_is_skipped(self) -> None:
        async def run() -> None:
            handler = _RecordingHandler()
            processor = PostAckMediaDispatchProcessor(handler=handler)

            await _process(processor, LLMConfigureOutputFrame(skip_tts=True), FrameDirection.DOWNSTREAM)
            await _process(processor, LLMFullResponseStartFrame(), FrameDirection.DOWNSTREAM)
            await _process(processor, LLMFullResponseEndFrame(), FrameDirection.DOWNSTREAM)

            self.assertEqual(handler.events, ["start_pending_media_analysis", "start_pending_thinking"])

        asyncio.run(run())

    def test_audio_mode_waits_for_bot_stopped_speaking(self) -> None:
        async def run() -> None:
            handler = _RecordingHandler()
            processor = PostAckMediaDispatchProcessor(handler=handler)

            await _process(processor, LLMConfigureOutputFrame(skip_tts=False), FrameDirection.DOWNSTREAM)
            await _process(processor, LLMFullResponseStartFrame(), FrameDirection.DOWNSTREAM)
            await _process(processor, LLMFullResponseEndFrame(), FrameDirection.DOWNSTREAM)
            await _process(processor, BotStartedSpeakingFrame(), FrameDirection.UPSTREAM)
            await _process(processor, BotStoppedSpeakingFrame(), FrameDirection.UPSTREAM)

            self.assertEqual(
                handler.events,
                [
                    "on_assistant_speaking_started",
                    "on_assistant_speaking_stopped",
                    "start_pending_media_analysis",
                    "start_pending_thinking",
                ],
            )

        asyncio.run(run())

    def test_skip_tts_persists_across_turns(self) -> None:
        async def run() -> None:
            handler = _RecordingHandler()
            processor = PostAckMediaDispatchProcessor(handler=handler)

            await _process(processor, LLMConfigureOutputFrame(skip_tts=True), FrameDirection.DOWNSTREAM)
            await _process(processor, LLMFullResponseStartFrame(), FrameDirection.DOWNSTREAM)
            await _process(processor, LLMFullResponseEndFrame(), FrameDirection.DOWNSTREAM)
            await _process(processor, LLMFullResponseStartFrame(), FrameDirection.DOWNSTREAM)
            await _process(processor, LLMFullResponseEndFrame(), FrameDirection.DOWNSTREAM)

            self.assertEqual(
                handler.events,
                [
                    "start_pending_media_analysis",
                    "start_pending_thinking",
                    "start_pending_media_analysis",
                    "start_pending_thinking",
                ],
            )

        asyncio.run(run())

    def test_text_mode_dispatches_once_per_response_boundary(self) -> None:
        async def run() -> None:
            handler = _RecordingHandler()
            processor = PostAckMediaDispatchProcessor(handler=handler)

            await _process(processor, LLMConfigureOutputFrame(skip_tts=True), FrameDirection.DOWNSTREAM)
            await _process(processor, LLMFullResponseStartFrame(), FrameDirection.DOWNSTREAM)
            await _process(processor, LLMFullResponseEndFrame(), FrameDirection.DOWNSTREAM)
            await _process(processor, LLMFullResponseEndFrame(), FrameDirection.DOWNSTREAM)

            self.assertEqual(handler.events, ["start_pending_media_analysis", "start_pending_thinking"])

        asyncio.run(run())


if __name__ == "__main__":
    unittest.main()
