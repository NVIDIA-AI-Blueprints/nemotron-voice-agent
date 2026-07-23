# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

# ruff: noqa: D100, D101, D102

import asyncio
import unittest

from pipecat.frames.frames import EndTaskFrame
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.frame_processor import FrameDirection

import examples_registry
from examples.shared.activity_check import (
    ActivityCheckProcessor,
    activity_check_instruction,
    create_activity_check_processor,
)


class _TestActivityCheckProcessor(ActivityCheckProcessor):
    """Use plain asyncio tasks without a running Pipecat pipeline."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.emitted_frames = []

    def create_task(self, coroutine, name=None):
        return asyncio.create_task(coroutine, name=name)

    async def push_frame(self, frame, direction=FrameDirection.DOWNSTREAM):
        self.emitted_frames.append((frame, direction))


class ActivityCheckProcessorTests(unittest.IsolatedAsyncioTestCase):
    async def test_warning_watchdog_advances_then_requests_graceful_stop(self) -> None:
        warnings: list[int] = []

        async def on_warning(stage: int) -> None:
            warnings.append(stage)

        processor = _TestActivityCheckProcessor(
            activity_check_interval_s=1.0,
            second_warning_s=1.0,
            warning_completion_timeout_s=0.01,
            on_warning=on_warning,
        )

        await processor._emit_warning(1)

        await asyncio.wait_for(self._wait_for_end_task(processor), timeout=0.5)

        self.assertEqual(warnings, [1, 2])
        frame, direction = processor.emitted_frames[-1]
        self.assertIsInstance(frame, EndTaskFrame)
        self.assertEqual(direction, FrameDirection.UPSTREAM)

    async def test_late_first_warning_completion_does_not_end_second_warning(self) -> None:
        warnings: list[int] = []

        async def on_warning(stage: int) -> None:
            warnings.append(stage)

        processor = _TestActivityCheckProcessor(
            activity_check_interval_s=1.0,
            second_warning_s=1.0,
            warning_completion_timeout_s=0.01,
            on_warning=on_warning,
        )

        await processor._emit_warning(1)
        await asyncio.wait_for(self._wait_for_warnings(warnings), timeout=0.5)

        processor._handle_tts_started()
        processor._handle_llm_response_ended()
        processor._handle_bot_stopped_speaking()

        self.assertEqual(processor._retired_warning_completions, 0)
        self.assertTrue(processor._disconnect_after_speech)
        self.assertIsNotNone(processor._warning_completion_timer)
        self.assertFalse(processor.emitted_frames)

        processor._handle_tts_started()
        processor._handle_llm_response_ended()
        processor._handle_bot_stopped_speaking()
        await asyncio.wait_for(self._wait_for_end_task(processor), timeout=0.5)

    async def test_empty_first_warning_does_not_retire_second_warning_completion(self) -> None:
        warnings: list[int] = []

        async def on_warning(stage: int) -> None:
            warnings.append(stage)

        processor = _TestActivityCheckProcessor(
            activity_check_interval_s=1.0,
            second_warning_s=1.0,
            warning_completion_timeout_s=0.05,
            on_warning=on_warning,
        )

        await processor._emit_warning(1)
        await asyncio.wait_for(self._wait_for_warnings(warnings), timeout=0.5)

        processor._handle_llm_response_ended()
        processor._handle_tts_started()
        processor._handle_llm_response_ended()
        processor._handle_bot_stopped_speaking()

        await asyncio.wait_for(self._wait_for_end_task(processor), timeout=0.5)
        self.assertEqual(processor._retired_warning_completions, 0)

    async def test_processor_appends_developer_instruction_to_context(self) -> None:
        context = LLMContext(messages=[])
        runs = 0

        async def queue_llm_run() -> None:
            nonlocal runs
            runs += 1

        processor = create_activity_check_processor(
            {
                "first_warning_s": 600,
                "second_warning_s": 30,
                "warning_completion_timeout_s": 45,
            },
            context=context,
            queue_llm_run=queue_llm_run,
            instruction_role="developer",
        )

        self.assertIsNotNone(processor)
        await processor._on_warning(1)

        self.assertEqual(runs, 1)
        self.assertEqual(context.get_messages()[-1]["role"], "developer")
        self.assertEqual(context.get_messages()[-1]["content"], activity_check_instruction(1))

    def test_activity_instructions_require_a_single_clean_spoken_sentence(self) -> None:
        for stage in (1, 2):
            instruction = activity_check_instruction(stage)
            self.assertIn("exactly one", instruction)
            self.assertIn("Output only that sentence", instruction)
            self.assertIn("think tags", instruction)

    def test_generic_example_enables_activity_check_in_registry(self) -> None:
        config = examples_registry.activity_check_config("generic-assistant")

        self.assertEqual(config["first_warning_s"], 600.0)
        self.assertEqual(config["second_warning_s"], 30.0)
        self.assertIsNone(examples_registry.activity_check_config("multilingual-assistant"))

    async def _wait_for_warnings(self, warnings: list[int]) -> None:
        while warnings != [1, 2]:
            await asyncio.sleep(0.01)

    async def _wait_for_end_task(self, processor: _TestActivityCheckProcessor) -> None:
        while not processor.emitted_frames:
            await asyncio.sleep(0.01)


if __name__ == "__main__":
    unittest.main()
