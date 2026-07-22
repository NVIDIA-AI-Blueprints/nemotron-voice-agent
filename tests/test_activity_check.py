# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

# ruff: noqa: D100, D101, D102

import asyncio
import unittest

from examples.shared.activity_check import ActivityCheckProcessor


class _TestActivityCheckProcessor(ActivityCheckProcessor):
    """Use plain asyncio tasks without a running Pipecat pipeline."""

    def create_task(self, coroutine, name=None):
        return asyncio.create_task(coroutine, name=name)


class ActivityCheckProcessorTests(unittest.IsolatedAsyncioTestCase):
    async def test_warning_watchdog_advances_then_disconnects(self) -> None:
        warnings: list[int] = []
        disconnected = asyncio.Event()

        async def on_warning(stage: int) -> None:
            warnings.append(stage)

        async def on_disconnect() -> None:
            disconnected.set()

        processor = _TestActivityCheckProcessor(
            activity_check_interval_s=1.0,
            second_warning_s=1.0,
            warning_completion_timeout_s=0.01,
            on_warning=on_warning,
            on_disconnect=on_disconnect,
        )

        await processor._emit_warning(1)

        await asyncio.wait_for(disconnected.wait(), timeout=0.5)

        self.assertEqual(warnings, [1, 2])

    async def test_late_first_warning_completion_does_not_end_second_warning(self) -> None:
        warnings: list[int] = []
        disconnected = asyncio.Event()

        async def on_warning(stage: int) -> None:
            warnings.append(stage)

        async def on_disconnect() -> None:
            disconnected.set()

        processor = _TestActivityCheckProcessor(
            activity_check_interval_s=1.0,
            second_warning_s=1.0,
            warning_completion_timeout_s=0.01,
            on_warning=on_warning,
            on_disconnect=on_disconnect,
        )

        await processor._emit_warning(1)
        await asyncio.wait_for(self._wait_for_warnings(warnings), timeout=0.5)

        processor._handle_tts_started()
        processor._handle_llm_response_ended()
        processor._handle_bot_stopped_speaking()

        self.assertEqual(processor._retired_warning_completions, 0)
        self.assertTrue(processor._disconnect_after_speech)
        self.assertIsNotNone(processor._warning_completion_timer)
        self.assertFalse(disconnected.is_set())

        processor._handle_tts_started()
        processor._handle_llm_response_ended()
        processor._handle_bot_stopped_speaking()
        await asyncio.wait_for(disconnected.wait(), timeout=0.5)

    async def test_empty_first_warning_does_not_retire_second_warning_completion(self) -> None:
        warnings: list[int] = []
        disconnected = asyncio.Event()

        async def on_warning(stage: int) -> None:
            warnings.append(stage)

        async def on_disconnect() -> None:
            disconnected.set()

        processor = _TestActivityCheckProcessor(
            activity_check_interval_s=1.0,
            second_warning_s=1.0,
            warning_completion_timeout_s=0.05,
            on_warning=on_warning,
            on_disconnect=on_disconnect,
        )

        await processor._emit_warning(1)
        await asyncio.wait_for(self._wait_for_warnings(warnings), timeout=0.5)

        processor._handle_llm_response_ended()
        processor._handle_tts_started()
        processor._handle_llm_response_ended()
        processor._handle_bot_stopped_speaking()

        await asyncio.wait_for(disconnected.wait(), timeout=0.5)
        self.assertEqual(processor._retired_warning_completions, 0)

    async def _wait_for_warnings(self, warnings: list[int]) -> None:
        while warnings != [1, 2]:
            await asyncio.sleep(0.01)


if __name__ == "__main__":
    unittest.main()
