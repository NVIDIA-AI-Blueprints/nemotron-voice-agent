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


if __name__ == "__main__":
    unittest.main()
