# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

# ruff: noqa: D100, D101, D102

import unittest
from unittest.mock import AsyncMock

from examples.omni_assistant_subagents.subagents.transport.proactive_gesture_controller import (
    ProactiveGestureController,
)


def _controller(resume_or_compliment: AsyncMock) -> ProactiveGestureController:
    return ProactiveGestureController(
        queue_frame=AsyncMock(),
        greet=AsyncMock(),
        barge_in=AsyncMock(),
        resume_or_compliment=resume_or_compliment,
        acknowledge_feedback=AsyncMock(),
        is_assistant_speaking=lambda: False,
        is_user_speaking=lambda: False,
    )


class GestureConfirmationTests(unittest.IsolatedAsyncioTestCase):
    async def test_single_thumbs_up_does_not_trigger(self) -> None:
        callback = AsyncMock()
        controller = _controller(callback)

        await controller.handle({"intent": "continue", "confidence": 0.99}, frame={})

        callback.assert_not_awaited()

    async def test_two_consecutive_thumbs_up_detections_trigger_once(self) -> None:
        callback = AsyncMock()
        controller = _controller(callback)

        await controller.handle({"intent": "continue", "confidence": 0.99}, frame={})
        await controller.handle({"intent": "continue", "confidence": 0.99}, frame={})
        await controller.handle({"intent": "continue", "confidence": 0.99}, frame={})

        callback.assert_awaited_once_with(False)

    async def test_none_resets_thumbs_up_confirmation(self) -> None:
        callback = AsyncMock()
        controller = _controller(callback)

        await controller.handle({"intent": "continue", "confidence": 0.99}, frame={})
        await controller.handle({"intent": "none", "confidence": 0.0}, frame={})
        await controller.handle({"intent": "continue", "confidence": 0.99}, frame={})

        callback.assert_not_awaited()

    async def test_low_confidence_thumbs_up_does_not_accumulate(self) -> None:
        callback = AsyncMock()
        controller = _controller(callback)

        await controller.handle({"intent": "continue", "confidence": 0.85}, frame={})
        await controller.handle({"intent": "continue", "confidence": 0.85}, frame={})

        callback.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()
