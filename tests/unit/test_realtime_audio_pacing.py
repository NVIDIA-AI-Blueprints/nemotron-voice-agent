# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

# ruff: noqa: D102

"""Deterministic tests for Realtime WebSocket output-audio pacing."""

from __future__ import annotations

import unittest
from unittest.mock import AsyncMock, patch

from realtime_helpers import FakeWebSocket

from realtime.controller import RealtimeSessionController
from realtime.transport import (
    _BoundedCatchUpFastAPIWebsocketOutputTransport,
    create_realtime_transport,
    shutdown_realtime_transport,
)


def _output(*, interval: float = 0.1, deadline: float = 0.0, catch_up: float = 0.3):
    output = object.__new__(_BoundedCatchUpFastAPIWebsocketOutputTransport)
    output._send_interval = interval
    output._next_send_time = deadline
    output._max_catch_up_seconds = catch_up
    return output


class RealtimeAudioPacingTests(unittest.IsolatedAsyncioTestCase):
    """Exercise the clock directly without wall-clock sleeps."""

    async def test_on_time_audio_preserves_realtime_pacing(self) -> None:
        output = _output(deadline=10.1)
        sleep = AsyncMock()

        with (
            patch("realtime.transport.time.monotonic", return_value=10.0),
            patch("realtime.transport.asyncio.sleep", sleep),
        ):
            await output._write_audio_sleep()

        sleep.assert_awaited_once()
        self.assertAlmostEqual(sleep.await_args.args[0], 0.1)
        self.assertAlmostEqual(output._next_send_time, 10.2)

    async def test_transient_lateness_retains_the_prior_audio_clock(self) -> None:
        output = _output(deadline=10.1)
        sleep = AsyncMock()

        with (
            patch("realtime.transport.time.monotonic", return_value=10.25),
            patch("realtime.transport.asyncio.sleep", sleep),
        ):
            await output._write_audio_sleep()

        sleep.assert_awaited_once_with(0.0)
        self.assertAlmostEqual(output._next_send_time, 10.2)
        self.assertLess(output._next_send_time, 10.25)

    async def test_available_chunks_repay_debt_then_resume_realtime_pacing(self) -> None:
        output = _output(deadline=10.1)
        sleep = AsyncMock()

        with (
            patch("realtime.transport.time.monotonic", return_value=10.35),
            patch("realtime.transport.asyncio.sleep", sleep),
        ):
            for _ in range(4):
                await output._write_audio_sleep()

        sleeps = [awaited.args[0] for awaited in sleep.await_args_list]
        self.assertEqual(sleeps[:3], [0.0, 0.0, 0.0])
        self.assertAlmostEqual(sleeps[3], 0.05)
        self.assertAlmostEqual(output._next_send_time, 10.5)

    async def test_large_lateness_is_clamped_to_the_catch_up_window(self) -> None:
        output = _output(deadline=1.0)
        sleep = AsyncMock()

        with (
            patch("realtime.transport.time.monotonic", return_value=10.0),
            patch("realtime.transport.asyncio.sleep", sleep),
        ):
            await output._write_audio_sleep()

        sleep.assert_awaited_once_with(0.0)
        self.assertAlmostEqual(output._next_send_time, 9.8)

    async def test_zero_catch_up_window_restores_reset_to_realtime_behavior(self) -> None:
        output = _output(deadline=1.0, catch_up=0.0)
        sleep = AsyncMock()

        with (
            patch("realtime.transport.time.monotonic", return_value=10.0),
            patch("realtime.transport.asyncio.sleep", sleep),
        ):
            await output._write_audio_sleep()

        sleep.assert_awaited_once_with(0.0)
        self.assertAlmostEqual(output._next_send_time, 10.1)

    async def test_realtime_transport_installs_configured_catch_up_output_edge(self) -> None:
        controller = RealtimeSessionController(
            model="test-realtime-model",
            voice="test-voice",
            runtime_config={"pipeline_mode": "generic-assistant"},
        )

        with patch.dict("os.environ", {"AUDIO_OUT_MAX_CATCH_UP_SECONDS": "0.125"}):
            transport = create_realtime_transport(FakeWebSocket([]), controller=controller)

        try:
            self.assertIsInstance(transport.output(), _BoundedCatchUpFastAPIWebsocketOutputTransport)
            self.assertEqual(transport.output()._max_catch_up_seconds, 0.125)
        finally:
            shutdown_realtime_transport(transport)
