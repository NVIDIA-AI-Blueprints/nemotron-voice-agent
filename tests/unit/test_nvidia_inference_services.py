# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""Tests for CI-only NVIDIA Inference API services."""

from unittest import IsolatedAsyncioTestCase
from unittest.mock import AsyncMock

from pipecat.frames.frames import ErrorFrame

from tests.pipecat_evals.service.nvidia_inference_services import InferenceMagpieTTSService


class InferenceMagpieTTSServiceTests(IsolatedAsyncioTestCase):
    """Verify hosted TTS error handling."""

    async def test_synthesis_failure_stops_ttfb_metrics(self) -> None:
        """Stop an active TTFB measurement when synthesis emits no audio."""
        service = InferenceMagpieTTSService()
        service.start_tts_usage_metrics = AsyncMock(side_effect=RuntimeError("synthesis failed"))
        service.stop_ttfb_metrics = AsyncMock()

        frames = [frame async for frame in service.run_tts("hello", "context-1")]

        service.stop_ttfb_metrics.assert_awaited_once_with()
        self.assertEqual(len(frames), 1)
        self.assertIsInstance(frames[0], ErrorFrame)
