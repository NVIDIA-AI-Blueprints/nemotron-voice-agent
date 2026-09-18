# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for CI-only NVIDIA Inference API services."""

from importlib import util
from pathlib import Path
from unittest import IsolatedAsyncioTestCase
from unittest.mock import AsyncMock

from pipecat.frames.frames import ErrorFrame

_SERVICE_PATH = Path(__file__).resolve().parents[1] / "pipecat_evals" / "service" / "nvidia_inference_services.py"
_SPEC = util.spec_from_file_location("nvidia_inference_services", _SERVICE_PATH)
assert _SPEC and _SPEC.loader
nvidia_inference_services = util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(nvidia_inference_services)
InferenceMagpieTTSService = nvidia_inference_services.InferenceMagpieTTSService


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
