# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for CI-only NVIDIA Inference API services."""

from importlib import import_module, util
from pathlib import Path
from unittest import IsolatedAsyncioTestCase, TestCase
from unittest.mock import AsyncMock

from pipecat.evals.scenario import EvalScenarioFile
from pipecat.frames.frames import ErrorFrame

_SERVICE_PATH = Path(__file__).resolve().parents[2] / "evals" / "pipecat" / "service" / "nvidia_inference_services.py"
_SPEC = util.spec_from_file_location("nvidia_inference_services", _SERVICE_PATH)
assert _SPEC and _SPEC.loader
nvidia_inference_services = util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(nvidia_inference_services)
InferenceMagpieTTSService = nvidia_inference_services.InferenceMagpieTTSService


class ServiceEvalFactoryTests(TestCase):
    """Validate the configured judge imports without calling hosted services."""

    def test_scenario_judge_factories_can_be_imported(self) -> None:
        """Load scenario includes and resolve the judge factory used by the harness."""
        checked = 0
        for path in sorted((_SERVICE_PATH.parent / "scenarios").glob("*.yaml")):
            if path.name.startswith("_"):
                continue
            for scenario in EvalScenarioFile.load(path):
                factory = scenario.judge.get("factory")
                if not factory:
                    continue
                with self.subTest(scenario=scenario.name):
                    module, name = factory.rsplit(".", 1)
                    self.assertTrue(callable(getattr(import_module(module), name)))
                checked += 1
        self.assertGreater(checked, 0)


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
