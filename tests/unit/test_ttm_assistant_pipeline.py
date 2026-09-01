# SPDX-FileCopyrightText: Copyright (c) 2024–2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

# ruff: noqa: D100, D101, D102

import os
import sys
import types
import unittest
from unittest.mock import AsyncMock, Mock, patch

from pipecat.runner.types import RunnerArguments
from pipecat.turns.user_turn_strategies import ExternalUserTurnStrategies

import examples_registry

try:
    import pipecat_ttm  # noqa: F401
except ModuleNotFoundError:
    pipecat_ttm = types.ModuleType("pipecat_ttm")
    pipecat_ttm.TTMUserTurnProcessor = Mock
    sys.modules["pipecat_ttm"] = pipecat_ttm

from examples.generic import pipeline_ttm
from examples.generic.pipeline_ttm import (
    DEFAULT_TTM_OPEN_TIMEOUT_SECS,
    DEFAULT_TTM_TURN_EVENTS_URL,
    _build_ttm_user_aggregator_params,
    _ttm_open_timeout_secs,
    _ttm_turn_events_url,
)


class TTMAssistantPipelineTests(unittest.IsolatedAsyncioTestCase):
    def test_default_turn_events_url_targets_local_ttm_service(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(_ttm_turn_events_url(), DEFAULT_TTM_TURN_EVENTS_URL)

    def test_configured_turn_events_url_is_trimmed(self) -> None:
        with patch.dict(
            os.environ,
            {"TTM_TURN_EVENTS_URL": "  ws://ttm.example:7860/v1/audio/turn-events  "},
        ):
            self.assertEqual(
                _ttm_turn_events_url(),
                "ws://ttm.example:7860/v1/audio/turn-events",
            )

    def test_ttm_open_timeout_is_configurable(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(_ttm_open_timeout_secs(), DEFAULT_TTM_OPEN_TIMEOUT_SECS)
        with patch.dict(os.environ, {"TTM_OPEN_TIMEOUT_SECS": "12.5"}):
            self.assertEqual(_ttm_open_timeout_secs(), 12.5)

    def test_ttm_is_the_only_turn_strategy(self) -> None:
        params = _build_ttm_user_aggregator_params(welcome_enabled=False)

        self.assertIsNone(params.vad_analyzer)
        self.assertIsInstance(params.user_turn_strategies, ExternalUserTurnStrategies)
        self.assertEqual(params.user_mute_strategies, [])

    def test_welcome_message_keeps_existing_user_mute_behavior(self) -> None:
        params = _build_ttm_user_aggregator_params(welcome_enabled=True)

        self.assertEqual(len(params.user_mute_strategies), 1)

    def test_registry_exposes_separate_ttm_assistant(self) -> None:
        example = examples_registry.find("ttm-assistant")

        self.assertEqual(example["bot"], "examples.generic.pipeline_ttm:bot")
        self.assertEqual(example["slots"], ["llm", "asr", "tts"])
        self.assertEqual(
            examples_registry.prompt_default_key("ttm-assistant"),
            "generic_assistant",
        )

    async def test_bot_injects_ttm_processor_into_generic_pipeline(self) -> None:
        runner_args = RunnerArguments()
        runner_args.body = {"pipeline_mode": "ttm-assistant"}
        processor = Mock()

        with (
            patch.object(pipeline_ttm, "TTMUserTurnProcessor", return_value=processor) as processor_class,
            patch.object(pipeline_ttm, "_run_bot", new_callable=AsyncMock) as run_bot,
        ):
            await pipeline_ttm.bot(runner_args)

        processor_class.assert_called_once_with(
            url=DEFAULT_TTM_TURN_EVENTS_URL,
            open_timeout=DEFAULT_TTM_OPEN_TIMEOUT_SECS,
        )
        self.assertIs(run_bot.await_args.kwargs["turn_processor"], processor)
        self.assertIsInstance(
            run_bot.await_args.kwargs["user_aggregator_params"].user_turn_strategies,
            ExternalUserTurnStrategies,
        )


if __name__ == "__main__":
    unittest.main()
