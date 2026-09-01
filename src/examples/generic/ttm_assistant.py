# SPDX-FileCopyrightText: Copyright (c) 2024–2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""TTM Assistant entry point with externally detected user turns."""

import os

from pipecat.processors.aggregators.llm_response_universal import LLMUserAggregatorParams
from pipecat.runner.types import RunnerArguments
from pipecat.turns.user_turn_strategies import ExternalUserTurnStrategies

import examples_registry
from examples.generic.pipeline import _run_bot
from examples.generic.ttm_user_turn_processor import TTMUserTurnProcessor
from examples.shared.pipeline_utils import build_user_mute_strategies
from utils import parse_env_float

DEFAULT_TTM_TURN_EVENTS_URL = "ws://127.0.0.1:7860/v1/audio/turn-events"
DEFAULT_TTM_OPEN_TIMEOUT_SECS = 10.0
DEFAULT_TTM_EOU_FALLBACK_SILENCE_SECS = 5.0


def _ttm_turn_events_url() -> str:
    """Return the configured TTM turn-event WebSocket endpoint."""
    return os.getenv("TTM_TURN_EVENTS_URL", "").strip() or DEFAULT_TTM_TURN_EVENTS_URL


def _ttm_open_timeout_secs() -> float:
    """Return the timeout for opening a TTM turn-event connection."""
    return parse_env_float(
        "TTM_OPEN_TIMEOUT_SECS",
        DEFAULT_TTM_OPEN_TIMEOUT_SECS,
        min_value=0.1,
    )


def _ttm_eou_fallback_silence_secs() -> float:
    """Return the Silero silence window used when TTM EOU is delayed."""
    return parse_env_float(
        "TTM_EOU_FALLBACK_SILENCE_SECS",
        DEFAULT_TTM_EOU_FALLBACK_SILENCE_SECS,
        min_value=0.1,
    )


def _build_ttm_user_aggregator_params(welcome_enabled: bool) -> LLMUserAggregatorParams:
    """Keep the local TTM processor as the sole turn-frame owner."""
    return LLMUserAggregatorParams(
        user_mute_strategies=build_user_mute_strategies(welcome_enabled),
        user_turn_strategies=ExternalUserTurnStrategies(),
    )


async def bot(runner_args: RunnerArguments) -> None:
    """Build and run the generic assistant with TTM turn detection."""
    body = runner_args.body if isinstance(runner_args.body, dict) else {}
    welcome_enabled = examples_registry.welcome_message_enabled(body.get("pipeline_mode", ""))
    await _run_bot(
        runner_args,
        turn_processor=TTMUserTurnProcessor(
            url=_ttm_turn_events_url(),
            open_timeout=_ttm_open_timeout_secs(),
            silence_fallback_secs=_ttm_eou_fallback_silence_secs(),
        ),
        user_aggregator_params=_build_ttm_user_aggregator_params(welcome_enabled),
    )
