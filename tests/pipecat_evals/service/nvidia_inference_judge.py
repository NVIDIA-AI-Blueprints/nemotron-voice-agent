# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""NVIDIA-hosted OpenAI-compatible judge factory for Pipecat evaluations."""

from __future__ import annotations

import os
from typing import Any

from pipecat.services.openai.llm import OpenAILLMService

NVIDIA_INFERENCE_API_BASE_URL = "https://inference-api.nvidia.com/v1"
NVIDIA_INFERENCE_JUDGE_MODEL = "google/gemma-4-31b-it"


def create_nvidia_inference_judge(config: dict[str, Any]) -> OpenAILLMService:
    """Create an OpenAI-compatible judge backed by the NVIDIA-hosted API.

    Pipecat's built-in ``service: openai`` judge does not pass its scenario config
    to the service constructor, so it cannot set a provider endpoint or credential.
    This thin factory configures Pipecat's own OpenAI-compatible service with the
    NVIDIA-hosted endpoint supplied by CI. The key remains in the
    ``NVIDIA_INFERENCE_API_KEY`` environment variable; ``NVIDIA_API_KEY`` is a
    backwards-compatible fallback for local cloud-service eval runs.
    """
    api_key = config.get("api_key") or os.environ.get("NVIDIA_INFERENCE_API_KEY") or os.environ.get("NVIDIA_API_KEY")
    if not api_key:
        raise RuntimeError(
            "NVIDIA_INFERENCE_API_KEY (or NVIDIA_API_KEY) is required for the NVIDIA-hosted evaluation judge"
        )

    return OpenAILLMService(
        api_key=api_key,
        base_url=(
            config.get("endpoint") or os.environ.get("NVIDIA_INFERENCE_API_BASE_URL") or NVIDIA_INFERENCE_API_BASE_URL
        ),
        settings=OpenAILLMService.Settings(
            model=config.get("model") or NVIDIA_INFERENCE_JUDGE_MODEL,
            temperature=float(config.get("temperature", 0.2)),
        ),
    )
