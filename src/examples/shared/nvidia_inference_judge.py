# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""NVIDIA-hosted judge factory for Pipecat evaluations."""

from __future__ import annotations

import asyncio
import os
from typing import Any

from openai import AsyncOpenAI, InternalServerError

NVIDIA_INFERENCE_API_BASE_URL = "https://inference-api.nvidia.com/v1"
NVIDIA_INFERENCE_JUDGE_MODEL = "nvidia/google/gemma-4-31b-it"


class NvidiaInferenceJudgeService:
    """NVIDIA-hosted client implementing Pipecat's judge interface."""

    def __init__(self, *, api_key: str, base_url: str, model: str, temperature: float) -> None:
        """Initialize the OpenAI-compatible NVIDIA-hosted client."""
        self._client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self._model = model
        self._temperature = temperature

    async def run_inference(
        self, context: Any, max_tokens: int | None = None, system_instruction: str | None = None
    ) -> str | None:
        """Retry transient NVIDIA-hosted server errors before failing the eval."""
        messages = [dict(message) for message in context.get_messages()]
        if system_instruction:
            messages.insert(0, {"role": "system", "content": system_instruction})

        for attempt in range(3):
            try:
                response = await self._client.chat.completions.create(
                    model=self._model,
                    messages=messages,
                    temperature=self._temperature,
                    max_tokens=max_tokens or 200,
                )
                return response.choices[0].message.content
            except InternalServerError:
                if attempt == 2:
                    raise
                await asyncio.sleep(2**attempt)

        raise AssertionError("unreachable")


def create_nvidia_inference_judge(config: dict[str, Any]) -> NvidiaInferenceJudgeService:
    """Create an OpenAI-compatible judge backed by the NVIDIA-hosted API.

    Pipecat's built-in ``service: openai`` judge only uses OpenAI's default API
    endpoint. This factory supplies NVIDIA's OpenAI-compatible hosted endpoint while
    keeping the key in the ``NVIDIA_INFERENCE_API_KEY``
    environment variable. ``NVIDIA_API_KEY`` remains a fallback for backwards
    compatibility, so cloud-service eval configurations that use one credential
    continue to work.
    """
    api_key = config.get("api_key") or os.environ.get("NVIDIA_INFERENCE_API_KEY") or os.environ.get("NVIDIA_API_KEY")
    if not api_key:
        raise RuntimeError(
            "NVIDIA_INFERENCE_API_KEY (or NVIDIA_API_KEY) is required for the NVIDIA-hosted evaluation judge"
        )

    return NvidiaInferenceJudgeService(
        api_key=api_key,
        base_url=(
            config.get("endpoint") or os.environ.get("NVIDIA_INFERENCE_API_BASE_URL") or NVIDIA_INFERENCE_API_BASE_URL
        ),
        model=config.get("model") or NVIDIA_INFERENCE_JUDGE_MODEL,
        temperature=float(config.get("temperature", 0.2)),
    )
