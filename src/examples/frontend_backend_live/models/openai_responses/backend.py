# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""OpenAI thinker: the Responses API, which keeps the conversation (``previous_response_id``)."""

from __future__ import annotations

from collections.abc import AsyncIterator

from openai import AsyncOpenAI

from ...config_manager.schema import ModelEndpoint, ReliabilityConfig
from ..base import Backend
from ..registry import BACKENDS


@BACKENDS.register("openai-responses")
class OpenAIResponsesBackend(Backend):
    """Thinker on the OpenAI Responses API."""

    stateful = True

    def __init__(self, endpoint: ModelEndpoint, reliability: ReliabilityConfig, client: AsyncOpenAI | None = None):
        """Create the client unless one is injected."""
        super().__init__(endpoint, reliability)
        self.client = client or AsyncOpenAI(
            api_key=endpoint.api_key or "not-needed",
            base_url=endpoint.base_url,
            timeout=reliability.api_timeout_seconds,
            max_retries=0,
        )

    async def stream(self, payload: dict) -> AsyncIterator[dict]:
        """Yield the Responses events of one round."""
        payload = {**payload, "model": payload.get("model") or self.endpoint.model}
        stream = await self.client.responses.create(**payload, stream=True)
        try:
            async for item in stream:
                yield item.model_dump(mode="json", exclude_none=True)
        finally:
            await stream.close()

    async def close(self) -> None:
        """Close the HTTP client."""
        await self.client.close()
