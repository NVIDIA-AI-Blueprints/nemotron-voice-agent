# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""OpenAI Responses backend: each round is one stateless Responses request with the whole history.

Every round sends ``store: false``, so a failed round loses nothing and the hosted API stores nothing. One
``prompt_cache_key`` per provider (one session's model) lets the hosted API reuse the shared prefix.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

from loguru import logger
from openai import AsyncOpenAI

from ...config import ModelEndpoint
from ...ids import uid
from .base import DEFAULT_LIMITS, Backend, ProviderLimits, make_client

# The round's settings a Responses request takes as they are.
SETTINGS = ("instructions", "max_output_tokens", "parallel_tool_calls", "reasoning", "text", "service_tier")
# The endpoint's catalog settings (``extra_params``, written for chat completions) a Responses request has, by name.
ENDPOINT_PARAMS = {"temperature": "temperature", "top_p": "top_p", "max_tokens": "max_output_tokens"}


class OpenAIResponsesBackend(Backend):
    """Backend on the OpenAI Responses API."""

    name = "openai-responses"

    def __init__(
        self, endpoint: ModelEndpoint, limits: ProviderLimits = DEFAULT_LIMITS, client: AsyncOpenAI | None = None
    ):
        """Bind the endpoint; make its client unless one is injected."""
        super().__init__(endpoint, limits)
        self.client = client or make_client(endpoint, limits)
        self.cache_key = uid("cache")

    def request(self, payload: dict) -> dict:
        """Map the round onto a Responses request: the endpoint's defaults, then the round's settings and input."""
        extra = self.endpoint.extra_params
        request = {name: extra[key] for key, name in ENDPOINT_PARAMS.items() if key in extra}
        if "extra_body" in extra:
            request["extra_body"] = extra["extra_body"]
        ignored = sorted(set(extra) - set(ENDPOINT_PARAMS) - {"extra_body"})
        if ignored:
            logger.debug(f"Endpoint settings a Responses request has no field for: {ignored}")
        request.update({key: payload[key] for key in SETTINGS if payload.get(key) is not None})
        if payload.get("tools"):
            request["tools"] = payload["tools"]
            request["tool_choice"] = payload.get("tool_choice", "auto")
        request.update(
            model=payload.get("model") or self.endpoint.model,
            input=payload["input"],
            store=False,
            prompt_cache_key=self.cache_key,
            stream=True,
        )
        return request

    async def _events(self, payload: dict) -> AsyncIterator[dict]:
        """Yield the Responses events of one round, as the API streams them."""
        stream = await self.client.responses.create(**self.request(payload))
        try:
            async for event in stream:
                yield event.model_dump(mode="json", exclude_none=True)
        finally:
            await stream.close()

    async def close(self) -> None:
        """Close the HTTP client."""
        await self.client.close()
