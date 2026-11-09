# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""OpenAI Responses backend: each round is one stateless Responses request with the whole history."""

from __future__ import annotations

from collections.abc import AsyncIterator, Mapping
from copy import deepcopy
from typing import Any

from loguru import logger
from openai import AsyncOpenAI

from examples.shared.delegation_contract import ModelEndpoint
from examples.shared.ids import uid

from .base import Backend, ProviderLimits, apply_thinking, client_run, make_client

SETTINGS = ("instructions", "max_output_tokens", "parallel_tool_calls", "reasoning", "text", "service_tier")
# The endpoint's extra_params, written for chat completions, that a Responses request has, by its name there.
ENDPOINT_PARAMS = {"temperature": "temperature", "top_p": "top_p", "max_tokens": "max_output_tokens"}


def _named(item: Any) -> tuple:
    """Return what a tool or a ``tool_choice`` names: its type, and an MCP tool's server."""
    return (item.get("type"), item.get("server_label")) if isinstance(item, Mapping) else (None, None)


class OpenAIResponsesBackend(Backend):
    """Backend on the OpenAI Responses API."""

    name = "openai-responses"

    def __init__(self, endpoint: ModelEndpoint, limits: ProviderLimits, client: AsyncOpenAI | None = None):
        """Bind the endpoint; make its client unless one is injected."""
        super().__init__(endpoint, limits)
        self.client = client or make_client(endpoint, limits)
        self.cache_key = uid("cache")

    def request(self, payload: dict) -> dict:
        """Map the round onto a Responses request: the endpoint's defaults, then the round's settings and input.

        ``store: false`` keeps nothing on the hosted API, and one ``prompt_cache_key`` per provider reuses the prefix.
        Thinking that the endpoint's switch or the round turns off is also sent as ``reasoning.effort: none``, which a
        server that ignores ``chat_template_kwargs`` reads. A tool the client runs is left out (:func:`client_run`), and
        a ``tool_choice`` that only a tool left out matches becomes ``auto``, as on chat completions.
        """
        extra = self.endpoint.extra_params
        request = {name: extra[key] for key, name in ENDPOINT_PARAMS.items() if key in extra}
        if "extra_body" in extra:
            request["extra_body"] = deepcopy(extra["extra_body"])
        ignored = sorted(set(extra) - set(ENDPOINT_PARAMS) - {"extra_body"})
        if ignored:
            logger.debug(f"Endpoint settings a Responses request has no field for: {ignored}")
        request.update({key: payload[key] for key in SETTINGS if payload.get(key) is not None})
        if apply_thinking(payload, request.get("extra_body")) is False:
            request["reasoning"] = {**(payload.get("reasoning") or {}), "effort": "none"}
        offered = payload.get("tools") or []
        tools = [tool for tool in offered if not client_run(tool)]
        dropped = [tool for tool in offered if client_run(tool)]
        if dropped:
            logger.debug(f"Tools whose calls the client runs, left out of the request: {[t['type'] for t in dropped]}")
        if tools:
            choice = payload.get("tool_choice", "auto")
            gone = {_named(tool) for tool in dropped} - {_named(tool) for tool in tools}
            request["tools"] = tools
            request["tool_choice"] = "auto" if _named(choice) in gone else choice
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
