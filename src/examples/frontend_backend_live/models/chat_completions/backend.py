# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Chat-completions thinker: function tools on any OpenAI-compatible endpoint, as a Responses event stream.

Chat completions are stateless, so the delegate resends its full history each round. Retries cover only
failures before the first chunk: after that a tool call may already be on its way to the executor, and
repeating the round could repeat an action.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

from loguru import logger
from openai import AsyncOpenAI

from examples.frontend_backend_live.common.retry import with_retries

from ...config_manager.schema import ModelEndpoint, ReliabilityConfig
from ..base import Backend
from ..registry import BACKENDS
from .translation import (
    ChatResponseTranslator,
    responses_items_to_messages,
    thinking_enabled,
    to_chat_tool_choice,
    to_chat_tools,
)

DEFAULT_SAMPLING = {"temperature": 0.7, "max_tokens": 1024}


@BACKENDS.register("chat-completions")
class ChatCompletionsBackend(Backend):
    """Thinker on an OpenAI-compatible chat-completions endpoint."""

    stateful = False

    def __init__(self, endpoint: ModelEndpoint, reliability: ReliabilityConfig, client: AsyncOpenAI | None = None):
        """Create the client unless one is injected."""
        super().__init__(endpoint, reliability)
        self.client = client or AsyncOpenAI(
            api_key=endpoint.api_key or "not-needed",
            base_url=endpoint.base_url,
            timeout=reliability.api_timeout_seconds,
            max_retries=0,
        )

    def validate_model(self, model: str) -> None:
        """Accept the configured model or a catalog-style id (``org/name``); anything else is not served here."""
        if model != self.endpoint.model and "/" not in model:
            raise ValueError(
                f"This endpoint serves catalog-style model ids; use {self.endpoint.model!r} (got {model!r})."
            )

    def request(self, payload: dict, messages: list[dict]) -> dict:
        """Build the chat request: sampling defaults, then the endpoint's ``extra_params``, then the round."""
        request = {
            **DEFAULT_SAMPLING,
            **self.endpoint.extra_params,
            "model": payload.get("model") or self.endpoint.model,
            "messages": messages,
            "stream": True,
        }
        if payload.get("max_output_tokens"):
            request["max_tokens"] = payload["max_output_tokens"]
        tools, dropped = to_chat_tools(payload.get("tools"))
        if dropped:
            logger.debug(f"Hosted tools unavailable on chat completions: {dropped}")
        if tools:
            request["tools"] = tools
            request["tool_choice"] = to_chat_tool_choice(payload.get("tool_choice", "auto"))
        return request

    async def stream(self, payload: dict) -> AsyncIterator[dict]:
        """Yield Responses events for one stateless round; ``payload["input"]`` is the complete history."""
        messages = responses_items_to_messages(payload.get("instructions"), payload["input"])
        request = self.request(payload, messages)
        translator = ChatResponseTranslator(request["model"], buffer_text=thinking_enabled(self.endpoint.extra_params))
        for event in translator.start():
            yield event

        async def open_stream():
            stream = None
            try:
                stream = await self.client.chat.completions.create(**request)
                iterator = stream.__aiter__()
                first = await anext(iterator, None)
                return stream, iterator, first
            except BaseException:
                if stream is not None:
                    await stream.close()
                raise

        stream, iterator, first = await with_retries(
            open_stream, attempts=self.reliability.provider_retries + 1, role="thinker"
        )
        try:

            async def chunks():
                if first is not None:
                    yield first
                async for chunk in iterator:
                    yield chunk

            async for chunk in chunks():
                for event in translator.feed(chunk):
                    yield event
        finally:
            await stream.close()
        for event in translator.finish():
            yield event

    async def close(self) -> None:
        """Close the HTTP client."""
        await self.client.close()
