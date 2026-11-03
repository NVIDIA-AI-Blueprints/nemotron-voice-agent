# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Chat-completions backend: function tools on any OpenAI-compatible endpoint, as a Responses event stream.

Each round is sent the whole history. ``response.created`` is emitted only once the first chunk arrives, so the
first-event limit and its retries cover the whole opening, and nothing is retried after a chunk arrived.
``parallel_tool_calls: false`` keeps only the reply's first call and stops reading once a second call starts;
``reasoning.effort: none`` turns thinking off where the endpoint's ``chat_template_kwargs`` has ``enable_thinking``;
``text.format`` becomes ``response_format``.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Mapping
from copy import deepcopy
from typing import Any

from loguru import logger
from openai import AsyncOpenAI

from ...config import ModelEndpoint
from .base import DEFAULT_LIMITS, Backend, ProviderLimits, make_client
from .translation import (
    ChatResponseTranslator,
    responses_items_to_messages,
    thinking_enabled,
    to_chat_response_format,
    to_chat_tool_choice,
    to_chat_tools,
)

DEFAULT_SAMPLING = {"temperature": 0.7, "max_tokens": 1024}
# The service tiers a chat endpoint honors: no special tier.
PLAIN_TIERS = (None, "auto", "default")


class ChatCompletionsBackend(Backend):
    """Backend on an OpenAI-compatible chat-completions endpoint."""

    name = "chat-completions"

    def __init__(
        self, endpoint: ModelEndpoint, limits: ProviderLimits = DEFAULT_LIMITS, client: AsyncOpenAI | None = None
    ):
        """Bind the endpoint; make its client unless one is injected."""
        super().__init__(endpoint, limits)
        self.client = client or make_client(endpoint, limits)

    @classmethod
    def unsupported(cls, settings: Mapping[str, Any]) -> list[tuple[str, str]]:
        """The request settings a chat endpoint cannot honor: hosted tools, a service tier, ``text.verbosity``."""
        _, hosted = to_chat_tools(settings.get("tools"))
        problems = [
            ("tools", f"The {kind} tool needs a Responses backend; this backend runs function tools only.")
            for kind in dict.fromkeys(hosted)
        ]
        if settings.get("service_tier") not in PLAIN_TIERS:
            problems.append(("service_tier", f"service_tier {settings['service_tier']!r} is not available here."))
        if (settings.get("text") or {}).get("verbosity") is not None:
            problems.append(("text.verbosity", "text.verbosity is not available on a chat-completions backend."))
        return problems

    def request(self, payload: dict) -> dict:
        """Build the chat request: sampling defaults, then the endpoint's ``extra_params``, then the round."""
        messages = responses_items_to_messages(payload.get("instructions"), payload["input"])
        request = {
            **DEFAULT_SAMPLING,
            **deepcopy(self.endpoint.extra_params),
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
        if (payload.get("reasoning") or {}).get("effort") == "none":
            switch = (request.get("extra_body") or {}).get("chat_template_kwargs")
            if isinstance(switch, dict) and "enable_thinking" in switch:
                switch["enable_thinking"] = False
        response_format = to_chat_response_format((payload.get("text") or {}).get("format"))
        if response_format is not None:
            request["response_format"] = response_format
        return request

    async def _events(self, payload: dict) -> AsyncIterator[dict | None]:
        """Open the round, then yield its Responses events: the opening ones only once the first chunk arrived.

        A chunk that gives no event yet (hidden reasoning, or answer text held while the model thinks) gives ``None``, a
        sign of life, so a long but steady thinking round is never cut by the silence limit.
        """
        request = self.request(payload)
        translator = ChatResponseTranslator(
            request["model"],
            buffer_text=thinking_enabled(request),
            single_call=payload.get("parallel_tool_calls") is False,
        )
        stream = await self.client.chat.completions.create(**request)
        try:
            opened = False
            async for chunk in stream:
                if not opened:
                    opened = True
                    for event in translator.start():
                        yield event
                events = translator.feed(chunk)
                for event in events:
                    yield event
                if translator.first_call_whole:
                    break  # one call per round, and it is whole: the model's later output is never read
                if not events:
                    yield None
            if not opened:
                for event in translator.start():
                    yield event
            for event in translator.finish():
                yield event
        finally:
            await stream.close()

    async def close(self) -> None:
        """Close the HTTP client."""
        await self.client.close()
