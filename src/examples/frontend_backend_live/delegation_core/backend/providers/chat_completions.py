# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Chat-completions backend: function tools on any OpenAI-compatible endpoint, as a Responses event stream."""

from __future__ import annotations

from collections.abc import AsyncIterator, Mapping
from copy import deepcopy
from typing import Any

from loguru import logger
from openai import AsyncOpenAI

from examples.shared.delegation_contract import CHAT_COMPLETIONS, ModelEndpoint

from .base import Backend, ProviderLimits, apply_thinking, client_run, make_client
from .translation import (
    ChatResponseTranslator,
    responses_items_to_messages,
    to_chat_response_format,
    to_chat_tool_choice,
    to_chat_tools,
)

DEFAULT_SAMPLING = {"temperature": 0.7, "max_tokens": 1024}
PLAIN_TIERS = (None, "auto", "default")


class ChatCompletionsBackend(Backend):
    """Backend on an OpenAI-compatible chat-completions endpoint."""

    name = CHAT_COMPLETIONS

    def __init__(self, endpoint: ModelEndpoint, limits: ProviderLimits, client: AsyncOpenAI | None = None):
        """Bind the endpoint; make its client unless one is injected."""
        super().__init__(endpoint, limits)
        self.client = client or make_client(endpoint, limits)

    @classmethod
    def unsupported(cls, settings: Mapping[str, Any]) -> list[tuple[str, str]]:
        """Return what a chat endpoint cannot honor: any tool but a function, a service tier, ``text.verbosity``.

        A tool the client runs gets every provider's refusal, and a hosted tool needs a Responses backend.
        ``reasoning`` is never refused: the endpoint's settings set the model's reasoning, so only an effort of
        ``none`` counts.
        """
        problems = super().unsupported(settings)
        hosted = [
            str(tool.get("type"))
            for tool in settings.get("tools") or []
            if tool.get("type") != "function" and not client_run(tool)
        ]
        problems += [
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
        apply_thinking(payload, request.get("extra_body"))
        response_format = to_chat_response_format((payload.get("text") or {}).get("format"))
        if response_format is not None:
            request["response_format"] = response_format
        return request

    async def _events(self, payload: dict) -> AsyncIterator[dict | None]:
        """Open the round, then yield its events; the opening ones wait for the first chunk, so the retries cover it.

        A chunk with no event yet, such as reasoning, gives ``None``, so a long but steady thinking round is never cut.
        """
        request = self.request(payload)
        translator = ChatResponseTranslator(request["model"], single_call=payload.get("parallel_tool_calls") is False)
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
                    break
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
