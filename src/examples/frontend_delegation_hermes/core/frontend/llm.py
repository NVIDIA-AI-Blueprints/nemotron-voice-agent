# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A thin async chat client for the frontend model (supports ``tool_choice: required``)."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Protocol

from examples.frontend_delegation_hermes.core.config import FrontendLLMConfig


@dataclass(frozen=True, slots=True)
class ChatReply:
    """What the decider and verbalizer need from one completion."""

    content: str | None
    tool_calls: list[tuple[str, str]] = field(default_factory=list)
    usage: dict[str, int] = field(default_factory=dict)
    latency_ms: float = 0.0


class ChatModel(Protocol):
    """Anything that completes a chat (the OpenAI client, or a fake in tests)."""

    async def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] = "auto",
        parallel_tool_calls: bool | None = None,
    ) -> ChatReply:
        """One completion."""
        ...


class OpenAIChatModel:
    """OpenAI-compatible endpoint (NVIDIA Inference Hub, NIM, vLLM)."""

    def __init__(self, config: FrontendLLMConfig) -> None:
        """Create the underlying ``AsyncOpenAI`` client (shared by all sessions)."""
        from openai import AsyncOpenAI

        self._config = config
        self._client = AsyncOpenAI(
            api_key=config.api_key or "not-set", base_url=config.base_url, timeout=config.timeout_s
        )

    async def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] = "auto",
        parallel_tool_calls: bool | None = None,
    ) -> ChatReply:
        """Call the endpoint once."""
        request: dict[str, Any] = {
            "model": self._config.model,
            "messages": messages,
            "temperature": self._config.temperature,
            "max_tokens": self._config.max_tokens,
        }
        if tools:
            request["tools"] = tools
            request["tool_choice"] = tool_choice
            if parallel_tool_calls is not None:
                request["parallel_tool_calls"] = parallel_tool_calls
        if self._config.extra_body:
            request["extra_body"] = dict(self._config.extra_body)
        started = time.perf_counter()
        response = await self._client.chat.completions.create(**request)
        latency_ms = (time.perf_counter() - started) * 1000.0
        choice = response.choices[0]
        raw_usage = getattr(response, "usage", None)
        usage = {
            "input_tokens": int(getattr(raw_usage, "prompt_tokens", 0) or 0),
            "output_tokens": int(getattr(raw_usage, "completion_tokens", 0) or 0),
        }
        calls = [
            (str(call.function.name), str(call.function.arguments or "{}"))
            for call in (getattr(choice.message, "tool_calls", None) or [])
        ]
        return ChatReply(content=choice.message.content, tool_calls=calls, usage=usage, latency_ms=latency_ms)

    async def aclose(self) -> None:
        """Close the HTTP client."""
        await self._client.close()
