# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Chat-client protocol and the OpenAI-compatible implementation.

``ChatClient`` is the seam the future tau2-bench adapter replaces with a
``tau2.utils.llm_utils.generate()``-backed client, so nothing else in the
package needs to know which provider is in play.
"""

from __future__ import annotations

import json
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from examples.frontend_backend_verdict.text.config import AccountingConfig, LLMConfig
from examples.frontend_backend_verdict.text.messages import Message, ToolCall, Usage


@dataclass(frozen=True, slots=True)
class ChatResponse:
    """One LLM completion plus its accounting."""

    content: str | None = None
    tool_calls: tuple[ToolCall, ...] = ()
    usage: Usage = field(default_factory=Usage)
    cost: float | None = None
    latency_ms: float = 0.0
    model: str = ""
    finish_reason: str = ""


@runtime_checkable
class ChatClient(Protocol):
    """Minimal async chat-completion interface."""

    async def complete(
        self,
        *,
        messages: Sequence[Message],
        tools: Sequence[dict[str, Any]] | None = None,
    ) -> ChatResponse:
        """Return one completion for ``messages``."""


def to_wire_messages(messages: Sequence[Message]) -> list[dict[str, Any]]:
    """Convert internal messages to OpenAI chat-completion payloads."""
    wire: list[dict[str, Any]] = []
    for message in messages:
        if message.role == "tool":
            wire.append({"role": "tool", "tool_call_id": message.tool_call_id, "content": message.content or ""})
            continue
        payload: dict[str, Any] = {"role": message.role, "content": message.content or ""}
        if message.tool_calls:
            payload["content"] = message.content or None
            payload["tool_calls"] = [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {"name": call.name, "arguments": call.arguments_json},
                }
                for call in message.tool_calls
            ]
        wire.append(payload)
    return wire


def compute_cost(model: str, usage: Usage, accounting: AccountingConfig) -> float | None:
    """Return the local cost for ``usage``, or ``None`` when no price is configured."""
    prices = accounting.pricing.get(model)
    if not prices:
        return None
    prompt_rate = prices.get("prompt_per_1k", 0.0)
    completion_rate = prices.get("completion_per_1k", 0.0)
    return (usage.prompt_tokens / 1000.0) * prompt_rate + (usage.completion_tokens / 1000.0) * completion_rate


class OpenAIChatClient:
    """Async client for any OpenAI-compatible endpoint (NVIDIA NIM included)."""

    def __init__(self, config: LLMConfig, *, accounting: AccountingConfig | None = None) -> None:
        """Create the underlying ``AsyncOpenAI`` client."""
        from openai import AsyncOpenAI

        self._config = config
        self._accounting = accounting or AccountingConfig()
        self._client = AsyncOpenAI(
            api_key=config.api_key or "not-set",
            base_url=config.base_url,
            timeout=config.timeout_seconds,
        )

    async def complete(
        self,
        *,
        messages: Sequence[Message],
        tools: Sequence[dict[str, Any]] | None = None,
    ) -> ChatResponse:
        """Call the endpoint and normalize the response."""
        request: dict[str, Any] = {
            "model": self._config.model,
            "messages": to_wire_messages(messages),
            "temperature": self._config.temperature,
            "max_tokens": self._config.max_tokens,
        }
        if tools:
            request["tools"] = list(tools)
            request["tool_choice"] = "auto"
        if self._config.extra_body:
            request["extra_body"] = dict(self._config.extra_body)
        started = time.perf_counter()
        response = await self._client.chat.completions.create(**request)
        latency_ms = (time.perf_counter() - started) * 1000.0
        return self._to_chat_response(response, latency_ms)

    def _to_chat_response(self, response: Any, latency_ms: float) -> ChatResponse:
        choice = response.choices[0]
        raw_usage = getattr(response, "usage", None)
        usage = Usage(
            prompt_tokens=int(getattr(raw_usage, "prompt_tokens", 0) or 0),
            completion_tokens=int(getattr(raw_usage, "completion_tokens", 0) or 0),
            total_tokens=int(getattr(raw_usage, "total_tokens", 0) or 0),
        )
        tool_calls = tuple(
            ToolCall(
                id=str(getattr(call, "id", "") or ""),
                name=str(call.function.name),
                arguments_json=_normalize_arguments_json(call.function.arguments),
            )
            for call in (getattr(choice.message, "tool_calls", None) or [])
        )
        return ChatResponse(
            content=choice.message.content,
            tool_calls=tool_calls,
            usage=usage,
            cost=compute_cost(self._config.model, usage, self._accounting),
            latency_ms=latency_ms,
            model=str(getattr(response, "model", self._config.model)),
            finish_reason=str(getattr(choice, "finish_reason", "") or ""),
        )


def _normalize_arguments_json(raw: Any) -> str:
    """Return tool-call arguments as a JSON object string."""
    if isinstance(raw, dict):
        return json.dumps(raw, sort_keys=True, ensure_ascii=False)
    text = str(raw or "{}").strip() or "{}"
    try:
        decoded = json.loads(text)
    except json.JSONDecodeError:
        return "{}"
    return json.dumps(decoded, sort_keys=True, ensure_ascii=False) if isinstance(decoded, dict) else "{}"
