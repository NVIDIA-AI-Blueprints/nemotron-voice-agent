# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Chat-completions talker: a strict ``{action, speech}`` JSON decision from any OpenAI-compatible endpoint.

Capability negotiation for ``response_format``: a 400 naming the format steps down to a weaker format and
the same decision request is issued again. No tool has run at that stage, so nothing can repeat. Retries
are bounded to complete decisions.
"""

from __future__ import annotations

import json
import re
import time

from loguru import logger
from openai import AsyncOpenAI, BadRequestError

from examples.frontend_backend_live.common.retry import current_attempt

from ...config_manager.schema import ModelEndpoint, ReliabilityConfig
from ..base import Frontend
from ..decision import TalkerDecision, decision_from_output_text, decision_schema, unusable_decision
from ..registry import FRONTENDS
from .translation import strip_reasoning

JSON_RULES = """
Reply with a single JSON object and nothing else: exactly the keys "action" and
"speech". No markdown, no code fences, no reasoning text before or after the object.
"""
FORMAT_ORDER = ("json_schema", "json_object", "none")
FORMAT_ERROR_HINTS = ("response_format", "json_schema", "json_object", "guided", "structured")
FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.I)
JSON_OBJECT = re.compile(r"\{.*\}", re.S)
DEFAULT_SAMPLING = {"temperature": 0.0, "max_tokens": 512}


def extract_json(text: str) -> str:
    """Tolerate fences and stray prose around the object; validation stays strict."""
    text = FENCE.sub("", strip_reasoning(text).strip()).strip()
    try:
        json.loads(text)
        return text
    except ValueError:
        match = JSON_OBJECT.search(text)
        return match.group(0) if match else text


def chat_messages(instructions: str, history: list[dict]) -> list[dict]:
    """Convert voice history to chat messages. Developer context joins the system message."""
    system = [instructions + "\n" + JSON_RULES]
    messages: list[dict] = []
    for entry in history:
        content = entry["content"]
        if not isinstance(content, str):
            content = "\n".join(p.get("text", "") for p in content if isinstance(p, dict))
        role = entry["role"]
        if role in ("developer", "system"):
            system.append(content)
        elif messages and messages[-1]["role"] == role:
            messages[-1]["content"] += "\n" + content
        else:
            messages.append({"role": role, "content": content})
    return [{"role": "system", "content": "\n\n".join(system)}, *messages]


@FRONTENDS.register("chat-completions")
class ChatCompletionsFrontend(Frontend):
    """Talker on an OpenAI-compatible chat-completions endpoint."""

    def __init__(self, endpoint: ModelEndpoint, reliability: ReliabilityConfig, client: AsyncOpenAI | None = None):
        """Create the client (unless injected) and read the endpoint's response format."""
        super().__init__(endpoint, reliability)
        self.client = client or AsyncOpenAI(
            api_key=endpoint.api_key or "not-needed",
            base_url=endpoint.base_url,
            timeout=reliability.api_timeout_seconds,
            max_retries=0,
        )
        self.format_level = FORMAT_ORDER.index(endpoint.response_format)

    async def complete(self, instructions, history, actions=("speak", "delegate"), max_output_tokens=400):
        """Return a decision, stepping ``response_format`` down once if the endpoint rejects it."""
        while True:
            level = FORMAT_ORDER[self.format_level]
            try:
                return await super().complete(instructions, history, actions, max_output_tokens)
            except BadRequestError as exc:
                rejected = any(hint in str(exc).lower() for hint in FORMAT_ERROR_HINTS)
                if not rejected or self.format_level >= len(FORMAT_ORDER) - 1:
                    raise
                self.format_level += 1
                logger.warning(
                    f"Talker endpoint rejected response_format={level}; using {FORMAT_ORDER[self.format_level]}"
                )

    async def complete_once(self, instructions, history, actions, max_output_tokens):
        """Make one attempt at the current ``response_format`` level."""
        messages = chat_messages(instructions, history)
        return await self.request_decision(messages, actions, FORMAT_ORDER[self.format_level])

    async def request_decision(self, messages: list[dict], actions, level: str) -> TalkerDecision:
        """Send one decision request and validate the reply."""
        request = {
            **DEFAULT_SAMPLING,
            **self.endpoint.extra_params,
            "model": self.endpoint.model,
            "messages": messages,
            "stream": True,
        }
        if level == "json_schema":
            request["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "voice_decision", "strict": True, "schema": decision_schema(actions)},
            }
        elif level == "json_object":
            request["response_format"] = {"type": "json_object"}

        started = time.monotonic()
        stream = None
        parts: list[str] = []
        completion_id = finish_reason = first_token_ms = None
        try:
            stream = await self.client.chat.completions.create(**request)
            async for chunk in stream:
                completion_id = completion_id or getattr(chunk, "id", None)
                if not chunk.choices:
                    continue
                choice = chunk.choices[0]
                if choice.delta.content:
                    if first_token_ms is None:
                        first_token_ms = round((time.monotonic() - started) * 1000, 3)
                    parts.append(choice.delta.content)
                finish_reason = choice.finish_reason or finish_reason
        finally:
            if stream is not None:
                await stream.close()
        text = "".join(parts)
        metadata = {
            "response_id": completion_id,
            "response_status": "incomplete" if finish_reason == "length" else "completed",
            "incomplete_reason": "max_output_tokens" if finish_reason == "length" else None,
            "output_types": ["message"] if text else [],
            "output_text_chars": len(text),
            "first_token_ms": first_token_ms,
            "response_format": level,
            "attempt": current_attempt.get(),
        }
        if finish_reason == "length":
            return unusable_decision(metadata, "incomplete_output")
        return decision_from_output_text(extract_json(text), metadata)

    async def close(self) -> None:
        """Close the HTTP client."""
        await self.client.close()
