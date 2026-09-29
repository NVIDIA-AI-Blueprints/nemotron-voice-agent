# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""Recover tool calls that a model emitted as literal text.

Nemotron models sometimes stream a tool call as plain ``content`` in their
native XML form instead of as structured ``delta.tool_calls``::

    <tool_call>
    <function=call_backend>
    <parameter=query>
    Get the details of reservation ABC123.
    </parameter>
    </function>
    </tool_call>

Whether the OpenAI-compatible endpoint parses this into structured deltas
depends on the server-side tool-call parser, and on shared endpoints it is not
reliable. A dropped tool call is indistinguishable from a refusal downstream,
so callers harvest the text form back into structured deltas before any
validation or dispatch decision is made.
"""

from __future__ import annotations

import json
import re
import uuid
from typing import Any

from openai.types.chat.chat_completion_chunk import (
    ChatCompletionChunk,
    Choice,
    ChoiceDelta,
    ChoiceDeltaToolCall,
    ChoiceDeltaToolCallFunction,
)

__all__ = ["harvest_text_tool_calls", "parse_text_tool_calls"]

_TOOL_CALL_BLOCK_RE = re.compile(r"<tool_call>\s*(?P<body>.*?)\s*</tool_call>", re.DOTALL | re.IGNORECASE)
_FUNCTION_RE = re.compile(
    r"<function=(?P<name>[A-Za-z0-9_.\-]+)\s*>\s*(?P<body>.*?)\s*</function\s*>",
    re.DOTALL | re.IGNORECASE,
)
_PARAMETER_RE = re.compile(
    r"<parameter=(?P<name>[A-Za-z0-9_.\-]+)\s*>\s*(?P<value>.*?)\s*</parameter\s*>",
    re.DOTALL | re.IGNORECASE,
)
_JSON_SCALAR_RE = re.compile(r"^(?:true|false|null|-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)$")


def _coerce(value: str) -> Any:
    """Decode a parameter value, keeping anything ambiguous as text.

    Only unambiguous JSON is decoded, so identifiers such as ``ABC123`` and
    free-form queries survive verbatim while typed arguments still arrive with
    the type the tool schema expects.
    """
    stripped = value.strip()
    if not stripped:
        return ""
    if stripped[0] in "{[" or _JSON_SCALAR_RE.match(stripped):
        try:
            return json.loads(stripped)
        except ValueError:
            return value
    return value


def parse_text_tool_calls(text: str) -> list[tuple[str, dict[str, Any]]]:
    """Return ``(name, arguments)`` for every tool call encoded in ``text``."""
    calls: list[tuple[str, dict[str, Any]]] = []
    for block in _TOOL_CALL_BLOCK_RE.finditer(text):
        body = block.group("body")
        for function in _FUNCTION_RE.finditer(body):
            arguments = {
                parameter.group("name"): _coerce(parameter.group("value"))
                for parameter in _PARAMETER_RE.finditer(function.group("body"))
            }
            calls.append((function.group("name"), arguments))
        if _FUNCTION_RE.search(body) is not None:
            continue
        # Some builds emit a JSON object inside the block instead of the XML form.
        try:
            payload = json.loads(body)
        except ValueError:
            continue
        if isinstance(payload, dict) and isinstance(payload.get("name"), str):
            raw_arguments = payload.get("arguments", payload.get("parameters", {}))
            if isinstance(raw_arguments, str):
                try:
                    raw_arguments = json.loads(raw_arguments)
                except ValueError:
                    raw_arguments = {}
            calls.append((payload["name"], raw_arguments if isinstance(raw_arguments, dict) else {}))
    return calls


def _completion_text(chunks: list[ChatCompletionChunk]) -> str:
    parts: list[str] = []
    for chunk in chunks:
        choices = getattr(chunk, "choices", None)
        delta = getattr(choices[0], "delta", None) if choices else None
        content = getattr(delta, "content", None) if delta is not None else None
        if isinstance(content, str):
            parts.append(content)
    return "".join(parts)


def _has_structured_tool_call(chunk: ChatCompletionChunk) -> bool:
    choices = getattr(chunk, "choices", None)
    if not choices:
        return False
    delta = getattr(choices[0], "delta", None)
    return delta is not None and bool(getattr(delta, "tool_calls", None))


def harvest_text_tool_calls(chunks: list[ChatCompletionChunk]) -> list[ChatCompletionChunk]:
    """Rewrite text-encoded tool calls in ``chunks`` into structured deltas.

    Returns ``chunks`` unchanged when the endpoint already produced structured
    tool calls, or when the streamed text holds no tool call at all.
    """
    if not chunks or any(_has_structured_tool_call(chunk) for chunk in chunks):
        return chunks
    calls = parse_text_tool_calls(_completion_text(chunks))
    if not calls:
        return chunks

    tool_calls = [
        ChoiceDeltaToolCall(
            index=index,
            id=f"call-{uuid.uuid4()}",
            type="function",
            function=ChoiceDeltaToolCallFunction(name=name, arguments=json.dumps(arguments)),
        )
        for index, (name, arguments) in enumerate(calls)
    ]
    template = next((chunk for chunk in chunks if getattr(chunk, "choices", None)), chunks[-1])
    harvested = template.model_copy(
        update={
            "choices": [
                Choice(
                    index=0,
                    delta=ChoiceDelta(role="assistant", content=None, tool_calls=tool_calls),
                    finish_reason="tool_calls",
                )
            ],
            "usage": None,
        },
        deep=True,
    )
    # Preserve trailing usage-only chunks so token metrics survive the rewrite.
    trailing = [chunk for chunk in chunks if getattr(chunk, "usage", None) is not None]
    return [harvested, *trailing]
