# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stored conversation rows -> the request's ``messages``, in the shape Hermes sends (llm-backend-plan.md 6).

``LLMAgent`` stores rows already in wire form (canonical tool arguments, stripped content, no
reasoning), so this only prepends the system message, keeps the request keys, sends an empty
non-final user or assistant message as the placeholder, and merges adjacent user messages.
Stdlib only.
"""

from __future__ import annotations

import re
from typing import Any

_KEYS = {
    "user": ("role", "content"),
    "assistant": ("role", "content", "tool_calls"),
    "tool": ("role", "tool_call_id", "content"),
}
_THINK = re.compile(r"<think>.*?(?:</think>|$)", re.DOTALL)


def strip_think(text: str) -> str:
    """``text`` without ``<think>`` blocks (an unclosed block runs to the end) and without a leading ``</think>``."""
    text = _THINK.sub("", text or "")
    if "</think>" in text:
        text = text.split("</think>", 1)[1]
    return text


def build(system: str, rows: list[dict[str, Any]], placeholder: str) -> list[dict[str, Any]]:
    """The request messages for ``rows`` (the stored conversation)."""
    out: list[dict[str, Any]] = [{"role": "system", "content": system}] if system else []
    last = len(rows) - 1
    for index, row in enumerate(rows):
        role = row.get("role")
        keys = _KEYS.get(str(role))
        if keys is None:
            continue
        message = {key: row[key] for key in keys if key in row and row[key] not in (None, [])}
        content = message.get("content")
        message["content"] = content.strip() if isinstance(content, str) else ("" if content is None else content)
        if role in ("user", "assistant") and not message["content"] and not message.get("tool_calls") and index < last:
            message["content"] = placeholder
        previous = out[-1] if out else None
        if role == "user" and previous is not None and previous["role"] == "user":
            previous["content"] = f"{previous['content']}\n\n{message['content']}"
            continue
        out.append(message)
    return out
