# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D103, D105, D107

"""Shared doubles of the delegation core's tests. They import only the core.

The core's tests (``tests/unit/delegation_core/``) import only the core and this module, because the core is its own
package; the tests of the other parts import from here too.
"""

from __future__ import annotations

import asyncio
import json

import httpx
from openai import APITimeoutError
from openai.types.chat import ChatCompletionChunk

from delegation_core.backend.providers import ProviderLimits
from delegation_core.config import ModelEndpoint
from delegation_core.im.stream import message_text


def answer_text(event: dict) -> str | None:
    """The answer a terminal event carries, or ``None``.

    The answer is the text of the last completed ``final_answer`` message of a ``response.completed`` whose output
    has no function call.
    """
    if event.get("type") != "response.completed":
        return None
    output = (event.get("response") or {}).get("output") or []
    if any(item.get("type") == "function_call" for item in output):
        return None
    messages = [
        item
        for item in output
        if item.get("type") == "message"
        and item.get("status") == "completed"
        and item.get("phase", "final_answer") == "final_answer"
    ]
    return message_text(messages[-1]) if messages else None


# ------------------------------------------------------------------- a scripted backend model and server tools
# The core's endpoint and provider limits for these doubles, in the core's own types.
ENDPOINT = ModelEndpoint(model="test/model", base_url="http://localhost:1/v1")
LIMITS = ProviderLimits(provider_retries=1, first_event_timeout_seconds=2.0, idle_timeout_seconds=2.0)


def _text_round(text: str, response_id: str = "resp") -> list[dict]:
    """A backend round that answers ``text``: one completed message, as a provider's translation writes it."""
    item = {
        "id": f"msg_{response_id}",
        "type": "message",
        "role": "assistant",
        "status": "completed",
        "content": [{"type": "output_text", "text": text, "annotations": []}],
    }
    return [
        {"type": "response.created", "response": {"id": response_id, "status": "in_progress"}},
        {"type": "response.output_item.done", "output_index": 0, "item": item},
        {"type": "response.completed", "response": {"id": response_id, "status": "completed", "output": [item]}},
    ]


def _call_round(name: str, arguments: dict, call_id: str = "call_1") -> list[dict]:
    """A backend round that calls ``name``: one function call, as a provider's translation writes it."""
    item = {
        "id": f"fc_{call_id}",
        "type": "function_call",
        "status": "completed",
        "call_id": call_id,
        "name": name,
        "arguments": json.dumps(arguments),
    }
    return [
        {"type": "response.created", "response": {"id": "resp_call", "status": "in_progress"}},
        {"type": "response.output_item.done", "output_index": 0, "item": item},
        {"type": "response.completed", "response": {"id": "resp_call", "status": "completed", "output": [item]}},
    ]


class RecordingTools:
    def __init__(self, result='{"ok": true}'):
        self.result, self.calls = result, []

    async def execute(self, call_id, name, arguments):
        self.calls.append((call_id, name, arguments))
        return self.result


class SlowTools(RecordingTools):
    """Tool execution that waits until released, so a task stays in its tool step."""

    def __init__(self):
        super().__init__('{"ok": true}')
        self.gate = asyncio.Event()

    async def execute(self, call_id, name, arguments):
        await self.gate.wait()
        return await super().execute(call_id, name, arguments)


# ------------------------------------------------------------------------ a scripted chat-completions client
class FakeStream:
    def __init__(self, chunks):
        self.chunks, self.closed = chunks, False

    def __aiter__(self):
        return self._iterate()

    async def _iterate(self):
        for item in self.chunks:
            yield item

    async def close(self):
        self.closed = True


class FakeChat:
    """Scripted ``chat.completions.create``: each entry is a chunk list or an exception."""

    def __init__(self, *script):
        self.script, self.requests, self.closed = list(script), [], False
        self.chat = self
        self.completions = self

    async def create(self, **kwargs):
        self.requests.append(kwargs)
        step = self.script.pop(0)
        if isinstance(step, Exception):
            raise step
        return FakeStream(step)

    async def close(self):
        self.closed = True


def chunk(content=None, tool_calls=None, finish=None, usage=None, **delta_extra):
    delta = {"role": "assistant", **delta_extra}
    if content is not None:
        delta["content"] = content
    if tool_calls is not None:
        delta["tool_calls"] = tool_calls
    body = {
        "id": "chatcmpl-1",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": "nvidia/test",
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
    }
    if usage:
        body["usage"] = usage
    return ChatCompletionChunk.model_validate(body)


def tool_fragment(index, call_id=None, name=None, arguments=None):
    function = {k: v for k, v in {"name": name, "arguments": arguments}.items() if v is not None}
    fragment = {"index": index, "type": "function", "function": function}
    if call_id:
        fragment["id"] = call_id
    return fragment


def _timeout():
    return APITimeoutError(request=httpx.Request("POST", "https://hub.example/v1"))
