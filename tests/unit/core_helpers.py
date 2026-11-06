# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D103, D105, D107

"""Shared doubles of the delegation core's tests. They import only the core and the helpers it shares.

The core's tests (``tests/unit/delegation_core/``) import only the core, the helpers it shares (``examples.shared``)
and this module, because the core is its own package; the tests of the other parts import from here too.
"""

from __future__ import annotations

from copy import deepcopy
from importlib.util import find_spec
from pathlib import Path

import httpx
import yaml
from openai import APITimeoutError
from openai.types.chat import ChatCompletionChunk

from examples.frontend_backend_live.delegation_core.backend.harness.source import GROUP_PACKAGES
from examples.frontend_backend_live.delegation_core.backend.providers import ProviderLimits
from examples.frontend_backend_live.delegation_core.items import message_text
from examples.shared.delegation_contract import ModelEndpoint

# The core's texts as the example's prompts.yaml words them (agent_prompts.Core), which a root hands to
# CoreSettings.parse with the core's settings. They are read here as a root reads them, so the core's tests load no
# module of the frontend.
PROMPTS_YAML = Path(__file__).resolve().parents[2] / "src/examples/frontend_backend_live/prompts.yaml"
CORE_TEXTS = {
    name: entry["content"]
    for name, entry in yaml.safe_load(PROMPTS_YAML.read_text(encoding="utf-8"))["agent_prompts"]["Core"].items()
}
# The example's own settings as examples_registry.yaml holds them (the config block of its entry), and the core's
# settings among them, read from the file as the registry reads it.
REGISTRY_YAML = Path(__file__).resolve().parents[2] / "examples_registry.yaml"
REGISTRY = yaml.safe_load(REGISTRY_YAML.read_text(encoding="utf-8"))
EXAMPLE_SETTINGS = REGISTRY["examples"]["frontend-backend-live"]["config"]
CORE_BLOCK = EXAMPLE_SETTINGS["core"]
# Whether the packages of the harness dependency group are installed: the tests that load the harness need them, and
# are skipped without them.
HARNESS_GROUP = all(find_spec(name) for name in GROUP_PACKAGES)
NO_HARNESS_GROUP = "the harness dependency group is not installed: uv sync --group harness"


def merged(base, changes):
    """``changes`` over ``base``, as the example merges settings: mappings key by key, anything else replaces."""
    if not isinstance(base, dict) or not isinstance(changes, dict):
        return deepcopy(changes)
    result = deepcopy(base)
    for key, value in changes.items():
        result[key] = merged(result[key], value) if key in result else deepcopy(value)
    return result


def core_block(**changes) -> dict:
    """The core's settings of the registry's block, with ``changes`` (its top-level blocks) merged over them."""
    return merged(CORE_BLOCK, changes)


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


# ---------------------------------------------------------------------------------- a scripted backend model
# The core's endpoint and provider limits for these doubles, in the core's own types.
ENDPOINT = ModelEndpoint(model="test/model", base_url="http://localhost:1/v1")
LIMITS = ProviderLimits(provider_retries=1, first_event_timeout_seconds=2.0, idle_timeout_seconds=2.0)


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
