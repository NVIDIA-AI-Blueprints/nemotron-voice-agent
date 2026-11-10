# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D103, D105, D107

"""Doubles that the delegation core's tests share with the other parts' tests. They import only the core."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping
from copy import deepcopy
from importlib.util import find_spec
from pathlib import Path

import httpx
import yaml
from openai import APITimeoutError
from openai.types.chat import ChatCompletionChunk

from examples.frontend_backend_live.delegation_core.backend.harness.source import GROUP_PACKAGES
from examples.frontend_backend_live.delegation_core.backend.port import message_phase

ROOT = Path(__file__).resolve().parents[2]
PROMPTS = yaml.safe_load((ROOT / "src/examples/frontend_backend_live/prompts.yaml").read_text(encoding="utf-8"))
CORE_TEXTS = {name: entry["content"] for name, entry in PROMPTS["agent_prompts"]["Core"].items()}
REGISTRY = yaml.safe_load((ROOT / "examples_registry.yaml").read_text(encoding="utf-8"))
EXAMPLE_SETTINGS = REGISTRY["examples"]["frontend-backend-live"]["config"]
CORE_BLOCK = EXAMPLE_SETTINGS["core"]
HARNESS_GROUP = all(find_spec(name) for name in GROUP_PACKAGES)
NO_HARNESS_GROUP = "the harness dependency group is not installed: uv sync --group harness"
ENDINGS = Path(__file__).parent / "fixtures" / "delegation_endings.json"


def merged(base, changes):
    """Merge ``changes`` over ``base`` as the example merges settings: mappings key by key, anything else replaced."""
    if not isinstance(base, dict) or not isinstance(changes, dict):
        return deepcopy(changes)
    result = deepcopy(base)
    for key, value in changes.items():
        result[key] = merged(result[key], value) if key in result else deepcopy(value)
    return result


def core_block(**changes) -> dict:
    """Return the registry's core settings with ``changes`` (its top-level blocks) merged over them."""
    return merged(CORE_BLOCK, changes)


class Sleep:
    def __init__(self, seconds):
        self.seconds = seconds


class FakeStream:
    """A model client's stream: a :class:`Sleep` in it pauses, and an exception in it is an error event."""

    def __init__(self, chunks):
        self.chunks, self.closed = chunks, False

    def __aiter__(self):
        return self._iterate()

    async def _iterate(self):
        for item in self.chunks:
            if isinstance(item, Sleep):
                await asyncio.sleep(item.seconds)
            elif isinstance(item, Exception):
                raise item
            else:
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


def load() -> dict[str, dict]:
    """Return the endings table's scenarios by name."""
    return json.loads(ENDINGS.read_text(encoding="utf-8"))["scenarios"]


def normal_form(event: dict, labels: dict[str, str], texts: Mapping[str, str] | None = None) -> str | None:
    """Return one official event as the table writes it, or ``None`` for an event the table does not compare.

    ``labels`` maps each response id to its label (``R1``, ``R2`` ... in order of first appearance); ``texts`` maps the
    names of ``texts`` lines to their words, so a message that carries one is marked ``[name]``.
    """
    kind = event.get("type", "")
    response = event.get("response") if isinstance(event.get("response"), dict) else {}
    if response.get("id") and response["id"] not in labels:
        labels[response["id"]] = f"R{len(labels) + 1}"
    if kind == "response.created":
        return f"created {response.get('status')}"
    if kind == "response.queued":
        return "queued"
    if kind == "response.in_progress":
        previous = response.get("previous_response_id")
        return "in_progress" + (f" previous={labels.get(previous, previous)}" if previous else "")
    if kind.startswith("response.mcp_call.") and kind.split(".")[-1] in ("in_progress", "completed", "failed"):
        return "mcp_call." + kind.split(".")[-1]
    if kind == "response.output_item.done":
        item = event.get("item") or {}
        if item.get("type") == "message":
            text = "".join(part.get("text", "") for part in item.get("content") or [] if isinstance(part, dict))
            names = [name for name, words in (texts or {}).items() if words and words == text]
            return f"message {message_phase(item)}" + (f" [{names[0]}]" if names else "")
        if item.get("type") == "mcp_call":
            return f"mcp_call {item.get('server_label')}/{item.get('name')} {item.get('status')}"
        return str(item.get("type"))
    if kind == "response.completed":
        return "completed"
    if kind == "response.failed":
        return f"failed {(response.get('error') or {}).get('code')}"
    if kind == "response.incomplete":
        if response.get("status") == "cancelled":
            return "incomplete cancelled"
        return f"incomplete {(response.get('incomplete_details') or {}).get('reason')}"
    return None


def normalized(events: list[tuple[str, dict]], texts: Mapping[str, str] | None = None) -> list[list[str]]:
    """Return events given as ``(delegation_id, event)`` pairs as a table timeline of ``[label, event]`` pairs."""
    labels: dict[str, str] = {}
    timeline = []
    for _, event in events:
        form = normal_form(event, labels, texts)
        response = event.get("response") if isinstance(event.get("response"), dict) else {}
        label = labels.get(response.get("id")) or labels.get(event.get("response_id"))
        if form is not None and label:
            timeline.append([label, form])
    return timeline


def official_events(scenario: dict, texts: Mapping[str, str] | None = None) -> list[tuple[str, dict]]:
    """Build the scenario's official events, ``(delegation_id, event)`` in order, to replay into a reader.

    Every event carries ``response_id`` besides its official fields; a message marked ``[key]`` carries that line of
    ``texts`` (or the key itself).
    """
    ids = {label: f"resp_{label.lower()}" for label in scenario["responses"]}
    outputs: dict[str, list[dict]] = {label: [] for label in scenario["responses"]}
    events = []
    for label, form in scenario["timeline"]:
        snapshot = {"id": ids[label], "object": "response", "output": []}
        words = form.split(" ")
        if words[0] == "created":
            event = {"type": "response.created", "response": {**snapshot, "status": words[1]}}
        elif form == "queued":
            event = {"type": "response.queued", "response": {**snapshot, "status": "queued"}}
        elif words[0] == "in_progress":
            event = {"type": "response.in_progress", "response": {**snapshot, "status": "in_progress"}}
            if len(words) > 1:
                event["response"]["previous_response_id"] = ids[words[1].removeprefix("previous=")]
        elif words[0].startswith("mcp_call."):
            event = {"type": f"response.{words[0]}", "output_index": len(outputs[label])}
        elif words[0] in ("message", "function_call", "mcp_call"):
            outputs[label].append(_item(words, len(outputs[label]), texts))
            event = {"type": "response.output_item.done", "output_index": len(outputs[label]) - 1}
            event["item"] = outputs[label][-1]
        elif form == "completed":
            event = {"type": "response.completed", "response": {**snapshot, "status": "completed"}}
            event["response"]["output"] = outputs[label]
        elif words[0] == "failed":
            error = {"code": words[1], "message": "The backend failed."}
            event = {"type": "response.failed", "response": {**snapshot, "status": "failed", "error": error}}
        elif form == "incomplete cancelled":
            cancelled = {**snapshot, "status": "cancelled", "incomplete_details": None}
            event = {"type": "response.incomplete", "response": cancelled}
        else:
            cut = {**snapshot, "status": "incomplete", "incomplete_details": {"reason": words[1]}}
            event = {"type": "response.incomplete", "response": cut}
        events.append((scenario["responses"][label], {**event, "response_id": ids[label]}))
    return events


def _item(words: list[str], index: int, texts: Mapping[str, str] | None) -> dict:
    item = {"id": f"item_{index}", "type": words[0], "status": "completed"}
    if words[0] == "message":
        key = words[2].strip("[]") if len(words) > 2 else None
        text = (texts or {}).get(key, key) if key else ("Done." if words[1] == "final_answer" else "Working on it.")
        content = [{"type": "output_text", "text": text, "annotations": []}]
        return {**item, "role": "assistant", "phase": words[1], "content": content}
    if words[0] == "function_call":
        return {**item, "call_id": f"call_{index}", "name": "get_menu", "arguments": "{}"}
    label, name = words[1].split("/")
    call = {"server_label": label, "name": name, "arguments": "{}", "output": "{}", "error": None}
    return {**item, **call, "status": words[2]}
