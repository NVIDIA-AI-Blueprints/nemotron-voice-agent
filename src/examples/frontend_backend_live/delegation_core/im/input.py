# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""What the backend is told besides the conversation: the settings, the composed instructions and the records."""

from __future__ import annotations

import json
from collections.abc import Iterable
from copy import deepcopy
from dataclasses import dataclass, field

from examples.shared.delegation_contract import OPTIONS

SETTINGS = ("model", "instructions", "tools", "tool_choice", "max_output_tokens", *OPTIONS)


@dataclass(frozen=True)
class BackendTask:
    """What every model call of a Work's response shares: model, instructions, the app's functions and options."""

    model: str
    instructions: str
    tools: list[dict]
    tool_choice: str | dict = "auto"
    max_output_tokens: int | None = None
    options: dict = field(default_factory=dict)

    def payload(self) -> dict:
        """Return the Responses-style request skeleton for one model call."""
        payload = {
            "model": self.model,
            "instructions": self.instructions,
            "tools": deepcopy(self.tools),
            "tool_choice": self.tool_choice,
        }
        if self.max_output_tokens:
            payload["max_output_tokens"] = self.max_output_tokens
        payload.update(deepcopy(self.options))
        return payload


def request_settings(params: dict) -> dict:
    """Return a copy of what a Responses request set of the ``SETTINGS`` fields."""
    return {key: deepcopy(params[key]) for key in SETTINGS if key in params}


def settings_for(settings: dict, defaults: BackendTask) -> BackendTask:
    """Return the settings record of a request: what it set, and ``defaults`` for what it left out.

    An empty model or empty instructions count as left out; tools it sets, an empty list too, replace the default tools.
    """
    return BackendTask(
        model=settings.get("model") or defaults.model,
        instructions=settings.get("instructions") or defaults.instructions,
        tools=deepcopy(settings.get("tools", defaults.tools)),
        tool_choice=deepcopy(settings.get("tool_choice", defaults.tool_choice)),
        max_output_tokens=settings.get("max_output_tokens", defaults.max_output_tokens),
        options={key: deepcopy(settings[key]) for key in OPTIONS if key in settings} or deepcopy(defaults.options),
    )


def _with_note(instructions: str | None, note: str) -> str | None:
    """Return ``instructions`` plus ``note`` on a line of its own, unchanged when the note is empty."""
    note = note.strip()
    if not note:
        return instructions
    return ((instructions or "").rstrip() + "\n" + note).strip()


def compose(instructions: str | None, *notes: str) -> str | None:
    """Return the composed instructions: ``instructions`` (the request's or the default), then each note, in order."""
    for note in notes:
        instructions = _with_note(instructions, note)
    return instructions


def task_request(settings: BackendTask, *notes: str) -> dict:
    """Return the skeleton of a task's next model call: ``settings``, its instructions composed with ``notes``."""
    request = settings.payload()
    instructions = compose(request.get("instructions"), *notes)
    return request if instructions is None else {**request, "instructions": instructions}


def is_user_message(item: dict) -> bool:
    """Return whether an input item is a user message (a typed one may leave its ``type`` out)."""
    return item.get("role") == "user" and item.get("type", "message") == "message"


def user_message(text: str) -> dict:
    """Return a user message whose one ``input_text`` part holds ``text``."""
    return {"type": "message", "role": "user", "content": [{"type": "input_text", "text": text}]}


def with_parts(items: list[dict], *texts: str) -> list[dict]:
    """Return ``items`` with each non-empty text as an ``input_text`` part of the last user message, copied if changed.

    The parts stay inside the request's own message, so a trim keeps or drops them with it.
    """
    parts = [{"type": "input_text", "text": text} for text in texts if text]
    if not parts:
        return items
    for index in range(len(items) - 1, -1, -1):
        if is_user_message(items[index]):
            items = deepcopy(items)
            content = items[index].get("content")
            text = [{"type": "input_text", "text": content}] if isinstance(content, str) else list(content or [])
            items[index]["content"] = text + parts
            return items
    return items


def records_parts(template: str, closing: str, records: Iterable[dict]) -> tuple[str, ...]:
    """Return the records block, one JSON line per Work, and the closing line as extra parts; none with no record."""
    lines = [json.dumps(record, ensure_ascii=False) for record in records]
    return (template.format(records="\n".join(lines)), closing) if lines else ()
