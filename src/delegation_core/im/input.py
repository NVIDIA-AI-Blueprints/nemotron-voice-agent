# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""What the backend is told besides the conversation: each request's settings, the composed instructions, the records.

Instructions are composed in one order: the request's instructions or the default, then the voice note, then the
result format's note. The work records go in as ``input_text`` parts of a request's user message: the newest request at
a Work's start, and every steered request when it lands.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from copy import deepcopy
from dataclasses import dataclass, field

# The options a request may carry beside the model, the instructions, the tools, the tool choice and the output limit.
# The Responses backend passes them on; the chat-completions backend maps those it has an equivalent for.
OPTIONS = ("parallel_tool_calls", "reasoning", "text", "service_tier")
# The fields of a Responses request that set how its Work runs on the backend model.
SETTINGS = ("model", "instructions", "tools", "tool_choice", "max_output_tokens", *OPTIONS)


@dataclass(frozen=True)
class BackendTask:
    """What every model call of a Work's response shares: model, instructions, the app's functions and options."""

    model: str
    instructions: str
    tools: list[dict]
    tool_choice: str | dict = "auto"
    max_output_tokens: int | None = None
    # The ``OPTIONS`` the client set.
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
    """What a Responses request set of the fields in ``SETTINGS``, copied; the rest of the request is left out."""
    return {key: deepcopy(params[key]) for key in SETTINGS if key in params}


def settings_for(settings: dict, defaults: BackendTask) -> BackendTask:
    """The settings record of a request: what it set (``request_settings``), and ``defaults`` for what it left out.

    An empty model or empty instructions count as left out. A request that sets tools runs with exactly those, none
    included; one that sets none runs with the default tools.
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
    """``instructions`` plus ``note`` on a line of its own; ``instructions`` unchanged when the note is empty."""
    note = note.strip()
    if not note:
        return instructions
    return ((instructions or "").rstrip() + "\n" + note).strip()


def compose(instructions: str | None, *notes: str) -> str | None:
    """The composed instructions: ``instructions`` (the request's or the default), then each note, in order."""
    for note in notes:
        instructions = _with_note(instructions, note)
    return instructions


def task_request(settings: BackendTask, *notes: str) -> dict:
    """The request skeleton of a task's next model call: ``settings``, its instructions composed with ``notes``."""
    request = settings.payload()
    instructions = compose(request.get("instructions"), *notes)
    return request if instructions is None else {**request, "instructions": instructions}


# ------------------------------------------------------------------------------------------------ input items
def is_user_message(item: dict) -> bool:
    """Whether an input item is a user message (a typed one may leave its ``type`` out)."""
    return item.get("role") == "user" and item.get("type", "message") == "message"


def user_message(text: str) -> dict:
    """A user message whose one ``input_text`` part holds ``text``."""
    return {"type": "message", "role": "user", "content": [{"type": "input_text", "text": text}]}


def with_parts(items: list[dict], *texts: str) -> list[dict]:
    """``items`` with each non-empty text added as an ``input_text`` part of the last user message, copied.

    The parts stay inside the request's own message, so a trim keeps or drops them with it. Items with no user message
    are returned unchanged.
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
    """The work records as a request's extra parts; none when there is no record.

    The parts are ``texts.records_block`` with one JSON line per Work record, then ``texts.closing_line``.
    """
    lines = [json.dumps(record, ensure_ascii=False) for record in records]
    return (template.format(records="\n".join(lines)), closing) if lines else ()
