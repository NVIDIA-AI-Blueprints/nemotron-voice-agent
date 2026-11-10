# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The conversation: the session's shared history, and each task's own view of it."""

from __future__ import annotations

import json
from collections.abc import Mapping
from copy import deepcopy

from examples.frontend_backend_live.delegation_core.im.input import is_user_message
from examples.shared.ids import approx_tokens


def starts_a_request(item: dict) -> bool:
    """Return whether an item opens a request: a user message, as every input and every brief begins with one."""
    return is_user_message(item)


def cut_output(item: dict, limit: int, marker: str) -> dict:
    """Return ``item`` with a tool output over ``limit`` characters cut to its head and tail around ``marker``."""
    output = item.get("output")
    if item.get("type") != "function_call_output" or not isinstance(output, str) or len(output) <= limit:
        return item
    half = limit // 2
    return {**item, "output": output[:half] + marker.format(chars=len(output) - 2 * half) + output[-half:]}


def item_size(item: dict) -> int:
    """Return an item's estimated size in tokens, as the budget counts it."""
    return approx_tokens(json.dumps(item))


def window_start(items: list[dict], sizes: list[int], budget: int) -> int:
    """Return where a window of ``items`` within ``budget`` starts, dropping whole requests and keeping the newest."""
    total, start = sum(sizes), 0
    for index in range(1, len(items)):
        if total <= budget:
            break
        if starts_a_request(items[index]):
            total -= sum(sizes[start:index])
            start = index
    return start


def owed_outputs(items: list[dict], text: str, reported: Mapping[str, dict] | None = None) -> list[dict]:
    """Return an output for each unanswered call, so the history stays valid: the ``reported`` one, else ``text``."""
    answered = {item.get("call_id") for item in items if item.get("type") == "function_call_output"}
    reported = reported or {}
    return [
        reported.get(item["call_id"]) or {"type": "function_call_output", "call_id": item["call_id"], "output": text}
        for item in items
        if item.get("type") == "function_call" and item.get("call_id") not in answered
    ]


class Conversation:
    """One IM session's shared conversation: only closed segments, so no Work reads another's unanswered calls."""

    def __init__(self, budget: int, output_chars: int, marker: str):
        """Start empty; a window holds whole requests within ``budget`` tokens, tool outputs cut to ``output_chars``."""
        self.budget, self.output_chars, self.marker = budget, output_chars, marker
        self._items: list[dict] = []
        self._sizes: list[int] = []
        self._base = 0

    def join(self, items: list[dict]) -> None:
        """Add a closed segment at the end."""
        for item in items:
            item = cut_output(deepcopy(item), self.output_chars, self.marker)
            self._items.append(item)
            self._sizes.append(item_size(item))
        start = window_start(self._items, self._sizes, self.budget)
        if start:
            del self._items[:start], self._sizes[:start]
            self._base += start

    def window(self) -> tuple[int, list[dict]]:
        """Return the position of its first item, and the conversation within the budget."""
        return self._base, deepcopy(self._items)


class TaskHistory:
    """One task's view of the conversation: where it started, then everything that joined for the task."""

    def __init__(self, prefix: list[dict], budget: int | None, output_chars: int | None, marker: str):
        """Start from ``prefix``, what the task found; ``budget`` bounds ``history()``, ``output_chars`` each output."""
        self.budget, self.output_chars, self.marker = budget, output_chars, marker
        self.items: list[dict] = []
        self._sizes: list[int] = []
        self.join(prefix)
        self.base = len(self.items)

    def join(self, items: list[dict]) -> None:
        """Add items at the end, in order, a long tool output cut once."""
        for item in items:
            item = (
                deepcopy(item)
                if self.output_chars is None
                else cut_output(deepcopy(item), self.output_chars, self.marker)
            )
            self.items.append(item)
            self._sizes.append(item_size(item))

    def history(self) -> list[dict]:
        """Return the view within the budget, oldest first; ``RuntimeError`` if the newest request alone is over it."""
        if self.budget is None:
            return deepcopy(self.items)
        start = window_start(self.items, self._sizes, self.budget)
        if sum(self._sizes[start:]) > self.budget:
            raise RuntimeError("Backend context budget exceeded")
        return deepcopy(self.items[start:])

    def owed(self, text: str, reported: Mapping[str, dict] | None = None) -> list[dict]:
        """Return an output for each call of the view that has none: its output in ``reported``, else ``text``."""
        return owed_outputs(self.items, text, reported)

    def own(self) -> list[dict]:
        """Return what joined for the task after it started: its brief, its rounds, its outputs, its steering."""
        return deepcopy(self.items[self.base :])
