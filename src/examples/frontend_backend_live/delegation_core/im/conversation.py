# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The conversation: the session's shared history, and each task's own view of it."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from copy import deepcopy

from examples.frontend_backend_live.delegation_core.im.input import is_scoped_input, is_user_message
from examples.frontend_backend_live.delegation_core.items import input_item, is_message
from examples.shared.ids import approx_tokens


def starts_a_request(item: dict) -> bool:
    """Start a context window at caller input or a host-authored task-specific projection."""
    return is_user_message(item) or is_scoped_input(item)


def with_ids(items: list[dict], key: str) -> list[dict]:
    """Return a copy of ``items`` whose messages carry ids: ``key``, then ``key.2``, ``key.3`` ... in order.

    Every task's view of the conversation knows an item by its id, or a call and its output by the call's id.
    """
    items = deepcopy(items)
    for number, item in enumerate((item for item in items if is_message(item)), 1):
        item["id"] = key if number == 1 else f"{key}.{number}"
    return items


def cut_output(item: dict, limit: int, marker: str) -> dict:
    """Return ``item`` with a tool output over ``limit`` characters cut to its head and tail around ``marker``."""
    output = item.get("output")
    if item.get("type") != "function_call_output" or not isinstance(output, str) or len(output) <= limit:
        return item
    half = limit // 2
    return {**item, "output": output[:half] + marker.format(chars=len(output) - 2 * half) + output[-half:]}


def item_size(item: dict) -> int:
    """Return an item's estimated size in tokens, as the budget counts it: the input item a request sends of it."""
    return approx_tokens(json.dumps(input_item(item)))


def window_start(items: Sequence[dict | None], sizes: list[int], budget: int) -> int:
    """Return where a window of ``items`` within ``budget`` starts, dropping whole requests and keeping the newest."""
    total, start = sum(sizes), 0
    for index in range(1, len(items)):
        if total <= budget:
            break
        if (item := items[index]) is not None and starts_a_request(item):
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
        if item.get("type") == "function_call" and item.get("call_id") and item.get("call_id") not in answered
    ]


class Conversation:
    """One IM session's shared conversation: only closed segments, so no Work reads another's unanswered calls."""

    def __init__(self, budget: int, output_chars: int, marker: str):
        """Start empty; a window holds whole requests within ``budget`` tokens, tool outputs cut to ``output_chars``."""
        self.budget, self.output_chars, self.marker = budget, output_chars, marker
        self._items: list[dict | None] = []
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
        """Return the start of the retained window and its visible items."""
        return self._base, deepcopy([item for item in self._items if item is not None])

    def discard(self, item_id: str) -> None:
        """Remove one item's content while keeping existing reader positions valid."""
        for index, item in enumerate(self._items):
            if item is not None and item.get("id") == item_id:
                self._items[index] = None
                self._sizes[index] = 0
                return

    def read(self, cursor: int) -> tuple[list[dict], int, int]:
        """Return visible items, the first unread index, and the absolute read-through position."""
        _, items = self.window()
        read = sum(item is not None for item in self._items[: max(0, cursor - self._base)])
        return items, read, self._base + len(self._items)


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

    def history(self, extra: Sequence[dict] = ()) -> list[dict]:
        """Return the view and required extra items within the budget; refuse if the newest request cannot fit."""
        if self.budget is None:
            return deepcopy([*self.items, *extra])
        available = self.budget - sum(map(item_size, extra))
        start = window_start(self.items, self._sizes, available)
        if sum(self._sizes[start:]) > available:
            raise RuntimeError("Backend context budget exceeded")
        return deepcopy([*self.items[start:], *extra])

    def extra_budget(self, required: Sequence[dict] = ()) -> int | None:
        """Return the room left after the newest whole request and any additional required items."""
        if self.budget is None:
            return None
        start = next((index for index in reversed(range(len(self.items))) if starts_a_request(self.items[index])), 0)
        return self.budget - sum(self._sizes[start:]) - sum(map(item_size, required))

    def owed(self, text: str, reported: Mapping[str, dict] | None = None) -> list[dict]:
        """Return an output for each call of the view that has none: its output in ``reported``, else ``text``."""
        return owed_outputs(self.items, text, reported)

    def own(self) -> list[dict]:
        """Return what joined for the task after it started: its brief, its rounds, its outputs, its steering."""
        return deepcopy(self.items[self.base :])
