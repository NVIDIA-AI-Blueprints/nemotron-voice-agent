# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The conversation: the session's shared history, and each task's own view of it.

A running Work's items stay out of the shared conversation, so no Work reads another Work's unanswered calls. Both keep
whole requests within ``delegation.context_tokens``: the oldest requests leave whole and the newest always stays, and a
tool output longer than ``delegation.tool_output_chars`` is cut to its head and tail when it joins. The records block
(``im/input.py``) carries each Work's steps, so a trim never hides what an action did.
"""

from __future__ import annotations

import json
from copy import deepcopy

from delegation_core.ids import approx_tokens
from delegation_core.im.input import is_user_message


def starts_a_request(item: dict) -> bool:
    """Whether an item opens a request: a user message, as every delegation's input and every brief begin with one."""
    return is_user_message(item)


def cut_output(item: dict, limit: int) -> dict:
    """A ``function_call_output`` over ``limit`` characters, cut to its head and tail; any other item as it is."""
    output = item.get("output")
    if item.get("type") != "function_call_output" or not isinstance(output, str) or len(output) <= limit:
        return item
    half = limit // 2
    marker = f"\n[... {len(output) - 2 * half} characters cut ...]\n"
    return {**item, "output": output[:half] + marker + output[-half:]}


def item_size(item: dict) -> int:
    """An item's size in tokens, as the budget counts it (estimated)."""
    return approx_tokens(json.dumps(item))


def window_start(items: list[dict], sizes: list[int], budget: int) -> int:
    """Where a window of ``items`` within ``budget`` starts: at a request, dropping whole requests, the newest kept."""
    total, start = sum(sizes), 0
    for index in range(1, len(items)):
        if total <= budget:
            break
        if starts_a_request(items[index]):
            total -= sum(sizes[start:index])
            start = index
    return start


def owed_outputs(items: list[dict], text: str) -> list[dict]:
    """An output saying ``text`` for each function call in ``items`` that has none, so the history stays valid."""
    answered = {item.get("call_id") for item in items if item.get("type") == "function_call_output"}
    return [
        {"type": "function_call_output", "call_id": item["call_id"], "output": text}
        for item in items
        if item.get("type") == "function_call" and item.get("call_id") not in answered
    ]


class Conversation:
    """One IM session's shared conversation: ended Works' segments and closed exchanges, in the order they closed."""

    def __init__(self, budget: int, output_chars: int):
        """Start empty; a window holds whole requests within ``budget`` tokens, tool outputs cut to ``output_chars``."""
        self.budget, self.output_chars = budget, output_chars
        self._items: list[dict] = []
        self._sizes: list[int] = []
        self._base = 0  # the position of self._items[0]: what came before has left every window

    @property
    def end(self) -> int:
        """The position after the last item."""
        return self._base + len(self._items)

    def join(self, items: list[dict]) -> None:
        """Add a closed segment at the end."""
        for item in items:
            item = cut_output(deepcopy(item), self.output_chars)
            self._items.append(item)
            self._sizes.append(item_size(item))
        start = window_start(self._items, self._sizes, self.budget)
        if start:
            del self._items[:start], self._sizes[:start]
            self._base += start

    def window(self) -> tuple[int, list[dict]]:
        """The conversation within the budget, and the position of its first item."""
        return self._base, deepcopy(self._items)


class TaskHistory:
    """One task's view of the conversation: where it started, then everything that joined for the task."""

    def __init__(self, prefix: list[dict], budget: int | None, output_chars: int | None):
        """Start from ``prefix``; ``budget`` bounds ``history()`` and ``output_chars`` each tool output.

        ``None`` for both keeps every item as given, whole (a one-shot session). ``prefix`` is what the task found when
        it started: the shared conversation's window and its Work's input. Everything that joins later is the task's
        own (``own()``).
        """
        self.budget, self.output_chars = budget, output_chars
        self.items: list[dict] = []
        self._sizes: list[int] = []
        self.join(prefix)
        self.base = len(self.items)

    def join(self, items: list[dict]) -> None:
        """Add items at the end, in order (a long tool output cut once)."""
        for item in items:
            item = deepcopy(item) if self.output_chars is None else cut_output(deepcopy(item), self.output_chars)
            self.items.append(item)
            self._sizes.append(item_size(item))

    def history(self) -> list[dict]:
        """The view within the budget, oldest first.

        Raises:
            RuntimeError: The newest request alone is over the budget.
        """
        if self.budget is None:
            return deepcopy(self.items)
        start = window_start(self.items, self._sizes, self.budget)
        if sum(self._sizes[start:]) > self.budget:
            raise RuntimeError("Backend context budget exceeded")
        return deepcopy(self.items[start:])

    def position(self) -> int:
        """The position after the view's last item."""
        return len(self.items)

    def since(self, position: int) -> list[dict]:
        """The view's items from ``position`` on, also those a trim leaves out of ``history()``."""
        return deepcopy(self.items[position:])

    def owed(self, text: str) -> list[dict]:
        """An output saying ``text`` for each call of the view that has none."""
        return owed_outputs(self.items, text)

    def own(self) -> list[dict]:
        """What joined for the task after it started: its brief, its rounds, its outputs, its steering."""
        return deepcopy(self.items[self.base :])
