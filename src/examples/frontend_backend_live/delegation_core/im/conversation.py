# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The conversation: the session's shared history, and each task's own view of it.

The session also keeps a record of the backend's answers and the questions they ask (:class:`AskedQuestions`).
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Final

from examples.frontend_backend_live.delegation_core.im.input import is_user_message
from examples.shared.delegation_contract import ReplyUnheard, ends_with_question, question_key, sentences
from examples.shared.ids import approx_tokens

MAX_QUESTIONS: Final = 50  # question entries kept; the oldest is dropped (a memory bound, not a behaviour)


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


@dataclass
class Asked:
    """One question the backend asked the caller."""

    text: str  # its newest wording
    key: str  # its letters and digits, case-folded: two asks are one question when their keys are equal
    asked: int = 1  # answers that asked it
    heard_in_full: int | None = None  # times the caller had heard it to the end when they next spoke; None: unknown
    awaiting: bool = True  # asked since the caller last spoke
    unheard: bool = False  # the caller's turn after its newest ask stated a question unheard
    challenged: bool = False  # the gate already held a change once for it (open_question); cleared when asked again


@dataclass
class AskedQuestions:
    """The backend's answers to the caller in this session, as they were written, and one entry per question."""

    answers: int = 0  # task answers written so far; conductor status answers and stop lines are not counted
    written: int = 0  # of them, answers with a sentence that ends_with_question
    # For each answer that asks: its index, and the requests of its Work when it was written.
    asking: list[tuple[int, frozenset[str]]] = field(default_factory=list)
    entries: dict[str, Asked] = field(default_factory=dict)  # by key, oldest ask first

    def answer_written(self, text: str, requests: Iterable[str]) -> list[Asked]:
        """A task answer was written into the response (never one that was held, or a stopped task's).

        Counts it, and records each of its question sentences once per answer: ``asked`` + 1, ``awaiting`` set,
        ``unheard`` and ``challenged`` cleared, the entry moved to the end. Returns the entries it touched, newest last.
        """
        index = self.answers
        self.answers += 1
        touched: list[Asked] = []
        for sentence in sentences(text):
            key = question_key(sentence)
            if not ends_with_question(sentence) or not key or any(entry.key == key for entry in touched):
                continue
            entry = self.entries.pop(key, None)
            if entry is None:
                entry = Asked(sentence, key)
            else:
                entry.text, entry.asked, entry.awaiting = sentence, entry.asked + 1, True
                entry.unheard = entry.challenged = False
            self.entries[key] = entry
            touched.append(entry)
        if touched:
            self.written += 1
            self.asking.append((index, frozenset(requests)))
        while len(self.entries) > MAX_QUESTIONS:
            del self.entries[next(iter(self.entries))]
        return touched

    def asked_since(self, mark: int, *, besides: str) -> bool:
        """Whether an answer that asks was written after ``mark`` answers by a Work not answering ``besides``.

        That Work's requests did not include ``besides`` when it wrote the answer.
        """
        return any(index >= mark and besides not in requests for index, requests in self.asking)

    def replied(self, level: ReplyUnheard | None) -> int:
        """The caller spoke (a frontend delegation was admitted) and stated ``level``; returns how many were awaiting.

        ``NONE`` or ``ANSWER``: each awaiting entry was heard in full (``heard_in_full`` + 1, ``unheard`` cleared).
        ``QUESTION``: each awaiting entry is ``unheard``. ``None`` (no fact): counts stay unknown. Each clears
        ``awaiting``.
        """
        awaiting = [entry for entry in self.entries.values() if entry.awaiting]
        for entry in awaiting:
            entry.awaiting = False
            if level is None:
                continue
            if level is ReplyUnheard.QUESTION:
                entry.unheard = True
            else:
                entry.heard_in_full = (entry.heard_in_full or 0) + 1
                entry.unheard = False
        return len(awaiting)

    def open_question(self) -> Asked | None:
        """The oldest entry that is ``unheard`` and not yet ``challenged``.

        That is a question the caller never heard and the backend has not asked again since.
        """
        return next((entry for entry in self.entries.values() if entry.unheard and not entry.challenged), None)

    def newest(self, count: int) -> list[Asked]:
        """The ``count`` most recently asked questions, oldest of them first."""
        return list(self.entries.values())[-count:] if count > 0 else []
