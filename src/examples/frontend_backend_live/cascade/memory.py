# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The talker's memory: what the talker and the writer read, owned by the cascade (:class:`TalkerMemory`).

Entries render as the messages the models read; a backend result and the application's text become user messages
marked as reference data by the prompt set's labels (``result_label``, ``context_label``). Above ``compact_at`` of the
token budget, the oldest unpinned turn groups leave, never the newest. The host's own buffer (Pipecat's context) is not
the memory: the cascade only takes the caller's committed turns from it (:class:`TurnBuffer`), so nothing Pipecat adds
there, such as its note after a turn with no words, reaches the talker.

The talker reads the memory with the set's ``caller_label`` in front of each caller turn right after reference data
(:meth:`TalkerMemory.labeled`); the writer and the normalizer read it as it is.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

from examples.frontend_backend_live.prompts import PromptSet


class TurnBuffer(Protocol):
    """The host's buffer of the call: the messages it seeds the cascade with, then each committed caller turn.

    Pipecat's ``LLMContext`` is one, behind :class:`ContextConversation`. The cascade only reads it, and bounds it.
    """

    def messages(self) -> list[dict]:
        """Return the messages in order (system, developer, user and assistant roles)."""

    def trim(self, limit: int) -> None:
        """Keep the messages the buffer started with that are system ones, and only the latest ``limit`` others.

        The cascade passes 0 once it has taken every caller turn: only the prompt stays.
        """


def trim_messages(messages: list[dict], preserve: int, limit: int) -> list[dict]:
    """Keep the first ``preserve`` messages (the prompt) and the latest ``limit`` of the rest (none for 0)."""
    if len(messages) <= preserve + max(limit, 0):
        return messages
    return messages[:preserve] + (messages[preserve:][-limit:] if limit > 0 else [])


class ContextConversation:
    """Pipecat's ``LLMContext`` as the host's :class:`TurnBuffer`."""

    def __init__(self, context):
        """Wrap ``context``; the system messages it starts with are the prompt that every trim keeps."""
        self.context = context
        messages = context.get_messages()
        self.prompt = next(
            (i for i, m in enumerate(messages) if m.get("role") not in ("system", "developer")), len(messages)
        )

    def messages(self) -> list[dict]:
        """Return the context's messages."""
        return list(self.context.get_messages())

    def trim(self, limit: int) -> None:
        """Keep the prompt and the latest ``limit`` other messages, so the buffer stays bounded over a long call."""
        messages = self.context.get_messages()
        trimmed = trim_messages(messages, self.prompt, limit)
        if trimmed is not messages:
            self.context.set_messages(trimmed)


class EntryKind(StrEnum):
    """What an entry of the talker's memory is."""

    PROMPT = "prompt"  # a system or developer message of the host's seed
    INSTRUCTION = "instruction"  # an instructions append, or the greeting's instructions
    CALLER = "caller"  # a caller turn; a normalizer's rewrite replaces it
    SPOKEN = "spoken"  # a line handed to speech, corrected by playout
    RESULT = "result"  # a backend result's speak part, written with the line that says it
    APP_CONTEXT = "app_context"  # the application's thinking, or its commentary once said
    TYPED = "typed"  # a question the application typed itself


@dataclass(frozen=True)
class Kind:
    """How the entries of one kind render and live."""

    role: str  # the role of its message; "" keeps the entry's own (a prompt entry)
    pinned: bool = False  # kept through every compaction
    opens_turn: bool = False  # a caller turn: the turn group it opens leaves as one
    said: bool = False  # a line the voice handed to speech, which playout corrects
    asks: bool = False  # a turn the talker decides (the caller's, or a question the application typed)


KINDS = {
    EntryKind.PROMPT: Kind("", pinned=True),
    EntryKind.INSTRUCTION: Kind("developer", pinned=True),
    EntryKind.CALLER: Kind("user", opens_turn=True, asks=True),
    EntryKind.SPOKEN: Kind("assistant", said=True),
    EntryKind.RESULT: Kind("user"),
    EntryKind.APP_CONTEXT: Kind("user"),
    EntryKind.TYPED: Kind("user", asks=True),
}


@dataclass
class Entry:
    """One entry of the talker's memory."""

    kind: EntryKind
    text: str
    role: str = ""  # a prompt entry's own role (system or developer)
    line: int | None = None  # the line that wrote it: a spoken line, or what a delivered line carried

    @property
    def traits(self) -> Kind:
        """How this entry renders and lives."""
        return KINDS[self.kind]


def approx_tokens(message: dict) -> int:
    """A cheap token estimate for one message: about four characters per token, plus its role."""
    return (len(json.dumps(message, ensure_ascii=False)) + 3) // 4


class TalkerMemory:
    """The talker's memory for one call; the cascade's parts are its only writers."""

    def __init__(self, prompts: PromptSet, *, context_tokens: int, compact_at: float):
        """Start empty, with a budget of ``context_tokens``, compacted above ``compact_at`` of it.

        The entries render in the words of ``prompts``: its labels mark reference data, its cut mark ends a cut line,
        and its caller label marks a caller turn right after reference data in the talker's view (:meth:`labeled`).
        """
        self.context_tokens, self.compact_at = context_tokens, compact_at
        # What the message of each kind of reference data starts with.
        self.marks = {
            EntryKind.RESULT: prompts.result_label,
            EntryKind.APP_CONTEXT: prompts.context_label,
            EntryKind.TYPED: prompts.context_label,
        }
        self.caller_label = prompts.caller_label
        self.cut_mark = prompts.cut_mark
        self.entries: list[Entry] = []
        # Moves with every entry but a turn the talker decides, and when a line is cut or taken back; never on a
        # rewrite of a caller's turn or a compaction. So a wording prepared earlier can tell whether anything it read
        # moved since besides those turns: a line of the voice, an instruction, the application's context.
        self.version = 0

    @classmethod
    def seeded(
        cls, prompts: PromptSet, messages: list[dict], *, context_tokens: int, compact_at: float
    ) -> TalkerMemory:
        """A memory that starts with the host's seed: its system and developer messages pinned, the rest as history.

        A user message under one of the set's labels is that kind of reference data again.
        """
        memory = cls(prompts, context_tokens=context_tokens, compact_at=compact_at)
        for message in messages:
            role, text = message.get("role"), message_text(message)
            if role in ("system", "developer"):
                memory.entries.append(Entry(EntryKind.PROMPT, text, role=role))
            elif role == "assistant":
                memory.entries.append(Entry(EntryKind.SPOKEN, text))
            elif role == "user" and text.startswith(prompts.result_label):
                memory.entries.append(Entry(EntryKind.RESULT, text.removeprefix(prompts.result_label)))
            elif role == "user" and text.startswith(prompts.context_label):
                memory.entries.append(Entry(EntryKind.APP_CONTEXT, text.removeprefix(prompts.context_label)))
            elif role == "user":
                memory.entries.append(Entry(EntryKind.CALLER, text))
        memory._compact()
        return memory

    def render(self, entry: Entry) -> dict:
        """One entry as the message the models read: reference data starts with its kind's mark."""
        return {"role": entry.traits.role or entry.role, "content": self.marks.get(entry.kind, "") + entry.text}

    def messages(self) -> list[dict]:
        """The memory as the messages the models read, in order."""
        return [self.render(entry) for entry in self.entries]

    def labeled(self) -> list[dict]:
        """The talker's view of the memory: :meth:`messages`, each caller turn right after reference data labeled.

        The set's caller label goes in front of a caller turn whose message before it in the voice history (the user
        and assistant ones) is reference data, so a caller's "yes" right after a backend result reads as the caller's,
        not as part of the result. Each entry's kind decides, never its text.
        """
        messages, after_reference = [], False
        for entry in self.entries:
            message = self.render(entry)
            if message["role"] in ("user", "assistant"):
                if after_reference and entry.traits.opens_turn:
                    message["content"] = self.caller_label + message["content"]
                after_reference = entry.kind in self.marks
            messages.append(message)
        return messages

    def kinds(self) -> list[EntryKind]:
        """The kind of each message of the voice history, the user and assistant ones (``split_messages``), in order."""
        return [entry.kind for entry in self.entries if self.render(entry)["role"] in ("user", "assistant")]

    def tokens(self) -> int:
        """The estimated tokens the memory holds now."""
        return sum(approx_tokens(self.render(entry)) for entry in self.entries)

    def caller(self, text: str) -> None:
        """A caller turn was heard."""
        self._add(Entry(EntryKind.CALLER, text))

    def replace_caller(self, original: str, text: str) -> None:
        """The newest caller turn ``original`` reads ``text`` (the normalizer rewrote it)."""
        for entry in reversed(self.entries):
            if entry.traits.opens_turn and entry.text == original:
                entry.text = text
                return

    def spoken(self, line: int, text: str) -> None:
        """The line ``line`` was handed to speech."""
        self._add(Entry(EntryKind.SPOKEN, text, line=line))

    def result(self, line: int, speak: str) -> None:
        """A result's speak part, delivered with the line ``line`` that says it."""
        self._add(Entry(EntryKind.RESULT, speak, line=line))

    def app_context(self, text: str, *, line: int | None = None) -> None:
        """The application's context: thinking, or commentary delivered with the line ``line`` that says it."""
        self._add(Entry(EntryKind.APP_CONTEXT, text, line=line))

    def instruction(self, text: str) -> None:
        """An instruction for the voice; pinned."""
        self._add(Entry(EntryKind.INSTRUCTION, text))

    def typed(self, text: str) -> None:
        """A question the application typed itself."""
        self._add(Entry(EntryKind.TYPED, text))

    def cut(self, line: int, heard: str, *, carried: bool = True) -> None:
        """The line ``line`` was cut after ``heard``: it reads ``heard`` then the cut mark, or leaves when it is empty.

        What the line carried (results, commentary) stays; with ``carried`` false it leaves too, to be said again.
        """
        before = len(self.entries)
        self.entries = [
            entry for entry in self.entries if entry.line != line or (bool(heard) if entry.traits.said else carried)
        ]
        said = [entry for entry in self.entries if entry.traits.said and entry.line == line]
        for entry in said:
            entry.text = heard.rstrip() + self.cut_mark
        if said or len(self.entries) != before:
            self.version += 1

    def unsay(self, line: int) -> None:
        """Nobody heard the line ``line``: it and what it carried (results, commentary) leave the memory."""
        kept = [entry for entry in self.entries if entry.line != line]
        if len(kept) != len(self.entries):
            self.entries = kept
            self.version += 1

    def _add(self, entry: Entry) -> None:
        self.entries.append(entry)
        if not entry.traits.asks:
            self.version += 1
        self._compact()

    def _compact(self) -> None:
        """Above ``compact_at`` of the budget, drop the oldest turn groups that are not pinned, never the newest."""
        limit = self.compact_at * self.context_tokens
        while self.tokens() > limit:
            groups = self._groups()
            if len(groups) < 2:
                return
            oldest = set(groups[0])
            self.entries = [entry for index, entry in enumerate(self.entries) if index not in oldest]

    def _groups(self) -> list[list[int]]:
        """The indexes of the unpinned entries, by turn group: each caller turn starts one."""
        groups: list[list[int]] = []
        for index, entry in enumerate(self.entries):
            if entry.traits.pinned:
                continue
            if entry.traits.opens_turn or not groups:
                groups.append([])
            groups[-1].append(index)
        return groups


def message_text(message: dict) -> str:
    """Return a message's text, joining content parts when the content is a list."""
    content = message.get("content") or ""
    if isinstance(content, str):
        return content
    return "\n".join(part.get("text", "") for part in content if isinstance(part, dict))


def split_messages(messages: list[dict]) -> tuple[str, list[dict]]:
    """Split a context into the talker's instructions and its plain-text voice history.

    Leading system messages are the instructions; later system or developer messages join them as context,
    exactly as the frontend does. Everything else is history with string content.
    """
    instructions: list[str] = []
    history: list[dict] = []
    for message in messages:
        role = message.get("role")
        text = message_text(message)
        if role in ("system", "developer"):
            instructions.append(text)
        elif role in ("user", "assistant"):
            history.append({"role": role, "content": text})
    return "\n\n".join(instructions), history


def caller_turns(messages: list[dict]) -> list[str]:
    """Return the text of each user message, oldest first: in the host's buffer, the seed's and the caller's turns."""
    return [message_text(message) for message in messages if message.get("role") == "user"]
