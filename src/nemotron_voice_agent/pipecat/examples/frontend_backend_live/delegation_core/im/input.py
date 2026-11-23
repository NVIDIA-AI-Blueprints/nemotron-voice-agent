# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""What the backend is told besides the conversation: the settings, the composed instructions and the records."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from copy import deepcopy
from dataclasses import dataclass, field

from nemotron_voice_agent.pipecat.shared.delegation_contract import OPTIONS, CallerContext
from nemotron_voice_agent.pipecat.shared.ids import uid

from ..backend.port import CallerLine, DecisionRejected
from ..items import WorkerData, is_message

SETTINGS = ("model", "instructions", "tools", "tool_choice", "max_output_tokens", *OPTIONS)


class ClarificationReceipt(dict):
    """A Core-authored record of routing actions linked to a clarification."""


def _scope_key(item: dict) -> tuple[str, str]:
    identity = item.get("id")
    return ("id", identity) if identity else ("item", json.dumps(item, sort_keys=True, ensure_ascii=False))


def _caller_parts(item: dict) -> tuple[str, ...]:
    content = item.get("content")
    if isinstance(content, str):
        return (content,)
    return tuple(
        part["text"]
        for part in content or ()
        if isinstance(part, dict) and part.get("type") == "input_text" and isinstance(part.get("text"), str)
    )


def _source_words(items: Iterable[dict], contexts: Mapping[str, CallerContext]) -> dict[str, dict[int, str]]:
    words: dict[str, dict[int, str]] = {}
    for item in items:
        source = item.get("id")
        if item.get("role") == "user" and source and source not in words:
            context = contexts.get(source)
            words[source] = dict(context["caller_turns"] if context is not None else enumerate(_caller_parts(item)))
    return words


def caller_lines(items: Iterable[dict], contexts: Mapping[str, CallerContext]) -> tuple[dict, ...]:
    """Give the conductor the same caller lines Core uses when binding selections."""
    return tuple(
        {"source_message_id": source, "sequence": index, "text": text}
        for source, parts in _source_words(items, contexts).items()
        for index, text in parts.items()
        if text.strip()
    )


def _selected_line_text(line: CallerLine, words: Mapping[str, dict[int, str]]) -> str:
    source, sequence, excerpt = line.source_message_id, line.sequence, line.excerpt
    if source not in words:
        raise DecisionRejected("Select caller lines only from the sources bound to this action")
    text = words[source].get(sequence)
    if text is None or not text.strip():
        raise DecisionRejected("Copy sequence from the selected caller line. It identifies the line, not the task")
    return text if excerpt is None else excerpt


def _scoped_playback(source: str, context: CallerContext, owners: frozenset[str]) -> Iterable[dict]:
    for receipt in context["playback"]:
        results = receipt["results"]
        if results and all(result.get("item_id") and result.get("task_ref") in owners for result in results):
            yield {
                "source_message_id": source,
                **{key: value for key, value in receipt.items() if key not in {"work_refs", "delegation_ids"}},
            }


@dataclass(frozen=True)
class InputScope:
    """A worker's routing interpretation with optional excerpts supplied by the conductor."""

    covered: frozenset[tuple[str, str]]
    item: dict
    selected_lines: frozenset[tuple[str, int]] = frozenset()
    whole_lines: frozenset[tuple[str, int]] = frozenset()
    whole_sources: frozenset[str] = frozenset()

    @classmethod
    def bind(
        cls,
        text: str,
        items: Iterable[dict],
        caller_context: Mapping[str, CallerContext] | None = None,
        playback_owners: frozenset[str] = frozenset(),
        *,
        require_lines: bool = False,
        selections: tuple[CallerLine, ...] = (),
        include_bound_lines: bool = False,
    ) -> InputScope:
        """Keep the original input in Core and build one model-facing data item for this worker."""
        items = tuple(items)
        if any(
            isinstance(part, dict) and part.get("type") != "input_text"
            for item in items
            if item.get("role") == "user" and isinstance(item.get("content"), list)
            for part in item["content"]
        ):
            raise DecisionRejected(
                "Cannot scope caller attachments by text excerpt. Route the complete message to one eligible task "
                "without excerpts."
            )
        contexts = caller_context or {}
        words = _source_words(items, contexts)
        playback = [
            receipt
            for source in words
            if source in contexts
            for receipt in _scoped_playback(source, contexts[source], playback_owners)
        ]
        if include_bound_lines and not selections:
            selections = tuple(
                CallerLine(source, index)
                for source, parts in words.items()
                if source in contexts
                for index, part in parts.items()
                if part.strip()
            )
        if require_lines and not selections and any(contexts.get(source, {}).get("caller_turns") for source in words):
            raise DecisionRejected("Each shared steer update needs a selection from its applicable caller lines")
        selected = dict.fromkeys(
            (line.source_message_id, line.sequence, _selected_line_text(line, words)) for line in selections
        )
        excerpts = []
        for source, index, line_text in selected:
            excerpt = {"source_message_id": source, "text": line_text}
            if source in contexts:
                excerpt["sequence"] = index
            excerpts.append(excerpt)
        selected_lines = frozenset((source, index) for source, index, _ in selected)
        whole_lines = frozenset((line.source_message_id, line.sequence) for line in selections if line.excerpt is None)
        whole_sources = frozenset(
            source
            for source in {identity for identity, _ in whole_lines}
            if all((source, index) in whole_lines for index, part in words[source].items() if part.strip())
        )
        note = {"task_input": text, "source_message_ids": list(words), "source_excerpts": excerpts}
        if playback:
            note["playback"] = playback
        item = routing_context(note)
        covered = frozenset(_scope_key(source) for source in items if not isinstance(source, ClarificationReceipt))
        return cls(covered, item, selected_lines, whole_lines, whole_sources)

    def conflicts(self, other: InputScope) -> bool:
        """Reject a whole source or line that another independent task also uses."""
        if any(("id", identity) in other.covered for identity in self.whole_sources) or any(
            ("id", identity) in self.covered for identity in other.whole_sources
        ):
            return True
        return bool(self.whole_lines & other.selected_lines or other.whole_lines & self.selected_lines)


def scoped_items(items: list[dict], scopes: Iterable[InputScope]) -> list[dict]:
    """Replace bound material in caller order, preserving other inputs and separate host lifecycle facts."""
    if not scopes:
        return items
    insertions: dict[int, list[dict]] = {}
    keys = list(map(_scope_key, items))
    positions = {key: index for index, key in enumerate(keys)}
    covered = set()
    for scope in scopes:
        point = max((positions[key] for key in scope.covered if key in positions), default=len(items))
        insertions.setdefault(point, []).append(scope.item)
        covered.update(scope.covered)
    projected = []
    for index, item in enumerate(items):
        if keys[index] not in covered:
            projected.append(item)
        projected.extend(insertions.get(index, ()))
    return [*projected, *insertions.get(len(items), ())]


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
    """Return whether the item is original caller input, excluding worker context."""
    return item.get("role") == "user" and is_message(item) and not isinstance(item, WorkerData)


def user_message(text: str) -> dict:
    """Return a user message whose one ``input_text`` part holds ``text``."""
    return {"type": "message", "role": "user", "content": [{"type": "input_text", "text": text}]}


def routing_context(note: dict, *, starts_request: bool = True) -> WorkerData:
    """Render a conductor's routing data for the assigned worker."""
    return WorkerData(
        user_message(json.dumps({"conductor_routing_context": note}, ensure_ascii=False)),
        id=uid("routing_context"),
        starts_request=starts_request,
    )


def with_parts(items: list[dict], *texts: str) -> list[dict]:
    """Append a separate immutable host-context snapshot without rewriting caller messages.

    Each constructed snapshot owns its identity. A later task can omit an old snapshot without presenting changed
    caller content under a previously served caller-message ID.
    """
    parts = [{"type": "input_text", "text": text} for text in texts if text]
    if not parts:
        return items
    return [*deepcopy(items), {"type": "message", "role": "developer", "id": uid("backend_context"), "content": parts}]


def records_parts(template: str, closing: str, records: Iterable[dict]) -> tuple[str, ...]:
    """Return the records block, one JSON line per Work, and the closing line as extra parts; none with no record."""
    lines = [json.dumps(record, ensure_ascii=False) for record in records]
    return (template.format(records="\n".join(lines)), closing) if lines else ()
