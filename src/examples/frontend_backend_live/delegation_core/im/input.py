# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""What the backend is told besides the conversation: the settings, the composed instructions and the records."""

from __future__ import annotations

import json
import re
from collections import defaultdict
from collections.abc import Iterable, Mapping
from copy import deepcopy
from dataclasses import dataclass, field
from itertools import islice

from examples.shared.delegation_contract import OPTIONS, CallerContext
from examples.shared.ids import uid

from ..backend.port import CallerLine, DecisionRejected

SETTINGS = ("model", "instructions", "tools", "tool_choice", "max_output_tokens", *OPTIONS)
SCOPED_INPUT_ID = "scoped_input"


def is_scoped_input(item: dict) -> bool:
    """Identify task-only input emitted by Core."""
    return item.get("role") == "developer" and str(item.get("id", "")).startswith(SCOPED_INPUT_ID + "_")


def _scope_key(item: dict) -> tuple[str, str]:
    identity = item.get("id")
    return ("id", identity) if identity else ("item", json.dumps(item, sort_keys=True, ensure_ascii=False))


def _covers_text(text: str, spans: Iterable[tuple[int, int]]) -> bool:
    """Return whether a set of copied spans covers every word of this content part."""
    covered = 0
    for start, end in sorted(spans):
        if text[covered:start].strip():
            return False
        covered = max(covered, end)
    return not text[covered:].strip()


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


def _quote_patterns(quote: str) -> tuple[re.Pattern, ...]:
    tokens = quote.split()
    # Resolve verbatim first, then account for ASR spacing when finding the stored words.
    return (re.compile(re.escape(quote)), re.compile(r"\s+".join(map(re.escape, tokens)))) if tokens else ()


def _selected_line_span(line: CallerLine, words: Mapping[str, dict[int, str]]) -> tuple[str, int, int, int]:
    text = words.get(line.source_message_id, {}).get(line.sequence)
    if text is None or not text.strip():
        raise DecisionRejected("Select caller lines only from the sources bound to this action")
    if line.excerpt is None:
        return line.source_message_id, line.sequence, 0, len(text)
    for pattern in _quote_patterns(line.excerpt):
        matches = list(islice(pattern.finditer(text), 2))
        if len(matches) == 1:
            return line.source_message_id, line.sequence, *matches[0].span()
        if matches:
            break
    raise DecisionRejected("Copy only the task's exact excerpt from the selected caller line, once")


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
    """A worker's routing interpretation with exact, optional text from the bound source messages."""

    covered: frozenset[tuple[str, str]]
    item: dict
    source_spans: tuple[tuple[str, int, int, int], ...] = ()
    whole_sources: frozenset[str] = frozenset()

    @classmethod
    def bind(
        cls,
        text: str,
        evidence: tuple[str, ...],
        items: Iterable[dict],
        caller_context: Mapping[str, CallerContext] | None = None,
        playback_owners: frozenset[str] = frozenset(),
        *,
        require_evidence: bool = False,
        selections: tuple[CallerLine, ...] = (),
        include_bound_lines: bool = False,
    ) -> InputScope:
        """Keep the original input in Core and build one model-facing data item for this worker."""
        items = tuple(items)
        contexts = caller_context or {}
        words = _source_words(items, contexts)
        playback = [
            receipt
            for source in words
            if source in contexts
            for receipt in _scoped_playback(source, contexts[source], playback_owners)
        ]
        if include_bound_lines and not (evidence or selections):
            selections = tuple(
                CallerLine(source, index)
                for source, parts in words.items()
                if source in contexts
                for index, part in parts.items()
                if part.strip()
            )
        selected = [_selected_line_span(line, words) for line in selections]
        for quote in evidence:
            for pattern in _quote_patterns(quote):
                matches = list(
                    islice(
                        (
                            (identity, index, match)
                            for identity, parts in words.items()
                            for index, part in parts.items()
                            for match in pattern.finditer(part)
                        ),
                        2,
                    )
                )
                if matches:
                    break
            else:
                raise DecisionRejected(
                    "Scoped input evidence must copy exact text from a bound caller source, one excerpt per caller line"
                )
            # The same words can refer to different caller turns. A bare quote cannot select one.
            if len(matches) != 1:
                continue
            source, index, match = matches[0]
            selected.append((source, index, *match.span()))
        if require_evidence and not selected and any(contexts.get(source, {}).get("caller_turns") for source in words):
            raise DecisionRejected("Each shared steer update needs a selection from its applicable caller lines")
        excerpts = []
        spans: dict[tuple[str, int], list[tuple[int, int]]] = defaultdict(list)
        for source, index, start, end in dict.fromkeys(selected):
            excerpt = {"source_message_id": source, "text": words[source][index][start:end]}
            if source in contexts:
                excerpt["sequence"] = index
            excerpts.append(excerpt)
            spans[source, index].append((start, end))
        whole_sources = frozenset(
            source
            for source in {identity for identity, _ in spans}
            if all(_covers_text(part, spans.get((source, index), ())) for index, part in words[source].items())
        )
        note = {"task_input": text, "source_message_ids": list(words), "source_excerpts": excerpts}
        if playback:
            note["playback"] = playback
        item = with_parts([], json.dumps({"conductor_routing_context": note}, ensure_ascii=False))[0]
        item["id"] = uid(SCOPED_INPUT_ID)
        source_spans = tuple(
            (source, index, start, end) for (source, index), ranges in spans.items() for start, end in ranges
        )
        return cls(frozenset(map(_scope_key, items)), item, source_spans, whole_sources)

    def conflicts(self, other: InputScope) -> bool:
        """Reject whole shared sources or overlapping excerpts when assigning independent new work."""
        if any(("id", identity) in other.covered for identity in self.whole_sources) or any(
            ("id", identity) in self.covered for identity in other.whole_sources
        ):
            return True
        return any(
            source == other_source and part == other_part and start < other_end and other_start < end
            for source, part, start, end in self.source_spans
            for other_source, other_part, other_start, other_end in other.source_spans
        )


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
    """Return whether an input item is a user message (a typed one may leave its ``type`` out)."""
    return item.get("role") == "user" and item.get("type", "message") == "message"


def user_message(text: str) -> dict:
    """Return a user message whose one ``input_text`` part holds ``text``."""
    return {"type": "message", "role": "user", "content": [{"type": "input_text", "text": text}]}


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
