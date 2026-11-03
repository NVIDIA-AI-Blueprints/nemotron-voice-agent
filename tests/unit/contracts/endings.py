# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The endings fixture (``tests/unit/fixtures/delegation_endings.json``): how each delegation ends on the wire.

Each scenario lists the official events of its responses in the order the IM writes them, as :func:`normal_form`
reads an event: its type and status, a message's phase, an item's type (an ``mcp_call``'s server label, name and
status), a terminal's kind and code or reason, ``previous_response_id`` as a response label, and ``[key]`` for the
``texts`` line a message carries. Deltas and the other events are not compared.

* The IM's test (``tests/unit/delegation_core/test_endings.py``) makes the IM write each scenario, normalizes the real
  events with :func:`normal_form` and compares them with ``timeline``.
* The reader's test (``test_cascade_reader.py``) replays :func:`official_events` into the reader and checks its outcome
  against ``reader``.
* :func:`timeline_problems` holds the rules every scenario keeps, so the fixture cannot contradict the wire.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

from delegation_core.backend.port import message_phase

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "delegation_endings.json"
# How a delegation ends, as the reader names it (the frontend's ``cascade.ports.Ending``).
ENDINGS = ("answered", "handed_over", "late", "failed", "cancelled", "lost")
READER_KEYS = ("results", "endings", "takers", "waits", "typed", "progress", "steps", "shown")
TERMINAL_WORDS = ("completed", "failed", "incomplete")
ITEM_WORDS = ("message", "function_call", "mcp_call", "mcp_call.in_progress", "mcp_call.completed", "mcp_call.failed")


def load() -> dict[str, dict]:
    """The scenarios, by name."""
    return json.loads(FIXTURE.read_text(encoding="utf-8"))["scenarios"]


# --------------------------------------------------------------------------------------------- reading events
def normal_form(event: dict, labels: dict[str, str], texts: Mapping[str, str] | None = None) -> str | None:
    """One official event as the fixture writes it, or ``None`` for an event the fixture does not compare.

    ``labels`` maps each response id to its label (``R1``, ``R2`` ... in order of first appearance); a new id gets the
    next label. ``texts`` maps the names of ``texts`` lines to their words, so a message that carries one is marked.
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
        return _item_form(event.get("item") or {}, texts)
    if kind == "response.completed":
        return "completed"
    if kind == "response.failed":
        return f"failed {(response.get('error') or {}).get('code')}"
    if kind == "response.incomplete":
        if response.get("status") == "cancelled":
            return "incomplete cancelled"
        return f"incomplete {(response.get('incomplete_details') or {}).get('reason')}"
    return None


def _item_form(item: dict, texts: Mapping[str, str] | None) -> str:
    if item.get("type") == "message":
        text = "".join(part.get("text", "") for part in item.get("content") or [] if isinstance(part, dict))
        names = [name for name, words in (texts or {}).items() if words and words == text]
        return f"message {message_phase(item)}" + (f" [{names[0]}]" if names else "")
    if item.get("type") == "mcp_call":
        return f"mcp_call {item.get('server_label')}/{item.get('name')} {item.get('status')}"
    return str(item.get("type"))


def normalized(events: list[tuple[str, dict]], texts: Mapping[str, str] | None = None) -> list[list[str]]:
    """Events as ``(delegation_id, event)`` pairs, as a fixture timeline of ``[label, event]`` pairs."""
    labels: dict[str, str] = {}
    timeline = []
    for _, event in events:
        form = normal_form(event, labels, texts)
        response = event.get("response") if isinstance(event.get("response"), dict) else {}
        label = labels.get(response.get("id")) or labels.get(event.get("response_id"))
        if form is not None and label:
            timeline.append([label, form])
    return timeline


# ------------------------------------------------------------------------------------- writing events (a replay)
def official_events(scenario: dict, texts: Mapping[str, str] | None = None) -> list[tuple[str, dict]]:
    """The scenario as official events, ``(delegation_id, event)`` in order, to replay into a reader.

    Every event carries ``response_id`` besides its official fields, so item events can be labeled when they are
    read back; a message marked ``[key]`` carries that line of ``texts`` (or the key itself).
    """
    ids = {label: f"resp_{label.lower()}" for label in scenario["responses"]}
    outputs: dict[str, list[dict]] = {label: [] for label in scenario["responses"]}
    events = []
    for label, form in scenario["timeline"]:
        delegation = scenario["responses"][label]
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
            item = _item(words, len(outputs[label]), texts)
            outputs[label].append(item)
            event = {"type": "response.output_item.done", "output_index": len(outputs[label]) - 1, "item": item}
        elif form == "completed":
            event = {
                "type": "response.completed",
                "response": {**snapshot, "status": "completed", "output": outputs[label]},
            }
        elif words[0] == "failed":
            error = {"code": words[1], "message": "The backend failed."}
            event = {"type": "response.failed", "response": {**snapshot, "status": "failed", "error": error}}
        elif form == "incomplete cancelled":
            event = {
                "type": "response.incomplete",
                "response": {**snapshot, "status": "cancelled", "incomplete_details": None},
            }
        else:
            details = {"reason": words[1]}
            event = {
                "type": "response.incomplete",
                "response": {**snapshot, "status": "incomplete", "incomplete_details": details},
            }
        events.append((delegation, {**event, "response_id": ids[label]}))
    return events


def _item(words: list[str], index: int, texts: Mapping[str, str] | None) -> dict:
    item_id = f"item_{index}"
    if words[0] == "message":
        key = words[2].strip("[]") if len(words) > 2 else None
        text = (texts or {}).get(key, key) if key else ("Done." if words[1] == "final_answer" else "Working on it.")
        content = [{"type": "output_text", "text": text, "annotations": []}]
        return {
            "id": item_id,
            "type": "message",
            "role": "assistant",
            "status": "completed",
            "phase": words[1],
            "content": content,
        }
    if words[0] == "function_call":
        return {
            "id": item_id,
            "type": "function_call",
            "status": "completed",
            "call_id": f"call_{index}",
            "name": "get_menu",
            "arguments": "{}",
        }
    label, name = words[1].split("/")
    return {
        "id": item_id,
        "type": "mcp_call",
        "server_label": label,
        "name": name,
        "arguments": "{}",
        "output": "{}",
        "status": words[2],
        "error": None,
    }


# --------------------------------------------------------------------------------------------- the wire rules
def timeline_problems(scenario: dict) -> list[str]:
    """What in a scenario breaks the wire's rules (empty: none)."""
    problems = []
    timeline, responses = scenario["timeline"], scenario["responses"]
    if {label for label, _ in timeline} != set(responses):
        problems.append("the timeline and the responses name different labels")
    ended_at = {label: i for i, (label, form) in enumerate(timeline) if form.split(" ")[0] in TERMINAL_WORDS}
    for label in responses:
        forms = [(i, form) for i, (each, form) in enumerate(timeline) if each == label]
        problems += [f"{label}: {problem}" for problem in _response_problems(label, forms, responses, ended_at)]
    reader = scenario["reader"]
    if set(reader) - set(READER_KEYS):
        problems.append(f"unknown reader keys {sorted(set(reader) - set(READER_KEYS))}")
    if set(reader.get("endings", {}).values()) - set(ENDINGS):
        problems.append(f"unknown endings {sorted(set(reader['endings'].values()) - set(ENDINGS))}")
    return problems


def _response_problems(label: str, forms: list[tuple[int, str]], responses: dict, ended_at: dict) -> list[str]:
    problems = []
    words = [form.split(" ") for _, form in forms]
    if not words or words[0][0] != "created":
        return ["it does not open with response.created"]
    if words[0][1] == "queued" and (len(words) < 2 or words[1] != ["queued"]):
        problems.append("a queued opening is not followed by response.queued")
    terminals = [i for i, word in enumerate(words) if word[0] in TERMINAL_WORDS]
    if len(terminals) != 1 or terminals[0] != len(words) - 1:
        problems.append("it has not exactly one terminal, last")
    running = False
    called = False
    for (position, form), word in zip(forms, words, strict=True):
        if word[0] == "in_progress":
            running = True
            if len(word) > 1:
                taken = word[1].removeprefix("previous=")
                if taken not in responses or responses[taken] == responses[label]:
                    problems.append(f"{form}: a takeover names another delegation's response")
                elif ended_at.get(taken, -1) < position:
                    problems.append(f"{form}: the taker must be named before the older response ends")
        elif word[0] in ITEM_WORDS:
            if not running:
                problems.append(f"{form}: an item before response.in_progress")
            if called and word[0] != "function_call":
                problems.append(f"{form}: only the app's calls may follow a call")
            if word[0] == "function_call":
                called = True
            if word[0] == "message" and (len(word) < 2 or word[1] not in ("commentary", "final_answer")):
                problems.append(f"{form}: a message without its phase")
    return problems
