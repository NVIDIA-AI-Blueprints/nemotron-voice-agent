# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Private bounded frontend text for replacing an audio attachment, without replaying audio or actions."""

from __future__ import annotations

import json
from dataclasses import asdict
from hashlib import sha256
from typing import TYPE_CHECKING

from examples.frontend_backend_live.heard_log import Heard, Noted
from examples.frontend_backend_live.spoken import Line, LineKind, Outcome

if TYPE_CHECKING:
    from examples.frontend_backend_live.voice import Voice

MAX_ATTACHMENT_BYTES = 256 * 1024
MAX_ATTACHMENT_ITEMS = 512


def _json_bytes(value: dict) -> int:
    return len(json.dumps(value, ensure_ascii=False).encode())


def _confirmed_requests(messages: list[dict]) -> dict[str, str]:
    """Recover only complete caller wording from retained records of an accepted handoff."""
    grouped: dict[str, list[dict]] = {}
    for row in messages:
        delivery = row.get("delivery") or {}
        source = delivery.get("delegation_id")
        if row["kind"] == "caller" and delivery.get("state") == "accepted" and isinstance(source, str) and source:
            grouped.setdefault(source, []).append(row)
    confirmed = {}
    for source, parts in grouped.items():
        wording = " ".join(row["content"] for row in parts)
        digest = sha256(wording.encode()).hexdigest()
        if all(row["delivery"].get("request_digest") == digest for row in parts):
            confirmed[source] = wording
    return confirmed


def _retained_messages(messages: list[dict], snapshot: dict) -> list[dict]:
    """Give pending input and its exact related tool rows priority within both snapshot limits."""
    # Charge the intervening `, ` separators with each row, then credit the first row's separator once.
    costs = [_json_bytes(row) + 2 for row in messages]
    room = MAX_ATTACHMENT_BYTES - _json_bytes(snapshot)
    if len(messages) <= MAX_ATTACHMENT_ITEMS and sum(costs) - (2 if costs else 0) <= room:
        return messages

    required = set()
    call_ids: list[set[str]] = []
    rows_by_call: dict[str, list[int]] = {}
    for index, row in enumerate(messages):
        delivery = row["delivery"] or {}
        if (row["kind"] == "caller" and not row["decided"]) or delivery.get("state") in {
            "never_forwarded",
            "unknown",
            "attempted",
        }:
            required.add(index)
        fields = row["call_fields"]
        linked = {call["id"] for call in fields.get("tool_calls", ())}
        linked.update(key for key in (fields.get("tool_call_id"), delivery.get("call_id")) if key is not None)
        call_ids.append(linked)
        for key in linked:
            rows_by_call.setdefault(key, []).append(index)

    def complete_group(initial: set[int]) -> set[int]:
        selected, pending = set(initial), list(initial)
        visited = set()
        while pending:
            for key in call_ids[pending.pop()] - visited:
                visited.add(key)
                for index in rows_by_call[key]:
                    if index not in selected:
                        selected.add(index)
                        pending.append(index)
        return selected

    selected = complete_group(required)
    if len(selected) > MAX_ATTACHMENT_ITEMS:
        raise ValueError("Pending frontend input and tool evidence exceeds the attachment item limit")
    # The snapshot already contains the empty array. Reserve the omission marker before adding any rows.
    remaining = MAX_ATTACHMENT_BYTES - _json_bytes({**snapshot, "messages_omitted": True})
    remaining += 2 - sum(costs[index] for index in selected)
    if remaining < 0:
        raise ValueError("Required frontend attachment state exceeds 256 KiB")
    considered = set(selected)
    for index in reversed(range(len(messages))):
        if index in considered:
            continue
        group = complete_group({index})
        considered.update(group)
        cost = sum(costs[member] for member in group)
        if cost <= remaining and len(selected) + len(group) <= MAX_ATTACHMENT_ITEMS:
            selected.update(group)
            remaining -= cost
    return [row for index, row in enumerate(messages) if index in selected]


def export_attachment(voice: Voice) -> dict:
    """Copy retained text; unsettled speech remains explicitly uncertain, never queued to play again."""
    flights = list(voice.spoken.in_flight_lines())
    in_flight = {line.number for line in flights}
    uncertain = list(dict.fromkeys(line.text for line in flights if line.text))
    # The retained port decides which final items were never forwarded. Voice does not repeat those it received.
    pending = [
        *voice.owed,
        *getattr(getattr(voice.call, "plan", None), "batch", ()),
        *(item for line in voice.waiting_lines for item in line.items),
    ]
    uncertain.extend(item.said.speak for item in pending if hasattr(item, "said") and item.said.speak)
    uncertain.extend(line.text for line in voice.waiting_lines if line.text)
    uncertain = list(dict.fromkeys(uncertain))
    # Assistant context contains the text the playout aggregator has already reported. A pending
    # line can repeat words from an earlier completed turn; text overlap does not revoke that turn.
    messages = []
    if voice.context is not None:
        voice._settle_results(voice.context)
    for message in voice._read():
        kind = voice.kind_of(message)
        delivery = voice.input_delivery.get(id(message))
        messages.append(
            {
                "content": voice.as_read(message, kind)["content"],
                "kind": kind.name,
                "decided": id(message) in voice.decided,
                "noted": id(message) in voice.noted,
                "delivery": delivery,
                "call_fields": {key: message[key] for key in ("tool_calls", "tool_call_id") if key in message},
            }
        )
    heard = [asdict(entry) for entry in voice.heard_log._carried() if entry.line not in in_flight]
    omitted_before = voice.heard_log.omitted_before
    if len(heard) > MAX_ATTACHMENT_ITEMS:
        omitted_before = max(omitted_before or 0, heard[-MAX_ATTACHMENT_ITEMS - 1]["seq"] + 1)
    snapshot = {
        "version": 3,
        "response_number": max([voice.responses, *in_flight, *(entry.line or 0 for entry in voice.heard_log.entries)]),
        "instructions": list(voice.instructions),
        "instruction_due": voice.instruction_due,
        "heard_sent": voice.heard_log._sent,
        "seed_roles": list(voice.seed_roles),
        "messages": [],
        "heard": heard[-MAX_ATTACHMENT_ITEMS:],
        "heard_omitted_before": omitted_before,
        "uncertain": uncertain[-MAX_ATTACHMENT_ITEMS:],
        "progress": voice.worknote.attachment_progress(),
        "presentation_sequence": voice.spoken._sequence,
        "presentation_cutoffs": dict(voice.spoken._work_cutoffs),
        "delegation_cutoffs": dict(voice.spoken._delegation_cutoffs),
        "presentations": [
            voice.spoken.presentation(line) for line in dict.fromkeys([*voice.spoken.presentations, *flights])
        ],
    }
    snapshot["messages"] = _retained_messages(messages, snapshot)
    if len(snapshot["messages"]) != len(messages):
        snapshot["messages_omitted"] = True
    # WorkNote/reader state comes from the retained canonical journal, never from duplicate mutable objects.
    encoded = json.dumps(snapshot, ensure_ascii=False)
    if len(encoded.encode()) > MAX_ATTACHMENT_BYTES:
        raise ValueError("Frontend attachment snapshot exceeds 256 KiB")
    return json.loads(encoded)


def restore_attachment(voice: Voice, snapshot: dict) -> None:
    """Load into a fresh, input-fenced voice; schedule no inference, speech, old tool call or work mutation."""
    from examples.frontend_backend_live.voice import APP_CONTEXT, CALL, CALL_RESULT, CALLER, RESULT, SPOKEN, TYPED

    if voice.call is not None or voice.spoken.in_flight() or voice.owed:
        raise ValueError("Frontend attachment restore requires a fresh voice")
    if snapshot.get("version") not in {2, 3} or _json_bytes(snapshot) > MAX_ATTACHMENT_BYTES:
        raise ValueError("Invalid or oversized frontend attachment snapshot")
    if any(len(snapshot.get(key, ())) > MAX_ATTACHMENT_ITEMS for key in ("messages", "heard", "uncertain")):
        raise ValueError("Frontend attachment snapshot exceeds its item limit")
    response_number = snapshot.get("response_number", 0)
    if type(response_number) is not int or response_number < 0:
        raise ValueError("Invalid frontend attachment response number")
    kinds = {kind.name: kind for kind in (CALLER, SPOKEN, CALL, CALL_RESULT, RESULT, APP_CONTEXT, TYPED)}
    voice.instructions = list(snapshot["instructions"])
    voice.seed_roles = list(snapshot["seed_roles"])
    messages = []
    for item in snapshot["messages"]:
        kind = kinds[item["kind"]]
        message = {"role": kind.role, "content": item["content"], **item.get("call_fields", {})}
        voice._register(message, kind)
        messages.append(message)
        delivery = item.get("delivery")
        if delivery is not None:
            voice.input_delivery[id(message)] = dict(delivery)
        # Only positively never-forwarded input may be routed automatically after attachment fencing.
        pending = (not item.get("decided", True)) or delivery and delivery.get("state") == "never_forwarded"
        if not pending:
            voice.decided[id(message)] = message
        if item.get("noted", False):
            voice.noted[id(message)] = message
    if snapshot.get("messages_omitted"):
        omitted = {"role": APP_CONTEXT.role, "content": voice.heard_log.words.omitted}
        if not any(voice.kind_of(message) is APP_CONTEXT and message == omitted for message in messages):
            voice._register(omitted, APP_CONTEXT)
            messages.insert(0, omitted)
    if snapshot["uncertain"]:
        note = voice.prompts.reconnect_note().format(text="\n".join(snapshot["uncertain"]))
        message = {"role": "developer", "content": voice.prompts.context_label + note}
        voice._register(message, APP_CONTEXT)
        messages.append(message)
    uncertain_inputs = [
        row
        for row in snapshot["messages"]
        if row.get("delivery") and row["delivery"].get("state") in {"unknown", "attempted"}
    ]
    if uncertain_inputs:
        message = {"role": "developer", "content": json.dumps({"private_input_delivery": uncertain_inputs})}
        voice._register(message, APP_CONTEXT)
        messages.append(message)
    # A restore preserves tool-message pairs without redispatching the existing routing call.
    present_results = {message.get("tool_call_id") for message in messages}
    paired = []
    for message in messages:
        paired.append(message)
        for call in message.get("tool_calls", ()):
            if call["id"] not in present_results:
                placeholder = {
                    "role": "tool",
                    "tool_call_id": call["id"],
                    "content": json.dumps({"status": "unknown"}),
                }
                voice._register(placeholder, CALL_RESULT)
                paired.append(placeholder)
    voice.context.set_messages(paired)
    # Text restore never decides an old caller request again or issues its action twice.
    voice.instruction_due = bool(snapshot.get("instruction_due", False))
    voice.heard_log._sent = snapshot.get("heard_sent", 0)
    voice.heard_log.omitted_before = snapshot.get("heard_omitted_before")
    voice.attachment_presentation = {
        "progress": snapshot.get("progress", []),
        "requests": _confirmed_requests(snapshot["messages"]),
        "presentations": snapshot.get("presentations", []),
        "sequence": snapshot.get("presentation_sequence", 0),
        "cutoffs": snapshot.get("presentation_cutoffs", {}),
        "delegation_cutoffs": snapshot.get("delegation_cutoffs", {}),
    }
    voice.heard_log.entries = [Noted(**{**entry, "kind": Heard(entry["kind"])}) for entry in snapshot["heard"]]
    voice.heard_log._noted = max(
        voice.heard_log._sent,
        voice.heard_log.omitted_before or 0,
        max((entry.seq for entry in voice.heard_log.entries), default=-1) + 1,
    )
    # New playout must not reuse an earlier line's identity and rewrite its retained heard-log entry.
    voice.responses = max([response_number, *(entry.line or 0 for entry in voice.heard_log.entries)])
    if snapshot["uncertain"]:
        voice.heard_log.uncertain_after_reconnect()


def resume_presentation(voice: Voice) -> None:
    """Bind minimal frontend evidence after canonical Work reconstruction; never replay old speech or actions."""
    state = getattr(voice, "attachment_presentation", None)
    if state is None:
        return
    voice.worknote.restore_request_wording(state["requests"], voice.reader.ownership)
    voice.worknote.restore_attachment_progress(state["progress"])
    voice.spoken._sequence = max(voice.spoken._sequence, state["sequence"])
    voice.spoken._work_cutoffs.update(state["cutoffs"])
    voice.spoken._delegation_cutoffs.update(state["delegation_cutoffs"])
    for row in state["presentations"]:
        refs = frozenset(row.get("context_work_refs", row["work_refs"])).intersection(voice.reader.ownership.bindings)
        line = Line(
            LineKind(row.get("kind", "result")),
            row["text"],
            delegation_ids=tuple(row.get("delegation_ids", ())),
            number=row["line_id"],
            outcome=Outcome(row["state"]),
            heard=row["recorded_sentence_prefix"],
            audio_started=row["audio_started"],
            work_refs=frozenset(row["work_refs"]),
            publication_items=tuple(row.get("results", ())),
        )
        voice.spoken._orders[line] = row.get("sequence", 0)
        voice.spoken._line_work[line] = set(refs)
        voice.spoken.presentations.append(line)
        for reference in row.get("context_delegation_ids", row.get("delegation_ids", ())):
            voice.spoken.told.setdefault(reference, []).append(line)
    del voice.attachment_presentation
