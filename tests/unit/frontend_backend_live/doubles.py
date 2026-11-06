# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D103, D107

"""The frontend's test doubles: the prompt set's words, the official events, and sinks that record what they hear."""

from __future__ import annotations

from examples.frontend_backend_live.prompts import load_prompts

# The example's prompt file: its prompt set, whose words the cascade writes in these tests, its words for the work
# note, which tests build the lines they expect with, and its lines.
LOADED = load_prompts()
PROMPTS, LINES = LOADED.cascade, LOADED.lines
WORK = PROMPTS.work_texts


class RecordingDisplay:
    def __init__(self):
        self.calls: list[tuple] = []

    def started(self, delegation_id, request):
        self.calls.append(("started", delegation_id, request))

    def progress(self, delegation_id, line):
        self.calls.append(("progress", delegation_id, line))

    def shown(self, delegation_id, item):
        self.calls.append(("shown", delegation_id, item.get("name")))

    def answered(self, result):
        self.calls.append(("answered", result.delegation_id, result.speak))

    def ended(self, delegation_id, how):
        self.calls.append(("ended", delegation_id, how.value))


class RecordingResults:
    """The reader's output, as the voice takes it."""

    def __init__(self):
        self.calls: list[tuple] = []

    def ready(self, result):
        self.calls.append(("ready", result.delegation_id, result.speak))

    def notice(self, delegation_id, text):
        self.calls.append(("notice", delegation_id, text))

    def ended(self, delegation_id):
        self.calls.append(("ended", delegation_id))

    def handed_over(self, delegation_id, taker):
        self.calls.append(("handed_over", delegation_id, taker))


# ----------------------------------------------------------------------------------------------- official events
def response(response_id, status, **fields):
    return {"id": response_id, "object": "response", "status": status, "output": [], **fields}


def created(response_id):
    return {"type": "response.created", "response": response(response_id, "queued")}


def queued(response_id):
    return {"type": "response.queued", "response": response(response_id, "queued")}


def in_progress(response_id, previous=None):
    extra = {"previous_response_id": previous} if previous else {}
    return {"type": "response.in_progress", "response": response(response_id, "in_progress", **extra)}


def message(text, phase="final_answer", item_id="msg_1", status="completed"):
    part = {"type": "output_text", "text": text, "annotations": []}
    item = {"id": item_id, "type": "message", "role": "assistant", "status": status, "content": [part]}
    if phase:
        item["phase"] = phase
    return {"type": "response.output_item.done", "item": item}


def function_call(name, call_id="call_1"):
    item = {"id": f"fc_{call_id}", "type": "function_call", "call_id": call_id, "name": name, "arguments": "{}"}
    return {"type": "response.output_item.done", "item": {**item, "status": "completed"}}


def mcp_call(label, name, item_id="mcp_1", output="{}", status="completed"):
    item = {"id": item_id, "type": "mcp_call", "server_label": label, "name": name, "arguments": "{}"}
    return {"type": "response.output_item.done", "item": {**item, "output": output, "status": status}}


def mcp_started(label, name, item_id="mcp_1"):
    """A backend call starts, as the IM writes it: its item ``in_progress``, with no output yet."""
    item = {"id": item_id, "type": "mcp_call", "server_label": label, "name": name, "arguments": "{}"}
    return {"type": "response.output_item.added", "item": {**item, "output": None, "status": "in_progress"}}


def completed(response_id):
    return {"type": "response.completed", "response": response(response_id, "completed")}


def failed(response_id):
    error = {"code": "server_error", "message": "The backend failed."}
    return {"type": "response.failed", "response": response(response_id, "failed", error=error)}


def incomplete(response_id, reason=None):
    if reason is None:
        return {"type": "response.incomplete", "response": response(response_id, "cancelled", incomplete_details=None)}
    details = {"reason": reason}
    return {"type": "response.incomplete", "response": response(response_id, "incomplete", incomplete_details=details)}
