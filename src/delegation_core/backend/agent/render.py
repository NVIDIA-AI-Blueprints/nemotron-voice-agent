# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""What the agent's conductor model reads for one ``decide()``: the new conversation, the work records, the requests.

The input is one user message whose text is one JSON object, so every instruction the model reads stays in
``texts.conductor_prompt``. Its keys:

* ``conversation``: the messages this conductor was not given before, each as its role and text;
* ``works``: every open Work, then the recently ended ones, each with its requests and tasks;
* ``new_requests``: the requests to place, oldest first, each with its delegation id and its messages.

The ids are those the control tools take (``work_id``, ``task_id``, ``delegations``, ``delegation_id``).
"""

from __future__ import annotations

import json
from collections.abc import Iterable

from delegation_core.backend.port import ConductorInput, Request, TaskSummary, WorkSummary
from delegation_core.backend.providers.translation import item_text

# Seconds are shown to a tenth.
SECONDS_DIGITS = 1


def said(items: Iterable[dict]) -> list[dict]:
    """The messages among ``items``, each as its role and its text; any other item, and an empty message, left out."""
    messages = []
    for item in items:
        if item.get("type", "message") != "message":
            continue
        text = item_text(item.get("content")).strip()
        if text:
            messages.append({"role": item.get("role", "user"), "text": text})
    return messages


def request(given: Request) -> dict:
    """One request: its delegation id and its messages."""
    return {"delegation_id": given.delegation_id, "messages": said(given.input)}


def task(summary: TaskSummary) -> dict:
    """One task of a Work, as its record says."""
    return {
        "task_id": summary.task_id,
        "state": summary.state,
        "brief": summary.brief,
        "steps": list(summary.steps),
        "acted": summary.acted,
        "progress": summary.progress,
        "seconds": round(summary.seconds, SECONDS_DIGITS),
    }


def work(summary: WorkSummary) -> dict:
    """One Work, open or recently ended, with the requests it carries and its tasks."""
    return {
        "work_id": summary.work_id,
        "state": summary.state,
        "requests": [request(each) for each in summary.requests],
        "tasks": [task(each) for each in summary.tasks],
        "seconds": round(summary.seconds, SECONDS_DIGITS),
    }


def round_input(given: ConductorInput) -> dict:
    """The user message of one decision round: what is new in the conversation, the work records, the new requests."""
    data = {
        "conversation": said(given.conversation[given.new_from :]),
        "works": [work(each) for each in given.works],
        "new_requests": [request(each) for each in given.arrivals],
    }
    text = json.dumps(data, ensure_ascii=False)
    return {"type": "message", "role": "user", "content": [{"type": "input_text", "text": text}]}
