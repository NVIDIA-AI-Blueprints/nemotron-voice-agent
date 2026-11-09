# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""What the agent's conductor model reads for one ``decide()``: one user message holding one JSON object."""

from __future__ import annotations

import json
from collections.abc import Iterable

from examples.frontend_backend_live.delegation_core.backend.port import (
    ConductorInput,
    Request,
    TaskSummary,
    WorkSummary,
)
from examples.frontend_backend_live.delegation_core.backend.providers.translation import item_text

SECONDS_DIGITS = 1


def said(items: Iterable[dict]) -> list[dict]:
    """Return the messages among ``items`` as role and text, leaving out other items and empty messages."""
    messages = []
    for item in items:
        if item.get("type", "message") != "message":
            continue
        text = item_text(item.get("content")).strip()
        if text:
            messages.append({"role": item.get("role", "user"), "text": text})
    return messages


def request(given: Request) -> dict:
    """Render one request: its delegation id and its messages."""
    return {"delegation_id": given.delegation_id, "messages": said(given.input)}


def task(summary: TaskSummary) -> dict:
    """Render one task of a Work from its record."""
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
    """Render one Work, open or recently ended, with its requests and its tasks."""
    return {
        "work_id": summary.work_id,
        "state": summary.state,
        "requests": [request(each) for each in summary.requests],
        "tasks": [task(each) for each in summary.tasks],
        "seconds": round(summary.seconds, SECONDS_DIGITS),
    }


def round_input(given: ConductorInput) -> dict:
    """Build one decision round's user message: the new conversation, the work records and the new requests.

    Its text is JSON only, so every instruction the model reads stays in ``texts.conductor_prompt``.
    """
    data = {
        "conversation": said(given.conversation[given.new_from :]),
        "works": [work(each) for each in given.works],
        "new_requests": [request(each) for each in given.arrivals],
    }
    text = json.dumps(data, ensure_ascii=False)
    return {"type": "message", "role": "user", "content": [{"type": "input_text", "text": text}]}
