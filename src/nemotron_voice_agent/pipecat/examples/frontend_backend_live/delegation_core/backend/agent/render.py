# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""What the agent's conductor model reads for one ``decide()``: one user message holding one JSON object."""

from __future__ import annotations

import json
from collections.abc import Iterable

from nemotron_voice_agent.pipecat.examples.frontend_backend_live.delegation_core.backend.port import (
    ConductorInput,
    Request,
    TaskSummary,
    WorkSummary,
)
from nemotron_voice_agent.pipecat.examples.frontend_backend_live.delegation_core.backend.providers.translation import (
    item_text,
)

SECONDS_DIGITS = 1


def said(items: Iterable[dict]) -> list[dict]:
    """Return the messages among ``items`` as role and text, leaving out other items and empty messages."""
    messages = []
    for item in items:
        if item.get("type", "message") != "message":
            continue
        text = item_text(item.get("content")).strip()
        if text:
            rendered = {"role": item.get("role", "user"), "text": text}
            if item.get("id"):
                rendered["id"] = item["id"]
            messages.append(rendered)
    return messages


def request(given: Request) -> dict:
    """Render one request: its delegation id and its messages."""
    rendered = {"delegation_id": given.delegation_id, "messages": said(given.input)}
    if given.caller_lines:
        rendered["source_lines"] = list(given.caller_lines)
    if given.delivery == "pending":
        rendered["delivery"] = "pending"
    return rendered


def task(summary: TaskSummary) -> dict:
    """Render one task of a Work from its record; its question only while it holds one, its answer only when given."""
    rendered = {
        "task_id": summary.task_id,
        "state": summary.state,
        "brief": summary.brief,
        "steps": list(summary.steps),
        "acted": summary.acted,
        "progress": summary.progress,
        "seconds": round(summary.seconds, SECONDS_DIGITS),
        "stop": summary.stop,
    }
    for name in ("continues", "successor_task_id", "successor_work_id", "steering"):
        value = getattr(summary, name)
        if value is not None:
            rendered[name] = value
    if summary.tool_results or summary.tool_results_omitted:
        rendered["tool_results"] = list(summary.tool_results)
        rendered["tool_results_omitted"] = summary.tool_results_omitted
    if summary.answer_publication != "none":
        rendered["answer_publication"] = summary.answer_publication
    if summary.question:
        rendered["question"] = summary.question
    if summary.questions:
        rendered["questions"] = list(summary.questions)
    if summary.answer:
        rendered["answer"] = summary.answer
    if summary.answer_end:
        rendered["answer_end"] = summary.answer_end
    return rendered


def work(summary: WorkSummary) -> dict:
    """Render one Work, open or recently ended, with its requests and its tasks; whether it outlives the call, if so."""
    rendered = {
        "work_id": summary.work_id,
        "state": summary.state,
        "requests": [request(each) for each in summary.requests],
        "tasks": [task(each) for each in summary.tasks],
        "seconds": round(summary.seconds, SECONDS_DIGITS),
    }
    if summary.after_call:
        rendered["continues_after_call"] = True
    return rendered


def round_input(given: ConductorInput) -> dict:
    """Build one decision round's user message: the new conversation, the work records and the new requests.

    Its text is JSON only, so every instruction the model reads stays in ``texts.conductor_prompt``.
    """
    data = {
        "conversation": said(given.conversation[given.new_from :]),
        "routing_targets": given.routing_targets(),
        "works": [work(each) for each in given.works],
        "new_requests": [request(each) for each in given.arrivals],
    }
    if given.available_worker_slots is not None:
        data["available_worker_slots"] = given.available_worker_slots
    if given.attempt > 1:
        data["placement_attempt"] = given.attempt
    if given.decision_rejection is not None:
        data["decision_rejection"] = given.decision_rejection
    if given.clarifications:
        data["pending_clarifications"] = list(given.clarifications)
    if given.control_actions:
        data["control_actions"] = list(given.control_actions)
    text = json.dumps(data, ensure_ascii=False)
    return {"type": "message", "role": "user", "content": [{"type": "input_text", "text": text}]}
