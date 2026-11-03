# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""What the agent reads, as text: a conductor's input, a worker's first prompt, a steering batch.

An agent takes prompts as text, so the IM's Responses items are written out as a transcript, each kind of item by its
template in ``backend.harness.prompts``. A worker's first prompt is the task's ``history()``; a conductor's input is the
new requests, the work records and the conversation its session has not read. The conductor's instructions are never
here: the agent class places them with its session.
"""

from __future__ import annotations

from collections.abc import Sequence

from delegation_core.backend.port import ConductorInput, Request, Steering, TaskSummary, WorkSummary

from .settings import HarnessPrompts

# How much of one request's text a work record shows: its newest words, as the IM's records do.
RECORD_TEXT_CHARS = 300


def content_text(content: object) -> str:
    """The text of a message's content: a string, or the text of its parts."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(str(part.get("text") or "") for part in content if isinstance(part, dict) and part.get("text"))
    return ""


def request_text(request: Request) -> str:
    """The words of a request's user messages."""
    return "\n".join(
        text
        for item in request.input
        if item.get("role") == "user" and (text := content_text(item.get("content")).strip())
    )


class Renderer:
    """Writes the IM's items as the agent's prompt text, by the harness's templates."""

    def __init__(self, prompts: HarnessPrompts):
        """Use ``prompts``'s templates."""
        self.prompts = prompts

    # ------------------------------------------------------------------------------------------- a transcript
    def transcript(self, items: Sequence[dict]) -> str:
        """Each item on its own paragraph: messages, calls and their outputs (an output names its call)."""
        names: dict[str, str] = {}
        parts = []
        for item in items:
            line = self._item(item, names)
            if line:
                parts.append(line)
        return "\n\n".join(parts)

    def _item(self, item: dict, names: dict[str, str]) -> str:
        kind = item.get("type", "message")
        if kind == "function_call":
            names[item.get("call_id", "")] = item.get("name", "")
            return self.prompts.call.format(name=item.get("name", ""), arguments=item.get("arguments") or "{}")
        if kind == "function_call_output":
            name = names.get(item.get("call_id", ""), self.prompts.unnamed_call)
            return self.prompts.output.format(name=name, output=item.get("output") or "")
        if kind != "message":
            return ""
        text = content_text(item.get("content")).strip()
        if not text:
            return ""
        template = self.prompts.assistant if item.get("role") == "assistant" else self.prompts.user
        return template.format(text=text)

    def worker_input(self, items: Sequence[dict]) -> str:
        """A worker's first prompt: the task's view of the conversation."""
        return self.prompts.worker_input.format(transcript=self.transcript(items))

    def steering(self, steering: Steering) -> str:
        """A steering batch as text: its requests with the records."""
        return self.transcript(steering.items)

    # ------------------------------------------------------------------------------------------- a conductor
    def conductor_input(self, given: ConductorInput, *, fresh: bool) -> str:
        """One ``decide()``'s input; ``fresh``: the session has read nothing of the conversation yet."""
        conversation = given.conversation if fresh else given.conversation[given.new_from :]
        arrivals = "\n\n".join(
            self.prompts.arrival.format(delegation=request.delegation_id, text=request_text(request))
            for request in given.arrivals
        )
        works = "\n".join(self._work(work) for work in given.works)
        return self.prompts.conductor_input.format(
            arrivals=arrivals or self.prompts.none,
            works=works or self.prompts.none,
            conversation=self.transcript(conversation) or self.prompts.none,
        )

    def _work(self, work: WorkSummary) -> str:
        requests = "; ".join(
            f"[{request.delegation_id}] {request_text(request)[-RECORD_TEXT_CHARS:]}" for request in work.requests
        )
        return self.prompts.work.format(
            work=work.work_id,
            state=work.state,
            carrier=work.carrier,
            requests=requests or self.prompts.none,
            tasks="\n".join(self._task(task) for task in work.tasks),
        )

    def _task(self, task: TaskSummary) -> str:
        return self.prompts.task.format(
            task=task.task_id,
            state=task.state,
            seconds=round(task.seconds),
            brief=self.prompts.brief.format(brief=task.brief) if task.brief else "",
            steps=", ".join(task.steps) or self.prompts.none,
            note=task.progress or self.prompts.none,
        )
