# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""What the agent reads, as text: a conductor's input, a worker's first prompt and a steering batch."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

from nemotron_voice_agent.pipecat.examples.frontend_backend_live.delegation_core.backend.port import (
    ConductorInput,
    Request,
    Steering,
    TaskSummary,
    WorkSummary,
)
from nemotron_voice_agent.pipecat.examples.frontend_backend_live.delegation_core.items import (
    WorkerData,
    is_output_content,
)
from nemotron_voice_agent.pipecat.shared.delegation_contract import CALLER_EXCERPT_CHARS

from .settings import HarnessPrompts


def content_text(content: object) -> str:
    """Return the text of a message's content: a string, or the text of its parts."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(str(part.get("text") or "") for part in content if isinstance(part, dict) and part.get("text"))
    return ""


def output_text(item: Mapping[str, Any] | None) -> str | None:
    """Return the text of a ``function_call_output``: a string as given, the text of content parts, else JSON."""
    if item is None:
        return None
    output = item.get("output")
    return content_text(output) if isinstance(output, str) or is_output_content(output) else json.dumps(output)


def request_text(request: Request) -> str:
    """Return the words of a request's user messages."""
    return "\n".join(
        text
        for item in request.input
        if item.get("role") == "user" and (text := content_text(item.get("content")).strip())
    )


class Renderer:
    """The writer of the IM's items as the agent's prompt text, by the ``backend.harness.prompts`` templates."""

    def __init__(self, prompts: HarnessPrompts):
        """Use ``prompts``'s templates."""
        self.prompts = prompts

    def transcript(self, items: Sequence[dict]) -> str:
        """Write ``items`` as a transcript: one paragraph per message, call or output, an output naming its call."""
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
            return self.prompts.output.format(name=name, output=output_text(item))
        if kind != "message":
            return ""
        text = content_text(item.get("content")).strip()
        if not text or isinstance(item, WorkerData):
            return text
        role = item.get("role")
        if role == "developer":
            return json.dumps({"role": "developer", "id": item.get("id"), "content": text}, ensure_ascii=False)
        template = self.prompts.assistant if role == "assistant" else self.prompts.user
        return template.format(text=text)

    def worker_input(self, items: Sequence[dict]) -> str:
        """Write a worker's first prompt: the task's view of the conversation."""
        return self.prompts.worker_input.format(transcript=self.transcript(items))

    def steering(self, steering: Steering) -> str:
        """Write a steering batch as text: its requests with the records."""
        return self.transcript(steering.items)

    def conductor_input(self, given: ConductorInput, *, fresh: bool) -> str:
        """Write one ``decide()``'s input; ``fresh`` when the session has read nothing of the conversation."""
        conversation = given.conversation if fresh else given.conversation[given.new_from :]
        requests = []
        for request in given.arrivals:
            text = request_text(request)
            if request.caller_lines:
                text = json.dumps({"source_lines": request.caller_lines}, ensure_ascii=False) + "\n" + text
            requests.append(self.prompts.arrival.format(delegation=request.delegation_id, text=text))
        arrivals = "\n\n".join(requests)
        works = "\n".join(
            [json.dumps({"routing_targets": given.routing_targets()}), *(self._work(work) for work in given.works)]
        )
        if given.available_worker_slots is not None:
            works += "\n" + json.dumps({"available_worker_slots": given.available_worker_slots})
        if given.attempt > 1:
            works += "\n" + json.dumps({"placement_attempt": given.attempt})
        if given.decision_rejection is not None:
            works += "\n" + json.dumps({"decision_rejection": given.decision_rejection})
        if given.clarifications:
            works += "\n" + json.dumps({"pending_clarifications": given.clarifications})
        if given.control_actions:
            works += "\n" + json.dumps({"control_actions": given.control_actions})
        return self.prompts.conductor_input.format(
            arrivals=arrivals or self.prompts.none,
            works=works or self.prompts.none,
            conversation=self.transcript(conversation) or self.prompts.none,
        )

    def _work(self, work: WorkSummary) -> str:
        requests = "; ".join(
            json.dumps(
                {
                    "delegation_id": request.delegation_id,
                    "delivery": request.delivery,
                    "text": (
                        request_text(request)[-CALLER_EXCERPT_CHARS:]
                        if request.caller_excerpt is None
                        else request.caller_excerpt
                    ),
                },
                ensure_ascii=False,
            )
            for request in work.requests
        )
        return self.prompts.work.format(
            work=work.work_id,
            state=work.state,
            after_call=self.prompts.after_call if work.after_call else "",
            carrier=work.carrier,
            requests=requests or self.prompts.none,
            tasks="\n".join(self._task(task) for task in work.tasks),
        )

    def _task(self, task: TaskSummary) -> str:
        excerpt = task.answer if task.answer is not None else task.answer_end
        state = {
            "state": task.state,
            "stop": task.stop,
            "answer_publication": task.answer_publication,
            "tool_results": task.tool_results,
            "tool_results_omitted": task.tool_results_omitted,
        }
        for name in ("continues", "successor_task_id", "successor_work_id", "steering"):
            value = getattr(task, name)
            if value is not None:
                state[name] = value
        return self.prompts.task.format(
            task=task.task_id,
            state=json.dumps(state),
            seconds=round(task.seconds),
            brief=self.prompts.brief.format(brief=task.brief) if task.brief else "",
            question=self.prompts.question.format(question=json.dumps(task.questions, ensure_ascii=False))
            if task.questions
            else self.prompts.question.format(question=task.question)
            if task.question
            else "",
            steps=", ".join(task.steps) or self.prompts.none,
            note=task.progress or self.prompts.none,
            answer=self.prompts.answer.format(answer=excerpt) if excerpt else "",
        )
