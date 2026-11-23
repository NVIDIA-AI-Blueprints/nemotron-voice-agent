# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The browser's prompt layout for the configured Live model calls."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from nemotron_voice_agent.pipecat.examples.frontend_backend_live.talker import Layout, PromptPart, RoutingTexts

if TYPE_CHECKING:
    from nemotron_voice_agent.pipecat.examples.frontend_backend_live.prompts import PromptSet

EDITABLE_FRONTEND = {
    "Cascade.routing",
    "Cascade.tool_routing",
    "Cascade.commentary",
    "Cascade.answer",
    "Cascade.routing_question",
    "Cascade.progress",
}
FRONTEND_LABELS = {
    "Cascade.frame_before": "Talker role",
    "Cascade.tool_frame_before": "Talker role",
    "Cascade.frame_after": "End of Talker app instructions",
    "Cascade.writer_frame_before": "Writer role",
    "Cascade.writer_frame_after": "End of Writer app instructions",
    "Cascade.routing": "Routing policy",
    "Cascade.tool_routing": "Routing policy",
    "Cascade.commentary": "Shared reply guidance",
    "Cascade.answer": "Answer guidance",
    "Cascade.progress": "Progress guidance",
    "Cascade.routing_question": "Routing question guidance",
    "Cascade.json_rules": "JSON reply rules",
    "Cascade.tool_rules": "Tool call rules",
}


def frontend_layout(prompts: PromptSet, *, routing_entries: tuple[str, ...], channel: str) -> list[dict[str, Any]]:
    """Read the shared instruction builders in the same order used by the Talker and Writer."""
    layout = Layout(prompts, prompts.json_rules)
    layout.routing_texts = RoutingTexts.named(prompts, *routing_entries)
    instructions = [PromptPart("service", ""), PromptPart("active", ""), PromptPart(f"Channel.{channel}", "")]

    def parts(items: list[PromptPart]) -> list[dict[str, Any]]:
        result = []
        for part in items:
            key = part.source.removeprefix("agent_prompts.")
            if key == "service":
                result.append({"label": "Service instructions, when configured"})
            elif key == "active":
                result.append({"active": True, "label": "Session preset", "editable": True})
            elif key == f"Channel.{channel}":
                result.append({"key": key, "label": "Shared channel context", "editable": True})
            else:
                result.append({"key": key, "label": FRONTEND_LABELS[key], "editable": key in EDITABLE_FRONTEND})
        return result

    history = {"label": "Conversation history"}
    answered = [
        *parts(layout.answer_parts(instructions)),
        history,
        {"label": "Earlier speech and newer caller requests, when relevant"},
        {"label": "Latest caller turn and current backend result, when available"},
        {"key": "Lines.intro", "label": "Startup greeting instruction, when greeting"},
    ]
    groups = [
        {
            "id": "talker",
            "label": "Frontend · Talker routing",
            "parts": [
                *parts([*layout.routing_parts(instructions), layout.routing_texts.rules]),
                history,
                {"label": "Current work, latest caller turn and any interruption note"},
            ],
        },
        {"id": "writer-answer", "label": "Frontend · Writer answer", "parts": answered},
        {
            "id": "writer-progress",
            "label": "Frontend · Writer progress",
            "parts": [
                *parts(layout.progress_parts(instructions)),
                history,
                {"label": "Prior speech context and the current work update"},
            ],
        },
    ]
    if prompts.routing_question:
        groups.insert(
            2,
            {
                "id": "writer-routing-question",
                "label": "Frontend · Writer routing question",
                "parts": [
                    *parts(layout.answer_parts(instructions, routing_question=True)),
                    {"label": "Earlier speech for this question and the current routing question"},
                ],
            },
        )
    return groups


def backend_layout(
    *, prompt: str, channel: str, result_format: str, model_conductor: bool, harness: bool
) -> list[dict[str, Any]]:
    """Show the worker's instructions and the optional model call that assigns work."""
    groups: list[dict[str, Any]] = []
    if model_conductor:
        decision_input = (
            {
                "key": "Core.backend.harness.prompts.conductor_input",
                "label": "Queued requests and work records template",
            }
            if harness
            else {"label": "Decision history, queued requests, work records and current capacity"}
        )
        groups.append(
            {
                "id": "conductor",
                "label": "Backend · Conductor",
                "parts": [
                    {"key": "Core.texts.conductor_prompt", "label": "Routing instructions and configured capacity"},
                    decision_input,
                    {"key": "Core.texts.control_tools.submit_decision", "label": "Decision tool schema"},
                ],
            }
        )
    worker_input = (
        {"key": "Core.backend.harness.prompts.worker_input", "label": "Worker transcript template"}
        if harness
        else {"label": "The assigned request, its conversation context and tool results"}
    )
    worker = [
        {"key": f"Backend.{prompt}", "label": "Worker instructions", "editable": True},
        {"key": f"Channel.{channel}", "editable": True},
        {"key": "Core.texts.caller_note", "label": "Caller context rules"},
        {"key": "Core.texts.voice_note", "label": "Voice channel rules"},
        {"key": f"Core.texts.result_formats.{result_format}", "label": "Result format rules"},
    ]
    if not harness:
        worker.append(
            {"key": "Core.texts.round_cap_note", "label": "Instruction for the last tool round, when reached"}
        )
    worker.extend([worker_input, {"label": "Tools available to this worker"}])
    groups.append({"id": "worker", "label": "Backend · Worker", "parts": worker})
    return groups
