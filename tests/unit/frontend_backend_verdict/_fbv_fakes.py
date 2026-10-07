# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D100, D101, D102, D103, D107

"""Offline doubles for the text Frontend/Backend Agent prototype tests."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from examples.frontend_backend_verdict.text.agent import FrontendBackendAgent, assemble_agent
from examples.frontend_backend_verdict.text.config import Config, build_config
from examples.frontend_backend_verdict.text.events import CollectingSink
from examples.frontend_backend_verdict.text.llm import ChatResponse
from examples.frontend_backend_verdict.text.messages import Message, ToolCall, Usage, canonical_json
from examples.frontend_backend_verdict.text.tools import ToolSpec

FRONTEND_PROMPT = "You are the frontend. {persona} {capabilities} {unsupported_reply}"
BACKEND_PROMPT = "You are the backend. {domain_policy}"

#: Inline stand-ins for the backend-history catalog keys (unused unless the flag is on).
HISTORY_PROMPTS = {
    "backend_history_request": "{earlier_turns}USER SAID:\n{user_message}\n\nFRONTEND ASKS:\n{query}",
    **{
        f"backend_history_context_{include}": f"CONTEXT {include}"
        for include in ("full", "backend_turns", "transcript")
    },
    **{
        f"backend_history_guidance_{include}": f"GUIDANCE {include}"
        for include in ("full", "backend_turns", "transcript")
    },
}


class FakeChatClient:
    """Replays scripted responses and records every request it was given."""

    def __init__(self, responses: Sequence[ChatResponse] | None = None) -> None:
        self.responses: list[ChatResponse] = list(responses or [])
        self.calls: list[dict[str, Any]] = []

    def queue(self, *responses: ChatResponse) -> FakeChatClient:
        self.responses.extend(responses)
        return self

    async def complete(
        self,
        *,
        messages: Sequence[Message],
        tools: Sequence[dict[str, Any]] | None = None,
    ) -> ChatResponse:
        self.calls.append({"messages": list(messages), "tools": list(tools or [])})
        if not self.responses:
            raise AssertionError("FakeChatClient ran out of scripted responses")
        return self.responses.pop(0)

    @property
    def last_messages(self) -> list[Message]:
        return self.calls[-1]["messages"]

    @property
    def last_tools(self) -> list[dict[str, Any]]:
        return self.calls[-1]["tools"]


def text_response(text: str, *, cost: float | None = 0.001, tokens: int = 10) -> ChatResponse:
    return ChatResponse(
        content=text,
        usage=Usage(prompt_tokens=tokens, completion_tokens=tokens, total_tokens=tokens * 2),
        cost=cost,
        latency_ms=1.0,
        model="fake",
    )


def tool_response(
    *calls: tuple[str, dict[str, Any]],
    ids: Sequence[str] | None = None,
    content: str | None = None,
    cost: float | None = 0.001,
) -> ChatResponse:
    resolved_ids = list(ids or [f"call_{index}" for index in range(len(calls))])
    return ChatResponse(
        content=content,
        tool_calls=tuple(
            ToolCall(id=resolved_ids[index], name=name, arguments_json=canonical_json(arguments))
            for index, (name, arguments) in enumerate(calls)
        ),
        usage=Usage(prompt_tokens=10, completion_tokens=10, total_tokens=20),
        cost=cost,
        latency_ms=1.0,
        model="fake",
    )


def delegate_response(query: str, filler: str = "", *, task: str | None = None, **kwargs: Any) -> ChatResponse:
    arguments: dict[str, Any] = {"query": query}
    if filler:
        arguments["filler_text"] = filler
    if task is not None:
        arguments["task"] = task
    return tool_response(("call_backend", arguments), ids=["call_delegate"], **kwargs)


def make_config(**overrides: Any) -> Config:
    """Build a Config with inline prompts so no file is touched."""
    raw: dict[str, Any] = {
        "agent": {"name": "Tester", "persona": "You are a tester.", "mode": overrides.pop("mode", "frontend_backend")},
        "prompts": {
            "path": "does-not-exist.yaml",
            "inline": {"frontend": FRONTEND_PROMPT, "backend": BACKEND_PROMPT, **HISTORY_PROMPTS},
        },
        "frontend": {
            "llm": {"model": "fake-frontend"},
            "delegation": overrides.pop("delegation", {}),
            "history": overrides.pop("frontend_history", {}),
        },
        "backend": {
            "llm": {"model": "fake-backend"},
            "tools": overrides.pop("tools_config", {}),
            "history": overrides.pop("backend_history", {}),
            "conversation_history": overrides.pop("conversation_history", {}),
        },
        "domain": overrides.pop("domain", {"policy": "Be helpful.", "capabilities": ["testing"]}),
        "logging": {"event_sink": "none"},
    }
    for key, value in overrides.items():
        raw[key] = value
    return build_config(raw)


def make_agent(
    *,
    frontend: FakeChatClient | None = None,
    backend: FakeChatClient | None = None,
    tools: Sequence[ToolSpec] = (),
    sink: CollectingSink | None = None,
    config: Config | None = None,
    **config_overrides: Any,
) -> tuple[FrontendBackendAgent, CollectingSink]:
    resolved_config = config or make_config(**config_overrides)
    resolved_sink = sink or CollectingSink()
    agent = assemble_agent(
        resolved_config,
        tools=tools,
        event_sink=resolved_sink,
        frontend_client=frontend or FakeChatClient(),
        backend_client=backend or FakeChatClient(),
    )
    return agent, resolved_sink


def echo_tool(name: str = "lookup", result: Any = None) -> ToolSpec:
    payload = {"ok": True} if result is None else result

    def _call(**kwargs: Any) -> Any:
        return payload

    return ToolSpec(
        name=name,
        description=f"{name} tool",
        parameters={"type": "object", "properties": {"value": {"type": "string"}}},
        callable=_call,
    )
