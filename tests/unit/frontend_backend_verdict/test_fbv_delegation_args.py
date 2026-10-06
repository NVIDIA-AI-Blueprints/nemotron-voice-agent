# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D100, D101, D102, D103, D107

"""Delegation tool schema and argument normalization."""

from __future__ import annotations

import unittest

from _fbv_fakes import FakeChatClient, delegate_response, make_agent, text_response, tool_response

from examples.frontend_backend_verdict.text import events
from examples.frontend_backend_verdict.text.delegation import CALL_BACKEND_TOOL, FRONTEND_TOOLS
from examples.frontend_backend_verdict.text.messages import ToolCall, canonical_json
from examples.frontend_backend_verdict.text.tools import decode_arguments


def test_only_delegation_tool_is_published() -> None:
    assert [tool["function"]["name"] for tool in FRONTEND_TOOLS] == ["call_backend"]


def test_schema_accepts_only_query_and_filler() -> None:
    properties = CALL_BACKEND_TOOL["function"]["parameters"]["properties"]
    assert set(properties) == {"query", "filler_text"}
    # Both are declared required so the frontend model reliably emits a filler phrase; the runtime
    # still tolerates a missing filler_text (see test_delegation_without_filler_still_succeeds).
    assert CALL_BACKEND_TOOL["function"]["parameters"]["required"] == ["query", "filler_text"]
    assert CALL_BACKEND_TOOL["function"]["parameters"]["additionalProperties"] is False


def test_no_cancel_tool_exists() -> None:
    names = {tool["function"]["name"] for tool in FRONTEND_TOOLS}
    assert not any("cancel" in name or "discard" in name for name in names)


def test_decode_arguments_unwraps_original_args() -> None:
    call = ToolCall(id="x", name="call_backend", arguments_json=canonical_json({"original_args": '{"query": "hi"}'}))
    assert decode_arguments(call) == {"query": "hi"}


def test_decode_arguments_passes_normal_payloads_through() -> None:
    call = ToolCall.create("call_backend", {"query": "hi", "filler_text": "one sec"})
    assert decode_arguments(call) == {"query": "hi", "filler_text": "one sec"}


class DelegationArgumentTests(unittest.IsolatedAsyncioTestCase):
    async def test_wrapped_arguments_still_delegate(self) -> None:
        frontend = FakeChatClient([tool_response(("call_backend", {"original_args": '{"query": "wrapped request"}'}))])
        backend = FakeChatClient([text_response("answered")])
        agent, _ = make_agent(frontend=frontend, backend=backend)
        turn, _ = await agent.send("go", agent.new_session())
        assert turn.final_text == "answered"
        assert [m.content for m in backend.last_messages if m.role == "user"] == ["wrapped request"]

    async def test_delegation_without_filler_still_succeeds(self) -> None:
        """filler_text is required in the schema to steer the model, not enforced at runtime."""
        frontend = FakeChatClient([delegate_response("q")])
        backend = FakeChatClient([text_response("answered")])
        agent, sink = make_agent(frontend=frontend, backend=backend)
        turn, _ = await agent.send("go", agent.new_session())
        assert turn.final_text == "answered"
        assert sink.of_kind(events.FILLER) == []

    async def test_extra_arguments_are_ignored(self) -> None:
        frontend = FakeChatClient([tool_response(("call_backend", {"query": "q", "origin_city": "X"}))])
        backend = FakeChatClient([text_response("fine")])
        agent, _ = make_agent(frontend=frontend, backend=backend)
        turn, _ = await agent.send("go", agent.new_session())
        assert turn.final_text == "fine"
