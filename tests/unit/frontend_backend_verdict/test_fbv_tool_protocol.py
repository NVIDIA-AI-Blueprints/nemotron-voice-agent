# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D100, D101, D102, D103, D107

"""Tool-call/result correlation, serialization, and pending-message policy."""

from __future__ import annotations

import json
import unittest

import pytest
from _fbv_fakes import FakeChatClient, echo_tool, make_agent, text_response, tool_response

from examples.frontend_backend_verdict.text.errors import ToolProtocolError, ToolResultSerializationError
from examples.frontend_backend_verdict.text.messages import ToolCall, ToolResult
from examples.frontend_backend_verdict.text.protocol import ensure_ids, validate_tool_results
from examples.frontend_backend_verdict.text.tools import error_payload, serialize_result

OUTSTANDING = ("call_a", "call_b")


def test_id_passthrough() -> None:
    calls = (ToolCall(id="provider-1", name="lookup"), ToolCall(id="provider-2", name="lookup"))
    assert [call.id for call in ensure_ids(calls)] == ["provider-1", "provider-2"]


def test_id_generated_only_when_absent() -> None:
    resolved = ensure_ids((ToolCall(id="", name="lookup"),))
    assert resolved[0].id.startswith("call_") and len(resolved[0].id) > 5


def test_duplicate_ids_in_one_batch_are_rejected() -> None:
    with pytest.raises(ToolProtocolError, match="duplicate tool-call id"):
        ensure_ids((ToolCall(id="same", name="a"), ToolCall(id="same", name="b")))


def test_results_are_reordered_to_emission_order() -> None:
    results = [ToolResult("call_b", "B"), ToolResult("call_a", "A")]
    assert [r.tool_call_id for r in validate_tool_results(OUTSTANDING, results)] == ["call_a", "call_b"]


def test_unknown_id_is_rejected() -> None:
    with pytest.raises(ToolProtocolError, match="unexpected tool_call_id"):
        validate_tool_results(OUTSTANDING, [ToolResult("call_z", "Z")])


def test_duplicate_result_is_rejected() -> None:
    with pytest.raises(ToolProtocolError, match="duplicate tool result"):
        validate_tool_results(OUTSTANDING, [ToolResult("call_a", "A"), ToolResult("call_a", "A")])


def test_missing_result_is_rejected_by_default() -> None:
    with pytest.raises(ToolProtocolError, match="missing tool results"):
        validate_tool_results(OUTSTANDING, [ToolResult("call_a", "A")])


def test_missing_result_can_be_synthesized() -> None:
    results = validate_tool_results(OUTSTANDING, [ToolResult("call_a", "A")], on_incomplete="synthesize_error_result")
    assert results[1].is_error
    assert json.loads(results[1].content)["error"]["message"]


def test_no_outstanding_calls_is_rejected() -> None:
    with pytest.raises(ToolProtocolError, match="no tool calls are outstanding"):
        validate_tool_results((), [])


def test_serialization() -> None:
    assert serialize_result({"b": 1, "a": 2}) == '{"a": 2, "b": 1}'
    assert serialize_result("already text") == "already text"
    assert serialize_result([1, 2]) == "[1, 2]"
    assert json.loads(error_payload(ValueError("boom")))["error"] == {"type": "ValueError", "message": "boom"}


def test_unserializable_result() -> None:
    with pytest.raises(ToolResultSerializationError):
        serialize_result(object())


def test_truncation_is_valid_json() -> None:
    long_dict = serialize_result({"k": "v" * 200}, max_result_chars=40)
    decoded = json.loads(long_dict)
    assert decoded["truncated"] is True and len(decoded["content"]) == 40
    long_text = serialize_result("t" * 100, max_result_chars=10)
    assert json.loads(long_text)["original_chars"] == 100


class PendingPolicyTests(unittest.IsolatedAsyncioTestCase):
    async def _suspended(self, **overrides):  # noqa: ANN003
        backend = FakeChatClient([tool_response(("lookup", {"value": "x"}), ids=["live"])])
        agent, sink = make_agent(
            backend=backend,
            mode="backend_only",
            tools=[echo_tool()],
            tools_config={"execution": "external", **overrides},
        )
        turn, session = await agent.send("look it up", agent.new_session())
        assert turn.is_tool_call
        return agent, backend, sink, session

    async def test_message_while_pending_errors_by_default(self) -> None:
        agent, _, _, session = await self._suspended()
        with self.assertRaises(ToolProtocolError):
            await agent.send("never mind", session)

    async def test_message_while_pending_can_discard(self) -> None:
        agent, backend, _, session = await self._suspended(on_user_message_while_pending="discard_pending")
        backend.queue(text_response("fresh answer"))
        turn, session = await agent.send("never mind", session)
        assert turn.final_text == "fresh answer"
        assert session.pending is None

    async def test_results_without_pending_are_rejected(self) -> None:
        agent, _ = make_agent(mode="backend_only", backend=FakeChatClient([text_response("hi")]))
        with self.assertRaises(ToolProtocolError):
            await agent.send_tool_results([ToolResult("x", "y")], agent.new_session())
