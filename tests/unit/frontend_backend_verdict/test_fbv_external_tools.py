# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D100, D101, D102, D103, D107

"""External execution: suspend, resume, parallel batches, sequential internal runs."""

from __future__ import annotations

import asyncio
import unittest

from _fbv_fakes import FakeChatClient, delegate_response, echo_tool, make_agent, text_response, tool_response

from examples.frontend_backend_verdict.text.messages import ToolResult
from examples.frontend_backend_verdict.text.tools import ToolSpec


def _ordered_tool(name: str, log: list[str], delay: float = 0.0) -> ToolSpec:
    async def _call(**kwargs: object) -> dict[str, object]:
        log.append(f"start:{name}")
        if delay:
            await asyncio.sleep(delay)
        log.append(f"end:{name}")
        return {"tool": name}

    return ToolSpec(name=name, description=name, parameters={"type": "object", "properties": {}}, callable=_call)


class ExternalExecutionTests(unittest.IsolatedAsyncioTestCase):
    async def test_tool_calls_are_surfaced_and_resumed(self) -> None:
        frontend = FakeChatClient([delegate_response("self-contained request")])
        backend = FakeChatClient(
            [tool_response(("lookup", {"value": "x"}), ids=["ext-1"]), text_response("final answer")]
        )
        agent, _ = make_agent(frontend=frontend, backend=backend, tools_config={"execution": "external"})

        turn, session = await agent.send("do it", agent.new_session())
        assert turn.is_tool_call and turn.final_text is None
        assert [call.name for call in turn.tool_calls] == ["lookup"]
        assert session.pending is not None and session.pending.outstanding == ("ext-1",)
        assert session.pending.delegation_query == "self-contained request"

        turn, session = await agent.send_tool_results([ToolResult("ext-1", '{"ok": true}')], session)
        assert turn.final_text == "final answer"
        assert session.pending is None
        assert [m.role for m in session.frontend_history.messages] == ["user", "assistant", "tool", "assistant"]
        assert session.frontend_history.messages[-1].content == "final answer"

    async def test_parallel(self) -> None:
        backend = FakeChatClient(
            [
                tool_response(("lookup", {"n": 1}), ("lookup", {"n": 2}), ("lookup", {"n": 3}), ids=["a", "b", "c"]),
                text_response("all three done"),
            ]
        )
        agent, _ = make_agent(backend=backend, mode="backend_only", tools_config={"execution": "external"})
        turn, session = await agent.send("look them up", agent.new_session())
        assert [call.id for call in turn.tool_calls] == ["a", "b", "c"]

        out_of_order = [ToolResult("c", "C"), ToolResult("a", "A"), ToolResult("b", "B")]
        turn, session = await agent.send_tool_results(out_of_order, session)
        assert turn.final_text == "all three done"
        tool_messages = [m for m in backend.last_messages if m.role == "tool"]
        assert [m.tool_call_id for m in tool_messages] == ["a", "b", "c"]

    async def test_multi_round_external_turn(self) -> None:
        backend = FakeChatClient(
            [
                tool_response(("lookup", {}), ids=["r1"]),
                tool_response(("lookup", {}), ids=["r2"]),
                text_response("done at last"),
            ]
        )
        agent, _ = make_agent(backend=backend, mode="backend_only", tools_config={"execution": "external"})
        turn, session = await agent.send("go", agent.new_session())
        turn, session = await agent.send_tool_results([ToolResult("r1", "{}")], session)
        assert turn.is_tool_call and session.pending is not None and session.pending.iterations == 2
        turn, session = await agent.send_tool_results([ToolResult("r2", "{}")], session)
        assert turn.final_text == "done at last"


class InternalExecutionTests(unittest.IsolatedAsyncioTestCase):
    async def test_internal_batch_is_sequential(self) -> None:
        log: list[str] = []
        backend = FakeChatClient([tool_response(("slow", {}), ("fast", {}), ids=["s", "f"]), text_response("done")])
        agent, _ = make_agent(
            backend=backend,
            mode="backend_only",
            tools=[_ordered_tool("slow", log, delay=0.01), _ordered_tool("fast", log)],
        )
        turn, _ = await agent.send("go", agent.new_session())
        assert turn.final_text == "done"
        assert log == ["start:slow", "end:slow", "start:fast", "end:fast"]

    async def test_parallel_execution_opt_in_overlaps(self) -> None:
        log: list[str] = []
        backend = FakeChatClient([tool_response(("slow", {}), ("fast", {}), ids=["s", "f"]), text_response("done")])
        agent, _ = make_agent(
            backend=backend,
            mode="backend_only",
            tools=[_ordered_tool("slow", log, delay=0.02), _ordered_tool("fast", log)],
            tools_config={"parallel_execution": True},
        )
        await agent.send("go", agent.new_session())
        assert log[:2] == ["start:slow", "start:fast"]
        tool_messages = [m for m in backend.last_messages if m.role == "tool"]
        assert [m.tool_call_id for m in tool_messages] == ["s", "f"]

    async def test_internal_mode_never_surfaces_tool_calls(self) -> None:
        backend = FakeChatClient([tool_response(("lookup", {"value": "x"})), text_response("resolved")])
        agent, _ = make_agent(backend=backend, mode="backend_only", tools=[echo_tool()])
        turn, session = await agent.send("go", agent.new_session())
        assert turn.final_text == "resolved"
        assert not turn.is_tool_call and session.pending is None

    async def test_failing_tool_becomes_an_error_result(self) -> None:
        def _boom(**kwargs: object) -> None:
            raise RuntimeError("tool exploded")

        failing = ToolSpec(name="lookup", description="x", parameters={"type": "object"}, callable=_boom)
        backend = FakeChatClient([tool_response(("lookup", {})), text_response("handled")])
        agent, _ = make_agent(backend=backend, mode="backend_only", tools=[failing])
        turn, _ = await agent.send("go", agent.new_session())
        assert turn.final_text == "handled"
        tool_messages = [m for m in backend.last_messages if m.role == "tool"]
        assert "tool exploded" in (tool_messages[0].content or "")
