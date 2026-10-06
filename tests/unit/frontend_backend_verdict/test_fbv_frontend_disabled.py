# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D100, D101, D102, D103, D107

"""Rule: backend_only runs standalone and keeps its own conversation history."""

from __future__ import annotations

import unittest

from _fbv_fakes import FakeChatClient, echo_tool, make_agent, text_response, tool_response

from examples.frontend_backend_verdict.text.history import group_is_complete


class BackendOnlyTests(unittest.IsolatedAsyncioTestCase):
    async def test_history_persists_across_turns(self) -> None:
        backend = FakeChatClient([text_response("first"), text_response("second")])
        agent, _ = make_agent(backend=backend, mode="backend_only")
        session = agent.new_session()
        turn, session = await agent.send("one", session)
        assert turn.final_text == "first"
        turn, session = await agent.send("two", session)
        assert turn.final_text == "second"
        assert [m.content for m in session.backend_history.messages] == ["one", "first", "two", "second"]
        assert len(session.frontend_history) == 0

    async def test_no_frontend_client_is_required(self) -> None:
        backend = FakeChatClient([text_response("standalone")])
        agent, _ = make_agent(frontend=None, backend=backend, mode="backend_only")
        turn, _ = await agent.send("hello", agent.new_session())
        assert turn.final_text == "standalone"

    async def test_tool_turn_forms_a_complete_group(self) -> None:
        backend = FakeChatClient([tool_response(("lookup", {"value": "x"})), text_response("done")])
        agent, _ = make_agent(backend=backend, mode="backend_only", tools=[echo_tool()])
        _, session = await agent.send("look it up", agent.new_session())
        groups = session.backend_history.groups()
        assert len(groups) == 1
        assert group_is_complete(groups[0])

    async def test_history_is_pruned_by_groups(self) -> None:
        backend = FakeChatClient([text_response(f"reply {i}") for i in range(4)])
        agent, _ = make_agent(backend=backend, mode="backend_only", backend_history={"max_groups": 2})
        session = agent.new_session()
        for index in range(4):
            _, session = await agent.send(f"turn {index}", session)
        assert len(session.backend_history.groups()) == 2
        assert session.backend_history.messages[0].content == "turn 2"
