# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D100, D101, D102, D103, D107

"""Rules: SessionState round-trips, and sessions on one agent stay isolated."""

from __future__ import annotations

import asyncio
import unittest

from _fbv_fakes import FakeChatClient, delegate_response, make_agent, text_response

from examples.frontend_backend_verdict.text.history import History
from examples.frontend_backend_verdict.text.messages import Message, ToolCall
from examples.frontend_backend_verdict.text.session import PendingTurn, SessionState


def test_round_trip_preserves_state() -> None:
    call = ToolCall.create("lookup", {"a": 1}, id="c1")
    state = SessionState(
        session_id="abc",
        frontend_history=History((Message.user("hi"), Message.assistant("hello"))),
        backend_history=History((Message.assistant_tool_calls((call,)), Message.tool("c1", "{}"))),
        pending=PendingTurn(
            delegation_query="q",
            user_message="hi",
            backend_history=History((Message.user("q"),)),
            outstanding=("c1",),
            iterations=2,
            frontend_assistant=Message.assistant_tool_calls((call,)),
        ),
    )
    restored = SessionState.from_dict(state.to_dict())
    assert restored.session_id == "abc"
    assert restored.frontend_history == state.frontend_history
    assert restored.backend_history == state.backend_history
    assert restored.pending == state.pending


class SessionIsolationTests(unittest.IsolatedAsyncioTestCase):
    async def test_two_sessions_on_one_agent_do_not_share_history(self) -> None:
        frontend = FakeChatClient([delegate_response("alpha request"), delegate_response("beta request")])
        backend = FakeChatClient([text_response("alpha answer"), text_response("beta answer")])
        agent, _ = make_agent(frontend=frontend, backend=backend)

        first, second = agent.new_session(), agent.new_session()
        _, first = await agent.send("alpha", first)
        _, second = await agent.send("beta", second)

        assert [m.content for m in first.frontend_history.messages if m.role == "user"] == ["alpha"]
        assert [m.content for m in second.frontend_history.messages if m.role == "user"] == ["beta"]
        assert first.session_id != second.session_id

    async def test_concurrent_sessions_keep_their_own_state(self) -> None:
        frontend = FakeChatClient([text_response("one"), text_response("two")])
        agent, _ = make_agent(frontend=frontend)
        sessions = [agent.new_session(), agent.new_session()]
        results = await asyncio.gather(agent.send("a", sessions[0]), agent.send("b", sessions[1]))
        assert {turn.final_text for turn, _ in results} == {"one", "two"}
        assert all(len(state.frontend_history) == 2 for _, state in results)
