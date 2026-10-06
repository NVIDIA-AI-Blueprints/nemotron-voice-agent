# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D100, D101, D102, D103, D107

"""Backend rules: self-contained queries, statelessness, caps, and error paths."""

from __future__ import annotations

import dataclasses
import unittest
from unittest import mock

from _fbv_fakes import (
    FakeChatClient,
    delegate_response,
    echo_tool,
    make_agent,
    make_config,
    text_response,
    tool_response,
)

from examples.frontend_backend_verdict.text import events
from examples.frontend_backend_verdict.text.backend import ERROR_TEXT, ITERATION_CAP_TEXT


class FailingClient:
    def __init__(self) -> None:
        self.calls = 0

    async def complete(self, *, messages, tools=None):  # noqa: ANN001, ANN003
        self.calls += 1
        raise RuntimeError("endpoint unavailable")


class BackendTests(unittest.IsolatedAsyncioTestCase):
    async def test_query_is_self_contained(self) -> None:
        frontend = FakeChatClient([delegate_response("Move the Friday booking for four people to 8:00 PM.")])
        backend = FakeChatClient([text_response("moved")])
        agent, _ = make_agent(frontend=frontend, backend=backend)
        await agent.send("make it eight", agent.new_session())
        user_messages = [m.content for m in backend.last_messages if m.role == "user"]
        assert user_messages == ["Move the Friday booking for four people to 8:00 PM."]
        assert "make it eight" not in " ".join(user_messages)

    async def test_stateless_paired(self) -> None:
        frontend = FakeChatClient([delegate_response("first request"), delegate_response("second request")])
        backend = FakeChatClient([text_response("one"), text_response("two")])
        agent, _ = make_agent(frontend=frontend, backend=backend)
        session = agent.new_session()
        _, session = await agent.send("a", session)
        _, session = await agent.send("b", session)
        assert [m.content for m in backend.last_messages if m.role == "user"] == ["second request"]
        assert len(session.backend_history) == 0

    async def test_simulated_delay_sleeps_once_per_delegation(self) -> None:
        config = make_config()
        config = dataclasses.replace(config, backend=dataclasses.replace(config.backend, simulated_delay_s=2.0))
        frontend = FakeChatClient([delegate_response("look it up")])
        backend = FakeChatClient([tool_response(("lookup", {"value": "x"})), text_response("done")])
        agent, _ = make_agent(frontend=frontend, backend=backend, tools=[echo_tool()], config=config)
        with mock.patch("examples.frontend_backend_verdict.text.agent.asyncio.sleep") as sleep:
            turn, _ = await agent.send("find x", agent.new_session())
        assert turn.final_text == "done"
        sleep.assert_awaited_once_with(2.0)

    async def test_iteration_cap(self) -> None:
        backend = FakeChatClient([tool_response(("lookup", {"value": "x"})) for _ in range(5)])
        agent, sink = make_agent(
            backend=backend,
            mode="backend_only",
            tools=[echo_tool()],
            tools_config={"max_tool_iterations": 2},
        )
        turn, _ = await agent.send("loop please", agent.new_session())
        assert turn.final_text == ITERATION_CAP_TEXT
        assert sink.of_kind(events.ITERATION_CAP)
        assert len(backend.calls) == 3

    async def test_error_path(self) -> None:
        agent, sink = make_agent(backend=FailingClient(), mode="backend_only")
        turn, session = await agent.send("hello", agent.new_session())
        assert turn.final_text == ERROR_TEXT
        assert sink.of_kind(events.BACKEND_ERROR)
        assert session.pending is None

    async def test_empty_backend_text_falls_back(self) -> None:
        backend = FakeChatClient([text_response("   ")])
        agent, _ = make_agent(backend=backend, mode="backend_only")
        turn, _ = await agent.send("hello", agent.new_session())
        assert turn.final_text == ERROR_TEXT
