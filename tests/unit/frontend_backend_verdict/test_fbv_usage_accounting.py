# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D100, D101, D102, D103, D107

"""Rules: usage aggregates per step and session; unknown cost stays unknown."""

from __future__ import annotations

import unittest

from _fbv_fakes import FakeChatClient, delegate_response, echo_tool, make_agent, text_response, tool_response

from examples.frontend_backend_verdict.text.messages import Usage, UsageTotals


def test_unknown_cost_is_contagious() -> None:
    totals = UsageTotals()
    totals = totals.add_call("backend", usage=Usage(total_tokens=5), latency_ms=1.0, cost=0.25)
    assert totals.cost == 0.25
    totals = totals.add_call("backend", usage=Usage(total_tokens=5), latency_ms=1.0, cost=None)
    assert totals.cost is None
    assert totals.cost_unknown_calls == 1
    assert totals.cost_known_subtotal == 0.25


def test_empty_totals_cost_zero() -> None:
    assert UsageTotals().cost == 0.0


class AccountingTests(unittest.IsolatedAsyncioTestCase):
    async def test_step_totals_sum_every_internal_call(self) -> None:
        frontend = FakeChatClient([delegate_response("q", cost=0.01)])
        backend = FakeChatClient(
            [
                tool_response(("lookup", {}), ids=["t1"], cost=0.02),
                tool_response(("lookup", {}), ids=["t2"], cost=0.03),
                text_response("done", cost=0.04),
            ]
        )
        agent, _ = make_agent(frontend=frontend, backend=backend, tools=[echo_tool()])
        turn, session = await agent.send("go", agent.new_session())

        assert turn.usage.calls == 4
        assert turn.usage.frontend.calls == 1
        assert turn.usage.backend.calls == 3
        assert round(turn.usage.cost or 0.0, 6) == 0.10
        assert turn.usage.usage.total_tokens == 80
        assert session.usage.calls == 4

    async def test_session_usage_accumulates(self) -> None:
        frontend = FakeChatClient([text_response("a", cost=0.01), text_response("b", cost=0.01)])
        agent, _ = make_agent(frontend=frontend)
        session = agent.new_session()
        first, session = await agent.send("one", session)
        second, session = await agent.send("two", session)
        assert first.usage.calls == second.usage.calls == 1
        assert session.usage.calls == 2
        assert round(session.usage.cost or 0.0, 6) == 0.02

    async def test_repair_attempts_are_counted(self) -> None:
        frontend = FakeChatClient([tool_response(("nope", {}), cost=0.01), text_response("recovered", cost=0.01)])
        agent, _ = make_agent(frontend=frontend)
        turn, _ = await agent.send("go", agent.new_session())
        assert turn.final_text == "recovered"
        assert turn.usage.frontend.calls == 2

    async def test_unknown_provider_cost_propagates_to_the_turn(self) -> None:
        frontend = FakeChatClient([text_response("hi", cost=None)])
        agent, _ = make_agent(frontend=frontend)
        turn, session = await agent.send("go", agent.new_session())
        assert turn.usage.cost is None
        assert session.usage.cost is None
