# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D100, D101, D102, D103, D107

"""Frontend rules: tool surface, fail-closed contract, and history closing."""

from __future__ import annotations

import unittest

from _fbv_fakes import FakeChatClient, delegate_response, make_agent, text_response, tool_response

from examples.frontend_backend_verdict.text import events
from examples.frontend_backend_verdict.text.errors import FrontendContractError
from examples.frontend_backend_verdict.text.history import group_is_complete


class FrontendToolSurfaceTests(unittest.IsolatedAsyncioTestCase):
    async def test_tool_surface(self) -> None:
        frontend = FakeChatClient([text_response("hello")])
        agent, _ = make_agent(frontend=frontend)
        await agent.send("hi", agent.new_session())
        names = [tool["function"]["name"] for tool in frontend.last_tools]
        assert names == ["call_backend"]

    async def test_unknown_tool_fails_closed(self) -> None:
        frontend = FakeChatClient(
            [tool_response(("get_order", {"order_id": "1"})), tool_response(("get_order", {"order_id": "1"}))]
        )
        agent, sink = make_agent(frontend=frontend, delegation={"fallback_text": "I could not process that."})
        turn, state = await agent.send("status?", agent.new_session())
        assert turn.final_text == "I could not process that."
        assert sink.of_kind(events.FRONTEND_CONTRACT_VIOLATION)
        assert "get_order" not in (turn.final_text or "")
        assert state.frontend_history.messages[-1].content == "I could not process that."

    async def test_repair_attempt_can_recover(self) -> None:
        frontend = FakeChatClient([tool_response(("get_order", {})), delegate_response("the real request")])
        backend = FakeChatClient([text_response("done")])
        agent, sink = make_agent(frontend=frontend, backend=backend)
        turn, _ = await agent.send("status?", agent.new_session())
        assert turn.final_text == "done"
        assert sink.of_kind(events.FRONTEND_REPAIR)

    async def test_contract_violation_can_raise(self) -> None:
        frontend = FakeChatClient([tool_response(("nope", {})), tool_response(("nope", {}))])
        agent, _ = make_agent(frontend=frontend, delegation={"on_contract_violation": "error"})
        with self.assertRaises(FrontendContractError):
            await agent.send("status?", agent.new_session())

    async def test_content_plus_tool_call(self) -> None:
        frontend = FakeChatClient([delegate_response("do the thing", content="Let me check that for you.")])
        backend = FakeChatClient([text_response("the answer")])
        agent, sink = make_agent(frontend=frontend, backend=backend)
        turn, _ = await agent.send("do it", agent.new_session())
        assert turn.final_text == "the answer"
        discarded = [e for e in sink.of_kind(events.FRONTEND_CONTRACT_VIOLATION) if "discarded_text" in e.data]
        assert discarded and discarded[0].data["discarded_text"] == "Let me check that for you."

    async def test_tool_call_typed_as_text_is_repaired_into_a_real_delegation(self) -> None:
        typed = (
            '{\n  "content": "",\n  "tool_calls": [{\n    "function": "call_backend",\n    "arguments": '
            '{"query": "The user asks about order 5513.", "filler_text": "Let me check."}\n  }]\n}'
        )
        frontend = FakeChatClient([text_response(typed), delegate_response("The user asks about order 5513.")])
        backend = FakeChatClient([text_response("Order 5513 has shipped.")])
        agent, sink = make_agent(frontend=frontend, backend=backend)
        turn, state = await agent.send("What is order 5513?", agent.new_session())
        assert turn.final_text == "Order 5513 has shipped."
        repairs = sink.of_kind(events.FRONTEND_REPAIR)
        assert repairs and repairs[0].data["rejected_text"] == typed
        assert all(typed != message.content for message in state.frontend_history.messages)

    async def test_call_arguments_typed_as_text_are_repaired(self) -> None:
        typed = '{\n  "query": "The user is asking about order 5513.",\n  "filler_text": "Let me look that up."\n}'
        frontend = FakeChatClient([text_response(typed), delegate_response("The user is asking about order 5513.")])
        backend = FakeChatClient([text_response("Order 5513 has shipped.")])
        agent, sink = make_agent(frontend=frontend, backend=backend)
        turn, _ = await agent.send("What is order 5513?", agent.new_session())
        assert turn.final_text == "Order 5513 has shipped."
        assert sink.of_kind(events.FRONTEND_REPAIR)

    async def test_tool_call_markup_as_text_fails_closed(self) -> None:
        markup = "<tool_call><function=call_backend><parameter=query>order 5513</parameter></function></tool_call>"
        frontend = FakeChatClient([text_response(markup), text_response(markup)])
        agent, _ = make_agent(frontend=frontend)
        turn, _ = await agent.send("What is order 5513?", agent.new_session())
        assert turn.final_text != markup
        assert "call_backend" not in turn.final_text

    async def test_empty_query_is_a_violation(self) -> None:
        frontend = FakeChatClient([tool_response(("call_backend", {"query": "  "})), text_response("plain answer")])
        agent, sink = make_agent(frontend=frontend)
        turn, _ = await agent.send("do it", agent.new_session())
        assert turn.final_text == "plain answer"
        assert sink.of_kind(events.FRONTEND_REPAIR)


class FrontendHistoryTests(unittest.IsolatedAsyncioTestCase):
    async def test_delegated_group_is_complete(self) -> None:
        frontend = FakeChatClient([delegate_response("self-contained request", filler="one moment")])
        backend = FakeChatClient([text_response("the backend answer")])
        agent, _ = make_agent(frontend=frontend, backend=backend)
        turn, state = await agent.send("do it", agent.new_session())

        messages = state.frontend_history.messages
        assert [m.role for m in messages] == ["user", "assistant", "tool", "assistant"]
        assert messages[1].tool_calls[0].name == "call_backend"
        assert messages[2].tool_call_id == messages[1].tool_calls[0].id
        assert messages[2].content == "the backend answer"
        assert messages[3].content == turn.final_text == "the backend answer"
        assert group_is_complete(messages)

    async def test_backend_answer_in_frontend_history_reaches_next_turn(self) -> None:
        frontend = FakeChatClient([delegate_response("first request"), text_response("follow up")])
        backend = FakeChatClient([text_response("order 5512 shipped")])
        agent, _ = make_agent(frontend=frontend, backend=backend)
        session = agent.new_session()
        _, session = await agent.send("where is it?", session)
        await agent.send("and when?", session)
        replayed = [m.content for m in frontend.last_messages]
        assert "order 5512 shipped" in replayed

    async def test_every_path_appends_one_group(self) -> None:
        cases = {
            "direct": (FakeChatClient([text_response("direct answer")]), FakeChatClient(), "direct answer"),
            "delegated": (
                FakeChatClient([delegate_response("q")]),
                FakeChatClient([text_response("delegated answer")]),
                "delegated answer",
            ),
            "fallback": (
                FakeChatClient([tool_response(("bad", {})), tool_response(("bad", {}))]),
                FakeChatClient(),
                "Sorry, I could not process that. Could you rephrase?",
            ),
        }
        for name, (frontend, backend, expected) in cases.items():
            agent, _ = make_agent(frontend=frontend, backend=backend)
            turn, state = await agent.send("go", agent.new_session())
            groups = state.frontend_history.groups()
            assert len(groups) == 1, name
            assert group_is_complete(groups[0]), name
            assert groups[0][-1].content == turn.final_text == expected, name
