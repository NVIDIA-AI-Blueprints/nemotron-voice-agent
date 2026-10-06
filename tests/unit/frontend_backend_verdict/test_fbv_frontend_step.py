# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D100, D101, D102, D103, D107

"""``send()`` split into ``decide_turn()`` + ``continue_turn()``, and the optional in-progress note."""

from __future__ import annotations

import unittest

from _fbv_fakes import FakeChatClient, delegate_response, make_agent, text_response, tool_response

from examples.frontend_backend_verdict.text import events
from examples.frontend_backend_verdict.text.delegation import (
    CALL_BACKEND_TOOL,
    FRONTEND_TOOLS,
    FRONTEND_TOOLS_IN_PROGRESS,
)
from examples.frontend_backend_verdict.text.frontend import Delegate, DirectAnswer

IN_PROGRESS = {"query": "check order 7", "filler": "One moment.", "filler_heard": True, "new_words": "okay"}


def _requests(client: FakeChatClient) -> list[tuple[list[tuple[str, str | None]], list]]:
    return [([(m.role, m.content) for m in call["messages"]], call["tools"]) for call in client.calls]


class SplitMatchesSendTests(unittest.IsolatedAsyncioTestCase):
    async def _both(self, frontend_responses, backend_responses, **config):
        results = []
        for split in (False, True):
            frontend = FakeChatClient(list(frontend_responses))
            backend = FakeChatClient(list(backend_responses))
            agent, sink = make_agent(frontend=frontend, backend=backend, **config)
            session = agent.new_session()
            if split:
                turn, state = await agent.continue_turn(await agent.decide_turn("hello", session))
            else:
                turn, state = await agent.send("hello", session)
            results.append((turn, state, [e.kind for e in sink.events], _requests(frontend), _requests(backend)))
        return results

    async def test_direct_answer_is_identical(self) -> None:
        sent, split = await self._both([text_response("Hi there.")], [])
        self.assertEqual(sent[0].final_text, split[0].final_text)
        self.assertEqual(sent[1].frontend_history, split[1].frontend_history)
        self.assertEqual(sent[2:], split[2:])

    async def test_delegation_is_identical(self) -> None:
        sent, split = await self._both(
            [delegate_response("check order 7", filler="One moment.")], [text_response("Shipped.")]
        )
        self.assertEqual(split[0].final_text, "Shipped.")
        self.assertEqual(sent[1].frontend_history, split[1].frontend_history)
        self.assertEqual(sent[2:], split[2:])  # same events, same order, same LLM requests

    async def test_external_tool_suspension_is_identical(self) -> None:
        sent, split = await self._both(
            [delegate_response("look it up")],
            [tool_response(("lookup", {"id": "1"}), ids=["call_x"])],
            tools_config={"execution": "external"},
        )
        self.assertEqual([c.name for c in split[0].tool_calls], ["lookup"])
        self.assertEqual(sent[1].pending, split[1].pending)
        self.assertEqual(sent[2:], split[2:])


class DecideTurnTests(unittest.IsolatedAsyncioTestCase):
    async def test_decide_turn_changes_nothing_and_emits_no_delegation(self) -> None:
        frontend = FakeChatClient([delegate_response("check order 7", filler="One moment.")])
        backend = FakeChatClient()
        agent, sink = make_agent(frontend=frontend, backend=backend)
        session = agent.new_session()
        step = await agent.decide_turn("check order 7", session)
        self.assertIs(step.session, session)
        self.assertIsInstance(step.decision, Delegate)
        self.assertEqual(backend.calls, [])
        kinds = {e.kind for e in sink.events}
        for kind in (events.DELEGATION, events.FILLER, events.BACKEND_CONTEXT, events.DIRECT_ANSWER):
            self.assertNotIn(kind, kinds)

    async def test_without_a_note_the_request_is_unchanged(self) -> None:
        via_send, via_decide = FakeChatClient([text_response("a")]), FakeChatClient([text_response("a")])
        agent, _ = make_agent(frontend=via_send)
        await agent.send("hi", agent.new_session())
        agent, _ = make_agent(frontend=via_decide)
        await agent.decide_turn("hi", agent.new_session(), in_progress=None)
        self.assertEqual(_requests(via_send), _requests(via_decide))
        self.assertEqual(via_decide.last_tools, list(FRONTEND_TOOLS))
        self.assertNotIn("task", CALL_BACKEND_TOOL["function"]["parameters"]["properties"])


class InProgressNoteTests(unittest.IsolatedAsyncioTestCase):
    async def test_in_progress_selects_the_task_tool(self) -> None:
        frontend = FakeChatClient([delegate_response("check order 7", task="continue")])
        agent, _ = make_agent(frontend=frontend)
        step = await agent.decide_turn("okay", agent.new_session(), in_progress=IN_PROGRESS)
        self.assertEqual(frontend.last_tools, list(FRONTEND_TOOLS_IN_PROGRESS))
        parameters = frontend.last_tools[0]["function"]["parameters"]
        self.assertIn("task", parameters["required"])
        self.assertEqual(parameters["properties"]["task"]["enum"], ["continue", "new"])
        self.assertEqual(step.decision.task, "continue")

    async def test_task_values(self) -> None:
        for given, expected in (("new", "new"), ("CONTINUE", "continue"), ("maybe", ""), (None, "")):
            with self.subTest(given=given):
                extra = {} if given is None else {"task": given}
                frontend = FakeChatClient([delegate_response("check order 7", **extra)])
                agent, sink = make_agent(frontend=frontend)
                step = await agent.decide_turn("okay", agent.new_session(), in_progress=IN_PROGRESS)
                self.assertEqual(step.decision.task, expected)
                self.assertEqual(len(frontend.calls), 1)  # an invalid task is no contract violation
                self.assertFalse(sink.of_kind(events.FRONTEND_REPAIR))

    async def test_task_is_ignored_without_a_note(self) -> None:
        frontend = FakeChatClient([delegate_response("check order 7", task="continue")])
        backend = FakeChatClient([text_response("done")])
        agent, _ = make_agent(frontend=frontend, backend=backend)
        step = await agent.decide_turn("check", agent.new_session())
        self.assertEqual(step.decision.task, "")

    async def test_direct_answer_with_a_note(self) -> None:
        frontend = FakeChatClient([text_response("It is still running.")])
        agent, _ = make_agent(frontend=frontend)
        step = await agent.decide_turn("how long?", agent.new_session(), in_progress=IN_PROGRESS)
        self.assertIsInstance(step.decision, DirectAnswer)
