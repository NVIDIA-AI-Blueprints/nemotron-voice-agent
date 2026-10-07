# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D100, D101, D102, D103, D107

"""Rule: filler text is observable only on the event sink, never on a turn."""

from __future__ import annotations

import dataclasses
import unittest

from _fbv_fakes import FakeChatClient, delegate_response, make_agent, text_response

from examples.frontend_backend_verdict.text import events
from examples.frontend_backend_verdict.text.messages import AgentTurn

FILLER = "Give me one moment."


class FillerIsolationTests(unittest.IsolatedAsyncioTestCase):
    async def test_filler_is_logged_but_never_returned(self) -> None:
        frontend = FakeChatClient([delegate_response("a self-contained request", filler=FILLER)])
        backend = FakeChatClient([text_response("the only answer the user sees")])
        agent, sink = make_agent(frontend=frontend, backend=backend)

        turn, state = await agent.send("do the thing", agent.new_session())

        assert turn.final_text == "the only answer the user sees"
        assert FILLER not in (turn.final_text or "")
        assert all(FILLER not in str(getattr(turn, field.name)) for field in dataclasses.fields(AgentTurn))
        assert [event.data["text"] for event in sink.of_kind(events.FILLER)] == [FILLER]

    async def test_agent_turn_has_no_events_field(self) -> None:
        assert "events" not in {field.name for field in dataclasses.fields(AgentTurn)}

    async def test_filler_does_not_enter_a_users_view_of_history(self) -> None:
        frontend = FakeChatClient([delegate_response("q", filler=FILLER)])
        backend = FakeChatClient([text_response("answer")])
        agent, _ = make_agent(frontend=frontend, backend=backend)
        _, state = await agent.send("go", agent.new_session())
        visible = [m.content for m in state.frontend_history.messages if m.role in ("user", "assistant") and m.content]
        assert FILLER not in " ".join(visible)
