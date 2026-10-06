# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D100, D101, D102, D103, D107

"""Rule: from_message_history() replays a foreign transcript, mode-specifically."""

from __future__ import annotations

import pytest

from examples.frontend_backend_verdict.text.errors import StateReplayError
from examples.frontend_backend_verdict.text.messages import Message, ToolCall
from examples.frontend_backend_verdict.text.session import SessionState

CALL = ToolCall.create("lookup", {"a": 1}, id="c1")
TRANSCRIPT = [
    Message.user("hello"),
    Message.assistant("hi there"),
    Message.user("look it up"),
    Message.assistant_tool_calls((CALL,)),
    Message.tool("c1", '{"ok": true}'),
    Message.assistant("here it is"),
]


def test_paired_replay_keeps_only_visible_turns() -> None:
    state = SessionState.from_message_history(TRANSCRIPT, backend_only=False)
    assert [m.role for m in state.frontend_history.messages] == ["user", "assistant", "user", "assistant"]
    assert [m.content for m in state.frontend_history.messages][-1] == "here it is"
    assert len(state.backend_history) == 0
    assert state.pending is None


def test_backend_only_replay_keeps_calls_with_their_results() -> None:
    state = SessionState.from_message_history(TRANSCRIPT, backend_only=True)
    assert len(state.frontend_history) == 0
    assert len(state.backend_history) == len(TRANSCRIPT)
    ids = [call.id for m in state.backend_history.messages for call in m.tool_calls]
    answered = [m.tool_call_id for m in state.backend_history.messages if m.role == "tool"]
    assert ids == answered == ["c1"]


def test_unresolved_tail_is_rejected() -> None:
    truncated = TRANSCRIPT[:4]
    for backend_only in (True, False):
        with pytest.raises(StateReplayError, match="unresolved tool calls"):
            SessionState.from_message_history(truncated, backend_only=backend_only)
