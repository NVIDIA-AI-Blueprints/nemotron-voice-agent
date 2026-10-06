# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D100, D101, D102, D103, D107

"""Rule: an outward turn carries text XOR tool calls, never both or neither."""

from __future__ import annotations

import pytest

from examples.frontend_backend_verdict.text.messages import AgentTurn, ToolCall


def test_text_only_turn_is_valid() -> None:
    assert AgentTurn(final_text="hello").final_text == "hello"


def test_tool_call_turn_is_valid() -> None:
    turn = AgentTurn(tool_calls=(ToolCall.create("lookup", {"a": 1}),))
    assert turn.is_tool_call and turn.final_text is None


def test_both_is_rejected() -> None:
    with pytest.raises(ValueError, match="both"):
        AgentTurn(final_text="hi", tool_calls=(ToolCall.create("lookup"),))


def test_neither_is_rejected() -> None:
    with pytest.raises(ValueError, match="either"):
        AgentTurn()
