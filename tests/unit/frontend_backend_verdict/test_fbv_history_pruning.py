# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D100, D101, D102, D103, D107

"""Rule: pruning drops whole logical groups and never splits a tool-call batch."""

from __future__ import annotations

from examples.frontend_backend_verdict.text.history import History, group_is_complete, validate_tool_pairing
from examples.frontend_backend_verdict.text.messages import Message, ToolCall


def _single_call_group(index: int) -> list[Message]:
    call = ToolCall.create("lookup", {"i": index}, id=f"c{index}")
    return [
        Message.user(f"user {index}"),
        Message.assistant_tool_calls((call,)),
        Message.tool(call.id, "{}"),
        Message.assistant(f"answer {index}"),
    ]


def _parallel_group(index: int) -> list[Message]:
    calls = tuple(ToolCall.create("lookup", {"n": n}, id=f"p{index}-{n}") for n in range(3))
    return [
        Message.user(f"user {index}"),
        Message.assistant_tool_calls(calls),
        *[Message.tool(call.id, "{}") for call in calls],
        Message.assistant(f"answer {index}"),
    ]


def _multi_round_group(index: int) -> list[Message]:
    first = ToolCall.create("lookup", {}, id=f"m{index}-a")
    second = ToolCall.create("lookup", {}, id=f"m{index}-b")
    return [
        Message.user(f"user {index}"),
        Message.assistant_tool_calls((first,)),
        Message.tool(first.id, "{}"),
        Message.assistant_tool_calls((second,)),
        Message.tool(second.id, "{}"),
        Message.assistant(f"answer {index}"),
    ]


def test_groups_split_on_user_messages() -> None:
    history = History(tuple(_single_call_group(0) + _single_call_group(1)))
    assert len(history.groups()) == 2


def test_pruning_keeps_whole_groups_single_call() -> None:
    history = History(tuple(sum((_single_call_group(i) for i in range(4)), [])))
    pruned = history.prune(2)
    assert len(pruned.groups()) == 2
    validate_tool_pairing(pruned.messages)
    assert all(group_is_complete(group) for group in pruned.groups())


def test_pruning_keeps_whole_groups_parallel_calls() -> None:
    history = History(tuple(sum((_parallel_group(i) for i in range(3)), [])))
    pruned = history.prune(1)
    assert len(pruned.groups()) == 1
    validate_tool_pairing(pruned.messages)
    assert pruned.messages[0].content == "user 2"


def test_pruning_keeps_whole_groups_multi_round() -> None:
    history = History(tuple(sum((_multi_round_group(i) for i in range(3)), [])))
    pruned = history.prune(2)
    validate_tool_pairing(pruned.messages)
    assert all(group_is_complete(group) for group in pruned.groups())


def test_oversized_single_group_survives() -> None:
    history = History(tuple(_parallel_group(0)))
    assert history.prune(1).messages == history.messages


def test_in_flight_group_is_never_pruned() -> None:
    call = ToolCall.create("lookup", {}, id="live")
    history = History(tuple(_single_call_group(0) + [Message.user("live"), Message.assistant_tool_calls((call,))]))
    pruned = history.prune(1)
    assert pruned.messages[-1].tool_calls[0].id == "live"


def test_validate_tool_pairing_detects_orphans() -> None:
    call = ToolCall.create("lookup", {}, id="x")
    try:
        validate_tool_pairing([Message.assistant_tool_calls((call,))])
    except ValueError as exc:
        assert "without a matching result" in str(exc)
    else:
        raise AssertionError("expected orphan tool call to be rejected")
