# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D100, D101, D102, D103, D107

"""Rule: nothing reachable from SessionState holds a mutable container."""

from __future__ import annotations

import dataclasses

from examples.frontend_backend_verdict.text.history import History
from examples.frontend_backend_verdict.text.messages import Message, ToolCall
from examples.frontend_backend_verdict.text.session import PendingTurn, SessionState

MUTABLE = (list, dict, set, bytearray)


def _walk(value: object, path: str, seen: set[int], offenders: list[str]) -> None:
    if id(value) in seen:
        return
    seen.add(id(value))
    if isinstance(value, MUTABLE):
        offenders.append(f"{path}: {type(value).__name__}")
        return
    if isinstance(value, tuple):
        for index, item in enumerate(value):
            _walk(item, f"{path}[{index}]", seen, offenders)
        return
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        assert type(value).__dataclass_params__.frozen, f"{path} is not frozen"
        for field in dataclasses.fields(value):
            _walk(getattr(value, field.name), f"{path}.{field.name}", seen, offenders)


def test_session_state_tree_has_no_mutable_containers() -> None:
    call = ToolCall.create("lookup", {"a": 1})
    state = SessionState(
        frontend_history=History((Message.user("hi"), Message.assistant_tool_calls((call,)))),
        backend_history=History((Message.tool(call.id, "{}"),)),
        pending=PendingTurn(
            delegation_query="q",
            user_message="hi",
            backend_history=History((Message.user("q"),)),
            outstanding=(call.id,),
            frontend_assistant=Message.assistant_tool_calls((call,)),
        ),
    )
    offenders: list[str] = []
    _walk(state, "SessionState", set(), offenders)
    assert offenders == []


def test_history_append_does_not_mutate_the_original() -> None:
    original = History((Message.user("one"),))
    extended = original.append(Message.assistant("two"))
    assert len(original) == 1
    assert len(extended) == 2
