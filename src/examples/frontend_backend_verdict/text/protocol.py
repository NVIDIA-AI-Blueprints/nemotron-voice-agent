# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tool-call / tool-result correlation and its enforced validation rules.

The backend sequence is::

    delegated query (or user message in backend_only)
      -> assistant message with 1..N tool calls   [NeedsTools]
      -> exactly N correlated tool results        [ToolResults]
      -> assistant final text [Final] OR another batch [NeedsTools]

Provider-generated tool-call ids are preserved; rewriting them would silently
break correlation for any caller that tracks its own calls (tau2 does).
"""

from __future__ import annotations

from collections.abc import Sequence

from examples.frontend_backend_verdict.text.errors import ToolProtocolError
from examples.frontend_backend_verdict.text.messages import ToolCall, ToolResult, new_tool_call_id
from examples.frontend_backend_verdict.text.tools import error_payload

MISSING_RESULT_MESSAGE = "tool result was not provided by the caller"


def ensure_ids(calls: Sequence[ToolCall]) -> tuple[ToolCall, ...]:
    """Return ``calls`` with ids intact, generating one only where absent.

    Raises when a provider emits duplicate ids inside one batch: results could
    not be correlated, so the batch is rejected rather than silently renamed.
    """
    resolved: list[ToolCall] = []
    seen: set[str] = set()
    for call in calls:
        call_id = (call.id or "").strip() or new_tool_call_id()
        if call_id in seen:
            raise ToolProtocolError(f"provider emitted duplicate tool-call id in one batch: {call_id}")
        seen.add(call_id)
        if call.id == call_id:
            resolved.append(call)
        else:
            resolved.append(ToolCall(id=call_id, name=call.name, arguments_json=call.arguments_json))
    return tuple(resolved)


def validate_tool_results(
    outstanding: Sequence[str],
    results: Sequence[ToolResult],
    *,
    on_incomplete: str = "error",
) -> tuple[ToolResult, ...]:
    """Validate and order ``results`` against the outstanding tool-call ids.

    Returns results ordered to match ``outstanding`` so the model always sees a
    batch in emission order regardless of how the caller returned them.
    """
    if not outstanding:
        raise ToolProtocolError("no tool calls are outstanding")
    expected = list(outstanding)
    by_id: dict[str, ToolResult] = {}
    for result in results:
        if result.tool_call_id not in expected:
            raise ToolProtocolError(
                f"unexpected tool_call_id {result.tool_call_id!r}; expected one of {sorted(expected)}"
            )
        if result.tool_call_id in by_id:
            raise ToolProtocolError(f"duplicate tool result for tool_call_id {result.tool_call_id!r}")
        by_id[result.tool_call_id] = result
    missing = [call_id for call_id in expected if call_id not in by_id]
    if missing:
        if on_incomplete == "error":
            raise ToolProtocolError(f"missing tool results for {sorted(missing)}")
        for call_id in missing:
            by_id[call_id] = ToolResult(
                tool_call_id=call_id,
                content=error_payload(RuntimeError(MISSING_RESULT_MESSAGE)),
                is_error=True,
            )
    return tuple(by_id[call_id] for call_id in expected)
