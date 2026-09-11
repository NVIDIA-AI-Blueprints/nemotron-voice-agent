# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Thinker plan parsing and multi-tool response combination."""

from __future__ import annotations

from typing import Any

from examples.frontend_backend_agent.src.protocol import response_hint, tool_result


def plan_tool_calls(plan: dict[str, Any]) -> list[dict[str, Any]]:
    """Normalize planner output into ordered tool-call dictionaries."""
    base_params = plan.get("params") if isinstance(plan.get("params"), dict) else {}
    raw_calls = plan.get("tool_calls")
    if raw_calls is None:
        raw_calls = plan.get("tools")
    if isinstance(raw_calls, list):
        tool_calls: list[dict[str, Any]] = []
        for raw_call in raw_calls:
            if isinstance(raw_call, str):
                tool_calls.append({"tool": raw_call, "params": dict(base_params)})
                continue
            if not isinstance(raw_call, dict):
                continue
            call = dict(raw_call)
            call_params = call.get("params") if isinstance(call.get("params"), dict) else {}
            call["params"] = {**base_params, **call_params}
            tool_calls.append(call)
        return tool_calls
    if plan.get("tool"):
        return [plan]
    return []


def combine_parallel_payloads(plan: dict[str, Any], payloads: list[dict[str, Any]]) -> dict[str, Any]:
    """Combine ordered parallel tool payloads into one Thinker response."""
    response_text = str(plan.get("response_text") or "").strip()
    if not response_text:
        response_text = " ".join(str(payload.get("response_text") or "").strip() for payload in payloads).strip()
    response_text = response_text or "I finished checking those items."
    contexts = [str(payload.get("context") or "") for payload in payloads if payload.get("context")]
    context = "multi_tool" if len(set(contexts)) != 1 else contexts[0]
    data = {"results": payloads}
    if any(payload.get("type") == "response_hint" for payload in payloads):
        return response_hint(
            reason="multi_tool_followup",
            action="review_results",
            response_text=response_text,
            context=context,
            data=data,
        )
    status = "success" if all(payload.get("status") == "success" for payload in payloads) else "partial"
    return tool_result(
        tool="multi_tool",
        status=status,
        data=data,
        response_text=response_text,
        context=context,
    )


def string_list(value: Any) -> list[str] | None:
    """Return a stripped string list when ``value`` is a non-empty list."""
    if not isinstance(value, list):
        return None
    result = [str(item).strip() for item in value if str(item).strip()]
    return result or None
