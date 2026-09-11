# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""LLM planner for the internal Thinker."""

from __future__ import annotations

import json
from datetime import date, timedelta
from typing import Any, Protocol

from loguru import logger
from pipecat.processors.aggregators.llm_context import LLMContext

from examples.frontend_backend_agent.src.runtime_context import runtime_today

_PLAN_CONTEXTS = ["flight_search", "booking", "pnr_status", "general"]
_PLAN_TOOLS = ["flight_search", "booking", "pnr_status", "response_hint"]
_PLAN_CALL_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "tool": {"type": "string", "enum": _PLAN_TOOLS},
        "params": {"type": "object", "additionalProperties": True},
        "reason": {"type": "string"},
        "action": {"type": "string"},
        "context": {"type": "string", "enum": _PLAN_CONTEXTS},
        "response_text": {"type": "string"},
        "params_needed": {"type": "array", "items": {"type": "string"}},
        "params_resolved": {"type": "object", "additionalProperties": True},
    },
    "required": ["tool"],
    "additionalProperties": False,
}
THINKER_PLAN_SCHEMA: dict[str, Any] = {
    "oneOf": [
        _PLAN_CALL_SCHEMA,
        {
            "type": "object",
            "properties": {
                "tool_calls": {
                    "type": "array",
                    "minItems": 1,
                    "items": _PLAN_CALL_SCHEMA,
                },
                "response_text": {"type": "string"},
            },
            "required": ["tool_calls"],
            "additionalProperties": False,
        },
    ]
}


class ThinkerPlanner(Protocol):
    """Planner interface for selecting internal Thinker tool calls."""

    async def plan(self, *, query: str, slots: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
        """Return a structured Thinker plan."""


class PlannerLLM(Protocol):
    """Provider operations used by the Thinker planner."""

    async def run_inference(self, context: LLMContext, *, max_tokens: int) -> str:
        """Return one unstructured completion."""

    async def run_structured_inference(
        self,
        context: LLMContext,
        *,
        schema: dict[str, Any],
        schema_name: str,
        max_tokens: int,
    ) -> Any:
        """Return one schema-constrained completion."""


class NvidiaThinkerPlanner:
    """Thinker planner backed by Nemotron reasoning."""

    def __init__(
        self,
        *,
        llm: PlannerLLM,
        system_prompt: str,
        max_tokens: int = 4096,
        structured_output: bool = False,
    ) -> None:
        """Create an NVIDIA-backed Thinker planner."""
        if not system_prompt.strip():
            raise ValueError("Thinker planner requires a non-empty system prompt")
        self._llm = llm
        self._system_prompt = system_prompt
        self._max_tokens = max_tokens
        self._structured_output = structured_output

    async def plan(self, *, query: str, slots: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
        """Ask the Thinker LLM for internal tool plan JSON."""
        today = runtime_today()
        tomorrow = today + timedelta(days=1)
        user_payload = {
            "query": query,
            "structured_fields": slots,
            "session_state": state,
            "runtime_context": {
                "today": today.isoformat(),
                "tomorrow": tomorrow.isoformat(),
            },
        }
        context = LLMContext(
            [
                {"role": "system", "content": f"{self._system_prompt}{_runtime_date_context(today)}"},
                {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
            ]
        )
        if self._structured_output:
            plan = await self._llm.run_structured_inference(
                context,
                schema=THINKER_PLAN_SCHEMA,
                schema_name="airline_plan",
                max_tokens=self._max_tokens,
            )
            if not isinstance(plan, dict):
                raise ValueError("Thinker LLM plan must be a JSON object")
            return plan

        raw = await self._llm.run_inference(context, max_tokens=self._max_tokens)
        if not raw:
            raise RuntimeError("Thinker LLM returned an empty plan")
        return parse_plan_json(raw)


def parse_plan_json(raw: str) -> dict[str, Any]:
    """Parse a Thinker LLM JSON object response."""
    cleaned = _strip_thinking(raw).strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`").strip()
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:].strip()
    try:
        decoded = json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start == -1 or end == -1 or end <= start:
            logger.warning(f"Thinker LLM returned non-JSON plan: {raw!r}")
            raise
        decoded = json.loads(cleaned[start : end + 1])
    if not isinstance(decoded, dict):
        raise ValueError("Thinker LLM plan must be a JSON object")
    return decoded


def _runtime_date_context(today: date) -> str:
    tomorrow = today + timedelta(days=1)
    return (
        "\n\nRuntime context:\n"
        f"- Today is {today.isoformat()}.\n"
        f"- Tomorrow is {tomorrow.isoformat()}.\n"
        "- Treat travel dates before today as past dates and ask for a future travel date.\n"
        "- For travel dates without a year, choose the next upcoming occurrence relative to today."
    )


def _strip_thinking(text: str) -> str:
    while "<think>" in text and "</think>" in text:
        start = text.find("<think>")
        end = text.find("</think>", start)
        text = text[:start] + text[end + len("</think>") :]
    return text
