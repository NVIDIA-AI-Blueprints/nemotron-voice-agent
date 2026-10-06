# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Immutable message, tool-call, usage, and outward-envelope types.

Every type here is deeply immutable: tuples instead of lists, and tool-call
arguments held as canonical JSON text rather than a mutable mapping. That is
what lets :class:`~examples.frontend_backend_verdict.text.session.SessionState`
be a genuine value object rather than a frozen shell around shared mutables.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field, replace
from typing import Any, Literal

Role = Literal["system", "user", "assistant", "tool"]

#: Roles that make LLM calls, for per-role accounting.
LLM_ROLES = ("frontend", "backend")


def new_tool_call_id() -> str:
    """Return a fresh tool-call id, used only when a provider omits one."""
    return f"call_{uuid.uuid4().hex[:16]}"


def canonical_json(value: Any) -> str:
    """Serialize ``value`` deterministically (sorted keys, no ASCII escaping)."""
    return json.dumps(value, sort_keys=True, ensure_ascii=False)


@dataclass(frozen=True, slots=True)
class ToolCall:
    """One tool call emitted by a model.

    ``arguments_json`` is stored as text so the instance stays hashable and
    immutable; use :attr:`arguments` for a fresh decoded copy.
    """

    id: str
    name: str
    arguments_json: str = "{}"

    @classmethod
    def create(cls, name: str, arguments: dict[str, Any] | None = None, *, id: str | None = None) -> ToolCall:
        """Build a tool call, generating an id only when one is not supplied."""
        return cls(id=id or new_tool_call_id(), name=name, arguments_json=canonical_json(arguments or {}))

    @property
    def arguments(self) -> dict[str, Any]:
        """Return the decoded arguments; ``{}`` when the payload is not an object."""
        try:
            decoded = json.loads(self.arguments_json or "{}")
        except json.JSONDecodeError:
            return {}
        return decoded if isinstance(decoded, dict) else {}


@dataclass(frozen=True, slots=True)
class ToolResult:
    """The outcome of one tool call, already serialized to text."""

    tool_call_id: str
    content: str
    is_error: bool = False


@dataclass(frozen=True, slots=True)
class Message:
    """One conversation message in either history."""

    role: Role
    content: str | None = None
    tool_calls: tuple[ToolCall, ...] = ()
    tool_call_id: str | None = None

    @classmethod
    def user(cls, content: str) -> Message:
        """Build a user message."""
        return cls(role="user", content=content)

    @classmethod
    def assistant(cls, content: str) -> Message:
        """Build a plain assistant message."""
        return cls(role="assistant", content=content)

    @classmethod
    def assistant_tool_calls(cls, tool_calls: tuple[ToolCall, ...], content: str | None = None) -> Message:
        """Build an assistant message that carries tool calls."""
        return cls(role="assistant", content=content, tool_calls=tuple(tool_calls))

    @classmethod
    def tool(cls, tool_call_id: str, content: str) -> Message:
        """Build a tool-result message."""
        return cls(role="tool", content=content, tool_call_id=tool_call_id)

    @classmethod
    def system(cls, content: str) -> Message:
        """Build a system message."""
        return cls(role="system", content=content)


@dataclass(frozen=True, slots=True)
class Usage:
    """Token counts for one or more LLM calls."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cached_tokens: int = 0

    def __add__(self, other: Usage) -> Usage:
        """Return the element-wise sum of two usage records."""
        return Usage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
            cached_tokens=self.cached_tokens + other.cached_tokens,
        )


@dataclass(frozen=True, slots=True)
class RoleTotals:
    """Accumulated accounting for one LLM role."""

    calls: int = 0
    usage: Usage = field(default_factory=Usage)
    latency_ms: float = 0.0
    cost_known_subtotal: float = 0.0
    cost_unknown_calls: int = 0

    def __add__(self, other: RoleTotals) -> RoleTotals:
        """Return the element-wise sum of two role totals."""
        return RoleTotals(
            calls=self.calls + other.calls,
            usage=self.usage + other.usage,
            latency_ms=self.latency_ms + other.latency_ms,
            cost_known_subtotal=self.cost_known_subtotal + other.cost_known_subtotal,
            cost_unknown_calls=self.cost_unknown_calls + other.cost_unknown_calls,
        )


@dataclass(frozen=True, slots=True)
class UsageTotals:
    """Accounting aggregated over every LLM call in a step or a session.

    ``cost`` is deliberately ``None`` whenever any contributing call had an
    unknown cost: treating a missing figure as ``0.0`` would understate a
    benchmark total, which is the failure this type exists to prevent.
    """

    frontend: RoleTotals = field(default_factory=RoleTotals)
    backend: RoleTotals = field(default_factory=RoleTotals)

    @property
    def calls(self) -> int:
        """Total number of LLM calls."""
        return self.frontend.calls + self.backend.calls

    @property
    def usage(self) -> Usage:
        """Token usage across both roles."""
        return self.frontend.usage + self.backend.usage

    @property
    def latency_ms(self) -> float:
        """Summed LLM latency in milliseconds."""
        return self.frontend.latency_ms + self.backend.latency_ms

    @property
    def cost_unknown_calls(self) -> int:
        """Number of calls whose cost could not be determined."""
        return self.frontend.cost_unknown_calls + self.backend.cost_unknown_calls

    @property
    def cost_known_subtotal(self) -> float:
        """Sum of the costs that *are* known; never report this as 'the cost'."""
        return self.frontend.cost_known_subtotal + self.backend.cost_known_subtotal

    @property
    def cost(self) -> float | None:
        """Total cost, or ``None`` when any contributing call's cost is unknown."""
        if self.calls == 0:
            return 0.0
        if self.cost_unknown_calls:
            return None
        return self.cost_known_subtotal

    def add_call(self, role: str, *, usage: Usage, latency_ms: float, cost: float | None) -> UsageTotals:
        """Return a copy with one more LLM call folded in."""
        totals = RoleTotals(
            calls=1,
            usage=usage,
            latency_ms=latency_ms,
            cost_known_subtotal=cost or 0.0,
            cost_unknown_calls=0 if cost is not None else 1,
        )
        if role == "frontend":
            return replace(self, frontend=self.frontend + totals)
        return replace(self, backend=self.backend + totals)

    def merge(self, other: UsageTotals) -> UsageTotals:
        """Return the element-wise sum of two totals."""
        return UsageTotals(frontend=self.frontend + other.frontend, backend=self.backend + other.backend)

    def summary(self) -> str:
        """Return a one-line human summary for logs and the REPL."""
        cost = "unknown" if self.cost is None else f"${self.cost:.4f}"
        return f"{self.calls} calls · {self.usage.total_tokens} tok · {self.latency_ms / 1000:.2f} s · cost {cost}"


@dataclass(frozen=True, slots=True)
class AgentTurn:
    """The single outward payload of one step.

    Carries user-visible text *or* backend tool calls for the caller to execute,
    never both and never neither — the same rule tau2-bench enforces in
    ``check_communication_error()``. There is deliberately no ``events`` field:
    diagnostics (including frontend filler text) go to the event sink only.
    """

    final_text: str | None = None
    tool_calls: tuple[ToolCall, ...] = ()
    usage: UsageTotals = field(default_factory=UsageTotals)

    def __post_init__(self) -> None:
        """Enforce the content/tool-calls exclusivity invariant."""
        if self.final_text is not None and self.tool_calls:
            raise ValueError("AgentTurn cannot carry both final_text and tool_calls")
        if self.final_text is None and not self.tool_calls:
            raise ValueError("AgentTurn must carry either final_text or tool_calls")

    @property
    def is_tool_call(self) -> bool:
        """Whether this turn hands tool calls back to the caller."""
        return bool(self.tool_calls)
