# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The ``llm:`` section of the gateway config (read only when ``backend.kind: llm``). Stdlib only."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, ClassVar

from examples.frontend_delegation_agent.core.backends.llm import HARNESS_KEYS


@dataclass(frozen=True, slots=True)
class LLMSettings:
    """Settings only the LLM backend uses; the shared ones (model, timeouts, prompts) are in ``backend:``."""

    DEFAULTS: ClassVar[dict[str, Any]] = {
        "reasoning_effort": "medium",
        "request_timeout_s": 180.0,
        "max_retries": 3,
        "verify_policy_follow_up_steps": True,
        "follow_up_rules": [],
        "follow_up_ignore_tools": [],
    }

    #: Sent top level on every request (not a request override).
    reasoning_effort: str
    #: OpenAI SDK per-read timeout: a stalled stream fails after it.
    request_timeout_s: float
    #: Attempts per model call (connection, timeout, rate-limit and 5xx errors).
    max_retries: int
    #: After a write that opens an "after X, always/must ..." rule of the client's policy, check the
    #: rule's follow-up steps before answering (``follow_up.py``). Does nothing without such a rule.
    verify_policy_follow_up_steps: bool = True
    #: Operator-declared rules, ``{tools: [...], text: "...", domain?: "..."}``, added to the extracted ones.
    follow_up_rules: tuple[dict[str, Any], ...] = ()
    #: Tools that never open an extracted rule.
    follow_up_ignore_tools: tuple[str, ...] = ()

    @classmethod
    def parse(cls, pick: Callable[[str, type], Any]) -> LLMSettings:
        """Typed fields; ``pick(key, type)`` reads and converts one key of the section."""
        settings = cls(
            reasoning_effort=pick("reasoning_effort", str),
            request_timeout_s=pick("request_timeout_s", float),
            max_retries=pick("max_retries", int),
            verify_policy_follow_up_steps=pick("verify_policy_follow_up_steps", bool),
            follow_up_rules=_follow_up_rules(pick("follow_up_rules", list)),
            follow_up_ignore_tools=_names(pick("follow_up_ignore_tools", list), "llm.follow_up_ignore_tools"),
        )
        if settings.request_timeout_s <= 0 or settings.max_retries < 1:
            raise ValueError("llm.request_timeout_s must be positive and llm.max_retries at least 1")
        return settings

    def worker_settings(self) -> dict[str, Any]:
        """The fields the LLM host reads from ``configure``."""
        return {
            "reasoning_effort": self.reasoning_effort,
            "request_timeout_s": self.request_timeout_s,
            "max_retries": self.max_retries,
            "verify_policy_follow_up_steps": self.verify_policy_follow_up_steps,
            "follow_up_rules": [dict(rule) for rule in self.follow_up_rules],
            "follow_up_ignore_tools": list(self.follow_up_ignore_tools),
        }

    def summary(self) -> dict[str, Any]:
        """For logs and ``/health`` (the rule lists only when set)."""
        summary = self.worker_settings()
        for key in ("follow_up_rules", "follow_up_ignore_tools"):
            if not summary[key]:
                del summary[key]
        return summary


def _names(value: list[Any], key: str) -> tuple[str, ...]:
    if not all(isinstance(item, str) and item.strip() for item in value):
        raise ValueError(f"{key} must be a list of tool names")
    return tuple(item.strip() for item in value)


def _follow_up_rules(value: list[Any]) -> tuple[dict[str, Any], ...]:
    """``llm.follow_up_rules``: each ``{tools: [name, ...], text: "...", domain?: "..."}``."""
    out = []
    for index, rule in enumerate(value):
        key = f"llm.follow_up_rules[{index}]"
        if not isinstance(rule, dict) or not {"tools", "text"} <= set(rule) <= {"tools", "text", "domain"}:
            raise ValueError(f"{key} must be a mapping with tools, text and an optional domain")
        tools = rule["tools"]
        if not isinstance(tools, list) or not tools:
            raise ValueError(f"{key}.tools must be a non-empty list of tool names")
        text = rule["text"]
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"{key}.text must be a non-empty string")
        domain = rule.get("domain", "")
        if not isinstance(domain, str):
            raise ValueError(f"{key}.domain must be a string")
        entry: dict[str, Any] = {"tools": list(_names(tools, f"{key}.tools")), "text": text.strip()}
        if domain.strip():
            entry["domain"] = domain.strip()
        out.append(entry)
    return tuple(out)


def worker_prompts(templates: Any) -> dict[str, Any]:
    """The prompt text the LLM host needs: the soul and the harness templates (rendered per call by the host)."""
    return {
        "soul": templates.render("backend_soul"),
        "harness": {key: templates.source(key) for key in HARNESS_KEYS},
    }
