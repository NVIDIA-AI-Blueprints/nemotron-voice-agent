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
    }

    #: Sent top level on every request: Hermes' ``custom`` provider adds it itself (not a request override).
    reasoning_effort: str
    #: OpenAI SDK per-read timeout, the base of Hermes' stale-stream detector.
    request_timeout_s: float
    #: Attempts per model call (connection, timeout, rate-limit and 5xx errors).
    max_retries: int

    @classmethod
    def parse(cls, pick: Callable[[str, type], Any]) -> LLMSettings:
        """Typed fields; ``pick(key, type)`` reads and converts one key of the section."""
        settings = cls(
            reasoning_effort=pick("reasoning_effort", str),
            request_timeout_s=pick("request_timeout_s", float),
            max_retries=pick("max_retries", int),
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
        }

    def summary(self) -> dict[str, Any]:
        """For logs and ``/health``."""
        return self.worker_settings()


def worker_prompts(templates: Any) -> dict[str, Any]:
    """The prompt text the LLM host needs: the soul and the harness templates (rendered per call by the host)."""
    return {
        "soul": templates.render("backend_soul"),
        "harness": {key: templates.source(key) for key in HARNESS_KEYS},
    }
