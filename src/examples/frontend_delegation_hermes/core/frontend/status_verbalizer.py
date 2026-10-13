# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Backend progress report -> one spoken sentence (D7).

``llm`` asks the frontend model (no tools) and falls back to the deterministic
template on timeout or error; ``template`` uses the template only.
"""

from __future__ import annotations

import asyncio
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol

from examples.frontend_delegation_hermes.core.frontend.llm import ChatModel
from examples.frontend_delegation_hermes.core.frontend.prompts import PromptCatalog


@dataclass(frozen=True, slots=True)
class Verbalized:
    """The sentence and where it came from."""

    text: str
    source: str  # llm | template | template_fallback
    usage: dict[str, int] = field(default_factory=dict)
    latency_ms: float = 0.0


class StatusVerbalizer(Protocol):
    """Turns a status summary into speech text."""

    async def verbalize(self, summary: Mapping[str, Any]) -> Verbalized:
        """Never raises for model failures."""
        ...


def summary_text(summary: Mapping[str, Any]) -> str:
    """A compact, factual rendering of the summary for the model."""
    lines: list[str] = []
    if summary.get("task"):
        lines.append(f"task: {summary['task']}")
    if summary.get("elapsed_s") is not None:
        lines.append(f"working for: {summary['elapsed_s']} s")
    current = summary.get("current_tool")
    lines.append(f"doing now: {current}" if current else "doing now: preparing the reply")
    for step in summary.get("tools_done") or ():
        result = f" -> {step.get('result_preview')}" if step.get("result_preview") else ""
        lines.append(f"done: {step.get('name')}({step.get('args_preview', '')}){result}")
    outstanding = summary.get("outstanding") or ()
    if outstanding:
        lines.append(f"waiting for: {', '.join(str(name) for name in outstanding)}")
    return "\n".join(lines)


def _tool_phrase(name: str) -> str:
    words = re.sub(r"[_\-]+", " ", name).strip()
    return f"working on {words}" if words else "working on it"


class TemplateVerbalizer:
    """Deterministic sentence from the ``status_template`` prompt."""

    def __init__(self, prompts: PromptCatalog) -> None:
        """Bind the catalog."""
        self._prompts = prompts

    async def verbalize(self, summary: Mapping[str, Any], *, source: str = "template") -> Verbalized:
        """Render the template."""
        current = summary.get("current_tool") or ""
        text = self._prompts.render(
            "status_template",
            current_tool=current,
            current_tool_phrase=_tool_phrase(str(current)),
            done_count=len(summary.get("tools_done") or ()),
        )
        return Verbalized(text=text, source=source)


class LLMVerbalizer:
    """The frontend model, bounded by a timeout, with the template as fallback."""

    def __init__(self, model: ChatModel, prompts: PromptCatalog, *, timeout_ms: int) -> None:
        """Bind a shared model client."""
        self._model = model
        self._prompts = prompts
        self._timeout = timeout_ms / 1000.0
        self._fallback = TemplateVerbalizer(prompts)

    async def verbalize(self, summary: Mapping[str, Any]) -> Verbalized:
        """One no-tools call; the template on any failure."""
        prompt = self._prompts.render("status_verbalize", summary_text=summary_text(summary))
        try:
            reply = await asyncio.wait_for(
                self._model.complete([{"role": "user", "content": prompt}]), timeout=self._timeout
            )
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 - timeout or endpoint error: speak the template
            return await self._fallback.verbalize(summary, source="template_fallback")
        text = " ".join((reply.content or "").split())
        if not text:
            return await self._fallback.verbalize(summary, source="template_fallback")
        return Verbalized(text=text, source="llm", usage=dict(reply.usage), latency_ms=reply.latency_ms)
