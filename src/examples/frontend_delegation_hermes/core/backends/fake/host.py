# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``FakeAgent`` host (``backend.kind: fake``): real processes and sockets, or in-process, no Hermes.

Tool schemas are kept per host instance, like Hermes' process-global registry is kept
per worker process, so tests can prove schema isolation between sessions.
"""

from __future__ import annotations

from typing import Any

from examples.frontend_delegation_hermes.core.backends.fake.agent import FakeAgent
from examples.frontend_delegation_hermes.core.worker.tool_futures import ToolFutures
from examples.frontend_delegation_hermes.core.worker.worker_core import ActivityEmit


class FakeAgentHost:
    """Builds a :class:`FakeAgent` bridged to the worker's tool futures."""

    def __init__(
        self,
        futures: ToolFutures,
        on_activity: ActivityEmit,
        *,
        session_id: str = "",
        in_process: bool = False,
        allow_crash: bool | None = None,
    ) -> None:
        """Bind the bridge and the activity emitter; ``[crash]`` exits only inside a worker process."""
        del session_id
        self._futures = futures
        self._on_activity = on_activity
        self._allow_crash = (not in_process) if allow_crash is None else allow_crash
        self._tools: dict[str, dict[str, Any]] = {}
        self._agent: FakeAgent | None = None
        self._default_tool = False
        self.instructions = ""

    def configure(self, *, tools: list[dict[str, Any]], instructions: str, settings: dict[str, Any]) -> list[str]:
        """Record the tools; build now unless construction is lazy."""
        self._tools = {str(t["name"]): dict(t.get("parameters") or {}) for t in tools}
        self.instructions = instructions
        self._default_tool = bool(settings.get("fake_default_tool", False))
        if str(settings.get("agent_construct", "eager")) == "eager":
            self.agent()
        return sorted(self._tools)

    def agent(self) -> FakeAgent:
        """The agent (built on first use)."""
        if self._agent is None:
            self._agent = FakeAgent(
                call_tool=self._futures.call,
                tools=self._tools,
                on_activity=self._on_activity,
                allow_crash=self._allow_crash,
                default_tool=self._default_tool,
            )
        return self._agent

    @property
    def built(self) -> bool:
        """Whether the agent exists."""
        return self._agent is not None

    def is_interrupted(self) -> bool:
        """The fake agent has no per-thread interrupts."""
        return False

    def version(self) -> str:
        """Implementation tag."""
        return "fake"

    def close(self) -> None:
        """Close the agent."""
        if self._agent is not None:
            self._agent.close()
