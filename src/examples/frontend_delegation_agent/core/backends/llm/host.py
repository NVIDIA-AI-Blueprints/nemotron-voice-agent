# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``LLMAgentHost``: builds the session's ``LLMAgent`` from the worker ``configure`` message.

``settings`` carries the shared ``backend:`` values, the ``llm:`` section and, from the spec's
``worker_prompts`` hook, the rendered soul and the ``harness_*`` templates of
``prompts/<profile>/backend/llm.yaml``. The system message is built once per session: each
part stripped, joined with a blank line, the instructions last.
"""

from __future__ import annotations

import os
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

import jinja2
import openai

from examples.frontend_delegation_agent.core.backends.llm.agent import LLMAgent
from examples.frontend_delegation_agent.core.backends.llm.client import ChatClient
from examples.frontend_delegation_agent.core.backends.llm.tools import model_tools
from examples.frontend_delegation_agent.core.backends.trace import BackendTrace
from examples.frontend_delegation_agent.core.worker.tool_futures import ToolFutures
from examples.frontend_delegation_agent.core.worker.worker_core import ActivityEmit


def conversation_started(now: datetime | None = None) -> str:
    """The session's start date as ``harness_system`` shows it: ``Wednesday, October 14, 2026 (UTC, UTC+00:00)``."""
    now = now if now is not None and now.tzinfo is not None else (now or datetime.now()).astimezone()
    zone = os.environ.get("TZ", "")
    abbreviation = now.tzname() or ""
    offset = now.strftime("%z")
    bits = [zone] if zone else []
    if abbreviation and abbreviation != zone:
        bits.append(abbreviation)
    if offset:
        bits.append(f"UTC{offset[:3]}:{offset[3:]}")
    suffix = f" ({', '.join(bits)})" if bits else ""
    return f"{now.strftime('%A, %B %d, %Y')}{suffix}"


def system_message(soul: str, harness_system: str, instructions: str) -> str:
    """The system message: the stripped parts joined with a blank line, the instructions appended last."""
    cached = "\n\n".join(part.strip() for part in (soul, harness_system) if part.strip())
    return f"{cached}\n\n{instructions}".strip() if instructions.strip() else cached


class LLMAgentHost:
    """Builds and owns the ``LLMAgent`` of one session (``Placement.IN_PROCESS``)."""

    def __init__(
        self, futures: ToolFutures, on_activity: ActivityEmit, *, session_id: str, in_process: bool = True
    ) -> None:
        """Bind the tool bridge; nothing is built until :meth:`configure`."""
        del in_process
        self._futures = futures
        self._on_activity = on_activity
        self._session_id = session_id
        self._tools: list[dict[str, Any]] = []
        self._instructions = ""
        self._settings: dict[str, Any] = {}
        self._agent: LLMAgent | None = None
        self._lock = threading.Lock()

    def configure(self, *, tools: list[dict[str, Any]], instructions: str, settings: dict[str, Any]) -> list[str]:
        """Keep the session's tools and prompt; build the agent now unless construction is lazy."""
        if self._agent is not None:
            raise RuntimeError("the agent already exists; its tools cannot change")
        names = [str(t["name"]) for t in tools]
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate tool names: {names}")
        self._tools, self._instructions, self._settings = list(tools), instructions, dict(settings)
        if str(settings.get("agent_construct", "eager")) == "eager":
            self.agent()
        return sorted(names)

    def agent(self) -> LLMAgent:
        """The agent (built on first use)."""
        with self._lock:
            if self._agent is None:
                self._agent = self._build()
            return self._agent

    @property
    def built(self) -> bool:
        """Whether the agent exists."""
        return self._agent is not None

    def is_interrupted(self) -> bool:
        """Whether the running turn was interrupted (the tool bridge stops waiting on it)."""
        return self._agent is not None and self._agent.is_interrupted()

    def version(self) -> str:
        """Implementation tag."""
        return f"fda-llm (openai {openai.__version__})"

    def close(self) -> None:
        """Close the agent's client."""
        if self._agent is not None:
            self._agent.close()

    def _build(self) -> LLMAgent:
        s = self._settings
        api_key = os.environ.get(str(s.get("api_key_env") or "NVIDIA_API_KEY"), "")
        if not api_key:
            raise RuntimeError(f"environment variable {s.get('api_key_env')} is not set")
        env = jinja2.Environment(undefined=jinja2.StrictUndefined, autoescape=False, keep_trailing_newline=False)  # noqa: S701 - plain text prompts
        templates = {key: env.from_string(source) for key, source in dict(s.get("harness") or {}).items()}

        def render(key: str, **variables: Any) -> str:
            return templates[key].render(**variables).strip()

        model = str(s["model"])
        system = system_message(
            str(s.get("soul") or ""),
            render("harness_system", started=conversation_started(), model=model),
            self._instructions,
        )
        request_tools = model_tools(self._tools)
        client = ChatClient(
            base_url=str(s["base_url"]),
            api_key=api_key,
            model=model,
            timeout_s=float(s.get("request_timeout_s", 180.0)),
            max_retries=int(s.get("max_retries", 3)),
            reasoning_effort=str(s.get("reasoning_effort") or ""),
            request_overrides=dict(s.get("request_overrides") or {}),
        )
        trace = None
        if s.get("backend_trace_dir"):
            trace = BackendTrace(
                Path(str(s["backend_trace_dir"])) / f"{self._session_id}.trace.jsonl", session_id=self._session_id
            )
            trace.session(
                tools=request_tools,
                model=model,
                instructions=self._instructions,
                request_overrides=dict(s.get("request_overrides") or {}),
            )
        return LLMAgent(
            client=client,
            system=system,
            tools=request_tools,
            call_tool=self._futures.call,
            render=render,
            max_iterations=int(s.get("max_iterations", 30)),
            on_activity=self._on_activity,
            trace=trace,
        )
