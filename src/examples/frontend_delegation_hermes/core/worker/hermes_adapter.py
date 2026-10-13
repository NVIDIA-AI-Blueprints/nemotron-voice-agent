# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The only module that imports Hermes (``workers.agent_kind: hermes``; Python 3.14 venv).

Facts verified live in the P0 spike (2026-09-29, Hermes ``af26acab73``):

* ``HERMES_HOME/config.yaml`` must set ``tools.tool_search.enabled: "off"``, otherwise
  every session tool hides behind ``tool_search`` / ``tool_describe`` / ``tool_call``
  (three model calls per tool). The gateway writes it together with
  ``model.context_length`` (the 64K floor) and the ``agent.*`` guidance switches.
* ``HERMES_YOLO_MODE=1`` and ``HERMES_HOME`` are frozen at import time, so the pool sets
  them in the worker's environment before this module imports Hermes.
* ``load_soul_identity=True`` is required with ``skip_context_files=True`` or SOUL.md is
  not loaded. Construction takes ~0.2-0.7 s, import ~0.5-1.9 s, RSS ~155 MB.
* Tool handlers get ``task_id="<session_id>:<epoch>"`` and ``session_id`` as kwargs.
* ``registry.register`` can silently no-op (shadow rejection): registration is
  verified, collisions with Hermes builtins are refused (no ``override=True``), and the
  constructed agent's tool surface must equal the session tools exactly.
* ``steer()`` mid-tool arrives as an out-of-band user row; ``redirect()`` works while
  ``agent._model_request_active`` is set; ``hard_interrupt`` unwinds in ~4 s during a
  blocking tool; ``session_*_tokens`` counters are cumulative.

With ``backend_trace_dir`` set (``hermes.backend_trace``, P0.3 of tau3-geval-failure-fixes-plan.md),
:class:`BackendTrace` appends each model request and response to
``<dir>/<session_id>.trace.jsonl`` through Hermes' ``pre_api_request`` / ``post_api_request``
lifecycle hooks. Logging only: nothing it does reaches the model or the call.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any

from examples.frontend_delegation_hermes.core.prompt_features import tools_canonical_sha256
from examples.frontend_delegation_hermes.core.worker.tool_futures import ToolFutures
from examples.frontend_delegation_hermes.core.worker.worker_core import ActivityEmit

TOOLSET = "fdh_client"
#: Bumped when the trace record layout changes (cli/backend_decision_replay.py reads it).
TRACE_VERSION = 1

logger = logging.getLogger(__name__)


class ToolSurfaceError(RuntimeError):
    """The agent would not see exactly the session's tools."""


class BackendTrace:
    """One worker's decision trace: a ``session`` record, then a ``request`` / ``response`` pair per model call.

    ``request`` carries the messages exactly as sent (system prompt included) and the hash of the
    tool schemas of the ``session`` record; ``response`` the assistant message (content and tool calls),
    the finish reason and the usage. Records are joined by ``api_request_id``. A write error is logged
    once and the trace stops; it never fails a model call.
    """

    def __init__(self, path: Path, *, session_id: str) -> None:
        """Append to ``path`` (respawned workers of a session append to the same file)."""
        self.path = path
        self._session_id = session_id
        self._lock = threading.Lock()
        self._broken = False
        path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, kind: str, **fields: Any) -> None:
        """Append one record."""
        if self._broken:
            return
        record = {"v": TRACE_VERSION, "ts": round(time.time(), 3), "kind": kind, "session_id": self._session_id}
        record.update(fields)
        try:
            line = json.dumps(record, ensure_ascii=False, default=_jsonable)
            with self._lock, self.path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")
        except Exception:  # noqa: BLE001 - tracing must never break a run
            self._broken = True
            logger.exception("backend trace disabled after a write error (%s)", self.path)

    def session(self, *, tools: Any, model: str, instructions: str, request_overrides: Any) -> None:
        """The session's tool schemas (as the model sees them), model and ephemeral system prompt."""
        self.write(
            "session",
            model=model,
            tools=tools,
            tools_sha256=_sha256_json(tools),
            tools_canonical_sha256=tools_canonical_sha256(tools),
            ephemeral_system_prompt=instructions,
            request_overrides=request_overrides,
        )

    def on_request(self, **kwargs: Any) -> None:
        """Hermes ``pre_api_request`` hook."""
        if str(kwargs.get("session_id") or self._session_id) != self._session_id:
            return
        self.write(
            "request",
            api_request_id=kwargs.get("api_request_id"),
            task_id=kwargs.get("task_id"),
            turn_id=kwargs.get("turn_id"),
            api_call_count=kwargs.get("api_call_count"),
            retry_count=kwargs.get("retry_count"),
            model=kwargs.get("model"),
            messages=kwargs.get("request_messages"),
            system_prompt=kwargs.get("system_prompt"),
            tool_count=kwargs.get("tool_count"),
        )

    def on_response(self, **kwargs: Any) -> None:
        """Hermes ``post_api_request`` hook."""
        if str(kwargs.get("session_id") or self._session_id) != self._session_id:
            return
        response = kwargs.get("response") or {}
        self.write(
            "response",
            api_request_id=kwargs.get("api_request_id"),
            task_id=kwargs.get("task_id"),
            turn_id=kwargs.get("turn_id"),
            api_call_count=kwargs.get("api_call_count"),
            finish_reason=kwargs.get("finish_reason"),
            assistant_message=response.get("assistant_message") if isinstance(response, dict) else None,
            usage=kwargs.get("usage"),
            api_duration=kwargs.get("api_duration"),
        )


def register_trace_hooks(trace: BackendTrace) -> bool:
    """Add the trace to Hermes' lifecycle hooks of this process (one worker process = one session)."""
    try:
        from hermes_cli.plugins import get_plugin_manager  # noqa: PLC0415 - Hermes, lazily
    except ImportError:
        logger.warning("backend trace: hermes_cli.plugins is not importable; no trace is written")
        return False
    manager = get_plugin_manager()
    discover = getattr(manager, "discover_and_load", None)
    if callable(discover):
        discover()  # what Hermes does lazily on its first hook check anyway; after it, hooks are kept
    hooks = getattr(manager, "_hooks", None)
    if not isinstance(hooks, dict):
        logger.warning("backend trace: this Hermes version has no hook registry; no trace is written")
        return False
    hooks.setdefault("pre_api_request", []).append(trace.on_request)
    hooks.setdefault("post_api_request", []).append(trace.on_response)
    return True


def _jsonable(value: Any) -> Any:
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        return dump()
    if hasattr(value, "__dict__"):
        return {k: v for k, v in vars(value).items() if not k.startswith("_")}
    return str(value)


def _sha256_json(value: Any) -> str:
    import hashlib  # noqa: PLC0415

    blob = json.dumps(value, sort_keys=True, ensure_ascii=False, default=_jsonable)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


class HermesAgentAdapter:
    """``AgentLike`` over a Hermes ``AIAgent``."""

    def __init__(self, agent: Any) -> None:
        """Wrap a constructed ``AIAgent``."""
        self._agent = agent

    def run_conversation(
        self, user_message: str, conversation_history: list[dict[str, Any]], task_id: str
    ) -> dict[str, Any]:
        """One blocking Hermes turn."""
        return self._agent.run_conversation(user_message, conversation_history=conversation_history, task_id=task_id)

    def steer(self, text: str) -> bool:
        """Deliver after the current tool batch."""
        return bool(self._agent.steer(text))

    def redirect(self, text: str) -> bool:
        """Abort the in-flight model request and retry with ``text``."""
        return bool(self._agent.redirect(text))

    def get_activity_summary(self) -> dict[str, Any]:
        """Hermes' activity summary."""
        return dict(self._agent.get_activity_summary())

    def clear_interrupt(self) -> None:
        """Drop an interrupt left over from an idle period."""
        self._agent.clear_interrupt()

    def hard_interrupt(self, message: str) -> None:
        """Stop the running turn."""
        self._agent.hard_interrupt(message)

    def model_request_active(self) -> bool:
        """Whether a provider call is in flight."""
        event = getattr(self._agent, "_model_request_active", None)
        return bool(event is not None and event.is_set())

    def usage_snapshot(self) -> dict[str, int]:
        """Cumulative counters."""
        a = self._agent
        return {
            "input_tokens": int(getattr(a, "session_input_tokens", 0) or getattr(a, "session_prompt_tokens", 0)),
            "output_tokens": int(getattr(a, "session_output_tokens", 0) or getattr(a, "session_completion_tokens", 0)),
            "reasoning_tokens": int(getattr(a, "session_reasoning_tokens", 0)),
        }

    def close(self) -> None:
        """Release clients and resources."""
        self._agent.close()


class HermesAgentHost:
    """Registers the session's tools in this process's Hermes registry and builds the ``AIAgent``."""

    def __init__(self, futures: ToolFutures, on_activity: ActivityEmit, *, session_id: str) -> None:
        """Bind the bridge; Hermes is imported on :meth:`configure`."""
        self._futures = futures
        self._on_activity = on_activity
        self._session_id = session_id
        self._names: list[str] = []
        self._instructions = ""
        self._hermes: dict[str, Any] = {}
        self._adapter: HermesAgentAdapter | None = None
        self._lock = threading.Lock()

    # -- AgentHost ---------------------------------------------------------------------

    def configure(self, *, tools: list[dict[str, Any]], instructions: str, hermes: dict[str, Any]) -> list[str]:
        """Register tools (refusing builtin collisions), then build the agent when eager."""
        if self._adapter is not None:
            raise ToolSurfaceError("the agent already exists; its tools cannot change")
        os.environ.setdefault("HERMES_YOLO_MODE", "1")
        from model_tools import get_tool_definitions  # noqa: PLC0415 - Hermes, lazily
        from tools.registry import registry  # noqa: PLC0415
        from toolsets import create_custom_toolset  # noqa: PLC0415

        for old in self._names:  # a reconfigure before the agent exists replaces the tools
            registry.deregister(old)
        names = [str(t["name"]) for t in tools]
        if len(set(names)) != len(names):
            raise ToolSurfaceError(f"duplicate tool names: {names}")
        builtins = {d["function"]["name"] for d in get_tool_definitions(quiet_mode=True)}
        clash = sorted(set(names) & builtins)
        if clash:
            raise ToolSurfaceError(f"session tools collide with Hermes builtin tools: {clash}")
        create_custom_toolset(TOOLSET, "Realtime session tools executed by the voice client", sorted(names))
        for tool in tools:
            name = str(tool["name"])
            registry.register(
                name=name,
                toolset=TOOLSET,
                schema={
                    "name": name,
                    "description": str(tool.get("description") or name),
                    "parameters": tool.get("parameters") or {"type": "object", "properties": {}},
                },
                handler=self._handler(name),
            )
        missing = sorted(set(names) - set(registry.get_all_tool_names()))
        if missing:
            raise ToolSurfaceError(f"Hermes' registry did not accept {missing} (look for 'registration REJECTED')")
        self._names = names
        self._instructions = instructions
        self._hermes = dict(hermes)
        if str(hermes.get("agent_construct", "eager")) == "eager":
            self.agent()
        return sorted(names)

    def agent(self) -> HermesAgentAdapter:
        """The ``AIAgent`` adapter (built on first use)."""
        with self._lock:
            if self._adapter is None:
                self._adapter = HermesAgentAdapter(self._build())
            return self._adapter

    @property
    def built(self) -> bool:
        """Whether the agent exists."""
        return self._adapter is not None

    def is_interrupted(self) -> bool:
        """Hermes' per-thread interrupt bit for the calling tool thread."""
        from tools.interrupt import is_interrupted  # noqa: PLC0415

        return bool(is_interrupted())

    def version(self) -> str:
        """Hermes package version."""
        try:
            from importlib.metadata import version  # noqa: PLC0415

            return f"hermes-agent {version('hermes-agent')}"
        except Exception:  # noqa: BLE001
            return "hermes-agent"

    def close(self) -> None:
        """Close the agent and deregister this session's tools."""
        if self._adapter is not None:
            try:
                self._adapter.close()
            finally:
                self._adapter = None
        try:
            from tools.registry import registry  # noqa: PLC0415

            for name in self._names:
                registry.deregister(name)
        except Exception:  # noqa: BLE001, S110 - the process is about to exit anyway
            pass

    # -- internals ----------------------------------------------------------------------

    def _handler(self, name: str) -> Any:
        futures = self._futures

        def handler(args: dict[str, Any], **kwargs: Any) -> str:
            return futures.call(name, dict(args or {}), kwargs.get("task_id"))

        return handler

    def _build(self) -> Any:
        from run_agent import AIAgent  # noqa: PLC0415

        h = self._hermes
        api_key = os.environ.get(str(h.get("api_key_env") or "NVIDIA_API_KEY"), "")
        if not api_key:
            raise RuntimeError(f"environment variable {h.get('api_key_env')} is not set in the worker")
        agent = AIAgent(
            model=str(h["model"]),
            base_url=str(h["base_url"]),
            api_key=api_key,
            provider=str(h.get("provider") or "custom"),
            enabled_toolsets=[TOOLSET],
            quiet_mode=True,
            skip_context_files=True,
            load_soul_identity=bool(h.get("load_soul_identity", True)),
            skip_memory=True,
            skip_background_review=True,
            save_trajectories=False,
            session_id=self._session_id,
            max_iterations=int(h.get("max_iterations", 30)),
            run_budget_seconds=float(h.get("run_budget_seconds", 300.0)),
            platform="voice",
            ephemeral_system_prompt=self._instructions or None,
            request_overrides=dict(h.get("request_overrides") or {}),
            tool_start_callback=self._tool_started,
            tool_complete_callback=self._tool_completed,
        )
        if h.get("disable_streaming"):
            agent._disable_streaming = True  # non-streaming test servers only (Hermes streams by default)
        check_tool_surface(set(self._names), agent)
        trace_dir = str(h.get("backend_trace_dir") or "")
        if trace_dir:
            trace = BackendTrace(Path(trace_dir) / f"{self._session_id}.trace.jsonl", session_id=self._session_id)
            if register_trace_hooks(trace):
                trace.session(
                    tools=getattr(agent, "tools", None),
                    model=str(h["model"]),
                    instructions=self._instructions,
                    request_overrides=dict(h.get("request_overrides") or {}),
                )
        return agent

    def _tool_started(self, tool_call_id: Any, name: Any, args: Any, *_: Any, **__: Any) -> None:
        self._on_activity("tool_started", str(name), str(args)[:200])

    def _tool_completed(self, call_id: Any, name: Any, args: Any, result: Any = None, *_: Any, **__: Any) -> None:
        self._on_activity("tool_completed", str(name), str(result)[:200])


def check_tool_surface(expected: set[str], agent: Any) -> None:
    """Raise unless the agent shows the model exactly ``expected`` (tau2-hermes pattern)."""
    actual = {
        t["function"]["name"]
        for t in (getattr(agent, "tools", None) or [])
        if isinstance(t, dict) and isinstance(t.get("function"), dict)
    }
    if actual == expected:
        return
    hint = ""
    if {"tool_search", "tool_describe", "tool_call"} & actual:
        hint = ' Set tools.tool_search.enabled: "off" in $HERMES_HOME/config.yaml.'
    raise ToolSurfaceError(
        f"Hermes' tool surface does not match the session tools: missing={sorted(expected - actual)} "
        f"unexpected={sorted(actual - expected)}.{hint}"
    )
