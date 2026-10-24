# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: D100, D101, D102, D103, D107

r"""HermesAgentHost against a real Hermes ``AIAgent`` and a fake OpenAI server.

Skipped unless Hermes is importable, i.e. it never runs in this repo's venv. Run it with
the Hermes Python 3.14 venv (no pytest there, so via unittest), from the repository
root, WITHOUT ``src`` on the path (``src/utils.py`` would shadow Hermes' ``utils``):

    PYTHONPATH=/tmp/fdh-workers/pythonpath:tests/unit/frontend_delegation_hermes/core \\
        ~/.cache/fdh/hermes-venv-314/bin/python -m unittest -v test_fdh_hermes_adapter

(``/tmp/fdh-workers/pythonpath`` holds only an ``examples`` symlink; the gateway creates it,
or ``mkdir -p /tmp/fdh-workers/pythonpath && ln -s $PWD/src/examples /tmp/fdh-workers/pythonpath/``.)
"""

from __future__ import annotations

import importlib.util
import os
import tempfile
import threading
import unittest
from pathlib import Path
from typing import Any

HERMES = importlib.util.find_spec("run_agent") is not None

SOUL_MARKER = "FDH-SOUL-MARKER voice backend"
CONFIG_YAML = """model:
  context_length: 131072
tools:
  tool_search:
    enabled: "off"
agent:
  coding_context: "off"
  task_completion_guidance: false
  parallel_tool_call_guidance: false
  tool_use_enforcement: false
  execution_guidance: false
"""

_HOME: str | None = None
_SERVER: Any = None


def setUpModule() -> None:  # noqa: N802
    global _HOME, _SERVER
    if not HERMES:
        return
    _HOME = tempfile.mkdtemp(prefix="fdh-hermes-test-")
    Path(_HOME, "SOUL.md").write_text(SOUL_MARKER + "\n")
    Path(_HOME, "config.yaml").write_text(CONFIG_YAML)
    os.environ["HERMES_HOME"] = _HOME  # before any Hermes import
    os.environ["HERMES_YOLO_MODE"] = "1"
    os.environ["FDH_FAKE_KEY"] = "sk-fake"
    from _fdh_fake_openai import FakeOpenAI  # noqa: PLC0415

    _SERVER = FakeOpenAI().start()


def tearDownModule() -> None:  # noqa: N802
    if _SERVER is not None:
        _SERVER.stop()


TOOL = {
    "name": "get_order",
    "description": "Look up an order.",
    "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}}, "required": ["order_id"]},
}


def hermes_settings() -> dict[str, Any]:
    return {
        "model": "fake-model",
        "base_url": _SERVER.base_url,
        "api_key_env": "FDH_FAKE_KEY",
        "provider": "custom",
        "agent_construct": "eager",
        "max_iterations": 5,
        "run_budget_seconds": 60,
        "load_soul_identity": True,
        "disable_streaming": True,
        "request_overrides": {},
    }


@unittest.skipUnless(HERMES, "Hermes is not importable in this interpreter")
class HermesAdapterTest(unittest.TestCase):
    def test_bridge_epoch_soul_surface_usage_and_close(self) -> None:
        from examples.frontend_delegation_hermes.core.backends.hermes.host import HermesAgentHost  # noqa: PLC0415
        from examples.frontend_delegation_hermes.core.worker.tool_futures import ToolFutures  # noqa: PLC0415

        sent: list[dict[str, Any]] = []
        activity: list[tuple[str, str]] = []
        holder: dict[str, ToolFutures] = {}

        def emit(message: dict[str, Any]) -> None:
            sent.append(message)
            if message["type"] == "tool.call":
                threading.Timer(0.05, lambda: holder["f"].resolve(message["call_id"], '{"status": "shipped"}')).start()

        futures = ToolFutures(session_id="s", emit=emit, poll_s=0.02)
        holder["f"] = futures
        host = HermesAgentHost(futures, lambda event, name, preview: activity.append((event, name)), session_id="s")
        names = host.configure(tools=[TOOL], instructions="POLICY-TEXT", settings=hermes_settings())
        self.assertEqual(names, ["get_order"])
        self.assertTrue(host.built)
        agent = host.agent()
        raw = agent._agent  # noqa: SLF001
        self.assertEqual({t["function"]["name"] for t in raw.tools}, {"get_order"})
        if hasattr(raw, "_build_system_prompt"):
            self.assertIn(SOUL_MARKER, raw._build_system_prompt())  # noqa: SLF001
        before = agent.usage_snapshot()
        futures.current_epoch = 1
        result = agent.run_conversation("What is the status of order 1?", [], "s:1")
        self.assertIn("shipped", result["final_response"])
        self.assertIn("schema: ['order_id']", result["final_response"])
        calls = [m for m in sent if m["type"] == "tool.call"]
        self.assertEqual((len(calls), calls[0]["epoch"], calls[0]["name"]), (1, 1, "get_order"))
        after = agent.usage_snapshot()
        self.assertGreater(after["input_tokens"], before["input_tokens"])
        summary = agent.get_activity_summary()
        for key in ("current_tool", "api_call_count"):
            self.assertIn(key, summary)
        self.assertIn(("tool_started", "get_order"), activity)
        self.assertEqual(_SERVER.requests[0]["messages"][0]["role"], "system")
        self.assertIn("POLICY-TEXT", _SERVER.requests[0]["messages"][0]["content"])
        # A stale epoch is refused without a tool.call.
        futures.current_epoch = 2
        sent.clear()
        result = agent.run_conversation("order 1 again please", [], "s:1")
        self.assertIn("stale run", result["final_response"])
        self.assertEqual([m for m in sent if m["type"] == "tool.call"], [])
        host.close()
        from tools.registry import registry  # noqa: PLC0415

        self.assertNotIn("get_order", registry.get_all_tool_names())

    def test_trace_records_each_request_and_response(self) -> None:
        """P0.3 (tau3-geval-failure-fixes-plan.md): Hermes' own hooks feed the trace; logging only."""
        import json  # noqa: PLC0415

        from examples.frontend_delegation_hermes.core.backends.hermes.host import HermesAgentHost  # noqa: PLC0415
        from examples.frontend_delegation_hermes.core.worker.tool_futures import ToolFutures  # noqa: PLC0415

        holder: dict[str, ToolFutures] = {}

        def emit(message: dict[str, Any]) -> None:
            if message["type"] == "tool.call":
                threading.Timer(0.05, lambda: holder["f"].resolve(message["call_id"], '{"status": "shipped"}')).start()

        with tempfile.TemporaryDirectory() as trace_dir:
            futures = ToolFutures(session_id="trace-s", emit=emit, poll_s=0.02)
            holder["f"] = futures
            host = HermesAgentHost(futures, lambda *_: None, session_id="trace-s")
            host.configure(
                tools=[TOOL],
                instructions="TRACE-POLICY",
                settings={**hermes_settings(), "backend_trace_dir": trace_dir},
            )
            futures.current_epoch = 1
            result = host.agent().run_conversation("What is the status of order 1?", [], "trace-s:1")
            self.assertIn("shipped", result["final_response"])
            host.close()
            records = [json.loads(line) for line in Path(trace_dir, "trace-s.trace.jsonl").read_text().splitlines()]
        kinds = [r["kind"] for r in records]
        self.assertEqual(kinds[0], "session")
        self.assertEqual(kinds.count("request"), kinds.count("response"))
        self.assertGreaterEqual(kinds.count("request"), 2)  # the tool call, then the answer
        self.assertEqual({t["function"]["name"] for t in records[0]["tools"]}, {"get_order"})
        request = next(r for r in records if r["kind"] == "request")
        self.assertIn("TRACE-POLICY", json.dumps(request["messages"]))
        first = next(r for r in records if r["kind"] == "response")
        self.assertEqual(first["api_request_id"], request["api_request_id"])
        self.assertIn("get_order", json.dumps(first["assistant_message"]))

    def test_builtin_collision_is_refused(self) -> None:
        from model_tools import get_tool_definitions  # noqa: PLC0415

        from examples.frontend_delegation_hermes.core.backends.hermes.host import (  # noqa: PLC0415
            HermesAgentHost,
            ToolSurfaceError,
        )
        from examples.frontend_delegation_hermes.core.worker.tool_futures import ToolFutures  # noqa: PLC0415

        builtin = get_tool_definitions(quiet_mode=True)[0]["function"]["name"]
        host = HermesAgentHost(ToolFutures(session_id="s", emit=lambda m: None), lambda *a: None, session_id="s")
        with self.assertRaisesRegex(ToolSurfaceError, "collide"):
            host.configure(tools=[{**TOOL, "name": builtin}], instructions="", settings=hermes_settings())


if __name__ == "__main__":
    unittest.main()
