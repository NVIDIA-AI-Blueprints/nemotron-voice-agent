# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: D100, D101, D102, D103, D107

"""The LLM backend in the voice process (llm-backend-plan.md sections 3 and 7).

One gateway config for both backends (only ``FDA_BACKEND`` differs), the eval profile chain
selecting the in-process link, and a whole session through ``InProcessBackendLink``: the shared
controller, an ``LLMAgent`` worker thread, a tool round trip and the answer.
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from typing import Any
from unittest import mock

from _fda_llm_server import Reply, ScriptedOpenAI, tool

from examples.frontend_delegation_agent.core.config import load_delegation_config
from examples.frontend_delegation_agent.core.sidecar import gateway_config as gc
from examples.frontend_delegation_agent.core.sidecar.inprocess_link import InProcessBackendLink

REPO_ROOT = Path(__file__).resolve().parents[4]
PROFILES = gc.CONFIG_DIR / "profiles"
TOOL = {
    "type": "function",
    "name": "get_order",
    "description": "Look up an order.",
    "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}}, "required": ["order_id"]},
}


def _env(**values: str) -> Any:
    env = {k: v for k, v in os.environ.items() if not k.startswith(("FDA_", "BACKEND_LLM_"))}
    env.update(values)
    return mock.patch.dict(os.environ, env, clear=True)


class ConfigTest(unittest.TestCase):
    def test_the_eval_gateway_gives_both_backends_the_same_settings(self) -> None:
        with _env():
            hermes = gc.load_gateway_config(gc.CONFIG_DIR / "gateway.eval.yaml")
        with _env(FDA_BACKEND="llm"):
            llm = gc.load_gateway_config(gc.CONFIG_DIR / "gateway.eval.yaml")
        self.assertEqual((hermes.backend.kind, llm.backend.kind), ("hermes", "llm"))
        self.assertEqual(llm.prompt_features, hermes.prompt_features)
        self.assertEqual(llm.domains, hermes.domains)
        self.assertEqual(Path(llm.backend.prompts_path).parent, Path(hermes.backend.prompts_path).parent)
        self.assertEqual(Path(llm.backend.prompts_path).parts[-3:], ("customer_service", "backend", "llm.yaml"))
        self.assertTrue(llm.backend.backend_trace)
        shared = ("model", "base_url", "max_iterations", "request_overrides", "run_budget_seconds")
        self.assertEqual([getattr(llm.backend, k) for k in shared], [getattr(hermes.backend, k) for k in shared])

    def test_the_eval_profile_selects_the_in_process_link(self) -> None:
        with _env():
            websocket = load_delegation_config(PROFILES / "pipecat_realtime_eval.yaml")
        self.assertEqual(websocket.backend.link, "websocket")
        with _env(FDA_BACKEND_LINK="in_process", FDA_GATEWAY_CONFIG="gateway.eval.yaml"):
            config = load_delegation_config(PROFILES / "pipecat_realtime_eval.yaml")
        self.assertEqual(config.backend.link, "in_process")
        # Declared only in delegation_agent.yaml, so the file name resolves next to it, not under profiles/.
        self.assertEqual(config.backend.gateway_config, gc.CONFIG_DIR / "gateway.eval.yaml")

    def test_the_old_link_name_names_the_new_one(self) -> None:
        with _env(FDA_BACKEND_LINK="in_process_fake"), self.assertRaisesRegex(Exception, "renamed to in_process"):
            load_delegation_config(PROFILES / "pipecat_demo.yaml")

    def test_hermes_cannot_run_in_the_voice_process(self) -> None:
        with _env(), self.assertRaisesRegex(ValueError, "cannot run in the voice process"):
            InProcessBackendLink(gc.load_gateway_config(gc.CONFIG_DIR / "gateway.yaml"))


class InProcessSessionTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.server = ScriptedOpenAI().start()
        self.logs = Path(tempfile.mkdtemp(prefix="fda-llm-"))
        self.env = _env(
            FDA_BACKEND="llm",
            BACKEND_LLM_BASE_URL=self.server.base_url,
            NVIDIA_API_KEY="sk-fake",
            FDA_BACKEND_TRACE="true",
            FDA_WORKER_LOG_DIR=str(self.logs / "workers"),
            FDA_GATEWAY_LOG=str(self.logs / "gateway.jsonl"),
        )
        self.env.start()

    async def asyncTearDown(self) -> None:
        self.env.stop()
        self.server.stop()

    async def session(self, name: str, order: str) -> list[dict[str, Any]]:
        link = InProcessBackendLink(gc.load_gateway_config(gc.CONFIG_DIR / "gateway.yaml"))
        self.assertEqual(link._config.workers.mode, "in_process")  # noqa: SLF001
        got: list[dict[str, Any]] = []

        def on(msg: dict[str, Any]) -> None:
            got.append(msg)
            if msg["type"] == "tool.call":
                args = json.loads(msg["arguments"])
                output = json.dumps({"order_id": args["order_id"], "status": "shipped"})
                link.send({"type": "tool.result", "call_id": msg["call_id"], "epoch": msg["epoch"], "output": output})

        await link.open(on)
        link.send({"type": "session.open", "session_id": name, "settings": {}})
        link.send({"type": "session.configure", "tools": [TOOL], "instructions": "Be brief."})
        link.send(
            {"type": "delegate", "turn_id": 1, "request": "task", "run_input": [{"seq": 1, "text": f"order {order}"}]}
        )
        deadline = time.monotonic() + 10
        while not any(m["type"] == "answer" for m in got) and time.monotonic() < deadline:
            await asyncio.sleep(0.02)
        await link.close()
        return got

    async def test_a_session_runs_a_tool_and_answers(self) -> None:
        self.server.queue(Reply(tool_calls=[tool("get_order", order_id="77")]), Reply(content="Order 77 shipped."))
        got = await self.session("llm1", "77")
        configured = next(m for m in got if m["type"] == "session.configured")
        self.assertTrue(configured["applied"])
        self.assertEqual(configured["backend_kind"], "llm")
        self.assertEqual(len(configured["backend_harness_sha256"]), 16)
        answer = next(m for m in got if m["type"] == "answer")
        self.assertEqual((answer["kind"], answer["text"]), ("hermes", "Order 77 shipped."))
        system = self.server.requests[0]["messages"][0]["content"]
        self.assertIn("## Mid-turn user steering", system)
        self.assertNotIn("Hermes runtime environment", system)
        self.assertIn("Be brief.", system)
        trace = (self.logs / "workers" / "llm1.trace.jsonl").read_text().splitlines()
        self.assertEqual(
            [json.loads(line)["kind"] for line in trace], ["session", "request", "response", "request", "response"]
        )
        events = [json.loads(line)["event"] for line in (self.logs / "gateway.jsonl").read_text().splitlines()]
        self.assertIn("backend_run_done", events)

    async def test_two_sessions_at_once(self) -> None:
        self.server.queue(
            Reply(tool_calls=[tool("get_order", order_id="1")], hold_s=0.2),
            Reply(tool_calls=[tool("get_order", order_id="2")], hold_s=0.2),
            Reply(content="first done"),
            Reply(content="second done"),
        )
        first, second = await asyncio.gather(self.session("a", "1"), self.session("b", "2"))
        answers = sorted(next(m for m in got if m["type"] == "answer")["text"] for got in (first, second))
        self.assertEqual(answers, ["first done", "second done"])


class VoiceProcessImportTest(unittest.TestCase):
    def test_an_llm_session_loads_no_hermes_module(self) -> None:
        code = (
            "import json, os, sys\n"
            "from examples.frontend_delegation_agent.core.sidecar import gateway_config as gc\n"
            "from examples.frontend_delegation_agent.core.sidecar.session_runtime import (\n"
            "    backend_templates, controller_settings)\n"
            "from examples.frontend_delegation_agent.core.worker.worker_main import make_host_factory\n"
            "config = gc.load_gateway_config(gc.CONFIG_DIR / 'gateway.yaml')\n"
            "controller_settings(config, backend_templates(config))\n"
            "make_host_factory(config.backend.kind, 's', in_process=True)\n"
            "print(json.dumps(sorted(sys.modules)))\n"
        )
        env = {"PYTHONPATH": str(REPO_ROOT / "src"), "FDA_BACKEND": "llm", "PATH": os.environ.get("PATH", "")}
        out = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, env=env, check=True, cwd=REPO_ROOT
        )  # noqa: S603
        loaded = json.loads(out.stdout.strip().splitlines()[-1])
        self.assertIn("examples.frontend_delegation_agent.core.backends.llm.host", loaded)
        for hermes in (
            "run_agent",
            "model_tools",
            "toolsets",
            "hermes_cli",
            "examples.frontend_delegation_agent.core.backends.hermes.host",
        ):
            self.assertNotIn(hermes, loaded)


if __name__ == "__main__":
    unittest.main()
