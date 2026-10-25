# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: D100, D101, D102, D103, D107

"""Two REAL Hermes workers (Python 3.14 venv) behind the gateway, against a fake OpenAI server.

This is the production topology: the gateway runs in this repo's environment and spawns
one Hermes worker process per session with the Hermes interpreter. Skipped unless that
interpreter exists and can ``import run_agent`` (default ``~/.cache/fda/hermes-venv-314``;
override with ``FDA_HERMES_PYTHON``). Runs with the normal suite:

    uv run pytest tests/unit/frontend_delegation_agent/core/test_fda_hermes_multisession.py -v
"""

from __future__ import annotations

import asyncio
import functools
import os
import shutil
import signal
import subprocess
import tempfile
import unittest
from pathlib import Path

from _fda_backend_fakes import TOOL
from _fda_fake_openai import FakeOpenAI
from test_fda_gateway_processes import Session, alive, process_config

HERMES_PYTHON = os.path.expanduser(os.environ.get("FDA_HERMES_PYTHON", "~/.cache/fda/hermes-venv-314/bin/python"))


@functools.cache
def hermes_available() -> bool:
    if not Path(HERMES_PYTHON).exists():
        return False
    try:
        result = subprocess.run(  # noqa: S603 - fixed interpreter path and arguments
            [HERMES_PYTHON, "-c", "import run_agent"],
            capture_output=True,
            timeout=60,
            cwd="/",
            env={"PATH": os.environ.get("PATH", "")},
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0


@unittest.skipUnless(hermes_available(), f"Hermes interpreter not usable: {HERMES_PYTHON}")
class HermesMultiSessionTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.root = Path(tempfile.mkdtemp(prefix="fda-hms-"))
        self.server = FakeOpenAI().start()
        os.environ["FDA_FAKE_KEY"] = "sk-fake"
        self.config = process_config(
            self.root,
            workers={
                "python": HERMES_PYTHON,
                "start_timeout_s": 30.0,
                "configure_timeout_s": 60.0,
                "stop_timeout_s": 12.0,
                "kill_grace_s": 2.0,
            },
            backend={
                "kind": "hermes",
                "model": "fake-model",
                "base_url": self.server.base_url,
                "api_key_env": "FDA_FAKE_KEY",
                "request_overrides": {},
                "unwind_timeout_s": 5.0,
                "run_budget_seconds": 60.0,
                "run_hard_deadline_s": 90.0,
            },
            hermes={"disable_streaming": True},
        )
        self.sessions: list[Session] = []

    async def asyncTearDown(self) -> None:
        for session in self.sessions:
            await session.runtime.close()
        self.server.stop()
        shutil.rmtree(self.root, ignore_errors=True)

    async def test_overlapping_tools_isolated_and_a_killed_worker_leaves_the_other_intact(self) -> None:
        a = Session(self.config, "ha", tool_output='{"a": "A-result"}')
        b = Session(self.config, "hb", tool_output='{"b": "B-result"}')
        self.sessions += [a, b]
        tool_b = {**TOOL, "parameters": {"type": "object", "properties": {"sku": {"type": "string"}}}}
        await asyncio.gather(a.open([TOOL]), b.open([tool_b]))
        await a.wait("session.configured", timeout=60)
        await b.wait("session.configured", timeout=60)
        await asyncio.gather(a.delegate(1, "look it up"), b.delegate(1, "look it up"))
        (answer_a,), (answer_b,) = await asyncio.gather(a.wait("answer", timeout=60), b.wait("answer", timeout=60))
        self.assertIn("A-result", answer_a["text"])
        self.assertIn("schema: ['order_id']", answer_a["text"])
        self.assertIn("B-result", answer_b["text"])
        self.assertIn("schema: ['sku']", answer_b["text"])
        for request in self.server.requests:
            self.assertEqual({t["function"]["name"] for t in request.get("tools") or []}, {"get_order"})

        # Kill A's worker while its tool call is outstanding; B keeps working.
        a.tool_output = None  # stop answering A's tool calls
        a._on = lambda msg: a.messages.append(msg)  # type: ignore[method-assign]  # noqa: SLF001
        a.runtime._send = a._on  # noqa: SLF001
        await a.delegate(2, "look it up again")
        await a.wait("tool.call", count=2, timeout=60)
        a_pid = a.pid()
        os.kill(a_pid, signal.SIGKILL)
        await b.delegate(2, "look it up again")
        answers_b = await b.wait("answer", count=2, timeout=60)
        self.assertIn("B-result", answers_b[1]["text"])
        done = await a.wait("backend_run_done", count=2, timeout=30)
        self.assertEqual(done[1]["status"], "worker_died")
        self.assertFalse(alive(a_pid))
        self.assertTrue(alive(b.pid()))


if __name__ == "__main__":
    unittest.main()
