# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D100, D101, D102, D103, D107

"""``agent_turn_done`` carries the per-role (frontend/backend) usage and latency of each agent step."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

from _fbv_voice_fakes import (
    FakeChatClient,
    SessionHarness,
    delegate_response,
    tau2_session_update,
    text_response,
    tool_response,
)

from examples.frontend_backend_verdict.voice.agent.runner import AgentClients
from examples.frontend_backend_verdict.voice.agent.scripted import ScriptedAgentPort
from examples.frontend_backend_verdict.voice.agent.sinks import EventLog

ROLE_KEYS = {"calls", "prompt_tokens", "completion_tokens", "cached_tokens", "total_tokens", "latency_ms"}


def _turn_records(path: Path) -> list[dict[str, Any]]:
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    return [record for record in records if record["kind"] == "agent_turn_done"]


class UsageLogTests(unittest.IsolatedAsyncioTestCase):
    async def test_paired_steps_split_usage_by_role(self) -> None:
        frontend = FakeChatClient([delegate_response("Look up the users.")])
        backend = FakeChatClient([tool_response(("get_users", {}), ids=["call_1"]), text_response("Found them.", 7)])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "events.jsonl"
            harness = SessionHarness(
                clients=AgentClients(backend=backend, frontend=frontend), event_log=EventLog(str(path))
            )
            await harness.start(tau2_session_update())
            await harness.speak()
            await harness.wait_for("response.done")
            await harness.send(
                {
                    "type": "conversation.item.create",
                    "item": {"type": "function_call_output", "call_id": "call_1", "output": "{}"},
                }
            )
            await harness.send({"type": "response.create"})
            await harness.wait_for("response.done", 2)
            await harness.close()
            respond, resume = _turn_records(path)

        self.assertEqual((respond["step"], resume["step"]), ("respond", "resume"))
        self.assertEqual(respond["turn_id"], resume["turn_id"])
        self.assertEqual(set(respond["frontend"]), ROLE_KEYS)
        self.assertEqual(
            (respond["frontend"]["calls"], respond["frontend"]["total_tokens"]), (1, 20)
        )  # the call_backend decision
        self.assertEqual((respond["backend"]["calls"], respond["backend"]["total_tokens"]), (1, 20))
        self.assertEqual((resume["frontend"]["calls"], resume["frontend"]["total_tokens"]), (0, 0))
        self.assertEqual((resume["backend"]["calls"], resume["backend"]["total_tokens"]), (1, 14))
        for record in (respond, resume):
            roles = record["frontend"]["total_tokens"] + record["backend"]["total_tokens"]
            self.assertEqual(roles, record["input_tokens"] + record["output_tokens"])

    async def test_scripted_agent_logs_zeroed_roles(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "events.jsonl"
            harness = SessionHarness(
                agent_factory=lambda _session_id: ScriptedAgentPort([{"say": "Hello there."}]),
                event_log=EventLog(str(path)),
            )
            await harness.start(tau2_session_update())
            await harness.speak()
            await harness.wait_for("response.done")
            await harness.close()
            (record,) = _turn_records(path)

        self.assertEqual(record["step"], "respond")
        for role in ("frontend", "backend"):
            self.assertEqual(record[role], dict.fromkeys(ROLE_KEYS, 0) | {"latency_ms": 0.0})


if __name__ == "__main__":
    unittest.main()
