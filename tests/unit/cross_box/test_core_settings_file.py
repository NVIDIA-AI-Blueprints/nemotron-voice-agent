# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102

"""The example's core settings files (``core.yaml`` and its profiles) load with the defaults the docs describe."""

import unittest
from pathlib import Path

from delegation_core.backend.agent import AgentSettings
from delegation_core.backend.harness import HarnessSettings
from delegation_core.backend.llm import LlmSettings
from delegation_core.config import CoreSettings

CORE_YAML = Path(__file__).resolve().parents[3] / "src/examples/frontend_backend_live/core.yaml"


class CoreSettingsFileTests(unittest.TestCase):
    def setUp(self):
        self.settings = CoreSettings.load(CORE_YAML)

    def test_the_file_loads_with_the_documented_defaults(self):
        delegation, backend = self.settings.delegation, self.settings.backend
        self.assertEqual((backend.adapter, backend.one_shot_adapter), ("llm", "llm"))
        self.assertEqual((backend.slot, backend.prompt), ("thinker-llm", "backend"))
        self.assertEqual(sorted(backend.blocks), ["harness", "llm", "llm-agent"])
        self.assertEqual(backend.blocks["llm"]["conductor"], "steer")
        self.assertEqual(
            (delegation.result_format, delegation.voice_note, delegation.work_tools),
            ("plain", "default", ("cancel_work",)),
        )
        # The backend LLM and the agent's workers get thirty rounds before the capped one.
        self.assertEqual(LlmSettings.parse(backend.blocks["llm"]).max_tool_rounds, 30)
        self.assertEqual(AgentSettings.parse(backend.blocks["llm-agent"]).worker.max_tool_rounds, 30)

    def test_every_profile_offers_cancel_work(self):
        # The steer rule never stops work: a caller's stop lands in the running task, which ends its own Work with the
        # cancel_work work tool, so no profile may leave that tool out.
        for path in sorted(CORE_YAML.parent.glob("core*.yaml")):
            with self.subTest(profile=path.name):
                self.assertIn("cancel_work", CoreSettings.load(path).delegation.work_tools)

    def test_the_benchmark_profile_runs_one_task_and_one_harness_session_at_a_time(self):
        # An eval task is one call: the agent's and the harness's tasks run one at a time, placed by the steer rule, and
        # the harness hands each answered task's session to the next task.
        shipped = self.settings.backend.blocks
        bench = CoreSettings.load(CORE_YAML.with_name("core.bench.yaml")).backend.blocks
        self.assertEqual(dict(bench["llm-agent"]), {**shipped["llm-agent"], "max_running": 1})
        self.assertEqual(dict(bench["harness"]), {**shipped["harness"], "max_running": 1, "session": "call"})
        self.assertEqual(AgentSettings.parse(bench["llm-agent"]).max_running, 1)
        harness = HarnessSettings.parse(bench["harness"]).describe()
        self.assertEqual((harness["max_running"], harness["session"]), (1, "call"))
