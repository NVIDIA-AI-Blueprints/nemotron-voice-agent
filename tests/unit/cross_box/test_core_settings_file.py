# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102

"""The example's core settings files (``core.yaml`` and its profiles) load with the defaults the docs describe."""

import re
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import yaml

from examples.frontend_backend_live.delegation_core.backend.agent import AgentSettings
from examples.frontend_backend_live.delegation_core.backend.harness import HarnessSettings
from examples.frontend_backend_live.delegation_core.backend.llm import LlmSettings
from examples.frontend_backend_live.delegation_core.config import CoreSettings
from tests.unit.core_helpers import CORE_TEXTS

CORE_YAML = Path(__file__).resolve().parents[3] / "src/examples/frontend_backend_live/core.yaml"


class CoreSettingsFileTests(unittest.TestCase):
    def setUp(self):
        self.settings = CoreSettings.load(CORE_YAML, CORE_TEXTS)

    def test_the_file_loads_with_the_documented_defaults(self):
        delegation, backend = self.settings.delegation, self.settings.backend
        self.assertEqual((backend.adapter, backend.one_shot_adapter), ("llm", "llm"))
        self.assertEqual((backend.slot, backend.prompt), ("thinker-llm", "generic"))
        self.assertEqual(sorted(backend.blocks), ["harness", "llm", "llm-agent"])
        self.assertEqual(backend.blocks["llm"]["conductor"], "steer")
        self.assertEqual(
            (delegation.result_format, delegation.caller_note, delegation.voice_note, delegation.work_tools),
            ("plain", "default", "default", ("cancel_work",)),
        )
        # The backend LLM and the agent's workers get thirty rounds before the capped one.
        self.assertEqual(LlmSettings.parse(backend.blocks["llm"]).max_tool_rounds, 30)
        self.assertEqual(AgentSettings.parse(backend.blocks["llm-agent"]).worker.max_tool_rounds, 30)

    def test_the_file_words_no_text_and_a_step_it_leaves_out_takes_none(self):
        with self.assertRaisesRegex(ValueError, "the core's texts is missing texts"):
            CoreSettings.load(CORE_YAML, {})  # every text comes from the prompt file
        self.assertEqual(list(self.settings.backend.app_output_steps), ["shape_hints"])
        shipped = yaml.safe_load(CORE_YAML.read_text(encoding="utf-8"))
        shipped["backend"]["app_output_steps"] = {}  # the step removed: its hint's text stays in the prompt file
        with TemporaryDirectory() as folder:
            bare = Path(folder) / "core.custom.yaml"
            bare.write_text(yaml.safe_dump(shipped), encoding="utf-8")
            self.assertEqual(dict(CoreSettings.load(bare, CORE_TEXTS).backend.app_output_steps), {})

    def test_a_profile_that_sets_a_place_of_the_cores_texts_names_where_they_are(self):
        # The core's texts are prompts.yaml's: a profile that sets one of their places stops loading, and no shipped
        # profile sets one, so each loads with the texts at their places.
        where = r"comes from agent_prompts\.Core in the example's prompts\.yaml"
        places = {
            "backend.harness.prompts.user": {"backend": {"harness": {"prompts": {"user": "{text}"}}}},
            "backend.harness.agents.hermes.command_guard": {
                "backend": {"harness": {"agents": {"hermes": {"command_guard": "Asked: "}}}}
            },
            "backend.app_output_steps.shape_hints.hint": {
                "backend": {"app_output_steps": {"shape_hints": {"hint": "Try {value}."}}}
            },
            "texts": {"texts": {"none": "nothing"}},
        }
        with TemporaryDirectory() as folder:
            profile = Path(folder) / "core.custom.yaml"
            for place, setting in places.items():
                profile.write_text(yaml.safe_dump({"extends": str(CORE_YAML), **setting}), encoding="utf-8")
                with self.subTest(place), self.assertRaisesRegex(ValueError, rf"\b{re.escape(place)}\b.* {where}"):
                    CoreSettings.load(profile, CORE_TEXTS)
        for path in sorted(CORE_YAML.parent.glob("core*.yaml")):
            with self.subTest(profile=path.name):
                block = CoreSettings.load(path, CORE_TEXTS).backend.blocks["harness"]
                self.assertEqual(block["prompts"]["user"], CORE_TEXTS["backend.harness.prompts.user"])

    def test_every_profile_offers_cancel_work(self):
        # The steer rule never stops work: a caller's stop lands in the running task, which ends its own Work with the
        # cancel_work work tool, so no profile may leave that tool out.
        for path in sorted(CORE_YAML.parent.glob("core*.yaml")):
            with self.subTest(profile=path.name):
                self.assertIn("cancel_work", CoreSettings.load(path, CORE_TEXTS).delegation.work_tools)

    def test_a_profile_runs_one_task_and_one_harness_session_at_a_time(self):
        # An eval task is one call: the agent's and the harness's tasks run one at a time, placed by the steer rule, and
        # the harness hands each answered task's session to the next task.
        shipped = self.settings.backend.blocks
        blocks = "{llm-agent: {max_running: 1}, harness: {max_running: 1, session: call}}"
        with TemporaryDirectory() as folder:
            profile = Path(folder) / "core.custom.yaml"
            profile.write_text(f"extends: {CORE_YAML}\nbackend: {blocks}\n", encoding="utf-8")
            bench = CoreSettings.load(profile, CORE_TEXTS).backend.blocks
        self.assertEqual(dict(bench["llm-agent"]), {**shipped["llm-agent"], "max_running": 1})
        self.assertEqual(dict(bench["harness"]), {**shipped["harness"], "max_running": 1, "session": "call"})
        self.assertEqual(AgentSettings.parse(bench["llm-agent"]).max_running, 1)
        harness = HarnessSettings.parse(bench["harness"]).describe()
        self.assertEqual((harness["max_running"], harness["session"]), (1, "call"))
