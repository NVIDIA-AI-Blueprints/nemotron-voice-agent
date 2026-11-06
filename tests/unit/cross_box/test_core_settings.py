# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102

"""The example's core settings in its registry entry load with the defaults the docs describe, on every adapter."""

import re
import unittest

from examples.frontend_backend_live.config_manager.schema import ExampleConfig
from examples.frontend_backend_live.delegation_core.backend.agent import AgentSettings
from examples.frontend_backend_live.delegation_core.backend.harness import HarnessSettings
from examples.frontend_backend_live.delegation_core.backend.llm import LlmSettings
from examples.frontend_backend_live.delegation_core.config import CoreSettings
from examples.frontend_backend_live.delegation_core.im.session import IMSession
from tests.unit.core_helpers import (
    CORE_BLOCK,
    CORE_TEXTS,
    EXAMPLE_SETTINGS,
    HARNESS_GROUP,
    NO_HARNESS_GROUP,
    core_block,
)

# The cores the example runs: the Live core on each backend adapter, and the UI bot's.
ADAPTERS = ("llm", "llm-agent", "harness")


def cores() -> dict[str, dict]:
    """The core settings of each place the example builds a core, by a name for the test's subtests."""
    shipped = {f"Live core on {name}": core_block(backend={"adapter": name}) for name in ADAPTERS}
    return {**shipped, "UI bot core": ExampleConfig.model_validate(EXAMPLE_SETTINGS).ui_bot_core}


class CoreSettingsTests(unittest.TestCase):
    def setUp(self):
        self.settings = CoreSettings.parse(CORE_BLOCK, CORE_TEXTS)

    def test_the_settings_load_with_the_documented_defaults(self):
        delegation, backend = self.settings.delegation, self.settings.backend
        self.assertEqual((backend.adapter, backend.one_shot_adapter), ("llm", "llm"))
        self.assertEqual((backend.slot, backend.prompt), ("thinker-llm", "generic"))
        self.assertEqual(sorted(backend.blocks), ["harness"])  # the other adapters' blocks keep their defaults
        self.assertEqual(delegation.result_format, "plain")
        # The backend LLM and the agent's workers get thirty rounds before the capped one.
        self.assertEqual(LlmSettings.model_validate(backend.blocks.get("llm", {})).max_tool_rounds, 30)
        self.assertEqual(AgentSettings.model_validate(backend.blocks.get("llm-agent", {})).worker.max_tool_rounds, 30)

    def test_the_settings_word_no_text_and_a_step_they_leave_out_takes_none(self):
        with self.assertRaisesRegex(ValueError, "texts\n.*Field required"):
            CoreSettings.parse(CORE_BLOCK, {})  # every text comes from the prompt file
        # The shape hints are off by default: no step, and the hint's text stays in the prompt file unused.
        self.assertEqual(dict(self.settings.backend.app_output_steps), {})
        steps = {"shape_hints": {"not_found_pattern": r"(?i)\b(not found|does not exist|no such)\b"}}
        on = CoreSettings.parse(core_block(backend={"app_output_steps": steps}), CORE_TEXTS).backend.app_output_steps
        self.assertEqual(list(on), ["shape_hints"])  # turning the step on gives it its hint's text
        self.assertEqual(on["shape_hints"]["hint"], CORE_TEXTS["backend.app_output_steps.shape_hints.hint"])

    def test_settings_that_set_a_place_of_the_cores_texts_name_where_they_are(self):
        # The core's texts are prompts.yaml's: settings that set one of their places stop loading, and no core the
        # example builds sets one, so each loads with the texts at their places.
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
        for place, setting in places.items():
            with self.subTest(place), self.assertRaisesRegex(ValueError, rf"\b{re.escape(place)}\b.* {where}"):
                CoreSettings.parse(core_block(**setting), CORE_TEXTS)
        for name, core in cores().items():
            with self.subTest(core=name):
                block = CoreSettings.parse(core, CORE_TEXTS).backend.blocks["harness"]
                self.assertEqual(block["prompts"]["user"], CORE_TEXTS["backend.harness.prompts.user"])

    def test_texts_that_hold_a_text_where_the_settings_have_a_block_name_where_they_are(self):
        where = r"comes from agent_prompts\.Core in the example's prompts\.yaml"
        others = {name: text for name, text in CORE_TEXTS.items() if not name.startswith("backend.")}
        steps = {"shape_hints": {"not_found_pattern": "no such"}}
        cases = {
            "backend": ({}, {**others, "backend": "A text."}),
            "backend.app_output_steps": (
                {"app_output_steps": steps},
                {**others, "backend.app_output_steps": "A text."},
            ),
        }
        for place, (backend, texts) in cases.items():
            with self.subTest(place), self.assertRaisesRegex(ValueError, rf"^{re.escape(place)} {where}"):
                CoreSettings.parse(core_block(backend=backend), texts)

    def test_every_core_offers_cancel_work(self):
        # The steer rule never stops work: a caller's stop lands in the running task, which ends its own Work with the
        # cancel_work work tool, so the IM gives it to every task of a conversation and no core's settings name it.
        self.assertEqual(IMSession.work_tools, ("cancel_work",))
        for name, core in cores().items():
            leave_out = {**core, "delegation": {**core.get("delegation", {}), "work_tools": []}}
            with self.subTest(core=name), self.assertRaisesRegex(ValueError, "delegation.work_tools\n.*Extra inputs"):
                CoreSettings.parse(leave_out, CORE_TEXTS)

    @unittest.skipUnless(HARNESS_GROUP, NO_HARNESS_GROUP)
    def test_settings_that_run_one_task_at_a_time(self):
        # An eval task is one call: the agent's and the harness's tasks run one at a time, placed by the steer rule.
        shipped = self.settings.backend.blocks
        one = {"llm-agent": {"max_running": 1}, "harness": {"max_running": 1}}
        bench = CoreSettings.parse(core_block(backend=one), CORE_TEXTS).backend.blocks
        self.assertEqual(dict(bench["llm-agent"]), {"max_running": 1})
        self.assertEqual(dict(bench["harness"]), {**shipped["harness"], "max_running": 1})
        self.assertEqual(AgentSettings.model_validate(bench["llm-agent"]).max_running, 1)
        self.assertEqual(HarnessSettings.model_validate(bench["harness"]).describe()["max_running"], 1)


if __name__ == "__main__":
    unittest.main()
