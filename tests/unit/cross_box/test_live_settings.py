# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102

"""The example's shipped settings: the core's block in its registry entry, and the Compose services' environments."""

import re
import unittest
from pathlib import Path

import yaml

from examples.frontend_backend_live.config_manager.schema import ExampleConfig, LiveConfig
from examples.frontend_backend_live.delegation_core.backend.agent import AgentSettings
from examples.frontend_backend_live.delegation_core.backend.llm import LlmSettings
from examples.frontend_backend_live.delegation_core.config import CoreSettings
from examples.frontend_backend_live.delegations import ReadyResult
from examples.frontend_backend_live.factory import adapter_block
from tests.unit.core_helpers import CORE_BLOCK, CORE_TEXTS, EXAMPLE_SETTINGS, core_block
from tests.unit.frontend_helpers import CallHarness

REPO = Path(__file__).resolve().parents[3]
WHERE = r"comes from agent_prompts\.Core in the example's prompts\.yaml"


class CoreSettingsTests(unittest.TestCase):
    def setUp(self):
        self.settings = CoreSettings.parse(CORE_BLOCK, CORE_TEXTS)

    def test_the_settings_load_with_the_documented_defaults(self):
        delegation, backend = self.settings.delegation, self.settings.backend
        self.assertEqual((backend.adapter, backend.one_shot_adapter), ("llm", "llm"))
        self.assertEqual((backend.slot, backend.prompt), ("thinker-llm", "generic"))
        self.assertEqual(sorted(backend.blocks), ["harness"])
        self.assertEqual((delegation.result_format, delegation.client_tool_timeout_seconds), ("plain", 60.0))
        self.assertEqual(LlmSettings.model_validate(backend.blocks.get("llm", {})).max_tool_rounds, 30)
        self.assertEqual(AgentSettings.model_validate(backend.blocks.get("llm-agent", {})).worker.max_tool_rounds, 30)

    def test_no_text_is_in_the_settings_and_no_step_is_on(self):
        with self.assertRaisesRegex(ValueError, "texts\n.*Field required"):
            CoreSettings.parse(CORE_BLOCK, {})
        self.assertEqual(dict(self.settings.backend.app_output_steps), {})
        steps = {"shape_hints": {"not_found_pattern": r"(?i)\b(not found|does not exist|no such)\b"}}
        on = CoreSettings.parse(core_block(backend={"app_output_steps": steps}), CORE_TEXTS).backend.app_output_steps
        self.assertEqual(list(on), ["shape_hints"])
        self.assertEqual(on["shape_hints"]["hint"], CORE_TEXTS["backend.app_output_steps.shape_hints.hint"])

    def test_a_setting_at_a_texts_place_names_where_the_text_is(self):
        places = {
            "backend.harness.prompts.user": {"backend": {"harness": {"prompts": {"user": "{text}"}}}},
            "backend.app_output_steps.shape_hints.hint": {
                "backend": {"app_output_steps": {"shape_hints": {"hint": "Try {value}."}}}
            },
            "texts": {"texts": {"none": "nothing"}},
        }
        for place, setting in places.items():
            with self.subTest(place), self.assertRaisesRegex(ValueError, rf"\b{re.escape(place)}\b.* {WHERE}"):
                CoreSettings.parse(core_block(**setting), CORE_TEXTS)
        cores = {name: core_block(backend={"adapter": name}) for name in ("llm", "llm-agent", "harness")}
        cores["UI bot"] = ExampleConfig.model_validate(EXAMPLE_SETTINGS).ui_bot_core
        for name, core in cores.items():
            with self.subTest(core=name):
                block = CoreSettings.parse(core, CORE_TEXTS).backend.blocks["harness"]
                self.assertEqual(block["prompts"]["user"], CORE_TEXTS["backend.harness.prompts.user"])

    def test_a_text_at_a_blocks_place_names_where_the_texts_are(self):
        others = {name: text for name, text in CORE_TEXTS.items() if not name.startswith("backend.")}
        steps = {"app_output_steps": {"shape_hints": {"not_found_pattern": "no such"}}}
        cases = {
            "backend": ({}, {**others, "backend": "A text."}),
            "backend.app_output_steps": (steps, {**others, "backend.app_output_steps": "A text."}),
        }
        for place, (backend, texts) in cases.items():
            with self.subTest(place), self.assertRaisesRegex(ValueError, rf"^{re.escape(place)} {WHERE}"):
                CoreSettings.parse(core_block(backend=backend), texts)


class ComposeTests(unittest.TestCase):
    def test_no_service_sets_the_live_settings(self):
        # The Live routes are open to anyone who can reach the server unless LIVE_API_TOKEN is set, so no recipe serves
        # them by default: a deployment sets the token, or LIVE_ENABLED=true, in its .env.
        compose = yaml.safe_load((REPO / "docker-compose.yml").read_text(encoding="utf-8"))
        services = dict(compose["services"])
        for include in compose.get("include", []):
            services.update(yaml.safe_load((REPO / include["path"]).read_text(encoding="utf-8")).get("services") or {})
        for name, service in services.items():
            # A mapping's keys, or a list's NAME=value and NAME items.
            names = {str(item).split("=", 1)[0] for item in service.get("environment") or ()}
            with self.subTest(service=name):
                self.assertEqual(names & {"LIVE_ENABLED", "LIVE_API_TOKEN"}, set())


class AnswersSettingsTests(unittest.IsolatedAsyncioTestCase):
    async def test_the_answers_keys_reach_the_voice_through_the_composition_root(self):
        # frontend.cascade.delegation.answers in the example's settings, read as the root reads the cascade's block.
        cascade = {**EXAMPLE_SETTINGS["frontend"]["cascade"], "delegation": {"answers": {"max_tries": 1}}}
        settings = {**EXAMPLE_SETTINGS, "frontend": {**EXAMPLE_SETTINGS["frontend"], "cascade": cascade}}
        live = LiveConfig.model_validate(adapter_block(ExampleConfig.model_validate(settings)))
        voice = CallHarness(config=live).voice
        voice._queue = lambda frame: None
        voice._word_early = lambda: None
        voice.worknote.confirm(voice.worknote.reserve("Hours?"), "del_1")
        voice.worknote.answered("del_1")
        voice.ready(ReadyResult("del_1", "item_1", "The gym opens at six."))
        batch = voice._take()
        voice.not_heard(voice.delivered(batch, "We open at six."))  # the first miss drops it
        self.assertEqual((list(voice.owed), voice.worknote.answer_ready()), ([], False))
