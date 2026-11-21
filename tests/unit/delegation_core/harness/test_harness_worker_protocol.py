# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102

"""Native worker instruction composition survives catalog overrides, session reuse, and recovery."""

import os
import unittest
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import yaml

from examples.frontend_backend_live import factory
from examples.frontend_backend_live.config_manager.schema import merged
from examples.frontend_backend_live.delegation_core.backend.harness.acp.hermes.agent import HermesAgent
from examples.frontend_backend_live.delegation_core.backend.harness.settings import HarnessSettings
from examples.frontend_backend_live.delegation_core.backend.harness.worker import WorkerRun
from tests.unit.core_helpers import PROMPTS
from tests.unit.delegation_core.harness.doubles import FakeHost, settings
from tests.unit.delegation_core.harness.test_harness_worker import close_events

PROTOCOL_KEY = "backend.harness.prompts.worker_protocol"
NOTE = "Use the current runtime-owned refusal record to handle pending input."


class WorkerProtocolCatalogTests(unittest.TestCase):
    def test_whole_catalog_override_reaches_the_live_and_ui_factory_settings(self):
        registered = factory.example_config()
        selected = merged(
            registered.core,
            {
                "backend": {"adapter": "harness", "harness": {"protocol": "acp", "agent": "hermes"}},
                "delegation": {"result_format": "speak_display"},
            },
        )
        expected_config = deepcopy(selected)
        config = registered.model_copy(update={"core": selected})
        guard = settings().agent.command_guard
        for catalog_kind in ("legacy", "shipped", "custom"):
            with self.subTest(catalog_kind):
                catalog = deepcopy(PROMPTS)
                core = catalog["agent_prompts"]["Core"]
                shipped = core[PROTOCOL_KEY]["content"]
                if catalog_kind == "legacy":
                    del core[PROTOCOL_KEY]
                    expected = ""
                elif catalog_kind == "custom":
                    core[PROTOCOL_KEY]["content"] = NOTE
                    expected = NOTE
                else:
                    expected = shipped
                with TemporaryDirectory() as folder:
                    path = Path(folder) / "custom-prompts.yaml"
                    path.write_text(yaml.safe_dump(catalog, sort_keys=False), encoding="utf-8")
                    with patch.dict(os.environ, {"PROMPT_FILE_PATH": str(path)}):
                        loaded = (factory.core_settings(config.core), factory.ui_bot_core(config))
                for parsed in loaded:
                    harness = HarnessSettings.model_validate(parsed.backend.blocks["harness"])
                    self.assertEqual(harness.prompts.worker_protocol, expected)
                    self.assertIs(type(harness.agent), HermesAgent)
                    self.assertEqual(harness.agent.command_guard, guard)
                self.assertEqual(config.core, expected_config)


class WorkerProtocolSessionTests(unittest.IsolatedAsyncioTestCase):
    def configure(self, harness, note):
        prompts = harness.prompts.model_copy(update={"worker_protocol": note})
        self.host = FakeHost(harness.model_copy(update={"prompts": prompts}))
        self.client = self.host.runtime.client

    async def start(self, task):
        """Open or reuse a native session through the actual WorkerRun event stream."""
        run = WorkerRun(task, self.host)
        events = run.events()
        self.addAsyncCleanup(close_events, events)
        self.assertEqual((await anext(events))["type"], "response.created")
        return run, events
