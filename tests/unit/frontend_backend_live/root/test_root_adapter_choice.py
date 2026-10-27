# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D103, D107
"""The frontend adapter chosen by config, and readiness."""

import asyncio
import os
import sys
import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from delegation_core.config import CoreSettings
from examples.frontend_backend_live import factory, pipeline
from examples.frontend_backend_live.config_manager.loader import CONFIG_PATH, load_live_config
from examples.frontend_backend_live.config_manager.schema import ConfigError
from live.protocol import ProtocolError, SessionConfig
from tests.unit.contracts.base import ContractCase
from tests.unit.frontend_backend_live.root.root_helpers import (
    CATALOG,
    NoRtviAdapter,
    StubAdapter,
    StubReadiness,
    StubSettings,
    clear_root_caches,
    reset_stubs,
    stub_root,
)

SESSION = SessionConfig.model_validate({"model": "live-1"})


class SecondAdapter(StubAdapter):
    """Another frontend adapter class: a second entry of the table."""


class AdapterChoiceTests(unittest.TestCase):
    def setUp(self):
        clear_root_caches(self)
        reset_stubs()
        self.enterContext(patch.object(factory.ExampleCatalog, "read", return_value=CATALOG))
        self.enterContext(patch.dict(os.environ))
        for name in ("CORE_CONFIG_PATH", "LIVE_CONFIG_PATH"):
            os.environ.pop(name, None)

    def test_the_named_class_parses_its_own_block_with_the_cores_display_label(self):
        with patch.object(factory, "frontend_adapters", return_value={"cascade": StubAdapter}):
            state = factory.root()
        block = dict(load_live_config(CONFIG_PATH).blocks["cascade"])
        self.assertIs(state.adapter, StubAdapter)
        self.assertEqual(StubSettings.parsed, [{**block, "display_label": state.core.delegation.display_label}])
        self.assertIsInstance(state.readiness, StubReadiness)

    def test_an_unknown_frontend_adapter_is_refused_with_the_known_names(self):
        with (
            patch.object(factory, "frontend_adapters", return_value={"other": StubAdapter}),
            self.assertRaisesRegex(ValueError, "Unknown frontend adapter 'cascade'; known: other"),
        ):
            factory.root()

    def test_a_second_adapter_is_one_class_and_one_table_entry(self):
        with TemporaryDirectory() as folder:
            profile = Path(folder) / "config.second.yaml"
            profile.write_text(
                f"extends: {CONFIG_PATH}\nfrontend:\n  adapter: second\n  second: {{voice: calm}}\n", encoding="utf-8"
            )
            table = {"cascade": StubAdapter, "second": SecondAdapter}
            with (
                patch.dict(os.environ, {"LIVE_CONFIG_PATH": str(profile)}),
                patch.object(factory, "frontend_adapters", return_value=table),
            ):
                state = factory.root()
        self.assertIs(state.adapter, SecondAdapter)
        self.assertEqual(state.settings.block, {"voice": "calm", "display_label": state.core.delegation.display_label})


class SessionFactoryTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        reset_stubs()

    async def test_each_session_is_built_by_the_class_from_its_settings_and_the_catalog(self):
        state = stub_root()
        with patch.object(factory, "root", return_value=state):
            parts = await factory.create_frontend_adapter(SESSION, "webrtc", "peer")
        self.assertEqual(parts.frontend.args, (SESSION, "webrtc", "peer", state.settings, CATALOG))
        self.assertEqual(parts, factory.SessionParts(frontend=parts.frontend))  # the frontend only

    async def test_readiness_is_asked_for_every_session(self):
        state = stub_root(readiness=StubReadiness())
        with patch.object(factory, "root", return_value=state):
            for _ in range(2):
                await factory.create_frontend_adapter(SESSION, "websocket", None)
        self.assertEqual(state.readiness.checks, 2)

    async def test_a_service_still_starting_refuses_the_session_as_the_servers_fault(self):
        state = stub_root(readiness=StubReadiness(down=("asr",)))
        with patch.object(factory, "root", return_value=state), self.assertRaises(ProtocolError) as refused:
            await factory.create_frontend_adapter(SESSION, "websocket", None)
        self.assertEqual(refused.exception.code, "server_error")
        self.assertIn("asr is still starting", str(refused.exception))
        self.assertEqual(StubAdapter.built, [])

    async def test_a_slow_read_of_the_roots_files_stays_off_the_event_loop(self):
        state, ticks = stub_root(), []

        def slow_root():
            time.sleep(0.2)
            return state

        async def tick():
            while True:
                ticks.append(1)
                await asyncio.sleep(0.01)

        ticker = asyncio.create_task(tick())
        try:
            with patch.object(factory, "root", slow_root):
                await factory.create_frontend_adapter(SESSION, "websocket", None)
        finally:
            ticker.cancel()
        self.assertGreater(len(ticks), 5)  # the loop kept running while the files were read


class TracingTests(unittest.TestCase):
    def test_the_live_root_loads_the_tracing_setup_as_the_ui_bots_root_does(self):
        # The adapter traces each session's pipeline when ENABLE_TRACING is on, but may not import the module that
        # starts OpenTelemetry; the roots import it.
        self.assertIs(vars(factory).get("tracing"), sys.modules["tracing"])


class UiBotHostTests(unittest.TestCase):
    def setUp(self):
        pipeline.bot_root.cache_clear()
        self.addCleanup(pipeline.bot_root.cache_clear)
        self.enterContext(patch.object(pipeline.ExampleCatalog, "read", return_value=CATALOG))

    def test_the_ui_bot_refuses_an_adapter_it_cannot_host_and_names_those_it_can(self):
        table = {"cascade": NoRtviAdapter, "stub": StubAdapter}
        with (
            patch.object(pipeline, "frontend_adapters", return_value=table),
            self.assertRaisesRegex(ConfigError, "'cascade' cannot run in the UI bot.*it can host: stub"),
        ):
            pipeline.bot_root()

    def test_the_ui_bot_hosts_an_adapter_that_can(self):
        with patch.object(pipeline, "frontend_adapters", return_value={"cascade": StubAdapter}):
            state = pipeline.bot_root()
        self.assertEqual((state.cascade.provider, state.cascade.slot), ("chat-completions", "llm"))
        self.assertEqual(state.cascade.display_label, state.core.delegation.display_label)


class ServerReadinessTests(unittest.IsolatedAsyncioTestCase):
    """The UI's readiness checks every LLM slot the example uses, each at the base URL the session will use."""

    EXAMPLE = {"services": {"llm": ["a"], "thinker-llm": ["a"], "asr": ["b"]}}

    def setUp(self):
        import server

        self.server, self.checked = server, []

        async def check(label, target, health_url, expects_ready_json=False):
            self.checked.append((label, target, health_url))

        defaults = {"llm": {"base_url": "http://talker:8000/v1"}, "thinker-llm": {"base_url": "http://thinker:8000/v1"}}
        self.enterContext(patch.object(server, "_run_http_readiness_check", check))
        self.enterContext(patch.object(server, "_llm_health_url", side_effect=lambda url: f"{url}/health"))
        self.enterContext(patch.object(server, "load_service_entry", side_effect=lambda slot, key: defaults[slot]))

    async def test_both_llm_slots_are_checked_at_the_sessions_choice(self):
        config = {"base_url": "http://picked:8000/v1", "thinker_base_url": "http://picked-thinker:8000/v1"}
        await self.server._ensure_llm_ready_for_connection(config, self.EXAMPLE)
        self.assertEqual(
            self.checked,
            [
                ("LLM", "http://picked:8000/v1", "http://picked:8000/v1/health"),
                ("Thinker LLM", "http://picked-thinker:8000/v1", "http://picked-thinker:8000/v1/health"),
            ],
        )

    async def test_a_slot_the_session_leaves_unset_is_checked_at_its_default_entry(self):
        await self.server._ensure_llm_ready_for_connection({}, self.EXAMPLE)
        self.assertEqual([target for _, target, _ in self.checked], ["http://talker:8000/v1", "http://thinker:8000/v1"])

    async def test_an_example_with_one_llm_slot_checks_that_slot_only(self):
        await self.server._ensure_llm_ready_for_connection({}, {"services": {"llm": ["a"], "tts": ["c"]}})
        self.assertEqual([label for label, _, _ in self.checked], ["LLM"])

    async def test_an_endpoint_with_no_health_path_is_not_checked(self):
        with patch.object(self.server, "_llm_health_url", return_value=""):
            await self.server._ensure_llm_ready_for_connection({}, self.EXAMPLE)
        self.assertEqual(self.checked, [])


class HealthUrlTests(unittest.TestCase):
    def test_an_entry_only_the_thinker_slot_offers_has_its_health_url(self):
        import server

        entry = {"base_url": "http://thinker-only:8000/v1", "health_path": "/v1/health/ready"}
        sources = [(server.SELF_HOSTED_SOURCE, {"llm": {}, "thinker-llm": {"t": entry}, "asr": {"a": {"server": "x"}}})]
        with patch.object(server, "service_catalog_sources", return_value=sources):
            url = server._llm_health_url("http://thinker-only:8000/v1")
        self.assertEqual(url, "http://thinker-only:8000/v1/health/ready")


class ShippedAdapterTests(ContractCase, unittest.TestCase):
    def test_the_shipped_file_names_a_registered_class_that_parses_its_block(self):
        from examples.frontend_backend_live.adapter.registry import FRONTEND_ADAPTERS

        config = load_live_config(CONFIG_PATH)
        core = CoreSettings.load(factory.core_config_path())
        adapter = FRONTEND_ADAPTERS[config.adapter]
        adapter.Settings.parse(factory.with_display_label(factory.adapter_block(config), core))
        self.assertTrue(adapter.hosts_rtvi)


if __name__ == "__main__":
    unittest.main()
