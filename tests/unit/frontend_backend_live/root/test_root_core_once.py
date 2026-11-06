# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D103, D107
"""The root reads the example's settings once and builds one delegation core per process."""

import asyncio
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import yaml

import examples_registry
from examples.frontend_backend_live import factory
from examples.frontend_backend_live.delegation_core.config import CoreSettings
from gateways.live.protocol import SessionConfig
from tests.unit.core_helpers import CORE_BLOCK, CORE_TEXTS, EXAMPLE_SETTINGS, core_block
from tests.unit.frontend_backend_live.root.root_helpers import (
    CATALOG,
    StubAdapter,
    StubSettings,
    clear_root_caches,
    reset_stubs,
    stub_root,
)


class RootCase(unittest.TestCase):
    def setUp(self):
        clear_root_caches(self)
        reset_stubs()
        self.enterContext(patch.object(factory.ExampleCatalog, "read", return_value=CATALOG))
        self.enterContext(patch.object(factory, "frontend_adapters", return_value={"cascade": StubAdapter}))

    def with_settings(self, settings: dict) -> None:
        """The example's registry entry holds ``settings`` for the rest of the test."""
        entry = examples_registry.EXAMPLES[factory.EXAMPLE_KEY]
        self.enterContext(patch.dict(entry, {"config": settings}))


class RootTests(RootCase):
    def test_the_root_reads_its_settings_once(self):
        with patch.object(examples_registry, "example_config", wraps=examples_registry.example_config) as reads:
            first, second = factory.root(), factory.root()
        self.assertIs(first, second)
        self.assertEqual(reads.call_count, 1)
        self.assertEqual(factory.ExampleCatalog.read.call_count, 1)
        self.assertEqual(first.core.backend.slot, "thinker-llm")
        self.assertEqual(len(StubSettings.parsed), 1)
        self.assertEqual(first.settings.block["display_label"], first.core.delegation.display_label)

    def test_concurrent_first_calls_read_the_settings_once(self):
        reads, started, read = [], threading.Event(), examples_registry.example_config

        def slow_read(example_key):
            reads.append(example_key)
            started.set()
            time.sleep(0.1)
            return read(example_key)

        results = []
        with patch.object(examples_registry, "example_config", side_effect=slow_read):
            threads = [threading.Thread(target=lambda: results.append(factory.root())) for _ in range(3)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(5)
        self.assertEqual(len(reads), 1)
        self.assertEqual(len({id(result) for result in results}), 1)

    def test_the_registry_entrys_core_settings_reach_the_core_and_the_frontend(self):
        self.with_settings({**EXAMPLE_SETTINGS, "core": core_block(delegation={"display_label": "screen"})})
        state = factory.root()
        self.assertEqual((state.core.delegation.display_label, state.settings.block["display_label"]), ("screen",) * 2)

    def test_malformed_settings_fail_the_root_and_are_read_again_next_time(self):
        self.with_settings({**EXAMPLE_SETTINGS, "core": core_block(delegation={"nope": 1})})
        with self.assertRaisesRegex(ValueError, "the core's settings"):
            factory.root()
        with self.assertRaisesRegex(ValueError, "the core's settings"):
            factory.root()  # not cached: corrected settings are read on the next use


class OneCoreTests(RootCase):
    def test_the_core_is_built_once_from_the_cores_settings_with_the_roots_defaults(self):
        built = []

        def build_core(settings, *, endpoints, defaults):
            built.append((settings, endpoints, defaults))
            return SimpleNamespace(session=None, one_shot_session=None)

        with (
            patch.object(factory, "build_core", build_core),
            patch.object(factory, "core_client", side_effect=lambda core: ("client of", core)),
        ):
            first = asyncio.run(factory.core())
            second = asyncio.run(factory.core())
        self.assertIs(first, second)
        self.assertEqual(len(built), 1)
        (settings, endpoints, defaults) = built[0]
        self.assertEqual(settings, CoreSettings.parse(CORE_BLOCK, CORE_TEXTS))
        self.assertEqual((defaults.model, defaults.tools), ("org/back", []))
        # Any client: the generic backend prompt as written, with none of the cafe demo's channel context.
        backend = yaml.safe_load(Path(factory.__file__).with_name("prompts.yaml").read_text(encoding="utf-8"))
        backend = backend["agent_prompts"]
        self.assertEqual(defaults.instructions, backend["Backend"][settings.backend.prompt]["content"].strip())
        self.assertEqual(settings.backend.prompt, "generic")
        self.assertEqual(endpoints(None).base_url, "http://back/v1")
        self.assertEqual(endpoints("org/other").base_url, "http://o/v1")

    def test_the_core_hook_builds_off_the_event_loop(self):
        threads = []

        def client():
            threads.append(threading.current_thread())
            return "client"

        with patch.object(factory, "responses_client", client):
            self.assertEqual(asyncio.run(factory.core()), "client")
        self.assertIsNot(threads[0], threading.main_thread())

    def test_the_core_client_answers_the_model_check_from_the_core_and_carries_its_close_budget_and_its_close(self):
        async def aclose():
            pass

        core = SimpleNamespace(
            session=lambda: "session", one_shot_session=lambda: "one-shot session", close_seconds=15, aclose=aclose
        )
        with (
            patch.object(factory, "ResponsesService") as service,
            patch.object(factory, "LocalResponsesClient") as made,
        ):
            factory.core_client(core)
        self.assertEqual(service.call_args.args, (core.session,))
        self.assertEqual(service.call_args.kwargs, {"stateless_session": core.one_shot_session, "backend": core})
        # The gateway sizes a session's close from the budget, and closes the core through its client at shutdown.
        self.assertEqual(made.call_args.kwargs, {"close_seconds": 15, "close_core": core.aclose})


class NoCorePerSessionTests(unittest.IsolatedAsyncioTestCase):
    """A session's factory builds its frontend adapter only: no core per session, none at all in client mode."""

    def setUp(self):
        reset_stubs()
        self.enterContext(patch.object(factory, "root", return_value=stub_root()))
        no_core = AssertionError("a session's factory built a delegation core")
        self.enterContext(patch.object(factory, "delegation_core", side_effect=no_core))
        self.enterContext(patch.object(factory, "responses_client", side_effect=no_core))

    async def test_no_session_builds_a_core_in_either_mode(self):
        for delegation in (None, {"type": "responses", "responses": {"model": "org/back"}}):
            config = SessionConfig.model_validate({"model": "live-1", "delegation": delegation})
            parts = await factory.create_frontend_adapter(config, "websocket", None)
            self.assertIsInstance(parts.frontend, StubAdapter)
        self.assertEqual(len(StubAdapter.built), 2)


if __name__ == "__main__":
    unittest.main()
