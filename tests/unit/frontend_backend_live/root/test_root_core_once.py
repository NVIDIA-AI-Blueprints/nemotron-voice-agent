# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D103, D107
"""The root reads its files once and builds one delegation core per process."""

import asyncio
import os
import threading
import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

from delegation_core.config import CoreSettings
from examples.frontend_backend_live import factory
from examples.frontend_backend_live.config_manager.loader import CONFIG_PATH, load_live_config
from examples.frontend_backend_live.prompts.catalog import role_instructions
from live.protocol import SessionConfig
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
        self.enterContext(patch.dict(os.environ))
        os.environ.pop("CORE_CONFIG_PATH", None)
        os.environ.pop("LIVE_CONFIG_PATH", None)


class RootTests(RootCase):
    def test_the_root_reads_its_files_once(self):
        with patch.object(factory, "load_live_config", wraps=load_live_config) as reads:
            first, second = factory.root(), factory.root()
        self.assertIs(first, second)
        self.assertEqual(reads.call_count, 1)
        self.assertEqual(factory.ExampleCatalog.read.call_count, 1)
        self.assertEqual(first.core.backend.slot, "thinker-llm")
        self.assertEqual(len(StubSettings.parsed), 1)
        self.assertEqual(first.settings.block["display_label"], first.core.delegation.display_label)

    def test_concurrent_first_calls_read_the_files_once(self):
        reads, started = [], threading.Event()

        def slow_read(path=None):
            reads.append(path)
            started.set()
            time.sleep(0.1)
            return load_live_config(path)

        results = []
        with patch.object(factory, "load_live_config", side_effect=slow_read):
            threads = [threading.Thread(target=lambda: results.append(factory.root())) for _ in range(3)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(5)
        self.assertEqual(len(reads), 1)
        self.assertEqual(len({id(result) for result in results}), 1)

    def test_core_config_path_points_the_root_at_another_core_file(self):
        with TemporaryDirectory() as folder:
            profile = Path(folder) / "core.custom.yaml"
            core_yaml = CONFIG_PATH.with_name("core.yaml")
            profile.write_text(f"extends: {core_yaml}\ndelegation: {{display_label: screen}}\n", encoding="utf-8")
            with patch.dict(os.environ, {"CORE_CONFIG_PATH": str(profile)}):
                state = factory.root()
        self.assertEqual((state.core.delegation.display_label, state.settings.block["display_label"]), ("screen",) * 2)

    def test_a_malformed_core_file_fails_the_root_and_is_read_again_next_time(self):
        with TemporaryDirectory() as folder:
            broken = Path(folder) / "core.yaml"
            broken.write_text("delegation: {nope: 1}\n", encoding="utf-8")
            with patch.dict(os.environ, {"CORE_CONFIG_PATH": str(broken)}):
                with self.assertRaisesRegex(ValueError, "the core's settings"):
                    factory.root()
                with self.assertRaisesRegex(ValueError, "the core's settings"):
                    factory.root()  # not cached: a fixed file is read on the next use


class OneCoreTests(RootCase):
    def test_the_core_is_built_once_from_the_cores_file_with_the_roots_defaults(self):
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
        self.assertEqual(settings, CoreSettings.load(CONFIG_PATH.with_name("core.yaml")))
        self.assertEqual((defaults.model, defaults.tools), ("org/back", []))
        self.assertEqual(defaults.instructions, role_instructions("thinker"))
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
