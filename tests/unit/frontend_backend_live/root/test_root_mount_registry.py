# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D103, D107
"""The mounted example in the registry."""

import asyncio
import os
import sys
import unittest
from unittest.mock import patch

import examples_registry
import live.mount
import server
from live.gateway import LiveSettings
from tests.unit.frontend_backend_live.root.root_helpers import module
from utils import clear_service_context, set_active_services

FBL = "frontend-backend-live"
HOOKS = {
    "factory": "examples.frontend_backend_live.factory:create_frontend_adapter",
    "describe": "examples.frontend_backend_live.factory:describe",
    "core": "examples.frontend_backend_live.factory:core",
}
LIVE_SETTINGS = ("LIVE_ENABLED", "LIVE_API_TOKEN", "LIVE_ADAPTER_FACTORY")


def entry(**changes) -> dict:
    """A minimal registry entry."""
    return {"label": "X", "bot": "pkg.mod:bot", "services": {"llm": ["a"]}, **changes}


def selection(key: str | None) -> examples_registry.Selection:
    """The registry's selection: locked to ``key``, or every example when ``None``."""
    if key is None:
        keys = tuple(examples_registry.EXAMPLES)
        return examples_registry.Selection("all", False, keys, keys[0])
    return examples_registry.Selection(key, True, (key,), key)


class RegistryTests(unittest.TestCase):
    def test_a_live_block_is_checked_when_the_registry_loads(self):
        loaded = examples_registry._load_examples({"examples": {"x": entry(live=dict(HOOKS))}})
        self.assertEqual(loaded["x"]["live"], HOOKS)
        self.assertIsNone(examples_registry._load_examples({"examples": {"x": entry()}})["x"]["live"])
        for bad, message in (
            ({"factory": HOOKS["factory"]}, "live must name factory, describe, core"),
            ({**HOOKS, "delegate": "m:f"}, "live must name factory, describe, core"),
            ({**HOOKS, "core": "no-function"}, r"live.core must be 'module.path:attr'"),
            ("m:f", "live must name"),
        ):
            with self.subTest(bad=bad), self.assertRaisesRegex(RuntimeError, message):
                examples_registry._load_examples({"examples": {"x": entry(live=bad)}})

    def test_the_live_example_is_the_selected_one_with_a_block_else_the_first_that_has_one(self):
        for locked in (None, "generic-assistant", FBL):
            with self.subTest(selection=locked), patch.object(examples_registry, "_SELECTION", selection(locked)):
                self.assertEqual(examples_registry.live_example()["key"], FBL)
        examples = {key: {**example, "live": None} for key, example in examples_registry.EXAMPLES.items()}
        with patch.object(examples_registry, "EXAMPLES", examples):
            self.assertIsNone(examples_registry.live_example())

    def test_a_hook_is_imported_only_when_it_is_resolved(self):
        async def factory(*args):
            return args

        stand_in = module("tests_live_hooks", factory=factory, not_callable=1)
        with patch.dict(sys.modules, stand_in):
            self.assertIs(examples_registry.resolve_hook("tests_live_hooks:factory", "live.factory"), factory)
            with self.assertRaisesRegex(RuntimeError, "live.core target is not callable"):
                examples_registry.resolve_hook("tests_live_hooks:not_callable", "live.core")
        with self.assertRaisesRegex(RuntimeError, "Example bot must be 'module.path:attr'"):
            examples_registry.resolve_bot({"bot": "no-function"})


class MountCase(unittest.TestCase):
    def setUp(self):
        self.mounts: list[tuple[tuple, dict]] = []
        self.enterContext(patch.dict(os.environ))
        for name in LIVE_SETTINGS:
            os.environ.pop(name, None)
        self.enterContext(patch.object(examples_registry, "_SELECTION", selection(None)))
        self.enterContext(
            patch.object(live.mount, "mount_live", lambda *args, **kwargs: self.mounts.append((args, kwargs)))
        )

    def mount(self) -> dict:
        """Mount on a new app; the keywords the mount got (empty when the routes are not served)."""
        app = object()
        server._mount_live(app)
        ((args, kwargs),) = self.mounts
        self.assertEqual(args, (app,))
        return kwargs


class ServeDecisionTests(MountCase):
    def test_another_examples_deployment_does_not_serve_the_routes(self):
        self.assertEqual(self.mount(), {})

    def test_live_enabled_decides_when_it_is_set(self):
        for value, served in (("true", True), ("1", True), ("on", True), ("false", False), ("no", False)):
            self.mounts.clear()
            with (
                self.subTest(LIVE_ENABLED=value),
                patch.dict(os.environ, {"LIVE_ENABLED": value, "LIVE_API_TOKEN": "t"}),
            ):
                self.assertEqual(bool(self.mount()), served)
        self.mounts.clear()
        with (
            self.subTest(LIVE_ENABLED="false", pinned=FBL),
            patch.object(examples_registry, "_SELECTION", selection(FBL)),
            patch.dict(os.environ, {"LIVE_ENABLED": "false"}),
        ):
            self.assertEqual(self.mount(), {})

    def test_a_token_serves_the_routes(self):
        with patch.dict(os.environ, {"LIVE_API_TOKEN": "secret"}):
            self.assertTrue(self.mount())

    def test_a_deployment_pinned_to_the_example_serves_the_routes(self):
        with patch.object(examples_registry, "_SELECTION", selection(FBL)):
            self.assertTrue(self.mount())


class HookTests(MountCase):
    def setUp(self):
        super().setUp()
        os.environ["LIVE_ENABLED"] = "true"
        self.calls: list[tuple] = []
        self.bound: list[str] = []
        self.enterContext(
            patch.object(server, "set_service_context", lambda example: self.bound.append(example["key"]))
        )

        def resolve(spec, what):
            async def hook(*args):
                self.calls.append((what, spec, args, list(self.bound)))
                return what

            return hook

        self.enterContext(patch.object(examples_registry, "resolve_hook", resolve))

    def test_the_routes_mount_with_the_examples_hooks_the_settings_and_the_turn_servers(self):
        kwargs = self.mount()
        self.assertEqual(set(kwargs), {"factory", "core", "describe", "settings", "ice_servers"})
        self.assertIsInstance(kwargs["settings"], LiveSettings)
        self.assertIs(kwargs["ice_servers"], server._build_ice_servers)
        self.assertEqual(asyncio.run(kwargs["factory"]("config", "websocket", None)), "live.factory")
        self.assertEqual(asyncio.run(kwargs["core"]()), "live.core")
        self.assertEqual(asyncio.run(kwargs["describe"]()), "live.describe")
        specs = {what: spec for what, spec, _, _ in self.calls}
        self.assertEqual(specs, {f"live.{name}": spec for name, spec in HOOKS.items()})
        self.assertEqual(self.calls[0][2], ("config", "websocket", None))

    def test_each_hook_runs_with_the_live_examples_services_bound_under_another_examples_lock(self):
        with patch.object(examples_registry, "_SELECTION", selection("generic-assistant")):
            kwargs = self.mount()
        asyncio.run(kwargs["core"]())
        self.assertEqual(self.calls[0][3], [FBL])  # bound before the hook ran

    def test_live_adapter_factory_overrides_the_factory_only(self):
        with patch.dict(os.environ, {"LIVE_ADAPTER_FACTORY": "custom.module:build"}):
            kwargs = self.mount()
        for name in ("factory", "core"):
            asyncio.run(kwargs[name](*(("c", "websocket", None) if name == "factory" else ())))
        self.assertEqual([spec for _, spec, _, _ in self.calls], ["custom.module:build", HOOKS["core"]])


class RealMountTests(unittest.TestCase):
    def setUp(self):
        self.enterContext(patch.dict(os.environ, {"LIVE_ENABLED": "true"}))
        for name in ("LIVE_API_TOKEN", "LIVE_ADAPTER_FACTORY"):
            os.environ.pop(name, None)
        # create_app binds the selected example's services process-wide; leave no binding behind.
        self.addCleanup(clear_service_context)
        self.addCleanup(set_active_services, None)

    def test_the_server_mounts_the_registry_hooks_on_the_gateway(self):
        app = server.create_app()
        # The info route is added only with the hooks; the session routes come as one included router, with no path.
        paths = {getattr(route, "path", None) for route in app.routes}
        self.assertIn("/v1/live/info", paths)


if __name__ == "__main__":
    unittest.main()
