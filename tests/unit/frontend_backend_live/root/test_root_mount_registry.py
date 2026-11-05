# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D103, D107
"""The mounted example in the registry."""

import asyncio
import dataclasses
import os
import sys
import unittest
from unittest.mock import patch

from loguru import logger

import examples_registry
import gateways.live.mount
import server
from gateways.live.gateway import LiveSettings
from tests.unit.frontend_backend_live.root.root_helpers import module
from utils import clear_service_context, listed_services, set_active_services

FBL = "frontend-backend-live"
HOOKS = {
    "factory": "examples.frontend_backend_live.factory:create_frontend_adapter",
    "describe": "examples.frontend_backend_live.factory:describe",
    "core": "examples.frontend_backend_live.factory:core",
}
LIVE_SETTINGS = ("LIVE_ENABLED", "LIVE_API_TOKEN")
# The services as GET /api/services lists them, each self-hosted one with its status.
LISTED = {"llm": [{"id": "self-hosted:a", "source": "self-hosted", "status": "ready", "selected": True}]}


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
        self.assertEqual(loaded["x"]["live"], {**HOOKS, "settings": {}})  # no settings: the gateway's defaults
        self.assertIsNone(examples_registry._load_examples({"examples": {"x": entry()}})["x"]["live"])
        for bad, message in (
            ({"factory": HOOKS["factory"]}, "live must name factory, describe, core"),
            ({**HOOKS, "delegate": "m:f"}, "live must name factory, describe, core"),
            ({**HOOKS, "core": "no-function"}, r"live.core must be 'module.path:attr'"),
            ("m:f", "live must name"),
        ):
            with self.subTest(bad=bad), self.assertRaisesRegex(RuntimeError, message):
                examples_registry._load_examples({"examples": {"x": entry(live=bad)}})

    def test_live_settings_are_checked_when_the_registry_loads(self):
        values = {"max_sessions": 3, "max_delegate_rounds": 1, "ice_urls": ["stun:stun.example:3478"]}
        loaded = examples_registry._load_examples({"examples": {"x": entry(live={**HOOKS, "settings": values})}})
        self.assertEqual(loaded["x"]["live"]["settings"], values)
        for bad, message in (
            ({"max_session": 3}, "live.settings has an unknown key 'max_session'; the keys are max_sessions, "),
            ({"api_token": "secret"}, "unknown key 'api_token'"),
            ({"max_sessions": "8"}, r"live.settings.max_sessions must be a whole number of at least 1"),
            ({"max_session_seconds": 0}, r"live.settings.max_session_seconds must be a whole number of at least 1"),
            ({"max_appends_per_minute": 2.5}, r"live.settings.max_appends_per_minute must be a whole number"),
            ({"max_delegate_rounds": True}, r"live.settings.max_delegate_rounds must be a whole number"),
            ({"allowed_hosts": "voice.lab"}, r"live.settings.allowed_hosts must be a list of strings"),
            ({"ice_urls": [3478]}, r"live.settings.ice_urls must be a list of strings"),
            (["max_sessions"], r"live.settings must be a mapping"),
        ):
            with self.subTest(bad=bad), self.assertRaisesRegex(RuntimeError, message):
                examples_registry._load_examples({"examples": {"x": entry(live={**HOOKS, "settings": bad})}})

    def test_each_live_settings_key_is_a_gateway_setting(self):
        fields = {field.name for field in dataclasses.fields(LiveSettings)}
        self.assertLessEqual({*examples_registry.LIVE_LIMITS, *examples_registry.LIVE_LISTS}, fields)

    def test_the_live_example_is_the_selected_one_only_when_it_has_a_block(self):
        keys = tuple(examples_registry.EXAMPLES)
        for chosen, served in (
            (selection(FBL), FBL),
            (examples_registry.Selection("all", False, keys, FBL), FBL),  # every example, the Live one by default
            (selection("generic-assistant"), None),
            (selection(None), None),  # every example, the first one by default
        ):
            with (
                self.subTest(selection=chosen.raw, default=chosen.default_key),
                patch.object(examples_registry, "_SELECTION", chosen),
            ):
                example = examples_registry.live_example()
                self.assertEqual(example and example["key"], served)
        examples = {key: {**example, "live": None} for key, example in examples_registry.EXAMPLES.items()}
        with (
            patch.object(examples_registry, "EXAMPLES", examples),
            patch.object(examples_registry, "_SELECTION", selection(FBL)),
        ):
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
        self.enterContext(patch.object(examples_registry, "_SELECTION", selection(FBL)))  # a deployment of the example
        self.enterContext(
            patch.object(gateways.live.mount, "mount_live", lambda *args, **kwargs: self.mounts.append((args, kwargs)))
        )

    def mount(self) -> dict:
        """Mount on a new app; the keywords the mount got (empty when the routes are not served)."""
        app = object()
        server._mount_live(app)
        ((args, kwargs),) = self.mounts
        self.assertEqual(args, (app,))
        return kwargs


class ServeDecisionTests(MountCase):
    def test_another_examples_deployment_does_not_serve_the_routes_whatever_its_settings(self):
        logged: list[str] = []
        self.addCleanup(logger.remove, logger.add(logged.append, level="INFO", format="{level} {message}"))
        for locked in ("generic-assistant", None):  # another example, or every example with the first one by default
            self.mounts.clear()
            with (
                self.subTest(selection=locked),
                patch.object(examples_registry, "_SELECTION", selection(locked)),
                patch.dict(os.environ, {"LIVE_API_TOKEN": "secret", "LIVE_ENABLED": "true"}),
            ):
                self.assertEqual(self.mount(), {})
        self.assertEqual(logged, [])  # no setting would serve them, so no startup line says how

    def test_live_enabled_decides_when_it_is_set(self):
        for value, served in (("true", True), ("1", True), ("on", True), ("false", False), ("no", False)):
            self.mounts.clear()
            with (
                self.subTest(LIVE_ENABLED=value),
                patch.dict(os.environ, {"LIVE_ENABLED": value, "LIVE_API_TOKEN": "t"}),
            ):
                self.assertEqual(bool(self.mount()), served)
        self.mounts.clear()
        with self.subTest(LIVE_ENABLED="false", token=None), patch.dict(os.environ, {"LIVE_ENABLED": "false"}):
            self.assertEqual(self.mount(), {})

    def test_a_token_serves_the_routes(self):
        with patch.dict(os.environ, {"LIVE_API_TOKEN": "secret"}):
            self.assertTrue(self.mount())

    def test_without_a_token_only_live_enabled_serves_the_routes(self):
        self.assertEqual(self.mount(), {})
        self.mounts.clear()
        with patch.dict(os.environ, {"LIVE_ENABLED": "true"}):
            self.assertTrue(self.mount())

    def test_a_startup_line_says_how_to_serve_the_routes_when_they_are_not_served(self):
        logged: list[str] = []
        self.addCleanup(logger.remove, logger.add(logged.append, level="DEBUG", format="{level} {message}"))
        self.assertEqual(self.mount(), {})
        (line,) = logged
        self.assertRegex(line, r"^INFO .*/v1/live.*LIVE_API_TOKEN.*LIVE_ENABLED=true")
        self.mounts.clear()
        with patch.dict(os.environ, {"LIVE_API_TOKEN": "secret"}):
            self.assertTrue(self.mount())
        self.mounts.clear()
        examples = {key: {**example, "live": None} for key, example in examples_registry.EXAMPLES.items()}
        with patch.object(examples_registry, "EXAMPLES", examples):
            self.assertEqual(self.mount(), {})  # no example offers Live, so neither setting would serve it
        self.assertEqual(len(logged), 1)


class HookTests(MountCase):
    def setUp(self):
        super().setUp()
        os.environ["LIVE_ENABLED"] = "true"
        self.calls: list[tuple] = []
        self.bound: list[str] = []
        self.listed: list[dict] = []  # the services each hook read, in order
        self.probes = 0  # how many times the services were listed with their status
        self.enterContext(
            patch.object(server, "set_service_context", lambda example: self.bound.append(example["key"]))
        )
        self.enterContext(patch.object(server, "build_services_api_response", dict))

        async def with_status(services):
            self.probes += 1
            return LISTED

        self.enterContext(patch.object(server, "_with_service_status", with_status))

        def resolve(spec, what):
            async def hook(*args):
                self.calls.append((what, spec, args, list(self.bound)))
                self.listed.append(listed_services())
                return what

            return hook

        self.enterContext(patch.object(examples_registry, "resolve_hook", resolve))

    def test_the_routes_mount_with_the_examples_hooks_the_settings_and_the_turn_servers(self):
        values = {"max_sessions": 3, "allowed_hosts": ["Voice.Lab"]}
        with (
            patch.dict(examples_registry.EXAMPLES[FBL]["live"], {"settings": values}),
            patch.dict(os.environ, {"LIVE_API_TOKEN": "secret"}),
        ):
            kwargs = self.mount()
        self.assertEqual(set(kwargs), {"factory", "core", "describe", "settings", "ice_servers"})
        self.assertEqual(
            kwargs["settings"], LiveSettings(api_token="secret", max_sessions=3, allowed_hosts=("voice.lab",))
        )
        self.assertIs(kwargs["ice_servers"], server._build_ice_servers)
        self.assertEqual(asyncio.run(kwargs["factory"]("config", "websocket", None)), "live.factory")
        self.assertEqual(asyncio.run(kwargs["core"]()), "live.core")
        self.assertEqual(asyncio.run(kwargs["describe"]()), "live.describe")
        specs = {what: spec for what, spec, _, _ in self.calls}
        self.assertEqual(specs, {f"live.{name}": spec for name, spec in HOOKS.items()})
        self.assertEqual(self.calls[0][2], ("config", "websocket", None))

    def test_each_hook_runs_with_the_live_examples_services_bound(self):
        kwargs = self.mount()
        asyncio.run(kwargs["core"]())
        self.assertEqual(self.calls[0][3], [FBL])  # bound before the hook ran

    def test_each_hook_reads_the_services_as_the_repository_ui_lists_them(self):
        kwargs = self.mount()
        asyncio.run(kwargs["factory"]("config", "websocket", None))
        asyncio.run(kwargs["describe"]())
        self.assertEqual(self.listed, [LISTED, LISTED])

    def test_the_hooks_list_the_services_once_and_a_later_call_probes_nothing(self):
        kwargs = self.mount()
        asyncio.run(kwargs["factory"]("config", "websocket", None))
        asyncio.run(kwargs["core"]())
        asyncio.run(kwargs["describe"]())
        asyncio.run(kwargs["factory"]("config", "websocket", None))
        self.assertEqual(self.probes, 1)
        self.assertEqual(self.listed, [LISTED] * 4)


class RealMountTests(unittest.TestCase):
    def setUp(self):
        self.enterContext(patch.dict(os.environ, {"LIVE_ENABLED": "true"}))
        os.environ.pop("LIVE_API_TOKEN", None)
        self.enterContext(patch.object(examples_registry, "_SELECTION", selection(FBL)))
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
