# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D103, D107
"""The Live example in the registry and the server: its block, whether the routes are served, its hooks, readiness."""

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
import utils
from gateways.live.gateway import LiveSettings
from tests.unit.frontend_helpers import module
from utils import listed_services

FBL = "frontend-backend-live"
HOOKS = {
    "factory": "examples.frontend_backend_live.factory:create_frontend_adapter",
    "describe": "examples.frontend_backend_live.factory:describe",
    "core": "examples.frontend_backend_live.factory:core",
}
LISTED = {"llm": [{"id": "self-hosted:a", "source": "self-hosted", "status": "ready", "selected": True}]}


def load(**changes) -> dict:
    """Load the registry with one minimal example ``x``, with ``changes`` to its entry."""
    entry = {"label": "X", "bot": "pkg.mod:bot", "services": {"llm": ["a"]}, **changes}
    return examples_registry._load_examples({"examples": {"x": entry, "y": {**entry, "live": None, "config": {}}}})


def selection(key: str | None) -> examples_registry.Selection:
    """Return the registry's selection: locked to ``key``, or every example when ``None``."""
    if key is None:
        keys = tuple(examples_registry.EXAMPLES)
        return examples_registry.Selection("all", False, keys, keys[0])
    return examples_registry.Selection(key, True, (key,), key)


def no_live_example():
    """Patch the registry so that no example has a Live block."""
    examples = {key: {**example, "live": None} for key, example in examples_registry.EXAMPLES.items()}
    return patch.object(examples_registry, "EXAMPLES", examples)


class RegistryTests(unittest.TestCase):
    def test_a_live_block_is_checked_when_the_registry_loads(self):
        self.assertEqual(load(live=dict(HOOKS))["x"]["live"], {**HOOKS, "settings": {}})
        self.assertIsNone(load()["x"]["live"])
        for bad, message in (
            ({"factory": HOOKS["factory"]}, "live must name factory, describe, core"),
            ({**HOOKS, "core": "no-function"}, r"live.core must be 'module.path:attr'"),
        ):
            with self.subTest(bad=bad), self.assertRaisesRegex(RuntimeError, message):
                load(live=bad)

    def test_an_examples_settings_are_a_mapping_read_as_a_copy(self):
        settings = {"frontend": {"adapter": "cascade"}}
        loaded = load(config=settings)
        self.assertEqual((loaded["x"]["config"], loaded["y"]["config"]), (settings, {}))
        with self.assertRaisesRegex(RuntimeError, "config must be a mapping"):
            load(config=["frontend"])
        copy = examples_registry.example_config(FBL)
        copy["core"]["backend"]["adapter"] = "harness"
        self.assertEqual(examples_registry.example_config(FBL)["core"]["backend"]["adapter"], "llm")
        with self.assertRaises(KeyError):
            examples_registry.example_config("no-such-example")

    def test_live_settings_are_checked_when_the_registry_loads(self):
        values = {"max_sessions": 3, "max_delegate_rounds": 1, "ice_urls": ["stun:stun.example:3478"]}
        self.assertEqual(load(live={**HOOKS, "settings": values})["x"]["live"]["settings"], values)
        for bad, message in (
            ({"api_token": "secret"}, "live.settings has an unknown key 'api_token'; the keys are max_sessions, "),
            ({"max_sessions": "8"}, r"live.settings.max_sessions must be a whole number of at least 1"),
            ({"max_session_seconds": 0}, r"live.settings.max_session_seconds must be a whole number of at least 1"),
            ({"max_delegate_rounds": True}, r"live.settings.max_delegate_rounds must be a whole number"),
            ({"allowed_hosts": "voice.lab"}, r"live.settings.allowed_hosts must be a list of strings"),
            ({"ice_urls": [3478]}, r"live.settings.ice_urls must be a list of strings"),
            (["max_sessions"], r"live.settings must be a mapping"),
        ):
            with self.subTest(bad=bad), self.assertRaisesRegex(RuntimeError, message):
                load(live={**HOOKS, "settings": bad})

    def test_each_live_settings_key_is_a_gateway_setting(self):
        fields = {field.name for field in dataclasses.fields(LiveSettings)}
        self.assertLessEqual({*examples_registry.LIVE_LIMITS, *examples_registry.LIVE_LISTS}, fields)

    def test_the_live_example_is_the_selected_one_with_a_block(self):
        keys = tuple(examples_registry.EXAMPLES)
        for chosen, served in (
            (selection(FBL), FBL),
            (examples_registry.Selection("all", False, keys, FBL), FBL),
            (selection("generic-assistant"), None),
            (selection(None), None),
        ):
            with (
                self.subTest(selection=chosen.raw, default=chosen.default_key),
                patch.object(examples_registry, "_SELECTION", chosen),
            ):
                example = examples_registry.live_example()
                self.assertEqual(example and example["key"], served)
        with no_live_example(), patch.object(examples_registry, "_SELECTION", selection(FBL)):
            self.assertIsNone(examples_registry.live_example())

    def test_a_hook_is_imported_only_when_it_is_resolved(self):
        async def factory(*args):
            return args

        with patch.dict(sys.modules, module("tests_live_hooks", factory=factory, not_callable=1)):
            self.assertIs(examples_registry.resolve_hook("tests_live_hooks:factory", "live.factory"), factory)
            with self.assertRaisesRegex(RuntimeError, "live.core target is not callable"):
                examples_registry.resolve_hook("tests_live_hooks:not_callable", "live.core")
        with self.assertRaisesRegex(RuntimeError, "Example bot must be 'module.path:attr'"):
            examples_registry.resolve_bot({"bot": "no-function"})


class MountCase(unittest.TestCase):
    def setUp(self):
        self.mounts: list[tuple[tuple, dict]] = []
        self.enterContext(patch.dict(os.environ))
        for name in ("LIVE_ENABLED", "LIVE_API_TOKEN"):
            os.environ.pop(name, None)
        self.enterContext(patch.object(examples_registry, "_SELECTION", selection(FBL)))

        def mount_live(*args, **kwargs):
            self.mounts.append((args, kwargs))

        self.enterContext(patch.object(gateways.live.mount, "mount_live", mount_live))
        self.logged: list[str] = []
        self.addCleanup(logger.remove, logger.add(self.logged.append, level="DEBUG", format="{level} {message}"))

    def mount(self) -> dict:
        """Mount on a new app and return the keywords the mount got, empty when the routes are not served."""
        app = object()
        self.mounts.clear()
        server._mount_live(app)
        ((args, kwargs),) = self.mounts
        self.assertEqual(args, (app,))
        return kwargs


class ServeTests(MountCase):
    def test_another_examples_deployment_never_serves_the_routes(self):
        for locked in ("generic-assistant", None):
            with (
                self.subTest(selection=locked),
                patch.object(examples_registry, "_SELECTION", selection(locked)),
                patch.dict(os.environ, {"LIVE_API_TOKEN": "secret", "LIVE_ENABLED": "true"}),
            ):
                self.assertEqual(self.mount(), {})
        self.assertEqual([line for line in self.logged if line.startswith("INFO")], [])

    def test_live_enabled_decides_and_alone_serves_with_no_token(self):
        token = {"LIVE_API_TOKEN": "t"}
        for settings, served in (
            *(({"LIVE_ENABLED": value, **token}, True) for value in ("true", "1", "yes", "on")),
            ({"LIVE_ENABLED": "false", **token}, False),
            ({}, False),
            ({"LIVE_ENABLED": "true"}, True),
        ):
            with self.subTest(settings=settings), patch.dict(os.environ, settings):
                self.assertEqual(bool(self.mount()), served)

    def test_a_startup_line_says_how_to_serve_unserved_routes(self):
        self.assertEqual(self.mount(), {})
        (line,) = self.logged
        self.assertRegex(line, r"^INFO .*/v1/live.*LIVE_API_TOKEN.*LIVE_ENABLED=true")
        with patch.dict(os.environ, {"LIVE_API_TOKEN": "secret"}):
            self.assertTrue(self.mount())
        with no_live_example():
            self.assertEqual(self.mount(), {})
        self.assertEqual(len(self.logged), 1)


class HookTests(MountCase):
    def setUp(self):
        super().setUp()
        os.environ["LIVE_ENABLED"] = "true"
        self.calls: list[tuple] = []
        self.bound: list[str] = []
        self.listed: list[dict] = []
        self.probes = 0
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

    def test_the_routes_mount_with_the_hooks_and_the_settings(self):
        values = {"max_sessions": 3, "allowed_hosts": ["Voice.Lab"]}
        with (
            patch.dict(examples_registry.EXAMPLES[FBL]["live"], {"settings": values}),
            patch.dict(os.environ, {"LIVE_API_TOKEN": "secret"}),
        ):
            kwargs = self.mount()
        self.assertEqual(set(kwargs), {"factory", "core", "describe", "settings", "ice_servers"})
        settings = LiveSettings(api_token="secret", max_sessions=3, allowed_hosts=("voice.lab",))
        self.assertEqual(kwargs["settings"], settings)
        self.assertIs(kwargs["ice_servers"], server._build_ice_servers)
        self.assertEqual(asyncio.run(kwargs["factory"]("config", "websocket", None)), "live.factory")
        self.assertEqual(asyncio.run(kwargs["core"]()), "live.core")
        self.assertEqual(asyncio.run(kwargs["describe"]()), "live.describe")
        specs = {what: spec for what, spec, _, _ in self.calls}
        self.assertEqual(specs, {f"live.{name}": spec for name, spec in HOOKS.items()})
        self.assertEqual(self.calls[0][2], ("config", "websocket", None))
        with patch.object(examples_registry, "resolve_hook", lambda spec, what: lambda *args: what):
            self.assertEqual(asyncio.run(kwargs["core"]()), "live.core")

    def test_each_hook_runs_with_the_services_bound_and_listed(self):
        kwargs = self.mount()
        asyncio.run(kwargs["factory"]("config", "websocket", None))
        asyncio.run(kwargs["core"]())
        asyncio.run(kwargs["describe"]())
        asyncio.run(kwargs["factory"]("config", "websocket", None))
        self.assertEqual([call[3] for call in self.calls], [[FBL], [FBL] * 2, [FBL] * 3, [FBL] * 4])
        self.assertEqual((self.probes, self.listed), (1, [LISTED] * 4))


class ReadinessTests(unittest.IsolatedAsyncioTestCase):
    EXAMPLE = {"services": {"llm": ["a"], "thinker-llm": ["a"], "asr": ["b"]}}

    def setUp(self):
        self.checked = []

        async def check(label, target, health_url, expects_ready_json=False):
            self.checked.append((label, target, health_url))

        defaults = {"llm": {"base_url": "http://talker:8000/v1"}, "thinker-llm": {"base_url": "http://thinker:8000/v1"}}
        self.enterContext(patch.object(server, "_run_http_readiness_check", check))
        self.enterContext(patch.object(server, "_llm_health_url", side_effect=lambda url: f"{url}/health"))
        self.enterContext(patch.object(server, "load_service_entry", side_effect=lambda slot, key: defaults[slot]))

    async def test_both_llm_slots_are_checked_at_the_sessions_choice(self):
        config = {"base_url": "http://picked:8000/v1", "thinker_base_url": "http://picked-thinker:8000/v1"}
        await server._ensure_llm_ready_for_connection(config, self.EXAMPLE)
        picked, thinker = "http://picked:8000/v1", "http://picked-thinker:8000/v1"
        self.assertEqual(
            self.checked, [("LLM", picked, f"{picked}/health"), ("Thinker LLM", thinker, f"{thinker}/health")]
        )

    async def test_an_unset_slot_is_checked_at_its_default_entry(self):
        await server._ensure_llm_ready_for_connection({}, self.EXAMPLE)
        self.assertEqual([target for _, target, _ in self.checked], ["http://talker:8000/v1", "http://thinker:8000/v1"])

    async def test_an_example_with_one_llm_slot_checks_that_slot_only(self):
        await server._ensure_llm_ready_for_connection({}, {"services": {"llm": ["a"], "tts": ["c"]}})
        self.assertEqual([label for label, _, _ in self.checked], ["LLM"])

    async def test_an_endpoint_with_no_health_path_is_not_checked(self):
        with patch.object(server, "_llm_health_url", return_value=""):
            await server._ensure_llm_ready_for_connection({}, self.EXAMPLE)
        self.assertEqual(self.checked, [])

    def test_a_thinker_only_entry_has_its_health_url(self):
        entry = {"base_url": "http://thinker-only:8000/v1", "health_path": "/v1/health/ready"}
        sources = [(utils.SELF_HOSTED_SOURCE, {"llm": {}, "thinker-llm": {"t": entry}, "asr": {"a": {"server": "x"}}})]
        with patch.object(utils, "service_catalog_sources", return_value=sources):
            url = utils._llm_health_url("http://thinker-only:8000/v1")
        self.assertEqual(url, "http://thinker-only:8000/v1/health/ready")
