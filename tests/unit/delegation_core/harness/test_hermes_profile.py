# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D103

"""The native extension preserves authored requests and refuses identity changes."""

import json
import os
import tempfile
import unittest
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from examples.frontend_backend_live.delegation_core.backend.harness import hermes_profile

BINDING = {
    "provider": "custom",
    "model": "bound/model",
    "base_url": "http://127.0.0.1:1/v1",
    "api": "chat-completions",
    "api_key_env": "NATIVE_PROFILE_UNIT_KEY",
    "extra_params": {
        "temperature": 0,
        "max_tokens": 4096,
        "extra_body": {"chat_template_kwargs": {"enable_thinking": True}},
    },
}


@dataclass
class NativeProfile:
    name: str = "custom"
    api_mode: str = "chat_completions"
    model_capabilities: dict = field(default_factory=lambda: {"bound/model": {"supports_reasoning": True}})
    inherited_body: dict = field(default_factory=dict)
    inherited_api_body: dict = field(default_factory=dict)
    inherited_top: dict = field(default_factory=dict)

    def default_reasoning_config(self, model=None):
        return {"effort": "medium"}

    def create_client(self, **kwargs):
        return None

    def build_extra_body(self, **context):
        return {"native_provider_field": True, **self.inherited_body}

    def build_api_kwargs_extras(self, **context):
        return self.inherited_api_body, self.inherited_top


class ProfileTests(unittest.TestCase):
    context = {"model": BINDING["model"], "base_url": BINDING["base_url"]}

    @contextmanager
    def registered(self, base, *, binding=BINDING, configured_reasoning=None, verify_install=False):
        with tempfile.TemporaryDirectory() as temporary:
            home = Path(temporary)
            hermes_profile.install_profile(home, binding)
            plugin = home / "plugins/model-providers" / hermes_profile.PLUGIN_NAME
            if verify_install:
                self.assertEqual(json.loads((plugin / "binding.json").read_text()), binding)
            captured = []
            modules = {
                "providers": SimpleNamespace(get_provider_profile=lambda name: base, register_provider=captured.append),
                "hermes_cli.config": SimpleNamespace(load_config=lambda: {"agent": {"reasoning_effort": "authored"}}),
                "hermes_constants": SimpleNamespace(resolve_reasoning_config=lambda cfg, model: configured_reasoning),
            }
            with (
                patch.dict("sys.modules", modules),
                patch.object(hermes_profile, "__file__", str(plugin / "__init__.py")),
                patch.dict(os.environ, {"NATIVE_PROFILE_UNIT_KEY": "unit-test-only"}),
            ):
                hermes_profile.register()
                if verify_install:
                    hermes_profile.require_registered_profile(home, binding)
                [profile] = captured
                yield profile

    def test_inherited_unauthored_reasoning_is_refused_before_any_request_can_be_sent(self):
        cases = (
            (NativeProfile(inherited_body={"reasoning_budget": 2048}), "build_extra_body", None),
            (NativeProfile(inherited_api_body={"reasoning": {"max_tokens": 2048}}), "build_api_kwargs_extras", None),
            (
                NativeProfile(inherited_top={"reasoning_effort": "none"}),
                "build_api_kwargs_extras",
                {"enabled": False, "effort": "none"},
            ),
        )
        for base, hook, reasoning in cases:
            with self.subTest(hook=hook, reasoning=reasoning), self.registered(base) as profile:
                forwarded = []
                with self.assertRaisesRegex(ValueError, "unauthored reasoning settings; refusing model request"):
                    forwarded.append(getattr(profile, hook)(**self.context, reasoning_config=reasoning))
                self.assertEqual(forwarded, [])

    def test_explicit_binding_reasoning_is_preserved_and_inherited_mismatch_is_refused(self):
        binding = {**BINDING, "extra_params": {"extra_body": {"reasoning_budget": 2048}}}
        for budget in (2048, 1024):
            with (
                self.subTest(budget=budget),
                self.registered(
                    NativeProfile(inherited_api_body={"reasoning_budget": budget}), binding=binding
                ) as profile,
            ):
                if budget == 2048:
                    self.assertEqual(profile.build_api_kwargs_extras(**self.context), ({"reasoning_budget": 2048}, {}))
                else:
                    with self.assertRaisesRegex(ValueError, "unauthored reasoning"):
                        profile.build_api_kwargs_extras(**self.context)

    def test_explicit_native_reasoning_survives_but_one_shot_override_does_not_gain_authority(self):
        for configured, wire in (({"enabled": True, "effort": "high"}, "high"), ({"enabled": False}, "none")):
            with (
                self.subTest(configured=configured),
                self.registered(
                    NativeProfile(inherited_top={"reasoning_effort": wire}), configured_reasoning=configured
                ) as profile,
            ):
                self.assertEqual(
                    profile.build_api_kwargs_extras(**self.context, reasoning_config=configured)[1]["reasoning_effort"],
                    wire,
                )
        with (
            self.registered(
                NativeProfile(inherited_top={"reasoning_effort": "none"}),
                configured_reasoning={"enabled": True, "effort": "high"},
            ) as profile,
            self.assertRaisesRegex(ValueError, "unauthored reasoning"),
        ):
            profile.build_api_kwargs_extras(**self.context, reasoning_config={"enabled": False, "effort": "none"})

    def test_invalid_request_fields_cannot_replace_identity_authentication_tools_or_transport(self):
        for key in ("model", "api_key", "Authorization", "messages", "tools", "stream", "temperature"):
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "cannot override"):
                hermes_profile.validated_binding({**BINDING, "extra_params": {"extra_body": {key: "changed"}}})
        for params in (
            {"max_tokens": True},
            {"temperature": float("nan")},
            {"extra_body": {"bad": object()}},
            {"extra_body": {"nested": {1: "not a JSON object key"}}},
            {"extra_body": {"nested": ("not", "a JSON array")}},
        ):
            with self.subTest(params=params), self.assertRaises(ValueError):
                hermes_profile.validated_binding({**BINDING, "extra_params": params})

    def test_public_profile_preserves_body_capability_and_route_and_checks_live_key_reference(self):
        with self.registered(NativeProfile(), verify_install=True) as profile:
            self.assertIsNone(profile.default_reasoning_config("bound/model"))
            self.assertTrue(profile.model_capabilities["bound/model"]["supports_reasoning"])
            self.assertEqual(
                profile.build_extra_body(**self.context),
                {"native_provider_field": True, "chat_template_kwargs": {"enable_thinking": True}},
            )
            assert profile.build_api_kwargs_extras(**self.context) == ({}, {"temperature": 0, "max_tokens": 4096})
            self.assertIsNone(profile.create_client(base_url=BINDING["base_url"], api_key="unit-test-only"))
            for changed in ({"model": "other/model"}, {"base_url": "https://another.invalid/v1"}):
                with self.subTest(changed=changed), self.assertRaisesRegex(ValueError, "differs"):
                    profile.build_extra_body(**{**self.context, **changed})
            with self.assertRaisesRegex(ValueError, "credential reference"):
                profile.create_client(base_url=BINDING["base_url"], api_key="other-key")
            with self.assertRaisesRegex(ValueError, "endpoint"):
                profile.create_client(base_url="https://another.invalid/v1", api_key="unit-test-only")

    def test_different_api_or_rebinding_an_owned_home_is_refused(self):
        with self.assertRaisesRegex(ValueError, "chat-completions"):
            hermes_profile.validated_binding({**BINDING, "api": "responses"})
        with tempfile.TemporaryDirectory() as temporary:
            home = Path(temporary)
            hermes_profile.install_profile(home, BINDING)
            hermes_profile.install_profile(home, BINDING)
            with self.assertRaisesRegex(ValueError, "different model-binding"):
                hermes_profile.install_profile(home, {**BINDING, "model": "other/model"})
