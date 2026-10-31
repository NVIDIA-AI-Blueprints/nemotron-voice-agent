# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D100, D101, D102, D103, D107

"""FDA as a registered example: registry entry, catalogs, ``FDA_PROFILE`` and ``bot()`` routing (steps 2-4)."""

from __future__ import annotations

import os
import unittest
from types import SimpleNamespace
from unittest import mock

import yaml
from _pipecat_fakes import PACKAGE, REPO_ROOT

import examples_registry
from examples.frontend_delegation_agent.adapters.pipecat import bot as bot_module
from examples.frontend_delegation_agent.adapters.pipecat import runtime as runtime_module
from examples.frontend_delegation_agent.adapters.pipecat.profiles import (
    DEFAULT_PROFILE,
    PROFILES,
    REALTIME_PROFILE,
    ProfileError,
    keepalive_mismatch,
    profile_name,
)
from examples.frontend_delegation_agent.adapters.pipecat.settings import (
    VAD_STOP_SECS_ENV,
    PipecatSettingsError,
    load_pipecat_settings,
    with_vad_stop_override,
)
from examples.frontend_delegation_agent.core.config import PROACTIVE_MAX_PER_RUN_ENV, DelegationConfigError
from examples.frontend_delegation_agent.core.text.demo_tools import TOOLS
from realtime.gateway import _SESSION_OWNED_PIPELINES

FDA = "frontend-delegation-agent"
MODEL = "nvidia/nemotron-realtime-frontend-delegation"
FBA_DIR = REPO_ROOT / "src" / "examples" / "frontend_backend_agent"


class RegistryTests(unittest.TestCase):
    def test_example_and_realtime_model_are_registered(self) -> None:
        example = examples_registry.find(FDA)
        self.assertEqual(example["key"], FDA)
        self.assertEqual(example["slots"], ["llm", "asr", "tts"])
        self.assertIs(examples_registry.resolve_bot(example), bot_module.bot)
        profiles = examples_registry.realtime_model_profiles()
        self.assertEqual(profiles[MODEL]["pipeline_mode"], FDA)
        self.assertIn(FDA, _SESSION_OWNED_PIPELINES)

    def test_tools_yaml_lists_the_demo_tools(self) -> None:
        catalog = yaml.safe_load((PACKAGE / "tools.yaml").read_text(encoding="utf-8"))
        expected = {
            tool.name: {"name": tool.name, "description": tool.description, "parameters": tool.parameters}
            for tool in TOOLS
        }
        self.assertEqual({name: entry["function"] for name, entry in catalog.items()}, expected)

    def test_public_prompt_exists(self) -> None:
        catalog = yaml.safe_load((PACKAGE / "prompts.yaml").read_text(encoding="utf-8"))
        self.assertIn("frontend", catalog)

    def test_speech_entries_are_the_ones_fda_used_before(self) -> None:
        # FDA read FBA's single-GPU and cloud catalogs before it had its own: speech must not change.
        fba_local = yaml.safe_load((FBA_DIR / "services.local.yaml").read_text(encoding="utf-8"))
        fda_local = yaml.safe_load((PACKAGE / "services.local.yaml").read_text(encoding="utf-8"))
        fba_cloud = yaml.safe_load((FBA_DIR / "services.cloud.yaml").read_text(encoding="utf-8"))
        fda_cloud = yaml.safe_load((PACKAGE / "services.cloud.yaml").read_text(encoding="utf-8"))
        for kind in ("asr", "tts"):
            with self.subTest(kind=kind):
                for section in ("singlegpu", "server"):
                    self.assertEqual(fda_local[section][kind], fba_local[section][kind])
                self.assertEqual(fda_cloud[kind], fba_cloud[kind])


class ProfileTests(unittest.TestCase):
    def test_profile_names(self) -> None:
        self.assertEqual(profile_name({}), DEFAULT_PROFILE)
        self.assertEqual(profile_name({"FDA_PROFILE": " eval "}), REALTIME_PROFILE)
        with self.assertRaises(ProfileError):
            profile_name({"FDA_PROFILE": "pipecat_demo.yaml"})

    def test_profile_files_are_todays(self) -> None:
        eval_profile, default = PROFILES[REALTIME_PROFILE], PROFILES[DEFAULT_PROFILE]
        self.assertEqual(
            (eval_profile.delegation, eval_profile.pipecat_settings, eval_profile.gateway),
            ("profiles/pipecat_realtime_eval.yaml", "adapters/pipecat.eval.yaml", "gateway.eval.yaml"),
        )
        self.assertEqual(
            (default.delegation, default.pipecat_settings, default.gateway),
            ("profiles/pipecat_demo.yaml", "adapters/pipecat.yaml", "gateway.yaml"),
        )
        for profile in PROFILES.values():
            self.assertTrue(profile.delegation_path.is_file())

    def test_keepalive_must_be_off(self) -> None:
        self.assertTrue(keepalive_mismatch(0, {}))
        self.assertTrue(keepalive_mismatch(0, {"UVICORN_WS_PING_INTERVAL": "20"}))
        self.assertEqual(keepalive_mismatch(0, {"UVICORN_WS_PING_INTERVAL": "0"}), "")


class RealtimeRuntimeTests(unittest.TestCase):
    def setUp(self) -> None:
        patcher = mock.patch.object(runtime_module, "_REALTIME", {})
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_realtime_sessions_always_run_eval(self) -> None:
        env = {
            "UVICORN_WS_PING_INTERVAL": "0",
            "FDA_PROFILE": "default",
            VAD_STOP_SECS_ENV: "",
            PROACTIVE_MAX_PER_RUN_ENV: "",
        }
        with mock.patch.dict(os.environ, env):
            os.environ.pop("FDA_PROACTIVE_AFTER_S", None)  # restored on exit; unset means the 15 s default
            runtime = runtime_module.realtime_runtime()
        self.assertEqual(runtime.profile, REALTIME_PROFILE)
        proactive = runtime.config.output.proactive_status
        self.assertEqual((proactive.after_s, proactive.max_per_run), (15.0, 1))
        self.assertIn(PROFILES[REALTIME_PROFILE].delegation_path, runtime.config.source_files)
        self.assertEqual(runtime.settings.summary()["vad"], {"stop_secs": 0.8, "audio_idle_timeout": 0.0})
        self.assertEqual(runtime.slots.max_sessions, runtime.config.voice.server.max_sessions)

    def test_vad_stop_override_applies_to_realtime_sessions(self) -> None:
        env = {"UVICORN_WS_PING_INTERVAL": "0", "FDA_PROFILE": "eval", VAD_STOP_SECS_ENV: "0.5"}
        with mock.patch.dict(os.environ, env):
            runtime = runtime_module.realtime_runtime()
        # Only stop_secs changes; the other eval settings stay as the file sets them.
        self.assertEqual(runtime.settings.summary()["vad"], {"stop_secs": 0.5, "audio_idle_timeout": 0.0})
        self.assertEqual(runtime.settings.summary()["turn"], {"user_speech_timeout": 0.0})

    def test_big_bench_audio_proactive_status_applies_to_realtime_sessions(self) -> None:
        env = {
            "UVICORN_WS_PING_INTERVAL": "0",
            "FDA_PROFILE": "eval",
            "FDA_PROACTIVE_AFTER_S": "1.0",
            PROACTIVE_MAX_PER_RUN_ENV: "3",
        }
        with mock.patch.dict(os.environ, env):
            runtime = runtime_module.realtime_runtime()
        proactive = runtime.config.output.proactive_status
        self.assertEqual((proactive.enabled, proactive.after_s, proactive.max_per_run), (True, 1.0, 3))

    def test_invalid_proactive_override_refuses_realtime(self) -> None:
        env = {"UVICORN_WS_PING_INTERVAL": "0", PROACTIVE_MAX_PER_RUN_ENV: "lots"}
        with mock.patch.dict(os.environ, env), self.assertRaises(DelegationConfigError):
            runtime_module.realtime_runtime()

    def test_invalid_vad_stop_override_refuses_realtime(self) -> None:
        env = {"UVICORN_WS_PING_INTERVAL": "0", VAD_STOP_SECS_ENV: "fast"}
        with mock.patch.dict(os.environ, env), self.assertRaises(PipecatSettingsError):
            runtime_module.realtime_runtime()

    def test_server_pings_refuse_realtime(self) -> None:
        with mock.patch.dict(os.environ, {"UVICORN_WS_PING_INTERVAL": "20"}), self.assertRaises(ProfileError):
            runtime_module.realtime_runtime()


class BotRoutingTests(unittest.IsolatedAsyncioTestCase):
    async def test_realtime_goes_to_serve_realtime(self) -> None:
        args = SimpleNamespace(body={"protocol": "realtime"}, websocket=object())
        with mock.patch.object(bot_module, "serve_realtime", mock.AsyncMock()) as serve:
            await bot_module.bot(args)
        serve.assert_awaited_once_with(args)

    async def test_an_eval_deployment_refuses_browser_sessions(self) -> None:
        with mock.patch.dict(os.environ, {"FDA_PROFILE": "eval"}), self.assertRaisesRegex(RuntimeError, "Realtime"):
            await bot_module.bot(SimpleNamespace(body={"protocol": "rtvi"}))

    async def test_browser_sessions_go_to_the_ui(self) -> None:
        args = SimpleNamespace(body={"protocol": "rtvi"})
        with (
            mock.patch.dict(os.environ, {"FDA_PROFILE": ""}),
            mock.patch("examples.frontend_delegation_agent.adapters.pipecat.ui_session.serve_ui") as serve_ui,
        ):
            await bot_module.bot(args)
        serve_ui.assert_awaited_once_with(args)

    async def test_a_profile_error_is_a_realtime_server_error(self) -> None:
        sent: list[str] = []

        async def send_text(text: str) -> None:
            sent.append(text)

        websocket = SimpleNamespace(query_params={}, send_text=send_text, close=mock.AsyncMock())
        with mock.patch.object(runtime_module, "realtime_runtime", side_effect=ProfileError("pings on")):
            await bot_module.serve_realtime(SimpleNamespace(body={"protocol": "realtime"}, websocket=websocket))
        self.assertIn('"server_error"', sent[0])
        websocket.close.assert_awaited_once_with(code=1011, reason="profile error")


class VadStopOverrideTests(unittest.TestCase):
    """``FDA_VAD_STOP_SECS``: the Realtime sessions' end-of-turn silence (the Big Bench Audio overlay sets 0.5)."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.eval_settings = load_pipecat_settings(PROFILES[REALTIME_PROFILE].pipecat_settings)

    def test_unset_or_empty_keeps_the_eval_file_value(self) -> None:
        for environ in ({}, {VAD_STOP_SECS_ENV: ""}, {VAD_STOP_SECS_ENV: "  "}):
            with self.subTest(environ=environ):
                settings = with_vad_stop_override(self.eval_settings, environ)
                self.assertIs(settings, self.eval_settings)
                self.assertEqual(settings.vad.stop_secs, 0.8)

    def test_a_valid_value_replaces_stop_secs_only(self) -> None:
        for raw, expected in (("0.5", 0.5), (" 0.5 ", 0.5), ("5", 5.0), ("0.05", 0.05)):
            with self.subTest(raw=raw):
                settings = with_vad_stop_override(self.eval_settings, {VAD_STOP_SECS_ENV: raw})
                self.assertEqual(settings.vad.stop_secs, expected)
                self.assertEqual(settings.vad.audio_idle_timeout, self.eval_settings.vad.audio_idle_timeout)
                self.assertEqual(settings.turn, self.eval_settings.turn)
                self.assertEqual(settings.source, self.eval_settings.source)
        self.assertEqual(self.eval_settings.vad.stop_secs, 0.8)  # the loaded settings are not mutated

    def test_an_invalid_value_names_the_variable(self) -> None:
        for raw in ("fast", "0", "-0.5", "5.01", "nan", "inf"):
            with self.subTest(raw=raw), self.assertRaisesRegex(PipecatSettingsError, VAD_STOP_SECS_ENV):
                with_vad_stop_override(self.eval_settings, {VAD_STOP_SECS_ENV: raw})

    def test_the_settings_loader_ignores_it(self) -> None:
        # Browser UI sessions load their settings with load_pipecat_settings only.
        with mock.patch.dict(os.environ, {VAD_STOP_SECS_ENV: "0.5"}):
            self.assertIsNone(load_pipecat_settings(PROFILES[DEFAULT_PROFILE].pipecat_settings).vad.stop_secs)
            self.assertEqual(load_pipecat_settings(PROFILES[REALTIME_PROFILE].pipecat_settings).vad.stop_secs, 0.8)


if __name__ == "__main__":
    unittest.main()
