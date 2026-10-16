# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D103

"""The browser profile's supporting pieces (plan ``ui-transport-plan.md``, sections 4.3, 4.6, 4.7).

* the browser pipeline's processors and the Smart Turn branch (off for ``eval``);
* the deterministic speech platform remap (decision D2), a no-op for ``singlegpu`` and unset;
* the tools: ``tools.yaml`` and the handlers copied from the Frontend/Backend Agent.
"""

from __future__ import annotations

import asyncio
import os
import unittest
from dataclasses import replace
from typing import Any
from unittest import mock

import yaml
from _fba_golden import pinned_environment
from _fba_pipecat_fakes import EnergyVADAnalyzer, FakeWebSocket, StubRecognizer, StubSTT, ToneSynthesizer, ToneTTS

from examples.frontend_backend_agent import tool_handlers, tools
from examples.frontend_backend_agent.adapters.pipecat import pipeline as pipeline_module
from examples.frontend_backend_agent.adapters.pipecat import ui_session
from examples.frontend_backend_agent.adapters.pipecat.pipeline import SpeechStack
from examples.frontend_backend_agent.adapters.pipecat.runtime import PipecatRuntime
from examples.frontend_backend_agent.adapters.pipecat.session import PipecatSession
from examples.frontend_backend_agent.adapters.pipecat.settings import load_pipecat_settings
from examples.frontend_backend_agent.core.agent.factory import ServerOptions
from examples.frontend_backend_agent.core.agent.messages import ToolCall
from examples.frontend_backend_agent.core.config.profiles import (
    DEFAULT_PROFILE,
    REALTIME_PROFILE,
    apply_speech_platform,
    profile_path,
)
from examples.frontend_backend_agent.core.config.voice import load_voice_config
from examples.frontend_backend_agent.core.tools.tools import ToolRegistry

UI_PROCESSORS = [
    "PipelineSource",
    "FastAPIWebsocketInputTransport",
    "RTVIProcessor",
    "VadParamsGate",
    "VADProcessor",
    "InputCursor",
    "AsrErrorTap",
    "StubSTT",
    "UserTurnProcessor",
    "FbaTurnBridge",
    "ToneTTS",
    "OutputTagger",
    "RealtimeFrameFilter",
    "FastAPIWebsocketOutputTransport",
    "PipelineSink",
]


def _speech(_session: Any) -> SpeechStack:
    return SpeechStack(vad=EnergyVADAnalyzer(), stt=StubSTT(StubRecognizer()), tts=ToneTTS(ToneSynthesizer()))


def _runtime(profile: str, *, smart_turn: bool | None = None) -> PipecatRuntime:
    settings = load_pipecat_settings("default.yaml" if profile == DEFAULT_PROFILE else "eval.yaml")
    if smart_turn is not None:
        settings = replace(settings, turn=replace(settings.turn, smart_turn=smart_turn))
    return PipecatRuntime(
        load_voice_config(profile_path(profile)),
        profile=profile,
        settings=settings,
        options=ServerOptions(stub_speech=True, stub_agent="scripted"),
        speech_factory=_speech,
    )


class UiPipelineTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        env = pinned_environment()
        env.__enter__()
        self.addCleanup(env.__exit__, None, None, None)

    def _ui_session(self, runtime: PipecatRuntime) -> ui_session.UiSession:
        from types import SimpleNamespace  # noqa: PLC0415

        from examples.shared.pipeline_utils import create_transport  # noqa: PLC0415

        args = SimpleNamespace(websocket=FakeWebSocket(), body={"protocol": "rtvi"})
        return ui_session.UiSession(
            runtime=ui_session._session_runtime(runtime, runtime.config),
            transport=create_transport(args),
            speech=_speech,
        )

    async def test_ui_processors(self) -> None:
        session = self._ui_session(_runtime(DEFAULT_PROFILE, smart_turn=False))
        processors = [type(p).__name__ for p in session.pipeline.worker._pipeline.processors]
        names = [name for name in processors if name not in ("Pipeline",)]
        self.assertEqual([n for n in names if n in UI_PROCESSORS], [n for n in UI_PROCESSORS if n in names])
        flat = [type(p).__name__ for p in session.pipeline.pipeline.processors]
        self.assertEqual(flat, [n for n in UI_PROCESSORS if n != "RTVIProcessor"])

    async def test_smart_turn_branch(self) -> None:
        stop = mock.Mock(name="smart-turn-stop")
        with mock.patch(
            "examples.shared.pipeline_utils.build_smart_turn_stop_strategies", return_value=[stop]
        ) as build:
            session = self._ui_session(_runtime(DEFAULT_PROFILE))
        build.assert_called_once_with(wait_for_transcript=False)
        self.assertTrue(session.settings.summary()["smart_turn"])
        stops = session.pipeline.user_turns._user_turn_controller._user_turn_strategies.stop
        self.assertEqual(stops, [stop])

    async def test_realtime_never_uses_smart_turn(self) -> None:
        wrapped = pipeline_module.SpeechTimeoutUserTurnStopStrategy
        with (
            mock.patch("examples.shared.pipeline_utils.build_smart_turn_stop_strategies") as build,
            mock.patch.object(pipeline_module, "SpeechTimeoutUserTurnStopStrategy", wraps=wrapped) as speech_timeout,
        ):
            PipecatSession(runtime=_runtime(REALTIME_PROFILE), websocket=FakeWebSocket(), speech=_speech)  # type: ignore[arg-type]
        build.assert_not_called()
        speech_timeout.assert_called_once()


class SpeechPlatformTests(unittest.TestCase):
    """Decision D2: Realtime and UI sessions follow the deployment platform for ASR/TTS."""

    def setUp(self) -> None:
        for name in ("_filter_reachable_entries", "_first_reachable_variant", "_load_effective_services_catalog"):
            patcher = mock.patch(f"utils.{name}", side_effect=AssertionError("no reachability probing"))
            patcher.start()
            self.addCleanup(patcher.stop)
        env = pinned_environment()
        env.__enter__()
        self.addCleanup(env.__exit__, None, None, None)
        self.config = load_voice_config(profile_path(REALTIME_PROFILE))

    def test_single_gpu_unset_and_unknown_return_the_same_config(self) -> None:
        for platform in (None, "", "singlegpu", "jetson"):
            env = {} if platform is None else {"REALTIME_SERVICE_PLATFORM": platform}
            with self.subTest(platform=platform):
                self.assertIs(apply_speech_platform(self.config, env), self.config)

    def test_server_reads_the_server_section(self) -> None:
        config = apply_speech_platform(self.config, {"REALTIME_SERVICE_PLATFORM": "server"})
        self.assertEqual(config.asr.endpoint.server, "nemotron-asr-streaming-english:50052")
        self.assertEqual(config.tts.endpoint.server, "magpie-multilingual-tts-service:50051")
        self.assertFalse(config.asr.endpoint.use_ssl)
        self.assertEqual(replace(config, asr=self.config.asr, tts=self.config.tts), self.config)

    def test_cloud_reads_the_cloud_catalog(self) -> None:
        env = {"REALTIME_SERVICE_PLATFORM": "cloud", "NVIDIA_API_KEY": "test-key"}
        config = apply_speech_platform(self.config, env)
        for endpoint in (config.asr.endpoint, config.tts.endpoint):
            self.assertEqual(endpoint.server, "grpc.nvcf.nvidia.com:443")
            self.assertTrue(endpoint.use_ssl)
            self.assertTrue(endpoint.function_id)
        self.assertEqual(config.tts.endpoint.voice_id, "Magpie-Multilingual.EN-US.Aria")

    def test_server_overrides_win_on_every_platform(self) -> None:
        env = {
            "REALTIME_SERVICE_PLATFORM": "server",
            "FBA_ASR_SERVER": "localhost:1",
            "FBA_TTS_SERVER": "localhost:2",
        }
        config = apply_speech_platform(self.config, env)
        self.assertEqual((config.asr.endpoint.server, config.tts.endpoint.server), ("localhost:1", "localhost:2"))


class ToolsTests(unittest.IsolatedAsyncioTestCase):
    def test_tools_yaml_and_handlers_match(self) -> None:
        catalog = yaml.safe_load(tools.TOOLS_FILE.read_text(encoding="utf-8"))
        self.assertEqual(list(catalog), [spec.name for spec in tools.TOOLS])
        self.assertEqual(set(catalog), set(tools.HANDLERS))
        for spec in tools.TOOLS:
            self.assertEqual(
                spec.to_openai_schema(),
                {**catalog[spec.name], "function": {**catalog[spec.name]["function"], "description": spec.description}},
            )
            self.assertNotIn("\n", spec.description)

    def test_the_default_profile_uses_these_tools(self) -> None:
        with pinned_environment():
            config = load_voice_config(profile_path(DEFAULT_PROFILE))
        self.assertEqual(config.tools.config_tool_specs, tuple(tools.TOOLS))

    async def test_registry_runs_a_handler_with_keyword_arguments(self) -> None:
        registry = ToolRegistry(tools.TOOLS)
        result = await registry.execute(ToolCall.create("calculate_bmi", {"weight_kg": 70, "height_m": 1.75}, id="c1"))
        self.assertFalse(result.is_error)
        self.assertIn('"bmi":22.86', result.content.replace(" ", ""))
        bad = await registry.execute(ToolCall.create("calculate_bmi", {"weight_kg": 0, "height_m": 1.75}, id="c2"))
        self.assertTrue(bad.is_error)

    async def test_a_missing_key_answers_unavailable(self) -> None:
        keys = ("WEATHERAPI_KEY", "FINNHUB_API_KEY", "PERPLEXITY_API_KEY")
        with mock.patch.dict(os.environ, dict.fromkeys(keys, "")):
            for name, arguments in (
                ("get_weather", {"city": "Pune"}),
                ("get_stock_price", {"company_name": "NVIDIA"}),
                ("web_search", {"query": "news"}),
            ):
                with self.subTest(tool=name):
                    result = await asyncio.wait_for(tool_handlers.__dict__[name](arguments), timeout=5)
                    self.assertEqual((result["status"], result["error_code"]), ("unavailable", "credential_missing"))


if __name__ == "__main__":
    unittest.main()
