# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D103

"""The OpenAI Realtime path is pinned to its state before the browser UI was added.

Plan ``misc/frontend-backend-verdict/ui-transport-plan.md`` (phase 0, condition C1). Every value
here was recorded from the code before the UI work started; none of them may change:

* the Realtime pipeline's processors, their order and the arguments they are built with;
* the Pipecat settings summary logged in ``session_start``;
* the ASR/TTS endpoints of the Tau profile, with and without ``FBA_ASR_SERVER``;
* what ``src/server.py`` hands the Realtime route for the FBV model (sanitized session config and
  server-side tools).
"""

from __future__ import annotations

import hashlib
import os
import unittest
from typing import Any
from unittest import mock

from _fbv_golden import pinned_environment
from _fbv_pipecat_fakes import (
    EnergyVADAnalyzer,
    FakeWebSocket,
    StubRecognizer,
    StubSTT,
    ToneSynthesizer,
    ToneTTS,
)

from examples.frontend_backend_verdict.adapters.pipecat import pipeline as pipeline_module
from examples.frontend_backend_verdict.adapters.pipecat import runtime as runtime_module
from examples.frontend_backend_verdict.adapters.pipecat.pipeline import SpeechStack
from examples.frontend_backend_verdict.adapters.pipecat.runtime import PipecatRuntime
from examples.frontend_backend_verdict.adapters.pipecat.session import PipecatSession
from examples.frontend_backend_verdict.adapters.pipecat.settings import load_pipecat_settings
from examples.frontend_backend_verdict.core.agent.factory import ServerOptions
from examples.frontend_backend_verdict.core.config.profiles import REALTIME_PROFILE, profile_path
from examples.frontend_backend_verdict.core.config.voice import load_voice_config

FBV = "frontend-backend-verdict-agent"

REALTIME_PROCESSORS = [
    "PipelineSource",
    "FastAPIWebsocketInputTransport",
    "VADProcessor",
    "InputCursor",
    "AsrErrorTap",
    "StubSTT",
    "UserTurnProcessor",
    "FbvTurnBridge",
    "ToneTTS",
    "OutputTagger",
    "FastAPIWebsocketOutputTransport",
    "PipelineSink",
]
TRANSPORT_PARAMS = {
    "audio_in_enabled": True,
    "audio_in_sample_rate": 16000,
    "audio_out_enabled": True,
    "audio_out_sample_rate": 22050,
    "audio_out_10ms_chunks": 2,
    "audio_out_auto_silence": False,
    "audio_out_end_silence_secs": 0,
    "audio_out_write_timeout_secs": 120.0,
    "serializer": "RealtimeSerializer",
    "session_timeout": None,
}
SETTINGS_SUMMARY = {
    "settings_file": "eval.yaml",
    "vad": {"stop_secs": 0.8, "audio_idle_timeout": 0.0},
    "turn": {"user_speech_timeout": 0.0, "stt_ttfs_p99": 3.0, "wait_for_transcript": False},
    "tts_synthesis_mode": "per_sentence",
    "smart_turn": False,
}
#: SHA-256 of the root ``prompts.yaml`` ``frontend`` content the Realtime route hands the pipeline.
FRONTEND_PROMPT_SHA256 = "1e58d7e99a01c5227286108221b69bb3d6d5ac47d2e11f63436d7527a613196c"


def _tau_config() -> Any:
    return load_voice_config(profile_path(REALTIME_PROFILE))


def _speech(_session: Any) -> SpeechStack:
    return SpeechStack(
        vad=EnergyVADAnalyzer(), stt=StubSTT(StubRecognizer(), ttfs_p99=3.0), tts=ToneTTS(ToneSynthesizer())
    )


def _record(target: str, calls: dict[str, list[dict[str, Any]]]) -> Any:
    original = getattr(pipeline_module, target)

    def wrapper(*args: Any, **kwargs: Any) -> Any:
        calls.setdefault(target, []).append(kwargs)
        return original(*args, **kwargs)

    return mock.patch.object(pipeline_module, target, side_effect=wrapper)


class RealtimePipelineBaselineTests(unittest.IsolatedAsyncioTestCase):
    async def test_realtime_pipeline_is_built_as_before(self) -> None:
        calls: dict[str, list[dict[str, Any]]] = {}
        names = (
            "VADProcessor",
            "SpeechTimeoutUserTurnStopStrategy",
            "UserTurnProcessor",
            "FastAPIWebsocketParams",
            "PipelineWorker",
            "PipelineParams",
        )
        with pinned_environment():
            config = _tau_config()
            patches = [_record(name, calls) for name in names]
            for patch in patches:
                patch.start()
            try:
                runtime = PipecatRuntime(
                    config,
                    profile=REALTIME_PROFILE,
                    settings=load_pipecat_settings("eval.yaml"),
                    options=ServerOptions(stub_speech=True, stub_agent="scripted"),
                    speech_factory=_speech,
                )
                session = PipecatSession(runtime=runtime, websocket=FakeWebSocket(), speech=_speech)  # type: ignore[arg-type]
            finally:
                for patch in patches:
                    patch.stop()
        pipe = session.pipeline
        self.assertEqual([type(p).__name__ for p in pipe.pipeline.processors], REALTIME_PROCESSORS)
        self.assertEqual([set(kw) - {"vad_analyzer"} for kw in calls["VADProcessor"]], [{"audio_idle_timeout"}])
        self.assertEqual(calls["VADProcessor"][0]["audio_idle_timeout"], 0.0)
        self.assertEqual(
            calls["SpeechTimeoutUserTurnStopStrategy"], [{"user_speech_timeout": 0.0, "wait_for_transcript": False}]
        )
        (turn_kwargs,) = calls["UserTurnProcessor"]
        self.assertEqual(set(turn_kwargs), {"user_turn_strategies"})
        strategies = turn_kwargs["user_turn_strategies"]
        self.assertEqual([type(s).__name__ for s in strategies.start], ["GatedVADStart", "GatedTranscriptionStart"])
        self.assertEqual([type(s).__name__ for s in strategies.stop], ["SpeechTimeoutUserTurnStopStrategy"])
        (params,) = calls["FastAPIWebsocketParams"]
        params = dict(params, serializer=type(params["serializer"]).__name__)
        self.assertEqual(params, TRANSPORT_PARAMS)
        (worker,) = calls["PipelineWorker"]
        self.assertEqual(
            {key: worker[key] for key in ("enable_rtvi", "idle_timeout_secs", "cancel_on_idle_timeout")},
            {"enable_rtvi": False, "idle_timeout_secs": None, "cancel_on_idle_timeout": False},
        )
        self.assertNotIn("observers", worker)
        self.assertEqual(calls["PipelineParams"], [{"audio_in_sample_rate": 16000, "audio_out_sample_rate": 22050}])
        # The profile's 800 ms silence reaches the VAD at session start (apply_turn_detection).
        self.assertEqual(pipe._vad_params.stop_secs, 0.8)
        self.assertEqual(runtime.settings.summary(), SETTINGS_SUMMARY)

    def test_settings_file_is_eval_whatever_the_environment(self) -> None:
        self.assertEqual(load_pipecat_settings().summary(), SETTINGS_SUMMARY)
        # FBV_PIPECAT_SETTINGS no longer selects the file: a leftover value cannot change the eval timing.
        env = {"FBV_PIPECAT_SETTINGS": "default.yaml", "UVICORN_WS_PING_INTERVAL": "0"}
        with pinned_environment(), mock.patch.dict(os.environ, env):
            self.assertEqual(load_pipecat_settings().summary(), SETTINGS_SUMMARY)
            with mock.patch.dict(runtime_module._RUNTIMES, clear=True):
                runtime = runtime_module.runtime_for({})
        self.assertEqual(runtime.profile, REALTIME_PROFILE)
        self.assertEqual(runtime.settings.summary(), SETTINGS_SUMMARY)

    def test_realtime_runtime_is_eval_whatever_fbv_profile(self) -> None:
        for value in ("", "default", "eval"):
            env = {"UVICORN_WS_PING_INTERVAL": "0", "FBV_PROFILE": value}
            with self.subTest(FBV_PROFILE=value), pinned_environment(), mock.patch.dict(os.environ, env):
                with mock.patch.dict(runtime_module._RUNTIMES, clear=True):
                    runtime = runtime_module.runtime_for({})
                self.assertEqual(runtime.profile, REALTIME_PROFILE)
                self.assertEqual(runtime.settings.summary(), SETTINGS_SUMMARY)
                self.assertEqual(runtime.config.turn_detection.silence_duration_ms, 800)
                self.assertEqual(runtime.config.tools.source, "client")
                self.assertEqual(runtime.config.agent.prompts_path.name, "tau.yaml")


class RealtimeSpeechEndpointBaselineTests(unittest.TestCase):
    def _endpoints(self, **env: str) -> list[tuple[Any, ...]]:
        with pinned_environment(), mock.patch.dict(os.environ, env):
            config = _tau_config()
        return [
            (e.kind, e.server, e.model, e.function_id, e.language_code, e.voice_id, e.use_ssl)
            for e in (config.asr.endpoint, config.tts.endpoint)
        ]

    def test_single_gpu_endpoints(self) -> None:
        self.assertEqual(
            self._endpoints(),
            [
                ("asr", "nemo-speech:50051", "nemotron-speech-streaming-en-0.6b", "", "en-US", "", False),
                ("tts", "nemo-speech:50051", "magpie-tts-multilingual", "", "en-US", "John", False),
            ],
        )

    def test_server_overrides_still_win(self) -> None:
        endpoints = self._endpoints(FBA_ASR_SERVER="localhost:50051", FBA_TTS_SERVER="localhost:50052")
        self.assertEqual([e[1] for e in endpoints], ["localhost:50051", "localhost:50052"])


class RealtimeRouteBaselineTests(unittest.TestCase):
    """What ``src/server.py`` gives the Realtime route for the FBV model (registry and prompt catalog)."""

    def test_sanitized_session_config_and_server_tools(self) -> None:
        with pinned_environment(), mock.patch.dict(os.environ, {"REALTIME_SERVICE_PLATFORM": "singlegpu"}):
            import server  # noqa: PLC0415

            config = server._sanitize_realtime_config({"pipeline_mode": FBV, "prompt_key": "frontend"}, FBV)
            tools = server._resolve_realtime_server_tools(config, FBV)
            schemas = server._resolve_realtime_server_tool_schemas(config, FBV)
        prompt = config.pop("prompt_content")
        self.assertEqual(hashlib.sha256(prompt.encode("utf-8")).hexdigest(), FRONTEND_PROMPT_SHA256)
        self.assertEqual(
            config,
            {
                "asr_model": "nemotron-speech-streaming-en-0.6b",
                "asr_server": "localhost:50051",
                "base_url": "https://inference-api.nvidia.com/v1",
                "extra_params": '{"extra_body":{"chat_template_kwargs":{"enable_thinking":false}}}',
                "max_tokens": 1024,
                "model_id": "nvidia/nvidia/nemotron-3.5-lightning",
                "pipeline_mode": FBV,
                "prompt_key": "frontend",
                "temperature": 0.0,
                "tools": [],
                "tts_model": "magpie-tts-multilingual",
                "tts_server": "localhost:50051",
                "tts_synthesis_mode": "stitched",
                "tts_voice_id": "John",
            },
        )
        self.assertEqual(tools, [])
        self.assertEqual(schemas, [])


if __name__ == "__main__":
    unittest.main()
