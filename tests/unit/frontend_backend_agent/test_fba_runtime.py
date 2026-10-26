# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D103

"""Profile selection, LLM endpoints, keepalive check, and admission of the Pipecat adapter (S12)."""

from __future__ import annotations

import asyncio
import json
import unittest
from dataclasses import replace
from types import SimpleNamespace
from typing import Any
from unittest import mock

from _fba_core_fakes import base_config

from examples.frontend_backend_agent.adapters.pipecat import runtime as prt
from examples.frontend_backend_agent.adapters.pipecat import warmup
from examples.frontend_backend_agent.core.agent.factory import ServerOptions
from examples.frontend_backend_agent.core.config import profiles as rt
from examples.frontend_backend_agent.core.config.voice import load_voice_config
from examples.frontend_backend_agent.core.errors import SpeechServiceError

_KEEPALIVE_OFF = {"UVICORN_WS_PING_INTERVAL": "0", "UVICORN_WS_PING_TIMEOUT": "20"}


class _Socket:
    def __init__(self) -> None:
        self.sent: list[dict] = []
        self.closed: tuple[int, str] | None = None

    async def send_text(self, text: str) -> None:
        self.sent.append(json.loads(text))

    async def close(self, code: int = 1000, reason: str = "") -> None:
        self.closed = (code, reason)


class ProfileTests(unittest.TestCase):
    def test_unset_profile_is_the_browser_profile(self) -> None:
        self.assertEqual(rt.profile_name({}), "default")

    def test_unshipped_profiles_are_refused(self) -> None:
        for name in ("backend_only", "live_demo", "browser_demo", "cloud_speech", "voice_agent", "old_eval"):
            with self.subTest(name=name), self.assertRaises(rt.ProfileError):
                rt.profile_name({rt.PROFILE_ENV: name})

    def test_eval_profile_has_the_benchmark_turn_taking(self) -> None:
        config = load_voice_config(rt.profile_path(rt.REALTIME_PROFILE))
        self.assertEqual(config.turn_detection.silence_duration_ms, 800)
        self.assertFalse(config.turn_detection.honor_client_values)
        self.assertEqual(config.turn_detection.threshold, 0.5)
        self.assertEqual(config.turn_detection.prefix_padding_ms, 300)

    def test_every_served_profile_turns_the_keepalive_off(self) -> None:
        # The uvicorn keepalive is process-wide and Realtime sessions (eval) need it off.
        for name in rt.SERVED_PROFILES:
            with self.subTest(profile=name):
                self.assertEqual(load_voice_config(rt.profile_path(name)).server.ws_ping_interval_s, 0)

    def test_browser_profile_values(self) -> None:
        config = load_voice_config(rt.profile_path(rt.DEFAULT_PROFILE))
        self.assertEqual(config.turn_detection.silence_duration_ms, 200)
        self.assertFalse(config.turn_detection.honor_client_values)
        self.assertTrue(config.protocol.greeting_enabled)
        self.assertEqual(config.protocol.seed_history_with_client_greeting, "")
        self.assertEqual(config.tools.source, "config")
        self.assertEqual(
            [tool.name for tool in config.tools.config_tool_specs],
            ["get_weather", "get_stock_price", "web_search", "calculate_bmi", "generate_random_number"],
        )
        self.assertTrue(config.normalization.transcript.enabled)
        self.assertFalse(config.normalization.tool_arguments.enabled)
        self.assertFalse(config.write_gate.enabled)
        self.assertEqual(config.barge_in.while_thinking, "frontend_verdict")
        self.assertEqual(config.filler.mode, "speak")

    def test_browser_profile_never_writes_the_eval_logs(self) -> None:
        env = {"FBA_VOICE_EVENT_LOG": "/tmp/events.jsonl", "FBA_FILLER_LOG": "/tmp/filler.jsonl"}
        with mock.patch.dict("os.environ", env):
            browser = load_voice_config(rt.profile_path(rt.DEFAULT_PROFILE))
            realtime = load_voice_config(rt.profile_path(rt.REALTIME_PROFILE))
        self.assertEqual((browser.logging.event_log, browser.filler.log_path), ("", ""))
        self.assertTrue(str(realtime.logging.event_log).endswith("events.jsonl"))
        self.assertTrue(str(realtime.filler.log_path).endswith("filler.jsonl"))

    def test_settings_file_per_profile(self) -> None:
        self.assertEqual(rt.pipecat_settings_file(rt.REALTIME_PROFILE), "eval.yaml")
        self.assertEqual(rt.pipecat_settings_file(rt.DEFAULT_PROFILE), "default.yaml")

    def test_eval_profile_values(self) -> None:
        config = load_voice_config(rt.profile_path(rt.REALTIME_PROFILE))
        self.assertEqual(config.barge_in.while_thinking, "frontend_verdict")
        self.assertEqual(config.barge_in.frontend_verdict.timeout_ms, 4000)
        self.assertTrue(config.barge_in.frontend_verdict.same_query_guard)
        self.assertEqual(config.filler.mode, "speak")
        self.assertEqual(config.filler.speak_after_ms, 300)
        history = config.agent.backend.conversation_history
        self.assertTrue(history.enabled)
        self.assertEqual(history.include, "full")
        self.assertTrue(config.normalization.transcript.enabled)
        self.assertTrue(config.normalization.tool_arguments.enabled)
        self.assertEqual(config.tools.source, "client")
        self.assertEqual(config.protocol.seed_history_with_client_greeting, "Hi! How can I help you today?")

    def test_prototype_llm_defaults_are_the_inference_hub_models(self) -> None:
        with mock.patch.dict("os.environ", {}, clear=False) as env:
            for key in ("FRONTEND_LLM_MODEL", "BACKEND_LLM_MODEL", "FRONTEND_LLM_BASE_URL", "BACKEND_LLM_BASE_URL"):
                env.pop(key, None)
            agent = load_voice_config(rt.profile_path(rt.REALTIME_PROFILE)).agent
        self.assertEqual(agent.frontend.llm.model, "nvidia/nvidia/nemotron-3.5-lightning")
        self.assertEqual(agent.frontend.llm.max_tokens, 1024)
        self.assertEqual(agent.frontend.llm.extra_body, {"chat_template_kwargs": {"enable_thinking": False}})
        self.assertEqual(agent.backend.llm.model, "nvidia/nvidia/nemotron-3-ultra")
        self.assertEqual(agent.backend.llm.max_tokens, 4096)
        self.assertEqual(
            agent.backend.llm.extra_body, {"chat_template_kwargs": {"enable_thinking": True}, "reasoning_budget": 1024}
        )


class EndpointTests(unittest.TestCase):
    def test_catalog_endpoints_replace_only_model_and_base_url(self) -> None:
        config = base_config()
        endpoints = rt.LLMEndpoints.from_body(
            {
                "model_id": "front/model",
                "base_url": "http://front/v1",
                "thinker_model_id": "back/model",
                "thinker_base_url": "http://back/v1",
            }
        )
        updated = rt.apply_llm_endpoints(config, endpoints)
        front, back = updated.agent.frontend.llm, updated.agent.backend.llm
        self.assertEqual((front.model, front.base_url), ("front/model", "http://front/v1"))
        self.assertEqual((back.model, back.base_url), ("back/model", "http://back/v1"))
        self.assertEqual(
            replace(front, model="", base_url=""), replace(config.agent.frontend.llm, model="", base_url="")
        )
        self.assertEqual(replace(back, model="", base_url=""), replace(config.agent.backend.llm, model="", base_url=""))

    def test_empty_body_keeps_the_prototype_endpoints(self) -> None:
        config = base_config()
        self.assertEqual(rt.apply_llm_endpoints(config, rt.LLMEndpoints.from_body({})), config)


class KeepaliveTests(unittest.TestCase):
    def test_matching_keepalive_passes(self) -> None:
        config = load_voice_config(rt.profile_path(rt.REALTIME_PROFILE))
        self.assertEqual(rt.keepalive_mismatch(config, _KEEPALIVE_OFF), "")

    def test_uvicorn_default_keepalive_is_reported(self) -> None:
        config = load_voice_config(rt.profile_path(rt.REALTIME_PROFILE))
        message = rt.keepalive_mismatch(config, {})
        self.assertIn("UVICORN_WS_PING_INTERVAL=0", message)

    def test_runtime_for_refuses_a_mismatched_server(self) -> None:
        with mock.patch.dict(prt._RUNTIMES, clear=True), self.assertRaises(rt.ProfileError):
            prt.runtime_for({}, {})

    def test_keepalive_values_parse_like_uvicorn_ws_ping_kwargs(self) -> None:
        self.assertEqual(rt._float_env({"K": "not-a-number"}, "K", 20.0), 20.0)
        self.assertEqual(rt._float_env({"K": "inf"}, "K", 20.0), 20.0)
        self.assertEqual(rt._float_env({"K": "-5"}, "K", 20.0), 0.0)
        self.assertEqual(rt._float_env({"K": " 7.5 "}, "K", 20.0), 7.5)
        self.assertEqual(rt._float_env({}, "K", 20.0), 20.0)


class AdmissionTests(unittest.IsolatedAsyncioTestCase):
    def _runtime(self, max_sessions: int = 1) -> prt.PipecatRuntime:
        config = base_config()
        config = replace(config, server=replace(config.server, warmup=False, max_sessions=max_sessions))
        return prt.PipecatRuntime(
            config, profile="test", options=ServerOptions(stub_speech=True, stub_agent="scripted")
        )

    async def test_capacity_follows_the_prototype_rule(self) -> None:
        runtime = self._runtime(max_sessions=1)
        first, second = _Socket(), _Socket()
        self.assertTrue(await runtime.admit(first))
        self.assertFalse(await runtime.admit(second))
        self.assertEqual(second.sent[0]["error"]["code"], "server_busy")
        self.assertEqual(second.closed, (1013, "server at capacity (server.max_sessions)"))
        runtime.release()
        self.assertTrue(await runtime.admit(_Socket()))

    async def test_failed_warmup_is_server_busy_and_retried(self) -> None:
        config = replace(base_config(), server=replace(base_config().server, warmup=True))
        runtime = prt.PipecatRuntime(config, profile="test", options=ServerOptions(stub_agent="scripted"))
        warm_up = mock.AsyncMock(side_effect=[RuntimeError("asr down"), None])
        socket = _Socket()
        with mock.patch.object(prt, "warm_up", warm_up):
            self.assertFalse(await runtime.admit(socket))
            self.assertEqual(socket.sent[0]["error"]["code"], "server_busy")
            self.assertEqual(socket.closed[0], 1013)
            self.assertEqual(runtime.state.startup_error, "asr down")
            self.assertTrue(await runtime.admit(_Socket()))
        self.assertEqual(warm_up.await_count, 2)

    async def test_concurrent_first_sessions_warm_up_once(self) -> None:
        config = replace(base_config(), server=replace(base_config().server, warmup=True))
        runtime = prt.PipecatRuntime(config, profile="test", options=ServerOptions(stub_agent="scripted"))
        warm_up = mock.AsyncMock(return_value=None)
        with mock.patch.object(prt, "warm_up", warm_up):
            results = await asyncio.gather(*(runtime.ensure_ready() for _ in range(5)))
        self.assertEqual(results, [True] * 5)
        warm_up.assert_awaited_once()

    async def test_stub_speech_skips_the_warmup(self) -> None:
        config = replace(base_config(), server=replace(base_config().server, warmup=True))
        runtime = prt.PipecatRuntime(
            config, profile="test", options=ServerOptions(stub_speech=True, stub_agent="scripted")
        )
        warm_up = mock.AsyncMock(return_value=None)
        with mock.patch.object(prt, "warm_up", warm_up):
            self.assertTrue(await runtime.ensure_ready())
        warm_up.assert_not_awaited()


class _FakeRiva:
    """Enough of ``riva.client`` for the warm-up probes."""

    def __init__(self, *, tts_audio: bytes = b"\x00\x01", asr_error: Exception | None = None) -> None:
        self.auth: list[tuple] = []
        self.asr_configs: list[Any] = []
        self.tts_calls: list[tuple[str, dict]] = []
        self._tts_audio = tts_audio
        self._asr_error = asr_error
        fake = self

        class ASRService:
            def __init__(self, auth: Any) -> None:
                pass

            def streaming_response_generator(self, *, audio_chunks: Any, streaming_config: Any) -> Any:
                fake.asr_configs.append(streaming_config)
                if fake._asr_error is not None:
                    raise fake._asr_error
                yield from ()

        class SpeechSynthesisService:
            def __init__(self, auth: Any) -> None:
                pass

            def synthesize(self, text: str, **kwargs: Any) -> Any:
                fake.tts_calls.append((text, kwargs))
                return SimpleNamespace(audio=fake._tts_audio)

        self.module = SimpleNamespace(
            Auth=lambda *args: self.auth.append(args) or object(),
            ASRService=ASRService,
            SpeechSynthesisService=SpeechSynthesisService,
            StreamingRecognitionConfig=lambda **kwargs: SimpleNamespace(**kwargs),
            RecognitionConfig=lambda **kwargs: SimpleNamespace(**kwargs),
            AudioEncoding=SimpleNamespace(LINEAR_PCM="LINEAR_PCM"),
        )


class WarmupTests(unittest.IsolatedAsyncioTestCase):
    """The speech warm-up probes ASR, then TTS, on the profile's endpoints (S12)."""

    async def _warm(self, fake: _FakeRiva) -> None:
        import riva  # noqa: PLC0415

        with (
            mock.patch.object(riva, "client", fake.module, create=True),
            mock.patch.dict("sys.modules", {"riva.client": fake.module}),
        ):
            await warmup.warm_up(base_config())

    async def test_probes_asr_then_tts_on_the_profile_endpoints(self) -> None:
        fake = _FakeRiva()
        config = base_config()
        await self._warm(fake)
        asr, tts = config.asr.endpoint, config.tts.endpoint
        self.assertEqual(fake.auth[0], (None, asr.use_ssl, asr.server, asr.metadata()))
        self.assertEqual(fake.auth[1], (None, tts.use_ssl, tts.server, tts.metadata()))
        self.assertEqual(fake.asr_configs[0].config.sample_rate_hertz, 16000)
        self.assertEqual(fake.tts_calls[0][0], "Hello.")
        self.assertEqual(fake.tts_calls[0][1]["sample_rate_hz"], config.tts.sample_rate)

    async def test_asr_failure_names_the_endpoint(self) -> None:
        with self.assertRaises(SpeechServiceError) as caught:
            await self._warm(_FakeRiva(asr_error=RuntimeError("connection refused")))
        self.assertIn("ASR warm-up failed", str(caught.exception))
        self.assertIn(base_config().asr.endpoint.server, str(caught.exception))

    async def test_tts_without_audio_fails(self) -> None:
        with self.assertRaises(SpeechServiceError) as caught:
            await self._warm(_FakeRiva(tts_audio=b""))
        self.assertIn("TTS warm-up failed", str(caught.exception))
