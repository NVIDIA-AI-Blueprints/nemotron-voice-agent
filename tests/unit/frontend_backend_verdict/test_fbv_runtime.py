# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D103

"""Profile selection, LLM endpoints, keepalive check, and admission of the Pipecat host."""

from __future__ import annotations

import asyncio
import json
import unittest
from dataclasses import replace
from unittest import mock

from _fbv_voice_fakes import base_config

from examples.frontend_backend_verdict.bridge import runtime as rt
from examples.frontend_backend_verdict.voice.config import load_voice_config
from examples.frontend_backend_verdict.voice.server import ServerOptions

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
    def test_default_profile_is_the_frontend_verdict_history_arm(self) -> None:
        self.assertEqual(rt.profile_name({}), "tau3_eval_frontend_verdict_speak_history")

    def test_unshipped_profiles_are_refused(self) -> None:
        for name in ("backend_only", "live_demo", "browser_demo", "cloud_speech", "voice_agent"):
            with self.subTest(name=name), self.assertRaises(rt.ProfileError):
                rt.profile_name({rt.PROFILE_ENV: name})

    def test_every_shipped_profile_has_the_prototype_tau3_turn_taking(self) -> None:
        for name in rt.SHIPPED_PROFILES:
            with self.subTest(profile=name):
                config = load_voice_config(rt.profile_path(name))
                self.assertEqual(config.turn_detection.silence_duration_ms, 800)
                self.assertFalse(config.turn_detection.honor_client_values)
                self.assertEqual(config.turn_detection.threshold, 0.5)
                self.assertEqual(config.turn_detection.prefix_padding_ms, 300)
                self.assertEqual(config.turn_detection.min_speech_ms, 120)
                self.assertEqual(config.server.ws_ping_interval_s, 0)

    def test_default_profile_values(self) -> None:
        config = load_voice_config(rt.profile_path(rt.DEFAULT_PROFILE))
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
            agent = load_voice_config(rt.profile_path(rt.DEFAULT_PROFILE)).agent
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
        config = load_voice_config(rt.profile_path(rt.DEFAULT_PROFILE))
        self.assertEqual(rt.keepalive_mismatch(config, _KEEPALIVE_OFF), "")

    def test_uvicorn_default_keepalive_is_reported(self) -> None:
        config = load_voice_config(rt.profile_path(rt.DEFAULT_PROFILE))
        message = rt.keepalive_mismatch(config, {})
        self.assertIn("UVICORN_WS_PING_INTERVAL=0", message)

    def test_runtime_for_refuses_a_mismatched_server(self) -> None:
        rt._RUNTIMES.clear()
        with self.assertRaises(rt.ProfileError):
            rt.runtime_for({}, {})


class AdmissionTests(unittest.IsolatedAsyncioTestCase):
    def _runtime(self, max_sessions: int = 1) -> rt.VerdictRuntime:
        config = base_config()
        config = replace(config, server=replace(config.server, warmup=False, max_sessions=max_sessions))
        return rt.VerdictRuntime(config, profile="test", options=ServerOptions(stub_speech=True, stub_agent="scripted"))

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
        runtime = rt.VerdictRuntime(config, profile="test", options=ServerOptions(stub_agent="scripted"))
        services = mock.Mock()
        services.warmup = mock.AsyncMock(side_effect=[RuntimeError("asr down"), None])
        runtime.state.services = services
        socket = _Socket()
        self.assertFalse(await runtime.admit(socket))
        self.assertEqual(socket.sent[0]["error"]["code"], "server_busy")
        self.assertEqual(socket.closed[0], 1013)
        self.assertTrue(await runtime.admit(_Socket()))

    async def test_concurrent_first_sessions_warm_up_once(self) -> None:
        config = replace(base_config(), server=replace(base_config().server, warmup=True))
        runtime = rt.VerdictRuntime(config, profile="test", options=ServerOptions(stub_agent="scripted"))
        services = mock.Mock()
        services.warmup = mock.AsyncMock(return_value=None)
        runtime.state.services = services
        results = await asyncio.gather(*(runtime.ensure_ready() for _ in range(5)))
        self.assertEqual(results, [True] * 5)
        services.warmup.assert_awaited_once()
