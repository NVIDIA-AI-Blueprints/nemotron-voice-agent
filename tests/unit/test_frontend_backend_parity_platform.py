# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

# ruff: noqa: D101, D102

"""Platform pieces of the generic parity work: route defaults, session metadata, readiness and neutrality."""

from __future__ import annotations

import asyncio
import os
import re
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

import examples_registry
import server
from examples.frontend_backend_agent.src.session_metadata import session_metadata
from realtime.controller import RealtimeSessionController, session_tools_sha256
from realtime.gateway import _validate_immutable_nvidia_patch
from realtime.protocol import RealtimeProtocolError, ServiceNotReadyError, services_not_ready_message
from realtime.session import CanonicalRealtimeSession, RealtimeSessionCapabilities

ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = ROOT / "src" / "examples" / "frontend_backend_agent"
VOICE = "Magpie-Multilingual.EN-US.Aria"


class RouteSessionDefaultTests(unittest.TestCase):
    def _session(self, silence_ms: int, *, honor_client: bool = True) -> CanonicalRealtimeSession:
        capabilities = RealtimeSessionCapabilities(
            voices=frozenset({VOICE}),
            default_silence_duration_ms=silence_ms,
            honor_client_silence_duration=honor_client,
        )
        return CanonicalRealtimeSession(model="m", voice=VOICE, capabilities=capabilities)

    def _silence(self, session: CanonicalRealtimeSession) -> int:
        return session.public_view()["audio"]["input"]["turn_detection"]["silence_duration_ms"]

    def test_the_route_default_applies_only_when_the_client_sets_none(self) -> None:
        session = self._session(800)
        self.assertEqual(self._silence(session), 800)

        session.apply_update({"audio": {"input": {"turn_detection": {"type": "server_vad", "threshold": 0.5}}}})
        self.assertEqual(self._silence(session), 800)

        for requested in (500, 800):
            session.apply_update(
                {"audio": {"input": {"turn_detection": {"type": "server_vad", "silence_duration_ms": requested}}}}
            )
            self.assertEqual(self._silence(session), requested)

    def test_a_server_owned_silence_ignores_the_client_value(self) -> None:
        session = self._session(800, honor_client=False)
        update = {"audio": {"input": {"turn_detection": {"type": "server_vad", "silence_duration_ms": 500}}}}
        with self.assertLogs_loguru() as messages:
            session.apply_update(update)
        self.assertEqual(self._silence(session), 800)
        self.assertTrue(any("client_silence_duration_ms=500 applied=800" in line for line in messages))

        session.apply_update({"audio": {"input": {"turn_detection": {"type": "server_vad", "threshold": 0.6}}}})
        turn_detection = session.public_view()["audio"]["input"]["turn_detection"]
        self.assertEqual((turn_detection["silence_duration_ms"], turn_detection["threshold"]), (800, 0.6))

    def test_a_server_owned_silence_still_rejects_an_invalid_patch(self) -> None:
        session = self._session(800, honor_client=False)
        with self.assertRaises(RealtimeProtocolError):
            session.apply_update({"audio": {"input": {"turn_detection": {"type": "server_vad", "threshold": "high"}}}})

    def assertLogs_loguru(self):  # noqa: N802 - mirrors assertLogs
        from contextlib import contextmanager

        from loguru import logger

        @contextmanager
        def capture():
            lines: list[str] = []
            sink = logger.add(lambda message: lines.append(str(message)), level="INFO")
            try:
                yield lines
            finally:
                logger.remove(sink)

        return capture()

    def test_the_gateway_applies_the_route_switch_to_the_session(self) -> None:
        from realtime.gateway import _controller_from_runtime

        update = {"audio": {"input": {"turn_detection": {"type": "server_vad", "silence_duration_ms": 500}}}}
        for honor, applied in ((False, 800), (True, 500), (None, 500)):
            turn_defaults: dict = {"silence_duration_ms": 800}
            if honor is not None:
                turn_defaults["honor_client_values"] = honor
            with self.subTest(honor=honor), patch.dict(os.environ, {"USE_SILERO_VAD_TURN_DETECTION": "true"}):
                controller = _controller_from_runtime(
                    {
                        "pipeline_mode": "generic-frontend-backend-agent",
                        "model_id": "m",
                        "tts_voice_id": VOICE,
                        "asr_model": "a",
                    },
                    server_tools=[],
                    delegate_tools=[],
                    session_defaults={"turn_detection": turn_defaults},
                )
                session = controller.session
                session.apply_update(update)
                self.assertEqual(self._silence(session), applied)

    def test_other_sessions_keep_500_ms(self) -> None:
        self.assertEqual(self._silence(CanonicalRealtimeSession(model="m", voice=VOICE)), 500)

    def test_only_the_generic_frontend_backend_route_sets_800_ms(self) -> None:
        profiles = examples_registry.realtime_model_profiles()
        self.assertEqual(
            profiles["nvidia/nemotron-realtime-generic-frontend-backend"]["session_defaults"],
            {"turn_detection": {"silence_duration_ms": 800, "honor_client_values": True}},
        )
        self.assertEqual(profiles["nvidia/nemotron-realtime-frontend-backend"]["session_defaults"], {})

    def test_the_honor_client_values_switch_loads_as_a_boolean(self) -> None:
        raw = {"turn_detection": {"silence_duration_ms": 800, "honor_client_values": False}}
        self.assertEqual(examples_registry._validate_realtime_session_defaults("m", raw), raw)

    def test_invalid_session_defaults_fail_registry_loading(self) -> None:
        for raw in (
            {"turn_detection": {"silence_duration_ms": -1}},
            {"turn_detection": {"silence_duration_ms": "800"}},
            {"turn_detection": {"threshold": 0.5}},
            {"turn_detection": {"honor_client_values": "false"}},
            {"turn_detection": {"honor_client_values": 0}},
            {"audio": {}},
            [],
        ):
            with self.subTest(raw=raw), self.assertRaises(RuntimeError):
                examples_registry._validate_realtime_session_defaults("m", raw)


class SessionMetadataTests(unittest.TestCase):
    def test_metadata_names_the_build_thinker_and_switches(self) -> None:
        config = {
            "domain_profile": "generic",
            "thinker_llm_id": "nemotron-super-reasoning",
            "thinker_model_id": "nvidia/nemotron-3-super-120b-a12b",
            "thinker_max_tokens": "4096",
            "thinker_extra_params": (
                '{"extra_body":{"chat_template_kwargs":{"enable_thinking":true},"reasoning_budget":1024}}'
            ),
        }
        with patch.dict(
            os.environ,
            {"AGENT_GIT_SHA": "abc1234", "GENERIC_THINKER_REASONING_BUDGET": "", "GENERIC_THINKER_MAX_TOKENS": ""},
        ):
            metadata = session_metadata(config)

        self.assertEqual(metadata["agent_git_sha"], "abc1234")
        self.assertEqual(
            metadata["thinker"],
            {
                "llm_id": "nemotron-super-reasoning",
                "model_id": "nvidia/nemotron-3-super-120b-a12b",
                "max_tokens": 4096,
                "reasoning_budget": 1024,
                "reasoning": True,
            },
        )
        self.assertTrue(metadata["flags"]["FRONTEND_BACKEND_NORMALIZATION"])

    def test_generic_env_overrides_are_reported_and_airline_ignores_them(self) -> None:
        with patch.dict(os.environ, {"GENERIC_THINKER_REASONING_BUDGET": "2048", "GENERIC_THINKER_MAX_TOKENS": "6000"}):
            generic = session_metadata({"domain_profile": "generic", "thinker_max_tokens": "4096"})
            airline = session_metadata({"domain_profile": "airline", "thinker_max_tokens": "4096"})

        self.assertEqual((generic["thinker"]["max_tokens"], generic["thinker"]["reasoning_budget"]), (6000, 2048))
        self.assertEqual((airline["thinker"]["max_tokens"], airline["thinker"]["reasoning_budget"]), (4096, None))

    def test_the_server_adds_metadata_that_a_client_cannot_supply(self) -> None:
        config = server._sanitize_session_config(
            {"pipeline_mode": "generic-frontend-backend-agent", "agent_metadata": {"agent_git_sha": "forged"}}
        )
        self.assertNotEqual(config["agent_metadata"]["agent_git_sha"], "forged")

        plain = server._sanitize_session_config({"pipeline_mode": "generic-assistant"})
        self.assertNotIn("agent_metadata", plain)

    def test_session_updated_publishes_metadata_and_a_tools_fingerprint(self) -> None:
        tool = {"type": "function", "name": "lookup_member", "description": "Look up.", "parameters": {}}
        controller = RealtimeSessionController(
            model="m",
            voice=VOICE,
            runtime_config={"agent_metadata": {"agent_git_sha": "abc1234"}},
            capabilities=RealtimeSessionCapabilities(voices=frozenset({VOICE})),
        )
        controller.apply_session_update({"tools": [tool]})

        agent = controller.public_session()["nvidia"]["agent"]

        self.assertEqual(agent["agent_git_sha"], "abc1234")
        self.assertEqual(agent["session_tools_sha256"], session_tools_sha256([tool]))

    def test_a_client_echoing_the_read_only_metadata_is_not_rejected(self) -> None:
        current = {"nvidia": {"pipeline_mode": "p", "agent": {"agent_git_sha": "abc1234"}}}

        _validate_immutable_nvidia_patch({"agent": {"anything": True}, "pipeline_mode": "p"}, current)
        with self.assertRaises(RealtimeProtocolError):
            _validate_immutable_nvidia_patch({"pipeline_mode": "other"}, current)


class ReadinessTests(unittest.IsolatedAsyncioTestCase):
    async def test_a_probe_that_fails_once_is_retried(self) -> None:
        attempts = 0

        async def check() -> None:
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise RuntimeError("Selected TTS service is still starting.")

        with patch.object(server, "_READINESS_RETRY_BUDGET_SECS", 2.0), patch.object(asyncio, "sleep", _no_sleep):
            await server._ready_with_retry("TTS", check, retry=True)
        self.assertEqual(attempts, 2)

    async def test_a_persistent_failure_names_the_service_without_its_address(self) -> None:
        async def check() -> None:
            raise RuntimeError("Selected TTS service is not available at grpc.internal.example:443.")

        with (
            patch.object(server, "_READINESS_RETRY_BUDGET_SECS", 0.0),
            self.assertRaises(ServiceNotReadyError) as caught,
        ):
            await server._ready_with_retry("TTS", check, retry=True)

        message = services_not_ready_message(caught.exception)
        self.assertEqual(message, "One or more required Realtime services are not ready: TTS (unavailable)")
        self.assertNotIn("grpc.internal", message)

    async def test_without_retry_one_attempt_is_made(self) -> None:
        attempts = 0

        async def check() -> None:
            nonlocal attempts
            attempts += 1
            raise RuntimeError("timed out")

        with self.assertRaises(ServiceNotReadyError) as caught:
            await server._ready_with_retry("ASR", check, retry=False)
        self.assertEqual((attempts, caught.exception.reason), (1, "timeout"))


async def _no_sleep(_seconds: float) -> None:
    return None


#: Benchmark vocabulary that must not appear in the domain-neutral generic code paths.
_BENCHMARK_TERMS = re.compile(
    r"(?i)\b(airline|reservations?|flights?|baggage|telecom|retail|passengers?|get_user_details|"
    r"sara_doe_496|omar_davis_3817|8JX2WO|ZFA04Y|cancel_reservation)\b"
)


class DomainNeutralityTests(unittest.TestCase):
    """The generic domain carries no benchmark names, codes or tools (code, prompts, tests)."""

    def _files(self) -> list[Path]:
        files = [
            *sorted((EXAMPLE / "generic").glob("*.py")),
            *sorted((EXAMPLE / "src" / "normalization").glob("*.py")),
            EXAMPLE / "src" / "frontend_verdict.py",
            EXAMPLE / "src" / "history.py",
            EXAMPLE / "src" / "flags.py",
            EXAMPLE / "src" / "session_metadata.py",
            *sorted((ROOT / "tests" / "unit").glob("test_generic_fba_*.py")),
            ROOT / "tests" / "unit" / "test_frontend_backend_normalization.py",
            ROOT / "tests" / "unit" / "test_frontend_backend_generic_parity.py",
            ROOT / "tests" / "unit" / "test_realtime_delegation_protocol.py",
        ]
        return [path for path in files if path.exists()]

    def test_generic_code_and_tests_are_neutral(self) -> None:
        for path in self._files():
            with self.subTest(path=path.relative_to(ROOT)):
                self.assertEqual(_BENCHMARK_TERMS.findall(path.read_text(encoding="utf-8")), [])

    def test_generic_prompts_are_neutral(self) -> None:
        catalog = yaml.safe_load((EXAMPLE / "prompts.yaml").read_text(encoding="utf-8"))
        keys = [
            key
            for key in catalog
            if key.startswith("generic_") or key in {"frontend_verdict_talker", "backend_history_thinker"}
        ]
        self.assertIn("generic_thinker", keys)
        for key in keys:
            with self.subTest(prompt=key):
                self.assertEqual(_BENCHMARK_TERMS.findall(yaml.safe_dump(catalog[key])), [])


class EffectiveFlagTests(unittest.TestCase):
    def test_switches_that_need_history_report_off_without_it(self) -> None:
        from examples.frontend_backend_agent.src.flags import effective_flags

        with patch.dict(os.environ, {"FRONTEND_BACKEND_BACKEND_HISTORY": "false"}):
            flags = effective_flags()

        self.assertFalse(flags["FRONTEND_BACKEND_DIRECT_WRITE"])
        self.assertFalse(flags["FRONTEND_BACKEND_LATE_ANSWERS"])
        self.assertFalse(flags["FRONTEND_BACKEND_DONE_GUARD"])
        self.assertTrue(flags["FRONTEND_BACKEND_NORMALIZATION"])
