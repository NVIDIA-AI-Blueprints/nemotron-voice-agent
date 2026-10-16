# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D103

"""Browser UI features added before the Frontend/Backend Agent is removed (plan section 5).

* 5.2: tracing: ``ENABLE_TRACING`` traces UI sessions; Realtime sessions also need
  ``FBV_REALTIME_TRACING`` and are never traced in an eval-mode container (the eval lock).
* 5.3: one RTVI ``metric-group`` message per answered turn (latency breakdown and LLM usage).
* 5.4: the UI's ``set-voice`` message changes the TTS voice from the next sentence on.
* 5.5: ``ENABLE_WELCOME_MESSAGE`` turns the UI greeting off; unset keeps it.
"""

from __future__ import annotations

import os
import sys
import unittest
from dataclasses import replace
from types import SimpleNamespace
from typing import Any
from unittest import mock

from _fbv_golden import pinned_environment
from _fbv_pipecat_fakes import EnergyVADAnalyzer, FakeWebSocket, StubRecognizer, StubSTT, ToneSynthesizer, ToneTTS

from examples.frontend_backend_verdict.adapters.pipecat import pipeline as pipeline_module
from examples.frontend_backend_verdict.adapters.pipecat import switches, ui_metrics, ui_pipeline, ui_session
from examples.frontend_backend_verdict.adapters.pipecat.pipeline import SpeechStack
from examples.frontend_backend_verdict.adapters.pipecat.runtime import PipecatRuntime
from examples.frontend_backend_verdict.adapters.pipecat.session import PipecatSession
from examples.frontend_backend_verdict.adapters.pipecat.settings import load_pipecat_settings
from examples.frontend_backend_verdict.core.agent.factory import ServerOptions
from examples.frontend_backend_verdict.core.config.profiles import DEFAULT_PROFILE, REALTIME_PROFILE, profile_path
from examples.frontend_backend_verdict.core.config.voice import load_voice_config
from examples.frontend_backend_verdict.core.engine.event_log import EventLog


def _speech(_session: Any) -> SpeechStack:
    return SpeechStack(vad=EnergyVADAnalyzer(), stt=StubSTT(StubRecognizer()), tts=ToneTTS(ToneSynthesizer()))


def ui_session_for_test() -> ui_session.UiSession:
    from examples.shared.pipeline_utils import create_transport  # noqa: PLC0415

    settings = load_pipecat_settings("default.yaml")
    settings = replace(settings, turn=replace(settings.turn, smart_turn=False))
    runtime = PipecatRuntime(
        load_voice_config(profile_path(DEFAULT_PROFILE)),
        profile=DEFAULT_PROFILE,
        settings=settings,
        options=ServerOptions(stub_speech=True, stub_agent="scripted"),
        speech_factory=_speech,
    )
    args = SimpleNamespace(websocket=FakeWebSocket(), body={"protocol": "rtvi"})
    return ui_session.UiSession(
        runtime=ui_session._session_runtime(runtime, runtime.config),
        transport=create_transport(args),
        speech=_speech,
    )


#: A stand-in for ``src/tracing.py`` with tracing initialized (importing the real one sets up OTel).
TRACING_ON = SimpleNamespace(IS_TRACING_ENABLED=True)


def _worker_kwargs(module: Any, build: Any) -> dict[str, Any]:
    calls: list[dict[str, Any]] = []
    original = module.PipelineWorker

    def record(*args: Any, **kwargs: Any) -> Any:
        calls.append(kwargs)
        return original(*args, **kwargs)

    with mock.patch.object(module, "PipelineWorker", side_effect=record):
        build()
    (kwargs,) = calls
    return kwargs


def _realtime_session() -> None:
    runtime = PipecatRuntime(
        load_voice_config(profile_path(REALTIME_PROFILE)),
        profile=REALTIME_PROFILE,
        settings=load_pipecat_settings("eval.yaml"),
        options=ServerOptions(stub_speech=True, stub_agent="scripted"),
        speech_factory=_speech,
    )
    PipecatSession(runtime=runtime, websocket=FakeWebSocket(), speech=_speech)  # type: ignore[arg-type]


class TracingTests(unittest.IsolatedAsyncioTestCase):
    async def test_realtime_needs_both_switches(self) -> None:
        cases = {
            "unset": ({}, False),
            "ENABLE_TRACING only": ({"ENABLE_TRACING": "true"}, False),
            "FBV_REALTIME_TRACING only": ({"FBV_REALTIME_TRACING": "true"}, False),
            "both": ({"ENABLE_TRACING": "true", "FBV_REALTIME_TRACING": "true"}, True),
            "both, eval mode": (
                {"ENABLE_TRACING": "true", "FBV_REALTIME_TRACING": "true", "FBV_PROFILE": "eval"},
                False,
            ),
        }
        for name, (env, traced) in cases.items():
            with self.subTest(case=name), pinned_environment(), mock.patch.dict(os.environ, env):
                with mock.patch.dict(sys.modules, {"tracing": TRACING_ON}):
                    kwargs = _worker_kwargs(pipeline_module, _realtime_session)
                if traced:
                    self.assertIs(kwargs["enable_tracing"], True)
                else:
                    self.assertNotIn("enable_tracing", kwargs)

    async def test_ui_follows_enable_tracing(self) -> None:
        for env, traced in (({}, False), ({"ENABLE_TRACING": "true"}, True), ({"ENABLE_TRACING": "false"}, False)):
            with self.subTest(env=env), pinned_environment(), mock.patch.dict(os.environ, env):
                with mock.patch.dict(sys.modules, {"tracing": TRACING_ON}):
                    kwargs = _worker_kwargs(ui_pipeline, ui_session_for_test)
                if traced:
                    self.assertIs(kwargs["enable_tracing"], True)
                else:
                    self.assertNotIn("enable_tracing", kwargs)

    def test_tracing_is_not_imported_while_off(self) -> None:
        with pinned_environment(), mock.patch.dict(sys.modules, {"tracing": None}):
            # ``None`` in sys.modules makes any ``import tracing`` fail: off, nothing imports it.
            self.assertFalse(switches.ui_tracing())
            self.assertFalse(switches.realtime_tracing())

    def test_eval_mode_ignores_every_realtime_switch(self) -> None:
        env = {"FBV_PROFILE": "eval", "FBV_REALTIME_TRACING": "true"}
        with mock.patch.dict(os.environ, env):
            self.assertTrue(switches.eval_mode())
            for name in switches.REALTIME_SWITCHES:
                self.assertFalse(switches.realtime_switch(name, {**env, name: "true"}))
        self.assertTrue(switches.realtime_switch("FBV_REALTIME_TRACING", {"FBV_REALTIME_TRACING": "TRUE"}))
        self.assertFalse(switches.realtime_switch("FBV_REALTIME_TRACING", {}))


class TurnMetricsTests(unittest.TestCase):
    def test_one_group_per_turn_with_latency_and_usage(self) -> None:
        sent: list[dict[str, Any]] = []
        metrics = ui_metrics.TurnMetrics(sent.append)
        usage = {"latency_ms": 120.0, "prompt_tokens": 50, "completion_tokens": 5}
        metrics("agent_turn_done", {"turn_id": 3, "frontend": usage, "backend": {"calls": 0}})
        metrics("agent_turn_done", {"turn_id": 3, "frontend": usage, "backend": {"latency_ms": 900.0}})
        metrics("speech_started", {"turn_id": 3})
        metrics(
            "turn_latency",
            {
                "turn_id": 3,
                "user_stop_to_asr_final_ms": 90,
                "asr_final_to_agent_start_ms": 2,
                "agent_start_to_first_audio_ms": 400,
                "user_stop_to_first_audio_ms": 492,
            },
        )
        (message,) = sent
        self.assertEqual(
            {k: message[k] for k in ("type", "group_id", "group_label", "category", "source")},
            {
                "type": "metric-group",
                "group_id": "fbv-turn-3",
                "group_label": "Turn 1",
                "category": "latency",
                "source": "frontend-backend-verdict",
            },
        )
        values = {m["key"]: (m["value"], m["unit"]) for m in message["metrics"]}
        self.assertEqual(
            values,
            {
                "user_stop_to_first_audio_ms": (492, "ms"),
                "user_stop_to_asr_final_ms": (90, "ms"),
                "asr_final_to_agent_start_ms": (2, "ms"),
                "agent_start_to_first_audio_ms": (400, "ms"),
                "frontend_latency_ms": (240.0, "ms"),
                "frontend_prompt_tokens": (100, "tokens"),
                "frontend_completion_tokens": (10, "tokens"),
                "backend_latency_ms": (900.0, "ms"),
            },
        )
        for metric in message["metrics"]:
            self.assertEqual(set(metric), {"key", "label", "value", "unit"})

    def test_a_turn_without_values_sends_nothing(self) -> None:
        sent: list[dict[str, Any]] = []
        ui_metrics.TurnMetrics(sent.append)("turn_latency", {"turn_id": 1, "user_stop_to_first_audio_ms": None})
        self.assertEqual(sent, [])

    def test_event_log_keeps_writing_and_never_raises(self) -> None:
        inner = mock.Mock(spec=EventLog)
        log = ui_metrics.UiEventLog(inner)
        seen: list[tuple[str, dict[str, Any]]] = []
        log.register("s1", lambda kind, data: seen.append((kind, data)))
        log.register("s2", mock.Mock(side_effect=RuntimeError("boom")))
        log.write("turn_latency", "s1", {"turn_id": 1})
        log.write("turn_latency", "s2", {"turn_id": 1})
        log.write("turn_latency", "s3", {"turn_id": 1})
        log.unregister("s1")
        log.write("turn_latency", "s1", {"turn_id": 2})
        self.assertEqual(seen, [("turn_latency", {"turn_id": 1})])
        self.assertEqual(inner.write.call_count, 4)

    def test_only_browser_runtimes_get_it(self) -> None:
        from examples.frontend_backend_verdict.adapters.pipecat import runtime as runtime_module  # noqa: PLC0415

        env = {"UVICORN_WS_PING_INTERVAL": "0", "UVICORN_WS_PING_TIMEOUT": "20"}
        with pinned_environment(), mock.patch.dict(os.environ, env):
            with mock.patch.dict(ui_session._UI_RUNTIMES, clear=True):
                ui_runtime = ui_session.ui_runtime_for(ui_session.UiSelection.from_body({}))
            with mock.patch.dict(runtime_module._RUNTIMES, clear=True):
                realtime = runtime_module.runtime_for({})
        self.assertIsInstance(ui_runtime.state.routing_sink.event_log, ui_metrics.UiEventLog)
        self.assertNotIsInstance(realtime.state.routing_sink.event_log, ui_metrics.UiEventLog)
        sink = ui_runtime.state.routing_sink
        ui_metrics.install(ui_runtime)
        self.assertIs(ui_runtime.state.routing_sink, sink)


class UiMetricsEndToEndTests(unittest.IsolatedAsyncioTestCase):
    async def test_an_answered_turn_sends_a_metric_group(self) -> None:
        import asyncio  # noqa: PLC0415

        from test_fbv_ui_session import UiFakeWebSocket, _silence, _tone  # noqa: PLC0415

        from examples.frontend_backend_verdict.core.engine.scripted import ScriptedAgentPort  # noqa: PLC0415
        from examples.shared.pipeline_utils import create_transport  # noqa: PLC0415

        with pinned_environment():
            settings = load_pipecat_settings("default.yaml")
            settings = replace(
                settings, turn=replace(settings.turn, smart_turn=False), vad=replace(settings.vad, min_volume=0.0)
            )
            synthesizer = ToneSynthesizer()

            def speech(_session: Any) -> SpeechStack:
                return SpeechStack(
                    vad=EnergyVADAnalyzer(),
                    stt=StubSTT(StubRecognizer(["hello there"]), ttfs_p99=0.5),
                    tts=ToneTTS(synthesizer),
                )

            runtime = PipecatRuntime(
                load_voice_config(profile_path(DEFAULT_PROFILE)),
                profile=DEFAULT_PROFILE,
                settings=settings,
                options=ServerOptions(stub_speech=True, stub_agent="scripted"),
                speech_factory=speech,
            )
            ui_metrics.install(runtime)
            agent = ScriptedAgentPort([{"say": "Okay."}])
            runtime.agent_factory = lambda _sid: agent
            socket = UiFakeWebSocket()
            args = SimpleNamespace(websocket=socket, body={"protocol": "rtvi"})
            session = ui_session.UiSession(
                runtime=ui_session._session_runtime(runtime, runtime.config),
                transport=create_transport(args),
                speech=speech,
            )
            task = asyncio.get_running_loop().create_task(session.run())
            try:
                socket.put_message({"label": "rtvi-ai", "type": "client-ready", "id": "1", "data": {}})
                for _ in range(25):
                    socket.put_audio(_silence(20))
                    await asyncio.sleep(0.02)
                for chunk in range(30):
                    socket.put_audio(_tone(20) if chunk < 15 else _silence(20))
                    await asyncio.sleep(0.02)
                for _ in range(150):
                    if any(m.get("data", {}).get("type") == "metric-group" for m in socket.received()[1]):
                        break
                    socket.put_audio(_silence(20))
                    await asyncio.sleep(0.02)
            finally:
                socket.disconnect()
                await asyncio.wait_for(task, timeout=10)
        groups = [m["data"] for m in socket.received()[1] if m.get("data", {}).get("type") == "metric-group"]
        self.assertTrue(groups, "an answered turn sends a metric-group")
        keys = {metric["key"] for metric in groups[0]["metrics"]}
        self.assertIn("user_stop_to_first_audio_ms", keys)
        self.assertNotIn(session.session_id, runtime.state.routing_sink.event_log._listeners)


class WelcomeMessageTests(unittest.IsolatedAsyncioTestCase):
    async def _greeted(self, env: dict[str, str]) -> bool:
        with pinned_environment(), mock.patch.dict(os.environ, env):
            os.environ.pop("ENABLE_WELCOME_MESSAGE", None)
            os.environ.update(env)
            session = ui_session_for_test()
            session._register_rtvi_handlers()
            handler = session.pipeline.worker.rtvi._event_handlers["on_client_ready"].handlers[-1]
            with mock.patch.object(session.turns, "speak_greeting") as greet:
                await handler(session.pipeline.worker.rtvi)
                await handler(session.pipeline.worker.rtvi)  # a second client-ready never greets again
        return greet.call_count == 1

    async def test_unset_greets(self) -> None:
        self.assertTrue(await self._greeted({}))

    async def test_true_greets(self) -> None:
        self.assertTrue(await self._greeted({"ENABLE_WELCOME_MESSAGE": "true"}))

    async def test_false_does_not_greet(self) -> None:
        self.assertFalse(await self._greeted({"ENABLE_WELCOME_MESSAGE": "false"}))


class SetVoiceTests(unittest.IsolatedAsyncioTestCase):
    async def _frames(self, payload: dict[str, Any], *, via_rtvi: bool = False) -> tuple[Any, list[Any]]:
        with pinned_environment():
            session = ui_session_for_test()
            session._register_rtvi_handlers()
            with mock.patch.object(session.pipeline.worker, "queue_frame", new=mock.AsyncMock()) as queue:
                if via_rtvi:
                    handler = session.pipeline.worker.rtvi._event_handlers["on_client_message"].handlers[-1]
                    await handler(session.pipeline.worker.rtvi, SimpleNamespace(type="set-voice", data=payload))
                else:
                    await session.apply_set_voice(payload)
        return session, [call.args[0] for call in queue.await_args_list]

    async def test_voice_and_language(self) -> None:
        from pipecat.frames.frames import TTSUpdateSettingsFrame  # noqa: PLC0415

        session, frames = await self._frames({"voice_id": "Magpie-Multilingual.EN-US.Ray", "language": "EN-us"})
        (frame,) = frames
        self.assertIsInstance(frame, TTSUpdateSettingsFrame)
        self.assertIs(frame.service, session.pipeline.tts)
        self.assertEqual(frame.delta.voice, "Magpie-Multilingual.EN-US.Ray")
        self.assertEqual(frame.delta.language, "en-US")

    async def test_the_rtvi_message_reaches_the_handler(self) -> None:
        _, frames = await self._frames({"voice_id": "John"}, via_rtvi=True)
        (frame,) = frames
        self.assertEqual(frame.delta.voice, "John")

    async def test_no_voice_is_ignored(self) -> None:
        _, frames = await self._frames({"language": "en-US"})
        self.assertEqual(frames, [])


if __name__ == "__main__":
    unittest.main()
