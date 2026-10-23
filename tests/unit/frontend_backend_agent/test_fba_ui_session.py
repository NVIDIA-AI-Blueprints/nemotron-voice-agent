# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D103, D107

"""Browser UI sessions on the ``default`` profile (plan ``ui-transport-plan.md``, sections 4.1-4.5, 8.2).

The end-to-end tests run a :class:`UiSession` on the repository's UI WebSocket transport, built
by ``examples/shared/pipeline_utils.create_transport`` exactly as for every other example
(Pipecat's FastAPI transport with the protobuf serializer), over a fake socket, with the stub
speech of ``_fba_pipecat_fakes``. The browser must receive audio and RTVI messages only, never a
Realtime event.
"""

from __future__ import annotations

import asyncio
import json
import struct
import unittest
from dataclasses import replace
from types import SimpleNamespace
from typing import Any
from unittest import mock

import pipecat.frames.protobufs.frames_pb2 as frame_protos
from _fba_golden import pinned_environment
from _fba_pipecat_fakes import EnergyVADAnalyzer, FakeWebSocket, StubRecognizer, StubSTT, ToneSynthesizer, ToneTTS
from pipecat.frames.frames import OutputAudioRawFrame, TextFrame, TTSTextFrame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from examples.frontend_backend_agent.adapters.pipecat import ui_session
from examples.frontend_backend_agent.adapters.pipecat.frames import (
    ItemAudioFrame,
    ResponseEventFrame,
    StreamEventFrame,
    UrgentEventFrame,
)
from examples.frontend_backend_agent.adapters.pipecat.pipeline import SpeechStack
from examples.frontend_backend_agent.adapters.pipecat.runtime import PipecatRuntime
from examples.frontend_backend_agent.adapters.pipecat.settings import load_pipecat_settings
from examples.frontend_backend_agent.adapters.pipecat.ui_pipeline import RealtimeFrameFilter
from examples.frontend_backend_agent.core.agent.factory import ServerOptions
from examples.frontend_backend_agent.core.config.profiles import DEFAULT_PROFILE, profile_path
from examples.frontend_backend_agent.core.config.voice import load_voice_config
from examples.frontend_backend_agent.core.engine.scripted import ScriptedAgentPort

FBA = "frontend-backend-agent"
SAMPLE_RATE = 16000


# -- the frame filter ---------------------------------------------------------------------------


async def _noop(*_: Any, **__: Any) -> None:
    pass


class RealtimeFrameFilterTests(unittest.IsolatedAsyncioTestCase):
    async def test_drops_realtime_frames_and_passes_the_rest(self) -> None:
        patcher = mock.patch.object(FrameProcessor, "process_frame", _noop)
        patcher.start()
        self.addCleanup(patcher.stop)
        flt = RealtimeFrameFilter()
        passed: list[Any] = []

        async def push_frame(frame: Any, direction: FrameDirection = FrameDirection.DOWNSTREAM) -> None:
            passed.append(frame)

        flt.push_frame = push_frame  # type: ignore[method-assign]
        dropped = [
            ItemAudioFrame(message=None, item_id="item_1"),
            StreamEventFrame(message={"type": "response.output_audio_transcript.delta"}),
            ResponseEventFrame(message={"type": "response.done"}),
            UrgentEventFrame(message={"type": "input_audio_buffer.speech_started"}),
        ]
        kept = [
            OutputAudioRawFrame(audio=b"\0\0" * 160, sample_rate=SAMPLE_RATE, num_channels=1),
            TTSTextFrame(text="Okay.", aggregated_by="sentence"),
            TextFrame(text="hello"),
        ]
        for frame in dropped + kept:
            await flt.process_frame(frame, FrameDirection.DOWNSTREAM)
        self.assertEqual(passed, kept)


# -- selections --------------------------------------------------------------------------------


def _no_probe(*_: Any, **__: Any) -> Any:
    raise AssertionError("a UI selection must never probe service reachability")


class SelectionTests(unittest.TestCase):
    def setUp(self) -> None:
        for name in ("_filter_reachable_entries", "_first_reachable_variant", "_load_effective_services_catalog"):
            patcher = mock.patch(f"utils.{name}", side_effect=_no_probe)
            patcher.start()
            self.addCleanup(patcher.stop)

    def _select(self, body: dict[str, Any], **env: str) -> ui_session.UiSelection:
        base = {"NVIDIA_API_KEY": "test-key"}
        return ui_session.UiSelection.from_body(body, {**base, **env})

    def test_catalog_ids_resolve_from_the_named_section(self) -> None:
        cloud = self._select(
            {"asr_id": "cloud-nim:nemotron-asr-streaming-english", "tts_id": "cloud-nim:magpie-multilingual-tts"}
        )
        self.assertEqual(cloud.asr.server, "grpc.nvcf.nvidia.com:443")
        self.assertTrue(cloud.asr.use_ssl)
        self.assertTrue(cloud.tts.function_id)
        server = self._select({"tts_id": "self-hosted:server:magpie-multilingual-tts"})
        self.assertEqual(server.tts.server, "magpie-multilingual-tts-service:50051")
        local = self._select({"asr_id": "self-hosted:nemotron-asr-streaming-english"})
        self.assertEqual(local.asr.server, "nemo-speech:50051")  # unqualified: this deployment's section
        on_server = self._select(
            {"asr_id": "self-hosted:nemotron-asr-streaming-english"}, REALTIME_SERVICE_PLATFORM="server"
        )
        self.assertEqual(on_server.asr.server, "nemotron-asr-streaming-english:50052")

    def test_server_override_still_wins(self) -> None:
        selection = self._select({"tts_id": "self-hosted:server:magpie-multilingual-tts"}, FBA_TTS_SERVER="localhost:9")
        self.assertEqual(selection.tts.server, "localhost:9")

    def test_custom_and_unknown_ids(self) -> None:
        custom = self._select({"asr_id": "custom-1", "asr_server": "asr.example:50051", "asr_model": "m"})
        self.assertEqual((custom.asr.server, custom.asr.model), ("asr.example:50051", "m"))
        self.assertIsNone(self._select({"asr_id": "self-hosted:nope"}).asr)
        self.assertIsNone(self._select({}).tts)

    def test_llm_prompt_and_voice(self) -> None:
        selection = self._select(
            {
                "llm_id": "self-hosted:singlegpu:nemotron-3.5-lightning",
                "tts_voice_id": "Magpie-Multilingual.EN-US.Aria",
                "prompt_key": "frontend",
                "prompt_content": "Be brief. {persona}",
            }
        )
        self.assertEqual(selection.llm.frontend_model, "nvidia/nvidia/nemotron-3.5-lightning")
        self.assertEqual(selection.llm.backend_model, "")  # the thinker stays the profile's
        self.assertEqual(selection.voice, "Magpie-Multilingual.EN-US.Aria")
        self.assertEqual(selection.prompt, "Be brief. {persona}")
        custom = self._select({"llm_id": "custom-x", "model_id": "my/model", "base_url": "http://llm/v1"})
        self.assertEqual((custom.llm.frontend_model, custom.llm.frontend_base_url), ("my/model", "http://llm/v1"))
        self.assertEqual(self._select({"prompt_key": "backend", "prompt_content": "x"}).prompt, "")


# -- runtimes and admission -----------------------------------------------------------------------


class RuntimeCacheTests(unittest.TestCase):
    def setUp(self) -> None:
        patcher = mock.patch.dict(ui_session._UI_RUNTIMES, clear=True)
        patcher.start()
        self.addCleanup(patcher.stop)

    def _runtime(self, body: dict[str, Any]) -> tuple[PipecatRuntime, ui_session.UiSelection]:
        with pinned_environment():
            selection = ui_session.UiSelection.from_body(body)
            return ui_session.ui_runtime_for(selection), selection

    def test_prompt_and_voice_edits_reuse_the_runtime(self) -> None:
        first, _ = self._runtime({"prompt_content": "One. {persona}"})
        second, selection = self._runtime({"prompt_content": "Two. {persona}", "tts_voice_id": "Aria"})
        self.assertIs(first, second)
        self.assertEqual(len(ui_session._UI_RUNTIMES), 1)
        config = ui_session.session_config(second, selection)
        self.assertEqual(config.tts.endpoint.voice_id, "Aria")
        self.assertEqual(config.agent.prompts.inline["frontend"], "Two. {persona}")
        self.assertNotIn("frontend", second.config.agent.prompts.inline)  # the runtime is not changed

    def test_a_different_endpoint_is_a_different_runtime(self) -> None:
        first, _ = self._runtime({})
        second, _ = self._runtime({"tts_id": "self-hosted:server:magpie-multilingual-tts"})
        self.assertIsNot(first, second)
        self.assertEqual(first.profile, DEFAULT_PROFILE)
        self.assertEqual(first.settings.summary()["settings_file"], "default.yaml")
        self.assertTrue(first.settings.summary()["smart_turn"])

    def test_admission_is_shared_and_capped(self) -> None:
        admission = ui_session.UiAdmission()
        self.assertTrue(admission.admit(2))
        self.assertTrue(admission.admit(2))
        self.assertFalse(admission.admit(2))
        admission.release()
        self.assertTrue(admission.admit(2))


class ServeUiTests(unittest.IsolatedAsyncioTestCase):
    async def test_a_full_ui_is_refused_without_realtime_json(self) -> None:
        runtime = mock.Mock(config=mock.Mock(server=mock.Mock(max_sessions=1)))
        runtime.ensure_ready = mock.AsyncMock(return_value=True)
        socket = FakeWebSocket()
        with (
            mock.patch.object(ui_session, "ui_runtime_for", return_value=runtime),
            mock.patch.object(ui_session, "UI_ADMISSION", ui_session.UiAdmission()) as admission,
            mock.patch("examples.shared.pipeline_utils.create_transport") as create_transport,
        ):
            admission.sessions = 1
            await ui_session.serve_ui(SimpleNamespace(websocket=socket, body={"protocol": "rtvi"}))
        create_transport.assert_not_called()
        self.assertEqual(socket.events, [])
        self.assertEqual(admission.sessions, 1)

    async def test_a_failed_warm_up_is_refused(self) -> None:
        runtime = mock.Mock(state=mock.Mock(startup_error="ASR down"))
        runtime.ensure_ready = mock.AsyncMock(return_value=False)
        with (
            mock.patch.object(ui_session, "ui_runtime_for", return_value=runtime),
            mock.patch("examples.shared.pipeline_utils.create_transport") as create_transport,
        ):
            await ui_session.serve_ui(SimpleNamespace(websocket=FakeWebSocket(), body={"protocol": "rtvi"}))
        create_transport.assert_not_called()


class UiAgentTests(unittest.IsolatedAsyncioTestCase):
    async def test_the_agent_gets_the_server_tools_and_the_profile_policy(self) -> None:
        from _fba_golden import _clients  # noqa: PLC0415

        from examples.frontend_backend_agent import tools  # noqa: PLC0415
        from examples.shared.pipeline_utils import create_transport  # noqa: PLC0415

        with pinned_environment():
            _, _, clients = _clients()
            runtime = PipecatRuntime(
                load_voice_config(profile_path(DEFAULT_PROFILE)),
                profile=DEFAULT_PROFILE,
                settings=replace(
                    load_pipecat_settings("default.yaml"),
                    turn=replace(load_pipecat_settings("default.yaml").turn, smart_turn=False),
                ),
                options=ServerOptions(stub_speech=True),
                clients=clients,
            )
            selection = ui_session.UiSelection.from_body(
                {"prompt_key": "frontend", "prompt_content": "Custom. {persona}"}
            )
            config = ui_session.session_config(runtime, selection)
            args = SimpleNamespace(websocket=FakeWebSocket(), body={"protocol": "rtvi"})
            session = ui_session.UiSession(
                runtime=ui_session._session_runtime(runtime, config),
                transport=create_transport(args),
                speech=lambda _s: SpeechStack(vad=EnergyVADAnalyzer(), stt=StubSTT(), tts=ToneTTS()),
            )
            session.configure_agent()
        runner = session.agent
        self.assertEqual([spec.name for spec in runner.tool_specs], [spec.name for spec in tools.TOOLS])
        self.assertEqual(runner.resolved_instructions.source, "config")
        self.assertEqual(runner._base.prompts.inline["frontend"], "Custom. {persona}")
        self.assertIsNot(runtime.agent_factory, ui_session._session_runtime(runtime, config).agent_factory)


# -- end to end on the UI WebSocket transport -------------------------------------------------------


class UiFakeWebSocket(FakeWebSocket):
    """The browser side of the UI WebSocket: protobuf frames both ways."""

    def __init__(self) -> None:
        super().__init__()
        self.sent: list[bytes] = []

    async def send_bytes(self, data: bytes) -> None:
        self.sent.append(data)

    def put_audio(self, pcm: bytes) -> None:
        proto = frame_protos.Frame()
        proto.audio.audio = pcm
        proto.audio.sample_rate = SAMPLE_RATE
        proto.audio.num_channels = 1
        self.inbox.put_nowait({"type": "websocket.receive", "bytes": proto.SerializeToString()})

    def put_message(self, message: dict[str, Any]) -> None:
        proto = frame_protos.Frame()
        proto.message.data = json.dumps(message)
        self.inbox.put_nowait({"type": "websocket.receive", "bytes": proto.SerializeToString()})

    def received(self) -> tuple[int, list[dict[str, Any]]]:
        """Bytes of audio and the JSON messages the browser received."""
        audio, messages = 0, []
        for data in self.sent:
            proto = frame_protos.Frame.FromString(data)
            which = proto.WhichOneof("frame")
            if which == "audio":
                audio += len(proto.audio.audio)
            elif which == "message":
                messages.append(json.loads(proto.message.data))
        return audio, messages


def _tone(ms: int) -> bytes:
    count = SAMPLE_RATE * ms // 1000
    return struct.pack(f"<{count}h", *(8000 if i % 20 < 10 else -8000 for i in range(count)))


def _silence(ms: int) -> bytes:
    return b"\0\0" * (SAMPLE_RATE * ms // 1000)


class UiSessionEndToEndTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        env = pinned_environment()
        env.__enter__()
        self.addCleanup(env.__exit__, None, None, None)
        config = load_voice_config(profile_path(DEFAULT_PROFILE))
        # Smart Turn's model is not needed here: its branch is covered by test_fba_ui_pipeline.
        settings = load_pipecat_settings("default.yaml")
        settings = replace(
            settings, turn=replace(settings.turn, smart_turn=False), vad=replace(settings.vad, min_volume=0.0)
        )
        self.synthesizer = ToneSynthesizer()
        self.agent = ScriptedAgentPort([{"say": "Okay."}])
        self.runtime = PipecatRuntime(
            config,
            profile=DEFAULT_PROFILE,
            settings=settings,
            options=ServerOptions(stub_speech=True, stub_agent="scripted"),
            speech_factory=self._speech,
        )
        self.runtime.agent_factory = lambda _sid: self.agent
        self.socket = UiFakeWebSocket()

    def _speech(self, _session: Any) -> SpeechStack:
        return SpeechStack(
            vad=EnergyVADAnalyzer(),
            stt=StubSTT(StubRecognizer(["hello there"]), ttfs_p99=0.5),
            tts=ToneTTS(self.synthesizer),
        )

    async def _run(self, feed: Any) -> tuple[ui_session.UiSession, int, list[dict[str, Any]]]:
        from examples.shared.pipeline_utils import create_transport  # noqa: PLC0415

        args = SimpleNamespace(websocket=self.socket, body={"protocol": "rtvi", "pipeline_mode": FBA})
        session = ui_session.UiSession(
            runtime=ui_session._session_runtime(self.runtime, self.runtime.config),
            transport=create_transport(args),
            speech=self._speech,
        )
        task = asyncio.get_running_loop().create_task(session.run())
        try:
            await feed(session)
        finally:
            self.socket.disconnect()
            await asyncio.wait_for(task, timeout=10)
        audio, messages = self.socket.received()
        return session, audio, messages

    async def test_greets_answers_and_sends_rtvi_only(self) -> None:
        async def feed(session: ui_session.UiSession) -> None:
            self.socket.put_message({"label": "rtvi-ai", "type": "client-ready", "id": "1", "data": {}})
            for _ in range(25):  # 500 ms of silence while the greeting plays
                self.socket.put_audio(_silence(20))
                await asyncio.sleep(0.02)
            for chunk in range(30):
                self.socket.put_audio(_tone(20) if chunk < 15 else _silence(20))
                await asyncio.sleep(0.02)
            for _ in range(100):
                if self.agent.inputs and self.synthesizer.requests[-1:] == ["Okay."]:
                    break
                self.socket.put_audio(_silence(20))
                await asyncio.sleep(0.02)
            await asyncio.sleep(0.5)

        session, audio, messages = await self._run(feed)
        self.assertEqual(self.agent.inputs, ["hello there"])
        # The greeting, one TTS request per sentence, before the answer.
        self.assertEqual(self.synthesizer.requests[:2], ["Hi!", "How can I help you today?"])
        self.assertIn("Okay.", self.synthesizer.requests)
        self.assertGreater(audio, 0)
        labels = {m.get("label") for m in messages}
        self.assertEqual(labels, {"rtvi-ai"}, "only RTVI messages reach the browser")
        types = {m.get("type") for m in messages}
        self.assertIn("bot-ready", types)
        self.assertIn("user-transcription", types)
        self.assertFalse(
            any(str(t).startswith(("response.", "conversation.", "input_audio_buffer.", "session.")) for t in types)
        )
        self.assertEqual(session.pipeline._vad_params.stop_secs, 0.2)  # default.yaml's turn end reached the VAD
        self.assertIsNone(session.pipeline.serializer)

    async def test_input_reaches_the_vad_after_its_params(self) -> None:
        async def feed(session: ui_session.UiSession) -> None:
            self.socket.put_audio(_silence(20))
            for _ in range(50):
                if session.pipeline.vad.sample_rate and not session.pipeline._vad_pending:
                    break
                await asyncio.sleep(0.02)
                self.socket.put_audio(_silence(20))

        session, _, _ = await self._run(feed)
        self.assertFalse(session.pipeline._vad_pending)
        self.assertEqual(session.pipeline.vad.params.stop_secs, 0.2)


if __name__ == "__main__":
    unittest.main()
