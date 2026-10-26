# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D100, D101, D102, D103, D107

"""Browser UI sessions on the ``default`` profile (``shared-stack-plan.md``, step 4).

The end-to-end tests run a :class:`UiSession` on the repository's UI WebSocket transport, built by
``examples/shared/pipeline_utils.create_transport`` exactly as for every other example (Pipecat's
FastAPI transport with the protobuf serializer), over a fake socket, with stub speech and the
in-process emulator backend. The browser must receive audio and RTVI messages only, never a Realtime event.
"""

from __future__ import annotations

import asyncio
import json
import os
import struct
import unittest
from collections.abc import AsyncGenerator
from dataclasses import replace
from types import SimpleNamespace
from typing import Any
from unittest import mock

import pipecat.frames.protobufs.frames_pb2 as frame_protos
from _pipecat_fakes import PACKAGE, stub_config
from pipecat.frames.frames import Frame, InterruptionFrame, OutputAudioRawFrame, TextFrame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from starlette.websockets import WebSocketState

from examples.frontend_delegation_agent.adapters.pipecat import ui_session
from examples.frontend_delegation_agent.adapters.pipecat.frames import (
    ItemAudioFrame,
    ResponseEventFrame,
    StreamEventFrame,
    UrgentEventFrame,
)
from examples.frontend_delegation_agent.adapters.pipecat.pipeline import SpeechStack
from examples.frontend_delegation_agent.adapters.pipecat.runtime import FdaRuntime
from examples.frontend_delegation_agent.adapters.pipecat.services import EnergyVADAnalyzer, StubSTT, ToneTTS
from examples.frontend_delegation_agent.adapters.pipecat.settings import load_pipecat_settings
from examples.frontend_delegation_agent.adapters.pipecat.ui_pipeline import UiOutputGate

FDA = "frontend-delegation-agent"
SAMPLE_RATE = 16000


# -- the output gate ----------------------------------------------------------------------------


async def _noop(*_: Any, **__: Any) -> None:
    pass


class _Recorder:
    def __init__(self) -> None:
        self.frames: list[Frame] = []

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        self.frames.append(frame)


class _Playout:
    def __init__(self) -> None:
        self.written: list[float] = []
        self.interruptions = 0

    def on_written(self, chunk_ms: float) -> None:
        self.written.append(chunk_ms)

    def on_interruption(self) -> None:
        self.interruptions += 1


class UiOutputGateTests(unittest.IsolatedAsyncioTestCase):
    async def test_drops_realtime_frames_and_counts_item_audio(self) -> None:
        # The gate alone, outside a started pipeline: the base class's frame bookkeeping is skipped.
        patcher = mock.patch.object(FrameProcessor, "process_frame", _noop)
        patcher.start()
        self.addCleanup(patcher.stop)
        playout = _Playout()
        gate = UiOutputGate()
        gate.bind(playout)  # type: ignore[arg-type]
        recorder = _Recorder()

        async def push(frame: Frame, direction: FrameDirection = FrameDirection.DOWNSTREAM) -> None:
            await recorder.process_frame(frame, direction)

        gate.push_frame = push  # type: ignore[method-assign]
        audio = OutputAudioRawFrame(audio=b"\0\0" * 1600, sample_rate=16000, num_channels=1)  # 100 ms
        text = TextFrame(text="kept")
        frames: list[Frame] = [
            UrgentEventFrame(message={"type": "session.updated"}),
            audio,  # before any item: played, not counted (as on the Realtime path)
            ItemAudioFrame(message={}, response_id="r", item_id="i", begin=True),
            StreamEventFrame(message={"type": "response.output_audio_transcript.delta"}),
            audio,
            ResponseEventFrame(message={"type": "response.done"}),
            ItemAudioFrame(message={}, response_id="r", item_id="i", begin=False),
            audio,
            text,
            InterruptionFrame(),
        ]
        for frame in frames:
            await gate.process_frame(frame, FrameDirection.DOWNSTREAM)
        passed = recorder.frames
        self.assertEqual(passed.count(audio), 3)
        self.assertIn(text, passed)
        self.assertFalse(any(isinstance(f, (ItemAudioFrame, StreamEventFrame, ResponseEventFrame)) for f in passed))
        self.assertFalse(any(isinstance(f, UrgentEventFrame) for f in passed))
        self.assertEqual(playout.written, [100.0])
        self.assertEqual(playout.interruptions, 1)


# -- selections, runtimes and admission ------------------------------------------------------------


class SelectionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.env = {"NVIDIA_API_KEY": "nvapi-test"}

    def test_catalog_ids_resolve_from_the_example_catalog(self) -> None:
        for entry_id in ("magpie-multilingual-tts", "self-hosted:magpie-multilingual-tts"):
            with self.subTest(entry_id=entry_id):
                endpoint = ui_session.speech_selection("tts", {"tts_id": entry_id}, self.env)
                assert endpoint is not None
                self.assertEqual((endpoint.server, endpoint.voice_id), ("nemo-speech:50051", "John"))
        cloud = ui_session.speech_selection("asr", {"asr_id": "cloud-nim:nemotron-asr-streaming-english"}, self.env)
        assert cloud is not None
        self.assertTrue(cloud.use_ssl)
        self.assertEqual(cloud.function_id, "4155ae85-73e1-4936-b47f-87b9de165651")

    def test_server_override_still_wins(self) -> None:
        env = {**self.env, "FBA_ASR_SERVER": "localhost:50051"}
        endpoint = ui_session.speech_selection("asr", {"asr_id": "nemotron-asr-streaming-english"}, env)
        assert endpoint is not None
        self.assertEqual(endpoint.server, "localhost:50051")

    def test_custom_unknown_and_missing_ids(self) -> None:
        self.assertIsNone(ui_session.speech_selection("asr", {}, self.env))
        self.assertIsNone(ui_session.speech_selection("asr", {"asr_id": "nope"}, self.env))
        custom = ui_session.speech_selection("asr", {"asr_id": "custom-1", "asr_server": "host:1"}, self.env)
        assert custom is not None
        self.assertEqual(custom.server, "host:1")

    def test_llm_voice_and_prompt(self) -> None:
        body = {"llm_id": "nemotron-3.5-lightning", "tts_voice_id": "Aria", "prompt_content": "edited"}
        selection = ui_session.UiSelection.from_body(body, self.env)
        self.assertEqual(selection.model, "nvidia/nvidia/nemotron-3.5-lightning")
        self.assertEqual((selection.voice, selection.prompt_edited), ("Aria", True))
        custom = ui_session.UiSelection.from_body({"llm_id": "custom-x", "model_id": "m", "base_url": "u"}, self.env)
        self.assertEqual((custom.model, custom.base_url), ("m", "u"))


class RuntimeCacheTests(unittest.TestCase):
    def setUp(self) -> None:
        patcher = mock.patch.object(ui_session, "_UI_RUNTIMES", {})
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_the_default_profile_is_todays_demo(self) -> None:
        runtime = ui_session.ui_runtime_for(ui_session.UiSelection.from_body({}))
        config = runtime.config
        self.assertIn(PACKAGE / "config" / "profiles" / "pipecat_demo.yaml", config.source_files)
        self.assertEqual(runtime.settings.summary()["settings_file"], str(PACKAGE / "config" / "adapters/pipecat.yaml"))
        self.assertEqual(config.tools.executor, "local")
        self.assertTrue(config.voice.protocol.greeting_enabled)

    def test_voice_edits_reuse_the_runtime(self) -> None:
        first = ui_session.ui_runtime_for(ui_session.UiSelection.from_body({}))
        second = ui_session.ui_runtime_for(ui_session.UiSelection.from_body({"tts_voice_id": "Aria"}))
        self.assertIs(first, second)
        voice = ui_session.session_voice_config(first, ui_session.UiSelection.from_body({"tts_voice_id": "Aria"}))
        self.assertEqual(voice.tts.endpoint.voice_id, "Aria")
        self.assertEqual(first.config.voice.tts.endpoint.voice_id, "John")

    def test_a_different_endpoint_is_a_different_runtime(self) -> None:
        first = ui_session.ui_runtime_for(ui_session.UiSelection.from_body({}))
        other = ui_session.ui_runtime_for(
            ui_session.UiSelection.from_body({"asr_id": "custom-1", "asr_server": "host:1"})
        )
        self.assertIsNot(first, other)

    def test_welcome_message_off_turns_the_greeting_off(self) -> None:
        with mock.patch.dict(os.environ, {"ENABLE_WELCOME_MESSAGE": "false"}):
            runtime = ui_session.ui_runtime_for(ui_session.UiSelection.from_body({}))
        self.assertFalse(runtime.config.voice.protocol.greeting_enabled)

    def test_admission_is_shared_and_capped(self) -> None:
        admission = ui_session.UiAdmission()
        self.assertTrue(admission.admit(2))
        self.assertTrue(admission.admit(2))
        self.assertFalse(admission.admit(2))
        admission.release()
        self.assertTrue(admission.admit(2))


class ServeUiTests(unittest.IsolatedAsyncioTestCase):
    async def test_a_full_ui_is_refused(self) -> None:
        runtime = mock.Mock(config=stub_config("pipecat_demo.yaml"), ensure_ready=mock.AsyncMock(return_value=True))
        admission = ui_session.UiAdmission()
        admission.sessions = 99
        with (
            mock.patch.object(ui_session, "ui_runtime_for", return_value=runtime),
            mock.patch.object(ui_session, "UI_ADMISSION", admission),
            mock.patch("examples.shared.pipeline_utils.create_transport") as create_transport,
        ):
            await ui_session.serve_ui(SimpleNamespace(body={"protocol": "rtvi"}))
        create_transport.assert_not_called()

    async def test_a_failed_start_is_refused(self) -> None:
        runtime = mock.Mock(ensure_ready=mock.AsyncMock(return_value=False), startup_error="gateway down")
        with (
            mock.patch.object(ui_session, "ui_runtime_for", return_value=runtime),
            mock.patch("examples.shared.pipeline_utils.create_transport") as create_transport,
        ):
            await ui_session.serve_ui(SimpleNamespace(body={"protocol": "rtvi"}))
        create_transport.assert_not_called()


# -- end to end on the UI WebSocket transport -------------------------------------------------------


class UiFakeWebSocket:
    """The browser side of the UI WebSocket: protobuf frames both ways."""

    def __init__(self) -> None:
        self.inbox: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self.sent: list[bytes] = []
        self.client_state = WebSocketState.CONNECTED
        self.application_state = WebSocketState.CONNECTED
        self.headers: dict[str, str] = {}
        self.query_params: dict[str, str] = {}

    async def receive(self) -> dict[str, Any]:
        return await self.inbox.get()

    async def send_text(self, text: str) -> None:  # pragma: no cover - the UI transport sends protobuf
        raise AssertionError(f"unexpected text frame: {text[:80]}")

    async def send_bytes(self, data: bytes) -> None:
        self.sent.append(data)

    async def close(self, code: int = 1000, reason: str = "") -> None:
        self.client_state = WebSocketState.DISCONNECTED
        self.application_state = WebSocketState.DISCONNECTED

    def disconnect(self) -> None:
        self.inbox.put_nowait({"type": "websocket.disconnect", "code": 1000})

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


class RecordingTTS(ToneTTS):
    """The tone TTS, recording each sentence it is asked to speak."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.requests: list[str] = []

    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame, None]:
        self.requests.append(text)
        async for frame in super().run_tts(text, context_id):
            yield frame


def _tone(ms: int) -> bytes:
    count = SAMPLE_RATE * ms // 1000
    return struct.pack(f"<{count}h", *(8000 if i % 20 < 10 else -8000 for i in range(count)))


def _silence(ms: int) -> bytes:
    return b"\0\0" * (SAMPLE_RATE * ms // 1000)


class UiSessionEndToEndTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        config = stub_config("pipecat_demo.yaml")
        config = replace(config, backend=replace(config.backend, delay_seconds=0.0))
        settings = load_pipecat_settings("adapters/pipecat.yaml")
        settings = replace(settings, vad=replace(settings.vad, min_volume=0.0))
        self.tts = RecordingTTS(sample_rate=16000, ms_per_char=3.0)
        self.runtime = FdaRuntime(config, settings, profile="default", speech_factory=self._speech)
        self.socket = UiFakeWebSocket()

    def _speech(self, _session: Any) -> SpeechStack:
        return SpeechStack(vad=EnergyVADAnalyzer(), stt=StubSTT(), tts=self.tts)

    async def test_configures_greets_answers_and_sends_rtvi_only(self) -> None:
        from examples.shared.pipeline_utils import create_transport  # noqa: PLC0415

        self.assertTrue(await self.runtime.ensure_ready())
        args = SimpleNamespace(websocket=self.socket, body={"protocol": "rtvi", "pipeline_mode": FDA})
        session = ui_session.UiSession(
            runtime=self.runtime,
            config=self.runtime.config.voice,
            transport=create_transport(args),
            speech=self._speech(None),
        )
        task = asyncio.get_running_loop().create_task(session.run())
        try:
            self.socket.put_message({"label": "rtvi-ai", "type": "client-ready", "id": "1", "data": {}})
            for _ in range(50):  # 1 s of silence while the greeting plays
                self.socket.put_audio(_silence(20))
                await asyncio.sleep(0.02)
            for chunk in range(40):  # 600 ms of speech, then silence
                self.socket.put_audio(_tone(20) if chunk < 30 else _silence(20))
                await asyncio.sleep(0.02)
            for _ in range(150):
                if len(self.tts.requests) > 2:
                    break
                self.socket.put_audio(_silence(20))
                await asyncio.sleep(0.02)
            await asyncio.sleep(0.3)
        finally:
            self.socket.disconnect()
            await asyncio.wait_for(task, timeout=10)
        audio, messages = self.socket.received()
        self.assertTrue(session.protocol.configured, "the demo page's session.update configured the session")
        # The greeting, one TTS request per sentence, before anything else.
        self.assertEqual(self.tts.requests[:2], ["Hi!", "How can I help you today?"])
        self.assertGreater(len(self.tts.requests), 2, "the user's turn got a spoken reply")
        self.assertGreater(audio, 0)
        self.assertGreater(session.playout.written_ms, 0, "agent audio fed the playout estimate")
        labels = {m.get("label") for m in messages}
        self.assertEqual(labels, {"rtvi-ai"}, "only RTVI messages reach the browser")
        types = {m.get("type") for m in messages}
        self.assertIn("bot-ready", types)
        self.assertIn("user-transcription", types)
        self.assertFalse(
            any(str(t).startswith(("response.", "conversation.", "input_audio_buffer.", "session.")) for t in types)
        )


if __name__ == "__main__":
    unittest.main()
