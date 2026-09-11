# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

# ruff: noqa: D100, D101, D102, D103

from __future__ import annotations

import base64
import json
import unittest
from typing import Any
from unittest.mock import AsyncMock, patch

from pipecat.frames.frames import InputAudioRawFrame, OutputAudioRawFrame
from pipecat.processors.aggregators.llm_context import LLMContext
from realtime_helpers import FakeWebSocket

from realtime.audio import (
    DEFAULT_CLIENT_PCM_RATE,
    PIPELINE_PCM_RATE,
    AudioResampler,
    decode_base64_audio,
    encode_base64_audio,
)
from realtime.controller import RealtimeSessionController
from realtime.events import SERVER_SESSION_UPDATED
from realtime.frames import RealtimeResponseCreateFrame
from realtime.gateway import handle_realtime_websocket
from realtime.serializer import RealtimeFrameSerializer
from realtime.session import RealtimeSessionCapabilities
from realtime.transport import RealtimeManualResponseGate

_MODEL = "test-realtime-model"
_VOICE = "Magpie-Multilingual.EN-US.Aria"
_OTHER_VOICE = "Magpie-Multilingual.EN-US.Claire"
_SERVER_VAD_DEFAULTS = {
    "type": "server_vad",
    "threshold": 0.5,
    "prefix_padding_ms": 300,
    "silence_duration_ms": 500,
    "create_response": True,
    "interrupt_response": True,
    "idle_timeout_ms": None,
}


def _controller(
    *,
    voices: frozenset[str] | None = None,
    input_transcription_model: str | None = None,
    manual: bool = False,
) -> RealtimeSessionController:
    controller = RealtimeSessionController(
        model=_MODEL,
        voice=_VOICE,
        runtime_config={
            "pipeline_mode": "generic-assistant",
            "model_id": _MODEL,
            "tts_voice_id": _VOICE,
        },
        input_transcription_model=input_transcription_model,
        capabilities=RealtimeSessionCapabilities(
            voices=voices or frozenset({_VOICE}),
            supports_manual_input=manual,
        ),
    )
    if manual:
        controller.apply_session_update(
            {
                "type": "realtime",
                "audio": {"input": {"turn_detection": None}},
            }
        )
    return controller


def _serializer(
    *,
    controller: RealtimeSessionController | None = None,
) -> tuple[RealtimeFrameSerializer, list[dict[str, Any]]]:
    emitted: list[dict[str, Any]] = []

    async def emit(event: dict[str, Any]) -> None:
        emitted.append(event)

    async def emit_batch(events: list[dict[str, Any]]) -> None:
        emitted.extend(events)

    serializer = RealtimeFrameSerializer(controller=controller or _controller())
    serializer.set_emit(emit, emit_batch)
    return serializer, emitted


class AudioHelperTests(unittest.IsolatedAsyncioTestCase):
    async def test_base64_roundtrip(self) -> None:
        raw = b"\x00\x01\x02\x03" * 10
        encoded = encode_base64_audio(raw)
        self.assertEqual(decode_base64_audio(encoded), raw)

    async def test_resample_uses_configured_pipeline_rate(self) -> None:
        resampler = AudioResampler()
        pcm = b"\x00\x00" * 80
        with patch.object(resampler, "_uplink") as uplink:
            uplink.resample = AsyncMock(return_value=b"up")
            out = await resampler.to_pipeline(pcm, DEFAULT_CLIENT_PCM_RATE, pipeline_rate=8000)
        self.assertEqual(out, b"up")
        uplink.resample.assert_awaited_once_with(pcm, DEFAULT_CLIENT_PCM_RATE, 8000)

        with patch.object(resampler, "_downlink") as downlink:
            downlink.resample = AsyncMock(return_value=b"down")
            out = await resampler.from_pipeline(pcm, DEFAULT_CLIENT_PCM_RATE, pipeline_rate=8000)
        self.assertEqual(out, b"down")
        downlink.resample.assert_awaited_once_with(pcm, 8000, DEFAULT_CLIENT_PCM_RATE)

    async def test_reset_recreates_uplink_and_resets_downlink_in_place(self) -> None:
        resampler = AudioResampler()
        uplink_before = resampler._uplink
        downlink_before = resampler._downlink
        await resampler.from_pipeline(
            b"\x00\x00" * 512,
            DEFAULT_CLIENT_PCM_RATE,
            pipeline_rate=22050,
        )
        self.assertIsNotNone(downlink_before._stream)

        resampler.reset()

        self.assertIsNot(resampler._uplink, uplink_before)
        self.assertIs(resampler._downlink, downlink_before)
        self.assertIsNone(downlink_before._stream)


class SerializerAudioTests(unittest.IsolatedAsyncioTestCase):
    async def test_client_event_envelopes_reject_unknown_root_fields(self) -> None:
        cases = (
            {"type": "session.update", "session": {}, "unexpected": True},
            {"type": "response.create", "response": {}, "unexpected": True},
            {
                "type": "conversation.item.create",
                "item": {"type": "message", "role": "user", "content": []},
                "unexpected": True,
            },
            {"type": "conversation.item.retrieve", "item_id": "item_missing", "unexpected": True},
        )
        for event in cases:
            with self.subTest(event_type=event["type"]):
                serializer, emitted = _serializer()
                frame = await serializer.deserialize(json.dumps(event))
                self.assertIsNone(frame)
                self.assertEqual(emitted[-1]["error"]["code"], "unknown_parameter")
                self.assertEqual(emitted[-1]["error"]["param"], "unexpected")

    async def test_client_event_id_has_the_native_length_bound(self) -> None:
        serializer, emitted = _serializer()
        frame = await serializer.deserialize(json.dumps({"type": "response.create", "event_id": "x" * 513}))
        self.assertIsNone(frame)
        self.assertEqual(emitted[-1]["error"]["code"], "invalid_value")
        self.assertEqual(emitted[-1]["error"]["param"], "event_id")
        self.assertNotIn("event_id", emitted[-1]["error"])

    async def test_oversized_client_event_is_rejected_before_json_decode(self) -> None:
        serializer, emitted = _serializer()
        with (
            patch("realtime.serializer.MAX_REALTIME_EVENT_BYTES", 32),
            patch("realtime.serializer.strict_json_loads") as decode,
        ):
            frame = await serializer.deserialize('{"type":"response.create","padding":"too-large"}')

        self.assertIsNone(frame)
        decode.assert_not_called()
        self.assertEqual(emitted[-1]["error"]["code"], "event_too_large")

    async def test_append_accepts_canonical_24khz_and_hands_pipeline_16khz(self) -> None:
        serializer, emitted = _serializer()
        client_pcm = b"\x00\x00" * 480  # 20 ms mono PCM16 at 24 kHz.
        pipeline_pcm = b"\x00\x00" * 320
        serializer._resampler.to_pipeline = AsyncMock(return_value=pipeline_pcm)

        frame = await serializer.deserialize(
            json.dumps(
                {
                    "type": "input_audio_buffer.append",
                    "audio": base64.b64encode(client_pcm).decode("ascii"),
                }
            )
        )

        self.assertIsInstance(frame, InputAudioRawFrame)
        assert isinstance(frame, InputAudioRawFrame)
        self.assertEqual(frame.sample_rate, PIPELINE_PCM_RATE)
        self.assertEqual(frame.audio, pipeline_pcm)
        serializer._resampler.to_pipeline.assert_awaited_once_with(
            client_pcm,
            DEFAULT_CLIENT_PCM_RATE,
            pipeline_rate=PIPELINE_PCM_RATE,
            format_type="audio/pcm",
        )
        self.assertEqual(emitted, [])

    async def test_output_audio_emits_only_canonical_delta_names(self) -> None:
        controller = _controller()
        serializer, emitted = _serializer(controller=controller)
        controller.start_response()
        client_pcm = b"\x01\x00" * 160
        serializer._resampler.from_pipeline = AsyncMock(return_value=client_pcm)

        payload = await serializer.serialize(
            OutputAudioRawFrame(audio=b"\x00\x00" * 160, sample_rate=22050, num_channels=1)
        )

        self.assertIsNone(payload)
        types = [event["type"] for event in emitted]
        self.assertIn("response.output_audio.delta", types)
        self.assertNotIn("response.audio.delta", types)
        delta = next(event for event in emitted if event["type"] == "response.output_audio.delta")
        self.assertEqual(base64.b64decode(delta["delta"]), client_pcm)

    async def test_output_audio_without_an_active_response_is_dropped(self) -> None:
        for completed_response in (False, True):
            with self.subTest(completed_response=completed_response):
                controller = _controller()
                serializer, emitted = _serializer(controller=controller)
                if completed_response:
                    controller.start_response()
                    controller.finish_response(status="completed")
                    emitted.clear()

                payload = await serializer.serialize(
                    OutputAudioRawFrame(audio=b"\x00\x00" * 160, sample_rate=22050, num_channels=1)
                )

                self.assertIsNone(payload)
                self.assertEqual(emitted, [])
                self.assertIsNone(controller.active_response_id)

    async def test_audio_delta_precedes_terminal_sequence(self) -> None:
        controller = _controller()
        serializer, emitted = _serializer(controller=controller)
        controller.start_response()
        serializer._resampler.from_pipeline = AsyncMock(return_value=b"\x00\x00" * 160)
        await serializer.serialize(OutputAudioRawFrame(audio=b"\x00\x00" * 160, sample_rate=22050, num_channels=1))
        emitted.extend(controller.finish_response(status="completed"))

        types = [event["type"] for event in emitted]
        self.assertLess(types.index("response.output_audio.delta"), types.index("response.output_audio.done"))
        self.assertLess(types.index("response.output_audio.done"), types.index("response.done"))
        self.assertNotIn("response.output_text.delta", types)
        self.assertNotIn("response.output_text.done", types)

    async def test_commit_is_unsupported_in_streaming_mode(self) -> None:
        serializer, emitted = _serializer()
        frame = await serializer.deserialize(json.dumps({"type": "input_audio_buffer.commit", "event_id": "e1"}))
        self.assertIsNone(frame)
        self.assertEqual(emitted[-1]["type"], "error")
        self.assertEqual(emitted[-1]["error"]["code"], "unsupported_capability")
        self.assertEqual(emitted[-1]["error"]["event_id"], "e1")

    async def test_streaming_append_does_not_enable_manual_buffer_controls(self) -> None:
        pcm = base64.b64encode(b"\x00\x00" * 480).decode("ascii")
        for event_type in ("input_audio_buffer.commit", "input_audio_buffer.clear"):
            with self.subTest(event_type=event_type):
                serializer, emitted = _serializer()
                serializer._resampler.to_pipeline = AsyncMock(return_value=b"\x00\x00" * 320)
                appended = await serializer.deserialize(json.dumps({"type": "input_audio_buffer.append", "audio": pcm}))
                frame = await serializer.deserialize(json.dumps({"type": event_type}))

                self.assertIsInstance(appended, InputAudioRawFrame)
                self.assertIsNone(frame)
                self.assertEqual(emitted[-1]["error"]["code"], "unsupported_capability")

    async def test_manual_append_buffers_until_commit_then_hands_off_once(self) -> None:
        controller = _controller(input_transcription_model="test-asr", manual=True)
        serializer, emitted = _serializer(controller=controller)
        gate = RealtimeManualResponseGate(controller=controller)
        handoff = AsyncMock()
        serializer.set_manual_input_handlers(commit_hook=handoff, response_gate=gate)
        client_pcm = b"\x01\x00" * 480
        pipeline_pcm = b"\x02\x00" * 320
        serializer._resampler.complete_input_to_pipeline = AsyncMock(return_value=pipeline_pcm)

        appended = await serializer.deserialize(
            json.dumps(
                {
                    "type": "input_audio_buffer.append",
                    "audio": base64.b64encode(client_pcm).decode("ascii"),
                }
            )
        )
        committed = await serializer.deserialize(
            json.dumps({"type": "input_audio_buffer.commit", "event_id": "commit_1"})
        )

        self.assertIsNone(appended)
        self.assertIsNone(committed)
        self.assertEqual(emitted, [])
        handoff.assert_awaited_once_with(pipeline_pcm, PIPELINE_PCM_RATE)
        serializer._resampler.complete_input_to_pipeline.assert_awaited_once_with(
            client_pcm,
            DEFAULT_CLIENT_PCM_RATE,
            pipeline_rate=PIPELINE_PCM_RATE,
            format_type="audio/pcm",
        )
        self.assertEqual(gate.pending_commits, 1)
        self.assertEqual(serializer._manual_input_audio, bytearray())

    async def test_manual_commit_is_independent_of_an_active_response(self) -> None:
        controller = _controller(input_transcription_model="test-asr", manual=True)
        serializer, emitted = _serializer(controller=controller)
        gate = RealtimeManualResponseGate(controller=controller)
        handoff = AsyncMock()
        serializer.set_manual_input_handlers(commit_hook=handoff, response_gate=gate)
        controller.start_response()
        client_pcm = b"\x01\x00" * 480
        pipeline_pcm = b"\x02\x00" * 320
        serializer._resampler.complete_input_to_pipeline = AsyncMock(return_value=pipeline_pcm)

        appended = await serializer.deserialize(
            json.dumps(
                {
                    "type": "input_audio_buffer.append",
                    "audio": base64.b64encode(client_pcm).decode("ascii"),
                }
            )
        )
        committed = await serializer.deserialize(
            json.dumps({"type": "input_audio_buffer.commit", "event_id": "commit_during_response"})
        )

        self.assertIsNone(appended)
        self.assertIsNone(committed)
        self.assertEqual(emitted, [])
        self.assertTrue(controller.response_in_progress)
        self.assertEqual(gate.pending_commits, 1)
        self.assertTrue(gate.has_unclaimed_manual_commit)
        handoff.assert_awaited_once_with(pipeline_pcm, PIPELINE_PCM_RATE)
        self.assertEqual(serializer._manual_input_audio, bytearray())

    async def test_manual_clear_discards_buffer_and_emits_canonical_ack(self) -> None:
        controller = _controller(input_transcription_model="test-asr", manual=True)
        serializer, emitted = _serializer(controller=controller)
        client_pcm = b"\x01\x00" * 480
        await serializer.deserialize(
            json.dumps(
                {
                    "type": "input_audio_buffer.append",
                    "audio": base64.b64encode(client_pcm).decode("ascii"),
                }
            )
        )

        frame = await serializer.deserialize(json.dumps({"type": "input_audio_buffer.clear", "event_id": "clear_1"}))

        self.assertIsNone(frame)
        self.assertEqual(serializer._manual_input_audio, bytearray())
        self.assertEqual([event["type"] for event in emitted], ["input_audio_buffer.cleared"])

    async def test_manual_commit_rejects_empty_buffer_without_registering_turn(self) -> None:
        controller = _controller(input_transcription_model="test-asr", manual=True)
        serializer, emitted = _serializer(controller=controller)
        gate = RealtimeManualResponseGate(controller=controller)
        serializer.set_manual_input_handlers(commit_hook=AsyncMock(), response_gate=gate)

        frame = await serializer.deserialize(
            json.dumps({"type": "input_audio_buffer.commit", "event_id": "empty_commit"})
        )

        self.assertIsNone(frame)
        self.assertEqual(emitted[-1]["error"]["code"], "input_audio_buffer_commit_empty")
        self.assertEqual(emitted[-1]["error"]["event_id"], "empty_commit")
        self.assertEqual(gate.pending_commits, 0)

    async def test_manual_response_create_returns_gate_marker(self) -> None:
        controller = _controller(input_transcription_model="test-asr", manual=True)
        serializer, emitted = _serializer(controller=controller)
        gate = RealtimeManualResponseGate(controller=controller)
        serializer.set_manual_input_handlers(commit_hook=AsyncMock(), response_gate=gate)

        frame = await serializer.deserialize(json.dumps({"type": "response.create"}))

        self.assertIsInstance(frame, RealtimeResponseCreateFrame)
        self.assertEqual(emitted[0]["type"], "response.created")
        self.assertEqual(frame.response_id, emitted[0]["response"]["id"])

    async def test_second_manual_commit_is_rejected_without_clearing_its_buffer(self) -> None:
        controller = _controller(input_transcription_model="test-asr", manual=True)
        serializer, emitted = _serializer(controller=controller)
        gate = RealtimeManualResponseGate(controller=controller)
        handoff = AsyncMock()
        serializer.set_manual_input_handlers(commit_hook=handoff, response_gate=gate)
        first_pcm = b"\x01\x00" * 480
        second_pcm = b"\x02\x00" * 480
        serializer._resampler.complete_input_to_pipeline = AsyncMock(side_effect=[b"first"])

        await serializer.deserialize(
            json.dumps(
                {
                    "type": "input_audio_buffer.append",
                    "audio": base64.b64encode(first_pcm).decode("ascii"),
                }
            )
        )
        await serializer.deserialize(json.dumps({"type": "input_audio_buffer.commit"}))
        await serializer.deserialize(
            json.dumps(
                {
                    "type": "input_audio_buffer.append",
                    "audio": base64.b64encode(second_pcm).decode("ascii"),
                }
            )
        )

        frame = await serializer.deserialize(json.dumps({"type": "input_audio_buffer.commit", "event_id": "commit_2"}))

        self.assertIsNone(frame)
        self.assertEqual(emitted[-1]["error"]["code"], "input_audio_buffer_commit_pending")
        self.assertEqual(emitted[-1]["error"]["event_id"], "commit_2")
        self.assertEqual(serializer._manual_input_audio, bytearray(second_pcm))
        handoff.assert_awaited_once_with(b"first", PIPELINE_PCM_RATE)

    async def test_live_voice_update_requires_installed_response_runtime(self) -> None:
        controller = _controller(voices=frozenset({_VOICE, _OTHER_VOICE}))
        serializer, emitted = _serializer(controller=controller)

        frame = await serializer.deserialize(
            json.dumps(
                {
                    "type": "session.update",
                    "session": {"audio": {"output": {"voice": _OTHER_VOICE}}},
                }
            )
        )

        self.assertIsNone(frame)
        audio = controller.public_session()["audio"]
        self.assertEqual(audio["input"]["turn_detection"], _SERVER_VAD_DEFAULTS)
        self.assertEqual(audio["input"]["format"]["rate"], DEFAULT_CLIENT_PCM_RATE)
        self.assertEqual(audio["output"]["voice"], _VOICE)
        self.assertEqual(emitted[-1]["type"], "error")
        self.assertEqual(emitted[-1]["error"]["code"], "response_gate_missing")
        self.assertEqual(emitted[-1]["error"]["param"], "session.audio.output.voice")

    async def test_malformed_live_voice_update_is_rejected_before_deferred_discovery(self) -> None:
        resolve = AsyncMock(side_effect=RuntimeError("catalog unavailable"))
        controller = RealtimeSessionController(
            model=_MODEL,
            voice=_VOICE,
            runtime_config={
                "pipeline_mode": "generic-assistant",
                "model_id": _MODEL,
                "tts_voice_id": _VOICE,
            },
            capabilities=RealtimeSessionCapabilities(voices=frozenset({_VOICE})),
            output_voice_resolver=resolve,
            output_voices_resolved=False,
        )
        serializer, emitted = _serializer(controller=controller)
        cases = (
            (
                {"audio": {"output": {"voice": _OTHER_VOICE, "unexpected": True}}},
                "unknown_parameter",
                "session.audio.output.unexpected",
            ),
            (
                {"audio": {"output": {"voice": []}}},
                "invalid_type",
                "session.audio.output.voice",
            ),
        )

        for session_patch, code, param in cases:
            with self.subTest(code=code, param=param):
                frame = await serializer.deserialize(
                    json.dumps(
                        {
                            "type": "session.update",
                            "session": session_patch,
                        }
                    )
                )
                self.assertIsNone(frame)
                self.assertEqual(emitted[-1]["error"]["code"], code)
                self.assertEqual(emitted[-1]["error"]["param"], param)

        resolve.assert_not_awaited()
        self.assertEqual(controller.public_session()["audio"]["output"]["voice"], _VOICE)
        self.assertEqual(controller.session.capabilities.voices, frozenset({_VOICE}))

    async def test_live_voice_update_rejects_an_audio_response_before_its_first_delta(self) -> None:
        controller = _controller(voices=frozenset({_VOICE, _OTHER_VOICE}))
        serializer, emitted = _serializer(controller=controller)
        gate = RealtimeManualResponseGate(controller=controller)
        gate.bind_tts_service(object())
        serializer.set_response_gate(gate)
        response_frame = await serializer.deserialize(json.dumps({"type": "response.create"}))
        self.assertIsInstance(response_frame, RealtimeResponseCreateFrame)
        response_id = controller.active_response_id
        self.assertEqual(controller.active_audio_output["voice"], _VOICE)

        frame = await serializer.deserialize(
            json.dumps(
                {
                    "type": "session.update",
                    "session": {"audio": {"output": {"voice": _OTHER_VOICE}}},
                    "event_id": "voice_during_audio_response",
                }
            )
        )

        self.assertIsNone(frame)
        self.assertEqual(emitted[-1]["type"], "error")
        self.assertEqual(emitted[-1]["error"]["code"], "immutable_field")
        self.assertEqual(emitted[-1]["error"]["param"], "session.audio.output.voice")
        self.assertEqual(emitted[-1]["error"]["event_id"], "voice_during_audio_response")
        self.assertEqual(controller.public_session()["audio"]["output"]["voice"], _VOICE)
        self.assertFalse(controller.session.output_audio_started)

        controller.finish_response(status="cancelled", reason="client_cancelled")
        serializer.notify_response_done_published(response_id)
        retry = await serializer.deserialize(
            json.dumps(
                {
                    "type": "session.update",
                    "session": {"audio": {"output": {"voice": _OTHER_VOICE}}},
                }
            )
        )
        self.assertIsNone(retry)
        self.assertEqual(emitted[-1]["type"], "session.updated")
        self.assertEqual(controller.public_session()["audio"]["output"]["voice"], _OTHER_VOICE)

    async def test_live_voice_update_is_safe_during_a_text_only_response(self) -> None:
        controller = _controller(voices=frozenset({_VOICE, _OTHER_VOICE}))
        controller.apply_session_update({"output_modalities": ["text"]})
        serializer, emitted = _serializer(controller=controller)
        gate = RealtimeManualResponseGate(controller=controller)
        gate.bind_tts_service(object())
        serializer.set_response_gate(gate)
        response_frame = await serializer.deserialize(json.dumps({"type": "response.create"}))
        self.assertIsInstance(response_frame, RealtimeResponseCreateFrame)
        self.assertEqual(controller.output_kind, "text")

        frame = await serializer.deserialize(
            json.dumps(
                {
                    "type": "session.update",
                    "session": {"audio": {"output": {"voice": _OTHER_VOICE}}},
                }
            )
        )

        self.assertIsNone(frame)
        self.assertEqual(emitted[-1]["type"], "session.updated")
        self.assertEqual(controller.public_session()["audio"]["output"]["voice"], _OTHER_VOICE)
        self.assertIsNone(controller.active_audio_output)

    async def test_live_voice_update_rejects_a_pending_audio_pipeline_owner(self) -> None:
        controller = _controller(voices=frozenset({_VOICE, _OTHER_VOICE}))
        serializer, emitted = _serializer(controller=controller)
        gate = RealtimeManualResponseGate(controller=controller)
        gate.bind_tts_service(object())
        serializer.set_response_gate(gate)
        controller.prepare_pipeline_response(
            owner_id="run_audio_pending",
            interruption_generation=controller.interruption_generation,
        )

        frame = await serializer.deserialize(
            json.dumps(
                {
                    "type": "session.update",
                    "session": {"audio": {"output": {"voice": _OTHER_VOICE}}},
                }
            )
        )

        self.assertIsNone(frame)
        self.assertEqual(emitted[-1]["error"]["code"], "immutable_field")
        self.assertEqual(controller.public_session()["audio"]["output"]["voice"], _VOICE)
        self.assertTrue(controller.pipeline_response_pending)

    async def test_live_voice_update_rejects_a_reserved_fused_audio_owner(self) -> None:
        controller = _controller(voices=frozenset({_VOICE, _OTHER_VOICE}))
        serializer, emitted = _serializer(controller=controller)
        gate = RealtimeManualResponseGate(controller=controller)
        gate.bind_tts_service(object())
        serializer.set_response_gate(gate)
        reservation_id = gate.reserve_service_audio_response()

        frame = await serializer.deserialize(
            json.dumps(
                {
                    "type": "session.update",
                    "session": {"audio": {"output": {"voice": _OTHER_VOICE}}},
                }
            )
        )

        self.assertIsNone(frame)
        self.assertEqual(emitted[-1]["error"]["code"], "immutable_field")
        self.assertEqual(controller.public_session()["audio"]["output"]["voice"], _VOICE)
        gate.release_service_audio_response(reservation_id)

    async def test_pre_ga_flat_audio_formats_are_rejected_not_canonicalized(self) -> None:
        controller = _controller()
        serializer, emitted = _serializer(controller=controller)

        frame = await serializer.deserialize(
            json.dumps(
                {
                    "type": "session.update",
                    "session": {
                        "input_audio_format": {"type": "audio/pcm", "rate": 16000},
                        "output_audio_format": {"type": "audio/pcm", "rate": 48000},
                    },
                }
            )
        )

        self.assertIsNone(frame)
        self.assertEqual(emitted[-1]["error"]["code"], "unknown_parameter")
        self.assertEqual(controller.public_session()["audio"]["input"]["format"]["rate"], 24000)
        self.assertEqual(controller.public_session()["audio"]["output"]["format"]["rate"], 24000)

    async def test_noncanonical_rate_is_rejected_without_resetting_resampler(self) -> None:
        controller = _controller()
        serializer, emitted = _serializer(controller=controller)
        uplink_before = serializer._resampler._uplink
        downlink_before = serializer._resampler._downlink

        frame = await serializer.deserialize(
            json.dumps(
                {
                    "type": "session.update",
                    "session": {
                        "audio": {
                            "input": {"format": {"type": "audio/pcm", "rate": 16000}},
                            "output": {"format": {"type": "audio/pcm", "rate": 16000}},
                        }
                    },
                }
            )
        )

        self.assertIsNone(frame)
        self.assertEqual(emitted[-1]["error"]["code"], "unsupported_capability")
        self.assertIs(serializer._resampler._uplink, uplink_before)
        self.assertIs(serializer._resampler._downlink, downlink_before)

    async def test_existing_transcription_selector_can_be_repeated(self) -> None:
        controller = _controller(input_transcription_model="nvidia/nemotron-asr")
        serializer, emitted = _serializer(controller=controller)
        selector = {"model": "nvidia/nemotron-asr"}

        frame = await serializer.deserialize(
            json.dumps(
                {
                    "type": "session.update",
                    "session": {"audio": {"input": {"transcription": selector}}},
                }
            )
        )

        self.assertIsNone(frame)
        self.assertEqual(emitted[-1]["type"], "session.updated")
        self.assertEqual(emitted[-1]["session"]["audio"]["input"]["transcription"], selector)

    async def test_pre_ga_transcription_selector_is_rejected(self) -> None:
        serializer, emitted = _serializer()
        frame = await serializer.deserialize(
            json.dumps(
                {
                    "type": "session.update",
                    "session": {"input_audio_transcription": {"model": "whisper-1"}},
                }
            )
        )
        self.assertIsNone(frame)
        self.assertEqual(emitted[-1]["error"]["code"], "unknown_parameter")

    async def test_unknown_voice_is_rejected_without_fallback(self) -> None:
        controller = _controller(voices=frozenset({_VOICE, _OTHER_VOICE}))
        serializer, emitted = _serializer(controller=controller)

        frame = await serializer.deserialize(
            json.dumps(
                {
                    "type": "session.update",
                    "session": {"audio": {"output": {"voice": "alloy"}}},
                }
            )
        )

        self.assertIsNone(frame)
        self.assertEqual(emitted[-1]["error"]["code"], "unsupported_capability")
        self.assertEqual(controller.public_session()["audio"]["output"]["voice"], _VOICE)
        self.assertEqual(controller.runtime_config["tts_voice_id"], _VOICE)

    async def test_mid_session_instructions_clear_updates_wire_and_owned_prompt_only(self) -> None:
        controller = _controller()
        controller.apply_session_update({"instructions": "public default"})
        serializer, emitted = _serializer(controller=controller)

        def render(instructions: str) -> list[dict[str, str]]:
            return [
                {"role": "system", "content": "trusted runtime control"},
                {"role": "user", "content": instructions},
            ]

        conversation_message = {"role": "user", "content": "already said"}
        context = LLMContext([*render("public default"), conversation_message])
        gate = RealtimeManualResponseGate(controller=controller)
        gate.push_frame = AsyncMock()
        serializer.bind_context(context, instructions_renderer=render)
        serializer.set_response_gate(gate)

        updated_frame = await serializer.deserialize(
            json.dumps(
                {
                    "type": "session.update",
                    "session": {"instructions": "live prompt"},
                }
            )
        )
        self.assertIsNone(updated_frame)
        self.assertEqual(controller.public_session()["instructions"], "live prompt")
        self.assertEqual(context.get_messages(), [*render("live prompt"), conversation_message])

        frame = await serializer.deserialize(
            json.dumps(
                {
                    "type": "session.update",
                    "session": {"instructions": ""},
                }
            )
        )

        self.assertIsNone(frame)
        self.assertEqual(emitted[-1]["type"], SERVER_SESSION_UPDATED)
        self.assertEqual(emitted[-1]["session"]["instructions"], "")
        self.assertEqual(controller.public_session()["instructions"], "")
        self.assertEqual(
            context.get_messages(),
            [
                {"role": "system", "content": "trusted runtime control"},
                {"role": "user", "content": ""},
                conversation_message,
            ],
        )
        self.assertEqual(controller.runtime_config["prompt_content"], "")

    async def test_mid_session_rejects_manual_turn_detection(self) -> None:
        controller = _controller()
        serializer, emitted = _serializer(controller=controller)
        frame = await serializer.deserialize(
            json.dumps(
                {
                    "type": "session.update",
                    "session": {"audio": {"input": {"turn_detection": None}}},
                }
            )
        )
        self.assertIsNone(frame)
        self.assertEqual(emitted[-1]["error"]["code"], "unsupported_capability")
        self.assertEqual(controller.public_session()["audio"]["input"]["turn_detection"], _SERVER_VAD_DEFAULTS)

    async def test_mid_session_rejects_server_vad_tuning_change(self) -> None:
        controller = _controller()
        serializer, emitted = _serializer(controller=controller)

        frame = await serializer.deserialize(
            json.dumps(
                {
                    "type": "session.update",
                    "session": {
                        "audio": {
                            "input": {
                                "turn_detection": {
                                    "type": "server_vad",
                                    "threshold": 0.65,
                                }
                            }
                        }
                    },
                }
            )
        )

        self.assertIsNone(frame)
        self.assertEqual(emitted[-1]["error"]["code"], "unsupported_live_session_update")
        self.assertEqual(
            emitted[-1]["error"]["param"],
            "session.audio.input.turn_detection",
        )
        self.assertEqual(controller.public_session()["audio"]["input"]["turn_detection"], _SERVER_VAD_DEFAULTS)

    async def test_live_max_output_tokens_requires_installed_response_runtime(self) -> None:
        controller = _controller()
        serializer, emitted = _serializer(controller=controller)

        frame = await serializer.deserialize(
            json.dumps(
                {
                    "type": "session.update",
                    "session": {"max_output_tokens": 128},
                }
            )
        )

        self.assertIsNone(frame)
        self.assertEqual(emitted[-1]["error"]["code"], "response_gate_missing")
        self.assertEqual(controller.public_session()["max_output_tokens"], "inf")

    async def test_response_output_override_requires_pipeline_response_gate(self) -> None:
        serializer, emitted = _serializer()
        frame = await serializer.deserialize(
            json.dumps(
                {
                    "type": "response.create",
                    "response": {"output_modalities": ["text"]},
                }
            )
        )
        self.assertIsNone(frame)
        self.assertEqual(emitted[-1]["error"]["code"], "response_gate_missing")

    async def test_append_rejects_oversized_payload(self) -> None:
        serializer, emitted = _serializer()
        with patch("realtime.serializer.MAX_AUDIO_APPEND_BYTES", 4):
            frame = await serializer.deserialize(
                json.dumps(
                    {
                        "type": "input_audio_buffer.append",
                        "audio": base64.b64encode(b"\x01\x00" * 3).decode("ascii"),
                    }
                )
            )
        self.assertIsNone(frame)
        self.assertEqual(emitted[-1]["error"]["code"], "input_buffer_overflow")

    async def test_append_rejects_invalid_base64_and_odd_pcm16(self) -> None:
        serializer, emitted = _serializer()
        self.assertIsNone(
            await serializer.deserialize(json.dumps({"type": "input_audio_buffer.append", "audio": "!!!!"}))
        )
        self.assertEqual(emitted[-1]["error"]["code"], "invalid_audio")
        self.assertIsNone(
            await serializer.deserialize(
                json.dumps(
                    {
                        "type": "input_audio_buffer.append",
                        "audio": base64.b64encode(b"\x00").decode("ascii"),
                    }
                )
            )
        )
        self.assertEqual(emitted[-1]["error"]["code"], "invalid_audio")


class GatewayHandoffTests(unittest.IsolatedAsyncioTestCase):
    @staticmethod
    def _sanitize(data: dict[str, Any], fallback_example_key: str = "") -> dict[str, Any]:
        output = dict(data)
        output.setdefault("pipeline_mode", "generic-assistant")
        output.setdefault("model_id", _MODEL)
        output.setdefault("tts_voice_id", _VOICE)
        return output

    async def test_too_many_failed_session_updates_closes_socket(self) -> None:
        update = json.dumps(
            {
                "type": "session.update",
                "session": {"nvidia": {"pipeline_mode": "generic-assistant"}},
            }
        )
        ws = FakeWebSocket([update])

        async def ensure_ready(config: dict[str, Any]) -> None:  # noqa: ARG001
            raise RuntimeError("services down")

        with patch.dict("os.environ", {"REALTIME_MAX_REJECTED_EVENTS": "1"}, clear=False):
            await handle_realtime_websocket(
                ws,
                sanitize_session_config=self._sanitize,
                ensure_services_ready=ensure_ready,
            )

        self.assertTrue(ws.closed)
        self.assertEqual(ws.close_code, 1008)
        self.assertEqual(ws.close_reason, "too many rejected session updates")


if __name__ == "__main__":
    unittest.main()
