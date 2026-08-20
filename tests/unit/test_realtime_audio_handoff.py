# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

# ruff: noqa: D100, D101, D102, D103

from __future__ import annotations

import asyncio
import base64
import json
import unittest
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

from pipecat.frames.frames import InputAudioRawFrame, OutputAudioRawFrame, TTSStoppedFrame
from pipecat.observers.base_observer import FramePushed
from pipecat.processors.frame_processor import FrameDirection
from pipecat.transports.base_output import BaseOutputTransport
from realtime_helpers import FakeWebSocket

from realtime.audio import (
    PIPELINE_PCM_RATE,
    AudioResampler,
    decode_base64_audio,
    encode_base64_audio,
)
from realtime.events import SERVER_ERROR, SERVER_SESSION_CREATED, SERVER_SESSION_UPDATED
from realtime.gateway import handle_realtime_websocket
from realtime.observer import RealtimeLifecycleObserver
from realtime.serializer import RealtimeFrameSerializer


class AudioHelperTests(unittest.IsolatedAsyncioTestCase):
    async def test_base64_roundtrip(self) -> None:
        raw = b"\x00\x01\x02\x03" * 10
        encoded = encode_base64_audio(raw)
        self.assertEqual(decode_base64_audio(encoded), raw)

    async def test_resample_identity_at_16k(self) -> None:
        resampler = AudioResampler()
        pcm = b"\x00\x00" * 160  # 10ms mono 16-bit at 16k
        out = await resampler.to_pipeline(pcm, PIPELINE_PCM_RATE)
        self.assertEqual(out, pcm)

    async def test_resample_uses_configured_pipeline_rate(self) -> None:
        resampler = AudioResampler()
        pcm = b"\x00\x00" * 80
        with patch.object(resampler, "_uplink") as uplink:
            uplink.resample = AsyncMock(return_value=b"up")
            out = await resampler.to_pipeline(pcm, 24000, pipeline_rate=8000)
        self.assertEqual(out, b"up")
        uplink.resample.assert_awaited_once_with(pcm, 24000, 8000)

        with patch.object(resampler, "_downlink") as downlink:
            downlink.resample = AsyncMock(return_value=b"down")
            out = await resampler.from_pipeline(pcm, 24000, pipeline_rate=8000)
        self.assertEqual(out, b"down")
        downlink.resample.assert_awaited_once_with(pcm, 8000, 24000)

    async def test_reset_replaces_stream_resamplers(self) -> None:
        resampler = AudioResampler()
        uplink_before = resampler._uplink
        downlink_before = resampler._downlink
        resampler.reset()
        self.assertIsNot(resampler._uplink, uplink_before)
        self.assertIsNot(resampler._downlink, downlink_before)


class SerializerAudioTests(unittest.IsolatedAsyncioTestCase):
    async def test_append_produces_input_audio_frame(self) -> None:
        emitted: list[dict[str, Any]] = []

        async def emit(event: dict[str, Any]) -> None:
            emitted.append(event)

        ser = RealtimeFrameSerializer(
            session_view={"audio": {"input": {"format": {"type": "audio/pcm", "rate": 16000}}}}
        )
        ser.set_emit(emit)
        # 20ms of silence at 16kHz s16le mono
        pcm = b"\x00\x00" * 320
        msg = json.dumps({"type": "input_audio_buffer.append", "audio": base64.b64encode(pcm).decode("ascii")})
        frame = await ser.deserialize(msg)
        self.assertIsInstance(frame, InputAudioRawFrame)
        assert isinstance(frame, InputAudioRawFrame)
        self.assertEqual(frame.sample_rate, 16000)
        self.assertEqual(frame.audio, pcm)

    async def test_output_audio_emits_delta_for_in_progress_response(self) -> None:
        emitted: list[dict[str, Any]] = []

        async def emit(event: dict[str, Any]) -> None:
            emitted.append(event)

        ser = RealtimeFrameSerializer(
            session_view={"audio": {"output": {"format": {"type": "audio/pcm", "rate": 16000}}}}
        )
        ser.set_emit(emit)
        from realtime.lifecycle import announce_response

        await announce_response(ser.conversation, emit)
        emitted.clear()
        pcm = b"\x00\x00" * 160
        payload = await ser.serialize(OutputAudioRawFrame(audio=pcm, sample_rate=16000, num_channels=1))
        self.assertIsNone(payload)
        delta = next(e for e in emitted if e["type"] == "response.output_audio.delta")
        self.assertEqual(delta["type"], "response.output_audio.delta")
        types = [e["type"] for e in emitted]
        self.assertNotIn("response.created", types)
        self.assertIn("response.audio.delta", types)

    async def test_output_audio_without_announced_response_is_dropped(self) -> None:
        emitted: list[dict[str, Any]] = []

        async def emit(event: dict[str, Any]) -> None:
            emitted.append(event)

        ser = RealtimeFrameSerializer(
            session_view={"audio": {"output": {"format": {"type": "audio/pcm", "rate": 16000}}}}
        )
        ser.set_emit(emit)
        pcm = b"\x00\x00" * 160
        payload = await ser.serialize(OutputAudioRawFrame(audio=pcm, sample_rate=16000, num_channels=1))
        self.assertIsNone(payload)
        self.assertEqual(emitted, [])

    async def test_trailing_output_audio_after_done_is_dropped(self) -> None:
        emitted: list[dict[str, Any]] = []

        async def emit(event: dict[str, Any]) -> None:
            emitted.append(event)

        ser = RealtimeFrameSerializer(
            session_view={"audio": {"output": {"format": {"type": "audio/pcm", "rate": 16000}}}}
        )
        ser.set_emit(emit)
        from realtime.lifecycle import announce_response, finish_response

        await announce_response(ser.conversation, emit)
        await finish_response(ser.conversation, emit, status="completed")
        emitted.clear()

        pcm = b"\x00\x00" * 160
        payload = await ser.serialize(OutputAudioRawFrame(audio=pcm, sample_rate=16000, num_channels=1))
        self.assertIsNone(payload)
        self.assertEqual(emitted, [])
        self.assertIsNone(ser.conversation.response_id)

    async def test_audio_delta_precedes_transport_drained_done_sequence(self) -> None:
        emitted: list[dict[str, Any]] = []

        async def emit(event: dict[str, Any]) -> None:
            emitted.append(event)

        ser = RealtimeFrameSerializer(
            session_view={"audio": {"output": {"format": {"type": "audio/pcm", "rate": 16000}}}}
        )
        ser.set_emit(emit)
        observer = RealtimeLifecycleObserver(emit=emit, conversation=ser.conversation)
        from realtime.lifecycle import announce_response

        await announce_response(ser.conversation, emit)
        await ser.serialize(
            OutputAudioRawFrame(
                audio=b"\x00\x00" * 160,
                sample_rate=16000,
                num_channels=1,
            )
        )
        stopped = TTSStoppedFrame()
        for source in (MagicMock(), MagicMock(spec=BaseOutputTransport)):
            await observer.on_push_frame(
                FramePushed(
                    source=source,
                    destination=MagicMock(),
                    frame=stopped,
                    direction=FrameDirection.DOWNSTREAM,
                    timestamp=0,
                )
            )

        types = [event["type"] for event in emitted]
        self.assertLess(
            types.index("response.output_audio.delta"),
            types.index("response.output_audio.done"),
        )
        self.assertLess(
            types.index("response.output_audio.done"),
            types.index("response.done"),
        )
        self.assertNotIn("response.output_text.delta", types)
        self.assertNotIn("response.output_text.done", types)

    async def test_commit_empty_errors(self) -> None:
        emitted: list[dict[str, Any]] = []

        async def emit(event: dict[str, Any]) -> None:
            emitted.append(event)

        ser = RealtimeFrameSerializer()
        ser.set_emit(emit)
        frame = await ser.deserialize(json.dumps({"type": "input_audio_buffer.commit"}))
        self.assertIsNone(frame)
        self.assertEqual(emitted[0]["type"], "error")
        self.assertEqual(emitted[0]["error"]["code"], "input_audio_buffer_commit_empty")

    async def test_commit_after_append_does_not_duplicate_vad_commit(self) -> None:
        emitted: list[dict[str, Any]] = []

        async def emit(event: dict[str, Any]) -> None:
            emitted.append(event)

        ser = RealtimeFrameSerializer(
            session_view={
                "audio": {
                    "input": {
                        "format": {"type": "audio/pcm", "rate": 16000},
                        "turn_detection": {"type": "server_vad"},
                    }
                },
            }
        )
        ser.set_emit(emit)
        pcm = b"\x00\x00" * 320
        appended = await ser.deserialize(
            json.dumps({"type": "input_audio_buffer.append", "audio": base64.b64encode(pcm).decode("ascii")})
        )
        frame = await ser.deserialize(json.dumps({"type": "input_audio_buffer.commit"}))
        self.assertIsInstance(appended, InputAudioRawFrame)
        self.assertIsNone(frame)
        self.assertNotIn("input_audio_buffer.committed", [e["type"] for e in emitted])

    async def test_append_always_streams_in_server_vad_mode(self) -> None:
        emitted: list[dict[str, Any]] = []

        async def emit(event: dict[str, Any]) -> None:
            emitted.append(event)

        ser = RealtimeFrameSerializer(
            session_view={
                "audio": {
                    "input": {
                        "turn_detection": {"type": "server_vad"},
                        "format": {"type": "audio/pcm", "rate": PIPELINE_PCM_RATE},
                    }
                }
            }
        )
        ser.set_emit(emit)
        pcm = b"\x01\x00" * 80
        frame = await ser.deserialize(
            json.dumps(
                {
                    "type": "input_audio_buffer.append",
                    "audio": base64.b64encode(pcm).decode("ascii"),
                }
            )
        )
        self.assertIsInstance(frame, InputAudioRawFrame)

    async def test_clear_resets_uncommitted_byte_count(self) -> None:
        emitted: list[dict[str, Any]] = []

        async def emit(event: dict[str, Any]) -> None:
            emitted.append(event)

        ser = RealtimeFrameSerializer(
            session_view={
                "audio": {
                    "input": {
                        "format": {"type": "audio/pcm", "rate": PIPELINE_PCM_RATE},
                        "turn_detection": {"type": "server_vad"},
                    }
                },
            }
        )
        ser.set_emit(emit)
        pcm = b"\x01\x00" * 40
        await ser.deserialize(
            json.dumps(
                {
                    "type": "input_audio_buffer.append",
                    "audio": base64.b64encode(pcm).decode("ascii"),
                }
            )
        )
        self.assertGreater(ser._bytes_since_commit, 0)
        frame = await ser.deserialize(json.dumps({"type": "input_audio_buffer.clear"}))
        self.assertIsNone(frame)
        self.assertEqual(ser._bytes_since_commit, 0)
        self.assertEqual(emitted[-1]["type"], "input_audio_buffer.cleared")

    async def test_mid_session_partial_audio_update_deep_merges(self) -> None:
        emitted: list[dict[str, Any]] = []

        async def emit(event: dict[str, Any]) -> None:
            emitted.append(event)

        ser = RealtimeFrameSerializer(
            session_view={
                "audio": {
                    "input": {
                        "format": {"type": "audio/pcm", "rate": PIPELINE_PCM_RATE},
                        "turn_detection": {"type": "server_vad"},
                    },
                    "output": {
                        "format": {"type": "audio/pcm", "rate": PIPELINE_PCM_RATE},
                        "voice": "",
                    },
                }
            }
        )
        ser.set_emit(emit)
        with patch("realtime.serializer.resolve_realtime_tts_voice", return_value="Magpie-Multilingual.EN-US.Aria"):
            frame = await ser.deserialize(
                json.dumps(
                    {
                        "type": "session.update",
                        "session": {"audio": {"output": {"voice": "Magpie-Multilingual.EN-US.Aria"}}},
                    }
                )
            )
        from pipecat.frames.frames import TTSUpdateSettingsFrame

        self.assertIsInstance(frame, TTSUpdateSettingsFrame)
        audio = ser._session_view["audio"]
        self.assertEqual(audio["input"]["turn_detection"], {"type": "server_vad"})
        self.assertEqual(audio["input"]["format"]["rate"], PIPELINE_PCM_RATE)
        self.assertEqual(audio["output"]["voice"], "Magpie-Multilingual.EN-US.Aria")
        self.assertEqual(emitted[-1]["type"], "session.updated")

    async def test_mid_session_rate_change_resets_resampler(self) -> None:
        emitted: list[dict[str, Any]] = []

        async def emit(event: dict[str, Any]) -> None:
            emitted.append(event)

        ser = RealtimeFrameSerializer(
            session_view={
                "audio": {
                    "input": {"format": {"type": "audio/pcm", "rate": 24000}},
                    "output": {"format": {"type": "audio/pcm", "rate": 24000}},
                }
            }
        )
        ser.set_emit(emit)
        uplink_before = ser._resampler._uplink
        frame = await ser.deserialize(
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
        self.assertEqual(ser._client_in_rate, 16000)
        self.assertEqual(ser._client_out_rate, 16000)
        self.assertIsNot(ser._resampler._uplink, uplink_before)
        self.assertEqual(emitted[-1]["type"], "session.updated")

    async def test_mid_session_transcription_selector_is_accepted_noop(self) -> None:
        emitted: list[dict[str, Any]] = []

        async def emit(event: dict[str, Any]) -> None:
            emitted.append(event)

        ser = RealtimeFrameSerializer()
        ser.set_emit(emit)
        frame = await ser.deserialize(
            json.dumps(
                {
                    "type": "session.update",
                    "session": {
                        "input_audio_transcription": {"model": "whisper-1"},
                    },
                }
            )
        )
        self.assertIsNone(frame)
        self.assertEqual(emitted[-1]["type"], "session.updated")
        self.assertEqual(
            emitted[-1]["session"]["input_audio_transcription"],
            {"model": "whisper-1"},
        )

    async def test_mid_session_unknown_voice_falls_back(self) -> None:
        emitted: list[dict[str, Any]] = []

        async def emit(event: dict[str, Any]) -> None:
            emitted.append(event)

        ser = RealtimeFrameSerializer(
            session_view={
                "audio": {
                    "input": {"format": {"type": "audio/pcm", "rate": PIPELINE_PCM_RATE}},
                    "output": {"format": {"type": "audio/pcm", "rate": PIPELINE_PCM_RATE}, "voice": ""},
                }
            },
            runtime_config={
                "tts_server": "tts.internal:443",
                "tts_function_id": "tts-fn",
                "tts_model": "magpie",
            },
        )
        ser.set_emit(emit)
        seen: dict[str, Any] = {}

        def _resolve(config, *, voice_was_set=False, tts_routing_changed=False):  # noqa: ARG001
            seen.update(config)
            config["tts_voice_id"] = "Magpie-Multilingual.EN-US.Aria"
            return "Magpie-Multilingual.EN-US.Aria"

        with patch("realtime.serializer.resolve_realtime_tts_voice", side_effect=_resolve):
            frame = await ser.deserialize(json.dumps({"type": "session.update", "session": {"voice": "alloy"}}))
        from pipecat.frames.frames import TTSUpdateSettingsFrame

        self.assertIsInstance(frame, TTSUpdateSettingsFrame)
        self.assertEqual(emitted[-1]["type"], "session.updated")
        self.assertEqual(ser._session_view["audio"]["output"]["voice"], "Magpie-Multilingual.EN-US.Aria")
        self.assertEqual(ser._session_view.get("voice"), "Magpie-Multilingual.EN-US.Aria")
        self.assertEqual(seen["tts_server"], "tts.internal:443")
        self.assertEqual(seen["tts_function_id"], "tts-fn")

    async def test_mid_session_rejects_unapplied_agent_config(self) -> None:
        emitted: list[dict[str, Any]] = []

        async def emit(event: dict[str, Any]) -> None:
            emitted.append(event)

        ser = RealtimeFrameSerializer(session_view={"instructions": "old"})
        ser.set_emit(emit)
        frame = await ser.deserialize(
            json.dumps(
                {
                    "type": "session.update",
                    "session": {"instructions": "new instructions", "temperature": 0.2},
                }
            )
        )
        self.assertIsNone(frame)
        self.assertEqual(emitted[-1]["type"], "error")
        self.assertEqual(emitted[-1]["error"]["code"], "unsupported_live_session_update")
        self.assertEqual(ser._session_view.get("instructions"), "old")

    async def test_mid_session_rejects_turn_detection_change(self) -> None:
        emitted: list[dict[str, Any]] = []

        async def emit(event: dict[str, Any]) -> None:
            emitted.append(event)

        ser = RealtimeFrameSerializer(
            session_view={
                "instructions": "Be brief.",
                "tools": [{"type": "function", "name": "get_weather"}],
                "turn_detection": {"type": "server_vad"},
                "audio": {"input": {"turn_detection": {"type": "server_vad"}}},
                "nvidia": {"pipeline_mode": "generic-assistant"},
            }
        )
        ser.set_emit(emit)
        frame = await ser.deserialize(
            json.dumps(
                {
                    "type": "session.update",
                    "session": {
                        "instructions": "Be brief.",
                        "tools": [{"type": "function", "name": "get_weather"}],
                        "turn_detection": None,
                        "nvidia": {"pipeline_mode": "generic-assistant"},
                    },
                }
            )
        )
        self.assertIsNone(frame)
        self.assertEqual(emitted[-1]["type"], "error")
        self.assertEqual(emitted[-1]["error"]["code"], "unsupported_live_session_update")
        self.assertEqual(ser._session_view["turn_detection"], {"type": "server_vad"})
        self.assertNotIn("temperature", ser._session_view)

    async def test_response_create_rejects_overrides(self) -> None:
        emitted: list[dict[str, Any]] = []

        async def emit(event: dict[str, Any]) -> None:
            emitted.append(event)

        ser = RealtimeFrameSerializer()
        ser.conversation.open_client_text()
        ser.set_emit(emit)
        frame = await ser.deserialize(
            json.dumps(
                {
                    "type": "response.create",
                    "response": {"instructions": "only this turn"},
                }
            )
        )
        self.assertIsNone(frame)
        self.assertEqual(emitted[-1]["type"], "error")
        self.assertEqual(emitted[-1]["error"]["code"], "unsupported_response_override")

    async def test_append_rejects_oversized_payload(self) -> None:
        from realtime.audio import MAX_PENDING_INPUT_BYTES

        emitted: list[dict[str, Any]] = []

        async def emit(event: dict[str, Any]) -> None:
            emitted.append(event)

        ser = RealtimeFrameSerializer(
            session_view={
                "audio": {
                    "input": {
                        "turn_detection": {"type": "server_vad"},
                        "format": {"type": "audio/pcm", "rate": PIPELINE_PCM_RATE},
                    }
                }
            }
        )
        ser.set_emit(emit)
        frame = await ser.deserialize(
            json.dumps(
                {
                    "type": "input_audio_buffer.append",
                    "audio": base64.b64encode(b"\x01\x00" * (MAX_PENDING_INPUT_BYTES // 2 + 1)).decode("ascii"),
                }
            )
        )
        self.assertIsNone(frame)
        self.assertEqual(emitted[-1]["type"], "error")
        self.assertEqual(emitted[-1]["error"]["code"], "input_buffer_overflow")

    async def test_append_rejects_invalid_base64_and_odd_pcm16(self) -> None:
        emitted: list[dict[str, Any]] = []

        async def emit(event: dict[str, Any]) -> None:
            emitted.append(event)

        ser = RealtimeFrameSerializer()
        ser.set_emit(emit)
        self.assertIsNone(await ser.deserialize(json.dumps({"type": "input_audio_buffer.append", "audio": "!!!!"})))
        self.assertEqual(emitted[-1]["error"]["code"], "invalid_audio")
        self.assertIsNone(
            await ser.deserialize(
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
    async def test_readiness_failure_keeps_socket_open_and_allows_retry(self) -> None:
        ws = FakeWebSocket(
            [
                json.dumps(
                    {
                        "type": "session.update",
                        "event_id": "e1",
                        "session": {"nvidia": {"pipeline_mode": "generic-assistant"}},
                    }
                ),
                json.dumps(
                    {
                        "type": "session.update",
                        "event_id": "e2",
                        "session": {"nvidia": {"pipeline_mode": "generic-assistant"}},
                    }
                ),
            ]
        )
        calls = {"ready": 0, "bot": 0}

        def sanitize(data: dict, fallback_example_key: str = "") -> dict:
            out = dict(data)
            out.setdefault("pipeline_mode", "generic-assistant")
            return out

        async def ensure_ready(config: dict) -> None:
            calls["ready"] += 1
            if calls["ready"] == 1:
                raise RuntimeError("TTS not ready")

        async def start_bot(websocket, config, session_view) -> None:
            calls["bot"] += 1
            # Consume no further messages; gateway returns after handoff.

        await handle_realtime_websocket(
            ws,
            sanitize_session_config=sanitize,
            ensure_services_ready=ensure_ready,
            start_bot=start_bot,
        )

        types = [m["type"] for m in ws.sent]
        self.assertEqual(types[0], SERVER_SESSION_CREATED)
        self.assertIn(SERVER_ERROR, types)
        self.assertEqual(ws.sent[types.index(SERVER_ERROR)]["error"]["code"], "services_not_ready")
        self.assertIn(SERVER_SESSION_UPDATED, types)
        self.assertEqual(calls["bot"], 1)
        self.assertFalse(ws.closed)

    async def test_too_many_failed_session_updates_closes_socket(self) -> None:
        update = json.dumps(
            {
                "type": "session.update",
                "session": {"nvidia": {"pipeline_mode": "generic-assistant"}},
            }
        )
        ws = FakeWebSocket([update, update])

        def sanitize(data: dict, fallback_example_key: str = "") -> dict:
            out = dict(data)
            out.setdefault("pipeline_mode", "generic-assistant")
            return out

        async def ensure_ready(config: dict) -> None:
            raise RuntimeError("services down")

        with patch.dict(
            "os.environ",
            {"REALTIME_MAX_REJECTED_EVENTS": "1"},
            clear=False,
        ):
            await handle_realtime_websocket(
                ws,
                sanitize_session_config=sanitize,
                ensure_services_ready=ensure_ready,
            )

        self.assertTrue(ws.closed)
        self.assertEqual(ws.close_code, 1008)
        self.assertIn("failed session.update", ws.close_reason)

    async def test_successful_update_hands_off_to_bot(self) -> None:
        ws = FakeWebSocket(
            [
                json.dumps(
                    {
                        "type": "session.update",
                        "session": {
                            "instructions": "Hi",
                            "voice": "Magpie-Multilingual.EN-US.Aria",
                            "nvidia": {
                                "pipeline_mode": "generic-assistant",
                                "prompt_key": "generic_assistant_without_tools",
                            },
                        },
                    }
                )
            ]
        )
        started = asyncio.Event()

        async def start_bot(websocket, config, session_view) -> None:
            self.assertEqual(config["pipeline_mode"], "generic-assistant")
            self.assertEqual(config.get("prompt_content"), "Hi")
            self.assertEqual(config.get("tts_voice_id"), "Magpie-Multilingual.EN-US.Aria")
            self.assertEqual(session_view.get("instructions"), "Hi")
            self.assertEqual(session_view.get("voice"), "Magpie-Multilingual.EN-US.Aria")
            self.assertIn("id", session_view)
            started.set()

        with patch("realtime.gateway.resolve_realtime_tts_voice", return_value=None):
            await handle_realtime_websocket(
                ws,
                sanitize_session_config=lambda data, **_: {**data, "pipeline_mode": "generic-assistant"},
                ensure_services_ready=lambda config: asyncio.sleep(0),
                start_bot=start_bot,
            )
        self.assertTrue(started.is_set())
        self.assertEqual(ws.sent[0]["type"], SERVER_SESSION_CREATED)
        self.assertEqual(ws.sent[1]["type"], SERVER_SESSION_UPDATED)


if __name__ == "__main__":
    unittest.main()
