# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

# ruff: noqa: D100, D101, D102

from __future__ import annotations

import base64
import json
import unittest
from typing import Any
from unittest.mock import AsyncMock

import numpy as np
from pipecat.frames.frames import InputAudioRawFrame, OutputAudioRawFrame

from realtime.audio import (
    DEFAULT_CLIENT_PCM_RATE,
    G711_PCM_RATE,
    PIPELINE_PCM_RATE,
    AudioResampler,
    decode_base64_audio,
    encode_base64_audio,
    extract_client_input_format_type,
    extract_client_output_format_type,
    extract_client_output_pcm_rate,
    extract_client_pcm_rate,
    max_base64_audio_chars,
    max_pending_input_bytes,
    require_audio_format_rate,
    require_supported_pipeline_pcm_rate,
)
from realtime.controller import RealtimeSessionController
from realtime.serializer import RealtimeFrameSerializer
from realtime.session import AudioFormatCapability, RealtimeSessionCapabilities

_MODEL = "test-realtime-model"
_VOICE = "Magpie-Multilingual.EN-US.Aria"


def _codec_controller() -> RealtimeSessionController:
    formats = frozenset(
        {
            AudioFormatCapability("audio/pcm", 8000),
            AudioFormatCapability("audio/pcm", 16000),
            AudioFormatCapability("audio/pcm", DEFAULT_CLIENT_PCM_RATE),
            AudioFormatCapability("audio/pcma"),
            AudioFormatCapability("audio/pcmu"),
        }
    )
    return RealtimeSessionController(
        model=_MODEL,
        voice=_VOICE,
        runtime_config={
            "pipeline_mode": "generic-assistant",
            "model_id": _MODEL,
            "tts_voice_id": _VOICE,
        },
        capabilities=RealtimeSessionCapabilities(
            input_formats=formats,
            output_formats=formats,
            voices=frozenset({_VOICE}),
        ),
    )


def _codec_serializer(
    controller: RealtimeSessionController,
) -> tuple[RealtimeFrameSerializer, list[dict[str, Any]]]:
    emitted: list[dict[str, Any]] = []

    async def emit(event: dict[str, Any]) -> None:
        emitted.append(event)

    async def emit_batch(events: list[dict[str, Any]]) -> None:
        emitted.extend(events)

    serializer = RealtimeFrameSerializer(controller=controller)
    serializer.set_emit(emit, emit_batch)
    return serializer, emitted


class RealtimeAudioFormatTests(unittest.TestCase):
    def test_g711_rate_is_exactly_8000_hz(self) -> None:
        for format_type in ("audio/pcmu", "audio/pcma"):
            self.assertEqual(require_audio_format_rate(format_type, None), G711_PCM_RATE)
            self.assertEqual(
                require_audio_format_rate(format_type, G711_PCM_RATE),
                G711_PCM_RATE,
            )
            for invalid in (16000, 8000.0, True):
                with self.subTest(format=format_type, invalid=invalid), self.assertRaises(ValueError):
                    require_audio_format_rate(format_type, invalid)

    def test_pipeline_rates_fail_closed(self) -> None:
        self.assertEqual(require_supported_pipeline_pcm_rate(16000), 16000)
        self.assertEqual(require_supported_pipeline_pcm_rate(22050), 22050)
        for invalid in (44100, 22050.0, True):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                require_supported_pipeline_pcm_rate(invalid)

    def test_session_view_extracts_format_and_fixed_rate(self) -> None:
        view = {
            "audio": {
                "input": {"format": {"type": "audio/pcmu"}},
                "output": {"format": {"type": "audio/pcma"}},
            }
        }
        self.assertEqual(extract_client_input_format_type(view), "audio/pcmu")
        self.assertEqual(extract_client_output_format_type(view), "audio/pcma")
        self.assertEqual(extract_client_pcm_rate(view), G711_PCM_RATE)
        self.assertEqual(extract_client_output_pcm_rate(view), G711_PCM_RATE)
        self.assertEqual(extract_client_pcm_rate(None), DEFAULT_CLIENT_PCM_RATE)

    def test_format_aware_wire_byte_and_base64_limits(self) -> None:
        self.assertEqual(max_pending_input_bytes("audio/pcm", 24000), 2_880_000)
        self.assertEqual(max_pending_input_bytes("audio/pcmu"), 480_000)
        self.assertEqual(max_pending_input_bytes("audio/pcma", duration_seconds=0.5), 4_000)
        self.assertEqual(max_base64_audio_chars(0), 0)
        self.assertEqual(max_base64_audio_chars(1), 4)
        self.assertEqual(max_base64_audio_chars(3), 4)
        self.assertEqual(max_base64_audio_chars(4), 8)
        for invalid in (0, -1, float("inf"), True):
            with self.subTest(duration=invalid), self.assertRaises(ValueError):
                max_pending_input_bytes("audio/pcmu", duration_seconds=invalid)

    def test_base64_validation_is_format_aware_and_bounded(self) -> None:
        one_g711_sample = base64.b64encode(b"\xff").decode("ascii")
        self.assertEqual(
            decode_base64_audio(one_g711_sample, format_type="audio/pcmu"),
            b"\xff",
        )
        with self.assertRaisesRegex(ValueError, "even number of bytes"):
            decode_base64_audio(one_g711_sample, format_type="audio/pcm")
        with self.assertRaisesRegex(ValueError, "invalid base64"):
            decode_base64_audio("!!!!", format_type="audio/pcma")

        two_bytes = base64.b64encode(b"\x00\x01").decode("ascii")
        with self.assertRaisesRegex(ValueError, "decoded limit"):
            decode_base64_audio(
                two_bytes,
                format_type="audio/pcmu",
                max_decoded_bytes=1,
            )
        with self.assertRaisesRegex(ValueError, "decoded limit"):
            decode_base64_audio(
                base64.b64encode(b"\x00" * 4).decode("ascii"),
                format_type="audio/pcmu",
                max_decoded_bytes=1,
            )

        self.assertEqual(
            encode_base64_audio(b"\xff", format_type="audio/pcmu"),
            one_g711_sample,
        )
        with self.assertRaisesRegex(ValueError, "even number of bytes"):
            encode_base64_audio(b"\x00", format_type="audio/pcm")


class RealtimeAudioCodecTests(unittest.IsolatedAsyncioTestCase):
    async def test_streaming_uplink_flush_preserves_every_input_sample(self) -> None:
        chunk_count = 26

        async def assert_complete_stream(
            *,
            format_type: str,
            wire_rate: int,
            wire_chunk: bytes,
        ) -> None:
            resampler = AudioResampler()
            output = bytearray()
            for _ in range(chunk_count):
                output.extend(
                    await resampler.to_pipeline(
                        wire_chunk,
                        wire_rate,
                        pipeline_rate=PIPELINE_PCM_RATE,
                        format_type=format_type,
                    )
                )
            output.extend(await resampler.flush_to_pipeline())
            expected = await AudioResampler.complete_input_to_pipeline(
                wire_chunk * chunk_count,
                wire_rate,
                pipeline_rate=PIPELINE_PCM_RATE,
                format_type=format_type,
            )
            actual_samples = np.frombuffer(output, dtype=np.int16)
            expected_samples = np.frombuffer(expected, dtype=np.int16)
            self.assertEqual(actual_samples.size, expected_samples.size)
            np.testing.assert_allclose(actual_samples, expected_samples, rtol=0, atol=2)
            self.assertEqual(await resampler.flush_to_pipeline(), b"")

        cases = (
            ("audio/pcm", 24000, b"\x00\x00" * 4800),
            ("audio/pcmu", G711_PCM_RATE, b"\xff" * 1600),
        )
        for format_type, wire_rate, wire_chunk in cases:
            with self.subTest(format_type=format_type):
                await assert_complete_stream(
                    format_type=format_type,
                    wire_rate=wire_rate,
                    wire_chunk=wire_chunk,
                )

        resampler = AudioResampler()
        uplink_before = resampler._uplink
        await resampler.to_pipeline(b"\x00\x00" * 480, 24000)
        self.assertIsNotNone(uplink_before._stream)
        resampler.reset_uplink()
        self.assertIs(resampler._uplink, uplink_before)
        self.assertIsNone(uplink_before._stream)
        self.assertIsNone(resampler._uplink_signature)

    async def test_pcm_8_16_24khz_preserve_one_second_across_the_pipeline_boundary(self) -> None:
        for wire_rate in (8000, 16000, 24000):
            with self.subTest(wire_rate=wire_rate):
                wire_input = b"\x00\x00" * wire_rate
                pipeline_pcm = await AudioResampler.complete_input_to_pipeline(
                    wire_input,
                    wire_rate,
                    pipeline_rate=PIPELINE_PCM_RATE,
                )
                resampler = AudioResampler()
                wire_output = await resampler.from_pipeline(
                    pipeline_pcm,
                    wire_rate,
                    pipeline_rate=PIPELINE_PCM_RATE,
                )
                wire_output += await resampler.flush_from_pipeline(
                    wire_rate,
                    pipeline_rate=PIPELINE_PCM_RATE,
                )

                self.assertEqual(len(pipeline_pcm) // 2, PIPELINE_PCM_RATE)
                self.assertEqual(len(wire_output) // 2, wire_rate)

    async def test_manual_complete_input_decodes_g711_before_resampling(self) -> None:
        for format_type, silence_byte in (("audio/pcmu", 0xFF), ("audio/pcma", 0xD5)):
            wire_audio = bytes([silence_byte]) * 800
            with self.subTest(format=format_type):
                pcm = await AudioResampler.complete_input_to_pipeline(
                    wire_audio,
                    G711_PCM_RATE,
                    pipeline_rate=PIPELINE_PCM_RATE,
                    format_type=format_type,
                )
                self.assertEqual(len(pcm), 3200)
                self.assertEqual(len(pcm) % 2, 0)

    async def test_downlink_resamples_before_g711_encoding_and_flushes(self) -> None:
        pipeline_pcm = b"\x00\x00" * 1600
        for format_type in ("audio/pcmu", "audio/pcma"):
            with self.subTest(format=format_type):
                resampler = AudioResampler()
                first = await resampler.from_pipeline(
                    pipeline_pcm,
                    G711_PCM_RATE,
                    pipeline_rate=PIPELINE_PCM_RATE,
                    format_type=format_type,
                )
                tail = await resampler.flush_from_pipeline(
                    G711_PCM_RATE,
                    pipeline_rate=PIPELINE_PCM_RATE,
                    format_type=format_type,
                )
                wire_audio = first + tail
                self.assertEqual(len(wire_audio), 800)
                self.assertNotEqual(wire_audio, pipeline_pcm)

    async def test_pcm_identity_behavior_is_preserved(self) -> None:
        pcm = b"\x01\x00" * 160
        resampler = AudioResampler()
        self.assertEqual(
            await resampler.to_pipeline(
                pcm,
                PIPELINE_PCM_RATE,
                format_type="audio/pcm",
            ),
            pcm,
        )
        self.assertEqual(
            await resampler.from_pipeline(
                pcm,
                PIPELINE_PCM_RATE,
                pipeline_rate=PIPELINE_PCM_RATE,
                format_type="audio/pcm",
            ),
            pcm,
        )

    async def test_format_transitions_require_directional_reset(self) -> None:
        resampler = AudioResampler()
        pcm = b"\x00\x00" * 160
        await resampler.to_pipeline(pcm, PIPELINE_PCM_RATE, format_type="audio/pcm")
        with self.assertRaisesRegex(ValueError, "reset_uplink"):
            await resampler.to_pipeline(
                b"\xff" * 80,
                G711_PCM_RATE,
                format_type="audio/pcmu",
            )
        resampler.reset_uplink()
        await resampler.to_pipeline(
            b"\xff" * 80,
            G711_PCM_RATE,
            format_type="audio/pcmu",
        )

        await resampler.from_pipeline(
            pcm,
            PIPELINE_PCM_RATE,
            pipeline_rate=PIPELINE_PCM_RATE,
            format_type="audio/pcm",
        )
        with self.assertRaisesRegex(ValueError, "reset_downlink"):
            await resampler.from_pipeline(
                pcm,
                G711_PCM_RATE,
                pipeline_rate=PIPELINE_PCM_RATE,
                format_type="audio/pcma",
            )
        resampler.reset_downlink()
        await resampler.from_pipeline(
            pcm,
            G711_PCM_RATE,
            pipeline_rate=PIPELINE_PCM_RATE,
            format_type="audio/pcma",
        )


class RealtimeSerializerCodecTests(unittest.IsolatedAsyncioTestCase):
    async def test_unknown_event_is_correlated_and_does_not_poison_next_event(self) -> None:
        serializer, emitted = _codec_serializer(_codec_controller())

        frame = await serializer.deserialize(
            json.dumps(
                {
                    "type": "future.realtime.event",
                    "event_id": "evt_unknown_1",
                }
            )
        )

        self.assertIsNone(frame)
        self.assertEqual(emitted[-1]["type"], "error")
        self.assertEqual(emitted[-1]["error"]["code"], "unsupported_event")
        self.assertEqual(emitted[-1]["error"]["param"], "type")
        self.assertEqual(emitted[-1]["error"]["event_id"], "evt_unknown_1")

        pipeline_pcm = b"\x00\x00" * 160
        serializer._resampler.to_pipeline = AsyncMock(return_value=pipeline_pcm)
        frame = await serializer.deserialize(
            json.dumps(
                {
                    "type": "input_audio_buffer.append",
                    "event_id": "evt_append_1",
                    "audio": base64.b64encode(b"\x00\x00" * 160).decode("ascii"),
                }
            )
        )

        self.assertIsInstance(frame, InputAudioRawFrame)
        self.assertEqual(len(emitted), 1)

    async def test_streaming_pcmu_input_is_decoded_at_the_pipeline_boundary(self) -> None:
        controller = _codec_controller()
        controller.apply_session_update({"type": "realtime", "audio": {"input": {"format": {"type": "audio/pcmu"}}}})
        serializer, emitted = _codec_serializer(controller)
        pipeline_pcm = b"\x00\x00" * 160
        serializer._resampler.to_pipeline = AsyncMock(return_value=pipeline_pcm)
        wire_audio = b"\xff" * 80

        frame = await serializer.deserialize(
            json.dumps(
                {
                    "type": "input_audio_buffer.append",
                    "audio": base64.b64encode(wire_audio).decode("ascii"),
                }
            )
        )

        self.assertIsInstance(frame, InputAudioRawFrame)
        assert isinstance(frame, InputAudioRawFrame)
        self.assertEqual(frame.audio, pipeline_pcm)
        self.assertEqual(frame.sample_rate, PIPELINE_PCM_RATE)
        serializer._resampler.to_pipeline.assert_awaited_once_with(
            wire_audio,
            G711_PCM_RATE,
            pipeline_rate=PIPELINE_PCM_RATE,
            format_type="audio/pcmu",
        )
        self.assertEqual(emitted, [])

    async def test_pcma_output_is_encoded_before_the_wire_delta(self) -> None:
        controller = _codec_controller()
        controller.apply_session_update({"type": "realtime", "audio": {"output": {"format": {"type": "audio/pcma"}}}})
        serializer, emitted = _codec_serializer(controller)
        controller.start_response()
        wire_audio = b"\xd5" * 80
        serializer._resampler.from_pipeline = AsyncMock(return_value=wire_audio)
        pipeline_pcm = b"\x00\x00" * 160

        await serializer.serialize(
            OutputAudioRawFrame(audio=pipeline_pcm, sample_rate=PIPELINE_PCM_RATE, num_channels=1)
        )

        serializer._resampler.from_pipeline.assert_awaited_once_with(
            pipeline_pcm,
            G711_PCM_RATE,
            pipeline_rate=serializer._pipeline_out_rate,
            format_type="audio/pcma",
        )
        delta = next(event for event in emitted if event["type"] == "response.output_audio.delta")
        self.assertEqual(base64.b64decode(delta["delta"]), wire_audio)

    async def test_live_format_change_resets_only_the_changed_directions_before_ack(self) -> None:
        controller = _codec_controller()
        serializer, emitted = _codec_serializer(controller)
        previous_uplink = serializer._resampler._uplink
        previous_downlink = serializer._resampler._downlink

        frame = await serializer.deserialize(
            json.dumps(
                {
                    "type": "session.update",
                    "session": {
                        "type": "realtime",
                        "audio": {
                            "input": {"format": {"type": "audio/pcmu"}},
                            "output": {"format": {"type": "audio/pcma"}},
                        },
                    },
                }
            )
        )

        self.assertIsNone(frame)
        self.assertIsNot(serializer._resampler._uplink, previous_uplink)
        self.assertIsNot(serializer._resampler._downlink, previous_downlink)
        self.assertEqual(emitted[-1]["type"], "session.updated")
        self.assertEqual(emitted[-1]["session"]["audio"]["input"]["format"], {"type": "audio/pcmu"})
        self.assertEqual(emitted[-1]["session"]["audio"]["output"]["format"], {"type": "audio/pcma"})

    async def test_live_input_format_change_flushes_old_pcm_before_accepting_new_pcm(self) -> None:
        pcm24_samples = (np.sin(np.arange(2400) * (2 * np.pi * 440 / 24000)) * 12000).astype(np.int16)
        cases = (
            ("audio/pcm", 24000, pcm24_samples.tobytes()),
            ("audio/pcmu", G711_PCM_RATE, b"\xff" * 800),
        )
        for format_type, wire_rate, wire_audio in cases:
            with self.subTest(format_type=format_type):
                controller = _codec_controller()
                if format_type != "audio/pcm":
                    controller.apply_session_update(
                        {"type": "realtime", "audio": {"input": {"format": {"type": format_type}}}}
                    )
                serializer, emitted = _codec_serializer(controller)

                first = await serializer.deserialize(
                    json.dumps(
                        {
                            "type": "input_audio_buffer.append",
                            "audio": base64.b64encode(wire_audio).decode("ascii"),
                        }
                    )
                )
                tail = await serializer.deserialize(
                    json.dumps(
                        {
                            "type": "session.update",
                            "session": {
                                "type": "realtime",
                                "audio": {"input": {"format": {"type": "audio/pcm", "rate": 16000}}},
                            },
                        }
                    )
                )

                self.assertIsInstance(tail, InputAudioRawFrame)
                streamed = first.audio if isinstance(first, InputAudioRawFrame) else b""
                assert isinstance(tail, InputAudioRawFrame)
                streamed += tail.audio
                expected = await AudioResampler.complete_input_to_pipeline(
                    wire_audio,
                    wire_rate,
                    pipeline_rate=PIPELINE_PCM_RATE,
                    format_type=format_type,
                )
                actual_samples = np.frombuffer(streamed, dtype=np.int16)
                expected_samples = np.frombuffer(expected, dtype=np.int16)
                self.assertEqual(actual_samples.size, expected_samples.size)
                np.testing.assert_allclose(actual_samples, expected_samples, rtol=0, atol=2)
                self.assertEqual(emitted[-1]["type"], "session.updated")

                new_pcm = b"\x01\x00" * 320
                following = await serializer.deserialize(
                    json.dumps(
                        {
                            "type": "input_audio_buffer.append",
                            "audio": base64.b64encode(new_pcm).decode("ascii"),
                        }
                    )
                )
                self.assertIsInstance(following, InputAudioRawFrame)
                assert isinstance(following, InputAudioRawFrame)
                self.assertEqual(following.audio, new_pcm)

    async def test_rejected_combined_update_preserves_the_old_uplink_stream(self) -> None:
        serializer, emitted = _codec_serializer(_codec_controller())
        wire_audio = (np.arange(960, dtype=np.int16) - 480).tobytes()
        accepted_frames: list[InputAudioRawFrame] = []

        first = await serializer.deserialize(
            json.dumps(
                {
                    "type": "input_audio_buffer.append",
                    "audio": base64.b64encode(wire_audio).decode("ascii"),
                }
            )
        )
        if isinstance(first, InputAudioRawFrame):
            accepted_frames.append(first)
        rejected = await serializer.deserialize(
            json.dumps(
                {
                    "type": "session.update",
                    "session": {
                        "type": "realtime",
                        "instructions": "This update cannot be prepared without a response gate.",
                        "audio": {"input": {"format": {"type": "audio/pcm", "rate": 16000}}},
                    },
                }
            )
        )
        self.assertIsNone(rejected)
        self.assertEqual(emitted[-1]["type"], "error")
        self.assertEqual(emitted[-1]["error"]["code"], "session_runtime_missing")

        second = await serializer.deserialize(
            json.dumps(
                {
                    "type": "input_audio_buffer.append",
                    "audio": base64.b64encode(wire_audio).decode("ascii"),
                }
            )
        )
        if isinstance(second, InputAudioRawFrame):
            accepted_frames.append(second)
        streamed = b"".join(frame.audio for frame in accepted_frames)
        streamed += await serializer._resampler.flush_to_pipeline()
        expected = await AudioResampler.complete_input_to_pipeline(
            wire_audio * 2,
            DEFAULT_CLIENT_PCM_RATE,
            pipeline_rate=PIPELINE_PCM_RATE,
        )
        actual_samples = np.frombuffer(streamed, dtype=np.int16)
        expected_samples = np.frombuffer(expected, dtype=np.int16)
        self.assertEqual(actual_samples.size, expected_samples.size)
        np.testing.assert_allclose(actual_samples, expected_samples, rtol=0, atol=2)

    async def test_partial_live_format_update_preserves_the_other_direction(self) -> None:
        cases = (
            (
                {"input": {"format": {"type": "audio/pcmu"}}},
                {"type": "audio/pcmu"},
                {"type": "audio/pcm", "rate": DEFAULT_CLIENT_PCM_RATE},
                True,
                False,
            ),
            (
                {"output": {"format": {"type": "audio/pcma"}}},
                {"type": "audio/pcm", "rate": DEFAULT_CLIENT_PCM_RATE},
                {"type": "audio/pcma"},
                False,
                True,
            ),
        )
        for audio_patch, expected_input, expected_output, reset_uplink, reset_downlink in cases:
            with self.subTest(audio_patch=audio_patch):
                serializer, emitted = _codec_serializer(_codec_controller())
                previous_uplink = serializer._resampler._uplink
                previous_downlink = serializer._resampler._downlink

                frame = await serializer.deserialize(
                    json.dumps(
                        {
                            "type": "session.update",
                            "session": {"type": "realtime", "audio": audio_patch},
                        }
                    )
                )

                self.assertIsNone(frame)
                self.assertEqual(serializer._resampler._uplink is not previous_uplink, reset_uplink)
                self.assertEqual(serializer._resampler._downlink is not previous_downlink, reset_downlink)
                self.assertEqual(emitted[-1]["type"], "session.updated")
                self.assertEqual(emitted[-1]["session"]["audio"]["input"]["format"], expected_input)
                self.assertEqual(emitted[-1]["session"]["audio"]["output"]["format"], expected_output)
