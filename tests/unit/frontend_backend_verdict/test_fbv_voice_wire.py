# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D100, D101, D102, D103, D107

"""Rules 2, 5, 22: G.711, client events, the GA session view, server events and response ordering."""

from __future__ import annotations

import base64
import copy
import random
import unittest

import numpy as np
from _fbv_voice_fakes import tau2_session_update

from examples.frontend_backend_verdict.voice.audio import g711
from examples.frontend_backend_verdict.voice.audio.formats import AudioFormat, parse_format
from examples.frontend_backend_verdict.voice.errors import WireProtocolError
from examples.frontend_backend_verdict.voice.wire import client_events as ce
from examples.frontend_backend_verdict.voice.wire import server_events as ev
from examples.frontend_backend_verdict.voice.wire.response_writer import ConversationOrder, ResponseWriter
from examples.frontend_backend_verdict.voice.wire.session_view import RealtimeSessionView, SessionDefaults


class G711Tests(unittest.TestCase):
    def test_itu_reference_points(self) -> None:
        # mu-law: 0xFF and 0x7F are +/-0, 0x80/0x00 are the extremes (+/-32124).
        decoded = np.frombuffer(g711.ulaw_decode(bytes([0xFF, 0x7F, 0x80, 0x00])), dtype="<i2").tolist()
        self.assertEqual(decoded, [0, 0, 32124, -32124])
        # A-law: 0xD5 -> 8, 0x55 -> -8, 0xAA -> 32256, 0x2A -> -32256.
        decoded = np.frombuffer(g711.alaw_decode(bytes([0xD5, 0x55, 0xAA, 0x2A])), dtype="<i2").tolist()
        self.assertEqual(decoded, [8, -8, 32256, -32256])
        self.assertEqual(g711.ulaw_encode(np.zeros(1, dtype="<i2").tobytes()), bytes([g711.ULAW_SILENCE]))
        self.assertEqual(g711.alaw_encode(np.zeros(1, dtype="<i2").tobytes()), bytes([g711.ALAW_SILENCE]))

    def test_decode_encode_is_identity_on_codes(self) -> None:
        codes = bytes(range(256))
        self.assertEqual(
            g711.ulaw_encode(g711.ulaw_decode(codes)).replace(b"\x7f", b"\xff"), codes.replace(b"\x7f", b"\xff")
        )
        self.assertEqual(g711.alaw_encode(g711.alaw_decode(codes)), codes)

    def test_pcmu_round_trip_error_is_small(self) -> None:
        samples = (np.sin(np.linspace(0, 40, 4000)) * 12000).astype("<i2")
        restored = np.frombuffer(g711.ulaw_decode(g711.ulaw_encode(samples.tobytes())), dtype="<i2")
        self.assertLess(np.max(np.abs(restored.astype(int) - samples.astype(int))), 12000 * 0.04)

    def test_formats(self) -> None:
        self.assertEqual(parse_format({"type": "audio/pcmu"}, param="p"), AudioFormat("audio/pcmu", 8000))
        self.assertEqual(parse_format({"type": "audio/pcm", "rate": 16000}, param="p").rate, 16000)
        for bad in ({"type": "audio/pcm", "rate": 11025}, {"type": "audio/pcmu", "rate": 16000}, "pcm16"):
            with self.assertRaises(ValueError):
                parse_format(bad, param="p")


class ClientEventTests(unittest.TestCase):
    def test_parses_every_tau2_event(self) -> None:
        self.assertIsInstance(ce.parse_client_event(tau2_session_update()), ce.SessionUpdate)
        append = ce.parse_client_event(
            {"type": "input_audio_buffer.append", "audio": base64.b64encode(b"\xff" * 160).decode()}
        )
        self.assertEqual(append.audio, b"\xff" * 160)
        output = ce.parse_client_event(
            {
                "type": "conversation.item.create",
                "item": {"type": "function_call_output", "call_id": "c1", "output": "Error: x"},
            }
        )
        self.assertEqual(output.item, ce.FunctionCallOutput(call_id="c1", output="Error: x"))
        self.assertIsInstance(ce.parse_client_event({"type": "response.create"}), ce.ResponseCreate)
        truncate = ce.parse_client_event(
            {"type": "conversation.item.truncate", "item_id": "i", "content_index": 0, "audio_end_ms": 1234}
        )
        self.assertEqual(truncate.audio_end_ms, 1234)

    def test_errors_are_typed(self) -> None:
        cases = [
            ({"type": "nope"}, "unknown_event"),
            ({"type": "input_audio_buffer.append", "audio": "@@@"}, "invalid_value"),
            (
                {"type": "conversation.item.create", "item": {"type": "message", "role": "assistant"}},
                "unsupported_item",
            ),
            ({"type": "conversation.item.create", "item": {"type": "function_call_output"}}, "invalid_value"),
        ]
        for data, code in cases:
            with self.subTest(data=data), self.assertRaises(WireProtocolError) as ctx:
                ce.parse_client_event(data)
            self.assertEqual(ctx.exception.code, code)
        with self.assertRaises(WireProtocolError):
            ce.decode_message("not json")


def _view() -> RealtimeSessionView:
    defaults = SessionDefaults(
        input_format=AudioFormat("audio/pcm", 24000),
        output_format=AudioFormat("audio/pcm", 24000),
        threshold=0.5,
        prefix_padding_ms=300,
        silence_duration_ms=500,
    )
    return RealtimeSessionView(session_id="sess_x", model="pine-x", defaults=defaults)


class SessionViewTests(unittest.TestCase):
    def test_tau2_payload_is_accepted_verbatim(self) -> None:
        view = _view()
        update = tau2_session_update("airline")["session"]
        result = view.apply(update)
        settings = result.settings
        self.assertEqual(settings.input_format, AudioFormat("audio/pcmu", 8000))
        self.assertEqual(settings.output_format, AudioFormat("audio/pcmu", 8000))
        self.assertEqual(settings.turn_detection.silence_duration_ms, 500)
        self.assertEqual(len(settings.tools), len(update["tools"]))
        self.assertTrue(result.tools_changed and result.instructions_changed)
        echo = view.public()
        self.assertEqual(echo["tools"], update["tools"])  # echoed as received
        self.assertEqual(echo["audio"]["input"]["format"], {"type": "audio/pcmu"})

    def test_unknown_keys_are_tolerated(self) -> None:
        view = _view()
        result = view.apply({"type": "realtime", "reasoning": {"effort": "low"}, "future_field": 1})
        self.assertEqual(view.public()["reasoning"], {"effort": "low"})
        self.assertTrue(any("future_field" in warning for warning in result.warnings))

    def test_beta_fields_and_formats_are_rejected_and_not_applied(self) -> None:
        view = _view()
        before = view.public()
        for patch, needle in (
            ({"modalities": ["audio"], "instructions": "x"}, "output_modalities"),
            ({"input_audio_format": "g711_ulaw"}, "audio.input.format"),
            ({"turn_detection": None}, "audio.input.turn_detection"),
            ({"temperature": 0.7}, "removed in GA"),
            ({"audio": {"input": {"format": "pcm16"}}}, "audio/pcm"),
        ):
            with self.subTest(patch=patch), self.assertRaises(WireProtocolError) as ctx:
                view.apply(patch)
            self.assertIn(needle, str(ctx.exception))
        self.assertEqual(view.public(), before)

    def test_semantic_vad_is_an_echoed_approximation(self) -> None:
        view = _view()
        result = view.apply({"audio": {"input": {"turn_detection": {"type": "semantic_vad", "eagerness": "low"}}}})
        self.assertTrue(result.semantic_vad)
        self.assertEqual(result.settings.turn_detection.silence_duration_ms, 800)
        echo = view.public()["audio"]["input"]["turn_detection"]
        self.assertEqual(echo["type"], "semantic_vad")
        self.assertEqual(echo["x_nvidia_effective"]["silence_duration_ms"], 800)
        self.assertTrue(any("approximated" in warning for warning in result.warnings))

    def test_manual_mode_and_invalid_values(self) -> None:
        view = _view()
        self.assertFalse(view.apply({"audio": {"input": {"turn_detection": None}}}).settings.turn_detection.vad_enabled)
        for patch in (
            {"tools": [{"type": "function", "name": "a"}, {"type": "function", "name": "a"}]},
            {"output_modalities": ["audio", "text"]},
            {"audio": {"input": {"turn_detection": {"type": "server_vad", "threshold": 2}}}},
        ):
            with self.subTest(patch=patch), self.assertRaises(WireProtocolError):
                view.apply(patch)


def _speak(writer: ResponseWriter, sentences: list[str]) -> None:
    item = writer.begin_message()
    for sentence in sentences:
        item.transcript_delta(sentence)
        item.audio_delta(base64.b64encode(b"\xff" * 80).decode())
    item.close()


class ServerEventTests(unittest.TestCase):
    def test_ordering_properties_over_generated_sequences(self) -> None:
        rng = random.Random(7)
        for _ in range(200):
            emitted: list[dict] = []
            writer = ResponseWriter(emitted.append, ConversationOrder())
            writer.open()
            for _ in range(rng.randint(0, 3)):
                if rng.random() < 0.5:
                    _speak(writer, [f"s{n}." for n in range(rng.randint(1, 3))])
                else:
                    writer.function_call(call_id=f"call_{rng.random()}", name="t", arguments="{}")
            if rng.random() < 0.3:
                item = writer.begin_message()
                item.transcript_delta("cut")
            writer.finish(rng.choice(["completed", "cancelled"]), reason="turn_detected")
            writer.finish("completed")  # idempotent
            types = [event["type"] for event in emitted]
            self.assertEqual(types.count("response.created"), 1)
            self.assertEqual(types.count("response.done"), 1)
            self.assertEqual(types[-1], "response.done")
            self.assertFalse(set(types) & ev.BETA_EVENT_TYPES)
            self.assertTrue(set(types) <= ev.GA_EVENT_TYPES)
            added = [e["item"]["id"] for e in emitted if e["type"] == "response.output_item.added"]
            self.assertEqual(len(added), len(set(added)))
            seen_transcript: set[str] = set()
            for event in emitted:
                if event["type"] == "response.output_audio_transcript.delta":
                    seen_transcript.add(event["item_id"])
                if event["type"] == "response.output_audio.delta":
                    self.assertIn(event["item_id"], seen_transcript)
            open_items = 0
            for event in emitted:
                if event["type"] == "response.output_item.added":
                    open_items += 1
                    self.assertEqual(open_items, 1, "items never interleave")
                if event["type"] == "response.output_item.done":
                    open_items -= 1
            done = emitted[-1]["response"]
            self.assertIsInstance(done["usage"], dict)
            self.assertIn("output_token_details", done["usage"])

    def test_function_call_done_has_top_level_fields(self) -> None:
        emitted: list[dict] = []
        writer = ResponseWriter(emitted.append, ConversationOrder())
        writer.function_call(call_id="call_abc", name="get_users", arguments='{"a": 1}')
        done = next(e for e in emitted if e["type"] == "response.function_call_arguments.done")
        self.assertEqual((done["call_id"], done["name"], done["arguments"]), ("call_abc", "get_users", '{"a": 1}'))

    def test_cancelled_item_keeps_only_sent_transcript(self) -> None:
        emitted: list[dict] = []
        writer = ResponseWriter(emitted.append, ConversationOrder())
        item = writer.begin_message()
        item.transcript_delta("First.")
        writer.finish("cancelled", reason="turn_detected")
        final = next(e for e in emitted if e["type"] == "response.output_item.done")["item"]
        self.assertEqual(final["status"], "incomplete")
        self.assertEqual(final["content"][0]["transcript"], "First.")
        done = emitted[-1]["response"]
        self.assertEqual(done["status_details"], {"type": "cancelled", "reason": "turn_detected"})

    def test_builders_copy_nothing_mutable(self) -> None:
        item = ev.user_audio_item("item_1")
        event = ev.item_added(copy.deepcopy(item), None)
        event["item"]["content"].append("x")
        self.assertEqual(len(item["content"]), 1)
