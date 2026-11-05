# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D103

"""The live protocol on its own: the startup schema, the error events and the wire audio codecs."""

import unittest

import numpy as np
from pydantic import ValidationError

from gateways.live.events import error_event, event
from gateways.live.media.audio import ALaw, MuLaw, WireCodec
from gateways.live.protocol import AudioFormat, ProtocolError, SessionConfig, validation_error

TEXT = {"type": "input_text", "text": "hi"}
USER = {"type": "message", "role": "user"}


def responses(**settings):
    """A Responses delegation whose backend block sets ``settings``."""
    return {"type": "responses", "responses": {"model": "org/model", **settings}}


def start_error(**session) -> ProtocolError:
    """The protocol error a ``session.start`` with these settings gets."""
    try:
        SessionConfig.model_validate({"model": "live-1", **session})
    except ValidationError as refused:
        return validation_error(refused, "session")
    raise AssertionError(f"the session was accepted: {session}")


class SchemaTests(unittest.TestCase):
    def test_an_omitted_null_or_empty_delegation_is_client_mode(self):
        for delegation in (None, {}, {"type": "client"}):
            with self.subTest(delegation=delegation):
                self.assertEqual(SessionConfig(model="live-1", delegation=delegation).mode, "client")
        self.assertEqual(SessionConfig(model="live-1").mode, "client")
        self.assertEqual(SessionConfig(model="live-1", delegation=responses()).mode, "responses")

    def test_every_documented_setting_is_accepted(self):
        tiers = ("auto", "default", "fast_tier_temp_pilot", "flex", "priority", "ultrafast", None)
        selector = {"type": "response.event", "response_event": "a.b"}
        for session in (
            {"delegation": {"type": "client", "acknowledgement_delay_ms": 800}},
            {"delegation": {**responses(), "acknowledgement_delay_ms": 800}},
            # Every official tool type: whether the backend supports one is for the background check to say.
            {"delegation": responses(tools=[{"type": "mcp"}, {"type": "function", "name": "get_menu"}])},
            {"delegation": responses(text={"verbosity": "low"}, parallel_tool_calls=None, reasoning=None)},
            *({"delegation": responses(service_tier=tier, instructions=None)} for tier in tiers),
            *({"audio": {"output": {"voice": voice}}} for voice in ("marin", "brise", "sillage", {"id": "voice_x"})),
            {"input": [{**USER, "role": "developer", "content": [TEXT]}]},
            {"client": {"data_channel": {"allowed_server_events": [selector]}}},
            {"triggers": []},
        ):
            with self.subTest(session=session):
                SessionConfig(model="live-1", **session)

    def test_a_refused_setting_names_its_param(self):
        server_events = "session.client.data_channel.allowed_server_events"
        for session, code, param in (
            ({"model": ""}, "invalid_value", "session.model"),
            ({"voice": "marin"}, "unknown_parameter", "session.voice"),
            ({"audio": {"output": {"speed": 2}}}, "unknown_parameter", "session.audio.output.speed"),
            ({"audio": {"output": {"voice": "no-such-voice"}}}, "invalid_value", "session.audio.output"),
            ({"audio": {"format": {"type": "audio/pcm", "rate": 8000}}}, "invalid_value", "session.audio.format"),
            ({"audio": {"format": {"type": "audio/pcmu", "rate": 24000}}}, "invalid_value", "session.audio.format"),
            # Responses settings belong to responses mode only, also when the type is left to its client default.
            ({"delegation": {"type": "client", "responses": {}}}, "unknown_parameter", "session.delegation.responses"),
            ({"delegation": {"responses": {"model": "x"}}}, "unknown_parameter", "session.delegation.responses"),
            ({"delegation": {"type": "responses"}}, "invalid_value", "session.delegation.responses"),
            ({"delegation": responses(text={"format": {}})}, "unknown_parameter", "session.delegation.responses.text"),
            ({"delegation": responses(tools=[{"type": "teleport"}])}, "invalid_value", "session.delegation.responses"),
            ({"delegation": responses(tools=[{"type": "function"}])}, "invalid_value", "session.delegation.responses"),
            ({"delegation": responses(tools=["get_menu"])}, "invalid_value", "session.delegation.responses.tools"),
            ({"delegation": {"acknowledgement_delay_ms": -1}}, "invalid_value", "session.delegation.acknowledgement"),
            ({"delegation": {"type": "decisions", "decisions": {}}}, "invalid_value", "session.delegation"),
            ({"triggers": [{"type": "decisions"}]}, "invalid_value", "session.triggers"),
            (
                {"client": {"data_channel": {"allowed_server_events": [{"type": "response.event"}]}}},
                "invalid_value",
                server_events,
            ),
            (
                {
                    "client": {
                        "data_channel": {"allowed_server_events": [{"type": "session.started", "response_event": "x"}]}
                    }
                },
                "invalid_value",
                server_events,
            ),
            *(
                ({"input": [item]}, "invalid_value", "session.input")
                for item in (
                    {**USER, "role": "system", "content": [TEXT]},
                    {**USER, "content": [TEXT, TEXT]},
                    {**USER, "content": [{"type": "input_image"}]},
                    {**USER, "content": [{"type": "input_text", "text": 5}]},
                    {"role": [], "content": []},
                    {"role": "user", "content": "x"},
                    "text",
                )
            ),
        ):
            with self.subTest(session=session):
                error = start_error(**session)
                self.assertEqual(error.code, code)
                self.assertTrue(error.param.startswith(param), error.param)
                if code == "unknown_parameter":
                    self.assertEqual(str(error), f"Unknown parameter: '{error.param}'.")
        self.assertIn("'decisions'", str(start_error(delegation={"type": "decisions"})))
        self.assertIn("Decisions triggers are not supported", str(start_error(triggers=[{"type": "decisions"}])))


class ErrorEventTests(unittest.TestCase):
    def test_every_event_has_a_fresh_id_and_an_error_takes_its_type_from_its_code(self):
        self.assertNotEqual(event("a")["event_id"], event("a")["event_id"])
        self.assertTrue(event("a")["event_id"].startswith("event_"))
        bare = error_event(ProtocolError("nope"))
        self.assertEqual(set(bare), {"type", "event_id", "error"})  # an optional field the server lacks is left out
        self.assertEqual(bare["error"], {"type": "invalid_request_error", "code": "invalid_value", "message": "nope"})
        full = error_event(ProtocolError("nope", "unknown_parameter", "session.voice"), "c1")
        self.assertEqual(
            (full["client_event_id"], full["error"]["param"], full["error"]["client_event_id"]),
            ("c1", "session.voice", "c1"),
        )
        for code in ("server_error", "pipeline_error"):
            self.assertEqual(error_event(ProtocolError("down", code))["error"]["type"], "server_error")


def reference(sample: int, alaw: bool) -> int:
    """ITU-T G.711 of one 16-bit sample, as the CCITT reference (Sun g711.c) encodes it: A-law or mu-law."""
    if alaw:
        pcm, mask = (sample >> 3, 0xD5) if sample >= 0 else (-(sample >> 3) - 1, 0x55)
        segment = sum(pcm > end for end in (0x1F, 0x3F, 0x7F, 0xFF, 0x1FF, 0x3FF, 0x7FF, 0xFFF))
        code = 0x7F if segment >= 8 else (segment << 4) | ((pcm >> (1 if segment < 2 else segment)) & 0xF)
        return code ^ mask
    pcm, mask = (-(sample >> 2), 0x7F) if sample < 0 else (sample >> 2, 0xFF)
    pcm = min(pcm, 8159) + 33
    segment = sum(pcm > end for end in (0x3F, 0x7F, 0xFF, 0x1FF, 0x3FF, 0x7FF, 0xFFF, 0x1FFF))
    return (0x7F if segment >= 8 else (segment << 4) | ((pcm >> (segment + 1)) & 0xF)) ^ mask


class CodecTests(unittest.TestCase):
    def test_wire_audio_converts_to_24_khz_pcm_and_back(self):
        self.assertEqual(
            [AudioFormat(type=kind).rate for kind in ("audio/pcm", "audio/pcmu", "audio/pcma")], [24000, 8000, 8000]
        )
        self.assertEqual(SessionConfig(model="m", audio={"format": {"type": "audio/pcma"}}).audio.format.rate, 8000)
        for audio_format, wire in (
            (AudioFormat(type="audio/pcm", rate=16000), b"\x10\x20" * 1600),
            (AudioFormat(type="audio/pcmu"), bytes(range(160)) * 2),
            (AudioFormat(type="audio/pcma"), bytes(range(160)) * 2),
        ):
            with self.subTest(audio_format=audio_format.type):
                codec = WireCodec(audio_format)
                pcm = codec.decode(wire)
                self.assertEqual(len(pcm) % 2, 0)
                self.assertGreater(len(pcm), len(wire))  # resampled up to 24 kHz
                self.assertTrue(0 < len(codec.encode(pcm)) <= len(wire))
        with self.assertRaises(ValueError):
            WireCodec().decode(b"\x01")  # an incomplete PCM sample

    def test_the_g711_encoders_match_the_reference_on_every_16_bit_sample(self):
        samples = np.arange(-32768, 32768, dtype="<i2")
        for codec, alaw in ((MuLaw(), False), (ALaw(), True)):
            with self.subTest(alaw=alaw):
                encoded = np.frombuffer(codec.encode(samples.tobytes()), dtype=np.uint8)
                expected = np.array([reference(int(sample), alaw) for sample in samples], dtype=np.uint8)
                self.assertEqual(int(np.count_nonzero(encoded != expected)), 0)

    def test_known_g711_codes_and_a_lossless_round_trip_of_every_code(self):
        known = {MuLaw(): {0: 0xFF, 32767: 0x80, -32768: 0x00}, ALaw(): {0: 0xD5, -1: 0x55, 32767: 0xAA}}
        for codec, codes in known.items():
            for sample, code in codes.items():
                self.assertEqual(codec.encode(np.array([sample], dtype="<i2").tobytes())[0], code, (codec, sample))
            decoded = codec.decode(bytes(range(256)))
            self.assertEqual(codec.decode(codec.encode(decoded)), decoded)


if __name__ == "__main__":
    unittest.main()
