# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D103

"""The live protocol on its own: the startup schema and the error events."""

import unittest

from pydantic import ValidationError

from gateways.live.events import error_event, event
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

    def test_a_wire_format_without_a_rate_takes_its_own(self):
        self.assertEqual(
            [AudioFormat(type=kind).rate for kind in ("audio/pcm", "audio/pcmu", "audio/pcma")], [24000, 8000, 8000]
        )
        self.assertEqual(SessionConfig(model="m", audio={"format": {"type": "audio/pcma"}}).audio.format.rate, 8000)

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

    def test_the_acknowledgement_delay_is_at_most_800_ms_in_either_delegation_type(self):
        param = "session.delegation.acknowledgement_delay_ms"
        refused = {
            "type": "invalid_request_error",
            "code": "invalid_value",
            "message": f"Invalid value for '{param}': Input should be less than or equal to 800",
            "param": param,
        }
        for delegation in ({"type": "client"}, responses()):
            with self.subTest(type=delegation["type"]):
                longest = SessionConfig(model="live-1", delegation={**delegation, "acknowledgement_delay_ms": 800})
                self.assertEqual(longest.delegation.acknowledgement_delay_ms, 800)
                started = start_error(delegation={**delegation, "acknowledgement_delay_ms": 801})
                self.assertEqual(error_event(started)["error"], refused)
                # A session.update is held to the same range.
                with self.assertRaises(ProtocolError) as updated:
                    longest.updated({"delegation": {"acknowledgement_delay_ms": 801}})
                self.assertEqual(error_event(updated.exception)["error"], refused)


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


if __name__ == "__main__":
    unittest.main()
