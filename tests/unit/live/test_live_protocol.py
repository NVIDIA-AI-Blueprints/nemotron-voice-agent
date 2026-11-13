# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D103

"""The live protocol on its own: the startup schema and the error events."""

import json
import unittest

from pydantic import ValidationError

from gateways.live.events import error_event
from gateways.live.protocol import AudioFormat, ProtocolError, SessionConfig, validation_error

TEXT = {"type": "input_text", "text": "hi"}
USER = {"type": "message", "role": "user"}


def responses(**settings):
    return {"type": "responses", "responses": {"model": "org/model", **settings}}


def start_error(**session) -> ProtocolError:
    try:
        SessionConfig.model_validate({"model": "live-1", **session})
    except ValidationError as refused:
        return validation_error(refused, "session")
    raise AssertionError(f"the session was accepted: {session}")


class ProtocolTests(unittest.TestCase):
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
        modes = [SessionConfig(model="live-1", delegation=d).mode for d in (None, {}, {"type": "client"}, responses())]
        self.assertEqual(modes, ["client", "client", "client", "responses"])
        rates = [AudioFormat(type=kind).rate for kind in ("audio/pcm", "audio/pcmu", "audio/pcma")]
        self.assertEqual(rates, [24000, 8000, 8000])

    def test_a_refused_setting_names_its_param(self):
        server_events = "session.client.data_channel.allowed_server_events"
        unnested = {"data_channel": {"allowed_server_events": [{"type": "response.event"}]}}
        misnested = {"data_channel": {"allowed_server_events": [{"type": "session.started", "response_event": "x"}]}}
        bad_items = (
            {**USER, "role": "system", "content": [TEXT]},
            {"role": [], "content": []},
            {**USER, "content": [TEXT, TEXT]},
            {**USER, "content": [{"type": "input_text", "text": 5}]},
            "text",
        )
        for session, code, param in (
            ({"model": ""}, "invalid_value", "session.model"),
            ({"audio": {"output": {"speed": 2}}}, "unknown_parameter", "session.audio.output.speed"),
            ({"audio": {"output": {"voice": "no-such-voice"}}}, "invalid_value", "session.audio.output"),
            ({"audio": {"output": {"voice": {"id": "   "}}}}, "invalid_value", "session.audio.output"),
            ({"audio": {"format": {"type": "audio/pcm", "rate": 8000}}}, "invalid_value", "session.audio.format"),
            ({"audio": {"format": {"type": "audio/pcmu", "rate": 24000}}}, "invalid_value", "session.audio.format"),
            ({"delegation": {"responses": {"model": "x"}}}, "unknown_parameter", "session.delegation.responses"),
            ({"delegation": {"type": "responses"}}, "invalid_value", "session.delegation.responses"),
            ({"delegation": responses(tools=[{"type": "teleport"}])}, "invalid_value", "session.delegation.responses"),
            ({"delegation": {"acknowledgement_delay_ms": -1}}, "invalid_value", "session.delegation.acknowledgement"),
            ({"delegation": {"type": "decisions", "decisions": {}}}, "invalid_value", "session.delegation"),
            ({"triggers": [{"type": "decisions"}]}, "invalid_value", "session.triggers"),
            ({"client": unnested}, "invalid_value", server_events),
            ({"client": misnested}, "invalid_value", server_events),
            *(({"input": [item]}, "invalid_value", "session.input") for item in bad_items),
        ):
            with self.subTest(session=session):
                error = start_error(**session)
                self.assertEqual(error.code, code)
                self.assertTrue(error.param.startswith(param), error.param)
                if code == "unknown_parameter":
                    self.assertEqual(str(error), f"Unknown parameter: '{error.param}'.")
        self.assertIn("'decisions'", str(start_error(delegation={"type": "decisions"})))
        self.assertIn("Decisions triggers are not supported", str(start_error(triggers=[{"type": "decisions"}])))

    def test_the_acknowledgement_delay_is_at_most_800_ms(self):
        param = "session.delegation.acknowledgement_delay_ms"
        message = f"Invalid value for '{param}': Input should be less than or equal to 800"
        refused = {"type": "invalid_request_error", "code": "invalid_value", "message": message, "param": param}
        for delegation in ({"type": "client"}, responses()):
            with self.subTest(type=delegation["type"]):
                longest = SessionConfig(model="live-1", delegation={**delegation, "acknowledgement_delay_ms": 800})
                self.assertEqual(longest.delegation.acknowledgement_delay_ms, 800)
                started = start_error(delegation={**delegation, "acknowledgement_delay_ms": 801})
                self.assertEqual(error_event(started)["error"], refused)
                with self.assertRaises(ProtocolError) as updated:
                    longest.updated({"delegation": {"acknowledgement_delay_ms": 801}})
                self.assertEqual(error_event(updated.exception)["error"], refused)

    def test_a_conversation_to_continue_needs_store_and_is_set_only_at_the_start(self):
        param = "session.delegation.responses.conversation"
        with self.assertRaises(ProtocolError) as unstored:
            SessionConfig(model="live-1", delegation=responses(conversation="conv_1"))
        self.assertEqual((unstored.exception.code, unstored.exception.param), ("invalid_value", param))
        blank = start_error(store=True, delegation=responses(conversation=" "))
        self.assertEqual((blank.code, blank.param), ("invalid_value", param))
        stored = SessionConfig(model="live-1", store=True, delegation=responses(conversation="conv_1"))
        same = stored.updated({"delegation": {"responses": {"conversation": "conv_1"}}})
        self.assertEqual(same.delegation.responses.conversation, "conv_1")
        fresh = SessionConfig(model="live-1", store=True, delegation=responses())
        for config, conversation in ((stored, "conv_2"), (stored, None), (fresh, "conv_1")):
            with self.subTest(conversation=conversation, named_at_start=config is stored):
                with self.assertRaises(ProtocolError) as changed:
                    config.updated({"delegation": {"responses": {"conversation": conversation}}})
                self.assertEqual((changed.exception.code, changed.exception.param), ("immutable_field_update", param))

    def test_an_update_merges_reasoning_and_replaces_tool_choice(self):
        function = {"type": "function", "name": "get_menu"}
        mcp = {"type": "mcp", "server_label": "crm"}
        reasoning = {"effort": "low", "summary": "auto"}
        config = SessionConfig(model="live-1", delegation=responses(tool_choice=function, reasoning=reasoning))
        updated = config.updated({"delegation": {"responses": {"tool_choice": mcp, "reasoning": {"effort": "high"}}}})
        backend = updated.delegation.responses
        self.assertEqual((backend.tool_choice, backend.reasoning), (mcp, {"effort": "high", "summary": "auto"}))

    def test_the_responses_settings_cannot_grow_past_one_event(self):
        config, piece, applied = SessionConfig(model="live-1", delegation=responses()), "y" * 300 * 1024, 0
        with self.assertRaises(ProtocolError) as refused:
            for n in range(8):
                config = config.updated({"delegation": {"responses": {"reasoning": {f"k{n}": piece}}}})
                applied += 1
        self.assertEqual((applied, refused.exception.code), (3, "invalid_value"))
        self.assertLess(len(json.dumps(config.delegation.responses.model_dump())), 2**20)

    def test_an_error_event_has_only_its_fields_and_its_codes_type(self):
        bare = error_event(ProtocolError("nope"))
        self.assertEqual(set(bare), {"type", "event_id", "error"})
        self.assertEqual(bare["error"], {"type": "invalid_request_error", "code": "invalid_value", "message": "nope"})
        self.assertEqual(error_event(ProtocolError("down", "pipeline_error"))["error"]["type"], "server_error")


if __name__ == "__main__":
    unittest.main()
