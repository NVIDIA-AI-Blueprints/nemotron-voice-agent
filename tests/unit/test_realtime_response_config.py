# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""Tests for native ``response.create`` validation and context projection."""

from __future__ import annotations

import unittest

from realtime.protocol import RealtimeProtocolError
from realtime.response_config import (
    project_response_input_messages,
    validate_response_conversation,
    validate_response_input,
    validate_response_output_modalities,
)


class ResponseCreateValidationTests(unittest.TestCase):
    """Verify response-local fields are strict and context remains isolated."""

    def test_default_conversation_and_each_native_modality_are_valid(self) -> None:
        """The default conversation and each Realtime output modality are accepted."""
        self.assertEqual(validate_response_conversation("auto"), "auto")
        self.assertEqual(validate_response_output_modalities(["text"]), ["text"])
        self.assertEqual(validate_response_output_modalities(["audio"]), ["audio"])

    def test_out_of_band_conversation_is_explicitly_unsupported(self) -> None:
        """OOB is rejected rather than leaking its output into default state."""
        with self.assertRaises(RealtimeProtocolError) as raised:
            validate_response_conversation("none")
        self.assertEqual(raised.exception.code, "unsupported_capability")
        self.assertEqual(raised.exception.param, "response.conversation")

    def test_inline_text_and_function_items_project_without_journal_mutation(self) -> None:
        """Inline text and tool history becomes standard provider messages."""
        response_input = validate_response_input(
            [
                {
                    "type": "message",
                    "role": "system",
                    "content": [{"type": "input_text", "text": "Be concise."}],
                },
                {
                    "type": "function_call",
                    "call_id": "call_1",
                    "name": "lookup",
                    "arguments": '{"id":7}',
                },
                {"type": "function_call_output", "call_id": "call_1", "output": '{"name":"Ada"}'},
            ]
        )

        messages = project_response_input_messages(
            response_input,
            resolve_item=lambda _item_id: self.fail("inline input must not read the journal"),
        )

        self.assertEqual(
            messages,
            [
                {"role": "system", "content": "Be concise."},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "lookup", "arguments": '{"id":7}'},
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "call_1", "content": '{"name":"Ada"}'},
            ],
        )

    def test_item_reference_prefers_exact_pipecat_context_message(self) -> None:
        """References reuse the exact canonical model message when available."""
        exact = {"role": "assistant", "content": "canonical context text"}
        response_input = validate_response_input([{"type": "item_reference", "id": "item_1"}])

        messages = project_response_input_messages(
            response_input,
            resolve_item=lambda _item_id: self.fail("exact message should win"),
            resolve_context_message=lambda item_id: exact if item_id == "item_1" else None,
        )

        self.assertEqual(messages, [exact])
        self.assertIsNot(messages[0], exact)

    def test_item_reference_does_not_replace_audio_with_its_reference_only_transcript(self) -> None:
        """A public audio transcript is not treated as the model's audio input."""
        response_input = validate_response_input([{"type": "item_reference", "id": "item_audio"}])
        with self.assertRaises(RealtimeProtocolError) as raised:
            project_response_input_messages(
                response_input,
                resolve_item=lambda _item_id: {
                    "id": "item_audio",
                    "type": "message",
                    "role": "user",
                    "status": "completed",
                    "content": [{"type": "input_audio", "transcript": "book NV123"}],
                },
            )
        self.assertEqual(raised.exception.code, "unsupported_capability")
        self.assertEqual(raised.exception.param, "response.input[0].content[0]")

    def test_missing_reference_reports_its_input_index(self) -> None:
        """A missing journal item is correlated to its response input entry."""
        response_input = validate_response_input([{"type": "item_reference", "id": "missing"}])

        def missing(_item_id: str) -> dict:
            raise RealtimeProtocolError(message="not found", code="item_not_found", param="item_id")

        with self.assertRaises(RealtimeProtocolError) as raised:
            project_response_input_messages(response_input, resolve_item=missing)
        self.assertEqual(raised.exception.code, "item_not_found")
        self.assertEqual(raised.exception.param, "response.input[0].id")

    def test_inline_audio_and_mcp_items_are_not_silently_flattened(self) -> None:
        """Unsupported custom context types fail instead of being approximated."""
        with self.assertRaises(RealtimeProtocolError) as audio:
            validate_response_input(
                [
                    {
                        "type": "message",
                        "role": "user",
                        "content": [{"type": "input_audio", "audio": "AAAA"}],
                    }
                ]
            )
        self.assertEqual(audio.exception.code, "unsupported_capability")
        self.assertEqual(audio.exception.param, "response.input[0].content[0].type")

        with self.assertRaises(RealtimeProtocolError) as mcp:
            validate_response_input([{"type": "mcp_approval_response", "approval_request_id": "x", "approve": True}])
        self.assertEqual(mcp.exception.code, "unsupported_capability")
        self.assertEqual(mcp.exception.param, "response.input[0].type")


if __name__ == "__main__":
    unittest.main()
