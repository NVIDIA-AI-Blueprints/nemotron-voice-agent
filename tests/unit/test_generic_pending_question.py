# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""A clarification question the caller did not answer comes back on a progress check."""

# ruff: noqa: D101, D102

from __future__ import annotations

import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from examples.frontend_backend_agent.generic import backend as backend_module
from examples.frontend_backend_agent.generic.backend import GenericThinkerBackend
from examples.frontend_backend_agent.generic.client_tools import build_client_tool_specs, format_client_result
from examples.frontend_backend_agent.generic.result_formatters import (
    confirmation_request,
    invalid_parameters,
    missing_client_parameters,
)
from examples.frontend_backend_agent.src.tool_handlers import _pending_question_for_turn


def _schema(name: str = "get_record") -> dict:
    return {
        "type": "function",
        "name": name,
        "description": "Get a record by id.",
        "parameters": {
            "type": "object",
            "properties": {"record_id": {"type": "string"}},
            "required": ["record_id"],
            "additionalProperties": False,
        },
    }


def _backend(*, enabled: bool = True) -> GenericThinkerBackend:
    with patch.dict(os.environ, {"FRONTEND_BACKEND_PENDING_QUESTION": str(enabled).lower()}):
        return GenericThinkerBackend(
            planner=SimpleNamespace(plan=None),
            tools={},
            enabled_tools=("get_record",),
            client_tools=build_client_tool_specs((_schema(),)),
        )


def _question() -> dict:
    return missing_client_parameters("get_record", ["record id"], ["record_id"])


class PendingQuestionTests(unittest.IsolatedAsyncioTestCase):
    async def test_a_question_is_offered_again_at_most_twice(self) -> None:
        backend = _backend()
        backend._retain_question(_question())

        first = backend.take_pending_question()
        second = backend.take_pending_question()
        third = backend.take_pending_question()

        self.assertEqual(first["response_text"], "Please tell me record id.")
        self.assertEqual(second, first)
        self.assertIsNone(third)
        self.assertIsNone(backend._pending_question)

    async def test_the_same_question_asked_again_keeps_its_count(self) -> None:
        backend = _backend()
        backend._retain_question(_question())
        backend.take_pending_question()
        backend.take_pending_question()

        backend._retain_question(_question())

        self.assertIsNone(backend.take_pending_question())

    async def test_only_code_written_clarifications_are_kept(self) -> None:
        backend = _backend()
        kept = invalid_parameters("get_record")
        for payload in (
            confirmation_request("get_record", {"record_id": "A1"}, description="Get a record by id."),
            format_client_result("get_record", {"record_id": "A1"}, {"id": "A1"}),
            {**_question(), "context": "call_backend"},
            {**_question(), "speakable": False},
        ):
            with self.subTest(reason=payload.get("reason") or payload.get("status")):
                backend._retain_question(payload)
                self.assertIsNone(backend._pending_question)

        backend._retain_question(kept)
        self.assertEqual(backend.take_pending_question()["reason"], "params_invalid")

    async def test_nothing_is_kept_with_the_switch_off(self) -> None:
        backend = _backend(enabled=False)

        backend._retain_question(_question())

        self.assertIsNone(backend.take_pending_question())

    async def test_never_offered_while_a_request_is_running(self) -> None:
        backend = _backend()
        backend._retain_question(_question())

        with patch.object(GenericThinkerBackend, "running_query", return_value="still checking"):
            self.assertIsNone(backend.take_pending_question())
        self.assertIsNotNone(backend.take_pending_question())

    async def test_an_old_question_expires(self) -> None:
        backend = _backend()
        backend._retain_question(_question())
        backend._pending_question.created_at -= backend_module._PENDING_QUESTION_TTL_SECONDS + 1

        self.assertIsNone(backend.take_pending_question())
        self.assertIsNone(backend._pending_question)

    async def test_a_successful_call_of_the_same_tool_answers_it(self) -> None:
        backend = _backend()
        backend._retain_question(_question())

        backend._clear_answered_question(format_client_result("other_tool", {}, {"id": "x"}))
        self.assertIsNotNone(backend._pending_question)
        backend._clear_answered_question(format_client_result("get_record", {"record_id": "A1"}, {"id": "A1"}))
        self.assertIsNone(backend._pending_question)

    async def test_cancellation_clears_it(self) -> None:
        backend = _backend()
        backend._retain_question(_question())

        backend.cancel_active("user_cancelled")

        self.assertIsNone(backend._pending_question)

    async def test_a_withdrawn_superseded_run_keeps_nothing(self) -> None:
        backend = _backend()
        backend._owner_gone.add("run-1")

        backend._retain_superseded_question("run-1", _question())
        self.assertIsNone(backend._pending_question)

        backend._retain_superseded_question("run-2", _question())
        self.assertIsNotNone(backend._pending_question)


class _Transcript:
    def __init__(self, text: str) -> None:
        self.text = text

    def latest_user_text(self) -> str:
        return self.text


class PendingQuestionTurnTests(unittest.IsolatedAsyncioTestCase):
    async def test_an_acknowledgement_or_progress_check_brings_it_back(self) -> None:
        for utterance in ("okay", "any update?"):
            with self.subTest(utterance=utterance):
                backend = _backend()
                backend._retain_question(_question())

                payload = _pending_question_for_turn(backend, _Transcript(utterance))

                self.assertEqual(payload["reason"], "params_missing")

    async def test_a_substantive_turn_clears_it(self) -> None:
        backend = _backend()
        backend._retain_question(_question())

        payload = _pending_question_for_turn(backend, _Transcript("my record is A as in apple one"))

        self.assertIsNone(payload)
        self.assertIsNone(backend._pending_question)

    async def test_a_backend_without_the_feature_is_untouched(self) -> None:
        self.assertIsNone(_pending_question_for_turn(SimpleNamespace(), _Transcript("okay")))


if __name__ == "__main__":
    unittest.main()
