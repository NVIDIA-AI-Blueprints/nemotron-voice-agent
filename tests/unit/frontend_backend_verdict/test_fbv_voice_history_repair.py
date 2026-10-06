# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D100, D101, D102, D103, D107

"""Rule 11c: every stored copy of an interrupted answer is rewritten, verified by exact text."""

from __future__ import annotations

import unittest

from examples.frontend_backend_verdict.text.history import History
from examples.frontend_backend_verdict.text.messages import Message, ToolCall
from examples.frontend_backend_verdict.text.session import SessionState
from examples.frontend_backend_verdict.voice.agent.history_repair import repair_interrupted_answer
from examples.frontend_backend_verdict.voice.errors import HistoryRepairError

FULL = "Your refund is approved. It arrives in five days."
HEARD = "Your refund is approved. [interrupted by the user]"


def _delegated(answer: str = FULL) -> SessionState:
    call = ToolCall(id="fc1", name="call_backend", arguments_json='{"query": "refund"}')
    history = History(
        (
            Message.assistant("Hi! How can I help you today?"),
            Message.user("earlier"),
            Message.assistant("earlier answer"),
            Message.user("refund please"),
            Message.assistant_tool_calls((call,)),
            Message.tool("fc1", answer),
            Message.assistant(answer),
        )
    )
    return SessionState(frontend_history=history)


class HistoryRepairTests(unittest.TestCase):
    def test_paired_delegated_turn_rewrites_both_copies(self) -> None:
        state = repair_interrupted_answer(_delegated(), full_text=FULL, replacement=HEARD, frontend_enabled=True)
        messages = state.frontend_history.messages
        self.assertEqual(messages[-2].content, HEARD)
        self.assertEqual(messages[-1].content, HEARD)
        self.assertEqual(messages[-3].tool_calls[0].id, "fc1")
        self.assertFalse(any("five days" in (m.content or "") for m in messages))
        self.assertEqual(messages[2].content, "earlier answer")

    def test_direct_answer(self) -> None:
        state = SessionState(frontend_history=History((Message.user("hi"), Message.assistant(FULL))))
        repaired = repair_interrupted_answer(state, full_text=FULL, replacement=HEARD, frontend_enabled=True)
        self.assertEqual(repaired.frontend_history.messages[-1].content, HEARD)

    def test_backend_only_rewrites_only_the_final_answer(self) -> None:
        call = ToolCall(id="t1", name="lookup", arguments_json="{}")
        history = History(
            (
                Message.user("refund"),
                Message.assistant_tool_calls((call,)),
                Message.tool("t1", "{}"),
                Message.assistant(FULL),
            )
        )
        repaired = repair_interrupted_answer(
            SessionState(backend_history=history), full_text=FULL, replacement=HEARD, frontend_enabled=False
        )
        self.assertEqual(repaired.backend_history.messages[-1].content, HEARD)
        self.assertEqual(repaired.backend_history.messages[2].content, "{}")

    def test_mismatch_raises_instead_of_rewriting(self) -> None:
        with self.assertRaises(HistoryRepairError):
            repair_interrupted_answer(
                _delegated("something else"), full_text=FULL, replacement=HEARD, frontend_enabled=True
            )
        with self.assertRaises(HistoryRepairError):
            repair_interrupted_answer(SessionState(), full_text=FULL, replacement=HEARD, frontend_enabled=True)

    def test_original_state_is_untouched(self) -> None:
        state = _delegated()
        repair_interrupted_answer(state, full_text=FULL, replacement=HEARD, frontend_enabled=True)
        self.assertEqual(state.frontend_history.messages[-1].content, FULL)
