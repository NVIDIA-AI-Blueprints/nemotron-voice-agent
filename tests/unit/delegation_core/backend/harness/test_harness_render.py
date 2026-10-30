# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D103

"""What the runtime reads (``harness/render.py``): a worker's transcript, a steering batch, a conductor's input."""

import unittest

from delegation_core.backend.harness.render import Renderer
from delegation_core.backend.harness.settings import HarnessPrompts
from delegation_core.backend.port import ConductorInput, Steering, TaskSummary, WorkSummary
from tests.unit.contracts.backend import a_request, an_ended_work, an_open_work
from tests.unit.delegation_core.backend.harness.doubles import PROMPTS

HISTORY = [
    {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "Caller: find teas"}]},
    {"type": "function_call", "call_id": "c1", "name": "get_menu", "arguments": '{"kind": "tea"}'},
    {"type": "function_call_output", "call_id": "c1", "output": '{"teas": ["green"]}'},
    {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "We have green tea."}]},
    {"type": "message", "role": "user", "content": "Task: also the hours"},
]


class RenderTests(unittest.TestCase):
    def setUp(self):
        self.render = Renderer(HarnessPrompts.parse(PROMPTS))

    def test_a_transcript_writes_each_item_and_names_each_output_by_its_call(self):
        self.assertEqual(
            self.render.transcript(HISTORY),
            "Caller: find teas\n\n"
            'Earlier step, already done: get_menu with {"kind": "tea"}\n\n'
            'Result of get_menu: {"teas": ["green"]}\n\n'
            "Assistant: We have green tea.\n\n"
            "Task: also the hours",
        )
        self.assertTrue(self.render.worker_input(HISTORY).startswith("The conversation and your task:\n\nCaller:"))

    def test_an_output_whose_call_is_not_in_the_transcript_names_it_by_the_unnamed_call_prompt(self):
        output = {"type": "function_call_output", "call_id": "c9", "output": "ok"}
        self.assertEqual(self.render.transcript([output]), "Result of the call: ok")

    def test_a_steering_batch_is_its_requests(self):
        steering = Steering("S1", ({"type": "message", "role": "user", "content": "Make it two."},))
        self.assertEqual(self.render.steering(steering), "Make it two.")

    def test_a_conductors_input_holds_the_requests_the_records_and_what_it_has_not_read(self):
        conversation = (
            {"type": "message", "role": "user", "content": "Old request."},
            {"type": "message", "role": "user", "content": "Find teas."},
        )
        given = ConductorInput((a_request("d_new", "Find teas."),), (an_open_work(), an_ended_work()), conversation, 1)
        text = self.render.conductor_input(given, fresh=False)
        self.assertIn("New requests to place:\n[d_new] Find teas.", text)
        self.assertIn("[W1] running, carried by d_old; requests: [d_old] Find teas.", text)
        self.assertIn("task W1.1: running for 3 s; steps: get_menu: done; newest note: Looking it up.", text)
        self.assertIn("[W0] answered, carried by d_done", text)
        self.assertNotIn("Old request.", text)  # read before
        self.assertIn("Old request.", self.render.conductor_input(given, fresh=True))

    def test_a_brief_and_empty_parts_read_as_none(self):
        task = TaskSummary("W2.1", "queued", "the hours", (), False, None, 0.0)
        work = WorkSummary("W2", "queued", "d_9", (), (task,), 0.0)
        text = self.render.conductor_input(ConductorInput((), (work,), (), 0), fresh=True)
        self.assertIn('task W2.1: queued for 0 s, brief "the hours"; steps: none; newest note: none', text)
        self.assertIn("New requests to place:\nnone", text)
        self.assertIn("Conversation since your last turn:\nnone", text)


if __name__ == "__main__":
    unittest.main()
