# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D103

"""The talker as the prompt set shapes it: the floor, the set's actions and rules, and the history it sends."""

import asyncio
import unittest

from examples.frontend_backend_live.cascade.talker.talker import Talker
from examples.frontend_backend_live.models.chat_completions import chat_messages
from examples.frontend_backend_live.models.decision import TalkerDecision, decision_schema
from examples.frontend_backend_live.prompts import get_prompt_set
from tests.unit.cascade_helpers import ScriptedFrontend

DEFAULT = get_prompt_set("default")


def talker(prompts, script=()):
    return Talker(ScriptedFrontend(list(script)), prompts)


class FloorTests(unittest.TestCase):
    def test_a_reply_and_a_delegation_always_stand(self):
        for decision in (TalkerDecision("speak", "Hello."), TalkerDecision("delegate", "")):
            with self.subTest(action=decision.action):
                self.assertIs(talker(DEFAULT).floored(decision), decision)

    def test_a_wait_stands_and_a_reply_with_nothing_to_say_waits(self):
        kept = talker(DEFAULT).floored(TalkerDecision("wait", "Hm.", {"response_id": "r1"}))
        self.assertEqual((kept.action, kept.speech, kept.diagnostics), ("wait", "", {"response_id": "r1"}))
        for decision in (TalkerDecision("speak"), TalkerDecision("speak", "  ")):
            with self.subTest(speech=decision.speech):
                floored = talker(DEFAULT).floored(decision)
                self.assertEqual((floored.action, floored.diagnostics["recovery"]), ("wait", "empty_speak"))


class SetShapeTests(unittest.TestCase):
    def test_the_set_declares_wait_in_its_actions_its_schema_and_its_hint(self):
        self.assertEqual(DEFAULT.actions, ("speak", "delegate", "wait"))
        schema = decision_schema(talker(DEFAULT).spec)
        self.assertEqual(schema["properties"]["action"]["enum"], ["speak", "delegate", "wait"])
        self.assertIn("For wait: empty.", schema["properties"]["speech"]["description"])
        self.assertIn('"wait"', DEFAULT.routing)

    def test_the_set_reads_spawnthinking_as_a_delegation_and_never_names_the_backend_to_the_caller(self):
        self.assertIn("<|SpawnThinking|>", DEFAULT.frame_after)
        self.assertIn("never mention a backend", DEFAULT.frame_before)
        self.assertIn("never mention a backend", DEFAULT.writer_frame_before)

    def test_the_rule_on_details_delegates_a_correction_or_an_addition(self):
        self.assertIn("corrects or adds to details given earlier", DEFAULT.routing)


class HistoryTests(unittest.TestCase):
    def test_the_talker_sends_the_history_it_is_given(self):
        # The memory labels the talker's view (TalkerMemory.labeled); the talker adds only its runtime data after it.
        history = [
            {"role": "user", "content": DEFAULT.result_label + "Morning slot is free. Shall I book it?"},
            {"role": "user", "content": "Yes."},
        ]
        frontend = ScriptedFrontend([TalkerDecision("delegate", "")])
        asyncio.run(Talker(frontend, DEFAULT).decide("Be Sam.", history))
        sent = frontend.calls[0]["history"]
        self.assertEqual(sent[:-1], history)
        self.assertEqual(sent[-1], {"role": "user", "content": "\n" + talker(DEFAULT).runtime_data()})

    def test_the_work_line_and_the_cue_come_after_the_conversation_and_never_in_the_system_message(self):
        # The work line and the cue close the caller's newest turn, a blank line after it, where the model reads last;
        # in client mode there is no work line, and the cue still closes the turn.
        work = DEFAULT.work_in_progress.format(items='1. running for 3 s: "Find teas."')
        history = [{"role": "assistant", "content": "Let me find our teas."}, {"role": "user", "content": "Okay."}]
        for mode, work_line in (("server", work), ("client", "")):
            with self.subTest(mode=mode):
                frontend = ScriptedFrontend([TalkerDecision("wait", "")])
                asyncio.run(Talker(frontend, DEFAULT).decide("Be Sam.", history, work_line))
                call = frontend.calls[0]
                messages = chat_messages(call["instructions"], call["history"])
                self.assertNotIn(DEFAULT.runtime_label, messages[0]["content"])
                self.assertNotIn("Work in progress", messages[0]["content"])
                expected = f"Okay.\n\n{DEFAULT.runtime_label}\n" + (f"{work_line}\n\n" if work_line else "")
                self.assertEqual(messages[-1], {"role": "user", "content": expected + DEFAULT.decision_cue})

    def test_with_no_caller_turn_last_the_runtime_data_starts_its_own_message(self):
        # An instruction decided with no caller turn waiting: the history ends with the voice's line.
        frontend = ScriptedFrontend([TalkerDecision("wait", "")])
        asyncio.run(Talker(frontend, DEFAULT).decide("Be Sam.", [{"role": "assistant", "content": "Hello."}]))
        self.assertEqual(frontend.calls[0]["history"][-1]["content"], talker(DEFAULT).runtime_data())


if __name__ == "__main__":
    unittest.main()
