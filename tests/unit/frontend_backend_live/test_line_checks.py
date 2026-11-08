# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D103
"""The relay's one wording check on a backend question: its last question stays the line's last sentence."""

import unittest

from examples.frontend_backend_live.models.decision import TalkerDecision
from examples.frontend_backend_live.talker import relay_line

UPDATE = "Your table is ready. Shall I hold it?"


def worded(line: str, asked: str = UPDATE, **checks) -> str | None:
    return relay_line(TalkerDecision("speak", line), "Update: " + asked, [], asked=asked, **checks)


class QuestionCheckTests(unittest.TestCase):
    def test_a_line_that_drops_the_last_question_is_rejected(self):
        self.assertIsNone(worded("Your table is ready."))

    def test_a_line_that_moves_the_question_forward_is_rejected(self):
        self.assertIsNone(worded("Shall I hold it? Your table is ready."))

    def test_a_line_that_keeps_the_question_last_in_other_words_passes(self):
        line = "Good news, your table is ready. Would you like me to hold it for you?"
        self.assertEqual(worded(line), line)

    def test_an_update_with_no_final_question_is_not_checked(self):
        self.assertEqual(worded("It is ready.", asked="Shall I hold it? Your table is ready."), "It is ready.")
        self.assertEqual(worded("It is ready.", asked=""), "It is ready.")  # a batch with commentary: no check

    def test_a_question_mark_in_another_script_counts(self):
        self.assertIsNone(worded("Hay una mesa.", asked="Hay una mesa. ¿Quiere una mesa?", language="es"))
        self.assertEqual(worded("¿La reservo?", asked="Hay una mesa. ¿Quiere una mesa?", language="es"), "¿La reservo?")
        self.assertIsNone(worded("席があります。", asked="席があります。予約しますか？", language="ja"))

    def test_the_check_off_keeps_todays_rules(self):
        self.assertEqual(worded("Your table is ready.", keep_questions=False), "Your table is ready.")

    def test_todays_rejections_still_hold(self):
        history = [{"role": "assistant", "content": "Your table is ready. Shall I hold it?"}]
        self.assertIsNone(relay_line(TalkerDecision("delegate", "x"), UPDATE, [], asked=UPDATE))
        self.assertIsNone(relay_line(TalkerDecision("speak", " "), UPDATE, [], asked=UPDATE))
        self.assertIsNone(relay_line(TalkerDecision("speak", UPDATE), UPDATE, [], asked=UPDATE))
        self.assertIsNone(relay_line(TalkerDecision("speak", UPDATE.upper()), "x", history, asked=UPDATE))


if __name__ == "__main__":
    unittest.main()
