# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D103

"""The Pipecat adapter speaks v1's sentences: its splitter gives v1's split (B3).

``core/``'s import rule is checked in ``test_fba_voice_config``.
"""

from __future__ import annotations

import unittest

from examples.frontend_backend_agent.adapters.pipecat.speech_output import split_sentences

#: v1's ``split_sentences`` (``voice/engine/output_path.py`` at ``fe7622d``) on these inputs.
V1_SPLITS: dict[str, list[str]] = {
    "Hello there. How can I help?": ["Hello there.", "How can I help?"],
    "One sentence only": ["One sentence only"],
    # v1's separator swallows a closing quote or bracket after the punctuation; kept as is.
    'He said "stop." Then left!': ['He said "stop.', "Then left!"],
    "Line one\nLine two": ["Line one", "Line two"],
    "Spaces   inside.   Next one?  ": ["Spaces inside.", "Next one?"],
    "Version 2.5 is out. Done.": ["Version 2.5 is out.", "Done."],
    "Wait (really?) yes.": ["Wait (really?", "yes."],
    "": [],
}


class SplitterTests(unittest.TestCase):
    def test_same_split_as_v1(self) -> None:
        for text, expected in V1_SPLITS.items():
            with self.subTest(text=text):
                self.assertEqual(split_sentences(text), expected)

    def test_disabled_keeps_one_sentence(self) -> None:
        self.assertEqual(split_sentences("A. B.", enabled=False), ["A. B."])


if __name__ == "__main__":
    unittest.main()
