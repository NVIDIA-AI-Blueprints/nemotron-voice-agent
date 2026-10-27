# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102

"""The example's core settings file (``core.yaml``): it loads with its defaults, voice note v2 included."""

import unittest
from pathlib import Path

from delegation_core.backend.control import ControlVocabulary
from delegation_core.config import CoreSettings

CORE_YAML = Path(__file__).resolve().parents[3] / "src/examples/frontend_backend_live/core.yaml"

# The voice-channel rules the v2 voice note starts with, byte for byte.
TODAYS_V2_RULES = """
## Voice channel
Your reply is spoken to the caller by a voice frontend, which may already have said a short
acknowledgment such as "Let me check." Start with the substance; do not greet or introduce
yourself again.
- Write plain spoken sentences: no markdown, lists, tables, links, or emoji.
- Put any question or request for confirmation at the end.
- Caller messages are speech-recognition transcripts. Spelled letters and spoken numbers arrive
  as separate words ("J O H N", "four four seven two"); join them before using them in a tool
  call. Words can be misheard, and the caller's latest correction wins.
"""


class CoreSettingsFileTests(unittest.TestCase):
    def setUp(self):
        self.settings = CoreSettings.load(CORE_YAML)

    def test_the_file_loads_with_the_frozen_defaults(self):
        delegation, backend = self.settings.delegation, self.settings.backend
        self.assertEqual((backend.adapter, backend.one_shot_adapter), ("llm", "llm"))
        self.assertEqual((backend.slot, backend.prompt), ("thinker-llm", "thinker"))
        self.assertEqual(sorted(backend.blocks), ["harness", "llm", "llm-agent"])
        self.assertEqual(backend.blocks["llm"]["conductor"], "steer")
        self.assertEqual((delegation.result_format, delegation.voice_note, delegation.work_tools), ("plain", "v2", ()))

    def test_the_v2_voice_note_keeps_todays_rules_and_carries_the_owners_rows(self):
        v2 = self.settings.texts.voice_notes["v2"]
        self.assertTrue(v2.startswith(TODAYS_V2_RULES), v2)
        self.assertEqual(
            [line for line in v2.splitlines() if line.startswith("## ")][1:], ["## Spoken ids", "## Requests"]
        )
        self.assertNotIn("spell them again", v2)
        self.assertEqual(self.settings.texts.voice_notes["v1"], "")

    def test_the_control_tools_make_a_vocabulary(self):
        vocabulary = ControlVocabulary(self.settings.texts.control_tools, self.settings.texts.conductor_prompt)
        names = [tool["name"] for tool in vocabulary.conductor_tools()]
        self.assertEqual(names, ["start_work", "steer_work", "cancel_work", "answer_status"])
