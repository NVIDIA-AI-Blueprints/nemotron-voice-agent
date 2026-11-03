# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D103

"""Named texts and the instruction composer."""

import unittest
from dataclasses import replace

from delegation_core.im.input import compose
from delegation_core.im.results import SpeakDisplay
from tests.unit.delegation_core.helpers import (
    DEFAULTS,
    TEXTS,
    IMCase,
    ScriptedAdapter,
    answer,
    ask,
    read_all,
)


class ComposeTests(unittest.TestCase):
    def test_the_order_is_the_instructions_then_the_voice_note_then_the_format_note(self):
        self.assertEqual(compose("Be brief.", "VOICE", "FORMAT"), "Be brief.\nVOICE\nFORMAT")
        note = TEXTS.voice_notes["default"]  # each note on a line of its own
        self.assertEqual(compose("Be brief.", note), "Be brief.\n" + note.strip())

    def test_an_empty_note_adds_nothing_and_keeps_the_instructions_byte_for_byte(self):
        self.assertEqual(compose("Be brief.", "", ""), "Be brief.")
        self.assertEqual(compose("Be brief.\n", ""), "Be brief.\n")
        self.assertIsNone(compose(None, ""))


class VoiceNoteTests(unittest.TestCase):
    def test_the_note_describes_the_voice_channel_and_how_speech_recognition_writes_ids(self):
        note = " ".join(TEXTS.voice_notes["default"].split())
        for line in (
            "## Voice channel",
            "Write plain spoken sentences",
            "join them before using them in a tool",
            "Speech recognition guesses capital letters, spacing and symbols",
            "with spoken numbers as digits",
            "and trust the caller's words over the assistant's read-backs.",
            "Write an id in the form its tool describes.",
            "you may retry once with only its letter case or separators changed; never try other letters or digits.",
            "Then read back what you heard, letter by letter, and ask the caller to spell the part you are unsure of.",
            "Once a lookup in this call has found what an id names, do not ask the caller for that id again.",
            "Offer two or three options at a time",
        ):
            with self.subTest(line=line):
                self.assertIn(line, note)
        self.assertNotIn("likely written forms", note)  # no guessed variants of an id
        for rule in ("transfer", "check the rules", "Finish every part", "the records already show"):
            with self.subTest(rule=rule):
                self.assertNotIn(rule, note)  # a task's or a domain's rules belong in the backend's own prompt


class SessionInstructionTests(IMCase):
    async def test_each_round_reads_the_composed_instructions(self):
        speak_display = SpeakDisplay(TEXTS.result_formats["speak_display"])
        adapter = ScriptedAdapter([answer()])
        im = self.session(adapter, result_format=speak_display)
        await read_all(im.admit(ask("Find teas."), delegation_id="D1", settings={"instructions": "Be brief."}).stream)
        expected = compose("Be brief.", TEXTS.voice_notes["default"], TEXTS.result_formats["speak_display"])
        self.assertEqual(adapter.runs[0].requests[0]["instructions"], expected)

    async def test_a_request_without_instructions_gets_the_default_and_an_empty_note_adds_nothing(self):
        adapter = ScriptedAdapter([answer()])
        im = self.session(adapter, texts=replace(TEXTS, voice_notes={"default": ""}))
        await read_all(im.admit(ask("Find teas."), delegation_id="D1").stream)
        self.assertEqual(adapter.runs[0].requests[0]["instructions"], DEFAULTS.instructions)

    async def test_every_snapshot_of_a_response_reports_the_instructions_as_composed(self):
        # The official response object's instructions are what its request runs on: a client reads them there,
        # and Live's /history shows them. The default, then the voice note, when the request sets none.
        adapter = ScriptedAdapter([answer()])
        im = self.session(adapter)
        events = await read_all(im.admit(ask("Find teas."), delegation_id="D1").stream)
        snapshots = [event["response"] for event in events if "response" in event]
        expected = compose(DEFAULTS.instructions, TEXTS.voice_notes["default"])
        self.assertEqual({snapshot["instructions"] for snapshot in snapshots}, {expected})
        self.assertEqual(adapter.runs[0].requests[0]["instructions"], expected)
