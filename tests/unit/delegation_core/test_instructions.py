# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D103

"""Named texts and the instruction composer."""

import unittest
from dataclasses import replace

from examples.frontend_backend_live.delegation_core.im.input import compose
from examples.frontend_backend_live.delegation_core.im.results import SpeakDisplay
from tests.unit.delegation_core.helpers import (
    DEFAULTS,
    TEXTS,
    IMCase,
    ScriptedAdapter,
    answer,
    ask,
    read_all,
)


def notes():
    """The notes the shipped settings add to every task's instructions, in order: the caller note, the voice note."""
    return TEXTS.caller_notes["default"], TEXTS.voice_notes["default"]


class ComposeTests(unittest.TestCase):
    def test_the_order_is_the_instructions_then_the_caller_note_the_voice_note_and_the_format_note(self):
        self.assertEqual(compose("Be brief.", "CALLER", "VOICE", "FORMAT"), "Be brief.\nCALLER\nVOICE\nFORMAT")
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


class CallerNoteTests(unittest.TestCase):
    def test_the_note_says_how_to_work_with_the_caller(self):
        note = " ".join(TEXTS.caller_notes["default"].split())
        for line in (
            "## Working with the caller",
            "a fact about the caller's situation that no tool shows",
            "ask for it in one short question early",
            "Do not hand the caller off or give up while that question is unasked",
            "Before you hand the caller off, check the steps your instructions give for this problem",
            "A refusal your instructions require is not a problem to work around.",
            "Before any change the caller cannot easily undo, describe the exact change",
            "including any amount and how it is paid or refunded",
            "Act only on a clear yes the caller gave after hearing that whole description",
            "a yes to something else, or the request itself, is not that yes.",
        ):
            with self.subTest(line=line):
                self.assertIn(line, note)

    def test_the_note_comes_before_the_voice_note_and_none_adds_nothing(self):
        composed = compose("Be brief.", *notes())
        self.assertLess(composed.index("## Working with the caller"), composed.index("## Voice channel"))
        self.assertEqual(TEXTS.caller_notes["none"], "")
        self.assertEqual(compose("Be brief.", TEXTS.caller_notes["none"]), "Be brief.")


class SessionInstructionTests(IMCase):
    async def test_each_round_reads_the_composed_instructions(self):
        speak_display = SpeakDisplay(TEXTS.result_formats["speak_display"])
        adapter = ScriptedAdapter([answer()])
        im = self.session(adapter, result_format=speak_display)
        await read_all(im.admit(ask("Find teas."), delegation_id="D1", settings={"instructions": "Be brief."}).stream)
        expected = compose("Be brief.", *notes(), TEXTS.result_formats["speak_display"])
        self.assertEqual(adapter.runs[0].requests[0]["instructions"], expected)

    async def test_a_request_without_instructions_gets_the_default_and_empty_notes_add_nothing(self):
        adapter = ScriptedAdapter([answer()])
        im = self.session(adapter, texts=replace(TEXTS, caller_notes={"default": ""}, voice_notes={"default": ""}))
        await read_all(im.admit(ask("Find teas."), delegation_id="D1").stream)
        self.assertEqual(adapter.runs[0].requests[0]["instructions"], DEFAULTS.instructions)

    async def test_every_snapshot_of_a_response_reports_the_instructions_as_composed(self):
        # The official response object's instructions are what its request runs on: a client reads them there,
        # and Live's /history shows them. The default, then the notes, when the request sets none.
        adapter = ScriptedAdapter([answer()])
        im = self.session(adapter)
        events = await read_all(im.admit(ask("Find teas."), delegation_id="D1").stream)
        snapshots = [event["response"] for event in events if "response" in event]
        expected = compose(DEFAULTS.instructions, *notes())
        self.assertEqual({snapshot["instructions"] for snapshot in snapshots}, {expected})
        self.assertEqual(adapter.runs[0].requests[0]["instructions"], expected)
