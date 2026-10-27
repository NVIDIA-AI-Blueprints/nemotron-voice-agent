# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D103

"""Named texts and the instruction composer."""

import unittest

from delegation_core.im.input import compose, task_request
from delegation_core.im.results import SpeakDisplay
from tests.unit.delegation_core.helpers import (
    CORE_SETTINGS,
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

    def test_an_empty_note_adds_nothing_and_today_s_v1_composes_byte_for_byte(self):
        self.assertEqual(compose("Be brief.", "", ""), "Be brief.")
        self.assertEqual(compose("Be brief.\n", TEXTS.voice_notes["v1"]), "Be brief.\n")
        self.assertIsNone(compose(None, ""))

    def test_v2_composes_as_todays_join_did(self):
        note = TEXTS.voice_notes["v2"]
        self.assertEqual(compose("Be brief.", note), "Be brief.\n" + note.strip())

    def test_task_request_keeps_the_settings_and_composes_the_instructions(self):
        request = task_request(DEFAULTS, "VOICE")
        self.assertEqual(request["instructions"], "Do the task.\nVOICE")
        self.assertEqual((request["model"], request["tools"], request["tool_choice"]), ("test/model", [], "auto"))


class VoiceNoteTests(unittest.TestCase):
    def test_v2_is_the_default(self):
        self.assertEqual(CORE_SETTINGS.delegation.voice_note, "v2")

    def test_v2_keeps_the_voice_channel_and_gains_the_owners_rules(self):
        note = TEXTS.voice_notes["v2"]
        for line in (
            "## Voice channel",
            "Write plain spoken sentences",
            "join them before using them in a tool",
            "look it up before asking them to spell it",
            "spoken numbers as digits",
            "trust what the caller says over your own read-back",
            "try it once more in lowercase",
            "Try the ids you have before you transfer the caller",
            "offer a transfer after a few failed",
            "Do not ask for any check your instructions do not require",
            "Finish every part of the request, and check the rules before each change",
            "Ask for all missing details in one question",
            "never ask for a detail the records already show",
            "offer two or three options at a time",
        ):
            with self.subTest(line=line):
                self.assertIn(line, " ".join(note.split()))


class SessionInstructionTests(IMCase):
    async def test_each_round_reads_the_composed_instructions(self):
        speak_display = SpeakDisplay(TEXTS.result_formats["speak_display"])
        adapter = ScriptedAdapter([answer()])
        im = self.session(adapter, result_format=speak_display)
        await read_all(im.admit(ask("Find teas."), delegation_id="D1", settings={"instructions": "Be brief."}).stream)
        expected = compose("Be brief.", TEXTS.voice_notes["v2"], TEXTS.result_formats["speak_display"])
        self.assertEqual(adapter.runs[0].requests[0]["instructions"], expected)

    async def test_a_request_without_instructions_gets_the_default_and_v1_adds_nothing(self):
        adapter = ScriptedAdapter([answer()])
        im = self.session(adapter, voice_note="v1")
        await read_all(im.admit(ask("Find teas."), delegation_id="D1").stream)
        self.assertEqual(adapter.runs[0].requests[0]["instructions"], DEFAULTS.instructions)

    async def test_every_snapshot_of_a_response_reports_the_instructions_as_composed(self):
        # The official response object's instructions are what its request runs on: a client reads them there,
        # and Live's /history shows them. The default, then the voice note, when the request sets none.
        adapter = ScriptedAdapter([answer()])
        im = self.session(adapter)
        events = await read_all(im.admit(ask("Find teas."), delegation_id="D1").stream)
        snapshots = [event["response"] for event in events if "response" in event]
        expected = compose(DEFAULTS.instructions, TEXTS.voice_notes["v2"])
        self.assertEqual({snapshot["instructions"] for snapshot in snapshots}, {expected})
        self.assertEqual(adapter.runs[0].requests[0]["instructions"], expected)
