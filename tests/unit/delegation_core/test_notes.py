# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D103

"""Progress notes: commentary that keeps the voice current, at most one per delegation per interval."""

import asyncio

from delegation_core.im.stream import message_text
from tests.unit.delegation_core.helpers import (
    IMCase,
    ScriptedAdapter,
    answer,
    ask,
    finished_items,
    progress,
    read_all,
    wait,
)


def notes(events: list[dict]) -> list[str]:
    return [
        item["content"][0]["text"]
        for item in finished_items(events)
        if item["type"] == "message" and item["phase"] == "commentary"
    ]


class NoteTests(IMCase):
    async def test_one_note_per_interval_and_the_newest_waiting_one_wins(self):
        gate = asyncio.Event()
        script = [progress("First."), progress("Second."), progress("Third."), wait(gate), answer()]
        im = self.session(ScriptedAdapter(script), progress_note_seconds=0.05)
        stream = im.admit(ask("Find teas."), delegation_id="D1").stream
        await self.turns()
        await asyncio.sleep(0.1)
        gate.set()
        self.assertEqual(notes(await read_all(stream)), ["First.", "Third."])

    async def test_a_waiting_note_is_dropped_when_the_response_ends(self):
        im = self.session(ScriptedAdapter([progress("First."), progress("Second."), answer()]), progress_note_seconds=5)
        events = await read_all(im.admit(ask("Find teas."), delegation_id="D1").stream)
        self.assertEqual(notes(events), ["First."])
        await asyncio.sleep(0)
        self.assertEqual(im.notes._waiting, {})

    async def test_zero_writes_no_notes(self):
        im = self.session(ScriptedAdapter([progress("First."), answer()]), progress_note_seconds=0)
        events = await read_all(im.admit(ask("Find teas."), delegation_id="D1").stream)
        self.assertEqual(notes(events), [])
        self.assertEqual(im.ledger.works["W1"].state, "answered")

    async def test_a_note_is_never_the_result(self):
        im = self.session(ScriptedAdapter([progress("Halfway."), answer("Two teas.")]))
        events = await read_all(im.admit(ask("Find teas."), delegation_id="D1").stream)
        output = events[-1]["response"]["output"]
        self.assertEqual(
            [(item["phase"], item["content"][0]["text"]) for item in output],
            [
                ("commentary", "Halfway."),
                ("final_answer", "Two teas."),
            ],
        )
        result = im.ledger.tasks["W1.1"].result  # the task's own result: its answer, written (pushed)
        self.assertEqual((result.state, message_text(result.items[-1])), ("pushed", "Two teas."))

    async def test_the_newest_note_is_the_tasks_progress_for_a_conductor(self):
        gate = asyncio.Event()
        im = self.session(ScriptedAdapter([progress("Checking stock."), wait(gate), answer()]))
        im.admit(ask("Find teas."), delegation_id="D1")
        await self.turns()
        self.assertEqual(im.works["W1"].summary().tasks[0].progress, "Checking stock.")
        self.assertEqual(im.works["W1"].record_line()["progress"], "Checking stock.")
        gate.set()
