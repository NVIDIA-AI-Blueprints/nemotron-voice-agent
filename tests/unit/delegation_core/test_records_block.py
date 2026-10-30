# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D103

"""The Work records block in every delegation input."""

import asyncio
import json

from tests.unit.delegation_core.helpers import (
    CORE_SETTINGS,
    TEXTS,
    IMCase,
    ScriptedAdapter,
    StartEach,
    answer,
    ask,
    checkpoint,
    own_call,
    read_all,
    wait,
)

MAX_ENDED = CORE_SETTINGS.delegation.records_max_works


def records_of(message: dict) -> list[dict]:
    """The work records a request's message carries, one per line."""
    parts = [part["text"] for part in message["content"]]
    block = next(part for part in parts if part.startswith(TEXTS.records_block.split("{records}")[0]))
    return [json.loads(line) for line in block.splitlines()[1:]]


class RecordsBlockTests(IMCase):
    async def test_the_first_request_carries_no_records(self):
        adapter = ScriptedAdapter([answer()])
        im = self.session(adapter)
        await read_all(im.admit(ask("Find teas."), delegation_id="D1").stream)
        self.assertEqual(adapter.runs[0].histories[0][-1]["content"], [{"type": "input_text", "text": "Find teas."}])

    async def test_a_new_request_after_an_answered_work_carries_its_steps(self):
        adapter = ScriptedAdapter([own_call("book_table"), answer("Booked.")], [answer()], conductor=StartEach())
        im = self.session(adapter)
        await read_all(im.admit(ask("Book a table."), delegation_id="D1").stream)
        await read_all(im.admit(ask("Is it booked?"), delegation_id="D2").stream)
        message = adapter.runs[1].histories[0][-1]
        self.assertEqual(message["content"][0]["text"], "Is it booked?")
        self.assertEqual(message["content"][-1]["text"], TEXTS.closing_line)  # after each records block
        (record,) = records_of(message)
        self.assertEqual(
            record, {"work": "W1", "state": "answered", "requests": ["Book a table."], "steps": ["book_table: done"]}
        )

    async def test_a_steered_request_carries_the_records_and_the_closing_line_when_it_lands(self):
        gate = asyncio.Event()
        adapter = ScriptedAdapter([own_call("search_flights"), wait(gate), checkpoint(), answer()])
        im = self.session(adapter)
        im.admit(ask("Book Denver."), delegation_id="D1")
        await self.turns()
        im.admit(ask("Make it Tuesday."), delegation_id="D2")
        await self.turns()
        gate.set()
        await self.turns()
        (steering,) = adapter.runs[0].applied
        parts = [part["text"] for part in steering.items[-1]["content"]]
        self.assertEqual(parts[0], "Make it Tuesday.")
        self.assertEqual(parts[-1], TEXTS.closing_line)
        (record,) = records_of(steering.items[-1])
        self.assertEqual(
            (record["work"], record["state"], record["steps"]), ("W1", "running", ["search_flights: done"])
        )
        history = im.works["W1"].tasks[0].view.items if "W1" in im.works else im.ended_works[-1].tasks[0].view.items
        self.assertIn(steering.items[-1], history)

    async def test_the_records_keep_the_open_works_and_the_newest_ended_ones(self):
        scripts = [[answer(f"Answer {n}.")] for n in range(MAX_ENDED + 2)]
        adapter = ScriptedAdapter(*scripts, conductor=StartEach())
        im = self.session(adapter)
        for n in range(MAX_ENDED + 2):
            await read_all(im.admit(ask(f"Question {n}."), delegation_id=f"D{n}").stream)
        works = [record["work"] for record in records_of(adapter.runs[-1].histories[0][-1])]
        self.assertEqual(len(works), MAX_ENDED)
        self.assertEqual(works[-1], f"W{MAX_ENDED + 1}")
