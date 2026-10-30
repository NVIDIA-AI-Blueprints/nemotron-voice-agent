# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D103

"""Result formats: ``plain``, and ``speak_display`` with its ``show`` display tool."""

import asyncio
import json

from delegation_core.im.results import RESULT_FORMATS, SpeakDisplay
from tests.unit.delegation_core.helpers import (
    TEXTS,
    IMCase,
    ScriptedAdapter,
    answer,
    ask,
    finished_items,
    read_all,
    show,
    wait,
)

SPEAK_DISPLAY = SpeakDisplay(TEXTS.result_formats["speak_display"])


class EnvelopeTests(IMCase):
    async def tools_of(self, **kwargs) -> list[str]:
        gate = asyncio.Event()
        im = self.session(ScriptedAdapter([wait(gate), answer()]), **kwargs)
        im.admit(ask("Find teas."), delegation_id="D1")
        await self.turns()
        names = [tool["name"] for tool in im.works["W1"].tasks[0].tools().tools()]
        gate.set()
        return names

    async def test_speak_display_offers_show_with_the_texts_schema(self):
        self.assertEqual(await self.tools_of(result_format=SPEAK_DISPLAY, work_tools=()), ["show"])
        schema = TEXTS.work_tools["show"]
        gate = asyncio.Event()
        im = self.session(ScriptedAdapter([wait(gate), answer()]), result_format=SPEAK_DISPLAY, work_tools=())
        im.admit(ask("Find teas."), delegation_id="D2")
        await self.turns()
        (tool,) = next(iter(im.works.values())).tasks[0].tools().tools()
        self.assertEqual((tool["description"], tool["parameters"]), (schema["description"], dict(schema["parameters"])))
        gate.set()

    async def test_cancel_work_is_offered_when_work_tools_lists_it_as_the_shipped_file_does(self):
        self.assertEqual(await self.tools_of(result_format=SPEAK_DISPLAY), ["show", "cancel_work"])
        self.assertEqual(await self.tools_of(), ["cancel_work"])
        self.assertEqual(await self.tools_of(result_format=SPEAK_DISPLAY, work_tools=()), ["show"])
        self.assertEqual(await self.tools_of(work_tools=()), [])  # plain, with no work tool listed

    async def test_the_show_items_label_is_the_display_label_setting(self):
        adapter = ScriptedAdapter([show("Earl Grey 3.25"), answer()])
        im = self.session(adapter, result_format=SPEAK_DISPLAY, display_label="screen")
        events = await read_all(im.admit(ask("Find teas."), delegation_id="D1").stream)
        self.assertEqual(finished_items(events)[0]["server_label"], "screen")

    async def test_show_then_the_answer_carries_the_display_and_the_speak_part(self):
        adapter = ScriptedAdapter([show("Earl Grey 3.25\nSencha 4.10"), answer("Two green teas.")])
        im = self.session(adapter, result_format=SPEAK_DISPLAY)
        events = await read_all(im.admit(ask("Find teas."), delegation_id="D1").stream)
        shown, said = finished_items(events)
        self.assertEqual((shown["type"], shown["server_label"], shown["name"]), ("mcp_call", "delegation_core", "show"))
        self.assertEqual(shown["output"], "Earl Grey 3.25\nSencha 4.10")
        self.assertEqual((said["phase"], said["content"][0]["text"]), ("final_answer", "Two green teas."))

    async def test_a_show_call_without_a_display_fails_and_a_tool_not_offered_is_unknown(self):
        gate = asyncio.Event()
        im = self.session(ScriptedAdapter([wait(gate), answer()]), result_format=SPEAK_DISPLAY, work_tools=())
        im.admit(ask("Find teas."), delegation_id="D1")
        await self.turns()
        tools = im.works["W1"].tasks[0].tools()
        with self.assertRaises(ValueError):
            await tools.execute("c1", "show", "{}")
        unknown = TEXTS.unknown_tool_output.format(name="cancel_work", offered="show")  # the tools it offers
        self.assertEqual(await tools.execute("c2", "cancel_work", "{}"), json.dumps({"error": unknown}))
        gate.set()

    def test_the_formats_are_named_in_one_table(self):
        self.assertEqual(set(RESULT_FORMATS), set(TEXTS.result_formats))
