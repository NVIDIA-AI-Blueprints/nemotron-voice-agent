# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102

"""Actual response writers keep old item ownership separate from current branch bindings."""

import unittest

from nemotron_voice_agent.pipecat.examples.frontend_backend_live.delegation_core.im.stream import (
    ResponseBook,
    ResponseStream,
)
from nemotron_voice_agent.pipecat.examples.frontend_backend_live.delegation_core.ledger.records import WorkLedger
from nemotron_voice_agent.pipecat.examples.frontend_backend_live.work_provenance import (
    Attribution,
    Binding,
    Provenance,
    decode_provenance,
)
from tests.unit.pipecat.delegation_core.helpers import call_item, finished_items, read_all


class WriterProvenanceTests(unittest.IsolatedAsyncioTestCase):
    async def test_late_mcp_end_retains_the_original_publication_owner(self):
        for incomplete in (False, True):
            with self.subTest(incomplete=incomplete):
                await self.check_old_holder(incomplete)

    async def check_old_holder(self, incomplete):
        ledger = WorkLedger("conversation")
        bindings = (Binding("A", "D4", 1), Binding("B", "D3", 0))
        book = ResponseBook(
            ledger, lambda source, response, item: Provenance("conversation", response, source, 1, bindings, (), item)
        )
        old_stream, new_stream = ResponseStream(), ResponseStream()
        for source in ("D3", "D4"):
            ledger.new_delegation(source, input_items=[])
        old = book.open("D3", old_stream, "model", running=True)
        new = book.open("D4", new_stream, "model", running=True)
        original = Attribution("A", "D3", 0)
        with old.attributed(original):
            old.mcp_started(call_item("lookup", {}, "original_call"), "fixture")
        if incomplete:
            with old.attributed(Attribution("B", "D3", 0)):
                book.end("D3", "response.incomplete")
        else:
            self.assertIs(book.running("D3"), old)
            self.assertTrue(book.mcp_ended("original_call", "actual output", "done"))
            book.end("D3", "response.completed")
        with new.attributed(Attribution("A", "D4", 1)):
            new.message("New A result.", "final_answer")
        book.end("D4", "response.completed")
        old_events, new_events = await read_all(old_stream), await read_all(new_stream)
        [mcp] = [e for e in old_events if e["type"] == "response.output_item.done" and e["item"]["type"] == "mcp_call"]
        self.assertEqual(decode_provenance(mcp).item_attribution, original)
        if incomplete:
            self.assertEqual(mcp["item"]["status"], "incomplete")
        else:
            self.assertEqual(mcp["item"]["output"], "actual output")
        self.assertFalse(any(item["type"] == "mcp_call" for item in finished_items(new_events)))
        for event in [*old_events, *new_events]:
            if event["type"] in {
                "response.created",
                "response.in_progress",
                "response.completed",
                "response.incomplete",
            }:
                self.assertIsNone(decode_provenance(event).item_attribution)
            if "item" in event:
                self.assertNotIn("_nva_work", event["item"])
