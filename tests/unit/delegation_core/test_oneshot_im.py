# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D103

"""One-shot sessions, the IM side."""

import unittest

from delegation_core.im.session import DelegationError
from tests.unit.delegation_core.helpers import (
    ScriptedAdapter,
    answer,
    app_call,
    ask,
    finished_items,
    one_shot_session,
    progress,
    read_all,
    say,
)


class OneShotTests(unittest.IsolatedAsyncioTestCase):
    async def serve(self, *steps) -> tuple:
        adapter = ScriptedAdapter(list(steps))
        im = one_shot_session(adapter)
        im.start()
        self.addAsyncCleanup(im.close)
        events = await read_all(im.admit(ask("Two lattes."), delegation_id="D1").stream)
        return im, adapter, events

    async def test_the_rounds_messages_are_written_as_they_are_and_no_note_is_written(self):
        _, _, events = await self.serve(progress("Working."), say("Checking the menu."), app_call("get_menu"))
        messages = [(item["phase"], item["content"][0]["text"]) for item in finished_items(events) if "phase" in item]
        self.assertEqual(messages, [("commentary", "Checking the menu.")])

    async def test_it_serves_one_request_and_ends_itself(self):
        im, adapter, _ = await self.serve(answer())
        with self.assertRaisesRegex(DelegationError, "exactly one request"):
            im.admit(ask("Again."))
        with self.assertRaisesRegex(DelegationError, "exactly one request"):
            im.admit_client_request(ask("Typed."))
        await im._closing
        self.assertTrue(adapter.closed)
        self.assertEqual(adapter.conductors, 0)  # a one-shot adapter needs no conductor
