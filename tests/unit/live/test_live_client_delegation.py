# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D103, D107

"""The gateway's client delegation: Live client mode announces each delegation and remembers its id."""

import re
import unittest

from live.client_delegation import ClientDelegation


class View:
    """The session's delegation view: the input clock, and the events the part sends."""

    id = "live_test"
    config = None

    def __init__(self, input_ms=0):
        self.input_ms = input_ms
        self.events: list[tuple[str, dict]] = []

    async def emit(self, kind, **fields):
        self.events.append((kind, fields))


class ClientDelegationTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.view, self.part = View(input_ms=1234), ClientDelegation()
        await self.part.open(self.view)

    async def test_a_delegation_is_minted_announced_and_remembered_before_delegate_returns(self):
        delegation_id = await self.part.delegate([{"role": "user", "content": "never sent anywhere"}])
        self.assertRegex(delegation_id, re.compile(r"^item_[0-9a-f]{32}$"))
        self.assertEqual(
            self.view.events,
            [
                (
                    "session.delegation.created",
                    {"offset_ms": 1234, "delegation": {"id": delegation_id, "type": "delegation", "target": "client"}},
                )
            ],
        )
        self.assertTrue(self.part.knows(delegation_id))
        self.assertEqual(len({await self.part.delegate([]), await self.part.delegate([])}), 2)

    async def test_the_id_is_known_before_the_announcement_goes_out(self):
        part, known = ClientDelegation(), []

        class Checking(View):
            async def emit(self, kind, **fields):
                known.append(part.knows(fields["delegation"]["id"]))  # an answer to it may arrive right away

        await part.open(Checking())
        await part.delegate([])
        self.assertEqual(known, [True])

    async def test_from_close_step_2_nothing_is_announced_and_after_close_delegate_raises(self):
        earlier = await self.part.delegate([])
        self.part.stop_handing()
        later = await self.part.delegate([])
        self.assertEqual([fields["delegation"]["id"] for _, fields in self.view.events], [earlier])
        self.assertFalse(self.part.knows(later))  # never announced, so no append may name it
        await self.part.close()
        await self.part.close()  # safe twice
        self.assertFalse(self.part.knows(earlier))
        with self.assertRaises(RuntimeError):
            await self.part.delegate([])

    async def test_open_and_close_are_safe_in_any_order(self):
        part = ClientDelegation()
        await part.close()
        await part.open(View())
        await part.open(View())
        with self.assertRaises(RuntimeError):
            await part.delegate([])
