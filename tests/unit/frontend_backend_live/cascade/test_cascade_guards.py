# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D103

"""The talker's guards in a call: the turn router, enforced or in shadow, and the repeat guard."""

from dataclasses import replace

from examples.frontend_backend_live.models.decision import TalkerDecision
from tests.unit.frontend_backend_live.cascade.doubles import CONFIG, Call

LATTE = ({"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Your latte is ready."})
STALE = "I am still unable to locate your account with the information provided."


def guarded(**guards):
    return replace(CONFIG, guards=replace(CONFIG.guards, **guards))


class RouterModeTests(Call):
    async def test_the_enforced_router_drops_a_backchannel_before_the_talker(self):
        await self.call([], seed=LATTE, config=guarded(turn_router=True))
        await self.turn("Okay.")
        self.assertEqual((self.frontend.calls, self.spoken, self.port.requests), ([], [], []))
        self.assertIn("router.dropped", self.kinds())
        self.assertEqual(self.session.turns.waiting, [])  # no later decision covers it again

    async def test_the_shadow_router_only_traces_and_the_talker_still_decides(self):
        await self.call(
            [TalkerDecision("speak", "Anything else?")], seed=LATTE, config=guarded(turn_router_shadow=True)
        )
        await self.turn("Okay.")
        self.assertEqual(self.spoken, ["Anything else?"])
        self.assertIn("router.would_drop", self.kinds())
        self.assertNotIn("router.dropped", self.kinds())

    async def test_the_router_is_off_by_default(self):
        await self.call([TalkerDecision("speak", "Anything else?")], seed=LATTE)
        await self.turn("Okay.")
        self.assertEqual(self.spoken, ["Anything else?"])
        self.assertNotIn("router.would_drop", self.kinds())


class RepeatGuardTests(Call):
    SEED = ({"role": "user", "content": "Check my account."}, {"role": "assistant", "content": STALE})

    async def test_with_the_guard_on_a_reply_that_repeats_the_last_one_becomes_a_delegation(self):
        await self.call([TalkerDecision("speak", STALE)], seed=self.SEED, config=guarded(repeat_guard=True))
        await self.turn("It is 4472.")
        self.assertIn("talker.repeat_replaced", self.kinds())
        self.assertNotIn(STALE, self.spoken)
        self.assertEqual(len(self.port.requests), 1)

    async def test_the_guard_is_off_by_default(self):
        await self.call([TalkerDecision("speak", STALE)], seed=self.SEED)
        await self.turn("It is 4472.")
        self.assertEqual((self.spoken, self.port.requests), ([STALE], []))
