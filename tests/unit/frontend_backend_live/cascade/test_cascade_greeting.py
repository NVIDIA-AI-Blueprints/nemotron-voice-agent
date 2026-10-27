# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D103

"""The greeting through the inbox: a pinned developer message plus a commentary append, worded and said."""

from examples.frontend_backend_live.models.decision import TalkerDecision
from tests.unit.frontend_backend_live.cascade.doubles import Call

INTRO = "Greet the caller now in English as Remy from Bluebird Cafe and ask what they would like. Then pause."


class GreetingTests(Call):
    async def test_greet_pins_the_instructions_and_says_the_worded_greeting_with_no_caller_turn(self):
        await self.call([TalkerDecision("speak", "Hi, I'm Remy at Bluebird Cafe. What can I get you?")])
        self.session.greet(INTRO)
        await self.said(1)
        self.assertEqual(self.spoken, ["Hi, I'm Remy at Bluebird Cafe. What can I get you?"])
        self.assertIn(INTRO, self.frontend.calls[0]["history"][-1]["content"])  # worded as the update
        self.assertEqual(self.session.turns.waiting, [])
        self.assertEqual([entry.kind.value for entry in self.session.heard_log.entries], ["Agent"])
        self.assertEqual(self.memory()[1], ("developer", INTRO))
        self.assertNotIn(("user", INTRO), self.memory())

    async def test_a_live_clients_documented_greeting_takes_the_same_path(self):
        await self.call(
            [TalkerDecision("speak", "Hello! How can I help?")], seed=[{"role": "developer", "content": INTRO}]
        )
        await self.session.on_commentary(INTRO)
        await self.said(1)
        self.assertEqual(self.spoken, ["Hello! How can I help?"])
        self.assertIn(INTRO, self.frontend.calls[0]["instructions"])  # the seed's developer item is pinned
        self.assertEqual(self.session.turns.waiting, [])

    async def test_the_greeting_waits_while_the_caller_speaks(self):
        await self.call([TalkerDecision("speak", "Hi there.")])
        self.session.caller_started()
        self.session.greet(INTRO)
        self.assertEqual(self.spoken, [])
        self.session.caller_stopped()
        await self.said(1)
