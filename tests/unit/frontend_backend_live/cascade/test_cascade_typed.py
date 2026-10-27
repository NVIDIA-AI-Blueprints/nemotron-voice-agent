# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D103

"""A typed question enters the talker's memory."""

from examples.frontend_backend_live.models.decision import TalkerDecision
from tests.unit.frontend_backend_live.cascade.doubles import Call, answered_events


class TypedTests(Call):
    async def test_a_typed_question_then_its_answer_are_both_in_the_memory(self):
        await self.call([TalkerDecision("speak", "Yes, there is parking behind the cafe.")])
        self.session.on_typed("Is there parking?")
        self.emit("typed_1", *answered_events("r1", "Parking behind the cafe."))
        await self.said(1)
        wording = self.frontend.calls[0]["history"]
        self.assertIn({"role": "user", "content": "Application context (reference data): Is there parking?"}, wording)
        self.assertEqual(
            self.memory()[-3:],
            [
                ("user", "Application context (reference data): Is there parking?"),
                ("user", "Backend result (reference data): Parking behind the cafe."),
                ("assistant", "Yes, there is parking behind the cafe."),
            ],
        )
        self.assertEqual([entry.kind.value for entry in self.session.heard_log.entries], ["Agent"])
