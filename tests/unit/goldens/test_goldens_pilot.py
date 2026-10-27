# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D100, D101, D102, D103, D107

from tests.unit.goldens.harness import GoldenCase, answer_round, decision, result_item, tool_call_round


class PilotGoldens(GoldenCase):
    """Goldens for a direct reply and for a delegation with one client-tool round."""

    async def test_g1_a_direct_reply(self):
        """G1: the caller says hello and the talker answers at once; nothing is delegated."""
        await self.serve(frontend=[decision("speak", "Hello there.")])
        await self.say("Hi.")
        await self.wait_quiet()
        await self.close()
        self.assert_golden("G1")

    async def test_g2_a_delegation_with_one_client_tool_round(self):
        """G2: a delegation with one client-tool round.

        The talker delegates with a lead-in; the backend calls the client's get_menu tool; the client returns the
        result and continues; the writer speaks the backend's answer.
        """
        await self.serve(
            frontend=[decision("delegate", "Let me check."), decision("speak", "We have Earl Grey and Matcha.")],
            backend=[tool_call_round(), answer_round("We have Earl Grey and Matcha.")],
        )
        await self.say("What teas do you have?")
        call = await self.wait_for_call("get_menu")
        # The client answers once the server waits for it: the lead-in is spoken and the first round has ended.
        await self.wait_quiet()
        self.send(result_item(call["call_id"], '{"teas": ["Earl Grey", "Matcha"]}'))
        self.send({"type": "response.create", "event_id": "continue"})
        await self.wait_quiet()
        await self.close()
        self.assert_golden("G2")
