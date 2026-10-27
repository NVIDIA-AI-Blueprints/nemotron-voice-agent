# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D100, D101, D102, D103, D107


import httpx
from openai import InternalServerError

from tests.unit.goldens.harness import GoldenCase, answer_round, chunk, decision, result_item, tool_call_round

# How often a round that always fails with a transient error is tried: the provider opens it once and retries it
# backend.llm.provider_retries (2) times, and the run tries the round once more, with the same retries.
BACKEND_TRIES = 6


def server_error():
    """The backend endpoint refuses the round with HTTP 500 before any output (the scripted client raises it)."""
    request = httpx.Request("POST", "http://localhost:1/v1/chat/completions")
    return InternalServerError("Internal Server Error", response=httpx.Response(500, request=request), body=None)


class DelegationGoldens(GoldenCase):
    """Delegations that do not go straight through: a round cut short, a failed one, and a second request."""

    async def test_g3_a_round_cut_by_the_output_limit(self):
        """G3: a round cut by the output limit, with partial text.

        The talker delegates with a lead-in. The backend's round streams part of its answer, then stops at the token
        limit. The cut text is no item and the round is not tried again: the delegation's response ends
        response.incomplete with the official reason max_output_tokens, and the voice says its own apology, which no
        model words.
        """
        await self.serve(
            frontend=[decision("delegate", "Let me look that up.")],
            backend=[
                # Cut off by the token limit after two text chunks.
                [chunk("Our caffeine-free teas"), chunk(" are rooibos,"), chunk(finish="length")],
            ],
        )
        await self.say("Which of your teas are caffeine free?")
        # Both attempts, the writer and the answer all follow from the one caller line.
        await self.wait_quiet()
        await self.close()
        self.assert_golden("G3")

    async def test_g4_a_backend_failure_with_the_apology(self):
        """G4: a backend failure with the apology.

        The talker delegates with a lead-in. Every try of the backend round fails with HTTP 500 before any output: the
        provider's own retries when it opens the round (backend.llm.provider_retries), and the run's one more attempt
        of the round. The delegation's response ends response.failed with the official code server_error, and the
        voice says its own apology, which no model words.
        """
        await self.serve(
            frontend=[decision("delegate", "Let me check.")],
            backend=[server_error()] * BACKEND_TRIES,
        )
        await self.say("Is the lemon cake gluten free?")
        await self.wait_quiet()
        await self.close()
        self.assert_golden("G4")

    async def test_g5_a_second_request_after_the_first_acted(self):
        """G5: a second request after the first acted.

        The first delegation calls the client's get_menu tool. While it waits for the output, the caller narrows the
        request; the talker sees the open work in its prompt and delegates again with a lead-in. The steer conductor
        joins the second request to the open Work: its words wait as steering, and the IM names the newer
        request as the work's taker. The client returns the output; the Work goes on with the output and the
        steering in one more round, which answers the narrowed request. The writer words that answer, with both
        lead-ins as already told.
        """
        await self.serve(
            frontend=[
                decision("delegate", "Let me check."),  # the talker, first line
                decision("delegate", "Okay, I'll narrow that down."),  # the talker, second line
                decision("speak", "Our green teas are Matcha and Sencha."),  # the answer
            ],
            backend=[
                tool_call_round("call_1", '{"category":"tea"}'),
                answer_round("Our green teas are Matcha and Sencha."),
            ],
        )
        await self.say("What teas do you have?")
        first = await self.wait_for_call("get_menu")
        # The first lead-in is spoken and written into the talker's context; the work waits for the client.
        await self.wait_quiet()
        await self.say("Actually, only green tea.")
        # The second lead-in is spoken; the second request waits as steering of the open work.
        await self.wait_quiet()
        self.send(result_item(first["call_id"], '{"teas": ["Earl Grey", "Matcha", "Sencha"]}'))
        self.send({"type": "response.create", "event_id": "continue_1"})
        await self.wait_quiet()
        await self.close()
        self.assert_golden("G5")
