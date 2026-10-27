# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D100, D101, D102, D103, D107

"""Goldens for joined work: a request that steers into a running round (G6, X8, X1) or a waiting Work (G7).

The core's conductor is ``steer`` (``core.yaml``): each request joins the newest open Work as steering. A
round that is still streaming is restarted with the steering (the commit gate refuses its calls); a Work that waits
for the app takes the steering at its next round. When the steering lands, the IM names the newest request as the
taker of the work in its ``response.in_progress``, and each older delegation's response ends with the handover
line; the frontend passes the older requests' words, clock, lead-ins and quiet-spell count to the taker
(``cascade/delegations.py``, ``cascade/worknote.py``). The goldens need backend work held at a chosen point, so this
file adds one double to the harness: :class:`Pause`, a point in a scripted model reply where the
stream waits until the test releases it, as a slow model would.
"""

import asyncio
from dataclasses import replace
from unittest.mock import patch

from delegation_core.backend.providers.translation import thinking_enabled
from examples.frontend_backend_live.cascade.lines import catalog_lines
from tests.unit.core_helpers import FakeStream
from tests.unit.goldens.harness import (
    ENDPOINT,
    GOLDEN_LIVE_CONFIG,
    RESULTS_PENDING,
    WORK_RUNNING,
    GoldenCase,
    ScriptedChat,
    answer_round,
    chunk,
    decision,
    result_item,
    tool_call_round,
)

# X1 tests the quiet spell's line, so it sets the timer small (the harness keeps it at an hour). A line comes after
# half a second without speech while work is open; each delegation gets at most one (the default), and joined work
# shares the count. The limit is what keeps the call one way only: after the first line, no silence says another.
STATUS_TIMER_CONFIG = replace(
    GOLDEN_LIVE_CONFIG,
    delegation=replace(GOLDEN_LIVE_CONFIG.delegation, progress_after_seconds=0.5, progress_max_per_delegation=1),
)
# X1's quiet spell is half a second, and its timer looks every eighth of a second while the floor is idle: eight looks
# while the joined round is held span two quiet spells, in which a line the shared count allowed would come.
TWO_QUIET_SPELLS_OF_LOOKS = 8


class Pause:
    """A point in a scripted model reply where the stream stops until the test calls :meth:`release`.

    Put it in a step's chunk list: the stream yields the chunks before it, then waits. A request cancelled while it
    waits (a fold drops the running round) never resumes. A held reply still runs under its timeouts (10 s for a
    backend round and for a writer attempt in the engine tests' config), so release it well within that.
    """

    def __init__(self, name, on_change):
        self.name, self.on_change = name, on_change
        self.reached, self.released = asyncio.Event(), asyncio.Event()
        self.cancelled = False
        self.role = None  # the model client whose reply it holds: "frontend" (the talker or the writer) or "backend"

    @property
    def holding(self):
        """Whether a reply waits here now."""
        return self.reached.is_set() and not self.released.is_set() and not self.cancelled

    def release(self):
        self.released.set()

    async def wait(self):
        self.reached.set()
        self.on_change()
        try:
            await self.released.wait()
        except asyncio.CancelledError:
            self.cancelled = True
            self.on_change()
            raise


class PausingStream(FakeStream):
    """``FakeStream`` that stops at each :class:`Pause` in its chunk list."""

    async def _iterate(self):
        for item in self.chunks:
            if isinstance(item, Pause):
                await item.wait()
            else:
                yield item


class JoinGoldens(GoldenCase):
    """A request that joins delegated work: folded into a running round, or merged into a waiting delegation."""

    def setUp(self):
        super().setUp()
        self.pauses: list[Pause] = []

    def pause(self, name):
        """A new :class:`Pause` for a script step; ``wait_quiet`` treats the work it holds as waiting for the test."""
        pause = Pause(name, self._changed)
        self.pauses.append(pause)
        return pause

    def instrument(self, patches):
        """The harness's wrappers, plus scripted replies that stop at a :class:`Pause`."""
        super().instrument(patches)
        create = ScriptedChat.create

        async def create_with_pauses(chat, **kwargs):
            stream = await create(chat, **kwargs)
            pauses = [item for item in stream.chunks if isinstance(item, Pause)]
            for pause in pauses:
                pause.role = chat.role
            if pauses:
                return PausingStream(stream.chunks)
            return stream

        patches.enter_context(patch.object(ScriptedChat, "create", create_with_pauses))

    def engine_pending(self):
        """The harness's reasons, except busy work while a reply is held at a pause.

        A held reply waits for the test, as a client-tool round waits for the client, so the call can go quiet. Since
        The presenter words and speaks results in its own task: a writer's reply held at a pause (X1) holds the
        result it words, so its reason goes too, while a held backend reply leaves the results to be spoken.
        """
        reasons = super().engine_pending()
        holding = [pause for pause in self.pauses if pause.holding]
        if holding:
            reasons = [reason for reason in reasons if reason != WORK_RUNNING]
        if any(pause.role == "frontend" for pause in holding):
            reasons = [reason for reason in reasons if reason != RESULTS_PENDING]
        return reasons

    async def wait_held(self, pause):
        """Wait until a model reply waits at ``pause``."""
        await self.wait_until(lambda: pause.holding, f"a model reply held at the {pause.name} pause")

    async def wait_spoken(self, text):
        """Wait until the text-to-speech has received ``text``."""
        await self.wait_until(lambda: text in self.tts.utterances, f"{text!r} spoken")

    # ------------------------------------------------------------------------------------ into a running round
    async def test_g6_a_request_steers_into_a_running_round_with_a_lead_in_on_each(self):
        """G6: a request steers into a running round, with a lead-in on each request.

        The caller asks for a booking; the talker delegates with a lead-in, and the backend round is held before the
        model's first chunk. The caller corrects the request; the talker delegates again with a second lead-in. The
        steer conductor joins it to the running Work: the held round is restarted with both requests, the IM names the
        newer request as the taker (previous_response_id), and the first delegation's response ends with the handover
        line. While the restarted round is held, the caller asks a side question that the talker answers itself; its
        work note shows both requests, joined as "first / correction". Then the backend answers in the taker's
        response, and the writer speaks the answer, told of both lead-ins, the first's first.
        """
        first, restarted = self.pause("first round"), self.pause("restarted round")
        await self.serve(
            frontend=[
                decision("delegate", "Let me check Friday for you."),
                decision("delegate", "Sure, I'll look at Saturday instead."),
                decision("speak", "Yes, there is free parking behind the restaurant."),
                decision("speak", "You have a table for two on Saturday at seven."),
            ],
            backend=[
                # Never released: the fold drops this round.
                [first, *answer_round("Friday at seven is free.")],
                [restarted, *answer_round("Booked a table for two on Saturday at seven.")],
            ],
        )
        await self.say("Book me a table for two on Friday.")
        await self.wait_held(first)
        await self.wait_quiet()
        await self.say("Actually, make it Saturday.")
        # The first round is dropped and the restarted one starts: wait until it is held too.
        await self.wait_held(restarted)
        await self.wait_quiet()
        await self.say("Is there parking nearby?")
        await self.wait_quiet()
        restarted.release()
        await self.wait_quiet()
        await self.close()
        self.assertTrue(first.cancelled)
        self.assert_golden("G6")

    async def test_x8_a_request_steers_into_a_round_that_streamed_partial_text(self):
        """X8: a request steers into a round that streamed partial text; thinking is off.

        The caller asks about a table; the talker delegates with a lead-in. The backend round streams the start of
        its answer and is held before it finishes; the backend LLM holds a round's text to its end, so nothing of it
        reaches the client. The caller corrects the request and the talker delegates again without a lead-in, so the
        voice says its own. The round is restarted with both requests, the newer one is named the taker, and
        the partial text is never spoken: the caller hears the lead-ins, then the restarted round's answer, which the
        writer words knowing both were said.
        """
        # Thinking off: the chat-completions translator streams text as it arrives instead of holding it to the end.
        self.assertFalse(thinking_enabled(ENDPOINT.extra_params))
        held = self.pause("partial text")
        await self.serve(
            frontend=[
                decision("delegate", "Let me check Friday."),
                decision("delegate", ""),
                decision("speak", "Saturday at eight is free for two."),
            ],
            backend=[
                [chunk("Friday at seven"), held, chunk(" is free.", finish="stop")],
                answer_round("Saturday at eight is free for two people."),
            ],
        )
        await self.say("Is there a table for two on Friday?")
        await self.wait_held(held)
        await self.wait_quiet()
        await self.say("Sorry, I meant Saturday.")
        await self.wait_until(lambda: held.cancelled, "the round with the partial text dropped")
        await self.wait_quiet()
        await self.close()
        self.assert_golden("X8")

    # ------------------------------------------------------------------------------- into a Work waiting for the app
    async def test_g7_two_requests_join_a_work_that_waits_for_the_app(self):
        """G7: two requests join a Work that waits for the app, with a lead-in on each request.

        The talker delegates the first request with a lead-in, and the backend calls the client's get_menu tool, so
        the Work waits for the app. A second request and a third, each with its own lead-in, join the Work as steering
        (the steer conductor). The client returns the tool result and continues: the Work's next round takes
        the output and both requests (held until the call settles), the IM names the newest request as the taker, and
        the two older delegations' responses end with the handover line. The one answer is worded for the taker, told
        of all three lead-ins, oldest first.
        """
        joined = self.pause("joined round")
        await self.serve(
            frontend=[
                decision("delegate", "Let me check the teas."),
                decision("delegate", "I'll check the oat milk too."),
                decision("delegate", "And the soy milk."),
                decision("speak", "We have Earl Grey and Matcha, and both oat milk and soy milk."),
            ],
            backend=[
                tool_call_round(),
                [joined, *answer_round("Earl Grey and Matcha; both oat milk and soy milk are available.")],
            ],
        )
        await self.say("What teas do you have?")
        call = await self.wait_for_call("get_menu")
        await self.wait_quiet()
        await self.say("Do you have oat milk?")
        await self.wait_quiet()
        await self.say("And soy milk?")
        await self.wait_quiet()
        self.send(result_item(call["call_id"], '{"teas": ["Earl Grey", "Matcha"]}'))
        self.send({"type": "response.create", "event_id": "continue"})
        await self.wait_held(joined)
        await self.wait_quiet()
        joined.release()
        await self.wait_quiet()
        await self.close()
        self.assert_golden("G7")

    async def test_x1_a_quiet_spells_line_then_a_request_joins_and_the_taker_shares_its_count(self):
        """X1: a quiet spell's line, then a request joins and the taker shares its count.

        The quiet-spell timer is set to half a second, one line per delegation, which joined work shares. The first
        request's backend round is held, and the quiet spell says the voice's still-working line, the first
        delegation's one line (the backend wrote no progress note to say instead). A second request steers into
        the running round, which restarts with both; the IM names the second request the taker, which carries the
        first one's count. The restarted round is held for a second, longer than the quiet spell, and no quiet spell
        says a second line. Then the answer is spoken, worded knowing the lead-in and the line already said.
        """
        first, joined = self.pause("first round"), self.pause("joined round")
        await self.serve(
            frontend=[
                decision("delegate", "Let me look at the pastries."),
                decision("delegate", "I'll check the croissant too."),
                decision("speak", "We have a croissant and a blueberry muffin, and the croissant is vegan."),
            ],
            backend=[
                # Never released: the steering restarts this round.
                [first, *answer_round("We have a croissant and a blueberry muffin.")],
                [joined, *answer_round("A croissant and a blueberry muffin; the croissant is vegan.")],
            ],
            live_config=STATUS_TIMER_CONFIG,
        )
        still_working = catalog_lines().still_working
        await self.say("What pastries do you have?")
        await self.wait_held(first)
        # The first request's one line; after it, the shared count keeps every later quiet spell silent.
        await self.wait_spoken(still_working)
        await self.wait_quiet()
        await self.say("Is the croissant vegan?")
        await self.wait_held(joined)
        await self.wait_quiet()
        # Two quiet spells pass while the joined round is held, counted in the timer's looks: the taker's shared count
        # leaves it no delegation to speak of.
        self.assertFalse(await self.wait_quiet_spells(TWO_QUIET_SPELLS_OF_LOOKS))
        joined.release()
        await self.wait_quiet()
        await self.close()
        self.assertTrue(first.cancelled)
        self.assertEqual(self.tts.utterances.count(still_working), 1)
        self.assert_golden("X1")
