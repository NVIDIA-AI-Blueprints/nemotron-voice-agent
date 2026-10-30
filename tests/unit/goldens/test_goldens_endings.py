# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D100, D101, D102, D103, D107

from dataclasses import replace
from unittest.mock import patch

from examples.frontend_backend_live.cascade.lines import catalog_lines
from tests.unit.core_helpers import _timeout
from tests.unit.goldens import harness
from tests.unit.goldens.harness import (
    GOLDEN_LIVE_CONFIG,
    WORK_RUNNING,
    GoldenCase,
    HeldRound,
    HoldingChat,
    answer_round,
    decision,
    result_item,
    tool_call_round,
)

# The waits for the app, cut so that G16's run out quickly, in the shipped order: the core's client-tool deadline
# (core.yaml delegation.client_tool_timeout_seconds) before the voice's app wait (frontend.cascade.delegation
# .app_wait_seconds). Nothing races them: each round is held until its lead-in is in the talker's context, and the
# client stays silent until the app-wait line was said.
SHORT_CLIENT_TOOL_WAIT = {"client_tool_timeout_seconds": 0.1}
SHORT_APP_WAIT = replace(GOLDEN_LIVE_CONFIG, delegation=replace(GOLDEN_LIVE_CONFIG.delegation, app_wait_seconds=0.3))
TEAS = '{"teas": ["Earl Grey", "Matcha"]}'
COFFEES = '{"coffees": ["espresso", "cappuccino"]}'
HOURS = "We are open from seven in the morning until five."
# How often a round that always times out is tried: the provider's own retries (backend.llm.provider_retries, 2) for
# each of the run's two attempts.
BACKEND_TRIES = 6


# -------------------------------------------------------------------------------------------------- goldens
class EndingGoldens(GoldenCase):
    """How delegated work ends: closed while it runs, cut off by the client-tool wait, or failed behind newer work."""

    async def serve(self, session=None, *, frontend=(), backend=(), live_config=None, delegation=None):
        # The harness builds a ScriptedChat for each role; HoldingChat behaves the same except for HeldRound steps.
        with patch.object(harness, "ScriptedChat", HoldingChat):
            return await super().serve(
                session, frontend=frontend, backend=backend, live_config=live_config, delegation=delegation
            )

    def engine_pending(self):
        """The harness's reasons, except that a held round counts as waiting for the test, like a client-tool wait.

        While a held round waits, the backend does nothing else: the next step is the test's. Results are
        worded and spoken in the presenter's own task, and the harness reports them under a reason of their own,
        which this filter keeps, so a result still being spoken while a round is held is waited for.
        """
        reasons = super().engine_pending()
        if self.backend.holding:
            reasons = [reason for reason in reasons if reason != WORK_RUNNING]
        return reasons

    async def release_when_settled(self, held, what):
        """Wait until ``held`` waits for its model and the rest of the call is quiet, then let the model answer."""
        await self.wait_until(lambda: held.waiting.is_set(), what)
        await self.wait_quiet()
        held.release()

    async def test_g13_close_while_work_runs(self):
        """G13: the client closes the session while a delegation's backend round runs.

        The talker delegates with a lead-in. The backend round is sent and its opening events are relayed, but the
        model has not answered when the client closes the session. The close stops the round: nothing more of the
        delegation is relayed, and session.closed comes once and last.
        """
        held = HeldRound()
        await self.serve(frontend=[decision("delegate", "Let me check.")], backend=[held])
        await self.say("What teas do you have?")
        await self.wait_until(lambda: held.waiting.is_set(), "the backend round to wait for its model")
        await self.wait_quiet()
        await self.close()
        # The IM session's close stopped the round: the model call was cancelled and its stream closed. Both model
        # clients were released.
        self.assertEqual((held.cancelled, held.closed), (True, True))
        self.assertEqual((self.frontend.closed, self.backend.closed), (True, True))
        self.assert_golden("G13")

    async def test_g16_the_client_tool_wait_runs_out_then_a_late_output_and_response_create_arrive(self):
        """G16: the client-tool wait runs out, then a late output and response.create arrive.

        The core's client-tool deadline is cut to 0.1 s and the voice's app wait to 0.3 s. The first request's backend
        round calls get_menu; the client stays silent. At the deadline the core ends the Work and writes nothing: no
        response is open (the endings fixture's deadline_while_waiting). When the app wait runs out the voice says its
        app-wait line once, and the delegation stays watched. A response.create without the output would
        continue nothing while the output is owed, so it is refused (function_call_outputs_required). The late
        output and its response.create then get the late continuation's empty response.completed, which the voice
        reads as too late: it says its unclear line. The second request's call runs out the same way, and a third
        request is answered. The late output for the second call and its response.create arrive only then: the call's
        route is still open, so the output is accepted, and its late continuation ends the second delegation the same
        way.
        """
        first = HeldRound(tool_call_round("call_1"))
        second = HeldRound(tool_call_round("call_2", '{"category":"coffee"}'))
        third = HeldRound(answer_round("We are open from 7 to 5."))
        await self.serve(
            frontend=[
                decision("delegate", "Let me check."),
                decision("delegate", "One moment."),
                decision("delegate", "Let me look that up."),
                decision("speak", HOURS),
            ],
            backend=[first, second, third],
            live_config=SHORT_APP_WAIT,
            delegation=SHORT_CLIENT_TOOL_WAIT,
        )
        lines = catalog_lines()
        await self.say("What teas do you have?")
        await self.release_when_settled(first, "the first round to wait for its model")
        call_1 = await self.wait_for_call("get_menu")
        await self.wait_until(lambda: lines.app_wait_line in self.tts.utterances, "the app-wait line for the first")
        await self.wait_quiet()
        self.send({"type": "response.create", "event_id": "continue_without_output"})
        await self.wait_quiet()
        self.send({**result_item(call_1["call_id"], TEAS), "event_id": "late_output_1"})
        self.send({"type": "response.create", "event_id": "late_continue_1"})
        await self.wait_until(lambda: lines.unclear in self.tts.utterances, "the unclear line for the first")
        await self.wait_quiet()

        after = len(self.ws.from_server)
        await self.say("Do you have coffee?")
        await self.release_when_settled(second, "the second round to wait for its model")
        call_2 = await self.wait_for_call("get_menu", after=after)
        await self.wait_until(
            lambda: self.tts.utterances.count(lines.app_wait_line) == 2, "the app-wait line for the second"
        )
        await self.wait_quiet()
        await self.say("Then just tell me your opening hours.")
        await self.release_when_settled(third, "the third round to wait for its model")
        await self.wait_quiet()
        # Late, after a newer request was answered: the call's route is still open, so the output is accepted.
        self.send({**result_item(call_2["call_id"], COFFEES), "event_id": "late_output_2"})
        self.send({"type": "response.create", "event_id": "late_continue_2"})
        await self.wait_until(lambda: self.tts.utterances.count(lines.unclear) == 2, "the unclear line for the second")
        await self.wait_quiet()
        await self.close()
        self.assert_golden("G16")

    async def test_x2_a_backend_failure_after_a_newer_request_joined_the_work(self):
        """X2: a backend failure after a newer request joined the work.

        The first request's backend round calls get_menu. While the client runs it, the caller asks something else;
        the steer conductor joins the newer request to the open Work. The client returns the result and
        continues; the IM names the newer request as the work's taker and the older response ends with the
        handover line. The Work's next round fails, through the provider's retries and the run's one more attempt:
        the taker's response ends response.failed. A tool already ran for the work, so the voice says its
        unclear line, not the apology (the endings' rule).
        """
        await self.serve(
            frontend=[
                decision("delegate", "Let me check."),
                decision("delegate", "I'll check the coffee too."),
            ],
            backend=[tool_call_round(), *[_timeout()] * BACKEND_TRIES],
        )
        await self.say("What teas do you have?")
        call = await self.wait_for_call("get_menu")
        await self.wait_quiet()
        await self.say("And do you have coffee?")
        await self.wait_quiet()
        self.send(result_item(call["call_id"], TEAS))
        self.send({"type": "response.create", "event_id": "continue"})
        await self.wait_until(lambda: catalog_lines().unclear in self.tts.utterances, "the unclear line")
        await self.wait_quiet()
        await self.close()
        self.assert_golden("X2")
