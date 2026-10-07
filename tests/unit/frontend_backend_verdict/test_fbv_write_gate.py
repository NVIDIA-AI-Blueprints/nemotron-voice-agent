# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D103

"""R4: the write gate holds consequential calls until the caller confirms the exact call.

The tools are tau2's airline tools (fixtures), but nothing in the gate is airline-specific:
which tools are held comes from the generic read/write classifier and configuration.
"""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from typing import Any

from _fbv_voice_fakes import (
    FakeChatClient,
    SessionHarness,
    delegate_response,
    normalize_trace,
    pcmu_silence,
    pcmu_speech,
    tau2_session_update,
    text_response,
    tool_response,
    voice_config,
)
from test_fbv_tool_resume import NO_OVERLAP_TRACE_SHA256

from examples.frontend_backend_verdict.text.delegation import FORCE_CALL_BACKEND, FRONTEND_TOOLS_CONFIRMATION
from examples.frontend_backend_verdict.text.messages import ToolCall, canonical_json
from examples.frontend_backend_verdict.text.tools import ToolSpec
from examples.frontend_backend_verdict.voice.agent.runner import AgentClients, TextAgentRunner
from examples.frontend_backend_verdict.voice.agent.sinks import EventLog, SessionRoutingSink
from examples.frontend_backend_verdict.voice.agent.tools import realtime_tools_to_specs
from examples.frontend_backend_verdict.voice.agent.write_gate import (
    CONFIRM_QUESTION,
    GateState,
    ToolGateSettings,
    WriteGate,
    WriteGateSettings,
    resolve_gate,
    summarize,
)
from examples.frontend_backend_verdict.voice.config import prompt_context
from examples.frontend_backend_verdict.voice.errors import VoiceConfigError
from examples.frontend_backend_verdict.voice.normalization import NormalizationSettings
from examples.frontend_backend_verdict.voice.normalization.arguments import ArgumentRule, ToolArgumentSettings

AIRLINE_TOOLS = tau2_session_update("airline")["session"]["tools"]
SPECS = {spec.name: spec for spec in realtime_tools_to_specs(AIRLINE_TOOLS)}
WRITES = {
    "book_reservation",
    "cancel_reservation",
    "send_certificate",
    "update_reservation_baggages",
    "update_reservation_flights",
    "update_reservation_passengers",
}
GATE = WriteGateSettings(enabled=True, exempt=("transfer_to_human_agents",))
BAGS = {"reservation_id": "ZFA04Y", "total_baggages": 3, "nonfree_baggages": 1, "payment_id": "credit_card_7815826"}
BAGS_SUMMARY = (
    "Update reservation baggages, with reservation id: ZFA04Y; total baggages: 3; nonfree baggages: 1; "
    "payment id: credit_card_7815826."
)
BOOKING = {
    "user_id": "anya_garcia_5412",
    "origin": "JFK",
    "destination": "SEA",
    "flight_type": "round_trip",
    "cabin": "economy",
    "flights": [{"flight_number": "HAT083", "date": "2024-05-20"}, {"date": "2024-05-27", "flight_number": "HAT112"}],
    "passengers": [{"first_name": "Anya", "last_name": "Garcia", "dob": "1990-04-02"}],
    "payment_methods": [{"payment_id": "gift_card_7815826", "amount": 410}],
    "total_baggages": 2,
    "nonfree_baggages": 0,
    "insurance": "no",
}


def bags_call(call_id: str, **changes: Any) -> Any:
    return tool_response(("update_reservation_baggages", {**BAGS, **changes}), ids=[call_id])


def confirm_response(verdict: str | None, query: str = "Add the bags to reservation ZFA04Y.") -> Any:
    arguments: dict[str, Any] = {"query": query, "filler_text": "One moment."}
    if verdict is not None:
        arguments["confirmation"] = verdict
    return tool_response(("call_backend", arguments), ids=["fcall_confirm"])


def call(call_id: str, name: str, arguments: dict[str, Any]) -> ToolCall:
    return ToolCall(id=call_id, name=name, arguments_json=canonical_json(arguments))


class _LogCase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.log_path = Path(self._tmp.name) / "events.jsonl"

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def records(self, kind: str) -> list[dict[str, Any]]:
        if not self.log_path.exists():
            return []
        return [r for r in map(json.loads, self.log_path.read_text(encoding="utf-8").splitlines()) if r["kind"] == kind]


# -- the gate itself -------------------------------------------------------------------------


class SelectionTests(unittest.TestCase):
    def test_non_read_selects_the_six_writes_and_the_handoff(self) -> None:
        resolved = resolve_gate(WriteGateSettings(enabled=True), list(SPECS.values()))
        self.assertEqual(resolved.gated, frozenset(WRITES | {"transfer_to_human_agents"}))

    def test_exempt_and_include(self) -> None:
        settings = WriteGateSettings(enabled=True, exempt=("transfer_to_human_agents",), include=("get_user_details",))
        resolved = resolve_gate(settings, list(SPECS.values()))
        self.assertEqual(resolved.gated, frozenset(WRITES | {"get_user_details"}))

    def test_none_default_and_disabled(self) -> None:
        self.assertEqual(
            resolve_gate(WriteGateSettings(enabled=True, default="none"), list(SPECS.values())).gated, set()
        )
        self.assertIsNone(resolve_gate(WriteGateSettings(enabled=False), list(SPECS.values())))

    def test_unknown_tool_names_and_paths_are_errors(self) -> None:
        bad = (
            WriteGateSettings(enabled=True, exempt=("no_such_tool",)),
            WriteGateSettings(enabled=True, include=("no_such_tool",)),
            WriteGateSettings(enabled=True, tools={"no_such_tool": ToolGateSettings()}),
            WriteGateSettings(
                enabled=True, tools={"book_reservation": ToolGateSettings(not_consequential=("passengers[].age",))}
            ),
        )
        for settings in bad:
            with self.subTest(settings=settings), self.assertRaises(VoiceConfigError):
                resolve_gate(settings, list(SPECS.values()))
        nested = WriteGateSettings(
            enabled=True, tools={"book_reservation": ToolGateSettings(not_consequential=("passengers[].dob",))}
        )
        self.assertIsNotNone(resolve_gate(nested, list(SPECS.values())))  # through anyOf/$ref

    def test_a_tool_the_session_did_not_offer_is_classified_by_name(self) -> None:
        resolved = resolve_gate(GATE, list(SPECS.values()))
        self.assertTrue(resolved.is_gated("delete_everything"))
        self.assertFalse(resolved.is_gated("get_weather"))


class SummaryTests(unittest.TestCase):
    def test_nested_lists_and_amounts_are_flattened_in_schema_order(self) -> None:
        fields, summary = summarize("book_reservation", BOOKING, SPECS["book_reservation"], ToolGateSettings())
        self.assertEqual(
            summary,
            "Book reservation, with user id: anya_garcia_5412; origin: JFK; destination: SEA; flight type: "
            "round_trip; cabin: economy; flights 1 flight number: HAT083; flights 1 date: 2024-05-20; flights 2 "
            "flight number: HAT112; flights 2 date: 2024-05-27; passengers 1 first name: Anya; passengers 1 last "
            "name: Garcia; passengers 1 dob: 1990-04-02; payment methods 1 payment id: gift_card_7815826; payment "
            "methods 1 amount: 410; total baggages: 2; nonfree baggages: 0; insurance: no.",
        )
        self.assertEqual(len(fields), 17)  # every leaf: nothing is counted instead of named

    def test_values_render_fixed(self) -> None:
        spec = ToolSpec(name="set_flags", description="Set flags.")
        _, summary = summarize(
            "set_flags", {"on": True, "off": False, "ratio": 2.0, "note": None}, spec, ToolGateSettings()
        )
        self.assertEqual(summary, "Set flags, with on: yes; off: no; ratio: 2; note: none.")

    def test_not_consequential_paths_are_omitted_and_labels_used(self) -> None:
        settings = ToolGateSettings(
            not_consequential=("passengers[].dob", "insurance"),
            labels={"payment_methods[].amount": "amount in dollars"},
        )
        fields, summary = summarize("book_reservation", BOOKING, SPECS["book_reservation"], settings)
        paths = [path for path, _ in fields]
        self.assertNotIn("passengers[0].dob", paths)
        self.assertNotIn("insurance", paths)
        self.assertEqual(len(paths), 15)
        self.assertIn("payment methods 1 payment id: gift_card_7815826; amount in dollars: 410", summary)


class GateStateTests(unittest.TestCase):
    def gate(self, **settings: Any) -> tuple[WriteGate, list[tuple[str, dict]]]:
        emitted: list[tuple[str, dict]] = []
        resolved = resolve_gate(WriteGateSettings(enabled=True, **settings), list(SPECS.values()))
        return WriteGate(resolved, lambda kind, **data: emitted.append((kind, data))), emitted

    def test_read_and_exempt_calls_are_never_held(self) -> None:
        gate, _ = self.gate(exempt=("transfer_to_human_agents",))
        calls = [
            call("c1", "get_reservation_details", {"reservation_id": "ZFA04Y"}),
            call("c2", "transfer_to_human_agents", {"summary": "wants a human"}),
        ]
        screened = gate.screen(calls, GateState())
        self.assertEqual([c.id for c in screened.allowed], ["c1", "c2"])
        self.assertEqual(screened.held, {})

    def test_two_gated_calls_in_one_step_need_their_own_yes(self) -> None:
        gate, _ = self.gate()
        other = {**BAGS, "reservation_id": "4WQ150"}
        first = gate.screen(
            [call("c1", "update_reservation_baggages", BAGS), call("c2", "update_reservation_baggages", other)],
            GateState(),
        )
        self.assertEqual([p.proposal_id for p in first.state.proposals], ["P1", "P2"])
        self.assertEqual(set(first.held), {"c1", "c2"})
        state = gate.present(first.state, gate.unpresented(first.state))
        self.assertEqual(state.presented, "P1")

        state, bound = gate.begin_caller_turn(state, {"P1": True})
        self.assertEqual(bound.proposal_id, "P1")
        state = gate.judge(state, bound, "yes")
        reissued = gate.screen(
            [call("c3", "update_reservation_baggages", BAGS), call("c4", "update_reservation_baggages", other)], state
        )
        self.assertEqual([c.id for c in reissued.allowed], ["c3"])  # only the confirmed call
        self.assertEqual(list(reissued.held), ["c4"])
        self.assertEqual(json.loads(reissued.held["c4"])["status"], "confirmation_required")
        self.assertIsNotNone(gate.unpresented(reissued.state))  # the second one is presented on its own

    def test_internal_error_fails_closed(self) -> None:
        gate, emitted = self.gate()

        def broken(*_args: Any, **_kwargs: Any) -> Any:
            raise RuntimeError("boom")

        gate._propose = broken  # type: ignore[method-assign]
        screened = gate.screen([call("c1", "update_reservation_baggages", BAGS)], GateState())
        self.assertEqual(screened.allowed, ())
        self.assertEqual(json.loads(screened.held["c1"])["status"], "confirmation_required")
        self.assertEqual(emitted[0][0], "write_gate_error")


# -- through the runner ----------------------------------------------------------------------


class RunnerTests(_LogCase):
    def make(
        self, gate: WriteGateSettings = GATE, normalization: NormalizationSettings | None = None
    ) -> tuple[TextAgentRunner, FakeChatClient, FakeChatClient]:
        config = voice_config()
        frontend, backend = FakeChatClient(), FakeChatClient()
        runner = TextAgentRunner(
            base_config=config.agent,
            tools_config=config.tools,
            instructions_config=config.instructions,
            clients=AgentClients(backend=backend, frontend=frontend),
            sink=SessionRoutingSink(EventLog(str(self.log_path))),
            session_id="sess_gate",
            normalization=normalization or config.normalization,
            prompt_context=prompt_context(config),
            write_gate=gate,
        )
        runner.configure(tools=AIRLINE_TOOLS, instructions="policy")
        return runner, frontend, backend

    def held_result(self, backend: FakeChatClient, index: int) -> dict[str, Any]:
        message = backend.calls[index]["messages"][-1]
        self.assertEqual(message.role, "tool")
        return json.loads(message.content)

    async def present(self, runner: TextAgentRunner, frontend: FakeChatClient, backend: FakeChatClient) -> None:
        frontend.queue(delegate_response("Add bags to reservation ZFA04Y."))
        backend.queue(bags_call("C1"), text_response("Here is the change."))
        reply = await runner.respond("Please add bags to ZFA04Y.")
        self.assertEqual(reply.calls, ())
        self.assertEqual(reply.text, f"Here is the change. {BAGS_SUMMARY} {CONFIRM_QUESTION}")
        self.assertEqual(reply.presentation, "P1")


class RunnerFlowTests(RunnerTests, unittest.IsolatedAsyncioTestCase):
    async def test_hold_full_summary_yes_then_the_matching_reissue_goes_out(self) -> None:
        runner, frontend, backend = self.make()
        self.assertFalse(runner._normalization.tool_arguments.enabled)  # gate holds without normalization
        await self.present(runner, frontend, backend)
        held = self.held_result(backend, 1)
        self.assertEqual(held["status"], "confirmation_required")
        self.assertEqual(held["proposal_id"], "P1")
        self.assertEqual(backend.calls[1]["messages"][-2].tool_calls[0].id, "C1")

        runner.presentation_heard("P1", complete=True)
        frontend.queue(confirm_response("yes"))
        backend.queue(bags_call("C2"), text_response("Done, the bags are added."))
        reply = await runner.respond("Yes, go ahead.")

        self.assertEqual([(c.call_id, c.name) for c in reply.calls], [("C2", "update_reservation_baggages")])
        self.assertEqual(json.loads(reply.calls[0].arguments), BAGS)
        self.assertEqual(frontend.tool_choices[1], FORCE_CALL_BACKEND)
        self.assertEqual(frontend.calls[1]["tools"], list(FRONTEND_TOOLS_CONFIRMATION))
        self.assertIn(BAGS_SUMMARY, frontend.calls[1]["messages"][0].content)
        self.assertIn('"confirmation": "yes"', backend.calls[2]["messages"][-1].content)

        final = await runner.resume({"C2": "ok"})
        self.assertEqual(final.text, "Done, the bags are added.")
        tail = backend.calls[3]["messages"][-2:]
        self.assertEqual((tail[0].tool_calls[0].id, tail[1].tool_call_id, tail[1].content), ("C2", "C2", "ok"))
        self.assertEqual(self.records("write_confirmed")[0]["call_id"], "C2")
        self.assertEqual(self.records("write_confirmed")[0]["proposal_id"], "P1")

    async def test_yes_then_different_arguments_is_a_new_proposal(self) -> None:
        runner, frontend, backend = self.make()
        await self.present(runner, frontend, backend)
        runner.presentation_heard("P1", complete=True)
        frontend.queue(confirm_response("yes"))
        backend.queue(bags_call("C2", total_baggages=2), text_response("Let me confirm the change."))
        reply = await runner.respond("Yes.")

        self.assertEqual(reply.calls, ())
        held = self.held_result(backend, 3)
        self.assertEqual(held["status"], "arguments_mismatch")
        self.assertEqual((held["proposal_id"], held["previous_proposal_id"]), ("P2", "P1"))
        self.assertEqual(held["differences"], {"total_baggages": {"before": 3, "after": 2}})
        self.assertEqual(reply.presentation, "P2")
        self.assertIn("total baggages: 2", reply.text)

    async def test_partial_or_no_invalidates_and_the_correction_is_a_new_proposal(self) -> None:
        for verdict, words in (("partial", "Yes to the date, but only two bags."), ("no", "No, two bags.")):
            with self.subTest(verdict=verdict):
                runner, frontend, backend = self.make()
                await self.present(runner, frontend, backend)
                runner.presentation_heard("P1", complete=True)
                frontend.queue(confirm_response(verdict, query="Add only two bags to ZFA04Y."))
                backend.queue(bags_call("C2", total_baggages=2), text_response("Here is the corrected change."))
                reply = await runner.respond(words)

                self.assertEqual(reply.calls, ())
                query = backend.calls[2]["messages"][-1].content
                self.assertIn(f'"status": "not_confirmed", "proposal_id": "P1", "reason": "{verdict}"', query)
                self.assertEqual(reply.presentation, "P2")
                self.assertIn("total baggages: 2", reply.text)
                invalidated = self.records("write_invalidated")
                self.assertIn(("P1", verdict), [(r["proposal_id"], r["reason"]) for r in invalidated])
                self.log_path.unlink()

    async def test_missing_invalid_or_no_call_backend_is_unclear(self) -> None:
        cases = {
            "missing": [confirm_response(None)],
            "invalid": [confirm_response("sure")],
            "no_call_backend": [text_response("Great, done!"), text_response("Sure.")],
        }
        for name, responses in cases.items():
            with self.subTest(case=name):
                runner, frontend, backend = self.make()
                await self.present(runner, frontend, backend)
                runner.presentation_heard("P1", complete=True)
                frontend.queue(*responses)
                backend.queue(bags_call("C2"), text_response("Here is the change again."))
                reply = await runner.respond("Okay.")

                self.assertEqual(reply.calls, ())
                self.assertIn(
                    ("P1", "unclear"), [(r["proposal_id"], r["reason"]) for r in self.records("write_invalidated")]
                )
                self.assertFalse(self.records("write_confirmed"))
                self.log_path.unlink()

    async def test_okay_and_okay_but_follow_the_frontends_verdict(self) -> None:
        for words, verdict, sent in (
            ("Okay.", "yes", True),
            ("Go ahead.", "yes", True),
            ("Okay, but two bags.", "partial", False),
        ):
            with self.subTest(words=words):
                runner, frontend, backend = self.make()
                await self.present(runner, frontend, backend)
                runner.presentation_heard("P1", complete=True)
                frontend.queue(confirm_response(verdict))
                backend.queue(bags_call("C2", **({} if sent else {"total_baggages": 2})), text_response("Here it is."))
                reply = await runner.respond(words)

                self.assertEqual(bool(reply.calls), sent)
                # The frontend sees the caller's exact words and the summary they answer.
                request = frontend.calls[1]["messages"]
                self.assertEqual(request[-1].content, words)
                self.assertIn(BAGS_SUMMARY, request[0].content)

    async def test_a_direct_answer_on_the_bound_turn_is_repaired_into_call_backend(self) -> None:
        runner, frontend, backend = self.make()
        await self.present(runner, frontend, backend)
        runner.presentation_heard("P1", complete=True)
        frontend.queue(text_response("You're welcome!"), confirm_response("yes"))
        backend.queue(bags_call("C2"), text_response("Done."))
        reply = await runner.respond("Yes please.")

        self.assertEqual([c.call_id for c in reply.calls], ["C2"])
        self.assertEqual(frontend.tool_choices[1:], [FORCE_CALL_BACKEND, FORCE_CALL_BACKEND])
        self.assertTrue(self.records("frontend_repair"))

    async def test_incomplete_presentation_never_binds(self) -> None:
        runner, frontend, backend = self.make()
        await self.present(runner, frontend, backend)
        runner.presentation_heard("P1", complete=False)
        frontend.queue(delegate_response("Add the bags."))
        backend.queue(bags_call("C2"), text_response("Here is the change."))
        reply = await runner.respond("Yes.")

        self.assertEqual(reply.calls, ())
        self.assertIsNone(frontend.tool_choices[1])  # not a confirmation turn
        self.assertIn(
            ("P1", "interrupted"), [(r["proposal_id"], r["reason"]) for r in self.records("write_invalidated")]
        )
        self.assertEqual(reply.presentation, "P2")

    async def test_repeated_reissues_before_a_confirmation_are_held_then_the_turn_stops(self) -> None:
        runner, frontend, backend = self.make()
        frontend.queue(delegate_response("Add bags to reservation ZFA04Y."))
        backend.queue(bags_call("C1"), bags_call("C1b"), bags_call("C1c"), text_response("never reached"))
        reply = await runner.respond("Add bags to ZFA04Y.")

        self.assertEqual(reply.calls, ())
        self.assertEqual(len(backend.calls), 3)  # the loop stopped after max_unconfirmed_reissues re-issues
        self.assertEqual(reply.text, f"{BAGS_SUMMARY} {CONFIRM_QUESTION}")
        self.assertEqual(reply.presentation, "P1")
        self.assertEqual([r["reissue"] for r in self.records("write_held")], [1, 2])
        frontend_roles = [m.role for m in runner.state.frontend_history.messages[-4:]]
        self.assertEqual(frontend_roles, ["user", "assistant", "tool", "assistant"])  # the turn closed cleanly

    async def test_summary_over_the_limit_is_blocked(self) -> None:
        runner, frontend, backend = self.make(WriteGateSettings(enabled=True, max_summary_chars=40))
        frontend.queue(delegate_response("Add bags to reservation ZFA04Y."))
        backend.queue(bags_call("C1"), text_response("I cannot do that in one step."))
        reply = await runner.respond("Add bags.")

        self.assertEqual(reply.calls, ())
        self.assertIsNone(reply.presentation)
        self.assertEqual(self.held_result(backend, 1)["status"], "summary_too_long")
        problem = self.records("write_gate_config_problem")[0]
        self.assertEqual((problem["problem"], problem["max_summary_chars"]), ("summary_too_long", 40))

    async def test_exempt_and_read_calls_go_out(self) -> None:
        runner, frontend, backend = self.make()
        frontend.queue(delegate_response("Look it up, then transfer me."))
        backend.queue(
            tool_response(("get_reservation_details", {"reservation_id": "ZFA04Y"}), ids=["R1"]),
            tool_response(("transfer_to_human_agents", {"summary": "wants a human"}), ids=["T1"]),
        )
        self.assertEqual([c.call_id for c in (await runner.respond("Check it, then a human please.")).calls], ["R1"])
        self.assertEqual([c.call_id for c in (await runner.resume({"R1": "{}"})).calls], ["T1"])
        self.assertFalse(self.records("write_proposed"))

    async def test_local_rounds_exhausted_never_sends_a_gated_call(self) -> None:
        rule = ArgumentRule(
            tool="update_reservation_baggages",
            argument="reservation_id",
            label="reservation ID",
            pattern=r"^[A-Z0-9]{6}$",
        )
        normalization = NormalizationSettings(
            tool_arguments=ToolArgumentSettings(enabled=True, rules=(rule,), max_local_rounds=0)
        )
        runner, frontend, backend = self.make(normalization=normalization)
        frontend.queue(delegate_response("Add bags."))
        backend.queue(bags_call("C1", reservation_id="ZFA0"), text_response("Here is the change."))
        reply = await runner.respond("Add bags to ZFA0.")

        self.assertTrue(self.records("local_rounds_exhausted"))
        self.assertEqual(reply.calls, ())
        self.assertEqual(self.held_result(backend, 1)["status"], "confirmation_required")


# -- through the session: the binding to playback --------------------------------------------


class PlaybackBindingTests(_LogCase, unittest.IsolatedAsyncioTestCase):
    def make(self, gate: WriteGateSettings = GATE) -> tuple[SessionHarness, FakeChatClient, FakeChatClient]:
        frontend, backend = FakeChatClient(), FakeChatClient()
        harness = SessionHarness(
            config=replace(voice_config(), write_gate=gate),
            clients=AgentClients(backend=backend, frontend=frontend),
            event_log=EventLog(str(self.log_path)),
        )
        return harness, frontend, backend

    async def presented(self) -> tuple[SessionHarness, FakeChatClient, FakeChatClient]:
        harness, frontend, backend = self.make()
        frontend.queue(delegate_response("Add bags to reservation ZFA04Y."))
        backend.queue(bags_call("C1"), text_response("Here is the change."))
        await harness.start(tau2_session_update("airline"))
        await harness.speak()
        await harness.wait_for("response.done")
        self.assertFalse(harness.of_type("response.function_call_arguments.done"))
        return harness, frontend, backend

    def function_calls(self, harness: SessionHarness) -> list[str]:
        return [event["call_id"] for event in harness.of_type("response.function_call_arguments.done")]

    async def test_fully_heard_summary_then_yes_sends_exactly_the_stored_call(self) -> None:
        harness, frontend, backend = await self.presented()
        await harness.idle(4000)  # the whole summary plays
        frontend.queue(confirm_response("yes"))
        backend.queue(bags_call("C2"), text_response("Done."))
        await harness.speak()
        await harness.wait_for("response.done", 2)

        self.assertEqual(self.records("write_presentation")[0]["complete"], True)
        self.assertEqual(self.function_calls(harness), ["C2"])
        arguments = harness.of_type("response.function_call_arguments.done")[0]["arguments"]
        self.assertEqual(json.loads(arguments), BAGS)
        await harness.close()

    async def test_barge_in_during_the_summary_then_yes_sends_nothing(self) -> None:
        harness, frontend, backend = await self.presented()
        frontend.queue(delegate_response("Add the bags.", filler=""))
        backend.queue(text_response("Okay."))
        await harness.feed(pcmu_silence(300))
        await harness.feed(pcmu_speech(300) + pcmu_silence(700))  # the caller says "yes" over the summary
        await harness.wait_for("response.done", 2)

        self.assertIsNone(frontend.tool_choices[1])  # not a confirmation turn

        self.assertEqual(self.records("write_presentation")[0]["complete"], False)
        self.assertEqual(self.function_calls(harness), [])
        self.assertIn("interrupted", [r["reason"] for r in self.records("write_invalidated")])
        await harness.close()

    async def test_no_gated_call_keeps_the_event_stream(self) -> None:
        harness, frontend, backend = self.make()
        frontend.queue(delegate_response("Look up the user."))
        backend.queue(tool_response(("get_users", {}), ids=["call_a"]), text_response("Found them."))
        await harness.start(tau2_session_update())
        await harness.speak()
        await harness.wait_for("response.done")
        await harness.send(
            {
                "type": "conversation.item.create",
                "item": {"type": "function_call_output", "call_id": "call_a", "output": "ok"},
            }
        )
        await harness.send({"type": "response.create"})
        await harness.wait_for("response.done", 2)
        await harness.idle(300)
        await harness.close()

        trace = json.dumps(normalize_trace(harness.events), sort_keys=True).encode("utf-8")
        self.assertEqual(hashlib.sha256(trace).hexdigest(), NO_OVERLAP_TRACE_SHA256)
        self.assertFalse(self.records("write_proposed"))


if __name__ == "__main__":
    unittest.main()
