# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D103, D107

"""Explicit gateway policy on real harness runs with provider-free session/placement doubles."""

import asyncio
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import mock
from uuid import uuid4

from acp import RequestError
from acp.schema import PromptResponse, Usage

from examples.frontend_backend_live.delegation_core.backend.harness import usage
from examples.frontend_backend_live.delegation_core.backend.harness.acp.client import AcpSession
from examples.frontend_backend_live.delegation_core.backend.harness.acp.hermes.agent import HermesAgent
from examples.frontend_backend_live.delegation_core.backend.harness.acp.hermes.evidence import ModelOrigin, OriginStore
from examples.frontend_backend_live.delegation_core.backend.harness.acp.hermes.gateway_control import (
    GatewayControl,
    GatewayProcess,
    GatewaySteer,
    GatewayTurn,
)
from examples.frontend_backend_live.delegation_core.backend.harness.acp.hermes.steer import HermesSteering
from examples.frontend_backend_live.delegation_core.backend.harness.acp.updates import SessionUpdates
from examples.frontend_backend_live.delegation_core.backend.harness.conductor import (
    ConductorRoundFailed,
    HarnessConductor,
)
from examples.frontend_backend_live.delegation_core.backend.harness.protocol import (
    MessageChunk,
    PromptEnded,
)
from examples.frontend_backend_live.delegation_core.backend.harness.worker import HarnessTurnError, WorkerRun
from examples.frontend_backend_live.delegation_core.backend.port import ConductorInput, DecisionRejected
from tests.unit.delegation_core.harness.doubles import FakeHost, FakeTaskContext, settings, until
from tests.unit.delegation_core.helpers import FakeDecisionTarget, a_request, answers, logged, read_all

COUNTERS = {"inputTokens": 40, "outputTokens": 8, "totalTokens": 48}
LOWER = {"inputTokens": 20, "outputTokens": 4, "totalTokens": 24}
HIGHER = {"inputTokens": 80, "outputTokens": 16, "totalTokens": 96}


def gateway():
    return settings()


def given(identifier="D1"):
    request = a_request(identifier, "Read the menu.")
    return ConductorInput((request,), (), request.input, 0)


class PolicyTests(unittest.TestCase):
    def setUp(self):
        self.agent = gateway().agent

    def test_normal_hermes_uses_one_supported_terminal_contract(self):
        selected = gateway()
        self.assertIs(type(selected.agent), HermesAgent)
        self.assertEqual(selected.describe()["agent"], "hermes")
        self.assertEqual(selected.agent.capabilities.delivery, "cancel_reprompt")
        self.assertEqual(selected.agent.capabilities.counter_scope, "unscoped")

    def test_typed_terminal_and_inference_evidence_are_independent(self):
        for counters in (COUNTERS, LOWER, HIGHER, None, {"totalTokens": 0}):
            for count, invoked in ((None, None), (0, False), (1, True), (-1, None), (False, None), (True, None)):
                with self.subTest(counters=counters, count=count):
                    result = self.agent.completion(PromptEnded(2, "end_turn", counters, completion_calls=count))
                    self.assertEqual(result.outcome, "succeeded")
                    self.assertIs(result.model_invoked, invoked)
        terminals = [PromptEnded(2, "end_turn", COUNTERS, "admitted_turn_failed")]
        terminals += [PromptEnded(2, stop, COUNTERS) for stop in ("cancelled", "refusal", "max_tokens", None)]
        for terminal in terminals:
            with self.subTest(terminal=terminal):
                self.assertNotEqual(self.agent.completion(terminal).outcome, "succeeded")

    def test_unscoped_snapshots_never_become_per_prompt_usage_or_zero_deltas(self):
        for current in (COUNTERS, LOWER, HIGHER, None, {}):
            with self.subTest(current=current):
                self.assertEqual(usage.round_counters(current, COUNTERS, agent=self.agent), {})
                observation = usage.observation(self.agent, current, None)
                self.assertEqual(observation["raw_reported_counters"], current)
                self.assertEqual(observation["usage_scope"], "unscoped")
        self.assertEqual(usage.round_counters(COUNTERS, COUNTERS), {})

    def test_explicit_invocation_counters_override_snapshots_without_invented_fields(self):
        exact = {"inputTokens": 7, "outputTokens": 2, "totalTokens": 9, "thoughtTokens": True}
        self.assertEqual(
            usage.round_counters(COUNTERS, HIGHER, exact, agent=self.agent),
            {"inputTokens": 7, "outputTokens": 2, "totalTokens": 9},
        )
        self.assertEqual(usage.round_counters(COUNTERS, LOWER, {}, agent=self.agent), {})
        self.assertEqual(usage.observation(self.agent, COUNTERS, {})["usage_scope"], "invocation")

    def test_completed_steer_publishes_only_the_signed_reply_when_hermes_combines_text(self):
        answer = "Tuesday at 10. ✓"
        cases = [(answer, "end_turn", answer), (answer[:-1], "end_turn", None), (answer, "cancelled", None)]
        with TemporaryDirectory() as folder:
            home = Path(folder)
            scope = OriginStore.create(home, "worker")
            steering = HermesSteering(scope, "session", home, {}, 1)
            for tail, stop_reason, expected in cases:
                with self.subTest(tail=tail, stop_reason=stop_reason):
                    steering.prompt(1, "invocation", "Find a time.")
                    current = steering.foreground
                    input_id = uuid4().hex
                    current.native = ("session", "turn")
                    current.required.append(input_id)
                    scope.issue_input(current.marker, input_id)
                    origin = ModelOrigin(current.marker, "session", "turn", "request", 0, (input_id,))
                    scope.record_model(origin, answer)
                    for text in ("Monday.", f"\n{tail}", f"\n{tail}"):
                        self.assertTrue(steering.capture(MessageChunk("combined", text, 1)))
                    result = steering.finish(PromptEnded(1, stop_reason, terminal_confirmed=True))
                    self.assertEqual(result, MessageChunk("combined", expected, 1) if expected else None)


class WorkerTests(unittest.IsolatedAsyncioTestCase):
    async def test_terminal_settles_only_known_steering_delivery(self):
        turn = GatewayTurn("admission", 3, 4)
        for blocked, gateway_status, delivery in (
            ("bind", "queued", "not_applied"),
            ("steer", "not_sent", "not_applied"),
            ("steer", "queued", "unknown"),
            ("cleanup", "queued", "unknown"),
        ):
            with self.subTest(blocked=blocked, gateway_status=gateway_status), TemporaryDirectory() as folder:
                home = Path(folder)
                scope = OriginStore.create(home, "worker")
                steering = HermesSteering(scope, "session", home, {}, 1)
                steering.prompt(1, "invocation", "Find a time.")
                scope.record_start(
                    ModelOrigin(scope.invocation(), "session", "turn", "request", 1.0, ()), sys.executable
                )
                reply = GatewaySteer(gateway_status, turn=turn)
                controller = SimpleNamespace(
                    bind=mock.AsyncMock(return_value=turn),
                    steer=mock.AsyncMock(return_value=reply),
                )
                connection = mock.MagicMock()
                connection.__aenter__.return_value = controller
                step, result = {
                    "bind": (controller.bind, turn),
                    "steer": (controller.steer, reply),
                    "cleanup": (connection.__aexit__, None),
                }[blocked]
                reached, resume = asyncio.Event(), asyncio.Event()

                async def wait_for_resume(*args, reached=reached, resume=resume, result=result, **kwargs):
                    reached.set()
                    await resume.wait()
                    return result

                step.side_effect = wait_for_resume
                with mock.patch.object(GatewayProcess, "connect", return_value=connection):
                    receipt = steering.submit(1, "invocation", "correction", "Tuesday.")
                    assert receipt is not None
                    (command,) = steering.commands
                    try:
                        async with asyncio.timeout(2):
                            await reached.wait()
                            self.assertIsNone(steering.finish(PromptEnded(1, "end_turn", terminal_confirmed=True)))
                            self.assertIsNone(steering.foreground)
                            self.assertEqual(receipt.done(), blocked != "steer")
                            resume.set()
                            await command
                            self.assertEqual((await receipt).delivery, delivery)
                            self.assertEqual(controller.steer.await_count, int(blocked != "bind"))
                    finally:
                        await steering.close()

    async def test_gateway_preserves_only_authoritative_refusals(self):
        class ClientError(ValueError):
            pass

        class RPCError(ClientError):
            pass

        turn = GatewayTurn("admission", 3, 4)
        sentinel = "SDK pre-write refusal"
        sdk = SimpleNamespace(GatewayClientError=ClientError, GatewayRPCError=RPCError, NOT_SENT=sentinel)
        with mock.patch.dict(sys.modules, {"hermes_cli": SimpleNamespace(gateway_client=sdk)}):
            for reply, outcome in (
                (ClientError(sentinel), "not_sent"),
                (ClientError("Socket closed during send"), "unknown"),
                (RPCError(sentinel), "unknown"),
                (RPCError("stale_generation"), "not_sent"),
                ({"status": "rejected", "delivered": False, **vars(turn)}, "not_sent"),
                ({"status": "rejected", **vars(turn)}, "unknown"),
                ({"status": "rejected", "delivered": False, **vars(turn), "execution_generation": 5}, "unknown"),
                ({"status": "redirected", **vars(turn)}, "queued"),
            ):
                with self.subTest(reply=reply):
                    rpc = mock.AsyncMock(side_effect=[{"status": "started", **vars(turn)}, reply])
                    result = await GatewayControl(SimpleNamespace(rpc=rpc), "session").steer(
                        turn, "Tuesday.", redirect=True
                    )
                    self.assertEqual((result.status, result.turn), (outcome, turn))

    async def open(self, host=None, task=None):
        host = host or FakeHost(gateway())
        task = task or FakeTaskContext()
        run = WorkerRun(task, host)
        self.addAsyncCleanup(run.release)
        events = run.events()
        first = await anext(events)
        self.assertEqual(first["type"], "response.created")
        return host, task, run, events, host.runtime.client.sessions[-1]

    async def test_named_current_text_with_missing_usage_is_usable_but_not_token_accounting(self):
        _, task, _, events, native = await self.open()
        native.answer(1, "named", "Read result.")
        result = await read_all(events)
        self.assertEqual(answers(result), ["Read result."])
        self.assertNotIn("usage", result[-1]["response"])
        self.assertIsNone(task.traced("harness.round.usage")[0]["raw_reported_counters"])

    async def test_unnamed_ack_empty_thought_and_invalid_named_id_do_not_become_worker_answers(self):
        for message_id, text in ((None, "Queued."), ("thought-only", ""), ("", "Done."), (7, "Done.")):
            with self.subTest(message_id=message_id):
                _, _, _, events, native = await self.open()
                native.put(MessageChunk(message_id, text))
                native.end(1, usage=HIGHER)
                with self.assertRaises(HarnessTurnError):
                    await read_all(events)

    async def test_error_cancel_refusal_and_zero_model_completion_cannot_publish_named_text(self):
        for stop, error, completed in (
            ("end_turn", "RPC: admitted_turn_failed", None),
            ("cancelled", None, None),
            ("refusal", None, None),
        ):
            with self.subTest(stop=stop, error=error, completed=completed):
                _, _, _, events, native = await self.open()
                native.say("named", "Never publish this.")
                native.end(1, stop, usage=COUNTERS, error=error, completion_calls=completed)
                with self.assertRaises(HarnessTurnError):
                    await read_all(events)


class AcpClientTests(unittest.IsolatedAsyncioTestCase):
    async def terminal(self, response=None, error=None):
        agent = gateway().agent
        mailbox = asyncio.Queue()

        async def prompt(**kwargs):
            session._dispatched(next(iter(session.outstanding)))
            if error is not None:
                raise error
            return response

        conn = SimpleNamespace(prompt=prompt)
        client = SimpleNamespace(
            agent=agent,
            alive=True,
            ended=None,
            conn=conn,
            prompts=set(),
            lose=mock.Mock(),
            connection_id="test-connection",
            transport_name="fixture",
            peer_name="hermes-agent",
            peer_version="fixture",
            _dispatching={},
        )
        session = AcpSession(client, "native-session", mailbox, SessionUpdates(agent, "c1"))
        number = session.prompt("A normal guarded request.")
        await session.ended(number)
        events = []
        while not mailbox.empty():
            events.append(mailbox.get_nowait())
        return next(event for event in events if isinstance(event, PromptEnded)), client

    async def test_actual_acp_client_preserves_repeated_raw_snapshots_without_invented_completion_or_invocation(self):
        for _ in range(2):
            ended, client = await self.terminal(
                PromptResponse(stop_reason="end_turn", usage=Usage(input_tokens=40, output_tokens=8, total_tokens=48))
            )
            self.assertEqual(ended.usage, COUNTERS)
            self.assertIsNone(ended.completion_calls)
            self.assertIsNone(ended.invocation_usage)
            client.lose.assert_not_called()

    async def test_rpc_failed_and_unknown_admission_stay_errors_even_with_positive_output_evidence(self):
        for message in ("admitted_turn_failed", "unknown_admission"):
            with self.subTest(message=message):
                ended, client = await self.terminal(error=RequestError(-32603, message))
                self.assertIn(message, ended.error)
                self.assertNotEqual(gateway().agent.completion(ended).outcome, "succeeded")
                client.lose.assert_not_called()

    async def test_own_slash_or_no_admission_terminal_without_named_output_is_rejected(self):
        ended, _ = await self.terminal(PromptResponse(stop_reason="end_turn"))
        self.assertIsNone(ended.usage)
        self.assertIsNone(gateway().agent.completion(ended).model_invoked)


class ConductorTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.host = FakeHost(gateway())
        self.conductor = HarnessConductor(self.host)
        self.target = FakeDecisionTarget()
        self.addAsyncCleanup(self.close)

    async def close(self):
        self.host.closed = True
        await self.conductor.close()

    async def opened(self, identifier="D1", place=None):
        task = asyncio.create_task(self.conductor.decide(given(identifier), place or self.target.place))
        await until(lambda: self.conductor.turn is not None)
        return task, self.conductor.session, self.conductor.turn

    async def place(self, identifier="D1"):
        return await self.conductor.door.call(
            "submit_decision",
            {"actions": [{"action": "start", "delegations": [identifier], "tasks": [{"brief": "Read the menu."}]}]},
        )

    async def test_completed_typed_callbacks_allow_same_or_decreasing_snapshot_with_no_model_text(self):
        for number, current in enumerate((COUNTERS, COUNTERS, LOWER, None), 1):
            task, native, turn = await self.opened(f"D{number}")
            self.assertFalse(turn.control_completed)
            await self.place(f"D{number}")
            self.assertTrue(turn.control_completed)
            native.end(native.last_prompt(), usage=current)
            await task
            self.assertFalse(native.retired)
        self.assertEqual(len(self.target.decisions), 4)

    async def test_a_clean_native_narrative_without_a_control_is_a_repairable_rejection(self):
        task, native, turn = await self.opened()
        native.answer(1, "named", "The already handled move needs no further placement.")
        with self.assertRaisesRegex(DecisionRejected, "one complete decision call"):
            await task
        self.assertFalse(turn.control_completed)

    async def test_unnamed_blank_or_invalid_id_is_not_current_conductor_evidence(self):
        for message_id, text in ((None, "Queued."), ("named", "  "), ("", "Own."), (7, "Own.")):
            with self.subTest(message_id=message_id):
                task, native, _ = await self.opened()
                native.put(MessageChunk(message_id, text))
                native.end(1, usage=HIGHER)
                with self.assertRaises(DecisionRejected):
                    await task
                self.assertTrue(native.retired)

    async def test_error_nonend_and_explicit_zero_override_completed_control(self):
        for stop, error, completed in (
            ("end_turn", "RPC: unknown", None),
            ("cancelled", None, None),
            ("refusal", None, None),
            ("max_tokens", None, None),
            (None, None, None),
        ):
            with self.subTest(stop=stop, error=error):
                task, native, _ = await self.opened()
                await self.place()
                native.say("named", "Done.")
                native.end(1, stop, usage=HIGHER, error=error, completion_calls=completed)
                with self.assertRaises(ConductorRoundFailed):
                    await task

    async def test_malformed_control_does_not_credit_a_callback(self):
        task, native, turn = await self.opened()
        reply = await self.conductor.door.call("submit_decision", {})
        self.assertIn("error", reply.lower())
        self.assertFalse(turn.control_completed)
        native.end(1, usage=HIGHER)
        with self.assertRaises(DecisionRejected):
            await task

    async def test_callback_exception_does_not_credit_a_completed_placement(self):
        async def failed(decision):
            raise RuntimeError("not completed")

        task, native, turn = await self.opened(place=failed)
        with self.assertRaisesRegex(RuntimeError, "not completed"):
            await self.place()
        self.assertFalse(turn.control_completed)
        native.end(1, usage=HIGHER)
        with self.assertRaises(ConductorRoundFailed):
            await task

    async def test_policy_refusal_from_completed_typed_callback_is_activity_not_business_acceptance(self):
        async def refused(decision):
            return "The request is no longer available; no placement accepted."

        task, native, turn = await self.opened(place=refused)
        reply = await self.place()
        self.assertIn("no placement accepted", reply)
        self.assertTrue(turn.control_completed)
        native.end(1)
        await task
        self.assertEqual(self.target.decisions, [])

    async def test_late_callback_cannot_credit_a_new_invocation(self):
        for detached in (False, True):
            with self.subTest(detached=detached):
                await self.check_late_callback(detached)

    async def check_late_callback(self, detached):
        entered, release = asyncio.Event(), asyncio.Event()

        async def held(decision):
            entered.set()
            await release.wait()
            return "Old callback completed."

        old_task, old_native, old_turn = await self.opened(place=held)
        call = asyncio.create_task(self.place())
        try:
            await entered.wait()
            self.assertFalse(old_turn.control_completed)
            if detached:
                await self.conductor.set_attachment(False)
                await old_task
                old_native.end(1, "cancelled")
                await self.conductor.set_attachment(True)
            else:
                old_native.say("first", "No further move.")
                old_native.end(1, usage=COUNTERS)
                with self.assertRaises(ConductorRoundFailed):
                    await old_task
            new_task, native, new_turn = await self.opened("D2")
            if not detached:
                self.assertIsNot(native, old_native)
        finally:
            release.set()
            await call
        self.assertFalse(old_turn.control_completed)
        self.assertFalse(new_turn.control_completed)
        native.end(native.last_prompt(), usage=HIGHER)
        with self.assertRaises(DecisionRejected):
            await new_task

    async def test_conductor_log_keeps_raw_scope_and_reports_normalized_usage_unknown(self):
        with logged("DEBUG") as lines:
            task, native, _ = await self.opened()
            await self.place()
            native.end(1, usage=COUNTERS)
            await task
        [row] = [line for line in lines if line.startswith("conductor.round.usage ")]
        self.assertIn("usage=None reported_counters=None usage_missing=True", row)
        self.assertIn("'raw_reported_counters': {'inputTokens': 40, 'outputTokens': 8, 'totalTokens': 48}", row)
        self.assertIn("'usage_scope': 'unscoped'", row)
