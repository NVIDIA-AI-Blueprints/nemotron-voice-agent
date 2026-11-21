# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D103, D107

"""Logical-call retention across fresh authenticated media attachments, with no action replay."""

import asyncio
import time
import unittest
from unittest import mock

from gateways.live.gateway import LiveGateway, LiveSettings
from gateways.live.gateway.websocket import run_primary_websocket
from gateways.live.protocol import ProtocolError, SessionConfig
from tests.unit.delegation_core.helpers import ask
from tests.unit.live_helpers import FakeAdapter, FakeWebSocket, Stream, response

ANSWER = {
    "id": "answer1",
    "type": "message",
    "role": "assistant",
    "phase": "final_answer",
    "status": "completed",
    "content": [{"type": "output_text", "text": "Your reservation is confirmed."}],
}
CALL = {
    "id": "item1",
    "call_id": "call1",
    "type": "function_call",
    "name": "reserve",
    "arguments": '{"private":"argument"}',
}
CONFIG = {
    "model": "live-1",
    "delegation": {"type": "responses", "responses": {"model": "model"}, "reconnect_grace_seconds": 30.0},
}


class RestorableAdapter(FakeAdapter):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.restored = None
        self.finals_seen = []
        self.attachments = []
        self.pending_calls = []
        self.fingerprint = "matching-config"

    def attachment_fingerprint(self):
        return self.fingerprint, self.config.model_dump_json() if self.config is not None else None

    def export_attachment(self):
        return {"history": ["Caller asked for a reservation"], "possibly_heard": True}

    def restore_attachment(self, snapshot):
        self.restored = snapshot
        if self.fail.get("restore"):
            raise self.fail["restore"]

    def restore_final(self, delegation_id):
        self.finals_seen.append(delegation_id)

    async def set_attachment(self, conversation_id, attached):
        self.attachments.append((conversation_id, attached))
        return {"attached": attached, "pending_calls": self.pending_calls if attached else []}


class ReconnectCase(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.core, self.adapters, self.sockets, self.runners = RestorableAdapter(), [], [], []
        self.factory_hold = None
        self.factory_failure = None

        async def factory(config, transport, ice_servers):
            if self.factory_hold is not None:
                await self.factory_hold.wait()
            adapter = RestorableAdapter()
            adapter.config = config
            if self.factory_failure is not None:
                adapter.fail["restore"] = self.factory_failure
            self.adapters.append(adapter)
            return adapter.parts()

        self.gateway = LiveGateway(factory, LiveSettings(), core=self.core.core)

    async def asyncTearDown(self):
        await self.gateway.close_all()
        for socket in self.sockets:
            socket.leave()
        await asyncio.gather(*self.runners, return_exceptions=True)

    async def socket(self, first):
        socket = FakeWebSocket()
        socket.send(first)
        self.sockets.append(socket)
        self.runners.append(asyncio.create_task(run_primary_websocket(self.gateway, socket)))
        return socket

    async def start(self):
        socket = await self.socket({"type": "session.start", "session": CONFIG, "reconnect": {"version": 1}})
        ready = await socket.wait_for("session.reconnect.ready")
        self.session = next(iter(self.gateway.sessions.values()))
        return socket, ready["reconnect"]

    async def drop(self, socket):
        socket.leave(1006)
        await asyncio.wait_for(self.runners[self.sockets.index(socket)], 2)
        self.assertEqual(self.session.attachment_state, "reconnecting")
        self.assertFalse(self.core.deleted.is_set())

    async def resume(self, authority, request_id="resume1"):
        socket = await self.resume_socket(authority, request_id)
        return socket, await socket.wait_for("session.resumed")

    async def resume_socket(self, authority, request_id="resume1"):
        resume = {key: authority[key] for key in ("token", "generation")}
        return await self.socket(
            {"type": "session.start", "resume": {"version": 1, **resume, "request_id": request_id}}
        )

    async def delegate(self, stream):
        self.core.streams.append(stream)
        return await self.session.delegation.delegate([{"role": "user", "content": "Make the reservation"}])


class ReconnectTests(ReconnectCase):
    async def test_journal_overflow_closes_for_capacity_without_relaying_the_unrecorded_result(self):
        socket, _ = await self.start()
        stream = Stream(response("too-much", ANSWER))
        with mock.patch("gateways.live.server_delegation.retention.MAX_ITEMS", 2):
            await self.delegate(stream)
            closed = await socket.wait_for("session.closed")
        self.assertEqual(closed["reason"], "rate_limit_exceeded")
        self.assertEqual(self.gateway.resume_leases, {})
        self.assertTrue(stream.closed)
        relayed = [event["event"] for event in socket.from_server if event["type"] == "response.event"]
        self.assertEqual([event["type"] for event in relayed], ["response.created"])
        self.assertEqual(len(self.core.requests), 1)

    async def test_journal_overflow_fences_a_paced_app_call_before_it_reaches_the_wire(self):
        socket, _ = await self.start()
        waiting, release = asyncio.Event(), asyncio.Event()

        async def pace_app_call():
            if CALL["call_id"] in self.session.delegation.routes.routes:
                waiting.set()
                await release.wait()

        with (
            mock.patch("gateways.live.server_delegation.retention.MAX_ITEMS", 3),
            mock.patch.object(self.session.fanout, "wait_for_room", pace_app_call),
        ):
            await self.delegate(response("with-app-call", CALL))
            try:
                await asyncio.wait_for(waiting.wait(), 2)
                with self.assertRaisesRegex(ProtocolError, "mailbox is full"):
                    await self.delegate(response("cannot-start", ANSWER))
            finally:
                release.set()
            closed = await socket.wait_for("session.closed")
        self.assertEqual(closed["reason"], "rate_limit_exceeded")
        self.assertEqual(len(self.core.requests), 1)
        self.assertNotIn(CALL["call_id"], self.session.delegation.mailbox.emitted)
        self.assertFalse(any(event.get("event", {}).get("item") == CALL for event in socket.from_server))

    async def test_a_stream_lost_while_away_replays_a_canonical_loss(self):
        socket, authority = await self.start()
        finish = asyncio.Event()
        delegation = await self.delegate(Stream(response("resp1", end=None), until=finish))
        await socket.wait_for("response.event")
        await self.drop(socket)
        finish.set()
        await asyncio.gather(*self.session.delegation.readers)
        last = self.session.delegation.mailbox.entries[-1]
        self.assertEqual((last.event["type"], last.lost), ("response.failed", True))
        await self.resume(authority)
        self.assertIn((delegation, "lost"), self.adapters[-1].heard)

    async def test_old_paced_event_cannot_reach_the_replacement_attachment(self):
        socket, authority = await self.start()
        waiting, release = asyncio.Event(), asyncio.Event()

        async def wait_for_room():
            waiting.set()
            await release.wait()

        with mock.patch.object(self.session.fanout, "wait_for_room", wait_for_room):
            old = asyncio.create_task(
                self.session.delegation_view.emit_paced("response.event", delegation_id="old", event={"type": "old"})
            )
            await waiting.wait()
            await self.drop(socket)
            resumed, _ = await self.resume(authority)
            release.set()
            await old
        await self.session.fanout.drain()
        self.assertFalse(any(e.get("delegation_id") == "old" for e in resumed.from_server))

    async def test_a_command_waiting_for_start_does_not_acquire_new_attachment_authority(self):
        socket, _ = await self.start()
        connection = self.session.socket.connection
        waiting, release = asyncio.Event(), asyncio.Event()

        async def wait_until_ready():
            waiting.set()
            await release.wait()

        self.session.status = "starting"
        message = {"type": "session.instructions.append", "operation_id": "stale", "text": "Old input"}
        with mock.patch.object(self.session, "_wait_until_ready", wait_until_ready):
            command = asyncio.create_task(self.session.handle(message, connection))
            await waiting.wait()
            self.session.status = "active"
            await self.drop(socket)
            release.set()
            await command
        self.assertNotIn("stale", self.session.reconnect.operations)
        self.assertEqual(self.adapters[0].appended, [])

    async def test_caller_effects_are_fenced_before_a_slow_old_writer_finishes_closing(self):
        socket, _ = await self.start()
        await self.delegate(response("resp1", ANSWER))
        await asyncio.gather(*self.session.delegation.readers)
        sub = self.session.fanout.subscribers[self.session.socket.connection]
        closing, release = asyncio.Event(), asyncio.Event()
        original_close = sub.close

        async def slow_close():
            closing.set()
            await release.wait()
            await original_close()

        with mock.patch.object(sub, "close", slow_close):
            loss = asyncio.create_task(self.session.reconnect.lost())
            await closing.wait()
            self.assertFalse(self.core.attachments[-1][1])
            self.assertFalse(self.session.delegation._attached)
            self.assertNotIn(sub, self.session.fanout.subscribers.values())
            release.set()
            await loss

    async def test_possible_app_execution_is_never_reissued_or_accepted_as_new_authority(self):
        socket, authority = await self.start()
        await self.delegate(response("resp1", CALL))
        await asyncio.gather(*self.session.delegation.readers)
        await self.session.fanout.drain()
        await self.drop(socket)
        resumed, result = await self.resume(authority)
        self.assertEqual(result["pending"]["unclear_calls"], [{"call_id": "call1", "name": "reserve"}])
        self.assertNotIn("private", str(result["pending"]))
        self.assertFalse(any(e.get("event", {}).get("item", {}).get("call_id") == "call1" for e in resumed.from_server))
        resumed.send(
            {
                "type": "response.item.create",
                "operation_id": "late-result",
                "item": {"type": "function_call_output", "call_id": "call1", "output": "success"},
            }
        )
        refused = await resumed.wait_for("error")
        self.assertEqual(refused["error"]["code"], "action_outcome_unclear")
        self.assertEqual(len(self.core.requests), 1)

    async def start_with_delayed_app_call(self):
        socket, authority = await self.start()
        finish = asyncio.Event()
        await self.delegate(Stream(response("resp1", end=None), until=finish, then=response("resp1", CALL)[1:]))
        await socket.wait_for("response.event")
        return socket, authority, finish

    @staticmethod
    def is_app_call(event):
        return event.get("event", {}).get("item", {}).get("call_id") == "call1"

    def app_calls(self, socket):
        return [event for event in socket.from_server if self.is_app_call(event)]

    async def app_call(self, socket):
        return await socket.wait_for("response.event", where=self.is_app_call)

    async def test_while_live_forwarding_is_paced_recovery_can_publish_the_same_app_call_once(self):
        for before_resume in (True, False):
            with self.subTest(before_resume=before_resume):
                await self.check_paced_app_call(before_resume)
                await self.asyncTearDown()
                await self.asyncSetUp()

    async def check_paced_app_call(self, before_resume):
        socket, authority, finish = await self.start_with_delayed_app_call()
        blocked, release = asyncio.Event(), asyncio.Event()
        original = self.core.set_attachment
        attempts = 0

        async def set_attachment(conversation, attached):
            nonlocal attempts
            if attached and not before_resume:
                attempts += 1
                if attempts == 2:
                    self.core.pending_calls = ["call1"]
                    finish.set()
                    await blocked.wait()
            return await original(conversation, attached)

        async def wait_for_room():
            if not blocked.is_set():
                blocked.set()
                await release.wait()

        with (
            mock.patch.object(self.core, "set_attachment", set_attachment),
            mock.patch.object(self.session.fanout, "wait_for_room", wait_for_room),
        ):
            try:
                if before_resume:
                    finish.set()
                    await asyncio.wait_for(blocked.wait(), 2)
                await self.drop(socket)
                if before_resume:
                    self.core.pending_calls = ["call1"]
                resumed, receipt = await self.resume(authority)
                self.assertEqual(receipt["pending"]["unclear_calls"], [])
                await self.app_call(resumed)
            finally:
                release.set()
            await asyncio.gather(*self.session.delegation.readers)
        await self.session.fanout.drain()
        self.assertEqual(self.app_calls(socket), [])
        self.assertEqual(len(self.app_calls(resumed)), 1)

    async def test_call_completed_while_away_or_resuming_is_replayed_once(self):
        for phase in ("detached", "core_reattach", "frontend_restore"):
            with self.subTest(phase=phase):
                await self.check_call_recovery(phase)
                await self.asyncTearDown()
                await self.asyncSetUp()

    async def check_call_recovery(self, phase):
        socket, authority, finish = await self.start_with_delayed_app_call()
        await self.drop(socket)
        original = self.core.set_attachment
        first_attach = True

        async def ready_call():
            finish.set()
            await asyncio.gather(*self.session.delegation.readers)
            self.core.pending_calls = ["call1"]

        async def attach(conversation, attached):
            nonlocal first_attach
            snapshot = await original(conversation, attached)
            if attached and first_attach:
                first_attach = False
                await ready_call()
            return snapshot

        async def restore(adapter):
            await ready_call()
            self.assertNotIn("call1", self.session.delegation.mailbox.emitted)

        if phase == "core_reattach":
            with mock.patch.object(self.core, "set_attachment", attach):
                resumed, receipt = await self.resume(authority)
        elif phase == "frontend_restore":
            with mock.patch.object(RestorableAdapter, "prepare_resume", restore, create=True):
                resumed, receipt = await self.resume(authority)
        else:
            await ready_call()
            resumed, receipt = await self.resume(authority)
        call = await self.app_call(resumed)
        self.assertLess(resumed.from_server.index(receipt), resumed.from_server.index(call))
        self.assertEqual(len(self.app_calls(resumed)), 1)

    async def test_revalidation_after_the_receipt_uses_the_original_grace(self):
        socket, authority = await self.start()
        await self.drop(socket)
        lease = self.session.reconnect
        original = self.core.set_attachment
        waiting, release = asyncio.Event(), asyncio.Event()
        attempts = 0

        async def stalled_revalidation(conversation, attached):
            nonlocal attempts
            if attached:
                attempts += 1
                if attempts == 2:
                    waiting.set()
                    await release.wait()
            return await original(conversation, attached)

        with mock.patch.object(self.core, "set_attachment", stalled_revalidation):
            lease.deadline = time.monotonic() + 0.2
            try:
                await self.resume(authority)
                await asyncio.wait_for(waiting.wait(), 1)
                await asyncio.wait_for(self.session.closed.wait(), 1)
            finally:
                release.set()
        self.assertEqual(self.gateway.resume_leases, {})

    async def test_exact_resume_retry_returns_receipt_without_another_attachment(self):
        socket, authority = await self.start()
        await self.drop(socket)
        await self.resume(authority)
        duplicate = await self.resume_socket(authority)
        receipt = await duplicate.wait_for("session.resume.receipt")
        self.assertEqual(receipt["reconnect"]["generation"], 2)
        self.assertEqual(len(self.adapters), 2)

    async def test_a_failed_new_frontend_does_not_extend_the_original_deadline(self):
        from gateways.live.gateway.websocket import PrimarySocketView

        for fence_fails in (False, True):
            with self.subTest(fence_fails=fence_fails):
                socket, authority = await self.start()
                await self.drop(socket)
                lease = self.session.reconnect
                deadline = lease.deadline
                self.factory_failure = RuntimeError("fresh pipeline failed")
                if fence_fails:
                    with mock.patch.object(self.core, "set_attachment", side_effect=RuntimeError("Core fence failed")):
                        replacement = await self.socket(
                            {"type": "session.start", "resume": {**authority, "request_id": "failed"}}
                        )
                        error = await replacement.wait_for("error")
                        self.assertEqual(error["error"]["code"], "resume_unavailable")
                        await asyncio.wait_for(self.session.closed.wait(), 1)
                    self.assertEqual(self.gateway.resume_leases, {})
                else:
                    view = PrimarySocketView(FakeWebSocket(), self.session.config.audio.format)
                    with self.assertRaises(RuntimeError):
                        await lease.attach(view, {**authority, "request_id": "failed"}, lease.scope)
                    self.assertEqual((lease.generation, lease.deadline, lease.state), (1, deadline, "reconnecting"))
                    self.assertFalse(self.core.attachments[-1][1])
                self.assertTrue(self.adapters[-1].closed)
                self.factory_failure = None
                if not fence_fails:
                    await self.resume(authority, "retry")
                    await self.asyncTearDown()
                    await self.asyncSetUp()

    async def test_expiry_revokes_authority_and_a_delayed_old_timer_cannot_close_a_new_generation(self):
        socket, authority = await self.start()
        await self.drop(socket)
        lease = self.session.reconnect
        await self.resume(authority)
        with mock.patch("gateways.live.session.reconnect.asyncio.sleep", new=mock.AsyncMock()):
            await lease._expire(1)
        self.assertEqual(self.session.status, "active")
        await lease.lost()
        lease.deadline = time.monotonic() - 1
        with mock.patch("gateways.live.session.reconnect.asyncio.sleep", new=mock.AsyncMock()):
            await lease._expire(2)
        self.assertEqual((self.session.status, self.gateway.resume_leases), ("closed", {}))

    async def test_operation_receipts_prevent_repeated_typed_input(self):
        socket, _ = await self.start()
        message = {
            "type": "response.item.create",
            "operation_id": "utterance1",
            "item": ask("Make the reservation")[0],
        }
        socket.send(message)
        await socket.wait_for("session.operation.receipt")
        socket.send({**message, "event_id": "retried"})
        await self.session.fanout.drain()
        await asyncio.sleep(0)
        self.assertEqual(len(self.session.delegation.routes.messages), 1)
        socket.send(
            {**message, "item": {**message["item"], "content": [{"type": "input_text", "text": "Different action"}]}}
        )
        error = await socket.wait_for("error")
        self.assertEqual(error["error"]["param"], "operation_id")

    async def test_receipted_tool_output_and_response_create_continue_once_across_attachments(self):
        socket, authority = await self.start()
        await self.delegate(response("resp1", CALL))
        await asyncio.gather(*self.session.delegation.readers)
        self.core.streams.append(response("resp2", ANSWER))
        output = {
            "type": "response.item.create",
            "event_id": "output-event",
            "operation_id": "output-operation",
            "item": {"type": "function_call_output", "call_id": "call1", "output": "Confirmed."},
        }
        create = {"type": "response.create", "event_id": "create-event", "operation_id": "create-operation"}
        socket.send(output)
        await socket.wait_for(
            "session.operation.receipt", where=lambda event: event["operation_id"] == "output-operation"
        )
        socket.send(create)
        receipt = await socket.wait_for(
            "session.operation.receipt", where=lambda event: event["operation_id"] == "create-operation"
        )
        self.assertEqual(receipt["status"], "recorded")
        await socket.wait_for(
            "response.event", where=lambda event: event.get("event", {}).get("response", {}).get("id") == "resp2"
        )
        await asyncio.gather(*self.session.delegation.readers)
        self.assertEqual(len(self.core.requests), 2)
        self.assertEqual(self.core.requests[-1]["input"], [output["item"]])
        self.assertNotIn("operation_id", self.core.requests[-1])
        await self.drop(socket)
        resumed, _ = await self.resume(authority)
        resumed.send({**create, "event_id": "retry-after-resume"})
        retried = await resumed.wait_for("session.operation.receipt")
        self.assertEqual((retried["operation_id"], retried["status"]), ("create-operation", "recorded"))
        self.assertEqual(len(self.core.requests), 2)

    async def test_receipted_appends_run_once_and_other_response_fields_stay_refused(self):
        socket, _ = await self.start()
        for kind in ("instructions", "thinking", "commentary"):
            command = {
                "type": f"session.{kind}.append",
                "event_id": kind,
                "operation_id": f"append-{kind}",
                "content": f"New {kind} context.",
                "delegation_id": None,
            }
            await self.session.handle(command, self.session.socket.connection)
            await self.session.handle({**command, "event_id": f"retry-{kind}"}, self.session.socket.connection)
            self.assertEqual(self.adapters[0].appended[-1], (kind, command["content"], kind, None))
        self.assertEqual(len(self.adapters[0].appended), 3)
        socket.send({"type": "response.create", "operation_id": "bad-create", "response": {}})
        error = await socket.wait_for("error")
        self.assertEqual(error["error"]["message"], "response.create accepts only type and event_id")
        self.assertEqual(self.core.requests, [])

    async def test_resume_authority_is_not_in_sideband_history_or_public_session_resources(self):
        socket, authority = await self.start()
        self.assertNotIn(authority["token"], str(self.session.snapshot()))
        self.assertNotIn(authority["token"], str(await self.session.history()))
        await self.drop(socket)
        from gateways.live.gateway.websocket import PrimarySocketView

        candidate = PrimarySocketView(FakeWebSocket(), self.session.config.audio.format)
        with self.assertRaises(ProtocolError) as refused:
            await self.session.reconnect.attach(candidate, {**authority, "request_id": "wrong-scope"}, "other")
        self.assertEqual(refused.exception.code, "resume_unavailable")
        self.assertEqual(len(self.adapters), 1)

    async def test_deliberate_and_policy_closes_end_the_call_instead_of_entering_grace(self):
        for code in (1000, 1001, 1008, 1011):
            with self.subTest(code=code):
                socket, _ = await self.start()
                socket.leave(code)
                await asyncio.wait_for(self.runners[-1], 2)
                self.assertEqual(self.session.status, "closed")
                self.assertEqual(self.gateway.resume_leases, {})

    async def test_no_status_asgi_disconnect_preserves_the_same_fenced_logical_call(self):
        socket, authority = await self.start()
        # Uvicorn's SansIO transport reports 1005 for TCP loss after an established handshake.
        socket.leave(1005)
        await asyncio.wait_for(self.runners[-1], 2)
        self.assertEqual((self.session.status, self.session.attachment_state), ("active", "reconnecting"))
        self.assertFalse(self.core.deleted.is_set())
        self.assertFalse(self.core.attachments[-1][1])
        resumed, result = await self.resume(authority)
        self.assertEqual(result["session"]["id"], self.session.id)
        self.assertNotEqual(result["reconnect"]["token"], authority["token"])
        self.assertEqual((len(self.core.requests), len(self.adapters)), (0, 2))
        resumed.send({"type": "session.close"})
        await resumed.wait_for("session.closed")
        self.assertEqual(self.gateway.resume_leases, {})


class MailboxTests(unittest.TestCase):
    def test_web_rtc_and_client_delegation_do_not_negotiate_reconnect(self):
        config = SessionConfig.model_validate(CONFIG)
        with self.assertRaises(ProtocolError):
            config.validate_transport("webrtc")
        with self.assertRaises(ValueError):
            SessionConfig(model="live", delegation={"type": "client", "reconnect_grace_seconds": 30})
