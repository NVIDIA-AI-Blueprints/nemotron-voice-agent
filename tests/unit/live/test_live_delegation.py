# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D103, D107

"""A session's delegation part: in ``responses`` mode on a scripted delegation core, and in client mode."""

import asyncio
import base64
import json
import unittest
from unittest import mock

from gateways.live.client_delegation import ClientDelegation
from gateways.live.gateway.websocket import PrimarySocketView
from gateways.live.protocol import ProtocolError, SessionConfig
from gateways.live.protocol.config import TOOL_TYPES
from gateways.live.server_delegation import ServerDelegation, passthrough
from gateways.live.session import LiveProtocolSession, lifecycle
from tests.unit.live_helpers import ClientSocket, Collector, FakeAdapter, Stream, response, responses_config

HI = [{"role": "user", "content": [{"type": "input_text", "text": "Hi."}]}]
AUDIO = base64.b64encode(b"\x01\x00" * 480).decode()
REFUSED_MODEL = ("responses_validation_failed", "session.delegation.responses.model")
SHOWN = {"type": "mcp", "server_label": "crm", "server_url": "https://crm.example/mcp"}
MCP = {**SHOWN, "authorization": "Bearer secret-token", "headers": {"X-Api-Key": "secret-key"}}


def typed(text="Make it two."):
    return {"type": "message", "role": "user", "content": [{"type": "input_text", "text": text}]}


def call(call_id):
    return {"type": "function_call", "call_id": call_id, "name": "get_menu", "arguments": "{}"}


def output(call_id, text="{}"):
    return {"type": "function_call_output", "call_id": call_id, "output": text}


def sent(request):
    """Return the delegation a request names, and its input."""
    return request["metadata"]["live_delegation_id"], request["input"]


class Stalling(Collector):
    """A client that reads once ``go`` is set, and yields to the loop at each send, as a socket does."""

    def __init__(self):
        super().__init__()
        self.go = asyncio.Event()

    async def send(self, message):
        await self.go.wait()
        await asyncio.sleep(0)
        await super().send(message)


class View:
    """The session's delegation view (``gateways.live.ports.DelegationView``): the input clock, and the events sent."""

    id, responses, acknowledgement_delay_ms = "live_test", None, 0

    def __init__(self, input_ms=0):
        self.input_ms, self.events = input_ms, []

    async def emit(self, kind, **fields):
        self.events.append((kind, fields))

    emit_paced = emit


class Case(unittest.IsolatedAsyncioTestCase):
    def build(self, *streams, config=None, client=None):
        self.core = FakeAdapter(streams=streams)
        config = (config or responses_config()).with_wire_format()
        self.client, self.sideband = client or Collector(), Collector()
        self.socket = PrimarySocketView(ClientSocket(self.client), config.audio.format)
        parts, core = self.core.parts(), self.core.core
        self.session = LiveProtocolSession(config, "websocket", parts, core=core, socket=self.socket)
        self.socket.attach(self.session)
        self.key = self.socket.connection
        self.addAsyncCleanup(self.session.close)

    async def start(self, *streams, config=None, client=None):
        self.build(*streams, config=config, client=client)
        await self.session.start("start_1")
        self.sideband_key = self.session.subscribe(self.sideband.send, "sideband")

    async def send(self, message, key=None):
        await self.session.handle(message, key or self.key)

    async def create(self, event_id, key=None):
        """Send ``response.create``; return the requests the core got for it."""
        before = len(self.core.requests)
        await self.send({"type": "response.create", "event_id": event_id}, key)
        await asyncio.sleep(0.05)
        return self.core.requests[before:]

    async def until(self, predicate, timeout=5.0):
        async with asyncio.timeout(timeout):
            while not predicate():
                await asyncio.sleep(0.005)

    async def delegate(self):
        """Delegate as the frontend does, then wait until the client saw the delegation's response complete."""
        delegation_id = await self.core.delegation.delegate(HI)
        await self.until(lambda: ("response.completed", delegation_id) in self.ends())
        return delegation_id

    def relayed(self):
        """Return the core events relayed to the client, in order."""
        return [event["event"] for event in self.client.events if event["type"] == "response.event"]

    def ends(self):
        events = self.client.events
        return [(e["event"]["type"], e["delegation_id"]) for e in events if e["type"] == "response.event"]

    def errors(self, collector=None):
        return [event["error"] for event in (collector or self.client).events if event["type"] == "error"]


class ConversationTests(Case):
    async def test_a_start_that_fails_or_times_out_closes_the_session(self):
        for step, fault, log in (
            ("create", RuntimeError("the core cannot create a conversation"), ["create", "close"]),
            ("create", None, ["create", "close"]),
            ("start", RuntimeError("pipeline failed"), ["create", "start", "close", "delete"]),
        ):
            with self.subTest(step=step, fault=fault), mock.patch.object(passthrough, "CONVERSATION_SECONDS", 0.1):
                self.build()
                if fault is None:
                    self.core.hold[step] = asyncio.Event()
                else:
                    self.core.fail[step] = fault
                with self.assertRaises(RuntimeError):
                    await self.session.start("start_1")
                self.assertEqual((self.session.status, self.core.log), ("closed", log))
                self.assertEqual([error["code"] for error in self.errors()], ["pipeline_error"])
                self.assertEqual(self.client.events[-1]["reason"], "connection_lost")
                self.assertNotIn("session.started", self.client.kinds())

    async def test_a_close_during_the_start_closes_what_started(self):
        for step, log in (
            ("create", ["create", "close", "delete", "close"]),
            ("start", ["create", "start", "close", "delete", "close"]),
        ):
            with self.subTest(step=step):
                self.build()
                self.core.hold[step] = asyncio.Event()
                starting = asyncio.create_task(self.session.start("start_1"))
                await asyncio.sleep(0.05)
                await self.session.close("remote_hangup")
                self.core.hold[step].set()
                await starting
                # The conversation the core created meanwhile is deleted; the frontend closes again after its start.
                self.assertEqual((self.session.status, self.core.log), ("closed", log))
                self.assertNotIn("session.started", self.client.kinds())
                self.assertEqual(self.client.kinds().count("session.closed"), 1)

    @mock.patch.object(lifecycle, "FRONTEND_CLOSE_SECONDS", 0.1)
    @mock.patch.object(passthrough, "RELAY_SECONDS", 0.1)
    async def test_a_part_failing_or_hanging_at_close_never_holds_it_open(self):
        hung = asyncio.Event()
        for step, fault in (
            ("close", RuntimeError("stuck")),
            ("close", hung),
            ("delete", ConnectionError()),
            ("delete", hung),
        ):
            with self.subTest(step=step, fault=fault):
                await self.start()
                self.core.close_seconds = 0.1
                self.core.streams.append(Stream(until=self.core.deleted))
                await self.core.delegation.delegate(HI)
                if fault is hung:
                    self.core.hold[step] = hung
                else:
                    self.core.fail[step] = fault
                async with asyncio.timeout(5):
                    await self.session.close("close_requested")
                self.assertEqual(self.core.log[-2:], ["close", "delete"])
                self.assertEqual((self.session.status, self.client.kinds()[-1]), ("closed", "session.closed"))

    async def test_the_conversation_outlives_the_frontend_at_close(self):
        await self.start()
        self.assertEqual(self.core.metadata, {"live_session_id": self.session.id})
        self.assertEqual((self.core.delegation.target, self.core.delegation.reports_events), ("responses", True))
        for number in (1, 2):
            cancelled = {"type": "response.incomplete", "response": {"id": f"resp_{number}", "status": "cancelled"}}
            self.core.streams.append(Stream(response(f"resp_{number}")[:1], until=self.core.deleted, then=[cancelled]))
        self.core.streams.append(Stream(response("resp_3")[:1], until=self.core.deleted))
        delegations = [await self.core.delegation.delegate(HI) for _ in range(3)]
        await self.until(lambda: self.client.kinds().count("session.delegation.created") == 3)
        await self.session.close("close_requested")
        self.assertEqual(self.core.log, ["create", "start", "close", "delete"])
        self.assertEqual(self.client.kinds()[-1], "session.closed")
        ended = sorted(delegation for kind, delegation in self.ends() if kind == "response.incomplete")
        self.assertEqual(ended, sorted(delegations[:2]))
        self.assertNotIn("response.failed", [kind for kind, _ in self.ends()])
        self.assertEqual([kind for _, kind in self.core.heard], ["response.created"] * 3)


class RelayTests(Case):
    async def test_a_long_burst_to_a_stalled_client_arrives_in_order(self):
        client = Stalling()
        await self.start(response("resp_1", *({"type": "message"} for _ in range(2002))), client=client)
        delegation_id = await self.core.delegation.delegate(HI)
        await asyncio.sleep(0.2)
        self.assertEqual((self.session.status, self.relayed()), ("active", []))
        client.go.set()
        await self.until(lambda: len(self.relayed()) == 2004)
        self.assertEqual([event["sequence_number"] for event in self.relayed()], list(range(1, 2005)))
        self.assertEqual((len(self.core.heard), self.session.status), (2004, "active"))
        self.assertEqual(self.client.kinds().count("session.delegation.created"), 1)
        self.assertTrue(all(e["delegation_id"] == delegation_id for e in self.client.events if "delegation_id" in e))

    async def test_a_bad_event_or_failing_frontend_skips_only_that_step(self):
        broken = {"type": "function_call", "name": "x"}
        for item, fails_on, sequences in (
            (broken, None, [1, 3, 4]),
            ({"type": "message"}, "response.output_item.done", [1, 2, 3, 4]),
        ):
            with self.subTest(fails_on=fails_on):
                await self.start(response("resp_1", item, {"type": "message"}))
                self.core.fails_on = fails_on
                await self.delegate()
                self.assertEqual([event["sequence_number"] for event in self.relayed()], sequences)
                self.assertEqual(len(self.core.heard), 4)

    async def test_a_stream_ended_before_its_terminal_is_lost_once(self):
        unended = response("resp_1", {"type": "message"}, end=None)
        for name, stream, lost, relayed in (
            ("no terminal", Stream(unended), True, 3),
            ("a failure", Stream(unended, fail=ConnectionError("the core went away")), True, 3),
            ("no event", Stream(), True, 0),
            ("a terminal", Stream(response("resp_1")), False, 2),
        ):
            with self.subTest(name):
                await self.start(stream)
                delegation_id = await self.core.delegation.delegate(HI)
                await asyncio.sleep(0.1)
                lost_heard = [heard for heard in self.core.heard if heard[1] == "lost"]
                self.assertEqual(lost_heard, [(delegation_id, "lost")] * lost)
                self.assertEqual((len(self.relayed()), self.session.status), (relayed, "active"))
                if relayed == 3:
                    failed = self.relayed()[-1]
                    self.assertEqual((failed["type"], failed["sequence_number"]), ("response.failed", 3))
                    self.assertEqual((failed["response"]["id"], failed["response"]["status"]), ("resp_1", "failed"))
                    self.assertEqual(failed["response"]["error"]["code"], "server_error")

    async def test_the_client_gets_redacted_snapshots_the_frontend_whole(self):
        snapshot = {"id": "resp_1", "previous_response_id": "resp_0", "instructions": "Be brief.", "input": HI}
        snapshot |= {"tools": [{"type": "function", "name": "get_menu"}], "output": [{"type": "message"}]}
        done = {"type": "response.completed", "response": {**snapshot, "status": "completed"}}
        await self.start([{"type": "response.created", "response": snapshot}, done])
        await self.delegate()
        shown = self.relayed()[0]["response"]
        redacted = (shown["instructions"], shown["tools"], shown["output"], "input" in shown)
        self.assertEqual(redacted, (None, [], [], False))
        self.assertEqual(shown["previous_response_id"], "resp_0")
        self.assertEqual(self.core.raw[0]["response"], snapshot)

    async def test_each_delegation_is_announced_once_with_its_event_id(self):
        await self.start(response("resp_1", call("c1")))
        delegation_id = await self.delegate()
        await self.send({"type": "response.item.create", "item": output("c1")})
        await self.create("rc7")
        [typed_run] = await self.create("rc8")
        await self.until(lambda: len(self.ends()) == 7)
        self.assertEqual((typed_run["input"], typed_run["metadata"]["live_typed_run"]), ([], "true"))
        run = typed_run["metadata"]["live_delegation_id"]
        announced = [event for event in self.client.events if event["type"] == "session.delegation.created"]
        self.assertEqual(
            [(e["delegation"]["id"], e["delegation"]["response_id"], e.get("client_event_id")) for e in announced],
            [(delegation_id, "resp_1", None), (run, "resp_3", "rc8")],
        )
        tagged = {(e["delegation_id"], e.get("client_event_id")) for e in self.client.events if "delegation_id" in e}
        self.assertEqual(tagged, {(delegation_id, None), (delegation_id, "rc7"), (run, "rc8")})


class StatedMetadataTests(Case):
    """What the frontend states about a delegation reaches the core beside the gateway's keys, which always win."""

    async def stated(self, metadata):
        delegation_id = await self.core.delegation.delegate(HI, metadata=metadata)
        await self.until(lambda: ("response.completed", delegation_id) in self.ends())
        return delegation_id, self.core.requests[-1]["metadata"]

    async def test_a_delegation_carries_the_frontends_metadata_beside_the_gateways_keys(self):
        await self.start()
        delegation_id, metadata = await self.stated({"live_reply_unheard": "question"})
        expected = {"live_reply_unheard": "question", "live_session_id": self.session.id}
        self.assertEqual(metadata, {**expected, "live_delegation_id": delegation_id})

    async def test_the_gateways_own_keys_win_over_the_frontends(self):
        await self.start()
        own = {"live_session_id": "other", "live_delegation_id": "item_x", "live_typed_run": "true"}
        delegation_id, metadata = await self.stated(own)
        self.assertEqual(metadata["live_session_id"], self.session.id)
        self.assertEqual(metadata["live_delegation_id"], delegation_id)
        self.assertNotIn("live_typed_run", metadata)  # the gateway's key: a frontend's value never makes a typed run
        self.assertEqual(self.core.requests[-1]["input"], HI)
        observed = []
        passthrough.stated(own, lambda kind, **fields: observed.append((kind, fields["reason"])))
        self.assertEqual(observed, [("delegation.metadata_dropped", "reserved")] * 3)

    async def test_metadata_outside_the_limits_is_dropped_and_the_delegation_still_goes(self):
        await self.start()
        many = {f"k{n}": "v" for n in range(passthrough.MAX_STATED_METADATA + 1)}
        long_key, long_value = "x" * (passthrough.MAX_METADATA_KEY + 1), "v" * (passthrough.MAX_METADATA_VALUE + 1)
        bad = {**many, long_key: "v", "long": long_value, "n": 1}
        _, metadata = await self.stated(bad)  # the delegation still goes, with the pairs within the limits
        self.assertEqual(sorted(key for key in metadata if key.startswith("k")), sorted(list(many)[:-1]))
        observed = []
        passthrough.stated(bad, lambda kind, **fields: observed.append((kind, fields["reason"])))
        reasons = sorted(reason for kind, reason in observed if kind == "delegation.metadata_dropped")
        self.assertEqual(reasons, ["not_text", "too_long", "too_long", "too_many"])

    async def test_continuations_and_typed_runs_carry_no_frontend_metadata(self):
        await self.start(response("resp_1", call("c1")))
        await self.core.delegation.delegate(HI, metadata={"live_reply_unheard": "answer"})
        await self.until(lambda: len(self.ends()) >= 1)
        await self.send({"type": "response.item.create", "item": output("c1")})
        continued, typed_run = (await self.create("rc7"))[0], (await self.create("rc8"))[0]
        for request in (continued, typed_run):
            self.assertNotIn("live_reply_unheard", request["metadata"])


class ContinuationTests(Case):
    async def test_responses_continue_in_call_order_the_newest_with_text(self):
        calls = (("resp_a", call("c1"), call("c2")), ("resp_b", call("c3")), ("resp_c", call("c4"), call("c5")))
        await self.start(*(response(*made) for made in calls))
        first, second, _ = await self.delegate(), await self.delegate(), await self.delegate()
        for item in (output("c3"), typed(), output("c2"), output("c1"), output("c4")):  # c5 is owed: resp_c waits
            await self.send({"type": "response.item.create", "item": item})
        requests = await self.create("rc1")
        continued = [(first, [output("c1"), output("c2")]), (second, [output("c3"), typed()])]
        self.assertEqual(([sent(request) for request in requests], self.errors()), (continued, []))
        self.assertEqual(self.core.typed_texts, ["Make it two."])

    async def test_a_create_with_results_owed_is_refused_and_keeps_input(self):
        await self.start(response("resp_a", call("c1"), call("c2")))
        delegation_id = await self.delegate()
        for number, item in enumerate((typed(), output("c1"))):
            await self.send({"type": "response.item.create", "item": item})
            self.assertEqual(await self.create(f"rc{number}"), [])
        refused = [(error["code"], error["client_event_id"]) for error in self.errors()]
        owed = "function_call_outputs_required"
        self.assertEqual(refused, [(owed, "rc0"), (owed, "rc1")])
        self.assertIn("c2", self.errors()[1]["message"])
        await self.send({"type": "response.item.create", "item": output("c2")})
        [request] = await self.create("rc2")
        self.assertEqual(sent(request), (delegation_id, [output("c1"), output("c2"), typed()]))

    async def test_results_pass_and_unknown_or_bad_items_are_refused(self):
        await self.start(response("resp_a", call("k0"), {"type": "custom_tool_call", "call_id": "k1", "name": "draw"}))
        delegation_id = await self.delegate()
        await self.send({"type": "response.item.create", "item": output("k0")})
        for item in (
            output("c404"),
            {"type": "custom_tool_call_output", "call_id": "k0", "output": "{}"},
            {**typed(), "content": [{"type": "input_text", "text": "Hi."}, {"type": "input_audio"}]},
            *({**typed(), "content": [part]} for part in ({"type": "input_text", "text": 5}, {"type": "input_text"})),
        ):
            await self.send({"type": "response.item.create", "event_id": "i1", "item": item})
        await self.until(lambda: len(self.errors()) >= 5)
        [request] = await self.create("rc1")
        self.assertEqual(sent(request), (delegation_id, [output("k0")]))
        [typed_run] = await self.create("rc2")
        self.assertEqual(typed_run["input"], [])
        params = ("call_id", "type", *["content"] * 3)
        errors = [(error["code"], error["param"], error["client_event_id"]) for error in self.errors()]
        self.assertEqual(errors, [("invalid_value", f"item.{param}", "i1") for param in params])
        self.assertEqual(self.core.typed_texts, [])

    async def test_a_refused_requests_items_are_dropped_and_the_next_goes(self):
        await self.start(response("resp_a", call("c1")))
        delegation_id, ends = await self.delegate(), asyncio.Event()
        for item in (output("c1", "first"), output("c1", "resent"), typed("Hello?")):
            await self.send({"type": "response.item.create", "item": item})
        self.core.fail["response"] = [ValueError("Duplicate result for call c1")]
        await self.create("rc1")
        await self.until(lambda: self.errors())
        [refused] = self.errors()
        reason = ("invalid_request_error", "invalid_value", "rc1", "Duplicate result for call c1")
        self.assertEqual((refused["type"], refused["code"], refused["client_event_id"], refused["message"]), reason)
        self.core.streams.append(Stream(response("resp_2")[:1], until=ends, then=response("resp_2")[1:]))
        for item in (output("c1", "again"), typed()):
            await self.send({"type": "response.item.create", "item": item})
        [continued] = await self.create("rc2")
        self.assertEqual(sent(continued), (delegation_id, [output("c1", "again"), typed()]))
        [typed_run] = await self.create("rc3")
        self.assertEqual(typed_run["metadata"]["live_typed_run"], "true")
        await self.send({"type": "response.item.create", "item": output("c1", "late")})
        self.core.fail["response"] = [ValueError("Duplicate result for call c1")]
        [repeat] = await self.create("rc4")
        self.assertEqual(sent(repeat), (delegation_id, [output("c1", "late")]))
        await self.until(lambda: len(self.errors()) == 2)
        ends.set()  # the continuation's response ends: the call's route closes, and the refused repeat with it
        await self.until(lambda: self.ends().count(("response.completed", delegation_id)) == 2)
        [last] = await self.create("rc5")
        self.assertEqual((last["input"], last["metadata"].get("live_typed_run")), ([], "true"))


class SenderTests(Case):
    async def test_a_slow_core_holds_neither_the_commands_nor_the_audio(self):
        await self.start()
        self.core.hold = {"response": asyncio.Event(), "check": asyncio.Event()}
        async with asyncio.timeout(1):
            await self.send({"type": "response.item.create", "item": typed()})
            await self.send({"type": "response.create", "event_id": "r1"})
            update = {"delegation": {"responses": {"model": "bad-model"}}}
            await self.send({"type": "session.update", "event_id": "u1", "session": update})
            audio = await self.socket.handle(json.dumps({"type": "session.input_audio.append", "audio": AUDIO}))
        self.assertEqual(audio, base64.b64decode(AUDIO))
        await self.client.wait_for("session.updated")
        self.assertEqual(self.errors(), [])
        for held in self.core.hold.values():
            held.set()
        self.assertEqual((await self.client.wait_for("error"))["error"]["client_event_id"], "u1")
        await self.client.wait_for("session.delegation.created")

    async def test_a_caller_that_gives_up_has_only_its_sent_work_read(self):
        await self.start()
        self.core.hold["response"] = asyncio.Event()
        sent, queued = (asyncio.create_task(self.core.delegation.delegate(HI)) for _ in range(2))
        await self.until(lambda: self.core.requests)
        for caller in (sent, queued):
            caller.cancel()
        self.core.hold["response"].set()
        await self.until(lambda: len(self.core.heard) == 2)
        await asyncio.sleep(0.05)
        [request] = self.core.requests
        delegation_id = request["metadata"]["live_delegation_id"]
        self.assertEqual(self.core.heard, [(delegation_id, "response.created"), (delegation_id, "response.completed")])
        for caller in (sent, queued):
            with self.assertRaises(asyncio.CancelledError):
                await caller

    async def test_requests_reach_the_core_in_order_with_their_offsets(self):
        await self.start(response("resp_a", call("c1")))
        owner = await self.delegate()
        self.core.hold["response"] = asyncio.Event()
        await self.core.session.advance_input_clock(1200)
        first = asyncio.create_task(self.core.delegation.delegate(HI))
        await asyncio.sleep(0)
        await self.core.session.advance_input_clock(4800)
        await self.send({"type": "response.item.create", "item": output("c1")})
        await self.send({"type": "response.create", "event_id": "r1"})
        third = asyncio.create_task(self.core.delegation.delegate(HI))
        await asyncio.sleep(0.01)
        self.core.hold["response"].set()
        made = [await first, owner, await third]
        await self.until(lambda: len(self.core.requests) == 4)
        self.assertEqual([request["metadata"]["live_delegation_id"] for request in self.core.requests[1:]], made)
        self.assertEqual(self.core.requests[2]["input"], [output("c1")])
        await self.until(lambda: self.client.kinds().count("session.delegation.created") == 3)
        offsets = {e["delegation"]["id"]: e["offset_ms"] for e in self.client.events if "offset_ms" in e}
        self.assertEqual((offsets[made[0]], offsets[made[2]]), (1200, 4800))

    async def test_a_refused_client_request_is_its_senders_error(self):
        await self.start()
        self.core.fail["response"] = [ProtocolError("The queue is full", "rate_limit_exceeded")]
        for text, event_id in (("first", "r1"), ("second", "r2")):
            await self.send({"type": "response.item.create", "item": typed(text)}, self.sideband_key)
            await self.send({"type": "response.create", "event_id": event_id}, self.sideband_key)
        error = (await self.sideband.wait_for("error"))["error"]
        self.assertEqual((error["code"], error["client_event_id"]), ("rate_limit_exceeded", "r1"))
        await self.until(lambda: len(self.core.requests) == 2)
        await self.create("r3")  # the next create sends the refused one again, with its own typed message
        inputs = [[typed(text)] for text in ("first", "second", "first")]
        self.assertEqual([request["input"] for request in self.core.requests], inputs)
        self.assertNotIn("error", self.client.kinds())

    async def test_a_refused_delegation_raises_before_any_event(self):
        await self.start()
        self.core.fail["response"] = ProtocolError("The queue is full", "queue_full")
        with self.assertRaisesRegex(Exception, "The queue is full"):
            await self.core.delegation.delegate(HI)
        await asyncio.sleep(0.05)
        self.assertEqual(self.core.heard, [])

    async def test_requests_queued_at_close_are_refused_and_never_sent(self):
        await self.start()
        self.core.hold["response"] = asyncio.Event()
        callers = [asyncio.create_task(self.core.delegation.delegate(HI)) for _ in range(2)]
        await self.until(lambda: self.core.requests)
        await self.session.close("close_requested")
        for caller in [*callers, self.core.delegation.delegate(HI)]:
            with self.assertRaisesRegex(RuntimeError, "Session is closing"):
                await caller
        self.assertEqual(len(self.core.requests), 1)


class SettingsCheckTests(Case):
    async def update(self, responses, event_id="u1", key=None):
        update = {"type": "session.update", "event_id": event_id, "session": {"delegation": {"responses": responses}}}
        await self.send(update, key)

    async def test_a_refused_setting_is_a_nonfatal_error_and_is_kept(self):
        await self.start()
        await self.update({"model": "bad-model"})
        error = (await self.client.wait_for("error"))["error"]
        self.assertEqual((error["type"], error["code"], error["param"]), ("invalid_request_error", *REFUSED_MODEL))
        self.assertEqual(error["client_event_id"], "u1")
        self.assertLess(self.client.kinds().index("session.updated"), self.client.kinds().index("error"))
        await self.update({"instructions": "Be brief."}, "u9", self.sideband_key)
        self.assertEqual((await self.sideband.wait_for("error"))["error"]["client_event_id"], "u9")
        await self.core.delegation.delegate(HI)
        self.assertEqual((self.core.requests[-1]["model"], self.session.status), ("bad-model", "active"))
        self.assertEqual(self.client.kinds().count("error"), 1)

    async def test_the_start_settings_are_checked_after_session_started(self):
        config = SessionConfig(model="live-1", delegation={"type": "responses", "responses": {"model": "bad-model"}})
        await self.start(config=config)
        error = (await self.client.wait_for("error"))["error"]
        self.assertEqual((error["code"], error["param"], error["client_event_id"]), (*REFUSED_MODEL, "start_1"))
        self.assertEqual(self.client.kinds()[0], "session.started")

    async def test_accepted_settings_give_no_error_and_run_nothing(self):
        await self.start()
        tools = [{"type": kind, "name": "get_menu"} if kind == "function" else {"type": kind} for kind in TOOL_TYPES]
        await self.update({"model": "org/other-model", "reasoning": {"effort": "high"}, "tools": tools})
        await self.until(lambda: len(self.core.checked) == 2)
        await asyncio.sleep(0.05)
        self.assertEqual([checked["model"] for checked in self.core.checked], ["org/org/model", "org/other-model"])
        checked = self.core.checked[-1]
        self.assertEqual((checked["reasoning"], checked["input"]), ({"effort": "high", "summary": "auto"}, []))
        self.assertEqual({tool["type"] for tool in checked["tools"]}, TOOL_TYPES)
        self.assertEqual((self.errors(), self.core.requests), ([], []))

    async def test_a_check_that_fails_or_outlives_the_session_says_nothing(self):
        for case in ("the core is down", "the session closes", "client mode"):
            with self.subTest(case):
                await self.start(config=SessionConfig(model="live-1") if case == "client mode" else None)
                if case == "the core is down":
                    self.core.fail["check"] = RuntimeError(case)
                if case == "the session closes":
                    self.core.hold["check"] = asyncio.Event()
                bad = {"responses": {"model": "bad-model"}}
                change = {"acknowledgement_delay_ms": 5} if case == "client mode" else bad
                await self.send({"type": "session.update", "event_id": "u1", "session": {"delegation": change}})
                if case == "the session closes":
                    await self.session.close()
                await asyncio.sleep(0.1)
                self.assertNotIn("error", self.client.kinds())
        self.assertEqual(self.core.checked, [])

    async def test_mcp_credentials_reach_the_core_but_never_come_back(self):
        responses = {"model": "org/org/model", "tools": [MCP]}
        await self.start(config=SessionConfig(model="live-1", delegation={"type": "responses", "responses": responses}))
        await self.update({"instructions": "Be brief."})
        for kind in ("session.started", "session.updated"):
            self.assertEqual((await self.client.wait_for(kind))["session"]["delegation"]["responses"]["tools"], [SHOWN])
        await self.core.delegation.delegate(HI)
        await self.until(lambda: len(self.core.checked) == 2)
        self.assertEqual((self.core.checked[-1]["tools"], self.core.requests[-1]["tools"]), ([MCP], [MCP]))
        self.assertEqual(self.core.requests[-1]["instructions"], "Be brief.")


class PartTests(Case):
    async def test_a_delegation_needs_a_listener_and_no_early_event(self):
        # The UI bot's part has no live session; a live session's part also names the session (live_session_id).
        core = FakeAdapter()
        part = ServerDelegation(core)
        await part.open(None)
        self.addAsyncCleanup(part.close)
        with self.assertRaisesRegex(RuntimeError, "listen"):
            await part.delegate(HI)
        part.listen(core)
        delegation_id = await part.delegate(HI)
        self.assertEqual([request["metadata"] for request in core.requests], [{"live_delegation_id": delegation_id}])
        self.assertEqual(core.heard, [])
        await self.until(lambda: len(core.heard) == 2)

    async def test_a_continuation_never_overtakes_the_end_it_continues(self):
        # The waiting response's terminal may come after the continuation is accepted; its reader still relays first.
        first, late, view = response("resp_1", call("c1")), asyncio.Event(), View()
        core = FakeAdapter(streams=[Stream(first[:2], until=late, then=first[2:]), response("resp_2", call("c2"))])
        part = ServerDelegation(core)
        part.listen(core)
        await part.open(view)
        self.addAsyncCleanup(part.close)

        def relayed():
            return [fields["event"]["type"] for kind, fields in view.events if kind == "response.event"]

        await part.delegate(HI)
        await self.until(lambda: "response.output_item.done" in relayed())
        await part.item_create(output("c1"), None)
        await part.response_create("continue")
        await asyncio.sleep(0.05)  # time enough for the continuation's events to overtake, were they not held
        late.set()
        await self.until(lambda: len(relayed()) >= 6)
        each_response = ["response.created", "response.output_item.done", "response.completed"]
        self.assertEqual((relayed(), [kind for _, kind in core.heard]), (each_response * 2, each_response * 2))

    async def test_a_client_delegation_is_known_until_close_step_2(self):
        part, view, known = ClientDelegation(), View(input_ms=1234), []
        emit = view.emit

        async def announce(kind, **fields):
            known.append(part.knows(fields["delegation"]["id"]))  # an answer to it may arrive right away
            await emit(kind, **fields)

        view.emit = announce
        await part.open(view)
        await part.open(view)
        delegation_id = await part.delegate([{"role": "user", "content": "sent nowhere"}])
        self.assertRegex(delegation_id, r"^item_[0-9a-f]{32}$")
        created = {"offset_ms": 1234, "delegation": {"id": delegation_id, "type": "delegation", "target": "client"}}
        self.assertEqual((view.events, known), ([("session.delegation.created", created)], [True]))
        other, another = await part.delegate([]), await part.delegate([])
        self.assertNotEqual(other, another)
        part.stop_handing()
        later = await part.delegate([])
        self.assertEqual([fields["delegation"]["id"] for _, fields in view.events], [delegation_id, other, another])
        self.assertEqual((part.knows(delegation_id), part.knows(later)), (True, False))
        for _ in range(2):
            await part.close()
        self.assertFalse(part.knows(delegation_id))
        closed_first = ClientDelegation()
        await closed_first.close()
        await closed_first.open(View())
        for closed in (part, closed_first):
            with self.assertRaises(RuntimeError):
                await closed.delegate([])

    async def test_a_client_delegation_takes_metadata_and_sends_it_nowhere(self):
        part, view = ClientDelegation(), View()
        await part.open(view)
        delegation_id = await part.delegate([], metadata={"live_reply_unheard": "question"})
        created = view.events[0][1]
        self.assertEqual(created["delegation"], {"id": delegation_id, "type": "delegation", "target": "client"})


if __name__ == "__main__":
    unittest.main()
