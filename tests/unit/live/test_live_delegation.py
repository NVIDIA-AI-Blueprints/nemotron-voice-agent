# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D103, D105, D107

"""A session's delegation part: in ``responses`` mode on a scripted delegation core, and in client mode.

The frontend adapter delegates through the port it got at start and listens to every delegation's events; the client
sends results, typed messages and ``response.create``. The fake is the frontend and the core in one, and notes the
conversation's create and delete and the frontend's start and close, in order. Both parts keep the ``DelegationPort``
contract (``tests/unit/contracts/delegation_port.py``).
"""

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
from tests.unit.contracts.delegation_port import DelegationPortContract
from tests.unit.live_helpers import ClientSocket, Collector, FakeAdapter, responses_config

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
    """The delegation a request names, and its input."""
    return request["metadata"]["live_delegation_id"], request["input"]


def response(response_id, *items, end="response.completed"):
    """One response's events, numbered from 1: created, each item done, then its end (none when ``end`` is None)."""
    events = [
        {"type": "response.created", "response": {"id": response_id}},
        *({"type": "response.output_item.done", "item": item} for item in items),
        *([{"type": end, "response": {"id": response_id, "status": "completed"}}] if end else []),
    ]
    return [{**event, "sequence_number": number} for number, event in enumerate(events, 1)]


class Stream:
    """A response stream: its events, then (once ``until`` is set) ``then``; then it ends, or raises ``fail``."""

    def __init__(self, events=(), until=None, then=(), fail=None):
        self.events, self.until, self.then, self.fail = list(events), until, list(then), fail

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self.events and self.until is not None:
            await self.until.wait()
            self.events, self.until = self.then, None
        if self.events:
            return self.events.pop(0)
        if self.fail is not None:
            raise self.fail
        raise StopAsyncIteration


class Core(FakeAdapter):
    """The session's frontend adapter and its delegation core in one.

    Each request gets the next of ``streams`` (a list of events, or a :class:`Stream`), then a response that completes
    at once. A step named in ``hold`` waits for that event first; one named in ``fail`` raises its exception, or the
    next of a list (``None`` lets one pass): ``create``, ``delete``, ``start``, ``close``, ``response``, ``check``.
    """

    def __init__(self, *streams):
        super().__init__()
        self.streams, self.requests, self.log, self.hold, self.fail = list(streams), [], [], {}, {}
        self.heard, self.raw, self.typed_texts = [], [], []  # what the frontend heard; the events as it got them
        self.fails_on, self.deleted = None, asyncio.Event()

    async def step(self, name, noted=True):
        if noted:
            self.log.append(name)
        if name in self.hold:
            await self.hold[name].wait()
        failure = self.fail.get(name)
        if isinstance(failure, list):
            failure = failure.pop(0) if failure else None
        if failure is not None:
            raise failure

    async def start(self, session, delegation):
        await self.step("start")
        await super().start(session, delegation)
        delegation.listen(self)

    async def close(self):
        await self.step("close")
        await super().close()

    async def create_conversation(self, metadata):
        await self.step("create")
        self.metadata = metadata
        return await super().create_conversation(metadata)

    async def delete_conversation(self, conversation_id):
        await self.step("delete")
        self.deleted.set()
        return await super().delete_conversation(conversation_id)

    async def create_response(self, params):
        self.requests.append(params)
        await self.step("response", noted=False)
        stream = self.streams.pop(0) if self.streams else response(f"resp_{len(self.requests)}")
        return stream if isinstance(stream, Stream) else Stream(stream)

    async def input_tokens(self, params):
        await self.step("check", noted=False)
        return await super().input_tokens(params)

    def on_delegation_event(self, delegation_id, event):
        self.heard.append((delegation_id, event["type"]))
        self.raw.append(event)
        if event["type"] == self.fails_on:
            raise RuntimeError("the frontend failed on this event")

    def on_delegation_lost(self, delegation_id):
        self.heard.append((delegation_id, "lost"))

    def typed(self, text):
        self.typed_texts.append(text)


class Stalling(Collector):
    """A client that reads once ``go`` is set, and yields to the loop at each send, as a socket does."""

    def __init__(self):
        super().__init__()
        self.go = asyncio.Event()

    async def send(self, message):
        await self.go.wait()
        await asyncio.sleep(0)
        await super().send(message)


class Case(unittest.IsolatedAsyncioTestCase):
    def build(self, *streams, config=None, client=None):
        self.core = Core(*streams)
        config = (config or responses_config()).with_wire_format()
        self.client, self.sideband = client or Collector(), Collector()
        # The primary connection, as the WebSocket edge makes it: the socket the frontend adapter's transport reads.
        self.socket = PrimarySocketView(ClientSocket(self.client), config.audio.format)
        self.session = LiveProtocolSession(
            config, "websocket", self.core.parts(), core=self.core.core, socket=self.socket
        )
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
        """The core events the client was relayed, in order."""
        return [event["event"] for event in self.client.events if event["type"] == "response.event"]

    def ends(self):
        events = self.client.events
        return [(e["event"]["type"], e["delegation_id"]) for e in events if e["type"] == "response.event"]

    def errors(self, collector=None):
        return [event["error"] for event in (collector or self.client).events if event["type"] == "error"]


class ConversationTests(Case):
    async def test_the_conversation_opens_before_the_frontend_starts_and_is_deleted_after_it_closes(self):
        await self.start()
        self.assertEqual(self.core.metadata, {"live_session_id": self.session.id})
        self.assertEqual((self.core.delegation.target, self.core.delegation.reports_events), ("responses", True))
        await self.session.close("close_requested")
        self.assertEqual(self.core.log, ["create", "start", "close", "delete"])
        self.assertEqual(self.client.kinds()[-1], "session.closed")

    async def test_a_start_that_fails_or_does_not_open_in_time_is_reported_and_closes_the_session(self):
        for step, fault, log in (
            ("create", RuntimeError("the core cannot create a conversation"), ["create", "close"]),
            ("create", None, ["create", "close"]),  # held past the open's limit
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

    async def test_a_session_closed_while_it_starts_stays_closed_and_closes_what_started(self):
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
    async def test_a_part_that_fails_or_hangs_at_close_never_keeps_the_session_open(self):
        hung = asyncio.Event()
        for step, fault in (
            ("close", RuntimeError("stuck")),
            ("close", hung),
            ("delete", ConnectionError()),
            ("delete", hung),
        ):
            with self.subTest(step=step, fault=fault):
                await self.start()
                self.core.close_seconds = 0.1  # the core's own close budget, as its client says
                self.core.streams.append(Stream(until=self.core.deleted))  # a response that ends at the delete
                await self.core.delegation.delegate(HI)
                if fault is hung:
                    self.core.hold[step] = hung
                else:
                    self.core.fail[step] = fault
                async with asyncio.timeout(5):
                    await self.session.close("close_requested")
                self.assertEqual(self.core.log[-2:], ["close", "delete"])  # the core is never skipped
                self.assertEqual((self.session.status, self.client.kinds()[-1]), ("closed", "session.closed"))

    async def test_every_delegations_end_reaches_the_client_before_session_closed(self):
        await self.start()
        for number in (1, 2):
            cancelled = {"type": "response.incomplete", "response": {"id": f"resp_{number}", "status": "cancelled"}}
            self.core.streams.append(Stream(response(f"resp_{number}")[:1], until=self.core.deleted, then=[cancelled]))
        self.core.streams.append(Stream(response("resp_3")[:1], until=self.core.deleted))  # ends at the delete, unended
        delegations = [await self.core.delegation.delegate(HI) for _ in range(3)]
        await self.until(lambda: self.client.kinds().count("session.delegation.created") == 3)
        await self.session.close("close_requested")
        self.assertEqual(self.client.kinds()[-1], "session.closed")
        ended = sorted(delegation for kind, delegation in self.ends() if kind == "response.incomplete")
        self.assertEqual(ended, sorted(delegations[:2]))
        self.assertNotIn("response.failed", [kind for kind, _ in self.ends()])  # one that ends at close is not lost
        self.assertEqual([kind for _, kind in self.core.heard], ["response.created"] * 3)  # nothing after close step 2


class RelayTests(Case):
    async def test_a_long_burst_to_a_client_that_stalls_arrives_in_order_and_the_session_stays_open(self):
        client = Stalling()
        await self.start(response("resp_1", *({"type": "message"} for _ in range(2002))), client=client)
        delegation_id = await self.core.delegation.delegate(HI)
        await asyncio.sleep(0.2)  # the client reads nothing, for less than the send limit: the relay waits
        self.assertEqual((self.session.status, self.relayed()), ("active", []))
        client.go.set()
        await self.until(lambda: len(self.relayed()) == 2004)
        self.assertEqual([event["sequence_number"] for event in self.relayed()], list(range(1, 2005)))
        self.assertEqual((len(self.core.heard), self.session.status), (2004, "active"))
        self.assertEqual(self.client.kinds().count("session.delegation.created"), 1)
        self.assertTrue(all(e["delegation_id"] == delegation_id for e in self.client.events if "delegation_id" in e))

    async def test_a_bad_event_or_a_failing_frontend_skips_one_step_and_the_rest_arrive(self):
        broken = {"type": "function_call", "name": "x"}  # a call without its call_id
        for item, fails_on, sequences in (
            (broken, None, [1, 3, 4]),
            ({"type": "message"}, "response.output_item.done", [1, 2, 3, 4]),
        ):
            with self.subTest(fails_on=fails_on):
                await self.start(response("resp_1", item, {"type": "message"}))
                self.core.fails_on = fails_on
                await self.delegate()
                self.assertEqual([event["sequence_number"] for event in self.relayed()], sequences)
                self.assertEqual(len(self.core.heard), 4)  # the frontend got every event, as the core wrote it

    async def test_a_stream_that_ends_before_its_terminal_is_lost_once_on_each_side(self):
        unended = response("resp_1", {"type": "message"}, end=None)
        for name, stream, lost, relayed in (
            ("no terminal", Stream(unended), True, 3),
            ("a failure", Stream(unended, fail=ConnectionError("the core went away")), True, 3),
            ("no event", Stream(), True, 0),  # the client was told of no response to end
            ("a terminal", Stream(response("resp_1")), False, 2),
        ):
            with self.subTest(name):
                await self.start(stream)
                delegation_id = await self.core.delegation.delegate(HI)
                await asyncio.sleep(0.1)
                self.assertEqual(
                    [heard for heard in self.core.heard if heard[1] == "lost"], [(delegation_id, "lost")] * lost
                )
                self.assertEqual((len(self.relayed()), self.session.status), (relayed, "active"))
                if relayed == 3:  # the client's response ends failed, after its last event
                    failed = self.relayed()[-1]
                    self.assertEqual((failed["type"], failed["sequence_number"]), ("response.failed", 3))
                    self.assertEqual((failed["response"]["id"], failed["response"]["status"]), ("resp_1", "failed"))
                    self.assertEqual(failed["response"]["error"]["code"], "server_error")

    async def test_the_client_gets_each_snapshot_redacted_and_the_frontend_gets_it_as_the_core_wrote_it(self):
        snapshot = {"id": "resp_1", "previous_response_id": "resp_0", "instructions": "Be brief.", "input": HI}
        snapshot |= {"tools": [{"type": "function", "name": "get_menu"}], "output": [{"type": "message"}]}
        done = {"type": "response.completed", "response": {**snapshot, "status": "completed"}}
        await self.start([{"type": "response.created", "response": snapshot}, done])
        await self.delegate()
        shown = self.relayed()[0]["response"]
        self.assertEqual(
            (shown["instructions"], shown["tools"], shown["output"], "input" in shown), (None, [], [], False)
        )
        self.assertEqual(shown["previous_response_id"], "resp_0")  # everything else is as the core wrote it
        self.assertEqual(self.core.raw[0]["response"], snapshot)

    async def test_each_delegation_is_announced_once_and_a_clients_own_request_carries_its_client_event_id(self):
        await self.start(response("resp_1", call("c1")))
        delegation_id = await self.delegate()
        await self.send({"type": "response.item.create", "item": output("c1")})
        await self.create("rc7")  # a continuation
        [typed_run] = await self.create("rc8")  # nothing waits: a typed run
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


class ContinuationTests(Case):
    async def test_two_waiting_responses_are_continued_one_at_a_time_in_either_order(self):
        for order in (("c1", "c2"), ("c2", "c1")):
            with self.subTest(order=order):
                await self.start(response("resp_a", call("c1")), response("resp_b", call("c2")))
                owner = {"c1": await self.delegate(), "c2": await self.delegate()}
                for number, call_id in enumerate(order):
                    await self.send({"type": "response.item.create", "item": output(call_id)})
                    [request] = await self.create(f"rc{number}")
                    self.assertEqual(sent(request), (owner[call_id], [output(call_id)]))
                self.assertEqual(self.errors(), [])

    async def test_complete_responses_are_continued_in_call_order_and_the_newest_takes_the_typed_messages(self):
        await self.start(
            response("resp_a", call("c1"), call("c2")),
            response("resp_b", call("c3")),
            response("resp_c", call("c4"), call("c5")),
        )
        first, second, _ = await self.delegate(), await self.delegate(), await self.delegate()
        for item in (output("c3"), typed(), output("c2"), output("c1"), output("c4")):  # c5 is owed: resp_c waits
            await self.send({"type": "response.item.create", "item": item})
        requests = await self.create("rc1")
        self.assertEqual(
            [sent(request) for request in requests],
            [(first, [output("c1"), output("c2")]), (second, [output("c3"), typed()])],
        )
        self.assertEqual(self.errors(), [])
        self.assertEqual(self.core.typed_texts, ["Make it two."])  # the voice reads the typed question, never a result

    async def test_a_create_that_would_continue_nothing_while_results_are_owed_is_refused_and_keeps_the_input(self):
        await self.start(response("resp_a", call("c1"), call("c2")))
        delegation_id = await self.delegate()
        for number, item in enumerate((typed(), output("c1"))):
            await self.send({"type": "response.item.create", "item": item})
            self.assertEqual(await self.create(f"rc{number}"), [])
        refused = [(error["code"], error["client_event_id"]) for error in self.errors()]
        self.assertEqual(
            refused, [("function_call_outputs_required", "rc0"), ("function_call_outputs_required", "rc1")]
        )
        self.assertIn("c2", self.errors()[1]["message"])
        await self.send({"type": "response.item.create", "item": output("c2")})
        [request] = await self.create("rc2")
        self.assertEqual(sent(request), (delegation_id, [output("c1"), output("c2"), typed()]))  # the text was kept
        self.core.streams.append(response("resp_b", {"type": "custom_tool_call", "call_id": "c9", "name": "draw"}))
        await self.delegate()
        self.assertEqual(await self.create("rc3"), [])
        self.assertEqual(self.errors()[-1]["code"], "tool_outputs_required")  # a result of another tool is owed

    async def test_an_item_for_a_call_never_relayed_or_an_unsupported_message_is_refused(self):
        await self.start()
        for item in (
            output("c404"),
            {"type": "mcp_approval_response", "approval_request_id": "x404", "approve": True},
            {**typed(), "content": [{"type": "input_text", "text": "Hi."}, {"type": "input_audio"}]},
        ):
            await self.send({"type": "response.item.create", "event_id": "i1", "item": item})
        await self.until(lambda: len(self.errors()) == 3)
        self.assertEqual(
            [(error["code"], error["param"], error["client_event_id"]) for error in self.errors()],
            [("invalid_value", f"item.{param}", "i1") for param in ("call_id", "approval_request_id", "content")],
        )
        self.assertEqual(self.core.typed_texts, [])  # a message the part refused never reaches the voice

    async def test_every_official_result_item_is_passed_on_for_a_relayed_call(self):
        kinds = ("function_call", "custom_tool_call", "computer_call", "shell_call", "apply_patch_call")
        kinds += ("tool_search_call",)
        answers = ("function_call_output", "custom_tool_call_output", "computer_call_output", "shell_call_output")
        answers += ("apply_patch_call_output", "tool_search_output")  # each answers the call of the same place
        calls = [{"type": kind, "call_id": f"k{number}"} for number, kind in enumerate(kinds)]
        results = [{"type": kind, "call_id": f"k{number}", "output": []} for number, kind in enumerate(answers)]
        calls.append({"type": "mcp_approval_request", "id": "k9", "name": "x", "server_label": "crm"})
        results.append({"type": "mcp_approval_response", "approval_request_id": "k9", "approve": True})
        await self.start(response("resp_a", *calls))
        delegation_id = await self.delegate()
        for item in results:
            await self.send({"type": "response.item.create", "item": item})
        [request] = await self.create("rc1")
        self.assertEqual(sent(request), (delegation_id, results))
        self.assertEqual(self.errors(), [])

    async def test_an_answered_call_owes_nothing_while_its_continuation_runs_and_a_late_repeat_is_dropped(self):
        await self.start(response("resp_a", call("c1")))
        delegation_id, ends = await self.delegate(), asyncio.Event()
        self.core.streams.append(Stream(response("resp_2")[:1], until=ends, then=response("resp_2")[1:]))
        await self.send({"type": "response.item.create", "item": output("c1")})
        [continued] = await self.create("rc1")  # its response runs until ``ends`` is set
        [typed_run] = await self.create("rc2")  # nothing is owed: a bare create is a typed run
        await self.send({"type": "response.item.create", "item": output("c1", "again")})  # still routed
        self.core.fail["response"] = [ValueError("Duplicate result for call c1")]  # the core judges a repeat
        [repeat] = await self.create("rc3")
        self.assertEqual((continued["input"], typed_run["metadata"]["live_typed_run"]), ([output("c1")], "true"))
        self.assertEqual(sent(repeat), (delegation_id, [output("c1", "again")]))
        await self.until(lambda: self.errors())
        [refused] = self.errors()  # the client's mistake, with the core's reason
        self.assertEqual(
            (refused["type"], refused["code"], refused["client_event_id"]),
            ("invalid_request_error", "invalid_value", "rc3"),
        )
        self.assertEqual(refused["message"], "Duplicate result for call c1")
        ends.set()  # the continuation's response ends: the call's route closes, and the refused repeat with it
        await self.until(lambda: self.ends().count(("response.completed", delegation_id)) == 2)
        [last] = await self.create("rc4")
        self.assertEqual((last["input"], last["metadata"].get("live_typed_run")), ([], "true"))


class SenderTests(Case):
    async def test_a_slow_core_holds_neither_the_commands_nor_the_audio(self):
        await self.start()
        self.core.hold = {"response": asyncio.Event(), "check": asyncio.Event()}
        async with asyncio.timeout(1):  # every command returns while the core still holds the request and the check
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
        self.assertEqual((await self.client.wait_for("error"))["error"]["client_event_id"], "u1")  # once it answered
        await self.client.wait_for("session.delegation.created")  # the typed run went out in its turn

    async def test_a_caller_that_gives_up_has_its_sent_work_read_and_its_queued_work_never_sent(self):
        await self.start()
        self.core.hold["response"] = asyncio.Event()
        sent, queued = (asyncio.create_task(self.core.delegation.delegate(HI)) for _ in range(2))
        await self.until(lambda: self.core.requests)
        for caller in (sent, queued):  # a barge-in
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

    async def test_requests_reach_the_core_in_the_order_they_were_made_where_each_was_made(self):
        await self.start(response("resp_a", call("c1")))
        owner = await self.delegate()
        self.core.hold["response"] = asyncio.Event()
        await self.core.session.advance_input_clock(1200)
        first = asyncio.create_task(self.core.delegation.delegate(HI))
        await asyncio.sleep(0)  # made first
        await self.core.session.advance_input_clock(4800)  # the caller spoke on while the request was sent
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

    async def test_a_refused_request_of_the_clients_is_its_senders_error_and_the_next_create_sends_it_again(self):
        await self.start()
        self.core.fail["response"] = [ProtocolError("The queue is full", "rate_limit_exceeded")]
        for text, event_id in (("first", "r1"), ("second", "r2")):  # two typed runs queued at once
            await self.send({"type": "response.item.create", "item": typed(text)}, self.sideband_key)
            await self.send({"type": "response.create", "event_id": event_id}, self.sideband_key)
        error = (await self.sideband.wait_for("error"))["error"]
        self.assertEqual((error["code"], error["client_event_id"]), ("rate_limit_exceeded", "r1"))
        await self.until(lambda: len(self.core.requests) == 2)
        await self.create("r3")  # the refused one kept its own typed message
        self.assertEqual(
            [request["input"] for request in self.core.requests],
            [[typed(text)] for text in ("first", "second", "first")],
        )
        self.assertNotIn("error", self.client.kinds())

    async def test_a_delegation_the_core_refuses_raises_to_its_caller_before_any_event(self):
        await self.start()
        self.core.fail["response"] = ProtocolError("The queue is full", "queue_full")
        with self.assertRaisesRegex(Exception, "The queue is full"):
            await self.core.delegation.delegate(HI)
        await asyncio.sleep(0.05)
        self.assertEqual(self.core.heard, [])

    async def test_requests_still_queued_at_close_are_refused_and_never_sent(self):
        await self.start()
        self.core.hold["response"] = asyncio.Event()
        callers = [asyncio.create_task(self.core.delegation.delegate(HI)) for _ in range(2)]
        await self.until(lambda: self.core.requests)
        await self.session.close("close_requested")
        for caller in [*callers, self.core.delegation.delegate(HI)]:
            with self.assertRaisesRegex(RuntimeError, "Session is closing"):
                await caller
        self.assertEqual(len(self.core.requests), 1)  # only the one the core held got there


class SettingsCheckTests(Case):
    async def update(self, responses, event_id="u1", key=None):
        update = {"type": "session.update", "event_id": event_id, "session": {"delegation": {"responses": responses}}}
        await self.send(update, key)

    async def test_a_refused_setting_is_a_nonfatal_error_for_its_sender_after_session_updated_and_is_kept(self):
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

    async def test_the_settings_a_session_starts_with_are_checked_after_session_started(self):
        config = SessionConfig(model="live-1", delegation={"type": "responses", "responses": {"model": "bad-model"}})
        await self.start(config=config)
        error = (await self.client.wait_for("error"))["error"]
        self.assertEqual((error["code"], error["param"], error["client_event_id"]), (*REFUSED_MODEL, "start_1"))
        self.assertEqual(self.client.kinds()[0], "session.started")

    async def test_settings_the_core_accepts_give_no_error_and_run_nothing(self):
        await self.start()
        tools = [{"type": kind, "name": "get_menu"} if kind == "function" else {"type": kind} for kind in TOOL_TYPES]
        await self.update({"model": "org/other-model", "reasoning": {"effort": "high"}, "tools": tools})
        await self.until(lambda: len(self.core.checked) == 2)
        await asyncio.sleep(0.05)
        self.assertEqual([checked["model"] for checked in self.core.checked], ["org/org/model", "org/other-model"])
        checked = self.core.checked[-1]
        self.assertEqual((checked["reasoning"], checked["input"]), ({"effort": "high", "summary": "auto"}, []))
        self.assertEqual({tool["type"] for tool in checked["tools"]}, TOOL_TYPES)
        self.assertEqual((self.errors(), self.core.requests), ([], []))  # only the settings are checked

    async def test_a_check_that_cannot_answer_or_outlives_the_session_reports_nothing(self):
        for case in ("the core is down", "the session closes", "client mode"):
            with self.subTest(case):
                await self.start(config=SessionConfig(model="live-1") if case == "client mode" else None)
                if case == "the core is down":
                    self.core.fail["check"] = RuntimeError(case)
                if case == "the session closes":
                    self.core.hold["check"] = asyncio.Event()
                change = (
                    {"responses": {"model": "bad-model"}} if case != "client mode" else {"acknowledgement_delay_ms": 5}
                )
                await self.send({"type": "session.update", "event_id": "u1", "session": {"delegation": change}})
                if case == "the session closes":
                    await self.session.close()
                await asyncio.sleep(0.1)
                self.assertNotIn("error", self.client.kinds())
        self.assertEqual(self.core.checked, [])  # the client-mode session checked nothing

    async def test_mcp_credentials_reach_the_core_but_never_come_back(self):
        responses = {"model": "org/org/model", "tools": [MCP]}
        await self.start(config=SessionConfig(model="live-1", delegation={"type": "responses", "responses": responses}))
        await self.update({"instructions": "Be brief."})  # the tools are left alone: the credentials stay
        for kind in ("session.started", "session.updated"):
            self.assertEqual((await self.client.wait_for(kind))["session"]["delegation"]["responses"]["tools"], [SHOWN])
        await self.core.delegation.delegate(HI)
        await self.until(lambda: len(self.core.checked) == 2)
        self.assertEqual((self.core.checked[-1]["tools"], self.core.requests[-1]["tools"]), ([MCP], [MCP]))
        self.assertEqual(self.core.requests[-1]["instructions"], "Be brief.")  # each request carries the settings now


class View:
    """The session's delegation view (``gateways.live.ports.DelegationView``): the input clock, and the events sent."""

    id, responses, acknowledgement_delay_ms = "live_test", None, 0

    def __init__(self, input_ms=0):
        self.input_ms, self.events = input_ms, []

    async def emit(self, kind, **fields):
        self.events.append((kind, fields))

    emit_paced = emit


class PortContractCase(DelegationPortContract):
    async def make_port(self):
        part = self.part()
        await part.open(View())
        self.addAsyncCleanup(part.close)
        return part


class ClientDelegationPortTests(PortContractCase, unittest.IsolatedAsyncioTestCase):
    part = ClientDelegation


class ServerDelegationPortTests(PortContractCase, unittest.IsolatedAsyncioTestCase):
    def part(self):
        return ServerDelegation(Core())  # a core that answers each request with a response that completes at once


class ClientDelegationTests(unittest.IsolatedAsyncioTestCase):
    """Live client mode: the client app does the work; the part announces each delegation and remembers its id."""

    async def test_a_delegation_is_minted_and_remembered_before_it_is_announced_at_the_input_clock(self):
        part, view, known = ClientDelegation(), View(input_ms=1234), []
        emit = view.emit

        async def announce(kind, **fields):
            known.append(part.knows(fields["delegation"]["id"]))  # an answer to it may arrive right away
            await emit(kind, **fields)

        view.emit = announce
        await part.open(view)
        delegation_id = await part.delegate([{"role": "user", "content": "sent nowhere"}])
        self.assertRegex(delegation_id, r"^item_[0-9a-f]{32}$")
        created = {"offset_ms": 1234, "delegation": {"id": delegation_id, "type": "delegation", "target": "client"}}
        self.assertEqual((view.events, known), ([("session.delegation.created", created)], [True]))
        self.assertNotEqual(await part.delegate([]), await part.delegate([]))

    async def test_from_close_step_2_nothing_is_announced_and_a_closed_part_delegates_nothing(self):
        view, part = View(), ClientDelegation()
        await part.open(view)
        await part.open(view)  # safe twice
        earlier = await part.delegate([])
        part.stop_handing()  # close step 2
        later = await part.delegate([])
        self.assertEqual([fields["delegation"]["id"] for _, fields in view.events], [earlier])
        self.assertEqual((part.knows(earlier), part.knows(later)), (True, False))  # an append may name only the first
        for _ in range(2):  # safe twice
            await part.close()
        self.assertFalse(part.knows(earlier))
        closed_first = ClientDelegation()
        await closed_first.close()
        await closed_first.open(View())  # open and close are safe in any order
        for closed in (part, closed_first):
            with self.assertRaises(RuntimeError):
                await closed.delegate([])


if __name__ == "__main__":
    unittest.main()
