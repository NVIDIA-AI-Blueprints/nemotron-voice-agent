# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D103, D105, D107

"""An ordered, cancel-safe sender off the command lock.

The server delegation sends every request through one sender task, in the order the requests were made, so a slow
core never holds the caller's audio.
"""

import asyncio
import base64
import time
import unittest

from live.gateway.websocket import wire_codec
from live.session import LiveProtocolSession
from tests.unit.live_helpers import Collector, FakeAdapter, responses_config

HI = [{"role": "user", "content": [{"type": "input_text", "text": "Hi."}]}]
TYPED = {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "When do you open?"}]}


class Stream:
    """A response stream with the given events."""

    def __init__(self, events):
        self.events = list(events)

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self.events:
            raise StopAsyncIteration
        return self.events.pop(0)


def answered(response_id):
    return [
        {"type": "response.created", "response": {"id": response_id, "status": "queued"}},
        {"type": "response.completed", "response": {"id": response_id, "status": "completed", "output": []}},
    ]


class Refusal(Exception):
    def __init__(self, message, code):
        super().__init__(message)
        self.code = code


class ScriptedCore(FakeAdapter):
    """A delegation core that answers each request after ``delay`` seconds, notes the order, and may refuse."""

    def __init__(self, delay=0.0, accepted_then=0.0):
        super().__init__()
        self.delay, self.accepted_then = delay, accepted_then
        self.sent: list[dict] = []  # each request's metadata and input, in the order the core accepted them
        self.refuse_typed: str | None = None  # a code to refuse typed runs with
        self.accepted = asyncio.Event()

    async def create_response(self, params):
        await asyncio.sleep(self.delay)
        if self.refuse_typed and params["metadata"].get("live_typed_run"):
            raise Refusal("A typed run must wait for the running work", self.refuse_typed)
        self.sent.append(params)
        self.accepted.set()
        await asyncio.sleep(self.accepted_then)  # accepted, the answer still on its way (over a network)
        return Stream(answered(f"resp_{len(self.sent)}"))


class Listener:
    def __init__(self):
        self.events: list[tuple[str, str]] = []

    def on_delegation_event(self, delegation_id, event):
        self.events.append((delegation_id, event["type"]))

    def on_delegation_lost(self, delegation_id):
        self.events.append((delegation_id, "lost"))


class SenderCase(unittest.IsolatedAsyncioTestCase):
    async def start(self, core):
        self.core = core
        config = responses_config()
        self.session = LiveProtocolSession(
            config, "websocket", self.core.parts(), core=self.core.core, codec=wire_codec(config.with_wire_format())
        )
        self.primary, self.sideband = Collector(), Collector()
        self.primary_key = self.session.subscribe(self.primary.send, "primary")
        await self.session.start("s")
        await self.primary.wait_for("session.started")
        self.sideband_key = self.session.subscribe(self.sideband.send, "sideband")
        self.listener = Listener()
        self.session.delegation.listen(self.listener)
        self.addAsyncCleanup(self.session.close)

    async def until(self, predicate, timeout=3.0):
        async with asyncio.timeout(timeout):
            while not predicate():
                await asyncio.sleep(0.005)


class OffTheLockTests(SenderCase):
    async def test_a_slow_core_adds_no_delay_to_the_callers_audio(self):
        await self.start(ScriptedCore(delay=0.3))
        await self.session.handle({"type": "response.item.create", "item": TYPED}, self.primary_key)
        began = time.monotonic()
        await self.session.handle({"type": "response.create", "event_id": "r1"}, self.primary_key)
        chunk = base64.b64encode(b"\x01\x00" * 480).decode()
        await self.session.handle({"type": "session.input_audio.append", "audio": chunk}, self.primary_key)
        self.assertEqual(len(self.core.audio), 1)
        self.assertLess(time.monotonic() - began, 0.1)  # the audio does not wait out the core's 300 ms
        await self.until(lambda: self.core.sent)  # the typed run still reaches the core, in turn


class CancelTests(SenderCase):
    async def test_a_caller_that_gives_up_after_acceptance_still_has_its_work_read(self):
        await self.start(ScriptedCore(accepted_then=0.05))
        caller = asyncio.create_task(self.session.delegation.delegate(HI))
        await self.core.accepted.wait()
        caller.cancel()  # a barge-in after the core accepted, before delegate() returned
        with self.assertRaises(asyncio.CancelledError):
            await caller
        await self.until(lambda: len(self.listener.events) == 2)
        delegation_id = self.core.sent[0]["metadata"]["live_delegation_id"]
        expected = [(delegation_id, "response.created"), (delegation_id, "response.completed")]
        self.assertEqual(self.listener.events, expected)

    async def test_a_caller_that_gives_up_before_its_request_is_sent_sends_nothing(self):
        await self.start(ScriptedCore(delay=0.1))
        first = asyncio.create_task(self.session.delegation.delegate(HI))
        await asyncio.sleep(0.01)  # the first request is being sent
        second = asyncio.create_task(self.session.delegation.delegate(HI))
        await asyncio.sleep(0.01)
        second.cancel()  # still queued behind the first
        await first
        with self.assertRaises(asyncio.CancelledError):
            await second
        await asyncio.sleep(0.15)
        self.assertEqual(len(self.core.sent), 1)


class OrderTests(SenderCase):
    async def test_requests_reach_the_core_in_the_order_they_were_made(self):
        await self.start(ScriptedCore(delay=0.02))
        call = {"type": "function_call", "call_id": "call_1", "name": "get_menu", "arguments": "{}"}
        self.session.delegation.routes.note_call("item_old", "resp_old", call)
        output = {"type": "function_call_output", "call_id": "call_1", "output": "{}"}
        first = asyncio.create_task(self.session.delegation.delegate(HI))
        await asyncio.sleep(0)  # made first
        await self.session.handle({"type": "response.item.create", "item": output}, self.primary_key)
        await self.session.handle({"type": "response.create", "event_id": "r1"}, self.primary_key)
        third = asyncio.create_task(self.session.delegation.delegate(HI))
        await asyncio.gather(first, third)
        await self.until(lambda: len(self.core.sent) == 3)
        self.assertEqual(
            [request["metadata"]["live_delegation_id"] for request in self.core.sent],
            [await first, "item_old", await third],
        )
        self.assertEqual(self.core.sent[1]["input"], [output])


class RefusalTests(SenderCase):
    async def test_a_refusal_reaches_the_connection_that_sent_the_command_with_its_client_event_id(self):
        core = ScriptedCore()
        core.refuse_typed = "response_in_progress"
        await self.start(core)
        await self.session.handle({"type": "response.item.create", "item": TYPED}, self.sideband_key)
        await self.session.handle({"type": "response.create", "event_id": "r9"}, self.sideband_key)
        error = (await self.sideband.wait_for("error"))["error"]
        self.assertEqual((error["code"], error["client_event_id"]), ("response_in_progress", "r9"))
        self.assertNotIn("error", self.primary.kinds())
        [kept] = self.session.delegation.routes.requests()  # kept, to be sent again
        self.assertEqual((kept.delegation_id, kept.input), (None, [TYPED]))

    async def test_an_older_refused_request_keeps_its_own_typed_message_when_a_newer_one_is_accepted(self):
        class RefusesTheFirst(ScriptedCore):
            async def create_response(self, params):
                if not self.sent and not getattr(self, "refused", False):
                    self.refused = True
                    await asyncio.sleep(self.delay)
                    raise Refusal("The queue is full", "rate_limit_exceeded")
                return await super().create_response(params)

        def typed(text):
            return {"type": "message", "role": "user", "content": [{"type": "input_text", "text": text}]}

        await self.start(RefusesTheFirst(delay=0.05))
        for number, text in enumerate(("first", "second"), start=1):  # two requests queued at once
            await self.session.handle({"type": "response.item.create", "item": typed(text)}, self.primary_key)
            await self.session.handle({"type": "response.create", "event_id": f"r{number}"}, self.primary_key)
        error = (await self.primary.wait_for("error"))["error"]
        await self.until(lambda: self.core.sent)
        await asyncio.sleep(0.05)
        self.assertEqual((error["code"], error["client_event_id"]), ("rate_limit_exceeded", "r1"))
        self.assertEqual([request["input"] for request in self.core.sent], [[typed("second")]])
        [kept] = self.session.delegation.routes.requests()  # the refused one stays, to be sent again; the sent one goes
        self.assertEqual((kept.delegation_id, kept.input), (None, [typed("first")]))

    async def test_a_delegation_the_core_refuses_raises_to_its_caller(self):
        class Refusing(ScriptedCore):
            async def create_response(self, params):
                raise Refusal("The queue is full", "queue_full")

        await self.start(Refusing())
        with self.assertRaisesRegex(Exception, "The queue is full"):
            await self.session.delegation.delegate(HI)
        self.assertEqual(self.listener.events, [])


class OffsetTests(SenderCase):
    async def test_offset_ms_is_where_the_delegation_was_made(self):
        await self.start(ScriptedCore(delay=0.05))
        await self.session.advance_input_clock(1200)
        caller = asyncio.create_task(self.session.delegation.delegate(HI))
        await asyncio.sleep(0)
        await self.session.advance_input_clock(4800)  # the caller spoke on while the request was sent
        await caller
        created = await self.primary.wait_for("session.delegation.created")
        self.assertEqual(created["offset_ms"], 1200)


class CloseTests(SenderCase):
    async def test_requests_still_queued_at_close_are_refused_and_never_sent(self):
        await self.start(ScriptedCore(delay=0.2))
        first = asyncio.create_task(self.session.delegation.delegate(HI))
        await asyncio.sleep(0.01)
        second = asyncio.create_task(self.session.delegation.delegate(HI))
        await asyncio.sleep(0.01)
        await self.session.close("close_requested")
        for caller in (first, second):
            with self.assertRaisesRegex(RuntimeError, "Session is closing"):
                await caller
        self.assertEqual(self.core.sent, [])


if __name__ == "__main__":
    unittest.main()
