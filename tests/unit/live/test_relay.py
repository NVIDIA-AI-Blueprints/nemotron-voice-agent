# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D103, D105, D107

"""A relay that never closes the session.

Each event goes to the frontend adapter first, then to the client, each step guarded, so one bad event is skipped.
A burst slows the client's view instead of closing the session; a sideband that falls behind is closed.
"""

import asyncio
import unittest
from unittest import mock

from live.events import Fanout, event
from live.events import fanout as fanout_module
from live.gateway.websocket import TRY_AGAIN_LATER, close_dropped, wire_codec
from live.protocol import SessionConfig
from live.server_delegation.passthrough import sanitize_response_event
from live.session import LiveProtocolSession
from tests.unit.live_helpers import Collector, FakeAdapter, responses_config

HI = [{"role": "user", "content": [{"type": "input_text", "text": "Hi."}]}]
CALL = {"type": "function_call", "call_id": "call_1", "name": "get_menu", "arguments": "{}"}
OUTPUT = {"type": "function_call_output", "call_id": "call_1", "output": "{}"}


def core_event(kind, sequence, **fields):
    return {"type": kind, "sequence_number": sequence, **fields}


def created(response_id, sequence=1):
    return core_event("response.created", sequence, response={"id": response_id, "object": "response"})


def completed(response_id, sequence):
    return core_event("response.completed", sequence, response={"id": response_id, "status": "completed"})


def delta(sequence):
    return core_event("response.output_text.delta", sequence, item_id="msg_1", output_index=0, delta=".")


class Stream:
    """A response stream: its events, then (after ``until`` is set, when given) an end, or ``fail`` raised."""

    def __init__(self, events, *, fail=None, until=None):
        self.events, self.fail, self.until = list(events), fail, until

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self.events:
            return self.events.pop(0)
        if self.until is not None:
            await self.until.wait()
        if self.fail is not None:
            raise self.fail
        raise StopAsyncIteration


class StreamCore(FakeAdapter):
    """A delegation core that answers each request with the next of its streams."""

    def __init__(self):
        super().__init__()
        self.streams: list[Stream] = []
        self.deleted = asyncio.Event()

    async def create_response(self, params):
        self.creates += 1
        return self.streams.pop(0)

    async def delete_conversation(self, conversation_id):
        self.deleted.set()
        return await super().delete_conversation(conversation_id)


class Frontend:
    """The frontend adapter's DelegationEvents: it notes each event and each loss, and may fail on one kind."""

    def __init__(self, fails_on=None):
        self.heard: list[tuple[str, object]] = []
        self.fails_on = fails_on

    def on_delegation_event(self, delegation_id, event):
        self.heard.append((delegation_id, event.get("type") if isinstance(event, dict) else event))
        if self.fails_on is not None and event.get("type") == self.fails_on:
            raise RuntimeError("the frontend failed on this event")

    def on_delegation_lost(self, delegation_id):
        self.heard.append((delegation_id, "lost"))


class YieldingClient(Collector):
    """A primary connection whose every send yields to the loop, as a real socket's does."""

    async def send(self, message):
        await asyncio.sleep(0)
        await super().send(message)


class RelayCase(unittest.IsolatedAsyncioTestCase):
    async def start(self, *streams, client=None, frontend=None):
        self.core = StreamCore()
        self.core.streams.extend(streams)
        config = responses_config()
        self.session = LiveProtocolSession(
            config, "websocket", self.core.parts(), core=self.core.core, codec=wire_codec(config.with_wire_format())
        )
        self.client = client or Collector()
        self.key = self.session.subscribe(self.client.send, "primary")
        await self.session.start("s")
        await self.client.wait_for("session.started")
        self.frontend = frontend or Frontend()
        self.session.delegation.listen(self.frontend)
        self.addAsyncCleanup(self.session.close)

    def relayed(self):
        """The core events the client was relayed, in order."""
        return [e["event"] for e in self.client.events if e["type"] == "response.event"]

    async def until(self, predicate, timeout=5.0):
        async with asyncio.timeout(timeout):
            while not predicate():
                await asyncio.sleep(0.005)


class BurstTests(RelayCase):
    async def test_2004_events_in_a_burst_all_arrive_and_the_session_stays_open(self):
        events = [created("resp_1"), *(delta(n) for n in range(2, 2004)), completed("resp_1", 2004)]
        await self.start(Stream(events), client=YieldingClient())
        delegation_id = await self.session.delegation.delegate(HI)
        await self.until(lambda: len(self.relayed()) == 2004)
        self.assertEqual([e["sequence_number"] for e in self.relayed()], list(range(1, 2005)))
        self.assertEqual(len(self.frontend.heard), 2004)
        self.assertEqual(self.session.status, "active")  # 128 events in a burst once closed it
        self.assertEqual(self.client.kinds().count("session.delegation.created"), 1)
        self.assertEqual(self.relayed()[-1]["type"], "response.completed")
        self.assertTrue(all(e["delegation_id"] == delegation_id for e in self.client.events if "delegation_id" in e))

    async def test_a_stalled_client_slows_the_relay_instead_of_closing_the_session(self):
        class StalledClient(Collector):
            def __init__(self):
                super().__init__()
                self.go = asyncio.Event()
                self.go.set()

            async def send(self, message):
                await self.go.wait()
                await super().send(message)

        client = StalledClient()
        events = [created("resp_1"), *(delta(n) for n in range(2, 1500)), completed("resp_1", 1500)]
        await self.start(Stream(events), client=client)
        client.go.clear()  # the client stops reading, for less than the send limit
        await self.session.delegation.delegate(HI)
        await asyncio.sleep(0.2)
        primary = next(iter(self.session.fanout.subscribers.values()))
        self.assertLessEqual(primary.queue.qsize(), primary.queue.maxsize // 2 + 1)  # the relay waits at half
        self.assertEqual(self.session.status, "active")
        client.go.set()
        await self.until(lambda: len(self.relayed()) == 1500)
        self.assertEqual(self.session.status, "active")


class GuardTests(RelayCase):
    async def test_a_bad_event_is_skipped_and_the_rest_still_arrive(self):
        broken_call = core_event("response.output_item.done", 2, item={"type": "function_call", "name": "x"})
        await self.start(Stream([created("resp_1"), broken_call, delta(3), completed("resp_1", 4)]))
        delegation_id = await self.session.delegation.delegate(HI)
        await self.until(lambda: len(self.relayed()) == 3)
        self.assertEqual([e["sequence_number"] for e in self.relayed()], [1, 3, 4])  # the call without call_id is not
        self.assertEqual(len(self.frontend.heard), 4)  # the frontend got every event, as the core wrote it
        self.assertEqual(self.frontend.heard[-1], (delegation_id, "response.completed"))

    async def test_a_frontend_that_fails_on_an_event_still_gets_the_later_ones_and_the_client_gets_them_all(self):
        events = [created("resp_1"), delta(2), delta(3), completed("resp_1", 4)]
        await self.start(Stream(events), frontend=Frontend(fails_on="response.output_text.delta"))
        await self.session.delegation.delegate(HI)
        await self.until(lambda: len(self.relayed()) == 4)
        self.assertEqual(len(self.frontend.heard), 4)

    async def test_the_frontend_is_handed_each_event_before_the_client_is_sent_it(self):
        order = []
        await self.start(Stream([created("resp_1"), completed("resp_1", 2)]))
        hear, publish = self.frontend.on_delegation_event, self.session.publish

        def on_delegation_event(delegation_id, event):
            order.append(("frontend", event["type"]))
            hear(delegation_id, event)

        async def published(message, audience="all"):
            if message["type"] == "response.event":
                order.append(("client", message["event"]["type"]))
            await publish(message, audience)

        self.frontend.on_delegation_event, self.session.publish = on_delegation_event, published
        await self.session.delegation.delegate(HI)
        await self.until(lambda: len(order) == 4)
        self.assertEqual(
            order,
            [
                ("frontend", "response.created"),
                ("client", "response.created"),
                ("frontend", "response.completed"),
                ("client", "response.completed"),
            ],
        )


class LostTests(RelayCase):
    def failed(self):
        return [e for e in self.relayed() if e["type"] == "response.failed"]

    async def test_a_stream_that_ends_before_its_terminal_is_lost_once_on_each_side(self):
        await self.start(Stream([created("resp_1"), core_event("response.in_progress", 2, response={"id": "resp_1"})]))
        delegation_id = await self.session.delegation.delegate(HI)
        await self.until(lambda: self.failed())
        await asyncio.sleep(0.05)
        self.assertEqual([heard for heard in self.frontend.heard if heard[1] == "lost"], [(delegation_id, "lost")])
        [failed] = self.failed()
        self.assertEqual(failed["sequence_number"], 3)
        self.assertEqual(
            (failed["response"]["id"], failed["response"]["status"], failed["response"]["error"]["code"]),
            ("resp_1", "failed", "server_error"),
        )
        self.assertEqual(self.session.status, "active")
        await self.session.close("close_requested")
        kinds = self.client.kinds()
        self.assertLess(kinds.index("response.event"), kinds.index("session.closed"))

    async def test_a_stream_that_fails_is_lost_too(self):
        await self.start(Stream([created("resp_1")], fail=ConnectionError("the core went away")))
        delegation_id = await self.session.delegation.delegate(HI)
        await self.until(lambda: self.failed())
        self.assertIn((delegation_id, "lost"), self.frontend.heard)
        self.assertEqual(len(self.failed()), 1)

    async def test_a_stream_lost_before_any_event_tells_only_the_frontend(self):
        await self.start(Stream([]))
        delegation_id = await self.session.delegation.delegate(HI)
        await self.until(lambda: self.frontend.heard)
        await asyncio.sleep(0.05)
        self.assertEqual(self.frontend.heard, [(delegation_id, "lost")])
        self.assertEqual(self.relayed(), [])  # the client was told of no response to end

    async def test_a_stream_that_reaches_its_terminal_is_not_lost(self):
        await self.start(Stream([created("resp_1"), completed("resp_1", 2)]))
        await self.session.delegation.delegate(HI)
        await self.until(lambda: len(self.relayed()) == 2)
        await asyncio.sleep(0.05)
        self.assertNotIn("lost", [heard for _, heard in self.frontend.heard])
        self.assertEqual(self.failed(), [])

    async def test_a_stream_that_ends_at_close_is_not_lost(self):
        await self.start()
        self.core.streams.append(Stream([created("resp_1")], until=self.core.deleted))
        await self.session.delegation.delegate(HI)
        await self.until(lambda: len(self.relayed()) == 1)
        await self.session.close("close_requested")  # deleting the conversation ends the stream, with no terminal
        self.assertNotIn("lost", [heard for _, heard in self.frontend.heard])
        self.assertEqual(self.failed(), [])
        self.assertEqual(self.client.kinds()[-1], "session.closed")


class AnnouncementTests(RelayCase):
    async def test_a_delegation_is_announced_once_by_the_request_that_started_it(self):
        call = core_event("response.output_item.done", 2, item=CALL)
        await self.start(
            Stream([created("resp_1"), call, completed("resp_1", 3)]),
            Stream([created("resp_2"), completed("resp_2", 2)]),
        )
        delegation_id = await self.session.delegation.delegate(HI)
        await self.until(lambda: len(self.relayed()) == 3)
        await self.session.handle({"type": "response.item.create", "item": OUTPUT}, self.key)
        await self.session.handle({"type": "response.create"}, self.key)
        await self.until(lambda: len(self.relayed()) == 5)
        announced = [e for e in self.client.events if e["type"] == "session.delegation.created"]
        self.assertEqual(
            [(e["delegation"]["id"], e["delegation"]["response_id"]) for e in announced], [(delegation_id, "resp_1")]
        )


class RedactionTests(unittest.TestCase):
    def test_the_snapshot_is_redacted_and_everything_else_is_the_cores(self):
        snapshot = {
            "id": "resp_2",
            "status": "in_progress",
            "previous_response_id": "resp_1",
            "instructions": "Be brief.",
            "tools": [{"type": "function", "name": "get_menu"}],
            "input": [{"role": "user", "content": "Hi."}],
            "output": [{"type": "message"}],
        }
        raw = {"type": "response.in_progress", "sequence_number": 2, "response": snapshot}
        before = {**raw, "response": dict(snapshot)}
        client = sanitize_response_event(raw)
        self.assertEqual(client["response"]["previous_response_id"], "resp_1")  # the taker is named, untouched
        self.assertEqual(
            {name: client["response"].get(name) for name in ("instructions", "tools", "output", "input")},
            {"instructions": None, "tools": [], "output": [], "input": None},
        )
        self.assertNotIn("input", client["response"])
        self.assertEqual(raw, before)  # the core's event, which the frontend holds, is unchanged

    def test_an_event_without_a_snapshot_is_not_copied(self):
        raw = core_event("response.output_item.done", 3, item=dict(CALL))
        self.assertIs(sanitize_response_event(raw), raw)


class FanoutCase(unittest.IsolatedAsyncioTestCase):
    def fanout(self):
        fanout = Fanout(lambda: SessionConfig(model="live-1"), "websocket", lambda: asyncio.sleep(0))
        self.addAsyncCleanup(fanout.close)
        return fanout


class CopyTests(FanoutCase):
    async def test_one_copy_of_an_event_goes_to_every_connection(self):
        fanout, received = self.fanout(), []

        async def keep(message):
            received.append(message)

        for role in ("primary", "sideband", "sideband"):
            fanout.subscribe(keep, role)
        message = event("session.usage.updated", usage={"seconds": 1.0})
        with mock.patch.object(fanout_module, "deepcopy", wraps=fanout_module.deepcopy) as copies:
            await fanout.publish(message)
        await fanout.drain()
        self.assertEqual(copies.call_count, 1)
        self.assertEqual(len(received), 3)
        self.assertTrue(all(copy is received[0] for copy in received))
        message["usage"]["seconds"] = 2.0  # the publisher's own object stays the publisher's
        self.assertEqual(received[0]["usage"]["seconds"], 1.0)


class PacingTests(FanoutCase):
    async def test_waiting_for_room_waits_while_the_primary_queue_is_above_half_and_no_longer(self):
        fanout, go = self.fanout(), asyncio.Event()

        async def stalled(message):
            await go.wait()

        fanout.subscribe(stalled, "primary")
        for _ in range(600):
            await fanout.publish(event("x"))
        waiter = asyncio.create_task(fanout.wait_for_room())
        await asyncio.sleep(0.05)
        self.assertFalse(waiter.done())
        go.set()
        async with asyncio.timeout(2):
            await waiter

    async def test_a_primary_that_closes_lets_every_waiter_go(self):
        fanout = self.fanout()

        async def never(message):
            await asyncio.Event().wait()

        fanout.subscribe(never, "primary")
        for _ in range(600):
            await fanout.publish(event("x"))
        waiter = asyncio.create_task(fanout.wait_for_room())
        await asyncio.sleep(0.01)
        await fanout.close()
        async with asyncio.timeout(2):
            await waiter

    async def test_without_a_primary_nothing_waits(self):
        fanout = self.fanout()
        async with asyncio.timeout(1):
            await fanout.wait_for_room()


class SidebandTests(FanoutCase):
    async def test_a_sideband_that_falls_behind_is_dropped_and_closed_with_the_reason(self):
        fanout, primary, closed = self.fanout(), [], []

        async def keep(message):
            primary.append(message)

        async def never(message):
            await asyncio.Event().wait()

        async def close(reason):
            closed.append(reason)

        fanout.subscribe(keep, "primary")
        key = fanout.subscribe(never, "sideband", close)
        for _ in range(200):
            await fanout.publish(event("x"))
        async with asyncio.timeout(2):
            while not closed:
                await asyncio.sleep(0.005)
        behind = "The sideband connection fell 128 events behind the session; attach again to observe"
        self.assertEqual(closed, [behind])
        self.assertIsNone(fanout.role(key))
        await fanout.drain()
        self.assertEqual(len(primary), 200)  # the primary connection is not touched

    async def test_a_sideband_whose_send_fails_is_closed_with_the_reason(self):
        fanout, closed = self.fanout(), []

        async def broken(message):
            raise ConnectionError("gone")

        async def close(reason):
            closed.append(reason)

        fanout.subscribe(broken, "sideband", close)
        await fanout.publish(event("x"))
        async with asyncio.timeout(2):
            while not closed:
                await asyncio.sleep(0.005)
        self.assertEqual(closed, ["An event could not be sent to the sideband connection; attach again to observe"])

    async def test_the_websocket_edge_closes_a_dropped_sideband_with_try_again_later_and_the_reason(self):
        class Socket:
            async def close(self, code=1000, reason=None):
                self.closed = (code, reason)

        socket = Socket()
        await close_dropped(socket, "behind")
        self.assertEqual(socket.closed, (TRY_AGAIN_LATER, "behind"))


if __name__ == "__main__":
    unittest.main()
