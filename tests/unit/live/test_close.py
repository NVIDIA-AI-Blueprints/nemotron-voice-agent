# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D103, D105, D107

"""A session's close, which never skips the core.

Each part has its own close limit. From close step 2 no new request is sent and the frontend gets nothing more, but
the client still gets the ends the core writes before ``session.closed``.
"""

import asyncio
import unittest
from unittest import mock

from live.gateway.websocket import wire_codec
from live.server_delegation import passthrough
from live.session import LiveProtocolSession, lifecycle
from tests.unit.live_helpers import Collector, FakeAdapter, responses_config

HI = [{"role": "user", "content": [{"type": "input_text", "text": "Hi."}]}]


class EndsAtDelete:
    """A response stream that opens, then waits; deleting the conversation ends it, ``incomplete``."""

    def __init__(self, core, response_id):
        self.core, self.response_id, self.step = core, response_id, 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        self.step += 1
        if self.step == 1:
            return {"type": "response.created", "sequence_number": 1, "response": {"id": self.response_id}}
        if self.step == 2:
            await self.core.deleted.wait()
            response = {"id": self.response_id, "status": "cancelled", "incomplete_details": None}
            return {"type": "response.incomplete", "sequence_number": 2, "response": response}
        raise StopAsyncIteration


class OpenStream:
    """A response stream whose next event never comes."""

    def __aiter__(self):
        return self

    async def __anext__(self):
        await asyncio.Event().wait()


class CloseCore(FakeAdapter):
    """A frontend and a core in one: its close may hang, its delete may fail, and its streams end at the delete."""

    def __init__(self):
        super().__init__()
        self.deleted = asyncio.Event()
        self.deletes = 0
        self.close_hangs = False
        self.delete_error: Exception | None = None
        self.create_hangs = False
        self.stream = "ends_at_delete"
        self.sent: list[dict] = []

    async def close(self):
        self.closed = True
        if self.close_hangs:
            await asyncio.Event().wait()

    async def create_conversation(self, metadata):
        if self.create_hangs:
            await asyncio.Event().wait()
        return await super().create_conversation(metadata)

    async def delete_conversation(self, conversation_id):
        self.deletes += 1
        if self.delete_error is not None:
            raise self.delete_error
        self.deleted.set()
        return await super().delete_conversation(conversation_id)

    async def create_response(self, params):
        self.sent.append(params)
        if self.stream == "open":
            return OpenStream()
        return EndsAtDelete(self, f"resp_{len(self.sent)}")


class Frontend:
    def __init__(self):
        self.heard: list[str] = []

    def on_delegation_event(self, delegation_id, event):
        self.heard.append(event["type"])

    def on_delegation_lost(self, delegation_id):
        self.heard.append("lost")


class CloseCase(unittest.IsolatedAsyncioTestCase):
    def build(self, core=None, on_closed=None):
        self.core = core or CloseCore()
        config = responses_config().with_wire_format()
        self.session = LiveProtocolSession(
            config, "websocket", self.core.parts(), core=self.core.core, codec=wire_codec(config), on_closed=on_closed
        )
        self.client = Collector()
        self.session.subscribe(self.client.send, "primary")
        self.frontend = Frontend()
        return self.session

    async def start(self, core=None, on_closed=None):
        self.build(core, on_closed)
        await self.session.start("start_1")
        await self.client.wait_for("session.started")
        self.session.delegation.listen(self.frontend)
        self.addAsyncCleanup(self.session.close)

    async def close(self):
        async with asyncio.timeout(5):  # a close that hangs fails here instead of hanging the test
            await self.session.close("close_requested")


class PartLimitTests(CloseCase):
    async def test_a_hung_frontend_close_still_deletes_the_conversation(self):
        await self.start()
        self.core.close_hangs = True
        with mock.patch.object(lifecycle, "FRONTEND_CLOSE_SECONDS", 0.1):
            await self.close()
        self.assertEqual(self.core.deletes, 1)
        self.assertEqual((self.session.status, self.client.kinds()[-1]), ("closed", "session.closed"))


class EndsBeforeClosedTests(CloseCase):
    async def test_every_delegations_end_reaches_the_client_before_session_closed(self):
        await self.start()
        first = await self.session.delegation.delegate(HI)
        second = await self.session.delegation.delegate(HI)
        async with asyncio.timeout(2):
            while self.client.kinds().count("session.delegation.created") < 2:
                await asyncio.sleep(0.005)
        await self.close()
        kinds = self.client.kinds()
        ends = [
            e["delegation_id"]
            for e in self.client.events
            if e["type"] == "response.event" and e["event"]["type"] == "response.incomplete"
        ]
        self.assertEqual(sorted(ends), sorted([first, second]))
        last_end = max(i for i, e in enumerate(self.client.events) if e["type"] == "response.event")
        self.assertLess(last_end, kinds.index("session.closed"))
        self.assertEqual(self.frontend.heard, ["response.created", "response.created"])  # nothing after step 2


class NoNewWorkTests(CloseCase):
    async def test_no_delegation_is_sent_from_close_step_2(self):
        await self.start()
        self.session.delegation.stop_handing()  # close step 2
        with self.assertRaisesRegex(RuntimeError, "Session is closing"):
            await self.session.delegation.delegate(HI)
        self.assertEqual(self.core.sent, [])

    async def test_a_request_still_queued_at_close_step_2_is_refused_and_never_sent(self):
        await self.start()
        sender = self.session.delegation.sender
        hold = asyncio.Event()
        send = sender.send

        async def held(request):
            await hold.wait()
            return await send(request)

        sender.send = held
        first = asyncio.create_task(self.session.delegation.delegate(HI))
        await asyncio.sleep(0.01)  # queued, its send held
        self.session.delegation.stop_handing()
        hold.set()
        with self.assertRaisesRegex(RuntimeError, "Session is closing"):
            await first
        self.assertEqual(self.core.sent, [])


class OpenLimitTests(CloseCase):
    async def test_a_conversation_the_core_does_not_open_in_time_fails_the_start(self):
        core = CloseCore()
        core.create_hangs = True
        self.build(core)
        with (
            mock.patch.object(passthrough, "CONVERSATION_SECONDS", 0.1),
            self.assertRaisesRegex(RuntimeError, "did not open a conversation within"),
        ):
            async with asyncio.timeout(5):
                await self.session.start("start_1")
        self.assertEqual(self.session.status, "closed")
        self.assertEqual((await self.client.wait_for("session.closed"))["reason"], "connection_lost")


class LeftoverReaderTests(CloseCase):
    async def test_a_failed_delete_stops_the_readers_and_the_session_still_closes(self):
        await self.start()
        self.core.stream = "open"
        self.core.delete_error = RuntimeError("the core is down")
        await self.session.delegation.delegate(HI)
        [reader] = self.session.delegation.readers
        await self.close()
        await asyncio.wait({reader}, timeout=1)
        self.assertTrue(reader.cancelled())
        self.assertEqual((self.session.status, self.client.kinds()[-1]), ("closed", "session.closed"))


class LastStepTests(CloseCase):
    async def test_a_raising_emit_still_closes(self):
        await self.start()
        emit = self.session.emit

        async def broken(kind, **fields):
            if kind == "session.closed":
                raise RuntimeError("the event could not be built")
            await emit(kind, **fields)

        self.session.emit = broken
        await self.close()
        self.assertEqual(self.session.status, "closed")
        self.assertTrue(self.session.closed.is_set())
        self.assertEqual(self.session.fanout.subscribers, {})

    async def test_an_on_closed_that_fails_still_detaches_every_connection(self):
        seen = []

        async def on_closed(session):
            seen.append(session.status)
            raise RuntimeError("the registry failed")

        await self.start(on_closed=on_closed)
        await self.close()
        self.assertEqual(seen, ["closed"])
        self.assertEqual(self.session.fanout.subscribers, {})
        self.assertEqual(self.client.kinds()[-1], "session.closed")


if __name__ == "__main__":
    unittest.main()
