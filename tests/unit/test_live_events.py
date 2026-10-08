# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D103

"""The live protocol's events package on its own: the timeline, permissions and fan-out."""

import asyncio
import unittest

from live.events import Fanout, Timeline, client_event_allowed, error_event, event, server_event_allowed
from live.protocol import ProtocolError, SessionConfig


def webrtc_config(**data_channel):
    return SessionConfig(model="live-1", client={"data_channel": data_channel})


class TimelineTests(unittest.TestCase):
    def test_acknowledgments_are_released_in_order_when_the_clock_reaches_their_end(self):
        timeline = Timeline()
        timeline.hold(lambda start, end: {"id": "a", "start": start, "end": end})
        timeline.advance(100)
        timeline.hold(lambda start, end: {"id": "b", "start": start, "end": end})
        self.assertEqual(timeline.advance(150), [])
        self.assertEqual([ack["id"] for ack in timeline.advance(250)], ["a"])
        self.assertEqual([ack["id"] for ack in timeline.advance(300)], ["b"])
        self.assertEqual(timeline.pending, [])

    def test_the_clock_never_moves_backwards(self):
        timeline = Timeline()
        timeline.advance(500)
        timeline.advance(100)
        self.assertEqual(timeline.input_ms, 500)

    def test_a_full_timeline_refuses_more_acknowledgments_and_discard_empties_it(self):
        timeline = Timeline()
        for _ in range(64):
            timeline.hold(lambda start, end: {"start": start})
        self.assertTrue(timeline.full)
        self.assertEqual(len(timeline.discard()), 64)
        self.assertFalse(timeline.full)


class PermissionTests(unittest.TestCase):
    def test_everything_is_allowed_without_data_channel_permissions(self):
        config = SessionConfig(model="live-1")
        self.assertTrue(server_event_allowed(config, "webrtc", {"type": "session.usage.updated"}, "primary"))
        self.assertTrue(client_event_allowed(config, "webrtc", "session.close", "primary"))

    def test_a_webrtc_primary_sees_only_the_allowed_server_events_but_a_sideband_sees_all(self):
        config = webrtc_config(allowed_server_events=["session.started"])
        self.assertTrue(server_event_allowed(config, "webrtc", {"type": "session.started"}, "primary"))
        self.assertFalse(server_event_allowed(config, "webrtc", {"type": "session.usage.updated"}, "primary"))
        self.assertTrue(server_event_allowed(config, "webrtc", {"type": "session.usage.updated"}, "sideband"))

    def test_a_backend_event_selector_can_name_the_inner_response_event(self):
        config = webrtc_config(
            allowed_server_events=[{"type": "response.event", "response_event": "response.completed"}]
        )
        done = {"type": "response.event", "event": {"type": "response.completed"}}
        delta = {"type": "response.event", "event": {"type": "response.output_text.delta"}}
        self.assertTrue(server_event_allowed(config, "webrtc", done, "primary"))
        self.assertFalse(server_event_allowed(config, "webrtc", delta, "primary"))

    def test_the_client_event_allow_list_applies_to_a_webrtc_primary_only(self):
        config = webrtc_config(allowed_client_events=["session.close"])
        self.assertTrue(client_event_allowed(config, "webrtc", "session.close", "primary"))
        self.assertFalse(client_event_allowed(config, "webrtc", "session.update", "primary"))
        self.assertTrue(client_event_allowed(config, "webrtc", "session.update", "sideband"))


class FanoutTests(unittest.IsolatedAsyncioTestCase):
    async def test_an_event_reaches_only_the_audience_it_is_for(self):
        fanout = Fanout(lambda: SessionConfig(model="live-1"), "websocket", on_primary_lost=lambda: asyncio.sleep(0))
        primary, sideband = [], []

        async def collect(into, message):
            into.append(message)

        fanout.subscribe(lambda m: collect(primary, m), "primary")
        fanout.subscribe(lambda m: collect(sideband, m), "sideband")
        await fanout.publish(event("a"), audience="primary")
        await fanout.publish(event("b"))
        await fanout.drain()
        self.assertEqual([m["type"] for m in primary], ["a", "b"])
        self.assertEqual([m["type"] for m in sideband], ["b"])
        await fanout.close()

    async def test_only_eight_sideband_observers_may_attach(self):
        fanout = Fanout(lambda: SessionConfig(model="live-1"), "websocket", on_primary_lost=lambda: asyncio.sleep(0))

        async def ignore(message):
            return None

        for _ in range(8):
            fanout.subscribe(ignore, "sideband")
        with self.assertRaises(ProtocolError) as raised:
            fanout.subscribe(ignore, "sideband")
        self.assertEqual(raised.exception.code, "rate_limit_exceeded")
        await fanout.close()


class FailingWriterTests(unittest.IsolatedAsyncioTestCase):
    async def test_a_primary_that_fails_closes_everything_including_the_observers(self):
        fanout = Fanout(lambda: SessionConfig(model="live-1"), "websocket", on_primary_lost=lambda: fanout.close())
        seen = []

        async def broken(message):
            raise ConnectionError("peer gone")

        async def observe(message):
            seen.append(message)

        fanout.subscribe(broken, "primary")
        fanout.subscribe(observe, "sideband")
        fanout.subscribe(observe, "sideband")
        await fanout.publish(event("a"))
        async with asyncio.timeout(5):
            while fanout.subscribers:
                await asyncio.sleep(0.01)
        self.assertEqual(fanout.subscribers, {})


class MessageTests(unittest.TestCase):
    def test_every_event_has_a_fresh_id_and_an_error_echoes_the_client_event(self):
        self.assertNotEqual(event("a")["event_id"], event("a")["event_id"])
        body = error_event(ProtocolError("nope", "bad", "field"), "client_1")["error"]
        self.assertEqual((body["code"], body["param"], body["client_event_id"]), ("bad", "field", "client_1"))


if __name__ == "__main__":
    unittest.main()
