# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D103, D105, D107
"""The Event log's ``live-event`` messages: what they carry, and that the tap changes nothing for the call."""

import asyncio
import copy
import json
import unittest
from unittest.mock import patch

from pipecat.processors.frameworks.rtvi.frames import RTVIServerMessageFrame

from nemotron_voice_agent.pipecat.examples.frontend_backend_live.media import live_events
from nemotron_voice_agent.pipecat.examples.frontend_backend_live.media.live_events import (
    MAX_BACKLOG,
    MAX_EVENT_CHARS,
    MAX_ITEMS,
    MAX_TEXT,
    MAX_TOOL_EVENT_CHARS,
    RELAY_CAPABILITY,
    RtviLiveEvents,
    TappedResponsesClient,
    clip,
    compact_event,
    relays_events,
    shrink,
    tapped,
    tool_names,
)
from nemotron_voice_agent.pipecat.shared.delegation_contract import DELEGATION_ID

PROMPT = "You are the backend for an assistant. " * 50
TOOLS = [
    {"type": "function", "name": "flights_search", "parameters": {"type": "object"}},
    {"type": "function", "function": {"name": "sign_in"}},
    {"type": "web_search"},
]
REQUEST = {
    "instructions": PROMPT,
    "tools": TOOLS,
    "input": [{"role": "user", "content": [{"type": "input_text", "text": "Find me a flight to Denver."}]}],
    "conversation": "conv_1",
    "metadata": {DELEGATION_ID: "item_abc"},
    "stream": True,
}


class Clock:
    def __init__(self):
        self.now = 100.0

    def __call__(self):
        return self.now


class Frames:
    """A stand-in for the pipeline worker's ``queue_frame``: it keeps the messages in the order they arrive."""

    def __init__(self):
        self.messages = []

    async def __call__(self, frame):
        assert isinstance(frame, RTVIServerMessageFrame)
        self.messages.append(frame.data)


class FakeStream:
    def __init__(self, events):
        self.events, self.closed = events, False

    def __aiter__(self):
        return self._iterate()

    async def _iterate(self):
        for event in self.events:
            yield event

    async def aclose(self):
        self.closed = True


class FakeClient:
    close_seconds = 7.5

    def __init__(self, events=None, refuse=None):
        self.events, self.refuse, self.requests = events or [], refuse, []

    async def create_conversation(self, metadata):
        return {"id": "conv_1", "object": "conversation"}

    async def update_conversation(self, conversation_id, metadata):
        return {"id": conversation_id}

    async def delete_conversation(self, conversation_id):
        return {"id": conversation_id, "deleted": True}

    async def create_response(self, params):
        self.requests.append(params)
        if self.refuse:
            raise self.refuse
        return FakeStream(self.events)

    async def retrieve_model(self, model):
        return {"id": model}


async def settle():
    for _ in range(5):
        await asyncio.sleep(0)


class BoundTests(unittest.TestCase):
    def test_a_long_string_is_cut_and_says_how_much_was_left_out(self):
        text = "x" * (MAX_TEXT + 25)
        self.assertEqual(clip(text), "x" * MAX_TEXT + "… (+25 chars)")
        self.assertEqual(clip("short"), "short")

    def test_a_long_list_is_cut_and_says_how_many_were_left_out(self):
        clipped = clip(list(range(MAX_ITEMS + 3)))
        self.assertEqual(len(clipped), MAX_ITEMS + 1)
        self.assertEqual(clipped[-1], "… +3 more")

    def test_clipping_leaves_its_argument_alone(self):
        value = {"a": ["y" * 1000, {"b": "z" * 1000}]}
        before = copy.deepcopy(value)
        clip(value)
        self.assertEqual(value, before)

    def test_a_value_that_is_not_json_becomes_a_string(self):
        self.assertEqual(clip({"x": {1, 2}}), {"x": "{1, 2}"})

    def test_a_request_shows_the_prompt_as_a_size_and_its_tools_as_names(self):
        shrunk = shrink(REQUEST)
        self.assertEqual(shrunk["instructions"], f"<{len(PROMPT):,} chars>")
        self.assertEqual(shrunk["tools"], ["flights_search", "sign_in", "web_search"])
        self.assertEqual(shrunk["input"][0]["content"][0]["text"], "Find me a flight to Denver.")
        self.assertEqual(shrunk["conversation"], "conv_1")

    def test_a_response_snapshot_leaves_out_the_input_that_the_request_row_has(self):
        event = {
            "type": "response.created",
            "response": {"id": "resp_1", "instructions": PROMPT, "tools": TOOLS, **{"input": [1, 2]}},
        }
        shown = compact_event(event)["response"]
        self.assertNotIn("input", shown)
        self.assertEqual(shown["instructions"], f"<{len(PROMPT):,} chars>")
        self.assertEqual(shown["tools"], ["flights_search", "sign_in", "web_search"])
        self.assertEqual(shown["id"], "resp_1")

    def test_a_completed_response_keeps_its_output_and_usage(self):
        event = {
            "type": "response.completed",
            "response": {
                "id": "resp_1",
                "output": [{"type": "message", "content": [{"type": "output_text", "text": "Done."}]}],
                "usage": {"input_tokens": 12, "output_tokens": 3},
            },
        }
        shown = compact_event(event)["response"]
        self.assertEqual(shown["output"][0]["content"][0]["text"], "Done.")
        self.assertEqual(shown["usage"], {"input_tokens": 12, "output_tokens": 3})

    def test_an_event_that_stays_too_large_is_replaced_by_a_marker(self):
        wide = {"type": "response.output_item.done", "item": {str(i): "v" * MAX_TEXT for i in range(30)}}
        shown = compact_event(wide)
        self.assertEqual(shown["type"], "response.output_item.done")
        self.assertTrue(shown["truncated"])
        self.assertGreater(shown["chars"], MAX_EVENT_CHARS)

    def test_compacting_leaves_the_event_alone(self):
        event = {"type": "response.created", "response": {"id": "r", "instructions": PROMPT, "tools": TOOLS}}
        before = copy.deepcopy(event)
        compact_event(event)
        self.assertEqual(event, before)

    def test_a_tool_calls_arguments_and_output_stay_whole_however_long(self):
        flights = [{"flight": f"DA {200 + n}", "note": "x" * 400, "price_usd": 300 + n} for n in range(8)]
        output = json.dumps({"status": "ok", "count": len(flights), "flights": flights})
        self.assertGreater(len(output), MAX_TEXT)
        done = {
            "type": "response.output_item.done",
            "item": {
                "type": "mcp_call",
                "id": "mcp_1",
                "server_label": "airline",
                "name": "flights_search",
                "arguments": json.dumps({"origin": "BOS", "destination": "DEN", "date": "2026-11-24"}),
                "output": output,
                "status": "completed",
            },
        }
        shown = compact_event(done)["item"]
        self.assertEqual(shown["output"], output)
        self.assertEqual(json.loads(shown["output"])["flights"][7]["flight"], "DA 207")
        self.assertEqual(json.loads(shown["arguments"])["origin"], "BOS")
        self.assertEqual(
            (shown["server_label"], shown["name"], shown["status"]), ("airline", "flights_search", "completed")
        )

    def test_a_tool_call_event_may_be_larger_than_other_events_but_not_without_bound(self):
        call = {"type": "mcp_call", "name": "flights_search", "output": "o" * (MAX_EVENT_CHARS + 5000)}
        kept = compact_event({"type": "response.output_item.done", "item": call})
        self.assertEqual(len(kept["item"]["output"]), MAX_EVENT_CHARS + 5000)
        huge = {"type": "mcp_call", "name": "flights_search", "output": "o" * (MAX_TOOL_EVENT_CHARS + 10)}
        cut = compact_event({"type": "response.output_item.done", "item": huge})
        self.assertNotIn("truncated", cut)
        self.assertEqual(cut["item"]["name"], "flights_search")
        self.assertTrue(cut["item"]["output"].startswith("ooo"))
        self.assertLessEqual(len(json.dumps(cut)), MAX_TOOL_EVENT_CHARS)

    def test_a_tool_call_too_large_for_the_event_keeps_its_name_and_the_start_of_each_field(self):
        # JSON in a string grows when it is escaped, so two fields of 40,000 characters do not fit in one event.
        quoted = json.dumps({"flights": ['"' * 50] * 800})
        call = {"type": "mcp_call", "name": "flights_search", "arguments": quoted, "output": quoted}
        kept = compact_event({"type": "response.output_item.done", "item": call})["item"]
        self.assertLessEqual(len(json.dumps({"item": kept})), MAX_TOOL_EVENT_CHARS)
        self.assertEqual(kept["name"], "flights_search")
        for field in ("arguments", "output"):
            with self.subTest(field):
                self.assertTrue(kept[field].startswith('{"flights"'))
                self.assertRegex(kept[field], r"… \(\+\d+ chars\)$")

    def test_a_completed_response_shows_its_tool_calls_as_sizes_not_a_second_copy(self):
        output = json.dumps({"status": "ok", "flights": ["a" * 100] * 30})
        call = {
            "type": "mcp_call",
            "name": "flights_search",
            "arguments": "{}",
            "output": output,
            "status": "completed",
        }
        event = {"type": "response.completed", "response": {"id": "resp_1", "output": [call, {"type": "message"}]}}
        shown = compact_event(event)["response"]["output"]
        self.assertEqual(shown[0]["output"], f"<{len(output):,} chars>")
        self.assertEqual(shown[0]["arguments"], "<2 chars>")
        self.assertEqual((shown[0]["name"], shown[0]["status"]), ("flights_search", "completed"))
        self.assertEqual(shown[1], {"type": "message"})
        self.assertEqual(event["response"]["output"][0]["output"], output)

    def test_any_other_item_is_still_clipped(self):
        other = {"type": "message", "content": [{"type": "output_text", "text": "t" * (MAX_TEXT + 50)}]}
        shown = compact_event({"type": "response.output_item.done", "item": other})["item"]
        self.assertIn("(+50 chars)", shown["content"][0]["text"])

    def test_tool_names_read_each_shape(self):
        self.assertEqual(tool_names(TOOLS), ["flights_search", "sign_in", "web_search"])
        self.assertEqual(tool_names(None), [])


class MessageTests(unittest.IsolatedAsyncioTestCase):
    async def test_events_wait_for_the_client_and_then_go_out_in_order(self):
        clock, frames = Clock(), Frames()
        log = RtviLiveEvents(clock)
        log.api("conversations.create", id="conv_1")
        clock.now = 101.5
        tag = log.request(REQUEST)
        log.event(tag, {"type": "response.created", "response": {"id": "resp_1"}})
        await settle()
        self.assertEqual(frames.messages, [])

        log.start(frames)
        await settle()
        self.assertEqual([m["seq"] for m in frames.messages], [1, 2, 3])
        self.assertEqual(
            [m["kind"] for m in frames.messages], ["conversations.create", "responses.create", "response.created"]
        )
        self.assertEqual([m["direction"] for m in frames.messages], ["out", "out", "in"])
        self.assertEqual([m["t"] for m in frames.messages], [0.0, 1.5, 1.5])

        log.event(tag, {"type": "response.completed", "response": {"id": "resp_1"}})
        await settle()
        self.assertEqual(frames.messages[-1]["seq"], 4)
        log.close()

    async def test_a_message_carries_the_fields_the_ui_reads(self):
        frames = Frames()
        log = RtviLiveEvents(Clock())
        log.start(frames)
        tag = log.request(REQUEST)
        log.event(tag, {"type": "response.output_text.delta", "item_id": "msg_1", "delta": "Hello"})
        await settle()
        request, event = frames.messages
        self.assertEqual(set(request), {"type", "v", "seq", "t", "direction", "kind", "stream", "delegation", "event"})
        self.assertEqual((request["type"], request["v"]), ("live-event", 1))
        self.assertEqual((request["stream"], request["delegation"]), ("r1", "item_abc"))
        self.assertEqual((event["stream"], event["delegation"]), ("r1", "item_abc"))
        self.assertEqual(event["event"], {"type": "response.output_text.delta", "item_id": "msg_1", "delta": "Hello"})
        json.dumps(frames.messages)
        log.close()

    async def test_each_request_starts_a_new_stream_label(self):
        log = RtviLiveEvents(Clock())
        first, second = log.request(REQUEST), log.request({**REQUEST, "metadata": {}})
        self.assertEqual((first.label, first.delegation), ("r1", "item_abc"))
        self.assertEqual((second.label, second.delegation), ("r2", None))

    async def test_the_backlog_keeps_the_newest_events_and_seq_shows_the_gap(self):
        frames = Frames()
        log = RtviLiveEvents(Clock())
        tag = log.request(REQUEST)
        for index in range(MAX_BACKLOG + 10):
            log.event(tag, {"type": "response.output_text.delta", "delta": str(index)})
        log.start(frames)
        await settle()
        self.assertEqual(len(frames.messages), MAX_BACKLOG)
        self.assertEqual(frames.messages[-1]["seq"], MAX_BACKLOG + 11)
        self.assertGreater(frames.messages[0]["seq"], 1)
        log.close()

    async def test_nothing_is_sent_after_close(self):
        frames = Frames()
        log = RtviLiveEvents(Clock())
        log.start(frames)
        log.close()
        log.api("conversations.delete", id="conv_1")
        await settle()
        self.assertEqual(frames.messages, [])

    async def test_a_failing_queue_never_stops_the_sender(self):
        sent = []

        async def queue_frame(frame):
            if len(sent) == 0:
                sent.append("failed")
                raise RuntimeError("the pipeline is gone")
            sent.append(frame.data["seq"])

        log = RtviLiveEvents(Clock())
        log.start(queue_frame)
        log.api("conversations.create")
        await settle()
        log.api("conversations.update")
        await settle()
        self.assertEqual(sent, ["failed", 2])
        log.close()


class TapTests(unittest.IsolatedAsyncioTestCase):
    async def test_the_stream_yields_the_same_events_in_the_same_order(self):
        events = [
            {"type": "response.created", "response": {"id": "resp_1", "instructions": PROMPT}},
            {"type": "response.output_text.delta", "delta": "Hi"},
            {"type": "response.completed", "response": {"id": "resp_1"}},
        ]
        inner = FakeClient(events)
        frames, log = Frames(), RtviLiveEvents(Clock())
        log.start(frames)
        client = TappedResponsesClient(inner, log)

        stream = await client.create_response(REQUEST)
        seen = [event async for event in stream]
        await settle()

        self.assertEqual(len(seen), 3)
        # Every event is the object the core produced: the log only reads it, and the prompt is still whole.
        self.assertTrue(all(a is b for a, b in zip(seen, events, strict=True)))
        self.assertEqual(seen[0]["response"]["instructions"], PROMPT)
        self.assertEqual(inner.requests, [REQUEST])
        self.assertEqual(
            [m["kind"] for m in frames.messages],
            ["responses.create", "response.created", "response.output_text.delta", "response.completed"],
        )
        self.assertEqual(frames.messages[1]["event"]["response"]["instructions"], f"<{len(PROMPT):,} chars>")
        log.close()

    async def test_a_log_that_fails_never_breaks_the_stream(self):
        events = [{"type": "response.created"}, {"type": "response.completed"}]
        log = RtviLiveEvents(Clock())

        def broken(*_args, **_kwargs):
            raise RuntimeError("the log is broken")

        log.event = broken
        client = TappedResponsesClient(FakeClient(events), log)
        stream = await client.create_response(REQUEST)
        self.assertEqual([event async for event in stream], events)

    async def test_a_log_that_cannot_record_a_request_still_returns_the_core_stream(self):
        inner = FakeClient([{"type": "response.created"}])
        log = RtviLiveEvents(Clock())

        def broken(*_args, **_kwargs):
            raise RuntimeError("the log is broken")

        log.request = broken
        client = TappedResponsesClient(inner, log)
        stream = await client.create_response(REQUEST)
        self.assertEqual([event async for event in stream], [{"type": "response.created"}])

    async def test_a_refused_request_is_logged_and_raised_again(self):
        refusal = ValueError("a repeated result")
        frames, log = Frames(), RtviLiveEvents(Clock())
        log.start(frames)
        client = TappedResponsesClient(FakeClient(refuse=refusal), log)
        with self.assertRaises(ValueError) as caught:
            await client.create_response(REQUEST)
        self.assertIs(caught.exception, refusal)
        await settle()
        self.assertEqual([m["kind"] for m in frames.messages], ["responses.create", "error"])
        self.assertEqual(frames.messages[1]["event"]["message"], "a repeated result")
        log.close()

    async def test_conversation_calls_are_logged_and_their_results_returned(self):
        frames, log = Frames(), RtviLiveEvents(Clock())
        log.start(frames)
        client = TappedResponsesClient(FakeClient(), log)
        self.assertEqual((await client.create_conversation({"a": "b"}))["id"], "conv_1")
        self.assertEqual((await client.update_conversation("conv_1", {}))["id"], "conv_1")
        self.assertTrue((await client.delete_conversation("conv_1"))["deleted"])
        await settle()
        self.assertEqual(
            [m["kind"] for m in frames.messages],
            ["conversations.create", "conversations.update", "conversations.delete"],
        )
        self.assertEqual(frames.messages[2]["event"], {"id": "conv_1", "deleted": True})
        log.close()

    async def test_other_endpoints_and_attributes_are_the_wrapped_clients_own(self):
        client = TappedResponsesClient(FakeClient(), RtviLiveEvents(Clock()))
        self.assertEqual(client.close_seconds, 7.5)
        self.assertEqual(await client.retrieve_model("m"), {"id": "m"})
        with self.assertRaises(AttributeError):
            client.no_such_endpoint  # noqa: B018

    async def test_closing_the_tapped_stream_closes_the_core_stream(self):
        inner = FakeClient([])
        client = TappedResponsesClient(inner, RtviLiveEvents(Clock()))
        stream = await client.create_response(REQUEST)
        await stream.aclose()
        self.assertTrue(stream._inner.closed)


class GateTests(unittest.TestCase):
    def registry(self, capabilities):
        return patch.object(live_events.examples_registry, "find", return_value={"capabilities": capabilities})

    def test_the_relay_is_on_when_the_entry_lists_the_event_log(self):
        with self.registry(["webcam", RELAY_CAPABILITY]):
            self.assertTrue(relays_events("frontend-backend-live"))

    def test_the_relay_is_off_without_it(self):
        for capabilities in ([], ["webcam", "attachments"], ["airline_desk"]):
            with self.subTest(capabilities), self.registry(capabilities):
                self.assertFalse(relays_events("frontend-backend-live"))

    def test_the_gate_reads_the_entry_even_when_the_deployment_locks_another_example(self):
        with self.registry([]) as find:
            relays_events("frontend-backend-live")
        find.assert_called_once_with("frontend-backend-live", ignore_lock=True)

    def test_the_registry_asks_for_the_relay_only_where_the_ui_needs_it(self):
        import nemotron_voice_agent.pipecat.runtime.examples_registry as registry

        self.assertIn(RELAY_CAPABILITY, registry.find("frontend-backend-live", ignore_lock=True)["capabilities"])
        for key in registry.EXAMPLES:
            if key != "frontend-backend-live":
                with self.subTest(key):
                    self.assertNotIn(RELAY_CAPABILITY, registry.EXAMPLES[key]["capabilities"])

    def test_without_a_log_the_client_is_returned_untouched(self):
        client = object()
        self.assertIs(tapped(client, None), client)
        self.assertIsInstance(tapped(client, RtviLiveEvents(Clock())), TappedResponsesClient)


class ModuleTests(unittest.TestCase):
    def test_the_message_type_is_the_one_the_client_reads(self):
        self.assertEqual(live_events.MESSAGE, "live-event")
        self.assertEqual(live_events.VERSION, 1)


if __name__ == "__main__":
    unittest.main()
