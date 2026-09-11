# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""Focused tests for OpenAI Realtime scaling-client measurements."""

from __future__ import annotations

import asyncio
import json
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

SCALING_PERF_DIR = Path(__file__).resolve().parents[2] / "benchmarking_tools" / "scaling-perf"
sys.path.insert(0, str(SCALING_PERF_DIR))

from benchmark import (
    REALTIME_RESPONSE_METRIC_KEYS,
    SERVER_METRIC_KEYS,
    PerfClient,
    _aggregate_run_dir,
    _aggregate_suite_dir,
    _benchmark_result_is_accepted,
    _run_aggregate_run,
    _run_aggregate_suite,
    _suite_results_are_accepted,
)
from openai_realtime_client import (
    HttpToolHandler,
    HttpToolInvocation,
    OpenAIRealtimePerfClient,
    RealtimeClientError,
    RealtimeClientToolsConfig,
    ScriptedToolHandler,
    _contains_user_visible_text,
    _ReceivedEvent,
    _TurnInputTrace,
    load_client_tools_config,
)


class _MemoryLogger:
    async def log(self, message: str) -> None:
        pass


class _RecordingWebSocket:
    def __init__(self) -> None:
        self.events: list[dict] = []

    async def send(self, message: str) -> None:
        self.events.append(json.loads(message))


class _FakeHttpResponse:
    def __init__(self, *, status: int, body: bytes) -> None:
        self.status = status
        self.content = self
        self.body = body

    async def readexactly(self, size: int) -> bytes:
        if len(self.body) < size:
            raise asyncio.IncompleteReadError(partial=self.body, expected=size)
        return self.body[:size]

    async def __aenter__(self) -> _FakeHttpResponse:
        return self

    async def __aexit__(self, *_args: object) -> None:
        return None


class _FakeHttpSession:
    def __init__(self, response: _FakeHttpResponse) -> None:
        self.response = response
        self.requests: list[dict] = []

    async def __aenter__(self) -> _FakeHttpSession:
        return self

    async def __aexit__(self, *_args: object) -> None:
        return None

    def post(self, url: str, *, json: dict, allow_redirects: bool) -> _FakeHttpResponse:
        self.requests.append(
            {
                "url": url,
                "json": json,
                "allow_redirects": allow_redirects,
            }
        )
        return self.response


def _scripted_tools_payload(name: str = "route_to_any_application") -> dict:
    return {
        "tools": [
            {
                "type": "function",
                "name": name,
                "description": "Route one request to the configured application.",
                "parameters": {
                    "type": "object",
                    "properties": {"request": {"type": "string"}},
                    "required": ["request"],
                    "additionalProperties": False,
                },
            }
        ],
        "handlers": {
            name: {
                "steps": [
                    {
                        "expected_arguments": {"request": "status"},
                        "output": {"ok": True, "marker": "APPLICATION_COMPLETE"},
                    }
                ],
                "repeat_last": False,
            }
        },
        "tool_choice": {"type": "function", "name": name},
        "parallel_tool_calls": False,
    }


def _load_tools_config(raw: str | dict) -> RealtimeClientToolsConfig:
    with tempfile.TemporaryDirectory() as temp_dir:
        path = Path(temp_dir) / "client-tools.json"
        path.write_text(raw if isinstance(raw, str) else json.dumps(raw), encoding="utf-8")
        return load_client_tools_config(path)


def _realtime_client(
    *,
    input_mode: str = "text",
    output_modality: str = "text",
    audio_files: list[Path] | None = None,
    text_inputs: list[str] | None = None,
    client_tools_config: RealtimeClientToolsConfig | None = None,
) -> OpenAIRealtimePerfClient:
    return OpenAIRealtimePerfClient(
        stream_id="client_1_test",
        host="localhost",
        port=7860,
        path="/v1/realtime",
        scheme="wss",
        insecure=False,
        ca_file=None,
        model="",
        voice="",
        instructions="",
        input_mode=input_mode,
        text_inputs=text_inputs if text_inputs is not None else ["Hello from the scaling client"],
        output_modality=output_modality,
        turn_mode="automatic",
        vad_silence_ms=800,
        api_key_env="",
        audio_files=audio_files or [],
        start_delay=0,
        metrics_start_time=None,
        session_end_time=None,
        test_duration=1,
        reverse_barge_in_threshold=0.4,
        turn_response_timeout=10,
        tool_completion_timeout=120,
        max_tool_rounds=4,
        client_tools_config=client_tools_config,
        session_init_timeout=60,
        audio_output_path=None,
        logger=_MemoryLogger(),
        server_metric_keys=SERVER_METRIC_KEYS,
        shutdown_requested=lambda: False,
    )


def _received_events(events: list[dict]) -> list[_ReceivedEvent]:
    base_time = time.time()
    return [
        _ReceivedEvent(event=event, received_at=base_time + index / 1000) for index, event in enumerate(events, start=1)
    ]


def _response(
    response_id: str,
    conversation_id: str,
    *,
    status: str = "in_progress",
    output: list[dict] | None = None,
    usage: dict | None = None,
) -> dict:
    return {
        "id": response_id,
        "object": "realtime.response",
        "conversation_id": conversation_id,
        "status": status,
        "status_details": None,
        "output": output or [],
        "usage": usage,
    }


def _text_response_events(
    *,
    conversation_id: str,
    response_id: str,
    item_id: str,
    previous_item_id: str,
    text: str,
    usage: dict | None = None,
) -> list[dict]:
    item = {
        "id": item_id,
        "object": "realtime.item",
        "type": "message",
        "status": "in_progress",
        "role": "assistant",
        "content": [],
    }
    done = {**item, "status": "completed", "content": [{"type": "output_text", "text": text}]}
    created = _response(response_id, conversation_id)
    content_fields = {
        "response_id": response_id,
        "item_id": item_id,
        "output_index": 0,
        "content_index": 0,
    }
    return [
        {"type": "response.created", "response": created},
        {"type": "conversation.item.added", "previous_item_id": previous_item_id, "item": item},
        {"type": "response.output_item.added", "response_id": response_id, "output_index": 0, "item": item},
        {"type": "response.content_part.added", **content_fields, "part": {"type": "text", "text": ""}},
        {"type": "response.output_text.delta", **content_fields, "delta": text},
        {"type": "response.output_text.done", **content_fields, "text": text},
        {"type": "response.content_part.done", **content_fields, "part": {"type": "text", "text": text}},
        {"type": "conversation.item.done", "item": done},
        {"type": "response.output_item.done", "response_id": response_id, "output_index": 0, "item": done},
        {
            "type": "response.done",
            "response": {
                **created,
                "status": "completed",
                "output": [done],
                "usage": usage or {"total_tokens": 5, "input_tokens": 4, "output_tokens": 1},
            },
        },
    ]


def _function_response_events(user_item: dict, *, tool_name: str, arguments: str) -> list[dict]:
    call = {
        "id": "item_application_status",
        "object": "realtime.item",
        "type": "function_call",
        "status": "in_progress",
        "call_id": "call_application_status",
        "name": tool_name,
        "arguments": "",
    }
    done = {**call, "status": "completed", "arguments": arguments}
    created = _response("resp_a", "conv_client_tool")
    argument_fields = {
        "response_id": "resp_a",
        "item_id": call["id"],
        "output_index": 0,
        "call_id": call["call_id"],
    }
    return [
        {"type": "conversation.item.added", "previous_item_id": None, "item": user_item},
        {"type": "conversation.item.done", "item": user_item},
        {"type": "response.created", "response": created},
        {"type": "conversation.item.added", "previous_item_id": user_item["id"], "item": call},
        {"type": "response.output_item.added", "response_id": "resp_a", "output_index": 0, "item": call},
        {"type": "response.function_call_arguments.delta", **argument_fields, "delta": arguments},
        {"type": "response.function_call_arguments.done", **argument_fields, "name": tool_name, "arguments": arguments},
        {"type": "conversation.item.done", "item": done},
        {"type": "response.output_item.done", "response_id": "resp_a", "output_index": 0, "item": done},
        {"type": "response.done", "response": {**created, "status": "completed", "output": [done]}},
    ]


class RealtimeClientToolConfigTests(unittest.TestCase):
    """Verify the file-backed client-tool registry's public contract."""

    def test_accepts_an_arbitrary_name_with_an_explicit_function_choice(self) -> None:
        """Treat function names as application configuration, not reserved routing words."""
        name = "route_to_any_application"
        config = _load_tools_config(_scripted_tools_payload(name))

        self.assertEqual([tool["name"] for tool in config.tools], [name])
        self.assertEqual(config.tool_choice, {"type": "function", "name": name})
        self.assertFalse(config.parallel_tool_calls)
        self.assertIsInstance(config.handlers[name], ScriptedToolHandler)

    def test_rejects_ambiguous_or_unsafe_configurations(self) -> None:
        """Fail closed for inputs that cannot define one safe, exact handler binding."""
        name = "route_to_any_application"
        duplicate_keys = '{"tools":[],"tools":[],"handlers":{},"tool_choice":"auto","parallel_tool_calls":true}'

        handler_mismatch = _scripted_tools_payload(name)
        handler_mismatch["handlers"] = {}

        unknown_forced_name = _scripted_tools_payload(name)
        unknown_forced_name["tool_choice"] = {"type": "function", "name": "not_declared"}

        nonfinite = _scripted_tools_payload(name)
        nonfinite["handlers"][name]["steps"][0]["output"] = {"score": float("nan")}

        unsafe_http_url = _scripted_tools_payload(name)
        unsafe_http_url["handlers"][name] = {
            "type": "http",
            "url": "https://user:password@example.com/run",
        }

        invalid_parameters_schema = _scripted_tools_payload(name)
        invalid_parameters_schema["tools"][0]["parameters"] = {"type": "not-a-json-schema-type"}

        cases = (
            ("duplicate object key", duplicate_keys, "duplicate JSON object key"),
            ("handler mismatch", handler_mismatch, "handlers must match tools exactly"),
            ("unknown forced name", unknown_forced_name, "selects unavailable function"),
            ("non-finite value", nonfinite, "non-finite JSON constant"),
            ("URL credentials", unsafe_http_url, "url must not contain credentials"),
            ("invalid parameters schema", invalid_parameters_schema, "parameters is not a valid JSON Schema"),
        )
        for label, raw, expected_error in cases:
            with self.subTest(label=label), self.assertRaisesRegex(RealtimeClientError, expected_error):
                _load_tools_config(raw)


class RealtimeTextLatencyTests(unittest.TestCase):
    """Verify which streamed text deltas can start user-visible latency."""

    def test_only_visible_text_starts_latency(self) -> None:
        """Ignore formatting whitespace and accept any visible character."""
        cases = (("", False), (" ", False), ("\n\n", False), ("\t \r\n", False), ("\u2003", False))
        cases += (("Hello", True), ("  Hello", True), (".", True), ("\nA", True))
        for delta, expected in cases:
            with self.subTest(delta=repr(delta)):
                self.assertIs(_contains_user_visible_text(delta), expected)


class RealtimeTextInputTests(unittest.IsolatedAsyncioTestCase):
    """Verify canonical text-turn submission independently of response modality."""

    async def test_connection_timeout_identifies_the_open_phase(self) -> None:
        """Do not misreport a WebSocket open timeout as the outer hard deadline."""
        client = _realtime_client()

        with (
            patch("openai_realtime_client.websockets.connect", new=AsyncMock(side_effect=TimeoutError)),
            self.assertRaises(RealtimeClientError) as raised,
        ):
            await client._run_connection()

        self.assertEqual(str(raised.exception), "WebSocket connection did not open within 30.0s")

    async def test_initialization_timeout_identifies_each_waiting_phase(self) -> None:
        """Preserve the handshake phase in session-initialization failures."""
        client = _realtime_client()

        async def receive(_websocket: object) -> _ReceivedEvent:
            raise TimeoutError

        client._receive_wire_event = receive
        for phase in (
            "session.created and conversation.created",
            "session.updated and MCP discovery completion",
        ):
            with self.subTest(phase=phase), self.assertRaises(RealtimeClientError) as raised:
                await client._receive_initialization_event(object(), phase)
            self.assertEqual(str(raised.exception), f"Timed out after 60.0s waiting for {phase}")

    async def test_missing_api_key_environment_name_is_not_exposed(self) -> None:
        """Keep the configured environment-variable name out of persisted failures."""
        client = _realtime_client()
        client.api_key_env = "SENTINEL_SECRET_ENV_NAME"

        with patch.dict("os.environ", {}, clear=True), self.assertRaises(RealtimeClientError) as raised:
            await client._run_connection()

        self.assertEqual(str(raised.exception), "Configured Realtime API key environment variable is not set")

    async def test_text_input_sends_item_then_response_create_for_both_outputs(self) -> None:
        """Send the same canonical input_text item for text and audio responses."""
        prompt = "Calculate my BMI for 70 kilograms and 1.75 meters."
        for output_modality in ("text", "audio"):
            with self.subTest(output_modality=output_modality):
                client = _realtime_client(output_modality=output_modality, text_inputs=[prompt])
                websocket = _RecordingWebSocket()
                trace = _TurnInputTrace()

                _, pending_ids = await client._submit_text_turn(websocket, prompt, trace)

                self.assertEqual(
                    [event["type"] for event in websocket.events],
                    [
                        "conversation.item.create",
                        "response.create",
                    ],
                )
                item = websocket.events[0]["item"]
                self.assertEqual(item["id"], trace.item_id)
                self.assertEqual(item["type"], "message")
                self.assertEqual(item["role"], "user")
                self.assertEqual(item["content"], [{"type": "input_text", "text": prompt}])
                self.assertEqual(trace.submitted_text, prompt)
                self.assertEqual(pending_ids, [event["event_id"] for event in websocket.events])
                self.assertEqual(client._pending_events[pending_ids[0]]["ack"], "conversation.item.done")
                self.assertEqual(client._pending_events[pending_ids[1]]["ack"], "response.created")

    async def test_input_mode_requires_the_matching_dataset(self) -> None:
        """Require WAVs only for audio input and prompts only for text input."""
        with self.assertRaisesRegex(ValueError, "audio_files"):
            _realtime_client(input_mode="audio", audio_files=[], text_inputs=[])
        with self.assertRaisesRegex(ValueError, "text_inputs"):
            _realtime_client(input_mode="text", text_inputs=[])

    async def test_text_turn_completes_the_correlated_response_lifecycle(self) -> None:
        """Consume a complete canonical text response after submitting input_text."""
        prompt = "Reply with one greeting."
        client = _realtime_client(text_inputs=[prompt])
        client._conversation_id = "conv_test"
        client._collecting_metrics = True
        websocket = _RecordingWebSocket()
        queued: list[_ReceivedEvent] | None = None

        async def next_event(_timeout: float) -> _ReceivedEvent:
            nonlocal queued
            if queued is None:
                user_id = websocket.events[0]["item"]["id"]
                user_item = {
                    **websocket.events[0]["item"],
                    "object": "realtime.item",
                    "status": "completed",
                }
                queued = _received_events(
                    [
                        {
                            "type": "conversation.item.added",
                            "previous_item_id": None,
                            "item": user_item,
                        },
                        {"type": "conversation.item.done", "item": user_item},
                        *_text_response_events(
                            conversation_id="conv_test",
                            response_id="resp_test",
                            item_id="item_assistant",
                            previous_item_id=user_id,
                            text="Hello",
                            usage={"total_tokens": 3, "input_tokens": 2, "output_tokens": 1},
                        ),
                    ]
                )
            return queued.pop(0)

        client._next_event = next_event
        turn = await client._run_turn(websocket, prompt)

        self.assertEqual(turn.response_ids, ["resp_test"])
        self.assertEqual(turn.transcript, "Hello")
        self.assertEqual(turn.output_bytes, 0)
        self.assertGreaterEqual(turn.first_output_at, turn.input_finished_at)
        self.assertEqual(client.server_metric_samples["response_total_tokens"], [3.0])
        self.assertEqual(client.server_metric_samples["response_input_tokens"], [2.0])
        self.assertEqual(client.server_metric_samples["response_output_tokens"], [1.0])
        self.assertEqual(len(client.server_metric_samples["response_lifecycle"]), 1)
        self.assertEqual(client._pending_events, {})

    async def test_text_turn_timeout_identifies_the_waiting_phase(self) -> None:
        """Persist an actionable error when no first response output arrives."""
        client = _realtime_client()
        client._conversation_id = "conv_test"

        async def next_event(_timeout: float) -> _ReceivedEvent:
            raise TimeoutError

        client._next_event = next_event
        with self.assertRaisesRegex(TimeoutError, "no first response output activity within 10.0s"):
            await client._run_turn(_RecordingWebSocket(), "Hello")


class RealtimeClientToolRoundTests(unittest.IsolatedAsyncioTestCase):
    """Verify one complete standard client-owned function-call round."""

    async def test_schema_validation_precedes_scripted_argument_expectations(self) -> None:
        """Distinguish schema-invalid arguments from a valid exact-match failure."""
        tool_name = "route_to_any_application"
        schema_invalid_message = (
            f"Client tool {tool_name!r} arguments did not satisfy its declared JSON Schema (constraint: type)"
        )
        cases = (
            (
                "schema-invalid",
                {"request": 42},
                "client_tool_arguments_invalid",
                schema_invalid_message,
            ),
            (
                "valid-but-unexpected",
                {"request": "different"},
                "client_tool_arguments_mismatch",
                f"Client tool {tool_name!r} arguments did not match the scripted expectation",
            ),
        )
        for label, arguments, expected_code, expected_message in cases:
            with self.subTest(label=label):
                config = _load_tools_config(_scripted_tools_payload(tool_name))
                client = _realtime_client(client_tools_config=config)
                call_id = f"call_{label}"
                client._tool_calls[call_id] = {
                    "name": tool_name,
                    "arguments": json.dumps(arguments),
                }

                [(returned_call_id, raw_output)] = await client._execute_client_tool_round(
                    [call_id],
                    tool_deadline_at=time.time() + 30,
                )

                self.assertEqual(returned_call_id, call_id)
                output = json.loads(raw_output)
                self.assertEqual(
                    output,
                    {
                        "ok": False,
                        "error": {
                            "code": expected_code,
                            "message": expected_message,
                        },
                    },
                )
                self.assertEqual(client._tool_calls[call_id]["handler_status"], "failed")
                self.assertEqual(client._tool_calls[call_id]["handler_error"], output["error"])

    async def test_output_acknowledgement_precedes_exactly_one_recovery_response(self) -> None:
        """Preserve call identity across Response A, output, and final Response B."""
        tool_name = "route_to_any_application"
        arguments = '{"request":"status"}'
        expected_result = {"ok": True, "marker": "APPLICATION_COMPLETE"}
        config = _load_tools_config(_scripted_tools_payload(tool_name))
        client = _realtime_client(client_tools_config=config, text_inputs=["Check the application status."])
        client._conversation_id = "conv_client_tool"
        client._collecting_metrics = True
        client._configure_tool_ownership(
            {
                "tools": [],
                "tool_choice": "auto",
                "parallel_tool_calls": True,
            }
        )
        websocket = _RecordingWebSocket()
        queued: list[_ReceivedEvent] = []
        delivered: list[dict] = []
        stage = 0

        def enqueue(raw_events: list[dict]) -> None:
            queued.extend(_received_events(raw_events))

        async def next_event(_timeout: float) -> _ReceivedEvent:
            nonlocal stage
            if not queued and stage == 0:
                self.assertEqual(
                    [event["type"] for event in websocket.events], ["conversation.item.create", "response.create"]
                )
                user_item = {
                    **websocket.events[0]["item"],
                    "object": "realtime.item",
                    "status": "completed",
                }
                enqueue(_function_response_events(user_item, tool_name=tool_name, arguments=arguments))
                stage += 1

            if not queued and stage == 1:
                output_events = [
                    event
                    for event in websocket.events
                    if event["type"] == "conversation.item.create" and event["item"]["type"] == "function_call_output"
                ]
                self.assertEqual(len(output_events), 1)
                self.assertEqual(delivered[-1]["type"], "response.done")
                self.assertEqual(sum(event["type"] == "response.create" for event in websocket.events), 1)
                output_item = {
                    **output_events[0]["item"],
                    "object": "realtime.item",
                    "status": "completed",
                }
                enqueue(
                    [
                        {
                            "type": "conversation.item.added",
                            "previous_item_id": "item_application_status",
                            "item": output_item,
                        },
                        {"type": "conversation.item.done", "item": output_item},
                    ]
                )
                stage += 1

            if not queued and stage == 2:
                output_ack = next(
                    event
                    for event in delivered
                    if event["type"] == "conversation.item.done" and event["item"]["type"] == "function_call_output"
                )
                response_creates = [event for event in websocket.events if event["type"] == "response.create"]
                self.assertEqual(len(response_creates), 2)
                enqueue(
                    _text_response_events(
                        conversation_id="conv_client_tool",
                        response_id="resp_b",
                        item_id="item_final_answer",
                        previous_item_id=output_ack["item"]["id"],
                        text="Application status is complete.",
                    )
                )
                stage += 1

            if not queued:
                raise AssertionError("The client requested an unexpected additional server event")
            received = queued.pop(0)
            delivered.append(received.event)
            return received

        client._next_event = next_event
        turn = await client._run_turn(websocket, "Check the application status.")

        self.assertEqual(turn.response_ids, ["resp_a", "resp_b"])
        self.assertEqual(turn.tool_call_ids, ["call_application_status"])
        self.assertEqual(turn.transcript, "Application status is complete.")
        self.assertEqual(stage, 3)

        outbound_types = [event["type"] for event in websocket.events]
        self.assertEqual(
            outbound_types,
            [
                "conversation.item.create",
                "response.create",
                "conversation.item.create",
                "response.create",
            ],
        )
        output_item = websocket.events[2]["item"]
        self.assertEqual(output_item["call_id"], "call_application_status")
        self.assertEqual(json.loads(output_item["output"]), expected_result)
        self.assertNotIn("response", websocket.events[1])
        self.assertEqual(websocket.events[3]["response"], {"tool_choice": "none"})

        call = client._tool_calls["call_application_status"]
        self.assertEqual(call["owner"], "client")
        self.assertEqual(call["response_id"], "resp_a")
        self.assertEqual(call["output_item_id"], output_item["id"])
        self.assertEqual(call["handler_status"], "completed")
        self.assertEqual(call["handler_error"], None)
        self.assertEqual(client._pending_events, {})

    def test_recovery_only_relaxes_session_choices_that_force_another_call(self) -> None:
        """Keep auto iterative rounds while preventing required or named call loops."""
        client = _realtime_client()
        cases = (
            ("auto", {}),
            ("none", {}),
            ("required", {"response": {"tool_choice": "none"}}),
            (
                {"type": "function", "name": "route_to_any_application"},
                {"response": {"tool_choice": "none"}},
            ),
        )
        for session_choice, expected_payload in cases:
            with self.subTest(session_choice=session_choice):
                client._selected_tool_choice = session_choice
                self.assertEqual(client._client_tool_recovery_response_payload(), expected_payload)


class RealtimeHttpClientToolTests(unittest.IsolatedAsyncioTestCase):
    """Verify deterministic HTTP client-tool execution without network access."""

    async def test_schema_invalid_arguments_never_reach_the_http_handler(self) -> None:
        """Return a structured function output without opening an HTTP session."""
        tool_name = "lookup_at_configured_origin"
        payload = _scripted_tools_payload(tool_name)
        payload["handlers"][tool_name] = {
            "type": "http",
            "url": "https://tools.example.com/lookup",
            "timeout_seconds": 3,
        }
        config = _load_tools_config(payload)
        client = _realtime_client(client_tools_config=config)
        call_id = "call_http_schema_invalid"
        client._tool_calls[call_id] = {
            "name": tool_name,
            "arguments": '{"request":42}',
        }

        with patch("openai_realtime_client.aiohttp.ClientSession") as session_factory:
            [(returned_call_id, raw_output)] = await client._execute_client_tool_round(
                [call_id],
                tool_deadline_at=time.time() + 30,
            )

        session_factory.assert_not_called()
        self.assertEqual(returned_call_id, call_id)
        output = json.loads(raw_output)
        self.assertEqual(output["error"]["code"], "client_tool_arguments_invalid")
        self.assertIn("declared JSON Schema", output["error"]["message"])
        self.assertEqual(client._tool_calls[call_id]["handler_status"], "failed")

    async def _execute_http_response(
        self,
        *,
        status: int,
        body: bytes,
        call_id: str,
    ) -> tuple[dict, _FakeHttpSession, dict]:
        client = _realtime_client()
        client._tool_calls[call_id] = {"name": "lookup_at_configured_origin"}
        session = _FakeHttpSession(_FakeHttpResponse(status=status, body=body))
        invocation = HttpToolInvocation(
            handler=HttpToolHandler(
                url="https://tools.example.com/lookup",
                timeout_seconds=3,
            ),
            arguments={"query": "weather"},
        )
        with patch("openai_realtime_client.aiohttp.ClientSession", return_value=session):
            output = await client._execute_client_tool_invocation(
                call_id,
                invocation,
                handler_deadline_at=time.time() + 2,
            )
        return json.loads(output), session, client._tool_calls[call_id]

    async def test_http_handler_posts_arguments_and_returns_json(self) -> None:
        """Return a successful endpoint JSON body as the function output."""
        output, session, call = await self._execute_http_response(
            status=200,
            body=b'{"ok":true,"temperature":24}',
            call_id="call_http_success",
        )

        self.assertEqual(output, {"ok": True, "temperature": 24})
        self.assertEqual(
            session.requests,
            [
                {
                    "url": "https://tools.example.com/lookup",
                    "json": {"query": "weather"},
                    "allow_redirects": False,
                }
            ],
        )
        self.assertEqual(call["handler_status"], "completed")
        self.assertIsNone(call["handler_error"])

    async def test_http_protocol_failures_become_structured_tool_outputs(self) -> None:
        """Expose redirects, non-success status, and invalid JSON to Response B."""
        cases = (
            ("redirect", 302, b"", "client_tool_http_redirect"),
            ("non-success", 503, b"temporarily unavailable", "client_tool_http_status"),
            ("invalid JSON", 200, b"not-json", "client_tool_http_invalid_json"),
        )
        for label, status, body, expected_code in cases:
            with self.subTest(label=label):
                output, session, call = await self._execute_http_response(
                    status=status,
                    body=body,
                    call_id=f"call_http_{status}_{expected_code}",
                )

                self.assertFalse(output["ok"])
                self.assertEqual(output["error"]["code"], expected_code)
                self.assertTrue(output["error"]["message"])
                self.assertEqual(len(session.requests), 1)
                self.assertFalse(session.requests[0]["allow_redirects"])
                self.assertEqual(call["handler_status"], "failed")
                self.assertEqual(call["handler_error"], output["error"])


def _write_client_result(
    run_dir: Path,
    *,
    ordinal: int,
    protocol: str,
    metric_average: dict[str, float],
    metric_counts: dict[str, int],
    valid_response: bool = True,
    error: str | None = None,
) -> None:
    stream_id = f"client_{ordinal}_test"
    client_dir = run_dir / stream_id
    client_dir.mkdir()
    run_config = {
        "protocol": protocol,
        "host": "localhost",
        "port": 7860,
        "metrics_start_time": 1.0,
        "session_end_time": 2.0,
        "test_duration": 1.0,
        "reverse_barge_in_threshold": 0.4,
        "turn_response_timeout": 10.0,
    }
    is_realtime = protocol == "openai-realtime"
    if is_realtime:
        run_config["realtime"] = {
            "input_mode": "text",
            "output_modality": "text",
            "tool_timeout": 120.0,
        }
    payload = {
        "stream_id": stream_id,
        "average_latency": 0.5 if valid_response else None,
        "individual_latencies": [0.5] if valid_response else [],
        "valid_latencies": [0.5] if valid_response else [],
        "num_turns": 1,
        "num_valid_turns": 1 if valid_response else 0,
        "failed_turns": 0,
        "reverse_barge_ins_count": 0,
        "glitch_detected": False,
        "reverse_barge_in_threshold": 0.4,
        "turn_response_timeout": 10.0,
        "realtime_tool_timeout": 120.0 if is_realtime else None,
        "metrics_start_time": 1.0,
        "test_duration": 1.0,
        "server_metrics": {
            "samples": {},
            "average": metric_average,
            "sample_counts": metric_counts,
        },
        "protocol": protocol,
        "error": error,
        "run_config": run_config,
    }
    (client_dir / f"result_{stream_id}.json").write_text(json.dumps(payload), encoding="utf-8")


class RTVIDeadlineTests(unittest.TestCase):
    """Protect the existing RTVI benchmark completion boundary."""

    def test_deadline_includes_configured_response_tail(self) -> None:
        """Do not terminate an admitted turn before its response timeout."""
        client = PerfClient(
            stream_id="rtvi_deadline",
            host="localhost",
            port=7860,
            audio_files=[],
            start_delay=0,
            metrics_start_time=1000,
            session_end_time=1060,
            test_duration=60,
            reverse_barge_in_threshold=0.4,
            audio_output_path=None,
            logger=_MemoryLogger(),
            turn_response_timeout=120,
        )

        with patch("benchmark.time.time", return_value=1000):
            self.assertEqual(client._hard_deadline_seconds(), 193)


class RealtimeMetricAggregationTests(unittest.TestCase):
    """Keep Realtime response metrics in summaries without widening RTVI output."""

    def test_realtime_response_metrics_survive_run_and_suite_aggregation(self) -> None:
        """Pool Realtime response samples and publish them in every aggregate."""
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir)
            first = {key: float(index + 1) for index, key in enumerate(REALTIME_RESPONSE_METRIC_KEYS)}
            second = {key: value * 2 for key, value in first.items()}
            _write_client_result(
                run_dir,
                ordinal=1,
                protocol="openai-realtime",
                metric_average=first,
                metric_counts={key: 1 for key in first},
            )
            _write_client_result(
                run_dir,
                ordinal=2,
                protocol="openai-realtime",
                metric_average=second,
                metric_counts={key: 3 for key in second},
            )

            summary_path = _aggregate_run_dir(run_dir, num_clients=2)
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            averages = summary["results"]["server_metrics"]["average"]
            counts = summary["results"]["server_metrics"]["sample_counts"]
            for key in REALTIME_RESPONSE_METRIC_KEYS:
                self.assertEqual(averages[key], (first[key] + 3 * second[key]) / 4)
                self.assertEqual(counts[key], 4)

            tsv_path, _text_path, json_path = _aggregate_suite_dir(run_dir)
            rows = json.loads(json_path.read_text(encoding="utf-8"))
            self.assertEqual(len(rows), 3)
            for row in rows:
                for key in REALTIME_RESPONSE_METRIC_KEYS:
                    self.assertIn(key, row)
                    self.assertIsNotNone(row[key])
            average_row = next(row for row in rows if row["client"] == "AVERAGE")
            for key in REALTIME_RESPONSE_METRIC_KEYS:
                self.assertEqual(average_row[key], (first[key] + 3 * second[key]) / 4)
            headers = tsv_path.read_text(encoding="utf-8").splitlines()[0].split("\t")
            self.assertIn("Response Lifecycle", headers)
            self.assertIn("Response Audio Bytes", headers)

    def test_rtvi_aggregate_shape_keeps_existing_metric_columns(self) -> None:
        """Leave pure RTVI run and table schemas unchanged."""
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir)
            _write_client_result(
                run_dir,
                ordinal=1,
                protocol="rtvi",
                metric_average={"llm_ttft": 0.25},
                metric_counts={"llm_ttft": 1},
            )

            summary_path = _aggregate_run_dir(run_dir, num_clients=1)
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertEqual(tuple(summary["results"]["server_metrics"]["average"]), SERVER_METRIC_KEYS)

            tsv_path, _text_path, json_path = _aggregate_suite_dir(run_dir)
            headers = tsv_path.read_text(encoding="utf-8").splitlines()[0].split("\t")
            self.assertNotIn("Response Lifecycle", headers)
            rows = json.loads(json_path.read_text(encoding="utf-8"))
            self.assertEqual(len(rows), 2)
            for row in rows:
                self.assertNotIn("response_lifecycle", row)


class BenchmarkAcceptanceTests(unittest.TestCase):
    """Make report generation independent from benchmark acceptance."""

    def test_worker_requires_a_valid_measured_turn(self) -> None:
        """Reject a clean process that observed no valid response."""
        self.assertTrue(_benchmark_result_is_accepted(SimpleNamespace(error=None, num_valid_turns=1)))
        self.assertFalse(_benchmark_result_is_accepted(SimpleNamespace(error=None, num_valid_turns=0)))
        self.assertFalse(_benchmark_result_is_accepted(SimpleNamespace(error="connection failed", num_valid_turns=1)))

    def test_aggregate_commands_write_reports_but_reject_no_response(self) -> None:
        """Return nonzero for a generated report with an unsuccessful worker."""
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir)
            _write_client_result(
                run_dir,
                ordinal=1,
                protocol="openai-realtime",
                metric_average={},
                metric_counts={},
                valid_response=False,
            )

            self.assertEqual(_run_aggregate_run(run_dir, num_clients=1), 1)
            self.assertTrue((run_dir / "benchmark_summary.json").is_file())
            self.assertEqual(_run_aggregate_suite(run_dir), 1)
            self.assertTrue((run_dir / "results.json").is_file())

    def test_suite_requires_a_concrete_run_row(self) -> None:
        """Do not accept an empty report or its synthetic average row."""
        self.assertFalse(_suite_results_are_accepted([]))
        self.assertFalse(_suite_results_are_accepted([{"client": "AVERAGE"}]))
