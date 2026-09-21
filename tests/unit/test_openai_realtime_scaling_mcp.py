# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""Focused native-MCP tests for the OpenAI Realtime scaling client."""

from __future__ import annotations

import asyncio
import copy
import json
import sys
import tempfile
import time
import unittest
from pathlib import Path

SCALING_PERF_DIR = Path(__file__).resolve().parents[2] / "benchmarking_tools" / "scaling-perf"
sys.path.insert(0, str(SCALING_PERF_DIR))

from benchmark import SERVER_METRIC_KEYS
from openai_realtime_client import (
    OpenAIRealtimePerfClient,
    RealtimeClientError,
    RealtimeClientToolsConfig,
    _ReceivedEvent,
    load_client_tools_config,
)


class _MemoryLogger:
    async def log(self, message: str) -> None:  # noqa: ARG002
        return None


class _RecordingWebSocket:
    def __init__(self) -> None:
        self.events: list[dict] = []

    async def send(self, message: str) -> None:
        self.events.append(json.loads(message))


def _mcp_tool() -> dict:
    return {
        "type": "mcp",
        "server_label": "openai_docs",
        "server_url": "https://developers.openai.com/mcp",
        "allowed_tools": ["search_openai_docs"],
        "require_approval": "always",
    }


def _function_tool() -> dict:
    return {
        "type": "function",
        "name": "local_lookup",
        "parameters": {"type": "object", "additionalProperties": False},
    }


def _scripted_handler() -> dict:
    return {"steps": [{"expected_arguments": {}, "output": {"ok": True}}]}


def _approval_policy(*, approve: bool) -> dict:
    return {
        "default": "reject",
        "rules": [
            {
                "server_label": "openai_docs",
                "name": "search_openai_docs",
                "approve": True,
            }
        ]
        if approve
        else [],
    }


def _load_config(payload: dict) -> RealtimeClientToolsConfig:
    with tempfile.TemporaryDirectory() as temp_dir:
        path = Path(temp_dir) / "tools.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        return load_client_tools_config(path)


def _client(client_tools_config: RealtimeClientToolsConfig | None = None) -> OpenAIRealtimePerfClient:
    return OpenAIRealtimePerfClient(
        stream_id="client_1_mcp",
        host="localhost",
        port=7860,
        path="/v1/realtime",
        scheme="wss",
        insecure=False,
        ca_file=None,
        model="",
        voice="",
        instructions="",
        input_mode="text",
        text_inputs=["Search the OpenAI docs."],
        output_modality="text",
        turn_mode="automatic",
        vad_silence_ms=800,
        api_key_env="",
        audio_files=[],
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


def _created_items(websocket: _RecordingWebSocket, item_type: str) -> list[dict]:
    return [
        event["item"]
        for event in websocket.events
        if event["type"] == "conversation.item.create" and event["item"]["type"] == item_type
    ]


def _session(*, tools: list[dict], tool_choice: object = "required") -> dict:
    return {
        "id": "sess_mcp",
        "object": "realtime.session",
        "type": "realtime",
        "model": "test-model",
        "output_modalities": ["text"],
        "tools": copy.deepcopy(tools),
        "tool_choice": copy.deepcopy(tool_choice),
        "parallel_tool_calls": True,
        "audio": {
            "input": {
                "format": {"type": "audio/pcm", "rate": 24000},
                "turn_detection": {"type": "server_vad"},
            },
            "output": {"format": {"type": "audio/pcm", "rate": 24000}},
        },
        "nvidia": {"server_tools": [], "delegate_tools": []},
    }


_MCP_ARGUMENTS = '{"query":"Realtime"}'


def _response(response_id: str, *, status: str = "in_progress", output: list[dict] | None = None) -> dict:
    return {
        "id": response_id,
        "object": "realtime.response",
        "conversation_id": "conv_mcp",
        "status": status,
        "status_details": None,
        "output": output or [],
        "usage": None,
    }


def _mcp_call(*, arguments: str = "", **fields: object) -> dict:
    return {
        "id": "item_mcp_call",
        "type": "mcp_call",
        "server_label": "openai_docs",
        "name": "search_openai_docs",
        "arguments": arguments,
        **fields,
    }


def _approval_request() -> dict:
    return {
        "id": "item_approval_request",
        "type": "mcp_approval_request",
        "server_label": "openai_docs",
        "name": "search_openai_docs",
        "arguments": _MCP_ARGUMENTS,
    }


def _mcp_response_events(user_item: dict) -> list[dict]:
    call = _mcp_call()
    request = _approval_request()
    created = _response("resp_a")
    event_fields = {
        "response_id": "resp_a",
        "item_id": call["id"],
        "output_index": 0,
    }
    return [
        {"type": "conversation.item.added", "previous_item_id": None, "item": user_item},
        {"type": "conversation.item.done", "item": user_item},
        {"type": "response.created", "response": created},
        {"type": "conversation.item.added", "previous_item_id": user_item["id"], "item": call},
        {"type": "response.output_item.added", "response_id": "resp_a", "output_index": 0, "item": call},
        {"type": "response.mcp_call_arguments.delta", **event_fields, "delta": _MCP_ARGUMENTS},
        {"type": "response.mcp_call_arguments.done", **event_fields, "arguments": _MCP_ARGUMENTS},
        {
            "type": "response.done",
            "response": {**created, "status": "completed", "output": [_mcp_call(arguments=_MCP_ARGUMENTS)]},
        },
        {"type": "conversation.item.added", "previous_item_id": call["id"], "item": request},
        {"type": "conversation.item.done", "item": request},
    ]


def _mcp_completion_events(approval: dict, *, approve: bool) -> list[dict]:
    terminal = "completed" if approve else "failed"
    call = _mcp_call(
        arguments=_MCP_ARGUMENTS,
        approval_request_id="item_approval_request",
        output='{"ok":true}' if approve else None,
        error=None if approve else {"type": "tool_execution_error", "message": "denied"},
    )
    lifecycle = [{"type": "response.mcp_call.in_progress", "item_id": call["id"], "output_index": 0}] if approve else []
    return [
        {"type": "conversation.item.added", "previous_item_id": "item_approval_request", "item": approval},
        {"type": "conversation.item.done", "item": approval},
        *lifecycle,
        {"type": f"response.mcp_call.{terminal}", "item_id": call["id"], "output_index": 0},
        {"type": "conversation.item.done", "item": call},
        {"type": "response.output_item.done", "response_id": "resp_a", "output_index": 0, "item": call},
    ]


def _final_response_events(previous_item_id: str) -> list[dict]:
    item = {
        "id": "item_final",
        "object": "realtime.item",
        "type": "message",
        "status": "in_progress",
        "role": "assistant",
        "content": [],
    }
    done = {**item, "status": "completed", "content": [{"type": "output_text", "text": "Done."}]}
    created = _response("resp_b")
    content_fields = {
        "response_id": "resp_b",
        "item_id": item["id"],
        "output_index": 0,
        "content_index": 0,
    }
    return [
        {"type": "response.created", "response": created},
        {"type": "conversation.item.added", "previous_item_id": previous_item_id, "item": item},
        {"type": "response.output_item.added", "response_id": "resp_b", "output_index": 0, "item": item},
        {"type": "response.content_part.added", **content_fields, "part": {"type": "text", "text": ""}},
        {"type": "response.output_text.delta", **content_fields, "delta": "Done."},
        {"type": "response.output_text.done", **content_fields, "text": "Done."},
        {"type": "response.content_part.done", **content_fields, "part": {"type": "text", "text": "Done."}},
        {"type": "conversation.item.done", "item": done},
        {"type": "response.output_item.done", "response_id": "resp_b", "output_index": 0, "item": done},
        {
            "type": "response.done",
            "response": {
                **created,
                "status": "completed",
                "output": [done],
                "usage": {"total_tokens": 3, "input_tokens": 2, "output_tokens": 1},
            },
        },
    ]


class RealtimeMCPConfigTests(unittest.TestCase):
    """Validate the benchmark's mixed native tool configuration."""

    def test_mixed_function_and_mcp_tools_keep_only_function_handlers(self) -> None:
        """Keep MCP entries on the wire without creating local handlers."""
        payload = {
            "tools": [_function_tool(), _mcp_tool()],
            "handlers": {"local_lookup": _scripted_handler()},
            "tool_choice": "required",
            "parallel_tool_calls": True,
            "mcp_approval": _approval_policy(approve=True),
        }

        config = _load_config(payload)

        self.assertEqual(config.tools, tuple(payload["tools"]))
        self.assertEqual(set(config.handlers), {"local_lookup"})
        self.assertEqual(set(config.argument_validators), {"local_lookup"})

    def test_mcp_approval_policy_rejects_unknown_fields_and_invalid_types(self) -> None:
        """Reject ambiguous tester policy before opening a benchmark socket."""
        cases = (
            ({"default": "reject", "interactive": True}, "unknown field 'interactive'"),
            ({"default": True}, "default must be approve or reject"),
            ({"rules": {}}, "rules must be a JSON array"),
            (
                {
                    "rules": [
                        {
                            "server_label": "openai_docs",
                            "name": "search_openai_docs",
                            "approve": "yes",
                        }
                    ]
                },
                "approve must be a boolean",
            ),
        )
        for policy, message in cases:
            with self.subTest(policy=policy):
                payload = {
                    "tools": [_mcp_tool()],
                    "handlers": {},
                    "mcp_approval": policy,
                }
                with self.assertRaisesRegex(RealtimeClientError, message):
                    _load_config(payload)


class RealtimeMCPLifecycleTests(unittest.IsolatedAsyncioTestCase):
    """Validate hosted MCP discovery, approval, and response boundaries."""

    async def test_session_update_can_interleave_mcp_discovery_before_acknowledgement(self) -> None:
        """Wait for both session acknowledgement and all discovery terminals."""
        tool = {
            **_mcp_tool(),
            "server_url": "https://developers.openai.com/mcp?access_token=test-url-secret",
            "authorization": "test-authorization",
            "headers": {"X-API-Key": "test-key", "X-Tenant": "test-tenant"},
        }
        public_tool = {key: value for key, value in tool.items() if key not in {"authorization", "headers"}}
        config = RealtimeClientToolsConfig(
            tools=(tool,),
            handlers={},
            argument_validators={},
            tool_choice="required",
            parallel_tool_calls=True,
            sha256="fixture",
        )
        client = _client(config)
        websocket = _RecordingWebSocket()
        discovery_added = {
            "id": "item_discovery",
            "type": "mcp_list_tools",
            "server_label": "openai_docs",
            "tools": [],
        }
        discovery_done = {
            **discovery_added,
            "tools": [
                {
                    "name": "search_openai_docs",
                    "description": "Search docs",
                    "input_schema": {"type": "object"},
                }
            ],
        }
        raw_events = [
            {"type": "session.created", "session": _session(tools=[], tool_choice="auto")},
            {"type": "conversation.created", "conversation": {"id": "conv_mcp", "object": "realtime.conversation"}},
            {"type": "conversation.item.added", "previous_item_id": None, "item": discovery_added},
            {"type": "mcp_list_tools.in_progress", "item_id": "item_discovery"},
            {"type": "session.updated", "session": _session(tools=[public_tool])},
            {"type": "conversation.item.done", "item": discovery_done},
            {"type": "mcp_list_tools.completed", "item_id": "item_discovery"},
        ]
        base_time = time.time()
        queued = [
            _ReceivedEvent(event=event, received_at=base_time + index / 1000) for index, event in enumerate(raw_events)
        ]

        async def receive(_websocket: object) -> _ReceivedEvent:
            await asyncio.sleep(0)
            return queued.pop(0)

        client._receive_wire_event = receive

        await client._initialize_session(websocket)

        self.assertEqual(queued, [])
        self.assertEqual(websocket.events[0]["type"], "session.update")
        self.assertEqual(websocket.events[0]["session"]["tools"], [tool])
        captured_update = next(
            event
            for event in client.protocol_events
            if event["direction"] == "client" and event["type"] == "session.update"
        )
        captured_tool = captured_update["data"]["session"]["tools"][0]
        self.assertEqual(
            websocket.events[0]["session"]["tools"][0]["server_url"],
            "https://developers.openai.com/mcp?access_token=test-url-secret",
        )
        self.assertEqual(captured_tool["server_url"], "https://developers.openai.com/mcp?[REDACTED]")
        self.assertEqual(captured_tool["authorization"], "[REDACTED]")
        self.assertEqual(set(captured_tool["headers"].values()), {"[REDACTED]"})
        self.assertEqual(client._pending_events, {})

    async def _assert_approval_round(self, approve: bool) -> None:
        payload = {
            "tools": [_mcp_tool()],
            "handlers": {},
            "tool_choice": "required",
            "parallel_tool_calls": True,
            "mcp_approval": _approval_policy(approve=approve),
        }
        client = _client(_load_config(payload))
        client._conversation_id = "conv_mcp"
        client._configure_tool_ownership({"tools": [], "tool_choice": "auto", "parallel_tool_calls": True})
        websocket = _RecordingWebSocket()
        queued: list[_ReceivedEvent] = []
        received_types: list[str] = []
        stage = 0

        def enqueue(events: list[dict]) -> None:
            base_time = time.time()
            queued.extend(
                _ReceivedEvent(event=event, received_at=base_time + index / 1000)
                for index, event in enumerate(events, start=1)
            )

        async def next_event(_timeout: float) -> _ReceivedEvent:
            nonlocal stage
            if not queued and stage == 0:
                user_item = {
                    **websocket.events[0]["item"],
                    "object": "realtime.item",
                    "status": "completed",
                }
                enqueue(_mcp_response_events(user_item))
                stage += 1
            if not queued and stage == 1:
                [approval_item] = _created_items(websocket, "mcp_approval_response")
                self.assertIs(approval_item["approve"], approve)
                self.assertEqual(approval_item["approval_request_id"], "item_approval_request")
                self.assertEqual(sum(event["type"] == "response.create" for event in websocket.events), 1)
                enqueue(_mcp_completion_events(approval_item, approve=approve))
                stage += 1
            if not queued and stage == 2:
                response_creates = [event for event in websocket.events if event["type"] == "response.create"]
                self.assertEqual(len(response_creates), 2)
                self.assertEqual(response_creates[-1]["response"], {"tool_choice": "none"})
                [approval_item] = _created_items(websocket, "mcp_approval_response")
                enqueue(_final_response_events(approval_item["id"]))
                stage += 1
            if not queued:
                raise AssertionError("The client requested an unexpected additional event")
            received = queued.pop(0)
            received_types.append(received.event["type"])
            return received

        client._next_event = next_event
        turn = await client._run_turn(websocket, "Search the OpenAI docs.")

        self.assertEqual(turn.response_ids, ["resp_a", "resp_b"])
        self.assertEqual(turn.transcript, "Done.")
        self.assertEqual(stage, 3)
        self.assertEqual(_created_items(websocket, "function_call_output"), [])
        self.assertEqual(sum(event["type"] == "response.create" for event in websocket.events), 2)
        self.assertEqual(client._pending_events, {})
        response_a_done = received_types.index("response.done")
        terminal = received_types.index(f"response.mcp_call.{'completed' if approve else 'failed'}")
        output_done = received_types.index("response.output_item.done")
        response_b_created = received_types.index("response.created", response_a_done)
        self.assertLess(response_a_done, terminal)
        self.assertLess(terminal, output_done)
        self.assertLess(output_done, response_b_created)

    async def test_approval_policy_completes_mcp_round_without_function_output(self) -> None:
        """Approve exactly or reject by default, then request one Response B."""
        for approve in (True, False):
            with self.subTest(approve=approve):
                await self._assert_approval_round(approve)
