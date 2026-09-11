# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

# ruff: noqa: D100, D101, D102, D103

"""In-process canonical Realtime WebSocket tests (no live NIM)."""

from __future__ import annotations

import asyncio
import base64
import json
import unittest
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, patch

from pipecat.frames.frames import (
    InputAudioRawFrame,
    InterruptionFrame,
    LLMContextFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.frame_processor import FrameDirection
from realtime_helpers import FakeWebSocket

from realtime.controller import RealtimeSessionController
from realtime.frames import (
    RealtimeASREndpointSilenceFrame,
    RealtimeConversationAppendFrame,
    RealtimeIdleTimeoutFrame,
    RealtimeManualUserStartedSpeakingFrame,
    RealtimeManualUserStoppedSpeakingFrame,
    RealtimeResponseCreateFrame,
)
from realtime.gateway import handle_realtime_websocket
from realtime.protocol import build_server_event
from realtime.session import RealtimeSessionCapabilities
from realtime.transport import (
    bind_realtime_context,
    bind_realtime_deferred_service_responses,
    create_realtime_transport,
    realtime_controller,
    realtime_response_gate_processors,
    shutdown_realtime_transport,
)

_MODEL = "test-realtime-model"
_VOICE = "Magpie-Multilingual.EN-US.Aria"


def _sanitize(data: dict[str, Any], **_: Any) -> dict[str, Any]:
    config = dict(data)
    config.setdefault("pipeline_mode", "generic-assistant")
    config.setdefault("model_id", _MODEL)
    config.setdefault("tts_voice_id", _VOICE)
    return config


def _deferred_response_runtime(websocket: FakeWebSocket | None = None) -> SimpleNamespace:
    ws = websocket or FakeWebSocket([])
    controller = RealtimeSessionController(
        model=_MODEL,
        voice=_VOICE,
        runtime_config={"pipeline_mode": "generic-assistant"},
        capabilities=RealtimeSessionCapabilities(voices=frozenset({_VOICE})),
    )
    controller.apply_session_update({"output_modalities": ["text"]})
    transport = create_realtime_transport(ws, controller=controller)
    realtime_response_gate_processors(transport)
    context = LLMContext([])
    bind_realtime_context(transport, context)
    bound: dict[str, Any] = {}
    service = SimpleNamespace(bind_realtime_deferred_response_snapshot=lambda hook: bound.__setitem__("hook", hook))
    bind_realtime_deferred_service_responses(transport, service)
    return SimpleNamespace(
        ws=ws,
        controller=controller,
        transport=transport,
        context=context,
        prepare=bound["hook"],
    )


class RealtimeWebSocketTests(unittest.IsolatedAsyncioTestCase):
    def test_manual_boundary_frames_reject_invalid_audio_clocks(self) -> None:
        for frame_type in (
            RealtimeManualUserStartedSpeakingFrame,
            RealtimeManualUserStoppedSpeakingFrame,
        ):
            for cursor in (-1, True):
                with (
                    self.subTest(frame_type=frame_type.__name__, cursor=cursor),
                    self.assertRaisesRegex(ValueError, "audio_sample_cursor"),
                ):
                    frame_type(audio_sample_cursor=cursor, sample_rate=16_000)
            for sample_rate in (0, True):
                with (
                    self.subTest(frame_type=frame_type.__name__, sample_rate=sample_rate),
                    self.assertRaisesRegex(ValueError, "sample_rate"),
                ):
                    frame_type(audio_sample_cursor=0, sample_rate=sample_rate)

    async def test_connect_two_event_handshake_and_handoff(self) -> None:
        ws = FakeWebSocket(
            [
                json.dumps(
                    {
                        "type": "session.update",
                        "session": {
                            "type": "realtime",
                            "instructions": "Be brief.",
                            "max_output_tokens": 128,
                            "output_modalities": ["text"],
                            "audio": {
                                "input": {"format": {"type": "audio/pcm"}},
                                "output": {
                                    "format": {"type": "audio/pcm"},
                                    "voice": _VOICE,
                                },
                            },
                            "nvidia": {
                                "pipeline_mode": "generic-assistant",
                                "model_id": _MODEL,
                            },
                        },
                    }
                )
            ]
        )
        handed_off: list[tuple[dict[str, Any], RealtimeSessionController]] = []

        async def start_bot(websocket, config, controller) -> None:  # noqa: ARG001
            handed_off.append((config, controller))

        await handle_realtime_websocket(
            ws,
            sanitize_session_config=_sanitize,
            ensure_services_ready=lambda config: _ready(config),
            start_bot=start_bot,
            default_example_key="generic-assistant",
        )

        self.assertTrue(ws.accepted)
        self.assertEqual(
            [event["type"] for event in ws.sent],
            [
                "session.created",
                "conversation.created",
                "session.updated",
            ],
        )
        created = ws.sent[0]["session"]
        self.assertEqual(created["audio"]["input"]["format"]["rate"], 24000)
        self.assertEqual(created["audio"]["output"]["format"]["rate"], 24000)
        self.assertEqual(created["model"], _MODEL)
        self.assertEqual(created["audio"]["output"]["voice"], _VOICE)
        updated = ws.sent[-1]["session"]
        self.assertEqual(updated["instructions"], "Be brief.")
        self.assertEqual(updated["max_output_tokens"], 128)
        self.assertEqual(updated["output_modalities"], ["text"])
        self.assertEqual(updated["audio"]["input"]["format"], {"type": "audio/pcm", "rate": 24000})
        self.assertEqual(updated["audio"]["output"]["format"], {"type": "audio/pcm", "rate": 24000})
        self.assertEqual(len(handed_off), 1)
        runtime, controller = handed_off[0]
        self.assertEqual(runtime["prompt_content"], "Be brief.")
        self.assertEqual(runtime["max_tokens"], 128)
        self.assertEqual(runtime["output_modalities"], ["text"])
        self.assertEqual(controller.id, created["id"])
        self.assertEqual(controller.public_session()["audio"], updated["audio"])

    async def test_initial_explicit_empty_instructions_survive_catalog_sanitization(self) -> None:
        """The public and pipeline instruction values must remain identical."""
        ws = FakeWebSocket(
            [
                json.dumps(
                    {
                        "type": "session.update",
                        "session": {"type": "realtime", "instructions": ""},
                    }
                )
            ]
        )
        handed_off: list[tuple[dict[str, Any], RealtimeSessionController]] = []

        def hydrate_catalog_default(data: dict[str, Any], **_: Any) -> dict[str, Any]:
            config = _sanitize(data)
            if not config.get("prompt_content"):
                config["prompt_key"] = "catalog_default"
                config["prompt_content"] = "catalog prompt that must not be restored"
            return config

        async def start_bot(websocket, config, controller) -> None:  # noqa: ARG001
            handed_off.append((config, controller))

        await handle_realtime_websocket(
            ws,
            sanitize_session_config=hydrate_catalog_default,
            ensure_services_ready=lambda config: _ready(config),
            start_bot=start_bot,
            default_example_key="generic-assistant",
        )

        self.assertEqual(ws.sent[-1]["session"]["instructions"], "")
        self.assertEqual(len(handed_off), 1)
        runtime, controller = handed_off[0]
        self.assertEqual(runtime["prompt_content"], "")
        self.assertEqual(runtime["prompt_key"], "catalog_default")
        self.assertIs(runtime["_realtime_instructions_explicit"], True)
        self.assertEqual(controller.runtime_config["prompt_content"], "")

    async def test_optional_authorization_header_does_not_change_contract(self) -> None:
        ws = FakeWebSocket([json.dumps({"type": "session.update", "session": {"type": "realtime"}})])
        ws.headers["authorization"] = "Bearer optional-client-header"

        await handle_realtime_websocket(
            ws,
            sanitize_session_config=_sanitize,
            ensure_services_ready=lambda config: _ready(config),
        )

        self.assertEqual([event["type"] for event in ws.sent[:2]], ["session.created", "conversation.created"])
        self.assertEqual(ws.sent[2]["type"], "session.updated")

    async def test_oversized_initial_event_is_rejected_before_json_and_socket_can_retry(self) -> None:
        ws = FakeWebSocket(
            [
                json.dumps({"type": "future.event", "padding": "x" * 256}),
                json.dumps({"type": "session.update", "session": {"type": "realtime"}}),
            ]
        )
        handed_off: list[RealtimeSessionController] = []

        async def start_bot(websocket, config, controller) -> None:  # noqa: ARG001
            handed_off.append(controller)

        with patch("realtime.gateway.MAX_REALTIME_EVENT_BYTES", 128):
            await handle_realtime_websocket(
                ws,
                sanitize_session_config=_sanitize,
                ensure_services_ready=lambda config: _ready(config),
                start_bot=start_bot,
            )

        self.assertEqual(ws.sent[2]["type"], "error")
        self.assertEqual(ws.sent[2]["error"]["code"], "event_too_large")
        self.assertEqual(ws.sent[3]["type"], "session.updated")
        self.assertEqual(len(handed_off), 1)
        self.assertFalse(ws.closed)

    async def test_readiness_failure_keeps_socket_open_for_atomic_retry(self) -> None:
        rejected_update = {
            "type": "session.update",
            "session": {
                "type": "realtime",
                "instructions": "rejected instructions must not leak",
                "output_modalities": ["text"],
                "max_output_tokens": 64,
                "tools": [
                    {
                        "type": "function",
                        "name": "rejected_tool",
                        "parameters": {"type": "object", "properties": {}},
                    }
                ],
                "tool_choice": "required",
            },
        }
        accepted_update = {
            "type": "session.update",
            "session": {"type": "realtime", "instructions": "retry atomically"},
        }
        ws = FakeWebSocket(
            [
                json.dumps({**rejected_update, "event_id": "configure-1"}),
                json.dumps({**accepted_update, "event_id": "configure-2"}),
            ]
        )
        prepared_configs: list[dict[str, Any]] = []
        handed_off: list[tuple[dict[str, Any], RealtimeSessionController]] = []

        async def ensure_ready(config: dict[str, Any]) -> None:
            prepared_configs.append(config)
            if len(prepared_configs) == 1:
                raise RuntimeError("ASR not reachable")

        async def start_bot(websocket, config, controller) -> None:  # noqa: ARG001
            handed_off.append((config, controller))

        await handle_realtime_websocket(
            ws,
            sanitize_session_config=_sanitize,
            ensure_services_ready=ensure_ready,
            start_bot=start_bot,
        )

        self.assertEqual([event["type"] for event in ws.sent[:2]], ["session.created", "conversation.created"])
        error = ws.sent[2]
        self.assertEqual(error["type"], "error")
        self.assertEqual(error["error"]["code"], "services_not_ready")
        self.assertEqual(error["error"]["event_id"], "configure-1")
        self.assertEqual(ws.sent[3]["type"], "session.updated")
        self.assertEqual(ws.sent[3]["session"]["instructions"], "retry atomically")
        self.assertEqual(len(prepared_configs), 2)
        self.assertEqual(prepared_configs[0]["prompt_content"], "rejected instructions must not leak")
        self.assertEqual(prepared_configs[0]["output_modalities"], ["text"])
        self.assertEqual(prepared_configs[0]["max_tokens"], 64)
        self.assertEqual([tool["name"] for tool in prepared_configs[0]["client_tools"]], ["rejected_tool"])

        self.assertEqual(len(handed_off), 1)
        runtime, controller = handed_off[0]
        public = controller.public_session()
        self.assertEqual(public["instructions"], "retry atomically")
        self.assertEqual(public["output_modalities"], ["audio"])
        self.assertEqual(public["max_output_tokens"], "inf")
        self.assertEqual(public["tools"], [])
        self.assertEqual(public["tool_choice"], "auto")
        self.assertEqual(runtime["prompt_content"], "retry atomically")
        self.assertEqual(runtime["output_modalities"], ["audio"])
        self.assertIsNone(runtime["max_tokens"])
        self.assertEqual(runtime["client_tools"], [])
        self.assertEqual(runtime["tool_choice"], "auto")
        self.assertFalse(ws.closed)

    async def test_requested_unknown_model_fails_closed(self) -> None:
        ws = FakeWebSocket([])
        ws.query_params["model"] = "unknown-model"

        await handle_realtime_websocket(ws, sanitize_session_config=_sanitize)

        self.assertEqual(len(ws.sent), 1)
        self.assertEqual(ws.sent[0]["type"], "error")
        self.assertEqual(ws.sent[0]["error"]["code"], "model_not_available")
        self.assertTrue(ws.closed)
        self.assertEqual(ws.close_code, 1008)
        self.assertEqual(ws.close_reason, "invalid realtime model")

    async def test_transport_constructor_binds_exact_controller(self) -> None:
        ws = FakeWebSocket([])
        controller = RealtimeSessionController(
            model=_MODEL,
            voice=_VOICE,
            runtime_config={"pipeline_mode": "generic-assistant"},
        )

        transport = create_realtime_transport(ws, controller=controller)
        try:
            self.assertIs(realtime_controller(transport), controller)
        finally:
            shutdown_realtime_transport(transport)

    async def test_unrecoverable_serializer_failure_closes_transport(self) -> None:
        ws = FakeWebSocket([])
        controller = RealtimeSessionController(
            model=_MODEL,
            voice=_VOICE,
            runtime_config={"pipeline_mode": "generic-assistant"},
        )
        transport = create_realtime_transport(ws, controller=controller)
        try:
            serializer = transport.input()._params.serializer

            await serializer._retire_failed_client_tool_connection()

            self.assertTrue(serializer.connection_closed)
            self.assertTrue(ws.closed)
            self.assertEqual(ws.close_code, 1011)
            self.assertEqual(ws.close_reason, "client tool context failed")
        finally:
            shutdown_realtime_transport(transport)

    async def test_transport_marks_response_done_only_after_wire_send(self) -> None:
        ws = FakeWebSocket([])
        controller = RealtimeSessionController(
            model=_MODEL,
            voice=_VOICE,
            runtime_config={"pipeline_mode": "generic-assistant"},
        )
        controller.apply_session_update(
            {
                "type": "realtime",
                "tools": [
                    {
                        "type": "function",
                        "name": "lookup",
                        "parameters": {"type": "object", "properties": {}},
                    }
                ],
            }
        )
        controller.start_function_call(call_id="call_wire", name="lookup", arguments={})
        record = controller.tool_call("call_wire")
        response_id = record.response_id
        transport = create_realtime_transport(ws, controller=controller)
        try:
            serializer = transport.input()._params.serializer
            events = controller.finish_response(status="completed")
            self.assertFalse(record.response_done_published)

            await serializer.emit_batch(events)

            self.assertEqual(ws.sent[-1]["type"], "response.done")
            self.assertEqual(ws.sent[-1]["response"]["id"], response_id)
            self.assertTrue(record.response_done_published)
        finally:
            shutdown_realtime_transport(transport)

    async def test_deferred_response_queues_its_owned_start_inside_the_atomic_transition(self) -> None:
        runtime = _deferred_response_runtime()
        ws, controller, transport, context = runtime.ws, runtime.controller, runtime.transport, runtime.context

        try:
            prepared = await runtime.prepare(context)
            self.assertIsNotNone(prepared)
            _, _, activate, abort = prepared
            callback_response_ids: list[str] = []

            async def queue_owned_start(response_id: str) -> None:
                self.assertTrue(controller.response_transition_lock.locked())
                self.assertEqual(ws.sent[-1]["type"], "response.created")
                self.assertEqual(ws.sent[-1]["response"]["id"], response_id)
                callback_response_ids.append(response_id)

            try:
                response_id = await activate(queue_owned_start)
            finally:
                if controller.active_response_id is None:
                    abort()
            self.assertEqual(callback_response_ids, [response_id])
            self.assertEqual(controller.active_response_id, response_id)

            terminal = controller.finish_response(status="completed")
            await transport.input()._params.serializer.emit_batch(terminal)
            self.assertEqual(ws.sent[-1]["type"], "response.done")
        finally:
            shutdown_realtime_transport(transport)

    async def test_deferred_response_queue_failure_gets_a_terminal_failed_response(self) -> None:
        runtime = _deferred_response_runtime()
        ws, controller, transport, context = runtime.ws, runtime.controller, runtime.transport, runtime.context

        async def fail_to_queue(_response_id: str) -> None:
            raise RuntimeError("queue closed")

        try:
            prepared = await runtime.prepare(context)
            self.assertIsNotNone(prepared)
            _, _, activate, abort = prepared
            with self.assertRaisesRegex(RuntimeError, "queue closed"):
                try:
                    await activate(fail_to_queue)
                finally:
                    abort()

            self.assertIsNone(controller.active_response_id)
            self.assertEqual(
                [event["type"] for event in ws.sent],
                ["response.created", "error", "response.done"],
            )
            self.assertEqual(ws.sent[-1]["response"]["status"], "failed")
            self.assertEqual(ws.sent[-1]["response"]["status_details"]["error"]["code"], "response_start_failed")
        finally:
            shutdown_realtime_transport(transport)

    async def test_closed_wire_during_response_created_never_starts_provider_work(self) -> None:
        ws = FakeWebSocket([])
        ws.send_text = AsyncMock(side_effect=RuntimeError("WebSocket is closed"))
        runtime = _deferred_response_runtime(ws)
        controller, transport, context = runtime.controller, runtime.transport, runtime.context
        provider_start = AsyncMock()

        try:
            prepared = await runtime.prepare(context)
            self.assertIsNotNone(prepared)
            _, _, activate, abort = prepared
            try:
                response_id = await activate(provider_start)
            finally:
                abort()

            self.assertIsNone(response_id)
            provider_start.assert_not_awaited()
            self.assertIsNone(controller.active_response_id)
            self.assertTrue(transport.input()._params.serializer.connection_closed)
            self.assertEqual(ws.sent, [])
        finally:
            shutdown_realtime_transport(transport)

    async def test_unowned_cancellation_during_provider_start_is_a_failed_response(self) -> None:
        runtime = _deferred_response_runtime()
        ws, controller, transport, context = runtime.ws, runtime.controller, runtime.transport, runtime.context

        async def cancelled_provider_start(_response_id: str) -> None:
            raise asyncio.CancelledError

        try:
            prepared = await runtime.prepare(context)
            self.assertIsNotNone(prepared)
            _, _, activate, abort = prepared
            with self.assertRaises(asyncio.CancelledError):
                try:
                    await activate(cancelled_provider_start)
                finally:
                    abort()

            self.assertIsNone(controller.active_response_id)
            self.assertEqual([event["type"] for event in ws.sent], ["response.created", "response.done"])
            done = ws.sent[-1]
            self.assertEqual(done["response"]["status"], "failed")
            self.assertEqual(
                done["response"]["status_details"]["error"]["code"],
                "response_start_cancelled",
            )
        finally:
            shutdown_realtime_transport(transport)

    async def test_owned_cancellation_during_provider_start_uses_its_canonical_reason(self) -> None:
        async def run_case(cancel_reason: str) -> None:
            runtime = _deferred_response_runtime()
            ws, controller, transport, context = runtime.ws, runtime.controller, runtime.transport, runtime.context

            async def cancelled_provider_start(response_id: str) -> None:
                controller.observe_interruption(
                    InterruptionFrame().id,
                    reason=cancel_reason,
                    response_id=response_id,
                )
                raise asyncio.CancelledError

            try:
                prepared = await runtime.prepare(context)
                self.assertIsNotNone(prepared)
                _, _, activate, abort = prepared
                with self.assertRaises(asyncio.CancelledError):
                    try:
                        await activate(cancelled_provider_start)
                    finally:
                        abort()

                done = ws.sent[-1]
                self.assertEqual(done["type"], "response.done")
                self.assertEqual(done["response"]["status"], "cancelled")
                self.assertEqual(done["response"]["status_details"]["reason"], cancel_reason)
            finally:
                shutdown_realtime_transport(transport)

        for cancel_reason in ("client_cancelled", "turn_detected"):
            with self.subTest(cancel_reason=cancel_reason):
                await run_case(cancel_reason)

    async def test_transport_idle_timeout_commits_empty_audio_before_next_response(self) -> None:
        ws = FakeWebSocket([])
        controller = RealtimeSessionController(
            model=_MODEL,
            voice=_VOICE,
            runtime_config={"pipeline_mode": "generic-assistant"},
            capabilities=RealtimeSessionCapabilities(voices=frozenset({_VOICE})),
        )
        controller.apply_session_update(
            {
                "audio": {
                    "input": {
                        "turn_detection": {
                            "type": "server_vad",
                            "idle_timeout_ms": 5_000,
                        }
                    }
                }
            }
        )
        transport = create_realtime_transport(ws, controller=controller)
        response_gate = realtime_response_gate_processors(transport)[0]
        context = LLMContext([])
        bind_realtime_context(transport, context)
        serializer = transport.input()._params.serializer
        delays: list[float] = []
        pushed_frames = []

        async def immediate_sleep(delay: float) -> None:
            delays.append(delay)
            if delay == 5.0:
                serializer._idle_timeout.record_input_audio(sample_count=1_200, sample_rate=24_000)

        async def push_frame(frame, *_args, **_kwargs) -> None:
            pushed_frames.append(frame)

        serializer._idle_timeout._sleep = immediate_sleep
        try:
            input_pcm = b"\x00\x00" * 2_400
            input_frame = await serializer.deserialize(
                json.dumps(
                    {
                        "type": "input_audio_buffer.append",
                        "audio": base64.b64encode(input_pcm).decode("ascii"),
                    }
                )
            )
            self.assertIsInstance(input_frame, InputAudioRawFrame)
            events = controller.start_response()
            events.extend(
                controller.output_audio_delta_event(
                    base64.b64encode(b"\x00\x00" * 2_400).decode("ascii"),
                    sample_count=2_400,
                    sample_rate=24_000,
                )
            )
            events.extend(controller.finish_response(status="completed"))
            # This audio arrived while the paced assistant response was still
            # playing, before response.done crossed the output boundary.
            serializer._idle_timeout.record_input_audio(sample_count=2_400, sample_rate=24_000)

            with patch.object(transport.input(), "push_frame", side_effect=push_frame):
                await serializer.emit_batch(events)
                for _ in range(5):
                    await asyncio.sleep(0)

            self.assertEqual(len(pushed_frames), 1)
            self.assertIsInstance(pushed_frames[0], RealtimeIdleTimeoutFrame)
            self.assertEqual(pushed_frames[0].preceding_assistant_item_id, controller.last_assistant_item_id)
            done_index = [event["type"] for event in ws.sent].index("response.done")
            idle_turn = asyncio.create_task(response_gate.process_frame(pushed_frames[0], FrameDirection.DOWNSTREAM))
            await asyncio.sleep(0)
            self.assertFalse(idle_turn.done())
            self.assertEqual([event["type"] for event in ws.sent[done_index + 1 :]], [])

            assistant_context_message = {"role": "assistant", "content": ""}
            context.add_message(assistant_context_message)
            self.assertTrue(serializer.bind_latest_assistant_context_message())
            await idle_turn

            event_types = [event["type"] for event in ws.sent]
            self.assertEqual(
                event_types[done_index + 1 :],
                [
                    "input_audio_buffer.timeout_triggered",
                    "input_audio_buffer.committed",
                    "conversation.item.added",
                    "conversation.item.done",
                    "response.created",
                ],
            )
            self.assertEqual(delays, [5.0])
            timeout_event = ws.sent[done_index + 1]
            self.assertEqual(timeout_event["audio_start_ms"], 200)
            self.assertEqual(timeout_event["audio_end_ms"], 250)
            self.assertEqual(
                context.get_messages(),
                [assistant_context_message, {"role": "user", "content": ""}],
            )
            self.assertIs(serializer._context_messages_by_item_id[timeout_event["item_id"]], context.get_messages()[1])
            self.assertTrue(serializer._context_applied_events[timeout_event["item_id"]].is_set())
            response_id = ws.sent[-1]["response"]["id"]
            self.assertEqual(controller.active_response_id, response_id)
            self.assertEqual(response_gate._running_response_id, response_id)
        finally:
            shutdown_realtime_transport(transport)

    async def _prepare_queued_idle_timeout(self):
        ws = FakeWebSocket([])
        controller = RealtimeSessionController(
            model=_MODEL,
            voice=_VOICE,
            runtime_config={"pipeline_mode": "generic-assistant"},
            capabilities=RealtimeSessionCapabilities(voices=frozenset({_VOICE})),
        )
        controller.apply_session_update(
            {
                "audio": {
                    "input": {
                        "turn_detection": {
                            "type": "server_vad",
                            "idle_timeout_ms": 5_000,
                        }
                    }
                }
            }
        )
        transport = create_realtime_transport(ws, controller=controller)
        response_gate = realtime_response_gate_processors(transport)[0]
        context = LLMContext([])
        bind_realtime_context(transport, context)
        serializer = transport.input()._params.serializer
        serializer._idle_timeout._sleep = AsyncMock(return_value=None)
        pushed_frames = []

        async def capture_frame(frame, *_args, **_kwargs) -> None:
            pushed_frames.append(frame)

        events = controller.start_response()
        events.extend(
            controller.output_audio_delta_event(
                base64.b64encode(b"\x00\x00" * 2_400).decode("ascii"),
                sample_count=2_400,
                sample_rate=24_000,
            )
        )
        events.extend(controller.finish_response(status="completed"))
        assistant_context_message = {"role": "assistant", "content": ""}
        context.add_message(assistant_context_message)
        self.assertTrue(serializer.bind_latest_assistant_context_message())
        with patch.object(transport.input(), "push_frame", side_effect=capture_frame):
            await serializer.emit_batch(events)
            for _ in range(5):
                await asyncio.sleep(0)

        self.assertEqual(len(pushed_frames), 1)
        self.assertIsInstance(pushed_frames[0], RealtimeIdleTimeoutFrame)
        return ws, controller, transport, response_gate, context, serializer, pushed_frames[0]

    async def test_stale_queued_idle_timeout_does_not_mutate_context_or_start_response(self) -> None:
        ws, controller, transport, response_gate, context, serializer, frame = await self._prepare_queued_idle_timeout()
        initial_events = list(ws.sent)
        initial_items = controller.conversation.ordered_item_ids()
        try:
            serializer._idle_timeout.observe_published_events([{"type": "input_audio_buffer.speech_started"}])
            with patch.object(response_gate, "push_frame", new_callable=AsyncMock) as push_response:
                await response_gate.process_frame(frame, FrameDirection.DOWNSTREAM)

            self.assertEqual(ws.sent, initial_events)
            self.assertEqual(controller.conversation.ordered_item_ids(), initial_items)
            self.assertEqual(context.get_messages(), [{"role": "assistant", "content": ""}])
            self.assertIsNone(controller.active_response_id)
            self.assertFalse(serializer.connection_closed)
            push_response.assert_not_awaited()
        finally:
            shutdown_realtime_transport(transport)

    async def test_queued_idle_timeout_wakes_when_generation_changes_during_context_wait(self) -> None:
        ws, controller, transport, response_gate, context, serializer, frame = await self._prepare_queued_idle_timeout()
        initial_events = list(ws.sent)
        try:
            serializer._context_messages_by_item_id.pop(frame.preceding_assistant_item_id)
            serializer._context_applied_events[frame.preceding_assistant_item_id] = asyncio.Event()
            context.set_messages([])
            idle_turn = asyncio.create_task(response_gate.process_frame(frame, FrameDirection.DOWNSTREAM))
            await asyncio.sleep(0)
            self.assertFalse(idle_turn.done())

            serializer._idle_timeout.observe_published_events([{"type": "input_audio_buffer.speech_started"}])
            await asyncio.wait_for(idle_turn, timeout=1.0)

            self.assertEqual(ws.sent, initial_events)
            self.assertEqual(context.get_messages(), [])
            self.assertIsNone(controller.active_response_id)
            self.assertFalse(serializer.connection_closed)
        finally:
            shutdown_realtime_transport(transport)

    async def test_preceding_assistant_edit_supersedes_queued_idle_timeout(self) -> None:
        for edit in ("delete", "truncate"):
            with self.subTest(edit=edit):
                (
                    ws,
                    controller,
                    transport,
                    response_gate,
                    context,
                    serializer,
                    frame,
                ) = await self._prepare_queued_idle_timeout()
                event_count = len(ws.sent)
                try:
                    if edit == "delete":
                        event = {
                            "type": "conversation.item.delete",
                            "item_id": frame.preceding_assistant_item_id,
                        }
                        expected_event = "conversation.item.deleted"
                    else:
                        event = {
                            "type": "conversation.item.truncate",
                            "item_id": frame.preceding_assistant_item_id,
                            "content_index": 0,
                            "audio_end_ms": 0,
                        }
                        expected_event = "conversation.item.truncated"
                    self.assertIsNone(await serializer.deserialize(json.dumps(event)))

                    with patch.object(response_gate, "push_frame", new_callable=AsyncMock) as push_response:
                        await response_gate.process_frame(frame, FrameDirection.DOWNSTREAM)

                    self.assertEqual(
                        [published["type"] for published in ws.sent[event_count:]],
                        [expected_event],
                    )
                    self.assertEqual(context.get_messages(), [])
                    self.assertIsNone(controller.active_response_id)
                    self.assertFalse(serializer.connection_closed)
                    self.assertFalse(ws.closed)
                    push_response.assert_not_awaited()
                finally:
                    shutdown_realtime_transport(transport)

    async def test_in_flight_assistant_edit_supersedes_queued_idle_timeout(self) -> None:
        for edit in ("delete", "truncate"):
            with self.subTest(edit=edit):
                (
                    ws,
                    controller,
                    transport,
                    response_gate,
                    context,
                    serializer,
                    frame,
                ) = await self._prepare_queued_idle_timeout()
                event_count = len(ws.sent)
                edit_send_started = asyncio.Event()
                release_edit_send = asyncio.Event()
                original_send = ws.send_text

                if edit == "delete":
                    client_event = {
                        "type": "conversation.item.delete",
                        "item_id": frame.preceding_assistant_item_id,
                    }
                    expected_event = "conversation.item.deleted"
                else:
                    client_event = {
                        "type": "conversation.item.truncate",
                        "item_id": frame.preceding_assistant_item_id,
                        "content_index": 0,
                        "audio_end_ms": 0,
                    }
                    expected_event = "conversation.item.truncated"

                async def block_edit_publication(
                    data: str,
                    *,
                    event_type: str = expected_event,
                    send_started: asyncio.Event = edit_send_started,
                    release_send: asyncio.Event = release_edit_send,
                    send: Any = original_send,
                ) -> None:
                    if json.loads(data)["type"] == event_type:
                        send_started.set()
                        await release_send.wait()
                    await send(data)

                ws.send_text = block_edit_publication
                edit_task = asyncio.create_task(serializer.deserialize(json.dumps(client_event)))
                try:
                    await asyncio.wait_for(edit_send_started.wait(), timeout=1.0)
                    with patch.object(response_gate, "push_frame", new_callable=AsyncMock) as push_response:
                        idle_task = asyncio.create_task(response_gate.process_frame(frame, FrameDirection.DOWNSTREAM))
                        await asyncio.sleep(0)
                        self.assertFalse(idle_task.done())

                        release_edit_send.set()
                        edit_result, idle_result = await asyncio.wait_for(
                            asyncio.gather(edit_task, idle_task),
                            timeout=1.0,
                        )

                    self.assertIsNone(edit_result)
                    self.assertIsNone(idle_result)
                    self.assertEqual(
                        [event["type"] for event in ws.sent[event_count:]],
                        [expected_event],
                    )
                    self.assertEqual(context.get_messages(), [])
                    self.assertIsNone(controller.active_response_id)
                    self.assertFalse(serializer.connection_closed)
                    self.assertFalse(ws.closed)
                    push_response.assert_not_awaited()
                finally:
                    release_edit_send.set()
                    if not edit_task.done():
                        edit_task.cancel()
                    await asyncio.gather(edit_task, return_exceptions=True)
                    shutdown_realtime_transport(transport)

    async def test_pending_client_item_supersedes_queued_idle_timeout(self) -> None:
        (
            ws,
            controller,
            transport,
            response_gate,
            context,
            serializer,
            idle_frame,
        ) = await self._prepare_queued_idle_timeout()
        try:
            client_frame = await serializer.deserialize(
                json.dumps(
                    {
                        "type": "conversation.item.create",
                        "item": {
                            "type": "message",
                            "role": "user",
                            "content": [{"type": "input_text", "text": "Hello"}],
                        },
                    }
                )
            )
            self.assertIsInstance(client_frame, RealtimeConversationAppendFrame)
            event_count = len(ws.sent)

            await response_gate.process_frame(idle_frame, FrameDirection.DOWNSTREAM)
            self.assertEqual(len(ws.sent), event_count)
            self.assertIsNone(controller.active_response_id)

            await response_gate.process_frame(client_frame, FrameDirection.DOWNSTREAM)
            self.assertEqual(
                [event["type"] for event in ws.sent[event_count:]],
                ["conversation.item.added", "conversation.item.done"],
            )
            self.assertEqual(
                context.get_messages(),
                [
                    {"role": "assistant", "content": ""},
                    {"role": "user", "content": "Hello"},
                ],
            )
            self.assertFalse(serializer.connection_closed)
        finally:
            shutdown_realtime_transport(transport)

    async def test_idle_timeout_context_failures_roll_back_and_close(self) -> None:
        for failure_point in ("add", "bind"):
            with self.subTest(failure_point=failure_point):
                (
                    ws,
                    controller,
                    transport,
                    response_gate,
                    context,
                    serializer,
                    frame,
                ) = await self._prepare_queued_idle_timeout()
                initial_event_count = len(ws.sent)
                initial_items = controller.conversation.ordered_item_ids()
                target = context if failure_point == "add" else serializer
                method = "add_message" if failure_point == "add" else "bind_conversation_context_message"
                try:
                    with (
                        patch.object(target, method, side_effect=RuntimeError("injected context failure")),
                        patch.object(response_gate, "push_frame", new_callable=AsyncMock) as push_response,
                    ):
                        await response_gate.process_frame(frame, FrameDirection.DOWNSTREAM)

                    self.assertEqual(
                        [event["type"] for event in ws.sent[initial_event_count:]],
                        ["error"],
                    )
                    self.assertEqual(ws.sent[-1]["error"]["code"], "idle_timeout_error")
                    self.assertEqual(controller.conversation.ordered_item_ids(), initial_items)
                    self.assertEqual(context.get_messages(), [{"role": "assistant", "content": ""}])
                    self.assertIsNone(controller.active_response_id)
                    self.assertTrue(serializer.connection_closed)
                    self.assertTrue(ws.closed)
                    self.assertEqual(ws.close_code, 1011)
                    self.assertEqual(ws.close_reason, "server-VAD idle timeout failed")
                    push_response.assert_not_awaited()
                finally:
                    shutdown_realtime_transport(transport)

    async def test_cancelled_idle_publication_terminalizes_response_and_closes(self) -> None:
        (
            ws,
            controller,
            transport,
            response_gate,
            _context,
            serializer,
            frame,
        ) = await self._prepare_queued_idle_timeout()
        event_count = len(ws.sent)
        send_started = asyncio.Event()
        release_send = asyncio.Event()
        original_send = ws.send_text

        async def block_first_idle_event(data: str) -> None:
            event = json.loads(data)
            if event["type"] == "input_audio_buffer.timeout_triggered":
                send_started.set()
                await release_send.wait()
            await original_send(data)

        ws.send_text = block_first_idle_event
        try:
            with patch.object(response_gate, "push_frame", new_callable=AsyncMock) as push_response:
                idle_turn = asyncio.create_task(response_gate.process_frame(frame, FrameDirection.DOWNSTREAM))
                await send_started.wait()
                idle_turn.cancel()
                release_send.set()
                with self.assertRaises(asyncio.CancelledError):
                    await idle_turn

            published = ws.sent[event_count:]
            self.assertEqual(
                [event["type"] for event in published],
                [
                    "input_audio_buffer.timeout_triggered",
                    "input_audio_buffer.committed",
                    "conversation.item.added",
                    "conversation.item.done",
                    "response.created",
                    "error",
                    "response.done",
                ],
            )
            response_id = published[4]["response"]["id"]
            self.assertEqual(published[5]["error"]["code"], "idle_timeout_error")
            self.assertEqual(published[6]["response"]["id"], response_id)
            self.assertEqual(published[6]["response"]["status"], "failed")
            self.assertIsNone(controller.active_response_id)
            self.assertIsNone(response_gate._pending_response_marker_id)
            self.assertIsNone(response_gate._running_response_id)
            self.assertTrue(serializer.connection_closed)
            self.assertTrue(ws.closed)
            self.assertEqual(ws.close_code, 1011)
            push_response.assert_not_awaited()
        finally:
            shutdown_realtime_transport(transport)

    async def test_manual_commit_publication_precedes_immediate_response_create(self) -> None:
        ws = FakeWebSocket([])
        controller = RealtimeSessionController(
            model=_MODEL,
            voice=_VOICE,
            runtime_config={"pipeline_mode": "generic-assistant", "asr_server": "localhost:50051"},
            input_transcription_model="test-asr",
            capabilities=RealtimeSessionCapabilities(
                voices=frozenset({_VOICE}),
                supports_manual_input=True,
            ),
        )
        controller.apply_session_update(
            {
                "type": "realtime",
                "audio": {"input": {"turn_detection": None}},
            }
        )
        transport = create_realtime_transport(ws, controller=controller)
        realtime_response_gate_processors(transport)
        serializer = transport.input()._params.serializer
        committed_send_started = asyncio.Event()
        allow_committed_send = asyncio.Event()
        original_send_text = ws.send_text

        async def blocking_send_text(data: str) -> None:
            if json.loads(data).get("type") == "input_audio_buffer.committed":
                committed_send_started.set()
                await allow_committed_send.wait()
            await original_send_text(data)

        ws.send_text = blocking_send_text
        item_id = "item_manual_commit"
        item = {
            "id": item_id,
            "object": "realtime.item",
            "type": "message",
            "status": "completed",
            "role": "user",
            "content": [{"type": "input_audio", "transcript": None}],
        }
        pushed_frames = []

        async def push_manual_frame(frame, *_args, **_kwargs) -> None:
            pushed_frames.append(frame)
            if isinstance(frame, UserStoppedSpeakingFrame):
                await serializer.emit_batch(
                    [
                        build_server_event(
                            "input_audio_buffer.committed",
                            item_id=item_id,
                            previous_item_id=None,
                        ),
                        build_server_event(
                            "conversation.item.added",
                            previous_item_id=None,
                            item=item,
                        ),
                        build_server_event(
                            "conversation.item.done",
                            previous_item_id=None,
                            item=item,
                        ),
                    ]
                )

        try:
            pcm = b"\x01\x00" * 480
            await serializer.deserialize(
                json.dumps(
                    {
                        "type": "input_audio_buffer.append",
                        "audio": base64.b64encode(pcm).decode("ascii"),
                    }
                )
            )
            with patch.object(transport.input(), "push_frame", side_effect=push_manual_frame):
                commit_task = asyncio.create_task(
                    serializer.deserialize(json.dumps({"type": "input_audio_buffer.commit"}))
                )
                await asyncio.wait_for(committed_send_started.wait(), timeout=1)
                self.assertFalse(commit_task.done())
                allow_committed_send.set()
                await asyncio.wait_for(commit_task, timeout=1)

            self.assertEqual(len(pushed_frames), 4)
            self.assertIsInstance(pushed_frames[0], RealtimeManualUserStartedSpeakingFrame)
            self.assertEqual(pushed_frames[0].audio_sample_cursor, 0)
            self.assertEqual(pushed_frames[0].sample_rate, 16_000)
            self.assertIsInstance(pushed_frames[1], InputAudioRawFrame)
            self.assertNotIsInstance(pushed_frames[1], RealtimeASREndpointSilenceFrame)
            self.assertIsInstance(pushed_frames[2], RealtimeASREndpointSilenceFrame)
            self.assertEqual(pushed_frames[2].sample_rate, 16_000)
            self.assertEqual(len(pushed_frames[2].audio), 32_000)
            self.assertIsInstance(pushed_frames[3], RealtimeManualUserStoppedSpeakingFrame)
            self.assertEqual(pushed_frames[3].audio_sample_cursor, len(pushed_frames[1].audio) // 2)
            self.assertEqual(pushed_frames[3].sample_rate, 16_000)

            response_frame = await serializer.deserialize(json.dumps({"type": "response.create"}))
            self.assertIsInstance(response_frame, RealtimeResponseCreateFrame)
            self.assertNotIn("response.created", [event["type"] for event in ws.sent])

            await serializer.emit_batch(
                [
                    build_server_event(
                        "conversation.item.input_audio_transcription.completed",
                        item_id=item_id,
                        content_index=0,
                        transcript="hello",
                    )
                ]
            )

            self.assertEqual(
                [event["type"] for event in ws.sent],
                [
                    "input_audio_buffer.committed",
                    "conversation.item.added",
                    "conversation.item.done",
                    "conversation.item.input_audio_transcription.completed",
                    "response.created",
                ],
            )
        finally:
            shutdown_realtime_transport(transport)

    async def test_manual_commit_cursor_advances_when_turn_enters_pipeline(self) -> None:
        ws = FakeWebSocket([])
        controller = RealtimeSessionController(
            model=_MODEL,
            voice=_VOICE,
            runtime_config={"pipeline_mode": "generic-assistant", "asr_server": "localhost:50051"},
            input_transcription_model="test-asr",
            capabilities=RealtimeSessionCapabilities(
                voices=frozenset({_VOICE}),
                supports_manual_input=True,
            ),
        )
        controller.apply_session_update(
            {
                "type": "realtime",
                "audio": {"input": {"turn_detection": None}},
            }
        )
        transport = create_realtime_transport(ws, controller=controller)
        response_gate = realtime_response_gate_processors(transport)[0]
        serializer = transport.input()._params.serializer
        pushed_frames = []
        commit_count = 0
        publish_commit = True

        async def push_manual_frame(frame, *_args, **_kwargs) -> None:
            nonlocal commit_count
            pushed_frames.append(frame)
            if publish_commit and isinstance(frame, RealtimeManualUserStoppedSpeakingFrame):
                commit_count += 1
                await serializer.emit_batch(
                    [
                        build_server_event(
                            "input_audio_buffer.committed",
                            item_id=f"item_{commit_count}",
                            previous_item_id=None,
                        )
                    ]
                )

        try:
            with patch.object(transport.input(), "push_frame", side_effect=push_manual_frame):
                first_pcm = b"\x01\x00" * 480
                await serializer.deserialize(
                    json.dumps(
                        {
                            "type": "input_audio_buffer.append",
                            "audio": base64.b64encode(first_pcm).decode("ascii"),
                        }
                    )
                )
                await serializer.deserialize(json.dumps({"type": "input_audio_buffer.commit"}))
                await response_gate.process_frame(
                    LLMContextFrame(LLMContext([{"role": "user", "content": "first"}])),
                    FrameDirection.DOWNSTREAM,
                )

                second_pcm = b"\x02\x00" * 720
                await serializer.deserialize(
                    json.dumps(
                        {
                            "type": "input_audio_buffer.append",
                            "audio": base64.b64encode(second_pcm).decode("ascii"),
                        }
                    )
                )
                publish_commit = False
                with patch("realtime.transport.asyncio.wait_for", new=AsyncMock(side_effect=TimeoutError)):
                    await serializer.deserialize(json.dumps({"type": "input_audio_buffer.commit"}))
                publish_commit = True
                await serializer.deserialize(json.dumps({"type": "input_audio_buffer.commit"}))

            starts = [frame for frame in pushed_frames if isinstance(frame, RealtimeManualUserStartedSpeakingFrame)]
            stops = [frame for frame in pushed_frames if isinstance(frame, RealtimeManualUserStoppedSpeakingFrame)]
            audio = [
                frame
                for frame in pushed_frames
                if isinstance(frame, InputAudioRawFrame) and not isinstance(frame, RealtimeASREndpointSilenceFrame)
            ]
            self.assertEqual(len(starts), 3)
            self.assertEqual(len(stops), 3)
            self.assertEqual(len(audio), 3)
            self.assertEqual(starts[0].audio_sample_cursor, 0)
            for index in range(3):
                self.assertEqual(
                    stops[index].audio_sample_cursor,
                    starts[index].audio_sample_cursor + len(audio[index].audio) // 2,
                )
            self.assertEqual(starts[1].audio_sample_cursor, stops[0].audio_sample_cursor)
            self.assertEqual(starts[2].audio_sample_cursor, stops[1].audio_sample_cursor)
            self.assertIn("event_processing_failed", [event.get("error", {}).get("code") for event in ws.sent])
        finally:
            shutdown_realtime_transport(transport)


async def _ready(config: dict[str, Any]) -> None:  # noqa: ARG001
    return None


if __name__ == "__main__":
    unittest.main()
