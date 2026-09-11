# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

# ruff: noqa: D100, D101, D102, D103

"""Realtime transcript ownership, response controls, and text items."""

from __future__ import annotations

import asyncio
import base64
import copy
import json
import time
import unittest
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

from pipecat.frames.frames import (
    ErrorFrame,
    FatalErrorFrame,
    FunctionCallInProgressFrame,
    InputAudioRawFrame,
    InterimTranscriptionFrame,
    InterruptionFrame,
    LLMConfigureOutputFrame,
    LLMContextFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMRunFrame,
    LLMTextFrame,
    OutputAudioRawFrame,
    TranscriptionFrame,
    TTSUpdateSettingsFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.observers.base_observer import FrameProcessed, FramePushed
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.processors.frameworks.rtvi.frames import RTVIServerMessageFrame
from pipecat.services.llm_service import LLMService
from pipecat.services.nvidia.llm import NvidiaLLMService as PipecatNvidiaLLMService
from pipecat.services.nvidia.stt import NvidiaSTTService as PipecatNvidiaSTTService
from pipecat.transports.base_output import BaseOutputTransport

import realtime.conversation as realtime_conversation
from examples.shared.frames import (
    USER_TRANSCRIPT_TURN_FRAME_ID_METADATA,
    UserTranscriptProducerEndedFrame,
)
from examples.shared.nvidia_llm import (
    NvidiaLLMService,
    NvidiaLLMSettings,
    _apply_realtime_completion_token_limit,
    _apply_realtime_parallel_tool_calls,
    _normalize_realtime_empty_tool_request,
    _truncate_realtime_context,
)
from realtime.asr import RealtimeASROwnershipError, RealtimeASRTurnOwnership, RealtimeNvidiaSTTService
from realtime.client_tools import (
    ClientToolBroker,
    ClientToolCancelledResult,
    ClientToolTimeoutResult,
)
from realtime.controller import RealtimeSessionController
from realtime.conversation import ConversationJournal
from realtime.frames import (
    RealtimeASREndpointSilenceFrame,
    RealtimeClientToolOutputFrame,
    RealtimeConversationAppendFrame,
    RealtimeDeferredResponseCreateFrame,
    RealtimeInputTranscriptionErrorFrame,
    RealtimeManualUserStartedSpeakingFrame,
    RealtimeManualUserStoppedSpeakingFrame,
    RealtimeOwnedLLMFullResponseStartFrame,
    RealtimeResponseContextFrame,
    RealtimeResponseCreateFrame,
    RealtimeResponseLLMContext,
    RealtimeResponseOrigin,
)
from realtime.observer import RealtimeLifecycleObserver
from realtime.protocol import RealtimeProtocolError
from realtime.serializer import RealtimeFrameSerializer
from realtime.session import AudioFormatCapability, RealtimeSessionCapabilities
from realtime.transport import RealtimeManualResponseGate, RealtimeManualTurnStopStrategy

_TOOL_SCHEMA = {
    "type": "function",
    "name": "lookup",
    "description": "Look up a value",
    "parameters": {"type": "object", "properties": {}},
}


def _controller(
    *,
    transcription: bool = True,
    fused: bool = False,
    output_kind: str = "audio",
    tool_owner: str | None = None,
    manual: bool = False,
) -> RealtimeSessionController:
    capabilities = RealtimeSessionCapabilities(supports_manual_input=True) if manual else None
    trusted_tools = None
    server_tools = None
    if tool_owner == "server":
        capabilities = RealtimeSessionCapabilities(trusted_function_tools=frozenset({"lookup"}))
        trusted_tools = [_TOOL_SCHEMA]
        server_tools = ["lookup"]
    controller = RealtimeSessionController(
        model="test-model",
        voice="test-voice",
        runtime_config={} if fused else {"asr_server": "localhost:50051"},
        input_transcription_model="test-asr" if transcription else None,
        server_tools=server_tools,
        trusted_tool_schemas=trusted_tools,
        capabilities=capabilities,
    )
    if output_kind == "text":
        controller.apply_session_update({"type": "realtime", "output_modalities": ["text"]})
    if tool_owner == "client":
        controller.apply_session_update({"type": "realtime", "tools": [_TOOL_SCHEMA]})
    if manual:
        controller.apply_session_update(
            {
                "type": "realtime",
                "audio": {"input": {"turn_detection": None}},
            }
        )
    return controller


class _Recorder:
    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    async def emit(self, event: dict[str, Any]) -> None:
        self.events.append(event)

    async def emit_batch(self, events: list[dict[str, Any]]) -> None:
        self.events.extend(events)

    @property
    def types(self) -> list[str]:
        return [event["type"] for event in self.events]


def _observer(
    recorder: _Recorder,
    controller: RealtimeSessionController,
    **kwargs: Any,
) -> RealtimeLifecycleObserver:
    return RealtimeLifecycleObserver(
        emit=recorder.emit,
        emit_batch=recorder.emit_batch,
        controller=controller,
        **kwargs,
    )


def _serializer(
    recorder: _Recorder,
    controller: RealtimeSessionController | None = None,
) -> RealtimeFrameSerializer:
    serializer = RealtimeFrameSerializer(controller=controller or _controller())
    serializer.set_emit(recorder.emit, recorder.emit_batch)
    return serializer


def _finish_playout_aligned_assistant(
    controller: RealtimeSessionController,
    *,
    transcript_fragments: tuple[str, ...] = ("Hello", "there"),
    context_fragments: tuple[str, ...] | None = None,
) -> str:
    """Finish audio with one 100 ms wire chunk before each text checkpoint."""
    context_fragments = context_fragments or transcript_fragments
    assert len(transcript_fragments) == len(context_fragments)
    controller.start_response()
    for fragment, context_fragment in zip(transcript_fragments, context_fragments, strict=True):
        controller.output_audio_delta_event(
            "encoded-audio",
            sample_count=2400,
            sample_rate=24000,
        )
        controller.append_assistant_audio_transcript(
            fragment,
            includes_inter_frame_spaces=False,
            context_text=context_fragment,
        )
    controller.output_audio_delta_event(
        "encoded-audio",
        sample_count=2400,
        sample_rate=24000,
    )
    controller.finish_response(status="completed")
    item_id = controller.last_assistant_item_id
    assert item_id is not None
    return item_id


def _serializer_with_active_client_tool(
    recorder: _Recorder,
) -> tuple[RealtimeFrameSerializer, RealtimeSessionController, Any]:
    controller = _controller(tool_owner="client")
    controller.start_function_call(call_id="call_client", name="lookup", arguments={"key": "value"})
    controller.finish_response(status="completed")
    serializer = _serializer(recorder, controller)
    broker = SimpleNamespace(
        stage_output=AsyncMock(),
        release_output=AsyncMock(),
        discard_staged_output=AsyncMock(),
        wait_context_applied=AsyncMock(),
    )
    serializer.set_client_tool_broker(broker)
    serializer.notify_response_done_published(controller.tool_call("call_client").response_id)
    return serializer, controller, broker


def _client_tool_params(
    results: list[Any],
    *,
    call_id: str = "call_client",
    name: str = "lookup",
) -> SimpleNamespace:
    async def result_callback(result: Any, *, properties: Any) -> None:
        results.append(result)
        if properties.on_context_updated is not None:
            await properties.on_context_updated()

    return SimpleNamespace(
        tool_call_id=call_id,
        function_name=name,
        result_callback=result_callback,
    )


def _bound_client_tool_broker(
    *call_ids: str,
    output_timeout_secs: float,
    context: LLMContext | None = None,
) -> ClientToolBroker:
    context = context or _client_tool_context(*call_ids)
    broker = ClientToolBroker(output_timeout_secs=output_timeout_secs)
    broker.bind_context(context)
    return broker


def _client_tool_context(*call_ids: str) -> LLMContext:
    return LLMContext(
        [
            {
                "role": "tool",
                "content": "IN_PROGRESS",
                "tool_call_id": call_id,
            }
            for call_id in call_ids
        ]
    )


def _install_client_tool_gate(
    serializer: RealtimeFrameSerializer,
    controller: RealtimeSessionController,
    context: LLMContext,
) -> RealtimeManualResponseGate:
    gate = RealtimeManualResponseGate(controller=controller)
    gate.push_frame = AsyncMock()
    serializer.bind_context(context, instructions_renderer=lambda _instructions: [])
    serializer.set_response_gate(gate)
    return gate


async def _push(
    observer: RealtimeLifecycleObserver,
    frame,
    *,
    source: FrameProcessor | Any,
    destination: FrameProcessor | Any,
    direction: FrameDirection = FrameDirection.DOWNSTREAM,
) -> None:
    await observer.on_push_frame(
        FramePushed(
            source=source,
            destination=destination,
            frame=frame,
            direction=direction,
            timestamp=0,
        )
    )


def _owned_by(frame: InterimTranscriptionFrame | TranscriptionFrame, boundary: FrameProcessor | Any):
    frame.metadata[USER_TRANSCRIPT_TURN_FRAME_ID_METADATA] = boundary.id
    return frame


async def _wait_until(predicate: Any) -> None:
    for _ in range(100):
        if predicate():
            return
        await asyncio.sleep(0)
    raise AssertionError("asynchronous condition did not become true")


class RealtimeControllerLifecycleTests(unittest.TestCase):
    def test_conversation_item_limit_is_bounded_for_the_session(self) -> None:
        with patch.object(realtime_conversation, "MAX_CONVERSATION_ITEMS", 2):
            journal = ConversationJournal()
            for item_id in ("item_a", "item_b"):
                journal.add_item(
                    {
                        "id": item_id,
                        "type": "message",
                        "role": "user",
                        "content": [{"type": "input_text", "text": item_id}],
                    }
                )
            journal.delete_item("item_a")
            with self.assertRaises(RealtimeProtocolError) as caught:
                journal.add_item(
                    {
                        "id": "item_c",
                        "type": "message",
                        "role": "user",
                        "content": [{"type": "input_text", "text": "new"}],
                    }
                )
        self.assertEqual(caught.exception.code, "conversation_limit_exceeded")

    def test_conversation_rejects_oversized_item_without_mutation(self) -> None:
        with patch.object(realtime_conversation, "MAX_CONVERSATION_ITEM_BYTES", 128):
            journal = ConversationJournal()
            with self.assertRaises(RealtimeProtocolError) as caught:
                journal.add_item(
                    {
                        "id": "item_large",
                        "type": "message",
                        "role": "user",
                        "content": [{"type": "input_text", "text": "x" * 256}],
                    }
                )
        self.assertEqual(caught.exception.code, "conversation_item_too_large")
        self.assertEqual(journal.ordered_item_ids(), ())

    def test_delete_releases_live_payload_budget(self) -> None:
        journal = ConversationJournal()
        first = {
            "id": "item_first",
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": "payload" * 20}],
        }
        second = copy.deepcopy(first)
        second["id"] = "item_second"
        journal.add_item(first)
        one_item_bytes = journal._live_item_bytes
        with patch.object(realtime_conversation, "MAX_LIVE_CONVERSATION_BYTES", one_item_bytes + 8):
            with self.assertRaises(RealtimeProtocolError) as caught:
                journal.add_item(second)
            journal.delete_item("item_first")
            journal.add_item(second)
        self.assertEqual(caught.exception.code, "conversation_limit_exceeded")
        self.assertEqual(journal.ordered_item_ids(), ("item_second",))

    def test_journal_history_is_bounded_and_does_not_retain_content(self) -> None:
        with patch.object(realtime_conversation, "MAX_JOURNAL_ENTRIES", 2):
            journal = ConversationJournal()
            journal.add_item(
                {
                    "id": "item_secret",
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "secret payload"}],
                }
            )
            journal.complete_item("item_secret")
            journal.delete_item("item_secret")
        entries = journal.entries()
        self.assertEqual([entry.sequence for entry in entries], [2, 3])
        self.assertEqual([entry.operation for entry in entries], ["done", "deleted"])
        self.assertTrue(all(entry.item is not None and "content" not in entry.item for entry in entries))

    def test_response_history_is_bounded_and_does_not_duplicate_output(self) -> None:
        with patch.object(realtime_conversation, "MAX_RESPONSE_RECORDS", 2):
            controller = _controller(output_kind="text")
            for response_text in ("first", "second"):
                controller.append_assistant_text(response_text)
                controller.finish_response(status="completed")
        records = controller.responses.records()
        self.assertEqual([record.generation for record in records], [2, 2])
        self.assertEqual([record.status for record in records], ["in_progress", "completed"])
        self.assertNotIn("content", records[-1].response["output"][0])

    def test_response_token_limit_projects_only_from_isolated_context(self) -> None:
        session_params = {"max_tokens": 256}
        _apply_realtime_completion_token_limit(session_params, LLMContext([]))
        self.assertEqual(session_params, {"max_tokens": 256})

        response_params = {"max_completion_tokens": 256, "max_tokens": 128}
        response_context = RealtimeResponseLLMContext([], max_output_tokens=32)
        _apply_realtime_completion_token_limit(response_params, response_context)
        self.assertEqual(response_params, {"max_completion_tokens": 32})

        unlimited_params = {"max_completion_tokens": 256, "max_tokens": 128}
        unlimited_context = RealtimeResponseLLMContext([], max_output_tokens="inf")
        _apply_realtime_completion_token_limit(unlimited_params, unlimited_context)
        self.assertEqual(unlimited_params, {})

    def test_parallel_tool_policy_projects_session_and_response_values_per_request(self) -> None:
        tools = [{"type": "function", "function": {"name": "lookup", "parameters": {}}}]
        session_params = {"tools": tools}
        effective = _apply_realtime_parallel_tool_calls(
            session_params,
            LLMContext([]),
            session_parallel_tool_calls=False,
        )
        self.assertFalse(effective)
        self.assertIs(session_params["parallel_tool_calls"], False)

        response_params = {"tools": tools, "parallel_tool_calls": False}
        response_context = RealtimeResponseLLMContext([], parallel_tool_calls=True)
        effective = _apply_realtime_parallel_tool_calls(
            response_params,
            response_context,
            session_parallel_tool_calls=False,
        )
        self.assertTrue(effective)
        self.assertIs(response_params["parallel_tool_calls"], True)

        no_tools_params = {"parallel_tool_calls": False}
        effective = _apply_realtime_parallel_tool_calls(
            no_tools_params,
            LLMContext([]),
            session_parallel_tool_calls=False,
        )
        self.assertFalse(effective)
        self.assertNotIn("parallel_tool_calls", no_tools_params)

    def test_empty_realtime_tools_keep_public_auto_but_omit_vacuous_provider_fields(self) -> None:
        for tool_choice in ("auto", "none"):
            with self.subTest(tool_choice=tool_choice):
                params = {
                    "messages": [{"role": "user", "content": "hello"}],
                    "tools": [],
                    "tool_choice": tool_choice,
                    "parallel_tool_calls": True,
                }
                context = RealtimeResponseLLMContext([], tool_choice=tool_choice)

                _normalize_realtime_empty_tool_request(params, context)

                self.assertEqual(params, {"messages": [{"role": "user", "content": "hello"}]})

    def test_empty_tool_normalization_is_realtime_only_and_keeps_forced_choices_strict(self) -> None:
        ordinary = {"tools": [], "tool_choice": "auto", "parallel_tool_calls": True}
        _normalize_realtime_empty_tool_request(ordinary, LLMContext([]))
        self.assertEqual(ordinary, {"tools": [], "tool_choice": "auto", "parallel_tool_calls": True})

        forced = {"tools": [], "tool_choice": "required"}
        _normalize_realtime_empty_tool_request(
            forced,
            RealtimeResponseLLMContext([], tool_choice="required"),
        )
        self.assertEqual(forced, {"tools": [], "tool_choice": "required"})

    def test_start_response_rejects_a_second_active_response(self) -> None:
        controller = _controller()
        created = controller.start_response()[0]
        with self.assertRaises(RealtimeProtocolError) as caught:
            controller.start_response()
        self.assertEqual(caught.exception.code, "response_in_progress")
        self.assertEqual(controller.active_response_id, created["response"]["id"])

    def test_finish_response_is_terminal_exactly_once(self) -> None:
        controller = _controller()
        controller.start_response()
        events = controller.finish_response(status="completed")
        self.assertEqual(events[-1]["type"], "response.done")
        self.assertEqual(events[-1]["response"]["status"], "completed")
        self.assertEqual(controller.finish_response(status="completed"), [])

    def test_new_response_has_an_independent_generation_and_id(self) -> None:
        controller = _controller()
        first = controller.start_response()[0]["response"]["id"]
        controller.finish_response(status="completed")
        second = controller.start_response()[0]["response"]["id"]
        self.assertNotEqual(first, second)
        records = controller.responses.records()
        self.assertEqual([record.status for record in records], ["in_progress", "completed", "in_progress"])

    def test_interruption_keeps_its_original_generation_and_response_owner(self) -> None:
        controller = _controller()
        response_a = controller.start_response()[0]["response"]["id"]
        interruption_a = InterruptionFrame()
        generation_a, first_observation = controller.observe_interruption(
            interruption_a.id,
            reason="client_cancelled",
            response_id=response_a,
        )
        self.assertTrue(first_observation)
        controller.finish_response(status="cancelled", reason="client_cancelled")

        response_b = controller.start_response()[0]["response"]["id"]
        interruption_b = InterruptionFrame()
        generation_b, _ = controller.observe_interruption(interruption_b.id)
        duplicate_generation, duplicate_observation = controller.observe_interruption(interruption_a.id)

        self.assertGreater(generation_b, generation_a)
        self.assertFalse(duplicate_observation)
        self.assertEqual(duplicate_generation, generation_a)
        self.assertEqual(controller.interruption_target(interruption_a.id), (response_a, "client_cancelled"))
        self.assertEqual(controller.interruption_target(interruption_b.id), (response_b, "turn_detected"))

    def test_abandon_response_releases_only_its_exact_owner(self) -> None:
        controller = _controller()
        response_a = controller.start_response()[0]["response"]["id"]

        self.assertFalse(controller.abandon_response("resp_not_active", reason="connection_closed"))
        self.assertEqual(controller.active_response_id, response_a)
        self.assertTrue(controller.abandon_response(response_a, reason="connection_closed"))
        self.assertIsNone(controller.active_response_id)
        self.assertFalse(controller.abandon_response(response_a, reason="connection_closed"))

        response_b = controller.start_response()[0]["response"]["id"]
        self.assertNotEqual(response_b, response_a)

    def test_assistant_audio_transcript_is_finalized_into_response_done(self) -> None:
        controller = _controller()
        controller.append_assistant_text("Hi ")
        controller.append_assistant_text("there.")
        events = controller.finish_response(status="completed")
        done = events[-1]
        self.assertEqual(done["response"]["output"][0]["content"][0]["type"], "output_audio")
        self.assertEqual(done["response"]["output"][0]["content"][0]["transcript"], "Hi there.")
        self.assertIn("response.output_audio_transcript.done", [event["type"] for event in events])

    def test_text_session_finishes_with_output_text(self) -> None:
        controller = _controller(output_kind="text")
        controller.append_assistant_text("No speech.")
        events = controller.finish_response(status="completed")
        done = events[-1]
        self.assertEqual(done["response"]["output"][0]["content"], [{"type": "output_text", "text": "No speech."}])
        self.assertIn("response.output_text.done", [event["type"] for event in events])

    def test_tool_call_and_output_use_separate_response_lifecycles(self) -> None:
        controller = _controller(tool_owner="client")
        call_events = controller.start_function_call(call_id="call_1", name="lookup", arguments={"key": "x"})
        response_a = controller.active_response_id
        self.assertIn("response.function_call_arguments.done", [event["type"] for event in call_events])
        controller.finish_response(status="completed")
        output_events = controller.add_function_output(call_id="call_1", output='{"value": 1}', owner="client")
        self.assertEqual(
            [event["type"] for event in output_events], ["conversation.item.added", "conversation.item.done"]
        )
        self.assertEqual(
            [event["item"]["status"] for event in output_events],
            ["completed", "completed"],
        )
        response_b = controller.start_response()[0]["response"]["id"]
        self.assertNotEqual(response_a, response_b)

    def test_parallel_calls_share_response_a_and_accept_outputs_out_of_order(self) -> None:
        controller = _controller(tool_owner="client")
        call_events = [
            *controller.start_function_call(call_id="call_1", name="lookup", arguments={"key": "first"}),
            *controller.start_function_call(call_id="call_2", name="lookup", arguments={"key": "second"}),
        ]
        response_a = controller.active_response_id

        added = [event for event in call_events if event["type"] == "response.output_item.added"]
        arguments_done = [event for event in call_events if event["type"] == "response.function_call_arguments.done"]
        self.assertEqual([event["output_index"] for event in added], [0, 1])
        self.assertEqual([event["item"]["call_id"] for event in added], ["call_1", "call_2"])
        self.assertEqual([event["call_id"] for event in arguments_done], ["call_1", "call_2"])
        self.assertTrue(all(event["response_id"] == response_a for event in added + arguments_done))

        done_a = controller.finish_response(status="completed")[-1]
        self.assertEqual(done_a["response"]["id"], response_a)
        self.assertEqual(
            [item["call_id"] for item in done_a["response"]["output"]],
            ["call_1", "call_2"],
        )

        second_output = controller.add_function_output(
            call_id="call_2",
            output='{"value":"second"}',
            owner="client",
        )
        first_output = controller.add_function_output(
            call_id="call_1",
            output='{"value":"first"}',
            owner="client",
        )
        self.assertEqual(
            [
                event["item"]["call_id"]
                for event in (*second_output, *first_output)
                if event["type"] == "conversation.item.done"
            ],
            ["call_2", "call_1"],
        )
        response_b = controller.start_response()[0]["response"]["id"]
        self.assertNotEqual(response_a, response_b)


class NvidiaRealtimeRunOwnershipTests(unittest.IsolatedAsyncioTestCase):
    async def test_provider_start_is_typed_with_the_immutable_context_owner(self) -> None:
        service = NvidiaLLMService(
            api_key="test",
            base_url="http://127.0.0.1:9/v1",
            settings=NvidiaLLMSettings(model="test-model"),
        )
        token = service._realtime_run_owner.set(("", "run_exact", 9))
        try:
            with patch.object(PipecatNvidiaLLMService, "push_frame", new_callable=AsyncMock) as base_push:
                await service.push_frame(LLMFullResponseStartFrame())
        finally:
            service._realtime_run_owner.reset(token)

        emitted, direction = base_push.await_args.args
        self.assertIsInstance(emitted, RealtimeOwnedLLMFullResponseStartFrame)
        self.assertEqual(emitted.response_id, "")
        self.assertEqual(emitted.run_owner_id, "run_exact")
        self.assertEqual(emitted.activation_generation, 9)
        self.assertEqual(direction, FrameDirection.DOWNSTREAM)

    async def test_upstream_tool_followup_is_activated_with_one_owned_start(self) -> None:
        service = NvidiaLLMService(
            api_key="test",
            base_url="http://127.0.0.1:9/v1",
            settings=NvidiaLLMSettings(model="test-model"),
        )
        original_skip_tts = service._skip_tts
        abort = MagicMock()
        activation_callbacks: list[Any] = []

        async def activate(on_started) -> str:
            activation_callbacks.append(on_started)
            await on_started("resp_followup")
            return "resp_followup"

        async def prepare(context: LLMContext):
            self.assertEqual(context.get_messages(), [{"role": "tool", "content": "result"}])
            return (
                RealtimeResponseLLMContext(
                    copy.deepcopy(context.get_messages()),
                    run_owner_id="run_followup",
                    activation_generation=4,
                ),
                (LLMConfigureOutputFrame(skip_tts=True),),
                activate,
                abort,
            )

        service.bind_realtime_deferred_response_snapshot(prepare)
        source_context = LLMContext([{"role": "tool", "content": "result"}])
        source_frame = LLMContextFrame(context=source_context)
        source_frame.metadata["origin"] = "assistant-aggregator"
        service._process_context = AsyncMock()
        service.start_processing_metrics = AsyncMock()
        service.stop_processing_metrics = AsyncMock()

        with (
            patch.object(LLMService, "process_frame", new_callable=AsyncMock) as base_process,
            patch.object(PipecatNvidiaLLMService, "push_frame", new_callable=AsyncMock) as base_push,
        ):
            await service.process_frame(source_frame, FrameDirection.UPSTREAM)

        base_process.assert_awaited_once_with(service, source_frame, FrameDirection.UPSTREAM)
        self.assertIs(source_frame.context, source_context)
        self.assertEqual(source_frame.metadata, {"origin": "assistant-aggregator"})
        self.assertEqual(len(activation_callbacks), 1)
        self.assertEqual(base_push.await_count, 3)
        setup_frame = base_push.await_args_list[0].args[0]
        owned_start = base_push.await_args_list[1].args[0]
        response_end = base_push.await_args_list[2].args[0]
        self.assertIsInstance(setup_frame, LLMConfigureOutputFrame)
        self.assertTrue(setup_frame.skip_tts)
        self.assertIsInstance(owned_start, RealtimeOwnedLLMFullResponseStartFrame)
        self.assertEqual(owned_start.response_id, "resp_followup")
        self.assertIsInstance(response_end, LLMFullResponseEndFrame)
        run_context = service._process_context.await_args.args[0]
        self.assertIsInstance(run_context, RealtimeResponseLLMContext)
        self.assertEqual(run_context.response_id, "resp_followup")
        self.assertEqual(run_context.run_owner_id, "run_followup")
        service.start_processing_metrics.assert_awaited_once_with()
        service.stop_processing_metrics.assert_awaited_once_with()
        self.assertIs(service._skip_tts, original_skip_tts)
        abort.assert_not_called()


class NvidiaRealtimeContextTruncationTests(unittest.IsolatedAsyncioTestCase):
    async def test_unavailable_exact_counter_preserves_provider_request_without_probing(self) -> None:
        messages = [{"role": "user", "content": "hello"}]
        context = RealtimeResponseLLMContext(copy.deepcopy(messages), truncation=None)
        count_tokens = AsyncMock(side_effect=AssertionError("unsupported tokenizer must not be called"))
        params = {"messages": copy.deepcopy(messages)}

        actual = await _truncate_realtime_context(params, context, count_tokens=count_tokens)

        self.assertIs(actual, params)
        count_tokens.assert_not_awaited()

    async def test_auto_drops_complete_oldest_turn_from_only_the_response_snapshot(self) -> None:
        messages = [
            {"role": "system", "content": "prompt"},
            {"role": "user", "content": "old question"},
            {"role": "assistant", "content": "old answer"},
            {"role": "user", "content": "latest question"},
        ]
        context = RealtimeResponseLLMContext(
            copy.deepcopy(messages),
            truncation="auto",
            preserve_prompt_messages=1,
            max_output_tokens=5,
        )
        params = {"messages": copy.deepcopy(messages), "tools": [{"type": "function"}]}

        async def count_tokens(candidate: dict[str, Any]) -> tuple[int, int]:
            return len(candidate["messages"]) * 10, 40

        truncated = await _truncate_realtime_context(params, context, count_tokens=count_tokens)

        self.assertEqual(
            truncated["messages"],
            [messages[0], messages[-1]],
        )
        self.assertEqual(truncated["tools"], params["tools"])
        self.assertEqual(context.get_messages(), messages)
        self.assertEqual(params["messages"], messages)

    async def test_retention_ratio_targets_post_prompt_budget(self) -> None:
        messages = [
            {"role": "system", "content": "prompt"},
            {"role": "user", "content": "one"},
            {"role": "assistant", "content": "one answer"},
            {"role": "user", "content": "two"},
            {"role": "assistant", "content": "two answer"},
            {"role": "user", "content": "latest"},
        ]
        context = RealtimeResponseLLMContext(
            copy.deepcopy(messages),
            truncation={"type": "retention_ratio", "retention_ratio": 0.5},
            preserve_prompt_messages=1,
            max_output_tokens=10,
        )

        async def count_tokens(candidate: dict[str, Any]) -> tuple[int, int]:
            return len(candidate["messages"]) * 20, 100

        truncated = await _truncate_realtime_context(
            {"messages": copy.deepcopy(messages)},
            context,
            count_tokens=count_tokens,
        )

        self.assertEqual(truncated["messages"], [messages[0], messages[-1]])

    async def test_post_instruction_limit_excludes_instruction_and_tool_prefix(self) -> None:
        messages = [
            {"role": "system", "content": "prompt"},
            {"role": "user", "content": "old"},
            {"role": "assistant", "content": "old answer"},
            {"role": "user", "content": "latest"},
        ]
        context = RealtimeResponseLLMContext(
            copy.deepcopy(messages),
            truncation={
                "type": "retention_ratio",
                "retention_ratio": 1.0,
                "token_limits": {"post_instructions": 20},
            },
            preserve_prompt_messages=1,
            max_output_tokens=10,
        )

        async def count_tokens(candidate: dict[str, Any]) -> tuple[int, int]:
            # The rendered instruction/tool prefix costs 30 tokens and each
            # post-instruction message costs 10. The custom limit therefore
            # permits the prompt plus two conversation messages, not 20 total.
            return 30 + max(0, len(candidate["messages"]) - 1) * 10, 100

        truncated = await _truncate_realtime_context(
            {"messages": copy.deepcopy(messages), "tools": [{"type": "function"}]},
            context,
            count_tokens=count_tokens,
        )

        self.assertEqual(truncated["messages"], [messages[0], messages[-1]])

    async def test_post_instruction_limit_cannot_exceed_exact_model_input_limit(self) -> None:
        messages = [
            {"role": "system", "content": "prompt"},
            {"role": "user", "content": "latest"},
        ]
        context = RealtimeResponseLLMContext(
            copy.deepcopy(messages),
            truncation={
                "type": "retention_ratio",
                "retention_ratio": 0.8,
                "token_limits": {"post_instructions": 91},
            },
            preserve_prompt_messages=1,
            max_output_tokens=10,
        )

        with self.assertRaises(RealtimeProtocolError) as raised:
            await _truncate_realtime_context(
                {"messages": copy.deepcopy(messages)},
                context,
                count_tokens=AsyncMock(return_value=(20, 100)),
            )

        self.assertEqual(raised.exception.code, "invalid_value")
        self.assertEqual(raised.exception.param, "session.truncation.token_limits.post_instructions")

    async def test_truncation_rejects_inconsistent_candidate_tokenizer_metadata(self) -> None:
        messages = [
            {"role": "system", "content": "prompt"},
            {"role": "user", "content": "old"},
            {"role": "user", "content": "latest"},
        ]
        context = RealtimeResponseLLMContext(
            copy.deepcopy(messages),
            truncation="auto",
            preserve_prompt_messages=1,
            max_output_tokens=10,
        )
        counts = AsyncMock(side_effect=[(60, 50), (20, 49)])

        with self.assertRaisesRegex(RuntimeError, "inconsistent candidate metadata"):
            await _truncate_realtime_context(
                {"messages": copy.deepcopy(messages)},
                context,
                count_tokens=counts,
            )

    async def test_disabled_and_irreducible_contexts_fail_with_native_error(self) -> None:
        messages = [
            {"role": "system", "content": "prompt"},
            {"role": "user", "content": "old"},
            {"role": "user", "content": "latest"},
        ]

        async def count_tokens(candidate: dict[str, Any]) -> tuple[int, int]:
            return len(candidate["messages"]) * 20, 30

        disabled = RealtimeResponseLLMContext(
            copy.deepcopy(messages),
            truncation="disabled",
            preserve_prompt_messages=1,
            max_output_tokens=1,
        )
        with self.assertRaises(RealtimeProtocolError) as disabled_error:
            await _truncate_realtime_context(
                {"messages": copy.deepcopy(messages)},
                disabled,
                count_tokens=count_tokens,
            )
        self.assertEqual(disabled_error.exception.code, "context_length_exceeded")
        self.assertEqual(disabled_error.exception.param, "session.truncation")

        irreducible = RealtimeResponseLLMContext(
            [messages[0], messages[-1]],
            truncation="auto",
            preserve_prompt_messages=1,
            max_output_tokens=1,
        )
        with self.assertRaises(RealtimeProtocolError) as irreducible_error:
            await _truncate_realtime_context(
                {"messages": [messages[0], messages[-1]]},
                irreducible,
                count_tokens=count_tokens,
            )
        self.assertEqual(irreducible_error.exception.code, "context_length_exceeded")
        self.assertEqual(irreducible_error.exception.param, "conversation")


class RealtimeASROwnershipTests(unittest.IsolatedAsyncioTestCase):
    @staticmethod
    def _result(*, audio_processed: object, start_ms: int, end_ms: int) -> SimpleNamespace:
        return SimpleNamespace(
            audio_processed=audio_processed,
            alternatives=[
                SimpleNamespace(
                    words=[SimpleNamespace(start_time=start_ms, end_time=end_ms)],
                )
            ],
        )

    def test_post_stop_silence_coordinate_remains_in_the_stopped_turn(self) -> None:
        ownership = RealtimeASRTurnOwnership()
        ownership.start_turn(101, start_sample=0)
        ownership.submit_samples(24_000)
        ownership.stop_turn()
        ownership.submit_samples(8_000)

        processed = ownership.processed_sample(1.9, sample_rate=16_000)

        self.assertEqual(processed, 30_400)
        self.assertEqual(ownership.owner_for_range(8_000, 12_800), 101)

    async def test_aggregator_upstream_vad_id_owns_submitted_audio_transcript(self) -> None:
        recorder = _Recorder()
        controller = _controller()
        observer = _observer(recorder, controller)
        service = RealtimeNvidiaSTTService(server="localhost:50051", use_ssl=False)
        service._sample_rate = 24_000
        service._audio_channel_count = 1
        iterator = SimpleNamespace(closed=False, put=AsyncMock())
        service._audio_iterator = iterator
        aggregator = LLMContextAggregatorPair(LLMContext([])).user()
        downstream = FrameProcessor(name="after-user-aggregator")
        audio = InputAudioRawFrame(
            audio=b"\x01\x00" * 24_000,
            sample_rate=24_000,
            num_channels=1,
        )

        await observer.on_process_frame(FrameProcessed(aggregator, audio, FrameDirection.DOWNSTREAM, 0))
        with patch.object(service, "start_processing_metrics", AsyncMock()):
            frames = [frame async for frame in service.run_stt(audio.audio)]

        queued_vad = AsyncMock()
        pushed_vad = AsyncMock()
        with (
            patch.object(aggregator, "queue_frame", queued_vad),
            patch.object(aggregator, "push_frame", pushed_vad),
        ):
            await aggregator._queued_broadcast_frame(
                VADUserStartedSpeakingFrame,
                start_secs=0.2,
            )
        downstream_vad = queued_vad.await_args.args[0]
        upstream_vad = pushed_vad.await_args.args[0]
        self.assertEqual(pushed_vad.await_args.args[1], FrameDirection.UPSTREAM)
        self.assertNotEqual(upstream_vad.id, downstream_vad.id)
        await _push(
            observer,
            upstream_vad,
            source=aggregator,
            destination=service,
            direction=FrameDirection.UPSTREAM,
        )
        base_process = AsyncMock()
        with patch.object(PipecatNvidiaSTTService, "process_frame", base_process):
            await service.process_frame(upstream_vad, FrameDirection.UPSTREAM)
        await _push(
            observer,
            downstream_vad,
            source=aggregator,
            destination=downstream,
        )

        transcript = TranscriptionFrame(
            text="please calculate my",
            user_id="",
            timestamp="",
            result=self._result(audio_processed=1.0, start_ms=850, end_ms=950),
            finalized=True,
        )
        base_push = AsyncMock()
        with patch.object(PipecatNvidiaSTTService, "push_frame", base_push):
            await service.push_frame(transcript)
        await _push(
            observer,
            transcript,
            source=service,
            destination=aggregator,
        )
        await _push(
            observer,
            UserStoppedSpeakingFrame(),
            source=aggregator,
            destination=downstream,
        )

        self.assertEqual(frames, [None])
        iterator.put.assert_awaited_once_with(audio.audio)
        self.assertEqual(service._realtime_ownership.submitted_samples, 24_000)
        self.assertEqual(
            transcript.metadata[USER_TRANSCRIPT_TURN_FRAME_ID_METADATA],
            upstream_vad.id,
        )
        base_process.assert_awaited_once_with(upstream_vad, FrameDirection.UPSTREAM)
        self.assertNotIn("error", recorder.types)
        completed = next(
            event
            for event in recorder.events
            if event["type"] == "conversation.item.input_audio_transcription.completed"
        )
        self.assertEqual(completed["transcript"], "please calculate my")
        observer.shutdown()

    def test_delayed_result_uses_word_interval_after_the_next_raw_turn_starts(self) -> None:
        ownership = RealtimeASRTurnOwnership()
        ownership.start_turn(101, start_sample=0)
        ownership.submit_samples(16_000)
        ownership.stop_turn()
        ownership.submit_samples(16_000)
        ownership.start_turn(202, start_sample=24_000)
        ownership.submit_samples(8_000)

        processed = ownership.processed_sample(2.4, sample_rate=16_000)

        self.assertEqual(processed, 38_400)
        self.assertEqual(ownership.owner_for_range(8_000, 12_000), 101)
        self.assertEqual(ownership.owner_for_range(26_000, 30_000), 202)
        self.assertIsNone(ownership.owner_for_range(18_000, 20_000))

    async def test_service_assigns_a_range_that_intersects_exactly_one_of_two_turns(self) -> None:
        service = RealtimeNvidiaSTTService(server="localhost:50051", use_ssl=False)
        service._sample_rate = 16_000
        ownership = service._realtime_ownership
        ownership.submit_samples(7_800)
        ownership.start_turn(72, start_sample=7_800)
        ownership.submit_samples(46_100)
        ownership.stop_turn()
        ownership.submit_samples(7_800)
        ownership.start_turn(396, start_sample=61_700)
        ownership.submit_samples(49_400)
        ownership.stop_turn()
        ownership.submit_samples(2_200)
        transcript = TranscriptionFrame(
            text="second turn",
            user_id="",
            timestamp="",
            result=self._result(audio_processed=7.08125, start_ms=3840, end_ms=6640),
            finalized=True,
        )

        base_push = AsyncMock()
        with patch.object(PipecatNvidiaSTTService, "push_frame", base_push):
            await service.push_frame(transcript)

        self.assertEqual(
            transcript.metadata[USER_TRANSCRIPT_TURN_FRAME_ID_METADATA],
            396,
        )
        self.assertIs(base_push.await_args.args[0], transcript)
        self.assertIsNone(ownership.owner_for_range(53_000, 62_000))
        self.assertIsNone(ownership.owner_for_range(55_000, 60_000))

    def test_invalid_and_untrustworthy_coordinates_fail_closed(self) -> None:
        invalid_values = (None, float("nan"), float("inf"), -0.1)
        for value in invalid_values:
            with self.subTest(value=value):
                ownership = RealtimeASRTurnOwnership()
                ownership.start_turn(101, start_sample=0)
                ownership.submit_samples(16_000)
                with self.assertRaises(RealtimeASROwnershipError) as caught:
                    ownership.processed_sample(value, sample_rate=16_000)
                self.assertEqual(caught.exception.code, "asr_transcription_coordinate_invalid")

        ownership = RealtimeASRTurnOwnership()
        ownership.start_turn(101, start_sample=0)
        ownership.submit_samples(16_000)
        with self.assertRaises(RealtimeASROwnershipError) as out_of_range:
            ownership.processed_sample(1.1, sample_rate=16_000)
        self.assertEqual(out_of_range.exception.code, "asr_transcription_coordinate_out_of_range")

        ownership = RealtimeASRTurnOwnership()
        ownership.start_turn(101, start_sample=0)
        ownership.submit_samples(16_000)
        ownership.processed_sample(0.75, sample_rate=16_000)
        with self.assertRaises(RealtimeASROwnershipError) as regressed:
            ownership.processed_sample(0.5, sample_rate=16_000)
        self.assertEqual(regressed.exception.code, "asr_transcription_coordinate_regressed")

    async def test_service_stamps_delayed_final_by_word_interval_before_stock_push(self) -> None:
        service = RealtimeNvidiaSTTService(server="localhost:50051", use_ssl=False)
        service._sample_rate = 16_000
        service._realtime_ownership.start_turn(101, start_sample=0)
        service._realtime_ownership.submit_samples(16_000)
        service._realtime_ownership.stop_turn()
        service._realtime_ownership.submit_samples(16_000)
        service._realtime_ownership.start_turn(202, start_sample=24_000)
        service._realtime_ownership.submit_samples(8_000)
        delayed = TranscriptionFrame(
            text="turn one",
            user_id="",
            timestamp="",
            result=self._result(audio_processed=2.4, start_ms=500, end_ms=750),
            finalized=True,
        )
        current = InterimTranscriptionFrame(
            text="turn two",
            user_id="",
            timestamp="",
            result=self._result(audio_processed=2.5, start_ms=1600, end_ms=1800),
        )

        base_push = AsyncMock()
        with patch.object(PipecatNvidiaSTTService, "push_frame", base_push):
            await service.push_frame(delayed)
            self.assertEqual(delayed.metadata[USER_TRANSCRIPT_TURN_FRAME_ID_METADATA], 101)
            await service.push_frame(current)
            self.assertEqual(current.metadata[USER_TRANSCRIPT_TURN_FRAME_ID_METADATA], 202)

        self.assertEqual(base_push.await_count, 2)
        self.assertIs(base_push.await_args_list[0].args[0], delayed)
        self.assertIs(base_push.await_args_list[1].args[0], current)

    async def test_service_back_projects_vad_onset_before_flushing_first_word(self) -> None:
        service = RealtimeNvidiaSTTService(server="localhost:50051", use_ssl=False)
        service._sample_rate = 16_000
        service._realtime_ownership.submit_samples(16_000)
        first_word = InterimTranscriptionFrame(
            text="hello",
            user_id="",
            timestamp="",
            result=self._result(audio_processed=0.9, start_ms=250, end_ms=500),
        )
        vad_start = VADUserStartedSpeakingFrame(start_secs=0.8)

        base_push = AsyncMock()
        base_process = AsyncMock()
        with (
            patch.object(PipecatNvidiaSTTService, "push_frame", base_push),
            patch.object(PipecatNvidiaSTTService, "process_frame", base_process),
        ):
            await service.push_frame(first_word)
            self.assertEqual(base_push.await_count, 0)
            await service.process_frame(vad_start, FrameDirection.DOWNSTREAM)

        self.assertEqual(
            first_word.metadata[USER_TRANSCRIPT_TURN_FRAME_ID_METADATA],
            vad_start.id,
        )
        base_process.assert_awaited_once_with(vad_start, FrameDirection.DOWNSTREAM)
        self.assertIs(base_push.await_args.args[0], first_word)

    async def test_final_in_an_inter_turn_gap_is_not_assigned_to_the_active_turn(self) -> None:
        service = RealtimeNvidiaSTTService(server="localhost:50051", use_ssl=False)
        service._sample_rate = 16_000
        service._realtime_ownership.start_turn(101, start_sample=0)
        service._realtime_ownership.submit_samples(16_000)
        service._realtime_ownership.stop_turn()
        service._realtime_ownership.submit_samples(16_000)
        service._realtime_ownership.start_turn(202, start_sample=24_000)
        service._realtime_ownership.submit_samples(8_000)
        gap_final = TranscriptionFrame(
            text="ambiguous gap",
            user_id="",
            timestamp="",
            result=self._result(audio_processed=2.4, start_ms=1100, end_ms=1200),
            finalized=True,
        )

        base_push = AsyncMock()
        with patch.object(PipecatNvidiaSTTService, "push_frame", base_push):
            await service.push_frame(gap_final)

        self.assertEqual(base_push.await_count, 2)
        error = base_push.await_args_list[0].args[0]
        self.assertIsInstance(error, RealtimeInputTranscriptionErrorFrame)
        self.assertEqual(error.code, "asr_transcription_owner_missing")
        self.assertIs(base_push.await_args_list[1].args[0], gap_final)
        self.assertNotIn(USER_TRANSCRIPT_TURN_FRAME_ID_METADATA, gap_final.metadata)

    async def test_manual_transcripts_keep_the_existing_semantic_commit_owner(self) -> None:
        service = RealtimeNvidiaSTTService(server="localhost:50051", use_ssl=False)
        service._sample_rate = 16_000
        manual_start = RealtimeManualUserStartedSpeakingFrame(
            audio_sample_cursor=0,
            sample_rate=16_000,
        )
        transcript = TranscriptionFrame(
            text="manual turn",
            user_id="",
            timestamp="",
            result=SimpleNamespace(audio_processed=1.0),
            finalized=True,
        )

        base_push = AsyncMock()
        base_process = AsyncMock()
        with (
            patch.object(PipecatNvidiaSTTService, "push_frame", base_push),
            patch.object(PipecatNvidiaSTTService, "process_frame", base_process),
        ):
            await service.process_frame(manual_start, FrameDirection.DOWNSTREAM)
            await service.push_frame(transcript)

        base_process.assert_awaited_once_with(manual_start, FrameDirection.DOWNSTREAM)
        self.assertIs(base_push.await_args.args[0], transcript)
        self.assertNotIn(USER_TRANSCRIPT_TURN_FRAME_ID_METADATA, transcript.metadata)

    async def test_missing_word_offsets_stay_pipeline_private_and_fail_public_final(self) -> None:
        service = RealtimeNvidiaSTTService(server="localhost:50051", use_ssl=False)
        service._sample_rate = 16_000
        service._realtime_ownership.start_turn(101, start_sample=0)
        service._realtime_ownership.submit_samples(16_000)
        missing_words = SimpleNamespace(
            audio_processed=0.75,
            alternatives=[SimpleNamespace(words=[])],
        )
        interim = InterimTranscriptionFrame(
            text="ambiguous",
            user_id="",
            timestamp="",
            result=missing_words,
        )
        final = TranscriptionFrame(
            text="ambiguous",
            user_id="",
            timestamp="",
            result=missing_words,
            finalized=True,
        )

        base_push = AsyncMock()
        with patch.object(PipecatNvidiaSTTService, "push_frame", base_push):
            await service.push_frame(interim)
            self.assertEqual(base_push.await_count, 1)
            self.assertNotIn(USER_TRANSCRIPT_TURN_FRAME_ID_METADATA, interim.metadata)
            self.assertIs(base_push.await_args.args[0], interim)
            await service.push_frame(final)

        self.assertEqual(base_push.await_count, 3)
        error = base_push.await_args_list[1].args[0]
        self.assertIsInstance(error, RealtimeInputTranscriptionErrorFrame)
        self.assertEqual(error.code, "asr_transcription_word_offsets_missing")
        self.assertIs(base_push.await_args_list[2].args[0], final)

    async def test_service_converts_coordinate_failure_to_a_typed_frame(self) -> None:
        service = RealtimeNvidiaSTTService(server="localhost:50051", use_ssl=False)
        service._sample_rate = 16_000
        service._realtime_ownership.start_turn(101, start_sample=0)
        service._realtime_ownership.submit_samples(16_000)
        transcript = TranscriptionFrame(
            text="must not escape",
            user_id="",
            timestamp="",
            result=self._result(audio_processed=float("nan"), start_ms=100, end_ms=200),
            finalized=True,
        )

        base_push = AsyncMock()
        with patch.object(PipecatNvidiaSTTService, "push_frame", base_push):
            await service.push_frame(transcript)

        self.assertEqual(base_push.await_count, 2)
        error = base_push.await_args_list[0].args[0]
        self.assertIsInstance(error, RealtimeInputTranscriptionErrorFrame)
        self.assertEqual(error.code, "asr_transcription_coordinate_invalid")
        self.assertIs(base_push.await_args_list[1].args[0], transcript)
        self.assertNotIn(USER_TRANSCRIPT_TURN_FRAME_ID_METADATA, transcript.metadata)

    def test_service_always_requests_word_offsets_for_realtime_ownership(self) -> None:
        service = RealtimeNvidiaSTTService(server="localhost:50051", use_ssl=False)
        service._sample_rate = 16_000

        config = service._create_recognition_config()

        self.assertTrue(config.config.enable_word_time_offsets)


class TranscriptControllerTests(unittest.TestCase):
    def test_final_segments_wait_for_semantic_stop_and_preserve_order(self) -> None:
        controller = _controller()
        item_id, _audio_start_ms = controller.begin_user_turn(audio_sample_cursor=0, sample_rate=16_000)
        for segment in ("The local speech", "pipeline is ready", "."):
            _item_id, events = controller.set_user_transcript(segment, item_id=item_id)
            self.assertEqual(events, [])

        item_id, audio_end_ms, events = controller.stop_user_turn(
            audio_sample_cursor=16_000,
            sample_rate=16_000,
        )
        completed = next(event for event in events if event["type"].endswith("transcription.completed"))
        self.assertEqual(completed["item_id"], item_id)
        self.assertEqual(completed["transcript"], "The local speech pipeline is ready .")
        self.assertEqual(audio_end_ms, 1000)

    def test_fused_transcript_after_stop_targets_the_exact_item(self) -> None:
        controller = _controller(fused=True)
        item_id, _audio_end_ms, stop_events = controller.stop_user_turn(
            audio_sample_cursor=16_000,
            sample_rate=16_000,
        )
        self.assertNotIn(
            "conversation.item.input_audio_transcription.completed",
            [event["type"] for event in stop_events],
        )
        correlated_item, events = controller.set_user_transcript("hello world", item_id=item_id)
        self.assertEqual(correlated_item, item_id)
        self.assertEqual(events[0]["item_id"], item_id)
        self.assertEqual(events[0]["transcript"], "hello world")

    def test_late_terminal_cannot_transfer_to_a_newer_turn(self) -> None:
        controller = _controller(fused=True)
        first_item, _audio_end_ms, _events = controller.stop_user_turn(
            audio_sample_cursor=16_000,
            sample_rate=16_000,
        )
        _item_id, first_events = controller.set_user_transcript("turn one", item_id=first_item)
        self.assertEqual(first_events[0]["item_id"], first_item)
        second_item, _audio_start_ms = controller.begin_user_turn(
            new_turn=True,
            audio_sample_cursor=16_000,
            sample_rate=16_000,
        )
        terminal_item, terminal_events = controller.end_user_transcript_producer(
            item_id=first_item,
            code="omni_transcription_cancelled",
            message="late terminal",
        )
        self.assertEqual(terminal_item, first_item)
        self.assertEqual(terminal_events, [])
        self.assertNotEqual(first_item, second_item)
        _item_id, second_events = controller.set_user_transcript("turn two", item_id=second_item)
        self.assertEqual(second_events, [])

    def test_blank_transcripts_do_not_allocate_a_phantom_item(self) -> None:
        controller = _controller(fused=True)
        item_id, events = controller.set_user_transcript(" \n")
        self.assertEqual((item_id, events), ("", []))
        self.assertIsNone(controller._user_item_id)
        self.assertEqual(controller.user_transcript_delta("\t"), [])
        self.assertIsNone(controller._user_item_id)

    def test_unowned_transcripts_after_a_completed_turn_do_not_create_the_next_turn(self) -> None:
        controller = _controller()
        first_item, _audio_start_ms = controller.begin_user_turn(audio_sample_cursor=0, sample_rate=16_000)
        _item_id, _events = controller.set_user_transcript("turn one", item_id=first_item)
        stopped_item, _audio_end_ms, events = controller.stop_user_turn(
            audio_sample_cursor=16_000,
            sample_rate=16_000,
        )
        self.assertEqual(stopped_item, first_item)
        self.assertTrue(any(event["type"].endswith("transcription.completed") for event in events))

        self.assertEqual(controller.user_transcript_delta("late interim"), [])
        self.assertEqual(controller.set_user_transcript("late final"), ("", []))
        self.assertIsNone(controller._user_item_id)
        self.assertEqual(controller._pending_user_interim, "")
        self.assertEqual(controller._pending_user_transcript.text, "")

        second_item, _audio_start_ms = controller.begin_user_turn(
            audio_sample_cursor=16_000,
            sample_rate=16_000,
        )
        self.assertNotEqual(second_item, first_item)
        _item_id, _audio_end_ms, second_events = controller.stop_user_turn(
            audio_sample_cursor=32_000,
            sample_rate=16_000,
        )
        second_done = next(event for event in second_events if event["type"] == "conversation.item.done")
        self.assertIsNone(second_done["item"]["content"][0]["transcript"])

    def test_unowned_transcripts_cannot_attach_to_an_active_or_stopped_new_turn(self) -> None:
        controller = _controller()
        first_item, _audio_start_ms = controller.begin_user_turn(
            audio_sample_cursor=0,
            sample_rate=16_000,
        )
        controller.set_user_transcript("turn one", item_id=first_item)
        stopped_first, _audio_end_ms, first_events = controller.stop_user_turn(
            audio_sample_cursor=16_000,
            sample_rate=16_000,
        )
        self.assertEqual(stopped_first, first_item)
        self.assertTrue(any(event["type"].endswith("transcription.completed") for event in first_events))

        second_item, _audio_start_ms = controller.begin_user_turn(
            audio_sample_cursor=24_000,
            sample_rate=16_000,
        )
        self.assertEqual(controller.user_transcript_delta("late active interim"), [])
        self.assertEqual(controller.set_user_transcript("late active final"), ("", []))
        self.assertEqual(controller._user_item_id, second_item)
        self.assertEqual(controller._pending_user_interim, "")
        self.assertEqual(controller._pending_user_transcript.text, "")

        stopped_second, _audio_end_ms, second_events = controller.stop_user_turn(
            audio_sample_cursor=32_000,
            sample_rate=16_000,
        )
        self.assertEqual(stopped_second, second_item)
        second_done = next(event for event in second_events if event["type"] == "conversation.item.done")
        self.assertIsNone(second_done["item"]["content"][0]["transcript"])
        self.assertEqual(controller.user_transcript_delta("late stopped interim"), [])
        self.assertEqual(controller.set_user_transcript("late stopped final"), ("", []))
        self.assertEqual(controller._stopped_user_transcripts[second_item].text, "")

        exact_item, exact_events = controller.set_user_transcript("turn two", item_id=second_item)
        self.assertEqual(exact_item, second_item)
        completed = next(event for event in exact_events if event["type"].endswith("transcription.completed"))
        self.assertEqual(completed["item_id"], second_item)
        self.assertEqual(completed["transcript"], "turn two")

    def test_exact_producer_terminal_fails_a_stopped_empty_turn(self) -> None:
        controller = _controller(fused=True)
        item_id, _audio_end_ms, _events = controller.stop_user_turn(
            audio_sample_cursor=16_000,
            sample_rate=16_000,
        )
        terminal_item, events = controller.end_user_transcript_producer(
            item_id=item_id,
            code="omni_transcription_missing",
            message="missing transcript",
        )
        self.assertEqual(terminal_item, item_id)
        self.assertEqual(events[0]["type"], "conversation.item.input_audio_transcription.failed")
        self.assertEqual(events[0]["item_id"], item_id)


class ObserverTranscriptTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.source = FrameProcessor(name="test-source")
        self.destination = FrameProcessor(name="test-destination")

    async def test_cascaded_interim_and_final_transcripts_complete_at_stop(self) -> None:
        recorder = _Recorder()
        observer = _observer(recorder, _controller())
        vad_start = VADUserStartedSpeakingFrame()
        await _push(observer, vad_start, source=self.source, destination=self.destination)
        await _push(observer, UserStartedSpeakingFrame(), source=self.source, destination=self.destination)
        await _push(
            observer,
            _owned_by(
                InterimTranscriptionFrame(text="hel", user_id="", timestamp=""),
                vad_start,
            ),
            source=self.source,
            destination=self.destination,
        )
        await _push(
            observer,
            _owned_by(
                TranscriptionFrame(text="hello", user_id="", timestamp=""),
                vad_start,
            ),
            source=self.source,
            destination=self.destination,
        )
        self.assertNotIn("conversation.item.created", recorder.types)
        await _push(observer, UserStoppedSpeakingFrame(), source=self.source, destination=self.destination)

        completed = next(
            event
            for event in recorder.events
            if event["type"] == "conversation.item.input_audio_transcription.completed"
        )
        started = next(event for event in recorder.events if event["type"] == "input_audio_buffer.speech_started")
        stopped = next(event for event in recorder.events if event["type"] == "input_audio_buffer.speech_stopped")
        committed = next(event for event in recorder.events if event["type"] == "input_audio_buffer.committed")
        added = next(event for event in recorder.events if event["type"] == "conversation.item.added")
        done = next(event for event in recorder.events if event["type"] == "conversation.item.done")
        self.assertEqual(
            {
                started["item_id"],
                stopped["item_id"],
                committed["item_id"],
                added["item"]["id"],
                done["item"]["id"],
                completed["item_id"],
            },
            {completed["item_id"]},
        )
        self.assertEqual(completed["transcript"], "hello")
        self.assertLess(
            recorder.types.index("input_audio_buffer.committed"), recorder.types.index("conversation.item.added")
        )
        self.assertLess(
            recorder.types.index("conversation.item.added"),
            recorder.types.index("conversation.item.input_audio_transcription.completed"),
        )
        observer.shutdown()

    async def test_cascaded_interim_at_stop_and_final_after_stop_keep_one_item_owner(self) -> None:
        recorder = _Recorder()
        observer = _observer(recorder, _controller())
        vad_start = VADUserStartedSpeakingFrame()
        await _push(observer, vad_start, source=self.source, destination=self.destination)
        await _push(observer, UserStartedSpeakingFrame(), source=self.source, destination=self.destination)
        await _push(
            observer,
            _owned_by(
                InterimTranscriptionFrame(text="after", user_id="", timestamp=""),
                vad_start,
            ),
            source=self.source,
            destination=self.destination,
        )
        await _push(
            observer,
            VADUserStoppedSpeakingFrame(),
            source=self.source,
            destination=self.destination,
        )
        await _push(observer, UserStoppedSpeakingFrame(), source=self.source, destination=self.destination)
        await _push(
            observer,
            _owned_by(
                TranscriptionFrame(text="after stop", user_id="", timestamp="", finalized=True),
                vad_start,
            ),
            source=self.source,
            destination=self.destination,
        )

        stopped = next(event for event in recorder.events if event["type"] == "input_audio_buffer.speech_stopped")
        committed = next(event for event in recorder.events if event["type"] == "input_audio_buffer.committed")
        added = next(event for event in recorder.events if event["type"] == "conversation.item.added")
        done = next(event for event in recorder.events if event["type"] == "conversation.item.done")
        delta = next(
            event for event in recorder.events if event["type"] == "conversation.item.input_audio_transcription.delta"
        )
        completed = next(
            event
            for event in recorder.events
            if event["type"] == "conversation.item.input_audio_transcription.completed"
        )
        self.assertEqual(
            {
                stopped["item_id"],
                committed["item_id"],
                added["item"]["id"],
                done["item"]["id"],
                delta["item_id"],
                completed["item_id"],
            },
            {completed["item_id"]},
        )
        self.assertEqual(delta["delta"], "after")
        self.assertEqual(completed["transcript"], "after stop")
        self.assertLess(
            recorder.types.index("input_audio_buffer.speech_stopped"),
            recorder.types.index("input_audio_buffer.committed"),
        )
        self.assertLess(
            recorder.types.index("input_audio_buffer.committed"), recorder.types.index("conversation.item.added")
        )
        self.assertLess(recorder.types.index("conversation.item.added"), recorder.types.index("conversation.item.done"))
        self.assertLess(
            recorder.types.index("conversation.item.done"),
            recorder.types.index("conversation.item.input_audio_transcription.completed"),
        )
        observer.shutdown()

    async def test_unowned_transcripts_after_completion_are_ignored_without_carryover(self) -> None:
        recorder = _Recorder()
        controller = _controller()
        observer = _observer(recorder, controller)
        first_vad_start = VADUserStartedSpeakingFrame()
        await _push(observer, first_vad_start, source=self.source, destination=self.destination)
        await _push(observer, UserStartedSpeakingFrame(), source=self.source, destination=self.destination)
        await _push(
            observer,
            _owned_by(
                TranscriptionFrame(text="turn one", user_id="", timestamp="", finalized=True),
                first_vad_start,
            ),
            source=self.source,
            destination=self.destination,
        )
        await _push(observer, UserStoppedSpeakingFrame(), source=self.source, destination=self.destination)
        first_completed = next(event for event in recorder.events if event["type"].endswith("transcription.completed"))
        event_count = len(recorder.events)

        await _push(
            observer,
            InterimTranscriptionFrame(text="late interim", user_id="", timestamp=""),
            source=self.source,
            destination=self.destination,
        )
        await _push(
            observer,
            TranscriptionFrame(text="late final", user_id="", timestamp="", finalized=True),
            source=self.source,
            destination=self.destination,
        )

        self.assertEqual(len(recorder.events), event_count)
        self.assertIsNone(controller._user_item_id)
        self.assertEqual(controller._pending_user_interim, "")
        self.assertEqual(controller._pending_user_transcript.text, "")
        self.assertEqual(list(controller._stopped_user_items), [])

        second_vad_start = VADUserStartedSpeakingFrame()
        await _push(observer, second_vad_start, source=self.source, destination=self.destination)
        await _push(observer, UserStartedSpeakingFrame(), source=self.source, destination=self.destination)
        await _push(observer, UserStoppedSpeakingFrame(), source=self.source, destination=self.destination)
        second_done = [event for event in recorder.events if event["type"] == "conversation.item.done"][-1]
        self.assertNotEqual(second_done["item"]["id"], first_completed["item_id"])
        self.assertIsNone(second_done["item"]["content"][0]["transcript"])
        self.assertNotIn("late", json.dumps(recorder.events))
        observer.shutdown()

    async def test_late_first_turn_transcripts_do_not_attach_to_active_or_stopped_second_turn(self) -> None:
        for stop_second_before_late_frame in (False, True):
            with self.subTest(stop_second_before_late_frame=stop_second_before_late_frame):
                recorder = _Recorder()
                controller = _controller()
                observer = _observer(recorder, controller)

                first_vad_start = VADUserStartedSpeakingFrame()
                await _push(observer, first_vad_start, source=self.source, destination=self.destination)
                await _push(observer, UserStartedSpeakingFrame(), source=self.source, destination=self.destination)
                await _push(
                    observer,
                    _owned_by(
                        TranscriptionFrame(text="turn one", user_id="", timestamp="", finalized=True),
                        first_vad_start,
                    ),
                    source=self.source,
                    destination=self.destination,
                )
                await _push(
                    observer,
                    VADUserStoppedSpeakingFrame(),
                    source=self.source,
                    destination=self.destination,
                )
                await _push(observer, UserStoppedSpeakingFrame(), source=self.source, destination=self.destination)
                first_completed = next(
                    event for event in recorder.events if event["type"].endswith("transcription.completed")
                )

                second_vad_start = VADUserStartedSpeakingFrame()
                await _push(observer, second_vad_start, source=self.source, destination=self.destination)
                await _push(observer, UserStartedSpeakingFrame(), source=self.source, destination=self.destination)
                if stop_second_before_late_frame:
                    await _push(
                        observer,
                        VADUserStoppedSpeakingFrame(),
                        source=self.source,
                        destination=self.destination,
                    )
                    await _push(
                        observer,
                        UserStoppedSpeakingFrame(),
                        source=self.source,
                        destination=self.destination,
                    )

                late_unowned = TranscriptionFrame(
                    text="late unowned turn one",
                    user_id="",
                    timestamp="",
                    finalized=True,
                )
                late_retired = _owned_by(
                    InterimTranscriptionFrame(text="late retired turn one", user_id="", timestamp=""),
                    first_vad_start,
                )
                await _push(observer, late_unowned, source=self.source, destination=self.destination)
                await _push(observer, late_retired, source=self.source, destination=self.destination)
                self.assertEqual(controller._pending_user_interim, "")
                self.assertNotIn("late", json.dumps(recorder.events))

                await _push(
                    observer,
                    _owned_by(
                        TranscriptionFrame(text="turn two", user_id="", timestamp="", finalized=True),
                        second_vad_start,
                    ),
                    source=self.source,
                    destination=self.destination,
                )
                if not stop_second_before_late_frame:
                    await _push(
                        observer,
                        VADUserStoppedSpeakingFrame(),
                        source=self.source,
                        destination=self.destination,
                    )
                    await _push(
                        observer,
                        UserStoppedSpeakingFrame(),
                        source=self.source,
                        destination=self.destination,
                    )

                completed = [event for event in recorder.events if event["type"].endswith("transcription.completed")]
                self.assertEqual([event["transcript"] for event in completed], ["turn one", "turn two"])
                self.assertNotEqual(first_completed["item_id"], completed[-1]["item_id"])
                second_events = [
                    event
                    for event in recorder.events
                    if event.get("item_id") == completed[-1]["item_id"]
                    or event.get("item", {}).get("id") == completed[-1]["item_id"]
                ]
                self.assertTrue(second_events)
                observer.shutdown()

    async def test_empty_manual_transcript_fails_item_and_queued_response(self) -> None:
        recorder = _Recorder()
        controller = _controller(manual=True)
        observer = _observer(recorder, controller)
        await _push(observer, UserStartedSpeakingFrame(), source=self.source, destination=self.destination)
        stopped_frame = UserStoppedSpeakingFrame()
        await _push(observer, stopped_frame, source=self.source, destination=self.destination)
        committed = next(event for event in recorder.events if event["type"] == "input_audio_buffer.committed")
        controller.start_response()

        await _push(
            observer,
            RTVIServerMessageFrame(
                data={
                    "type": "user-turn-finalized",
                    "turn_frame_id": stopped_frame.id,
                    "transcript": "",
                }
            ),
            source=self.source,
            destination=self.destination,
        )

        failed = next(
            event for event in recorder.events if event["type"] == "conversation.item.input_audio_transcription.failed"
        )
        self.assertEqual(failed["item_id"], committed["item_id"])
        self.assertEqual(failed["error"]["code"], "empty_transcript")
        error = next(
            event
            for event in recorder.events
            if event["type"] == "error" and event["error"]["code"] == "input_audio_transcription_empty"
        )
        self.assertEqual(error["error"]["param"], "session.audio.input.transcription")
        done = next(event for event in recorder.events if event["type"] == "response.done")
        self.assertEqual(done["response"]["status"], "failed")
        self.assertEqual(
            done["response"]["status_details"]["error"]["code"],
            "input_audio_transcription_empty",
        )
        self.assertFalse(controller.response_in_progress)
        observer.shutdown()

    async def test_multiple_provider_final_segments_share_one_semantic_turn(self) -> None:
        recorder = _Recorder()
        observer = _observer(recorder, _controller())
        vad_start = VADUserStartedSpeakingFrame()
        await _push(observer, vad_start, source=self.source, destination=self.destination)
        await _push(observer, UserStartedSpeakingFrame(), source=self.source, destination=self.destination)
        for text in ("hello", "world"):
            await _push(
                observer,
                _owned_by(
                    TranscriptionFrame(text=text, user_id="", timestamp="", finalized=True),
                    vad_start,
                ),
                source=self.source,
                destination=self.destination,
            )
        self.assertNotIn("conversation.item.input_audio_transcription.completed", recorder.types)
        await _push(observer, UserStoppedSpeakingFrame(), source=self.source, destination=self.destination)
        completed = [event for event in recorder.events if event["type"].endswith("transcription.completed")]
        self.assertEqual(len(completed), 1)
        self.assertEqual(completed[0]["transcript"], "hello world")
        observer.shutdown()

    async def test_vad_events_use_the_processor_audio_clock(self) -> None:
        recorder = _Recorder()
        observer = _observer(recorder, _controller(transcription=False))
        first_audio = InputAudioRawFrame(audio=b"\x00\x00" * 16_000, sample_rate=16_000, num_channels=1)
        second_audio = InputAudioRawFrame(audio=b"\x00\x00" * 8_000, sample_rate=16_000, num_channels=1)
        await observer.on_process_frame(FrameProcessed(self.source, first_audio, FrameDirection.DOWNSTREAM, 0))
        await _push(observer, UserStartedSpeakingFrame(), source=self.source, destination=self.destination)
        await observer.on_process_frame(FrameProcessed(self.source, second_audio, FrameDirection.DOWNSTREAM, 1))
        await _push(observer, UserStoppedSpeakingFrame(), source=self.source, destination=self.destination)

        started = next(event for event in recorder.events if event["type"] == "input_audio_buffer.speech_started")
        stopped = next(event for event in recorder.events if event["type"] == "input_audio_buffer.speech_stopped")
        committed = next(event for event in recorder.events if event["type"] == "input_audio_buffer.committed")
        self.assertEqual(started["item_id"], stopped["item_id"])
        self.assertEqual(stopped["item_id"], committed["item_id"])
        self.assertEqual(started["audio_start_ms"], 1000)
        self.assertEqual(stopped["audio_end_ms"], 1500)
        observer.shutdown()

    async def test_manual_audio_uses_typed_cursor_across_processor_sources(self) -> None:
        recorder = _Recorder()
        controller = _controller(manual=True)
        observer = _observer(recorder, controller)
        boundary_source = FrameProcessor(name="manual-input-transport")
        await _push(
            observer,
            RealtimeManualUserStartedSpeakingFrame(audio_sample_cursor=0, sample_rate=16_000),
            source=boundary_source,
            destination=self.destination,
        )
        client_audio = InputAudioRawFrame(
            audio=b"\x01\x00" * 16_000,
            sample_rate=16_000,
            num_channels=1,
        )
        endpoint_silence = RealtimeASREndpointSilenceFrame(
            audio=b"\x00\x00" * 16_000,
            sample_rate=16_000,
            num_channels=1,
        )
        await observer.on_process_frame(FrameProcessed(self.source, client_audio, FrameDirection.DOWNSTREAM, 0))
        await observer.on_process_frame(FrameProcessed(self.source, endpoint_silence, FrameDirection.DOWNSTREAM, 1))
        stopped_frame = RealtimeManualUserStoppedSpeakingFrame(
            audio_sample_cursor=16_000,
            sample_rate=16_000,
        )
        await _push(observer, stopped_frame, source=boundary_source, destination=self.destination)
        await _push(
            observer,
            TranscriptionFrame(text="hello", user_id="", timestamp="", finalized=True),
            source=self.source,
            destination=self.destination,
        )
        await _push(
            observer,
            RTVIServerMessageFrame(
                data={
                    "type": "user-turn-finalized",
                    "turn_frame_id": stopped_frame.id,
                    "transcript": "hello",
                }
            ),
            source=self.source,
            destination=self.destination,
        )

        completed = next(event for event in recorder.events if event["type"].endswith("transcription.completed"))
        self.assertEqual(completed["usage"], {"type": "duration", "seconds": 1.0})
        observer.shutdown()

    async def test_manual_cumulative_cursors_report_each_turn_duration(self) -> None:
        recorder = _Recorder()
        observer = _observer(recorder, _controller(manual=True))
        boundary_source = FrameProcessor(name="manual-input-transport")

        async def complete_turn(*, start: int, stop: int, transcript: str) -> None:
            await _push(
                observer,
                RealtimeManualUserStartedSpeakingFrame(
                    audio_sample_cursor=start,
                    sample_rate=16_000,
                ),
                source=boundary_source,
                destination=self.destination,
            )
            stopped_frame = RealtimeManualUserStoppedSpeakingFrame(
                audio_sample_cursor=stop,
                sample_rate=16_000,
            )
            await _push(observer, stopped_frame, source=boundary_source, destination=self.destination)
            await _push(
                observer,
                TranscriptionFrame(text=transcript, user_id="", timestamp="", finalized=True),
                source=self.source,
                destination=self.destination,
            )
            await _push(
                observer,
                RTVIServerMessageFrame(
                    data={
                        "type": "user-turn-finalized",
                        "turn_frame_id": stopped_frame.id,
                        "transcript": transcript,
                    }
                ),
                source=self.source,
                destination=self.destination,
            )

        await complete_turn(start=0, stop=3_200, transcript="first")
        await complete_turn(start=3_200, stop=9_600, transcript="second")

        completed = [
            event
            for event in recorder.events
            if event["type"] == "conversation.item.input_audio_transcription.completed"
        ]
        self.assertEqual(
            [event["usage"] for event in completed],
            [
                {"type": "duration", "seconds": 0.2},
                {"type": "duration", "seconds": 0.4},
            ],
        )
        self.assertNotEqual(completed[0]["item_id"], completed[1]["item_id"])
        observer.shutdown()

    async def test_server_vad_speech_start_applies_negotiated_prefix_padding(self) -> None:
        recorder = _Recorder()
        controller = _controller(transcription=False)
        controller.apply_session_update(
            {
                "audio": {
                    "input": {
                        "turn_detection": {
                            "type": "server_vad",
                            "prefix_padding_ms": 250,
                        }
                    }
                }
            }
        )
        observer = _observer(recorder, controller)
        audio = InputAudioRawFrame(audio=b"\x00\x00" * 16_000, sample_rate=16_000, num_channels=1)
        await observer.on_process_frame(FrameProcessed(self.source, audio, FrameDirection.DOWNSTREAM, 0))

        await _push(
            observer,
            VADUserStartedSpeakingFrame(start_secs=0.1),
            source=self.source,
            destination=self.destination,
        )
        await _push(observer, UserStartedSpeakingFrame(), source=self.source, destination=self.destination)

        started = [event for event in recorder.events if event["type"] == "input_audio_buffer.speech_started"]
        self.assertEqual(len(started), 1)
        self.assertEqual(started[0]["audio_start_ms"], 650)
        observer.shutdown()

    async def test_fused_transcript_after_stop_uses_immutable_frame_token(self) -> None:
        recorder = _Recorder()
        observer = _observer(recorder, _controller(fused=True))
        await _push(observer, UserStartedSpeakingFrame(), source=self.source, destination=self.destination)
        stopped_frame = UserStoppedSpeakingFrame()
        await _push(observer, stopped_frame, source=self.source, destination=self.destination)
        transcript = TranscriptionFrame(text="hello world", user_id="user", timestamp="", finalized=True)
        transcript.metadata[USER_TRANSCRIPT_TURN_FRAME_ID_METADATA] = stopped_frame.id
        await _push(
            observer,
            transcript,
            source=self.source,
            destination=self.destination,
            direction=FrameDirection.UPSTREAM,
        )
        await _push(
            observer,
            UserTranscriptProducerEndedFrame(status="completed", turn_frame_id=stopped_frame.id),
            source=self.source,
            destination=self.destination,
            direction=FrameDirection.UPSTREAM,
        )

        committed = next(event for event in recorder.events if event["type"] == "input_audio_buffer.committed")
        completed = next(event for event in recorder.events if event["type"].endswith("transcription.completed"))
        self.assertEqual(completed["item_id"], committed["item_id"])
        self.assertEqual(completed["transcript"], "hello world")
        self.assertNotIn("error", recorder.types)
        observer.shutdown()

    async def test_late_fused_frames_for_a_retired_turn_token_are_ignored(self) -> None:
        recorder = _Recorder()
        close = AsyncMock()
        controller = _controller(fused=True)
        observer = _observer(recorder, controller, fail_input_audio_turn=close)
        await _push(observer, UserStartedSpeakingFrame(), source=self.source, destination=self.destination)
        first_stop = UserStoppedSpeakingFrame()
        await _push(observer, first_stop, source=self.source, destination=self.destination)
        first_transcript = TranscriptionFrame(text="turn one", user_id="user", timestamp="", finalized=True)
        first_transcript.metadata[USER_TRANSCRIPT_TURN_FRAME_ID_METADATA] = first_stop.id
        await _push(
            observer,
            first_transcript,
            source=self.source,
            destination=self.destination,
            direction=FrameDirection.UPSTREAM,
        )
        await _push(
            observer,
            UserTranscriptProducerEndedFrame(status="completed", turn_frame_id=first_stop.id),
            source=self.source,
            destination=self.destination,
            direction=FrameDirection.UPSTREAM,
        )
        first_completed = next(event for event in recorder.events if event["type"].endswith("transcription.completed"))
        event_count = len(recorder.events)

        late_interim = InterimTranscriptionFrame(text="late interim", user_id="user", timestamp="")
        late_interim.metadata[USER_TRANSCRIPT_TURN_FRAME_ID_METADATA] = first_stop.id
        await _push(
            observer,
            late_interim,
            source=self.source,
            destination=self.destination,
            direction=FrameDirection.UPSTREAM,
        )
        late_final = TranscriptionFrame(text="late final", user_id="user", timestamp="", finalized=True)
        late_final.metadata[USER_TRANSCRIPT_TURN_FRAME_ID_METADATA] = first_stop.id
        await _push(
            observer,
            late_final,
            source=self.source,
            destination=self.destination,
            direction=FrameDirection.UPSTREAM,
        )
        await _push(
            observer,
            UserTranscriptProducerEndedFrame(status="completed", turn_frame_id=first_stop.id),
            source=self.source,
            destination=self.destination,
            direction=FrameDirection.UPSTREAM,
        )

        self.assertEqual(len(recorder.events), event_count)
        self.assertIsNone(controller._user_item_id)
        self.assertEqual(controller._pending_user_interim, "")
        self.assertEqual(controller._pending_user_transcript.text, "")
        self.assertFalse(observer._input_transcription_terminal)
        close.assert_not_awaited()

        await _push(observer, UserStartedSpeakingFrame(), source=self.source, destination=self.destination)
        second_stop = UserStoppedSpeakingFrame()
        await _push(observer, second_stop, source=self.source, destination=self.destination)
        second_transcript = TranscriptionFrame(text="turn two", user_id="user", timestamp="", finalized=True)
        second_transcript.metadata[USER_TRANSCRIPT_TURN_FRAME_ID_METADATA] = second_stop.id
        await _push(
            observer,
            second_transcript,
            source=self.source,
            destination=self.destination,
            direction=FrameDirection.UPSTREAM,
        )
        await _push(
            observer,
            UserTranscriptProducerEndedFrame(status="completed", turn_frame_id=second_stop.id),
            source=self.source,
            destination=self.destination,
            direction=FrameDirection.UPSTREAM,
        )
        completed = [event for event in recorder.events if event["type"].endswith("transcription.completed")]
        self.assertEqual([event["transcript"] for event in completed], ["turn one", "turn two"])
        self.assertNotEqual(first_completed["item_id"], completed[-1]["item_id"])
        self.assertNotIn("late", json.dumps(recorder.events))
        observer.shutdown()

    async def test_fused_transcript_without_owner_token_closes_the_input_turn(self) -> None:
        recorder = _Recorder()
        close = AsyncMock()
        observer = _observer(recorder, _controller(fused=True), fail_input_audio_turn=close)
        await _push(observer, UserStartedSpeakingFrame(), source=self.source, destination=self.destination)
        await _push(observer, UserStoppedSpeakingFrame(), source=self.source, destination=self.destination)
        await _push(
            observer,
            TranscriptionFrame(text="must not use FIFO", user_id="user", timestamp="", finalized=True),
            source=self.source,
            destination=self.destination,
            direction=FrameDirection.UPSTREAM,
        )
        self.assertNotIn("conversation.item.input_audio_transcription.completed", recorder.types)
        error = next(event for event in recorder.events if event["type"] == "error")
        self.assertEqual(error["error"]["code"], "omni_transcription_owner_missing")
        close.assert_awaited_once()
        observer.shutdown()

    async def test_realtime_asr_ownership_error_is_published_and_closes_input(self) -> None:
        recorder = _Recorder()
        close = AsyncMock()
        observer = _observer(recorder, _controller(), fail_input_audio_turn=close)

        await _push(
            observer,
            RealtimeInputTranscriptionErrorFrame(
                code="asr_transcription_word_offsets_missing",
                message="NVIDIA ASR returned no word offsets",
            ),
            source=self.source,
            destination=self.destination,
        )

        error = next(event for event in recorder.events if event["type"] == "error")
        self.assertEqual(error["error"]["code"], "asr_transcription_word_offsets_missing")
        self.assertEqual(error["error"]["param"], "session.audio.input.transcription")
        self.assertTrue(observer._input_transcription_terminal)
        close.assert_awaited_once()
        observer.shutdown()

    async def test_realtime_asr_ownership_is_private_when_public_transcription_is_disabled(self) -> None:
        recorder = _Recorder()
        close = AsyncMock()
        observer = _observer(
            recorder,
            _controller(transcription=False),
            fail_input_audio_turn=close,
        )

        await _push(
            observer,
            RealtimeInputTranscriptionErrorFrame(
                code="asr_transcription_word_offsets_missing",
                message="NVIDIA ASR returned no word offsets",
            ),
            source=self.source,
            destination=self.destination,
        )
        await _push(
            observer,
            TranscriptionFrame(text="pipeline-only text", user_id="", timestamp="", finalized=True),
            source=self.source,
            destination=self.destination,
        )

        self.assertEqual(recorder.events, [])
        self.assertFalse(observer._input_transcription_terminal)
        close.assert_not_awaited()
        observer.shutdown()

    async def test_skipped_fused_turn_is_nonfatal_and_next_turn_can_complete(self) -> None:
        recorder = _Recorder()
        close = AsyncMock()
        observer = _observer(recorder, _controller(fused=True), fail_input_audio_turn=close)
        await _push(observer, UserStartedSpeakingFrame(), source=self.source, destination=self.destination)
        skipped_stop = UserStoppedSpeakingFrame()
        await _push(observer, skipped_stop, source=self.source, destination=self.destination)
        await _push(
            observer,
            UserTranscriptProducerEndedFrame(status="skipped", turn_frame_id=skipped_stop.id),
            source=self.source,
            destination=self.destination,
            direction=FrameDirection.UPSTREAM,
        )
        failure = next(event for event in recorder.events if event["type"].endswith("transcription.failed"))
        self.assertEqual(failure["error"]["code"], "omni_transcription_skipped")
        self.assertNotIn("error", recorder.types)
        close.assert_not_awaited()

        await _push(observer, UserStartedSpeakingFrame(), source=self.source, destination=self.destination)
        next_stop = UserStoppedSpeakingFrame()
        await _push(observer, next_stop, source=self.source, destination=self.destination)
        transcript = TranscriptionFrame(text="next turn works", user_id="user", timestamp="", finalized=True)
        transcript.metadata[USER_TRANSCRIPT_TURN_FRAME_ID_METADATA] = next_stop.id
        await _push(
            observer,
            transcript,
            source=self.source,
            destination=self.destination,
            direction=FrameDirection.UPSTREAM,
        )
        completed = [event for event in recorder.events if event["type"].endswith("transcription.completed")]
        self.assertEqual(len(completed), 1)
        self.assertEqual(completed[0]["transcript"], "next turn works")
        observer.shutdown()

    async def test_overflowed_fused_turn_reports_native_buffer_error_and_stays_usable(self) -> None:
        recorder = _Recorder()
        close = AsyncMock()
        observer = _observer(recorder, _controller(fused=True), fail_input_audio_turn=close)
        await _push(observer, UserStartedSpeakingFrame(), source=self.source, destination=self.destination)
        overflowed_stop = UserStoppedSpeakingFrame()
        await _push(observer, overflowed_stop, source=self.source, destination=self.destination)
        await _push(
            observer,
            UserTranscriptProducerEndedFrame(status="overflowed", turn_frame_id=overflowed_stop.id),
            source=self.source,
            destination=self.destination,
            direction=FrameDirection.UPSTREAM,
        )

        failure = next(event for event in recorder.events if event["type"].endswith("transcription.failed"))
        self.assertEqual(failure["error"]["code"], "input_buffer_overflow")
        error = next(event for event in recorder.events if event["type"] == "error")
        self.assertEqual(error["error"]["type"], "invalid_request_error")
        self.assertEqual(error["error"]["code"], "input_buffer_overflow")
        self.assertEqual(error["error"]["param"], "audio")
        close.assert_not_awaited()

        await _push(observer, UserStartedSpeakingFrame(), source=self.source, destination=self.destination)
        next_stop = UserStoppedSpeakingFrame()
        await _push(observer, next_stop, source=self.source, destination=self.destination)
        transcript = TranscriptionFrame(text="next turn works", user_id="user", timestamp="", finalized=True)
        transcript.metadata[USER_TRANSCRIPT_TURN_FRAME_ID_METADATA] = next_stop.id
        await _push(
            observer,
            transcript,
            source=self.source,
            destination=self.destination,
            direction=FrameDirection.UPSTREAM,
        )
        completed = [event for event in recorder.events if event["type"].endswith("transcription.completed")]
        self.assertEqual(completed[-1]["transcript"], "next turn works")
        observer.shutdown()

    async def test_late_first_turn_terminal_does_not_poison_second_turn(self) -> None:
        recorder = _Recorder()
        observer = _observer(recorder, _controller(fused=True))
        await _push(observer, UserStartedSpeakingFrame(), source=self.source, destination=self.destination)
        first_stop = UserStoppedSpeakingFrame()
        await _push(observer, first_stop, source=self.source, destination=self.destination)
        first_transcript = TranscriptionFrame(text="turn one", user_id="user", timestamp="", finalized=True)
        first_transcript.metadata[USER_TRANSCRIPT_TURN_FRAME_ID_METADATA] = first_stop.id
        await _push(
            observer,
            first_transcript,
            source=self.source,
            destination=self.destination,
            direction=FrameDirection.UPSTREAM,
        )

        await _push(observer, UserStartedSpeakingFrame(), source=self.source, destination=self.destination)
        second_stop = UserStoppedSpeakingFrame()
        await _push(observer, second_stop, source=self.source, destination=self.destination)
        await _push(
            observer,
            UserTranscriptProducerEndedFrame(status="cancelled", turn_frame_id=first_stop.id),
            source=self.source,
            destination=self.destination,
            direction=FrameDirection.UPSTREAM,
        )
        second_transcript = TranscriptionFrame(text="turn two", user_id="user", timestamp="", finalized=True)
        second_transcript.metadata[USER_TRANSCRIPT_TURN_FRAME_ID_METADATA] = second_stop.id
        await _push(
            observer,
            second_transcript,
            source=self.source,
            destination=self.destination,
            direction=FrameDirection.UPSTREAM,
        )
        completed = [
            event["transcript"] for event in recorder.events if event["type"].endswith("transcription.completed")
        ]
        self.assertEqual(completed, ["turn one", "turn two"])
        self.assertNotIn("conversation.item.input_audio_transcription.failed", recorder.types)
        observer.shutdown()

    async def test_late_rtvi_user_turn_marker_is_diagnostic_only(self) -> None:
        recorder = _Recorder()
        observer = _observer(recorder, _controller())
        marker = RTVIServerMessageFrame(data={"type": "user-turn-finalized", "transcript": "stale"})
        await _push(observer, marker, source=self.source, destination=self.destination)
        self.assertEqual(recorder.events, [])
        observer.shutdown()

    async def test_cascaded_transcript_watchdog_fails_then_closes(self) -> None:
        recorder = _Recorder()
        close = AsyncMock()
        observer = _observer(
            recorder,
            _controller(),
            fail_input_audio_turn=close,
            input_transcription_timeout_secs=0.01,
        )
        await _push(observer, UserStartedSpeakingFrame(), source=self.source, destination=self.destination)
        await _push(observer, UserStoppedSpeakingFrame(), source=self.source, destination=self.destination)
        await asyncio.sleep(0.03)
        failure = next(event for event in recorder.events if event["type"].endswith("transcription.failed"))
        self.assertEqual(failure["error"]["code"], "transcription_timeout")
        error = next(event for event in recorder.events if event["type"] == "error")
        self.assertEqual(error["error"]["code"], "asr_transcription_timeout")
        close.assert_awaited_once()
        observer.shutdown()

    async def test_manual_transcript_watchdog_finishes_queued_response_before_close(self) -> None:
        recorder = _Recorder()
        close = AsyncMock()
        controller = _controller(manual=True)
        observer = _observer(
            recorder,
            controller,
            fail_input_audio_turn=close,
            input_transcription_timeout_secs=0.01,
        )
        await _push(observer, UserStartedSpeakingFrame(), source=self.source, destination=self.destination)
        await _push(observer, UserStoppedSpeakingFrame(), source=self.source, destination=self.destination)
        controller.start_response()

        await asyncio.sleep(0.03)

        failure = next(event for event in recorder.events if event["type"].endswith("transcription.failed"))
        self.assertEqual(failure["error"]["code"], "transcription_timeout")
        error = next(event for event in recorder.events if event["type"] == "error")
        self.assertEqual(error["error"]["code"], "asr_transcription_timeout")
        done = next(event for event in recorder.events if event["type"] == "response.done")
        self.assertEqual(done["response"]["status"], "failed")
        self.assertEqual(
            done["response"]["status_details"]["error"]["code"],
            "asr_transcription_timeout",
        )
        self.assertLess(
            recorder.types.index("conversation.item.input_audio_transcription.failed"),
            recorder.types.index("error"),
        )
        self.assertLess(recorder.types.index("error"), recorder.types.index("response.done"))
        self.assertFalse(controller.response_in_progress)
        close.assert_awaited_once()
        observer.shutdown()


class ObserverResponseTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.source = FrameProcessor(name="response-source")
        self.destination = FrameProcessor(name="response-destination")

    async def test_downstream_interruption_cancels_active_response_once(self) -> None:
        recorder = _Recorder()
        controller = _controller()
        controller.append_assistant_text("partial")
        observer = _observer(recorder, controller)
        interruption = InterruptionFrame()
        await _push(observer, interruption, source=self.source, destination=self.destination)
        done = next(event for event in recorder.events if event["type"] == "response.done")
        self.assertEqual(done["response"]["status"], "cancelled")
        self.assertIsNone(controller.active_response_id)
        recorder.events.clear()
        await _push(observer, interruption, source=self.source, destination=self.destination)
        self.assertEqual(recorder.events, [])
        observer.shutdown()

    async def test_delayed_interruption_cannot_cancel_a_replacement_response(self) -> None:
        recorder = _Recorder()
        controller = _controller()
        observer = _observer(recorder, controller)
        response_a = controller.start_response()[0]["response"]["id"]
        interruption_a = InterruptionFrame()
        controller.observe_interruption(
            interruption_a.id,
            reason="client_cancelled",
            response_id=response_a,
        )
        controller.finish_response(status="cancelled", reason="client_cancelled")
        response_b = controller.start_response()[0]["response"]["id"]

        await _push(observer, interruption_a, source=self.source, destination=self.destination)

        self.assertEqual(controller.active_response_id, response_b)
        self.assertNotIn("response.done", recorder.types)
        observer.shutdown()

    async def test_accepted_client_cancel_wins_a_racing_provider_completion(self) -> None:
        recorder = _Recorder()
        controller = _controller(output_kind="text")
        response_id = controller.start_response()[0]["response"]["id"]
        observer = _observer(recorder, controller)
        serializer = _serializer(recorder, controller)
        cancel_frame = await serializer.deserialize(json.dumps({"type": "response.cancel", "response_id": response_id}))
        self.assertIsInstance(cancel_frame, InterruptionFrame)

        await _push(
            observer,
            RealtimeOwnedLLMFullResponseStartFrame(response_id=response_id),
            source=self.source,
            destination=self.destination,
        )
        end = LLMFullResponseEndFrame()
        await _push(observer, end, source=self.source, destination=self.destination)
        output = MagicMock(spec=BaseOutputTransport)
        await _push(observer, end, source=output, destination=self.destination)

        done = next(event for event in recorder.events if event["type"] == "response.done")
        self.assertEqual(done["response"]["status"], "cancelled")
        self.assertEqual(done["response"]["status_details"]["reason"], "client_cancelled")
        self.assertIsNone(controller.active_response_id)

        recorder.events.clear()
        await _push(observer, cancel_frame, source=self.source, destination=self.destination)
        self.assertEqual(recorder.events, [])
        observer.shutdown()

    async def test_first_observer_interruption_rejects_delayed_owned_llm_start(self) -> None:
        recorder = _Recorder()
        controller = _controller(output_kind="text")
        observer = _observer(recorder, controller)
        gate = RealtimeManualResponseGate(controller=controller)
        gate.push_frame = AsyncMock()
        gate.register_automatic_response_context()

        await gate.process_frame(
            LLMContextFrame(LLMContext([{"role": "user", "content": "start a response"}])),
            FrameDirection.DOWNSTREAM,
        )
        self.assertTrue(controller.pipeline_response_pending)
        response_context = gate.push_frame.await_args_list[-1].args[0].context
        stale_start = RealtimeOwnedLLMFullResponseStartFrame(
            run_owner_id=response_context.run_owner_id,
            activation_generation=response_context.activation_generation,
        )

        # Model Pipecat's processor ordering: the observer can see the system
        # interruption before it reaches the response gate, while a provider
        # start from the now-stale unowned run is already queued behind it.
        interruption = InterruptionFrame()
        await _push(observer, interruption, source=self.source, destination=self.destination)
        await _push(
            observer,
            stale_start,
            source=self.source,
            destination=self.destination,
        )

        self.assertFalse(controller.pipeline_response_pending)
        self.assertFalse(controller.response_in_progress)
        self.assertNotIn("response.created", recorder.types)
        gate.reset()
        observer.shutdown()

    async def test_stale_owned_start_cannot_claim_replacement_pending_run(self) -> None:
        recorder = _Recorder()
        controller = _controller(output_kind="text")
        observer = _observer(recorder, controller)
        generation_a = controller.interruption_generation
        controller.prepare_pipeline_response(
            owner_id="run_A",
            interruption_generation=generation_a,
        )

        # The interruption retires run A. A new turn can reserve run B before
        # the cancelled provider task's already-queued start reaches the
        # observer; that stale start must not consume run B's snapshot.
        await _push(
            observer,
            InterruptionFrame(),
            source=self.source,
            destination=self.destination,
        )
        generation_b = controller.interruption_generation
        controller.prepare_pipeline_response(
            owner_id="run_B",
            interruption_generation=generation_b,
        )
        await _push(
            observer,
            RealtimeOwnedLLMFullResponseStartFrame(
                run_owner_id="run_A",
                activation_generation=generation_a,
            ),
            source=self.source,
            destination=self.destination,
        )

        self.assertTrue(controller.pipeline_response_pending)
        self.assertFalse(controller.response_in_progress)
        self.assertNotIn("response.created", recorder.types)

        await _push(
            observer,
            RealtimeOwnedLLMFullResponseStartFrame(
                run_owner_id="run_B",
                activation_generation=generation_b,
            ),
            source=self.source,
            destination=self.destination,
        )
        self.assertFalse(controller.pipeline_response_pending)
        self.assertTrue(controller.response_in_progress)
        self.assertEqual(recorder.types.count("response.created"), 1)
        observer.shutdown()

    async def test_unowned_start_cannot_attach_to_an_active_response(self) -> None:
        recorder = _Recorder()
        controller = _controller(output_kind="text")
        response_id = controller.start_response()[0]["response"]["id"]
        observer = _observer(recorder, controller)

        stale_start = LLMFullResponseStartFrame()
        await _push(observer, stale_start, source=self.source, destination=self.destination)
        await _push(
            observer,
            LLMTextFrame(text="stale provider output"),
            source=self.source,
            destination=self.destination,
        )
        stale_end = LLMFullResponseEndFrame()
        await _push(observer, stale_end, source=self.source, destination=self.destination)
        output = MagicMock(spec=BaseOutputTransport)
        await _push(observer, stale_end, source=output, destination=self.destination)

        self.assertEqual(controller.active_response_id, response_id)
        self.assertTrue(controller.response_in_progress)
        self.assertNotIn("response.done", recorder.types)
        errors = [event for event in recorder.events if event["type"] == "error"]
        self.assertEqual(len(errors), 1)
        self.assertEqual(errors[0]["error"]["code"], "llm_response_owner_missing")
        observer.shutdown()

    async def test_uncorrelated_nonfatal_error_preserves_active_response(self) -> None:
        recorder = _Recorder()
        controller = _controller()
        controller.start_response()
        observer = _observer(recorder, controller)
        await _push(
            observer, ErrorFrame(error="unowned service warning"), source=self.source, destination=self.destination
        )
        self.assertEqual(recorder.types, ["error"])
        self.assertTrue(controller.response_in_progress)
        observer.shutdown()

    async def test_correlated_realtime_context_error_preserves_native_error_fields(self) -> None:
        recorder = _Recorder()
        controller = _controller(output_kind="text")
        controller.start_response()
        response_id = controller.active_response_id
        self.assertIsNotNone(response_id)
        observer = _observer(recorder, controller)
        await _push(
            observer,
            RealtimeOwnedLLMFullResponseStartFrame(response_id=response_id or ""),
            source=self.source,
            destination=self.destination,
        )
        failure = RealtimeProtocolError(
            message="The conversation exceeds the model input-token limit and truncation is disabled",
            code="context_length_exceeded",
            param="session.truncation",
            event_id="response_create_1",
        )

        await _push(
            observer,
            ErrorFrame(error=str(failure), exception=failure),
            source=self.source,
            destination=self.destination,
        )

        error = next(event for event in recorder.events if event["type"] == "error")
        self.assertEqual(
            error["error"],
            {
                "type": "invalid_request_error",
                "message": failure.message,
                "code": "context_length_exceeded",
                "param": "session.truncation",
                "event_id": "response_create_1",
            },
        )
        done = next(event for event in recorder.events if event["type"] == "response.done")
        self.assertEqual(done["response"]["status"], "failed")
        self.assertEqual(done["response"]["status_details"]["error"]["code"], "context_length_exceeded")
        self.assertFalse(controller.response_in_progress)
        observer.shutdown()

    async def test_fatal_pipeline_error_finishes_active_response_as_failed(self) -> None:
        recorder = _Recorder()
        controller = _controller()
        controller.start_response()
        observer = _observer(recorder, controller)
        await _push(
            observer, FatalErrorFrame(error="pipeline unavailable"), source=self.source, destination=self.destination
        )
        error_index = recorder.types.index("error")
        done_index = recorder.types.index("response.done")
        self.assertLess(error_index, done_index)
        self.assertEqual(recorder.events[error_index]["error"]["code"], "pipeline_error")
        done = next(event for event in recorder.events if event["type"] == "response.done")
        self.assertEqual(done["response"]["status"], "failed")
        self.assertEqual(done["response"]["status_details"]["error"]["code"], "pipeline_error")
        observer.shutdown()

    async def test_function_call_is_announced_as_canonical_response_output(self) -> None:
        recorder = _Recorder()
        controller = _controller(tool_owner="client")
        observer = _observer(recorder, controller)
        await _push(
            observer,
            FunctionCallInProgressFrame(
                function_name="lookup",
                tool_call_id="call_abc",
                arguments={"key": "intro"},
            ),
            source=self.source,
            destination=self.destination,
        )
        self.assertIn("response.created", recorder.types)
        self.assertIn("response.function_call_arguments.done", recorder.types)
        arguments = next(event for event in recorder.events if event["type"] == "response.function_call_arguments.done")
        self.assertEqual(json.loads(arguments["arguments"]), {"key": "intro"})
        self.assertEqual(controller.tool_call("call_abc").owner, "client")
        observer.shutdown()


class SerializerControlTests(unittest.IsolatedAsyncioTestCase):
    async def test_audio_response_lazily_resolves_voice_catalog_and_retries_after_failure(self) -> None:
        voice_a = "voice-a"
        voice_b = "voice-b"
        attempts = 0

        async def resolve() -> frozenset[str]:
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise RuntimeError("catalog unavailable")
            return frozenset({voice_a, voice_b})

        recorder = _Recorder()
        controller = RealtimeSessionController(
            model="test-model",
            voice=voice_a,
            runtime_config={},
            instructions="prompt",
            capabilities=RealtimeSessionCapabilities(voices=frozenset({voice_a})),
            output_voice_resolver=resolve,
            output_voices_resolved=False,
        )
        controller.apply_session_update({"output_modalities": ["text"]})
        serializer = _serializer(recorder, controller)
        context = LLMContext([{"role": "system", "content": "prompt"}])
        serializer.bind_context(context, instructions_renderer=lambda text: [{"role": "system", "content": text}])
        gate = RealtimeManualResponseGate(controller=controller)
        gate.bind_tts_service(object())
        gate.push_frame = AsyncMock()
        serializer.set_response_gate(gate)
        request = json.dumps(
            {
                "type": "response.create",
                "response": {
                    "output_modalities": ["audio"],
                    "audio": {"output": {"voice": voice_b}},
                },
            }
        )

        rejected = await serializer.deserialize(request)
        self.assertIsNone(rejected)
        self.assertEqual(recorder.events[-1]["error"]["code"], "services_not_ready")
        self.assertEqual(controller.public_session()["output_modalities"], ["text"])
        self.assertFalse(controller.response_in_progress)

        accepted = await serializer.deserialize(request)
        self.assertIsInstance(accepted, RealtimeResponseCreateFrame)
        self.assertEqual(attempts, 2)
        self.assertEqual(controller.session.capabilities.voices, frozenset({voice_a, voice_b}))

    async def test_live_transcription_toggle_redacts_public_text_without_changing_asr_route(self) -> None:
        recorder = _Recorder()
        controller = RealtimeSessionController(
            model="test-model",
            voice="test-voice",
            runtime_config={
                "asr_server": "localhost:50051",
                "realtime_input_transcription_model": "test-asr",
                "asr_language_code": "en-US",
            },
            input_transcription_model="test-asr",
            capabilities=RealtimeSessionCapabilities(
                input_transcription_models=frozenset({"test-asr"}),
                input_transcription_language_aliases=(("en-us", "en-US"),),
                supports_input_transcription_disable=True,
            ),
        )
        serializer = _serializer(recorder, controller)
        asr_route = copy.deepcopy(controller.runtime_config)

        snapshot_item_id, _audio_start_ms = controller.begin_user_turn(
            audio_sample_cursor=0,
            sample_rate=16_000,
        )
        _item_id, snapshot_segments = controller.set_user_transcript(
            "snapshot partial",
            item_id=snapshot_item_id,
        )
        self.assertEqual(snapshot_segments, [])
        committed_item_id, _audio_end_ms, snapshot_events = controller.commit_user_turn(
            audio_sample_cursor=8_000,
            sample_rate=16_000,
        )
        self.assertEqual(committed_item_id, snapshot_item_id)
        await recorder.emit_batch(snapshot_events)

        disabled = await serializer.deserialize(
            json.dumps(
                {
                    "type": "session.update",
                    "session": {"audio": {"input": {"transcription": None}}},
                }
            )
        )
        self.assertIsNone(disabled)
        self.assertIsNone(recorder.events[-1]["session"]["audio"]["input"]["transcription"])
        self.assertEqual(controller.runtime_config, asr_route)

        finalized_item_id, snapshot_final_events = controller.finalize_user_transcript(
            item_id=snapshot_item_id,
            transcript="snapshot final",
        )
        self.assertEqual(finalized_item_id, snapshot_item_id)
        await recorder.emit_batch(snapshot_final_events)
        snapshot_completed = next(
            event
            for event in snapshot_final_events
            if event["type"] == "conversation.item.input_audio_transcription.completed"
        )
        self.assertEqual(snapshot_completed["transcript"], "snapshot final")
        self.assertNotIn(snapshot_item_id, controller._stopped_user_transcripts)
        self.assertNotIn(snapshot_item_id, controller._stopped_user_transcript_publication_enabled)

        hidden_item_id, _audio_start_ms = controller.begin_user_turn(
            audio_sample_cursor=8_000,
            sample_rate=16_000,
        )
        hidden_events = controller.user_transcript_delta("private interim", item_id=hidden_item_id)
        _item_id, final_events = controller.set_user_transcript("private final", item_id=hidden_item_id)
        hidden_events.extend(final_events)
        stopped_hidden_item_id, _audio_end_ms, stopped_events = controller.stop_user_turn(
            audio_sample_cursor=24_000,
            sample_rate=16_000,
        )
        self.assertEqual(stopped_hidden_item_id, hidden_item_id)
        hidden_events.extend(stopped_events)
        await recorder.emit_batch(hidden_events)

        self.assertNotIn("private", json.dumps(hidden_events))
        self.assertFalse(any("input_audio_transcription" in event["type"] for event in hidden_events))
        hidden_done = next(event for event in hidden_events if event["type"] == "conversation.item.done")
        self.assertIsNone(hidden_done["item"]["content"][0]["transcript"])
        self.assertIsNone(controller.conversation.item(hidden_item_id)["content"][0]["transcript"])
        self.assertEqual(controller.runtime_config, asr_route)

        enabled = await serializer.deserialize(
            json.dumps(
                {
                    "type": "session.update",
                    "session": {
                        "audio": {
                            "input": {
                                "transcription": {
                                    "model": "test-asr",
                                    "language": "en-US",
                                }
                            }
                        }
                    },
                }
            )
        )
        self.assertIsNone(enabled)
        self.assertEqual(
            recorder.events[-1]["session"]["audio"]["input"]["transcription"],
            {"model": "test-asr", "language": "en-US"},
        )
        self.assertEqual(controller.runtime_config, asr_route)

        visible_item_id, _audio_start_ms = controller.begin_user_turn(
            audio_sample_cursor=24_000,
            sample_rate=16_000,
        )
        visible_events = controller.user_transcript_delta("visible interim", item_id=visible_item_id)
        stopped_visible_item_id, _audio_end_ms, stopped_events = controller.stop_user_turn(
            audio_sample_cursor=40_000,
            sample_rate=16_000,
        )
        self.assertEqual(stopped_visible_item_id, visible_item_id)
        visible_events.extend(stopped_events)
        _item_id, final_events = controller.set_user_transcript(
            "visible final",
            item_id=visible_item_id,
        )
        visible_events.extend(final_events)
        await recorder.emit_batch(visible_events)

        visible_types = {event["type"] for event in visible_events}
        self.assertIn("conversation.item.input_audio_transcription.delta", visible_types)
        self.assertIn("conversation.item.input_audio_transcription.completed", visible_types)
        visible_done = next(event for event in visible_events if event["type"] == "conversation.item.done")
        self.assertIsNone(visible_done["item"]["content"][0]["transcript"])
        self.assertEqual(controller.conversation.item(visible_item_id)["content"][0]["transcript"], "visible final")
        self.assertIn("visible final", json.dumps(visible_events))
        self.assertEqual(controller.runtime_config, asr_route)

    async def test_invalid_metadata_is_rejected_before_mcp_preparation(self) -> None:
        recorder = _Recorder()
        controller = RealtimeSessionController(
            model="test-model",
            voice="test-voice",
            runtime_config={"asr_server": "localhost:50051"},
            capabilities=RealtimeSessionCapabilities(
                voices=frozenset({"test-voice"}),
                mcp_tools=True,
            ),
        )
        serializer = _serializer(recorder, controller)
        gate = RealtimeManualResponseGate(controller=controller)
        serializer.set_response_gate(gate)
        mcp_runtime = SimpleNamespace(prepare_tools=AsyncMock())
        serializer.set_mcp_runtime(mcp_runtime)
        metadata = {f"key_{index}": "value" for index in range(17)}

        frame = await serializer.deserialize(
            json.dumps(
                {
                    "type": "response.create",
                    "response": {
                        "metadata": metadata,
                        "tools": [
                            {
                                "type": "mcp",
                                "server_label": "remote",
                                "server_url": "https://mcp.example.test/rpc",
                            }
                        ],
                    },
                }
            )
        )

        self.assertIsNone(frame)
        mcp_runtime.prepare_tools.assert_not_awaited()
        self.assertEqual(recorder.types, ["error"])
        self.assertEqual(recorder.events[0]["error"]["code"], "invalid_value")
        self.assertEqual(recorder.events[0]["error"]["param"], "response.metadata")

    async def test_non_finite_json_number_is_rejected_before_dispatch(self) -> None:
        recorder = _Recorder()
        controller = _controller()
        serializer = _serializer(recorder, controller)

        frame = await serializer.deserialize('{"type":"response.create","response":{"metadata":{"score":NaN}}}')

        self.assertIsNone(frame)
        self.assertFalse(controller.response_in_progress)
        self.assertEqual(recorder.types, ["error"])
        self.assertEqual(recorder.events[0]["error"]["code"], "invalid_json")

    async def test_response_cancel_waits_for_downstream_interruption(self) -> None:
        recorder = _Recorder()
        controller = _controller()
        controller.append_assistant_text("partial")
        serializer = _serializer(recorder, controller)
        observer = _observer(recorder, controller)
        serializer.set_on_response_cancel(observer.on_response_cancelled)

        frame = await serializer.deserialize(json.dumps({"type": "response.cancel"}))
        self.assertIsInstance(frame, InterruptionFrame)
        self.assertNotIn("response.done", recorder.types)
        await _push(
            observer,
            frame,
            source=FrameProcessor(name="cancel-source"),
            destination=FrameProcessor(name="cancel-destination"),
        )
        done = next(event for event in recorder.events if event["type"] == "response.done")
        self.assertEqual(done["response"]["status"], "cancelled")
        self.assertEqual(done["response"]["status_details"]["reason"], "client_cancelled")
        observer.shutdown()

    async def test_manual_response_cancel_before_transcript_cannot_launch_late_inference(self) -> None:
        recorder = _Recorder()
        controller = _controller(manual=True)
        serializer = _serializer(recorder, controller)
        observer = _observer(recorder, controller)
        gate = RealtimeManualResponseGate(controller=controller)
        gate.push_frame = AsyncMock()
        serializer.set_manual_input_handlers(commit_hook=AsyncMock(), response_gate=gate)
        serializer.set_on_response_cancel(observer.on_response_cancelled)
        gate.register_commit()

        create_frame = await serializer.deserialize(json.dumps({"type": "response.create"}))
        self.assertIsInstance(create_frame, RealtimeResponseCreateFrame)
        response_id = controller.active_response_id
        cancel_frame = await serializer.deserialize(json.dumps({"type": "response.cancel", "response_id": response_id}))
        self.assertIsInstance(cancel_frame, InterruptionFrame)
        await _push(
            observer,
            cancel_frame,
            source=FrameProcessor(name="manual-cancel-source"),
            destination=FrameProcessor(name="manual-cancel-destination"),
        )

        await gate.process_frame(create_frame, FrameDirection.DOWNSTREAM)
        await gate.process_frame(
            LLMContextFrame(LLMContext([{"role": "user", "content": "late transcript"}])),
            FrameDirection.DOWNSTREAM,
        )

        self.assertEqual(recorder.types.count("response.created"), 1)
        self.assertEqual(recorder.types.count("response.done"), 1)
        done = next(event for event in recorder.events if event["type"] == "response.done")
        self.assertEqual(done["response"]["status"], "cancelled")
        self.assertNotIn("response.output_item.added", recorder.types)
        gate.push_frame.assert_not_awaited()
        self.assertEqual(gate.pending_commits, 0)
        observer.shutdown()

    async def test_manual_empty_transcript_during_response_publication_cannot_launch_inference(self) -> None:
        recorder = _Recorder()
        controller = _controller(manual=True)
        serializer = _serializer(recorder, controller)
        gate = RealtimeManualResponseGate(controller=controller)
        gate.push_frame = AsyncMock()
        serializer.set_manual_input_handlers(commit_hook=AsyncMock(), response_gate=gate)
        gate.register_commit()
        item_id, _audio_start_ms = controller.begin_user_turn(audio_sample_cursor=0, sample_rate=16_000)
        stopped_item_id, _audio_end_ms, _events = controller.stop_user_turn(
            audio_sample_cursor=320,
            sample_rate=16_000,
        )
        self.assertEqual(stopped_item_id, item_id)
        failure_observed = False

        async def fail_while_response_created_is_published(events: list[dict[str, Any]]) -> None:
            nonlocal failure_observed
            recorder.events.extend(events)
            if not any(event["type"] == "response.created" for event in events):
                return
            response_id = controller.active_response_id
            self.assertEqual(gate._pending_response_marker_id, response_id)
            failure_observed = True
            failure_events = controller.fail_user_transcription(
                code="empty_transcript",
                message="Committed input audio did not produce a non-empty transcript",
                item_id=item_id,
            )
            terminal_events = controller.finish_response(
                status="failed",
                reason="input_audio_transcription_empty",
            )
            gate.fail_commit()
            recorder.events.extend([*failure_events, *terminal_events])

        serializer.set_emit(recorder.emit, fail_while_response_created_is_published)

        marker = await serializer.deserialize(json.dumps({"type": "response.create"}))

        self.assertTrue(failure_observed)
        self.assertIsInstance(marker, RealtimeResponseCreateFrame)
        self.assertFalse(controller.response_in_progress)
        await gate.process_frame(marker, FrameDirection.DOWNSTREAM)
        gate.push_frame.assert_not_awaited()
        self.assertIsNone(gate._pending_response_marker_id)
        self.assertFalse(gate._cancelled_response_marker_ids)
        self.assertEqual(recorder.types.count("response.created"), 1)
        self.assertEqual(recorder.types.count("response.done"), 1)

    async def test_manual_gate_releases_pre_marker_cancellation_state_each_time(self) -> None:
        gate = RealtimeManualResponseGate(controller=_controller(manual=True))
        gate.push_frame = AsyncMock()

        for index in range(20):
            response_id = f"resp_cancel_{index}"
            gate.register_response_marker(response_id)
            gate.cancel_response(response_id)
            await gate.process_frame(
                RealtimeResponseCreateFrame(response_id=response_id),
                FrameDirection.DOWNSTREAM,
            )

        self.assertIsNone(gate._pending_response_marker_id)
        self.assertFalse(gate._cancelled_response_marker_ids)
        gate.push_frame.assert_not_awaited()

    async def test_manual_commit_context_does_not_infer_before_response_create(self) -> None:
        controller = _controller(manual=True)
        gate = RealtimeManualResponseGate(controller=controller)
        gate.push_frame = AsyncMock()
        context = LLMContext([{"role": "user", "content": "committed transcript"}])
        gate.register_commit()

        await gate.process_frame(LLMContextFrame(context), FrameDirection.DOWNSTREAM)

        self.assertEqual(gate.pending_commits, 0)
        gate.push_frame.assert_not_awaited()

        controller.start_response()
        response_id = controller.active_response_id
        gate.register_response_marker(response_id)
        await gate.process_frame(
            RealtimeResponseCreateFrame(response_id=response_id),
            FrameDirection.DOWNSTREAM,
        )
        gate.push_frame.assert_awaited_once()
        released_frame, released_direction = gate.push_frame.await_args.args
        self.assertIsInstance(released_frame, RealtimeResponseContextFrame)
        self.assertEqual(released_frame.response_id, response_id)
        self.assertIs(released_frame.canonical_context, context)
        self.assertIsNot(released_frame.context, context)
        self.assertEqual(released_frame.context.get_messages(), context.get_messages())
        self.assertEqual(released_direction, FrameDirection.DOWNSTREAM)

    async def test_detected_turn_does_not_infer_when_automatic_response_is_disabled(self) -> None:
        controller = _controller()
        controller.apply_session_update(
            {
                "audio": {
                    "input": {
                        "turn_detection": {
                            "type": "server_vad",
                            "create_response": False,
                        }
                    }
                }
            }
        )
        gate = RealtimeManualResponseGate(controller=controller)
        gate.push_frame = AsyncMock()
        context = LLMContext([{"role": "user", "content": "detected transcript"}])
        gate.register_automatic_response_context()

        await gate.process_frame(LLMContextFrame(context), FrameDirection.DOWNSTREAM)

        gate.push_frame.assert_not_awaited()
        self.assertIs(gate._llm_context, context)

        controller.start_response()
        response_id = controller.active_response_id
        assert response_id is not None
        gate.register_response_marker(response_id)
        await gate.process_frame(
            RealtimeResponseCreateFrame(response_id=response_id),
            FrameDirection.DOWNSTREAM,
        )

        gate.push_frame.assert_awaited_once()
        released_frame, released_direction = gate.push_frame.await_args.args
        self.assertIsInstance(released_frame, RealtimeResponseContextFrame)
        self.assertEqual(released_frame.response_id, response_id)
        self.assertIs(released_frame.canonical_context, context)
        self.assertIsNot(released_frame.context, context)
        self.assertEqual(released_frame.context.get_messages(), context.get_messages())
        self.assertEqual(released_direction, FrameDirection.DOWNSTREAM)

    async def test_automatic_response_freezes_inference_and_public_response_defaults(self) -> None:
        controller = RealtimeSessionController(
            model="test-model",
            voice="test-voice",
            runtime_config={},
            capabilities=RealtimeSessionCapabilities(sequential_tool_calls=True),
        )
        controller.apply_session_update(
            {
                "output_modalities": ["text"],
                "max_output_tokens": 32,
                "parallel_tool_calls": True,
            }
        )
        gate = RealtimeManualResponseGate(controller=controller)
        gate.push_frame = AsyncMock()
        canonical = LLMContext([{"role": "user", "content": "automatic turn"}])
        gate.register_automatic_response_context()

        await gate.process_frame(LLMContextFrame(canonical), FrameDirection.DOWNSTREAM)

        self.assertEqual(gate.push_frame.await_count, 2)
        config = gate.push_frame.await_args_list[0].args[0]
        self.assertIsInstance(config, LLMConfigureOutputFrame)
        self.assertTrue(config.skip_tts)
        inference = gate.push_frame.await_args_list[1].args[0]
        self.assertIsInstance(inference, LLMContextFrame)
        self.assertIsInstance(inference.context, RealtimeResponseLLMContext)
        self.assertEqual(inference.context.max_output_tokens, 32)
        self.assertIs(inference.context.parallel_tool_calls, True)
        self.assertTrue(inference.context.run_owner_id)
        self.assertEqual(inference.context.activation_generation, controller.interruption_generation)
        self.assertIsNot(inference.context, canonical)

        controller.apply_session_update({"max_output_tokens": 64, "parallel_tool_calls": False})
        created = controller.ensure_response()[0]["response"]
        self.assertEqual(created["max_output_tokens"], 32)
        self.assertEqual(created["output_modalities"], ["text"])
        self.assertNotIn("parallel_tool_calls", created)
        self.assertEqual(controller.public_session()["max_output_tokens"], 64)
        self.assertIs(controller.public_session()["parallel_tool_calls"], False)

    async def test_server_tool_followup_without_vad_marker_uses_fresh_response_snapshot(self) -> None:
        controller = _controller(output_kind="text", tool_owner="server")
        controller.apply_session_update({"max_output_tokens": 48})
        gate = RealtimeManualResponseGate(controller=controller)
        gate.push_frame = AsyncMock()
        context = LLMContext(
            [
                {"role": "user", "content": "look it up"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_server",
                            "type": "function",
                            "function": {"name": "lookup", "arguments": "{}"},
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "call_server", "content": '{"ok":true}'},
            ]
        )

        await gate.process_frame(LLMContextFrame(context), FrameDirection.DOWNSTREAM)

        self.assertEqual(gate.push_frame.await_count, 2)
        self.assertIsInstance(gate.push_frame.await_args_list[0].args[0], LLMConfigureOutputFrame)
        followup = gate.push_frame.await_args_list[1].args[0]
        self.assertIsInstance(followup.context, RealtimeResponseLLMContext)
        self.assertEqual(followup.context.max_output_tokens, 48)
        created = controller.ensure_response()[0]["response"]
        self.assertEqual(created["max_output_tokens"], 48)
        self.assertEqual(created["output_modalities"], ["text"])

    async def test_internal_tool_continuation_uses_auto_without_mutating_forced_session_choice(self) -> None:
        choices = [
            ("required", "required"),
            (
                {"type": "function", "name": "lookup"},
                {"type": "function", "function": {"name": "lookup"}},
            ),
        ]
        for public_choice, pipeline_choice in choices:
            for deferred in (True, False):
                with self.subTest(tool_choice=public_choice, deferred=deferred):
                    controller = _controller(output_kind="text", tool_owner="server")
                    controller.apply_session_update({"tool_choice": copy.deepcopy(public_choice)})
                    gate = RealtimeManualResponseGate(controller=controller)
                    gate.push_frame = AsyncMock()

                    async def activate(
                        owner_id: str,
                        generation: int,
                        on_started=None,
                        _controller=controller,
                    ) -> str | None:
                        activated = _controller.start_pending_pipeline_response(
                            owner_id=owner_id,
                            expected_generation=generation,
                        )
                        response_id = activated[0] if activated is not None else None
                        if response_id is not None and on_started is not None:
                            await on_started(response_id)
                        return response_id

                    gate.set_pipeline_response_start_handler(activate)
                    canonical = LLMContext(
                        [{"role": "tool", "tool_call_id": "call_server", "content": '{"ok":true}'}],
                        tool_choice=copy.deepcopy(pipeline_choice),
                    )
                    prepared = (
                        await gate.prepare_deferred_service_response_snapshot(canonical)
                        if deferred
                        else await gate.prepare_service_response_snapshot(
                            canonical,
                            origin=RealtimeResponseOrigin.INTERNAL_TOOL_CONTINUATION,
                        )
                    )
                    self.assertIsNotNone(prepared)
                    continuation, _, activate_continuation, _ = prepared

                    self.assertEqual(continuation.tool_choice, "auto")
                    self.assertEqual(canonical.tool_choice, pipeline_choice)
                    self.assertEqual(controller.public_session()["tool_choice"], public_choice)
                    response_id = await activate_continuation()
                    self.assertIsNotNone(response_id)
                    controller.finish_response(status="completed")
                    gate.finish_response(response_id)

                    gate.register_automatic_response_context()
                    await gate.process_frame(LLMContextFrame(canonical), FrameDirection.DOWNSTREAM)
                    next_user_turn = gate.push_frame.await_args_list[-1].args[0].context

                    self.assertEqual(next_user_turn.tool_choice, pipeline_choice)
                    self.assertEqual(canonical.tool_choice, pipeline_choice)
                    self.assertEqual(controller.public_session()["tool_choice"], public_choice)

    async def test_unmarked_programmatic_context_preserves_forced_session_choice(self) -> None:
        choices = [
            ("required", "required"),
            (
                {"type": "function", "name": "lookup"},
                {"type": "function", "function": {"name": "lookup"}},
            ),
        ]
        for public_choice, pipeline_choice in choices:
            for wait_for_session_tools in (False, True):
                with self.subTest(tool_choice=public_choice, wait_for_session_tools=wait_for_session_tools):
                    controller = _controller(output_kind="text", tool_owner="server")
                    controller.apply_session_update({"tool_choice": copy.deepcopy(public_choice)})
                    gate = RealtimeManualResponseGate(controller=controller)
                    gate.push_frame = AsyncMock()
                    canonical = LLMContext(
                        [{"role": "assistant", "content": "scheduled service update"}],
                        tool_choice=copy.deepcopy(pipeline_choice),
                    )
                    if wait_for_session_tools:
                        await gate.reserve_session_tool_preparation("request_programmatic")

                    await gate.process_frame(LLMContextFrame(canonical), FrameDirection.DOWNSTREAM)
                    if wait_for_session_tools:
                        self.assertEqual(gate.push_frame.await_count, 0)
                        await gate.release_session_tool_preparation("request_programmatic")

                    inference = gate.push_frame.await_args_list[-1].args[0].context
                    self.assertIsInstance(inference, RealtimeResponseLLMContext)
                    self.assertEqual(inference.tool_choice, pipeline_choice)
                    self.assertEqual(canonical.tool_choice, pipeline_choice)
                    self.assertEqual(controller.public_session()["tool_choice"], public_choice)

    async def test_live_response_defaults_are_consumed_by_next_pipeline_run(self) -> None:
        voice_a = "voice-a"
        voice_b = "voice-b"
        recorder = _Recorder()
        controller = RealtimeSessionController(
            model="test-model",
            voice=voice_a,
            runtime_config={},
            instructions="prompt",
            capabilities=RealtimeSessionCapabilities(
                voices=frozenset({voice_a, voice_b}),
                sequential_tool_calls=True,
            ),
        )
        serializer = _serializer(recorder, controller)
        context = LLMContext([{"role": "system", "content": "prompt"}])
        serializer.bind_context(context, instructions_renderer=lambda text: [{"role": "system", "content": text}])
        gate = RealtimeManualResponseGate(controller=controller)
        gate.bind_tts_service(object())
        gate.push_frame = AsyncMock()
        serializer.set_response_gate(gate)

        frame = await serializer.deserialize(
            json.dumps(
                {
                    "type": "session.update",
                    "session": {
                        "max_output_tokens": 96,
                        "parallel_tool_calls": False,
                        "output_modalities": ["audio"],
                        "audio": {"output": {"voice": voice_b}},
                    },
                }
            )
        )
        self.assertIsNone(frame)
        self.assertEqual(recorder.types[-1], "session.updated")

        gate.register_automatic_response_context()
        await gate.process_frame(LLMContextFrame(context), FrameDirection.DOWNSTREAM)
        calls = gate.push_frame.await_args_list
        self.assertIsInstance(calls[0].args[0], LLMConfigureOutputFrame)
        self.assertFalse(calls[0].args[0].skip_tts)
        self.assertIsInstance(calls[1].args[0], TTSUpdateSettingsFrame)
        self.assertEqual(calls[1].args[0].delta.voice, voice_b)
        inference = calls[2].args[0].context
        self.assertEqual(inference.max_output_tokens, 96)
        self.assertIs(inference.parallel_tool_calls, False)
        created = controller.ensure_response()[0]["response"]
        self.assertEqual(created["max_output_tokens"], 96)
        self.assertEqual(created["output_modalities"], ["audio"])
        self.assertEqual(created["audio"]["output"]["voice"], voice_b)

    async def test_disabled_auto_response_does_not_block_explicit_server_tool_followup(self) -> None:
        controller = _controller()
        controller.apply_session_update(
            {
                "audio": {
                    "input": {
                        "turn_detection": {
                            "type": "server_vad",
                            "create_response": False,
                        }
                    }
                }
            }
        )
        gate = RealtimeManualResponseGate(controller=controller)
        gate.push_frame = AsyncMock()
        context = LLMContext([{"role": "user", "content": "look this up"}])
        gate.register_automatic_response_context()
        await gate.process_frame(LLMContextFrame(context), FrameDirection.DOWNSTREAM)
        gate.push_frame.assert_not_awaited()

        controller.start_response()
        response_a_id = controller.active_response_id
        assert response_a_id is not None
        gate.register_response_marker(response_a_id)
        await gate.process_frame(
            RealtimeResponseCreateFrame(response_id=response_a_id),
            FrameDirection.DOWNSTREAM,
        )
        gate.push_frame.assert_awaited_once()

        controller.finish_response(status="completed")
        gate.finish_response(response_a_id)
        gate.push_frame.reset_mock()
        context.add_messages(
            [
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "call_server",
                            "type": "function",
                            "function": {"name": "lookup", "arguments": "{}"},
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "call_server", "content": '{"ok":true}'},
            ]
        )
        tool_followup = LLMContextFrame(context)

        await gate.process_frame(tool_followup, FrameDirection.DOWNSTREAM)

        self.assertEqual(gate.push_frame.await_count, 2)
        output_config = gate.push_frame.await_args_list[0].args[0]
        self.assertIsInstance(output_config, LLMConfigureOutputFrame)
        followup = gate.push_frame.await_args_list[1].args[0]
        self.assertIsInstance(followup, LLMContextFrame)
        self.assertIsInstance(followup.context, RealtimeResponseLLMContext)
        self.assertIsNot(followup.context, context)
        self.assertEqual(followup.context.get_messages(), context.get_messages())
        self.assertTrue(controller.pipeline_response_pending)

    async def test_empty_auto_turn_marker_is_retired_at_user_stop(self) -> None:
        controller = _controller()
        controller.apply_session_update(
            {
                "audio": {
                    "input": {
                        "turn_detection": {
                            "type": "server_vad",
                            "create_response": False,
                        }
                    }
                }
            }
        )
        gate = RealtimeManualResponseGate(controller=controller)
        gate.push_frame = AsyncMock()
        gate.register_automatic_response_context()
        stopped = UserStoppedSpeakingFrame()
        await gate.process_frame(stopped, FrameDirection.DOWNSTREAM)
        unrelated = LLMContextFrame(LLMContext([{"role": "developer", "content": "continue"}]))

        await gate.process_frame(unrelated, FrameDirection.DOWNSTREAM)

        self.assertEqual(gate._pending_automatic_response_contexts, 0)
        gate.push_frame.assert_any_await(stopped, FrameDirection.DOWNSTREAM)
        self.assertIsInstance(gate.push_frame.await_args_list[-2].args[0], LLMConfigureOutputFrame)
        released = gate.push_frame.await_args_list[-1].args[0]
        self.assertIsInstance(released, LLMContextFrame)
        self.assertIsInstance(released.context, RealtimeResponseLLMContext)
        self.assertIsNot(released.context, unrelated.context)
        self.assertEqual(released.context.get_messages(), unrelated.context.get_messages())

    async def test_manual_commit_gate_survives_pipecat_queued_context_boundary(self) -> None:
        gate = RealtimeManualResponseGate(controller=_controller(manual=True))
        gate.push_frame = AsyncMock()
        gate.register_commit()
        queued_frames: list[LLMContextFrame] = []
        strategy = RealtimeManualTurnStopStrategy(response_gate=gate)
        context_frame = LLMContextFrame(LLMContext([{"role": "user", "content": "queued transcript"}]))

        async def queue_context_after_strategy_returns() -> None:
            queued_frames.append(context_frame)

        strategy.trigger_user_turn_stopped = AsyncMock(side_effect=queue_context_after_strategy_returns)
        await strategy.handle_user_turn_started()
        await strategy.process_frame(TranscriptionFrame(text="queued transcript", user_id="", timestamp=""))
        await strategy.process_frame(UserStoppedSpeakingFrame())
        self.assertEqual(queued_frames, [context_frame])

        # Pipecat's processor queue delivers this only after the strategy call
        # stack has returned. The durable commit count, not task-local scope,
        # must still identify and suppress it.
        await gate.process_frame(queued_frames.pop(), FrameDirection.DOWNSTREAM)

        self.assertEqual(gate.pending_commits, 0)
        gate.push_frame.assert_not_awaited()

    async def test_manual_response_create_before_transcript_waits_then_releases_owned_snapshot(self) -> None:
        controller = _controller(manual=True)
        gate = RealtimeManualResponseGate(controller=controller)
        gate.push_frame = AsyncMock()
        context_frame = LLMContextFrame(LLMContext([{"role": "user", "content": "late transcript"}]))
        gate.register_commit()
        controller.start_response()
        response_id = controller.active_response_id
        gate.register_response_marker(response_id)

        await gate.process_frame(
            RealtimeResponseCreateFrame(response_id=response_id),
            FrameDirection.DOWNSTREAM,
        )
        gate.push_frame.assert_not_awaited()

        await gate.process_frame(context_frame, FrameDirection.DOWNSTREAM)

        gate.push_frame.assert_awaited_once()
        released, direction = gate.push_frame.await_args.args
        self.assertIsInstance(released, RealtimeResponseContextFrame)
        self.assertEqual(released.response_id, response_id)
        self.assertIs(released.canonical_context, context_frame.context)
        self.assertIsNot(released.context, context_frame.context)
        self.assertEqual(released.context.get_messages(), context_frame.context.get_messages())
        self.assertEqual(direction, FrameDirection.DOWNSTREAM)
        self.assertEqual(gate.pending_commits, 0)

    async def test_manual_turn_during_response_defers_next_response_until_done_and_context(self) -> None:
        recorder = _Recorder()
        controller = _controller(manual=True, output_kind="text")
        serializer = _serializer(recorder, controller)
        context = LLMContext([{"role": "user", "content": "first turn"}])
        serializer.bind_context(context, instructions_renderer=lambda _instructions: [])
        gate = RealtimeManualResponseGate(controller=controller)
        gate.push_frame = AsyncMock()
        serializer.set_manual_input_handlers(commit_hook=AsyncMock(), response_gate=gate)

        response_a_frame = await serializer.deserialize(json.dumps({"type": "response.create"}))
        self.assertIsInstance(response_a_frame, RealtimeResponseCreateFrame)
        response_a_id = controller.active_response_id
        await gate.process_frame(response_a_frame, FrameDirection.DOWNSTREAM)
        gate.push_frame.reset_mock()

        client_pcm = b"\x01\x00" * 480
        serializer._resampler.complete_input_to_pipeline = AsyncMock(return_value=b"\x02\x00" * 320)
        await serializer.deserialize(
            json.dumps(
                {
                    "type": "input_audio_buffer.append",
                    "audio": base64.b64encode(client_pcm).decode("ascii"),
                }
            )
        )
        await serializer.deserialize(json.dumps({"type": "input_audio_buffer.commit"}))

        response_b_frame = await serializer.deserialize(json.dumps({"type": "response.create"}))
        self.assertIsInstance(response_b_frame, RealtimeDeferredResponseCreateFrame)
        self.assertEqual(recorder.types.count("response.created"), 1)
        self.assertFalse(gate.has_unclaimed_manual_commit)

        response_b_task = asyncio.create_task(gate.process_frame(response_b_frame, FrameDirection.DOWNSTREAM))
        await asyncio.sleep(0)
        self.assertFalse(response_b_task.done())

        await recorder.emit_batch(controller.finish_response(status="completed"))
        await asyncio.sleep(0)
        self.assertFalse(response_b_task.done())
        self.assertEqual(recorder.types.count("response.created"), 1)

        serializer.notify_response_done_published(response_a_id)
        await asyncio.wait_for(response_b_task, timeout=1)
        response_b_id = controller.active_response_id
        self.assertIsNotNone(response_b_id)
        self.assertNotEqual(response_b_id, response_a_id)
        self.assertEqual(recorder.types.count("response.created"), 2)
        self.assertEqual(gate._waiting_response_id, response_b_id)
        gate.push_frame.assert_not_awaited()

        context.add_message({"role": "user", "content": "committed during response A"})
        await gate.process_frame(LLMContextFrame(context), FrameDirection.DOWNSTREAM)

        self.assertEqual(gate.pending_commits, 0)
        self.assertIsNone(gate._waiting_response_id)
        released = gate.push_frame.await_args_list[-1].args[0]
        self.assertIsInstance(released, RealtimeResponseContextFrame)
        self.assertEqual(released.response_id, response_b_id)
        self.assertEqual(released.context.get_messages(), context.get_messages())

    async def test_cancel_then_manual_commit_preserves_following_deferred_generation(self) -> None:
        recorder = _Recorder()
        controller = _controller(manual=True, output_kind="text")
        serializer = _serializer(recorder, controller)
        context = LLMContext([{"role": "user", "content": "first turn"}])
        serializer.bind_context(context, instructions_renderer=lambda _instructions: [])
        gate = RealtimeManualResponseGate(controller=controller)
        gate.push_frame = AsyncMock()
        serializer.set_manual_input_handlers(commit_hook=AsyncMock(), response_gate=gate)

        response_a_frame = await serializer.deserialize(json.dumps({"type": "response.create"}))
        await gate.process_frame(response_a_frame, FrameDirection.DOWNSTREAM)
        response_a_id = controller.active_response_id
        generation_before_cancel = controller.interruption_generation
        cancel_frame = await serializer.deserialize(
            json.dumps({"type": "response.cancel", "response_id": response_a_id})
        )
        self.assertIsInstance(cancel_frame, InterruptionFrame)
        self.assertEqual(controller.interruption_generation, generation_before_cancel + 1)

        client_pcm = b"\x01\x00" * 480
        serializer._resampler.complete_input_to_pipeline = AsyncMock(return_value=b"\x02\x00" * 320)
        await serializer.deserialize(
            json.dumps(
                {
                    "type": "input_audio_buffer.append",
                    "audio": base64.b64encode(client_pcm).decode("ascii"),
                }
            )
        )
        await serializer.deserialize(json.dumps({"type": "input_audio_buffer.commit"}))
        response_b_frame = await serializer.deserialize(json.dumps({"type": "response.create"}))
        self.assertIsInstance(response_b_frame, RealtimeDeferredResponseCreateFrame)
        self.assertEqual(response_b_frame.activation_generation, controller.interruption_generation)

        delayed_generation, first_observation = controller.observe_interruption(cancel_frame.id)
        self.assertFalse(first_observation)
        self.assertEqual(delayed_generation, response_b_frame.activation_generation)
        gate._cancel_deferred_response(older_than_generation=delayed_generation)
        self.assertEqual(gate._pending_deferred_response_id, response_b_frame.request_id)
        response_b_task = asyncio.create_task(gate.process_frame(response_b_frame, FrameDirection.DOWNSTREAM))
        await asyncio.sleep(0)
        self.assertFalse(response_b_task.done())

        await recorder.emit_batch(controller.finish_response(status="cancelled", reason="client_cancelled"))
        serializer.notify_response_done_published(response_a_id)
        await asyncio.wait_for(response_b_task, timeout=1)

        self.assertIsNotNone(controller.active_response_id)
        self.assertNotEqual(controller.active_response_id, response_a_id)
        self.assertNotIn("response_cancelled", [event.get("error", {}).get("code") for event in recorder.events])

    async def test_manual_empty_final_transcript_releases_gate_without_context_or_inference(self) -> None:
        controller = _controller(manual=True)
        gate = RealtimeManualResponseGate(controller=controller)
        gate.push_frame = AsyncMock()
        gate.register_commit()
        controller.start_response()
        response_id = controller.active_response_id
        gate.register_response_marker(response_id)
        await gate.process_frame(
            RealtimeResponseCreateFrame(response_id=response_id),
            FrameDirection.DOWNSTREAM,
        )
        strategy = RealtimeManualTurnStopStrategy(response_gate=gate)
        strategy.trigger_user_turn_stopped = AsyncMock()
        await strategy.handle_user_turn_started()

        await strategy.process_frame(TranscriptionFrame(text="", user_id="", timestamp=""))
        await strategy.process_frame(UserStoppedSpeakingFrame())

        self.assertEqual(gate.pending_commits, 0)
        self.assertIsNone(gate._waiting_response_id)
        gate.push_frame.assert_not_awaited()
        strategy.trigger_user_turn_stopped.assert_awaited_once()

    async def test_response_cancel_while_idle_is_a_protocol_error(self) -> None:
        recorder = _Recorder()
        serializer = _serializer(recorder)
        frame = await serializer.deserialize(json.dumps({"type": "response.cancel"}))
        self.assertIsNone(frame)
        self.assertEqual(recorder.events[0]["error"]["code"], "response_not_found")

    async def test_response_create_emits_created_and_returns_llm_run(self) -> None:
        recorder = _Recorder()
        serializer = _serializer(recorder)
        frame = await serializer.deserialize(json.dumps({"type": "response.create", "response": {}}))
        self.assertIsInstance(frame, LLMRunFrame)
        self.assertEqual(recorder.types, ["response.created"])

    async def test_cancelled_response_created_publication_finishes_as_failed(self) -> None:
        controller = _controller(output_kind="text")
        serializer = RealtimeFrameSerializer(controller=controller)
        attempted_batches: list[list[dict[str, Any]]] = []

        async def emit_batch(events: list[dict[str, Any]]) -> None:
            attempted_batches.append(events)
            if len(attempted_batches) == 1:
                raise asyncio.CancelledError

        serializer.set_emit(AsyncMock(), emit_batch)

        with self.assertRaises(asyncio.CancelledError):
            await serializer.deserialize(json.dumps({"type": "response.create"}))

        self.assertIsNone(controller.active_response_id)
        self.assertEqual([batch[0]["type"] for batch in attempted_batches], ["response.created", "response.done"])
        done = attempted_batches[-1][-1]
        self.assertEqual(done["response"]["status"], "failed")
        self.assertEqual(
            done["response"]["status_details"]["error"]["code"],
            "response_start_cancelled",
        )

    async def test_connection_close_during_response_created_abandons_without_terminal_wire_event(self) -> None:
        controller = _controller(output_kind="text")
        serializer = RealtimeFrameSerializer(controller=controller)
        attempted_types: list[str] = []

        async def close_during_emit(events: list[dict[str, Any]]) -> None:
            attempted_types.extend(event["type"] for event in events)
            serializer.notify_connection_closed()

        serializer.set_emit(AsyncMock(), close_during_emit)

        frame = await serializer.deserialize(json.dumps({"type": "response.create"}))

        self.assertIsNone(frame)
        self.assertTrue(serializer.connection_closed)
        self.assertIsNone(controller.active_response_id)
        self.assertEqual(attempted_types, ["response.created"])

    async def test_duplicate_response_create_is_rejected(self) -> None:
        recorder = _Recorder()
        serializer = _serializer(recorder)
        await serializer.deserialize(json.dumps({"type": "response.create"}))
        frame = await serializer.deserialize(json.dumps({"type": "response.create"}))
        self.assertIsNone(frame)
        self.assertEqual(recorder.events[-1]["error"]["code"], "response_in_progress")

    async def test_response_inference_overrides_are_isolated_without_noncanonical_echoes(self) -> None:
        recorder = _Recorder()
        controller = RealtimeSessionController(
            model="test-model",
            voice="test-voice",
            runtime_config={"asr_server": "localhost:50051"},
            input_transcription_model="test-asr",
            capabilities=RealtimeSessionCapabilities(sequential_tool_calls=True),
        )
        controller.apply_session_update(
            {
                "instructions": "session prompt",
                "max_output_tokens": 256,
                "parallel_tool_calls": False,
            }
        )
        serializer = _serializer(recorder, controller)

        def render(instructions: str) -> list[dict[str, str]]:
            return [
                {"role": "system", "content": "trusted runtime control"},
                {"role": "user", "content": instructions},
            ]

        canonical_messages = [
            *render("session prompt"),
            {"role": "user", "content": "question"},
        ]
        context = LLMContext(canonical_messages)
        gate = RealtimeManualResponseGate(controller=controller)
        gate.push_frame = AsyncMock()
        serializer.bind_context(context, instructions_renderer=render)
        serializer.set_response_gate(gate)
        frame = await serializer.deserialize(
            json.dumps(
                {
                    "type": "response.create",
                    "response": {
                        "instructions": "one response",
                        "max_output_tokens": 32,
                        "parallel_tool_calls": True,
                    },
                }
            )
        )
        self.assertIsInstance(frame, RealtimeResponseCreateFrame)
        self.assertNotIn("instructions", recorder.events[0]["response"])
        self.assertEqual(recorder.events[0]["response"]["max_output_tokens"], 32)
        self.assertNotIn("parallel_tool_calls", recorder.events[0]["response"])
        self.assertEqual(controller.public_session()["instructions"], "session prompt")
        self.assertEqual(controller.public_session()["max_output_tokens"], 256)
        self.assertIs(controller.public_session()["parallel_tool_calls"], False)
        self.assertEqual(context.get_messages(), canonical_messages)

        await gate.process_frame(frame, FrameDirection.DOWNSTREAM)
        response_frame = gate.push_frame.await_args.args[0]
        self.assertIsInstance(response_frame, RealtimeResponseContextFrame)
        self.assertIsInstance(response_frame.context, RealtimeResponseLLMContext)
        self.assertEqual(response_frame.context.max_output_tokens, 32)
        self.assertIs(response_frame.context.parallel_tool_calls, True)
        self.assertEqual(
            response_frame.context.get_messages(),
            [*render("one response"), {"role": "user", "content": "question"}],
        )
        self.assertEqual(context.get_messages(), canonical_messages)

        done = controller.finish_response(status="completed")[0]
        self.assertNotIn("instructions", done["response"])
        self.assertEqual(done["response"]["max_output_tokens"], 32)
        self.assertNotIn("parallel_tool_calls", done["response"])
        inherited = controller.start_response()[0]["response"]
        self.assertNotIn("instructions", inherited)
        self.assertEqual(inherited["max_output_tokens"], 256)
        self.assertNotIn("parallel_tool_calls", inherited)

    async def test_custom_response_input_and_default_conversation_are_isolated(self) -> None:
        recorder = _Recorder()
        controller = _controller(output_kind="text")
        controller.apply_session_update({"instructions": "trusted prompt"})
        serializer = _serializer(recorder, controller)

        def render(instructions: str) -> list[dict[str, str]]:
            return [{"role": "system", "content": instructions}]

        canonical_messages = [*render("trusted prompt"), {"role": "user", "content": "default history"}]
        context = LLMContext(canonical_messages)
        serializer.bind_context(context, instructions_renderer=render)
        gate = RealtimeManualResponseGate(controller=controller)
        gate.push_frame = AsyncMock()
        serializer.set_response_gate(gate)
        item_events = controller.create_conversation_item(
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "referenced context"}],
            }
        )
        item_id = item_events[0]["item"]["id"]

        frame = await serializer.deserialize(
            json.dumps(
                {
                    "type": "response.create",
                    "response": {
                        "conversation": "auto",
                        "input": [
                            {"type": "item_reference", "id": item_id},
                            {
                                "type": "message",
                                "role": "user",
                                "content": [{"type": "input_text", "text": "inline context"}],
                            },
                        ],
                        "instructions": "one-response prompt",
                        "output_modalities": ["text"],
                    },
                }
            )
        )

        self.assertIsInstance(frame, RealtimeResponseCreateFrame)
        created = recorder.events[-1]["response"]
        self.assertEqual(created["conversation_id"], controller.conversation.id)
        self.assertEqual(created["output_modalities"], ["text"])
        self.assertNotIn("audio", created)
        await gate.process_frame(frame, FrameDirection.DOWNSTREAM)
        calls = gate.push_frame.await_args_list
        self.assertIsInstance(calls[0].args[0], LLMConfigureOutputFrame)
        self.assertTrue(calls[0].args[0].skip_tts)
        response_frame = calls[-1].args[0]
        self.assertIsInstance(response_frame, RealtimeResponseContextFrame)
        self.assertEqual(
            response_frame.context.get_messages(),
            [
                *render("one-response prompt"),
                {"role": "user", "content": "referenced context"},
                {"role": "user", "content": "inline context"},
            ],
        )
        self.assertEqual(context.get_messages(), canonical_messages)

    async def test_response_conversation_none_is_rejected_without_starting_lifecycle(self) -> None:
        recorder = _Recorder()
        controller = _controller(output_kind="text")
        serializer = _serializer(recorder, controller)

        frame = await serializer.deserialize(
            json.dumps(
                {
                    "event_id": "oob",
                    "type": "response.create",
                    "response": {"conversation": "none"},
                }
            )
        )

        self.assertIsNone(frame)
        self.assertFalse(controller.response_in_progress)
        self.assertEqual(recorder.events[-1]["error"]["code"], "unsupported_capability")
        self.assertEqual(recorder.events[-1]["error"]["param"], "response.conversation")

    async def test_response_audio_format_voice_and_next_response_restore_exact_snapshot(self) -> None:
        voice_a = "voice-a"
        voice_b = "voice-b"
        recorder = _Recorder()
        controller = RealtimeSessionController(
            model="test-model",
            voice=voice_a,
            runtime_config={},
            instructions="prompt",
            capabilities=RealtimeSessionCapabilities(
                output_formats=frozenset(
                    {
                        AudioFormatCapability("audio/pcm", 24000),
                        AudioFormatCapability("audio/pcmu"),
                    }
                ),
                voices=frozenset({voice_a, voice_b}),
            ),
        )
        serializer = _serializer(recorder, controller)
        context = LLMContext([{"role": "system", "content": "prompt"}])
        serializer.bind_context(context, instructions_renderer=lambda text: [{"role": "system", "content": text}])
        gate = RealtimeManualResponseGate(controller=controller)
        gate.bind_tts_service(object())
        gate.push_frame = AsyncMock()
        serializer.set_response_gate(gate)

        first = await serializer.deserialize(
            json.dumps(
                {
                    "type": "response.create",
                    "response": {
                        "output_modalities": ["audio"],
                        "audio": {
                            "output": {
                                "format": {"type": "audio/pcmu"},
                                "voice": voice_b,
                            }
                        },
                    },
                }
            )
        )
        self.assertIsInstance(first, RealtimeResponseCreateFrame)
        self.assertEqual(serializer._client_out_format, "audio/pcmu")
        self.assertEqual(serializer._client_out_rate, 8000)
        self.assertEqual(
            recorder.events[-1]["response"]["audio"],
            {"output": {"format": {"type": "audio/pcmu"}, "voice": voice_b}},
        )
        await gate.process_frame(first, FrameDirection.DOWNSTREAM)
        first_calls = gate.push_frame.await_args_list
        self.assertIsInstance(first_calls[0].args[0], LLMConfigureOutputFrame)
        self.assertFalse(first_calls[0].args[0].skip_tts)
        self.assertIsInstance(first_calls[1].args[0], TTSUpdateSettingsFrame)
        self.assertEqual(first_calls[1].args[0].delta.voice, voice_b)

        with patch.object(serializer._resampler, "from_pipeline", AsyncMock(return_value=b"\xff" * 80)) as resample:
            await serializer.serialize(OutputAudioRawFrame(b"\x00" * 480, 24000, 1))
        resample.assert_awaited_once_with(
            b"\x00" * 480,
            8000,
            pipeline_rate=serializer._pipeline_out_rate,
            format_type="audio/pcmu",
        )

        first_response_id = controller.active_response_id
        controller.finish_response(status="completed")
        serializer.notify_response_done_published(first_response_id)
        self.assertEqual(serializer._client_out_format, "audio/pcm")
        self.assertEqual(serializer._client_out_rate, 24000)
        gate.push_frame.reset_mock()

        rejected = await serializer.deserialize(json.dumps({"type": "response.create"}))
        self.assertIsNone(rejected)
        self.assertEqual(recorder.events[-1]["error"]["code"], "immutable_field")
        self.assertEqual(recorder.events[-1]["error"]["param"], "response.audio.output.voice")
        controller.apply_session_update({"audio": {"output": {"voice": voice_b}}})

        second = await serializer.deserialize(json.dumps({"type": "response.create"}))
        self.assertIsInstance(second, RealtimeResponseCreateFrame)
        self.assertEqual(serializer._client_out_format, "audio/pcm")
        self.assertEqual(serializer._client_out_rate, 24000)
        await gate.process_frame(second, FrameDirection.DOWNSTREAM)
        second_calls = gate.push_frame.await_args_list
        self.assertIsInstance(second_calls[1].args[0], TTSUpdateSettingsFrame)
        self.assertEqual(second_calls[1].args[0].delta.voice, voice_b)

    async def test_live_output_format_update_waits_for_next_response_snapshot(self) -> None:
        recorder = _Recorder()
        controller = RealtimeSessionController(
            model="test-model",
            voice="voice-a",
            runtime_config={},
            instructions="prompt",
            capabilities=RealtimeSessionCapabilities(
                output_formats=frozenset(
                    {
                        AudioFormatCapability("audio/pcm", 24000),
                        AudioFormatCapability("audio/pcmu"),
                    }
                ),
            ),
        )
        serializer = _serializer(recorder, controller)
        context = LLMContext([{"role": "system", "content": "prompt"}])
        serializer.bind_context(context, instructions_renderer=lambda text: [{"role": "system", "content": text}])
        gate = RealtimeManualResponseGate(controller=controller)
        gate.bind_tts_service(object())
        gate.push_frame = AsyncMock()
        serializer.set_response_gate(gate)

        first = await serializer.deserialize(json.dumps({"type": "response.create"}))
        await gate.process_frame(first, FrameDirection.DOWNSTREAM)
        self.assertEqual(serializer._client_out_format, "audio/pcm")
        active_response_id = controller.active_response_id

        update = await serializer.deserialize(
            json.dumps(
                {
                    "type": "session.update",
                    "session": {"audio": {"output": {"format": {"type": "audio/pcmu"}}}},
                }
            )
        )
        self.assertIsNone(update)
        self.assertEqual(controller.public_session()["audio"]["output"]["format"], {"type": "audio/pcmu"})
        self.assertEqual(serializer._client_out_format, "audio/pcm")

        controller.finish_response(status="completed")
        gate.finish_response(active_response_id)
        second = await serializer.deserialize(json.dumps({"type": "response.create"}))
        self.assertIsInstance(second, RealtimeResponseCreateFrame)
        self.assertEqual(serializer._client_out_format, "audio/pcmu")
        self.assertEqual(serializer._client_out_rate, 8000)

    async def test_pipeline_response_codec_snapshot_survives_concurrent_session_update(self) -> None:
        recorder = _Recorder()
        controller = RealtimeSessionController(
            model="test-model",
            voice="voice-a",
            runtime_config={},
            capabilities=RealtimeSessionCapabilities(
                output_formats=frozenset(
                    {
                        AudioFormatCapability("audio/pcm", 24000),
                        AudioFormatCapability("audio/pcmu"),
                    }
                ),
            ),
        )
        serializer = _serializer(recorder, controller)
        gate = RealtimeManualResponseGate(controller=controller)
        gate.bind_tts_service(object())
        gate.push_frame = AsyncMock()
        serializer.set_response_gate(gate)
        context = LLMContext([{"role": "user", "content": "automatic response"}])
        gate.register_automatic_response_context()

        await gate.process_frame(LLMContextFrame(context), FrameDirection.DOWNSTREAM)
        self.assertTrue(controller.pipeline_response_pending)
        self.assertEqual(serializer._client_out_format, "audio/pcm")

        await serializer.deserialize(
            json.dumps(
                {
                    "type": "session.update",
                    "session": {"audio": {"output": {"format": {"type": "audio/pcmu"}}}},
                }
            )
        )
        self.assertEqual(controller.public_session()["audio"]["output"]["format"], {"type": "audio/pcmu"})
        self.assertEqual(serializer._client_out_format, "audio/pcm")

        created = controller.ensure_response()[0]["response"]
        self.assertEqual(created["audio"]["output"]["format"], {"type": "audio/pcm", "rate": 24000})
        response_id = controller.active_response_id
        controller.finish_response(status="completed")
        serializer.notify_response_done_published(response_id)
        self.assertEqual(serializer._client_out_format, "audio/pcmu")
        self.assertEqual(serializer._client_out_rate, 8000)

    async def test_fused_service_snapshot_drives_context_output_and_created_resource(self) -> None:
        voice_a = "voice-a"
        voice_b = "voice-b"
        recorder = _Recorder()
        controller = RealtimeSessionController(
            model="test-model",
            voice=voice_a,
            runtime_config={},
            capabilities=RealtimeSessionCapabilities(
                output_formats=frozenset(
                    {
                        AudioFormatCapability("audio/pcm", 24000),
                        AudioFormatCapability("audio/pcmu"),
                    }
                ),
                voices=frozenset({voice_a, voice_b}),
                sequential_tool_calls=True,
            ),
        )
        controller.apply_session_update(
            {
                "max_output_tokens": 72,
                "parallel_tool_calls": False,
                "audio": {
                    "output": {
                        "format": {"type": "audio/pcmu"},
                        "voice": voice_b,
                    }
                },
            }
        )
        serializer = _serializer(recorder, controller)
        gate = RealtimeManualResponseGate(controller=controller)
        tts = object()
        gate.bind_tts_service(tts)
        serializer.set_response_gate(gate)
        canonical = LLMContext([{"role": "user", "content": "fused audio turn"}])

        prepared = await gate.prepare_service_response_snapshot(canonical)
        self.assertIsNotNone(prepared)
        snapshot, setup, _, abort = prepared

        self.assertIsInstance(snapshot, RealtimeResponseLLMContext)
        self.assertIsNot(snapshot, canonical)
        self.assertEqual(snapshot.get_messages(), canonical.get_messages())
        self.assertEqual(snapshot.max_output_tokens, 72)
        self.assertIs(snapshot.parallel_tool_calls, False)
        self.assertEqual([type(frame) for frame in setup], [LLMConfigureOutputFrame, TTSUpdateSettingsFrame])
        self.assertFalse(setup[0].skip_tts)
        self.assertIs(setup[1].service, tts)
        self.assertEqual(setup[1].delta.voice, voice_b)
        self.assertEqual(serializer._client_out_format, "audio/pcmu")
        self.assertEqual(serializer._client_out_rate, 8000)

        # A concurrent live update is a default for the next response. The
        # observer must consume exactly the already-frozen service snapshot.
        controller.apply_session_update(
            {
                "max_output_tokens": 99,
                "parallel_tool_calls": True,
                "audio": {"output": {"format": {"type": "audio/pcm", "rate": 24000}}},
            }
        )
        created = controller.ensure_response()[0]["response"]
        self.assertEqual(created["max_output_tokens"], 72)
        self.assertEqual(created["output_modalities"], ["audio"])
        self.assertEqual(
            created["audio"]["output"],
            {"format": {"type": "audio/pcmu"}, "voice": voice_b},
        )
        self.assertEqual(serializer._client_out_format, "audio/pcmu")
        abort()
        self.assertTrue(controller.response_in_progress)

        response_id = controller.active_response_id
        controller.finish_response(status="completed")
        serializer.notify_response_done_published(response_id)
        self.assertEqual(serializer._client_out_format, "audio/pcm")
        self.assertEqual(serializer._client_out_rate, 24000)

    async def test_aborted_fused_service_snapshot_restores_current_session_codec(self) -> None:
        recorder = _Recorder()
        controller = RealtimeSessionController(
            model="test-model",
            voice="voice-a",
            runtime_config={},
            capabilities=RealtimeSessionCapabilities(
                output_formats=frozenset(
                    {
                        AudioFormatCapability("audio/pcm", 24000),
                        AudioFormatCapability("audio/pcmu"),
                    }
                ),
            ),
        )
        controller.apply_session_update({"audio": {"output": {"format": {"type": "audio/pcmu"}}}})
        serializer = _serializer(recorder, controller)
        gate = RealtimeManualResponseGate(controller=controller)
        gate.bind_tts_service(object())
        serializer.set_response_gate(gate)

        prepared = await gate.prepare_service_response_snapshot(LLMContext([]))
        self.assertIsNotNone(prepared)
        _, _, _, abort = prepared
        self.assertTrue(controller.pipeline_response_pending)
        self.assertEqual(serializer._client_out_format, "audio/pcmu")
        controller.apply_session_update({"audio": {"output": {"format": {"type": "audio/pcm", "rate": 24000}}}})

        abort()

        self.assertFalse(controller.pipeline_response_pending)
        self.assertFalse(controller.response_in_progress)
        self.assertEqual(serializer._client_out_format, "audio/pcm")
        self.assertEqual(serializer._client_out_rate, 24000)

    async def test_deferred_service_result_waits_for_published_done_and_gets_a_distinct_response(self) -> None:
        controller = RealtimeSessionController(
            model="test-model",
            voice="voice-a",
            runtime_config={},
            capabilities=RealtimeSessionCapabilities(voices=frozenset({"voice-a"})),
        )
        controller.apply_session_update({"output_modalities": ["text"]})
        gate = RealtimeManualResponseGate(controller=controller)

        async def activate(owner_id: str, generation: int, on_started=None) -> str | None:
            activated = controller.start_pending_pipeline_response(
                owner_id=owner_id,
                expected_generation=generation,
            )
            response_id = activated[0] if activated is not None else None
            if response_id is not None and on_started is not None:
                await on_started(response_id)
            return response_id

        gate.set_pipeline_response_start_handler(activate)
        foreground = await gate.prepare_service_response_snapshot(LLMContext([]))
        self.assertIsNotNone(foreground)
        _, _, activate_foreground, _ = foreground
        foreground_id = await activate_foreground()
        self.assertIsNotNone(foreground_id)

        deferred_task = asyncio.create_task(
            gate.prepare_deferred_service_response_snapshot(LLMContext([{"role": "user", "content": "newest context"}]))
        )
        await asyncio.sleep(0)
        self.assertFalse(deferred_task.done())

        # The controller becomes terminal before response.done is necessarily
        # on the wire. A separate gate wakeup must not let the background result
        # reuse that response or start before the publication barrier.
        controller.finish_response(status="completed")
        reservation = gate.reserve_service_audio_response()
        gate.release_service_audio_response(reservation)
        await asyncio.sleep(0)
        self.assertFalse(deferred_task.done())

        gate.finish_response(foreground_id)
        deferred = await asyncio.wait_for(deferred_task, timeout=1.0)
        self.assertIsNotNone(deferred)
        snapshot, _, activate_deferred, _ = deferred
        started_ids: list[str] = []

        async def record_started(response_id: str) -> None:
            started_ids.append(response_id)

        deferred_id = await activate_deferred(record_started)

        self.assertIsInstance(snapshot, RealtimeResponseLLMContext)
        self.assertNotEqual(deferred_id, foreground_id)
        self.assertEqual(deferred_id, controller.active_response_id)
        self.assertEqual(started_ids, [deferred_id])
        controller.finish_response(status="completed")
        gate.finish_response(deferred_id)

    async def test_waiting_deferred_service_result_uses_generation_at_its_own_claim(self) -> None:
        controller = RealtimeSessionController(
            model="test-model",
            voice="voice-a",
            runtime_config={},
            capabilities=RealtimeSessionCapabilities(voices=frozenset({"voice-a"})),
        )
        controller.apply_session_update({"output_modalities": ["text"]})
        gate = RealtimeManualResponseGate(controller=controller)

        async def activate(owner_id: str, generation: int, on_started=None) -> str | None:
            activated = controller.start_pending_pipeline_response(
                owner_id=owner_id,
                expected_generation=generation,
            )
            response_id = activated[0] if activated is not None else None
            if response_id is not None and on_started is not None:
                await on_started(response_id)
            return response_id

        gate.set_pipeline_response_start_handler(activate)
        foreground = await gate.prepare_service_response_snapshot(LLMContext([]))
        self.assertIsNotNone(foreground)
        _, _, activate_foreground, _ = foreground
        foreground_id = await activate_foreground()
        original_generation = controller.interruption_generation

        deferred_task = asyncio.create_task(gate.prepare_deferred_service_response_snapshot(LLMContext([])))
        await asyncio.sleep(0)
        self.assertFalse(deferred_task.done())

        interruption = InterruptionFrame()
        interruption_generation, _ = controller.observe_interruption(interruption.id)
        gate._abort_pending_pipeline_response(older_than_generation=interruption_generation)
        gate._response_slot_changed.set()
        self.assertGreater(controller.interruption_generation, original_generation)
        controller.finish_response(status="cancelled", reason="turn_detected")
        gate.finish_response(foreground_id)

        deferred = await asyncio.wait_for(deferred_task, timeout=1.0)
        self.assertIsNotNone(deferred)
        snapshot, _, activate_deferred, _ = deferred
        self.assertEqual(snapshot.activation_generation, controller.interruption_generation)
        self.assertIsNotNone(await activate_deferred())

    async def test_fused_audio_claim_preempts_response_after_async_preparation(self) -> None:
        controller = RealtimeSessionController(
            model="test-model",
            voice="voice-a",
            runtime_config={},
            capabilities=RealtimeSessionCapabilities(voices=frozenset({"voice-a"})),
        )
        controller.apply_session_update({"output_modalities": ["text"]})
        gate = RealtimeManualResponseGate(controller=controller)

        await gate.reserve_response_preparation("request_explicit")
        audio_reservation = gate.reserve_service_audio_response()
        snapshot_task = asyncio.create_task(gate.prepare_service_response_snapshot(LLMContext([]), audio_reservation))
        await asyncio.sleep(0)
        self.assertFalse(snapshot_task.done())

        with self.assertRaises(RealtimeProtocolError) as raised:
            gate.validate_response_preparation("request_explicit")
        self.assertEqual(raised.exception.code, "response_in_progress")
        gate.release_response_preparation("request_explicit")

        prepared = await asyncio.wait_for(snapshot_task, timeout=1.0)
        self.assertIsNotNone(prepared)
        _, _, _, abort = prepared
        self.assertTrue(controller.pipeline_response_pending)
        abort()
        self.assertFalse(controller.pipeline_response_pending)

    async def test_fused_snapshot_waits_for_session_update_and_uses_new_defaults(self) -> None:
        controller = RealtimeSessionController(
            model="test-model",
            voice="voice-a",
            runtime_config={},
            capabilities=RealtimeSessionCapabilities(voices=frozenset({"voice-a"})),
        )
        controller.apply_session_update({"output_modalities": ["text"], "max_output_tokens": 40})
        gate = RealtimeManualResponseGate(controller=controller)

        await gate.reserve_session_tool_preparation("request_update")
        audio_reservation = gate.reserve_service_audio_response()
        snapshot_task = asyncio.create_task(gate.prepare_service_response_snapshot(LLMContext([]), audio_reservation))
        await asyncio.sleep(0)
        self.assertFalse(snapshot_task.done())

        controller.apply_session_update({"max_output_tokens": 81})
        await gate.release_session_tool_preparation("request_update")
        prepared = await asyncio.wait_for(snapshot_task, timeout=1.0)
        self.assertIsNotNone(prepared)
        snapshot, _, _, abort = prepared
        self.assertEqual(snapshot.max_output_tokens, 81)
        abort()

    async def test_item_create_rejects_unknown_input_text_field_before_context_append(self) -> None:
        recorder = _Recorder()
        serializer = _serializer(recorder)

        frame = await serializer.deserialize(
            json.dumps(
                {
                    "type": "conversation.item.create",
                    "event_id": "bad_text_part",
                    "item": {
                        "type": "message",
                        "role": "user",
                        "content": [{"type": "input_text", "text": "Hello bot", "unexpected": True}],
                    },
                }
            )
        )

        self.assertIsNone(frame)
        self.assertEqual(recorder.types, ["error"])
        self.assertEqual(recorder.events[0]["error"]["code"], "unknown_parameter")
        self.assertEqual(recorder.events[0]["error"]["param"], "item.content[0].unexpected")
        self.assertEqual(recorder.events[0]["error"]["event_id"], "bad_text_part")

    async def test_function_call_output_rejects_unknown_call_id(self) -> None:
        recorder = _Recorder()
        serializer = _serializer(recorder)
        frame = await serializer.deserialize(
            json.dumps(
                {
                    "type": "conversation.item.create",
                    "item": {
                        "type": "function_call_output",
                        "call_id": "missing",
                        "output": "{}",
                    },
                }
            )
        )
        self.assertIsNone(frame)
        self.assertEqual(recorder.events[0]["error"]["code"], "call_not_found")

    async def test_function_call_output_validates_complete_item_before_broker_wakeup(self) -> None:
        invalid_items = (
            ({"id": 7}, "invalid_type", "item.id"),
            ({"id": ""}, "invalid_value", "item.id"),
            ({"object": 7}, "invalid_type", "item.object"),
            ({"object": "wrong.item"}, "invalid_value", "item.object"),
            ({"status": []}, "invalid_type", "item.status"),
            ({"status": "queued"}, "invalid_value", "item.status"),
            ({"unexpected": True}, "unknown_parameter", "item.unexpected"),
        )

        for item_patch, code, param in invalid_items:
            with self.subTest(item_patch=item_patch):
                recorder = _Recorder()
                serializer, _, broker = _serializer_with_active_client_tool(recorder)
                item = {
                    "type": "function_call_output",
                    "call_id": "call_client",
                    "output": "{}",
                    **item_patch,
                }

                frame = await serializer.deserialize(
                    json.dumps(
                        {
                            "type": "conversation.item.create",
                            "event_id": "client_output_invalid",
                            "item": item,
                        }
                    )
                )

                self.assertIsNone(frame)
                self.assertEqual(recorder.types, ["error"])
                self.assertEqual(recorder.events[0]["error"]["code"], code)
                self.assertEqual(recorder.events[0]["error"]["param"], param)
                self.assertEqual(recorder.events[0]["error"]["event_id"], "client_output_invalid")
                broker.stage_output.assert_not_awaited()
                broker.release_output.assert_not_awaited()
                broker.wait_context_applied.assert_not_awaited()

    async def test_valid_function_call_output_preserves_supplied_item_id(self) -> None:
        recorder = _Recorder()
        serializer, controller, broker = _serializer_with_active_client_tool(recorder)
        gate = _install_client_tool_gate(serializer, controller, LLMContext([]))

        frame = await serializer.deserialize(
            json.dumps(
                {
                    "type": "conversation.item.create",
                    "item": {
                        "id": "item_client_output",
                        "object": "realtime.item",
                        "type": "function_call_output",
                        "status": "completed",
                        "call_id": "call_client",
                        "output": '{"ok":true}',
                    },
                }
            )
        )

        self.assertIsInstance(frame, RealtimeClientToolOutputFrame)
        self.assertEqual(recorder.events, [])
        await gate.process_frame(frame, FrameDirection.DOWNSTREAM)
        self.assertEqual(recorder.types, ["conversation.item.added", "conversation.item.done"])
        self.assertEqual(recorder.events[-1]["item"]["id"], "item_client_output")
        self.assertEqual(controller.tool_call("call_client").output_item_id, "item_client_output")
        broker.stage_output.assert_awaited_once_with(
            call_id="call_client",
            name="lookup",
            output='{"ok":true}',
        )
        broker.release_output.assert_awaited_once_with(
            call_id="call_client",
            name="lookup",
        )
        broker.wait_context_applied.assert_awaited_once_with(
            "call_client",
            timeout=10,
        )

    async def test_fast_official_client_output_and_response_create_wait_for_response_a_done(self) -> None:
        recorder = _Recorder()
        controller = _controller(tool_owner="client")
        function_events = controller.start_function_call(
            call_id="call_client",
            name="lookup",
            arguments={"key": "value"},
        )
        self.assertEqual(function_events[-1]["type"], "response.output_item.done")
        response_a_id = controller.tool_call("call_client").response_id
        serializer = _serializer(recorder, controller)
        context = _client_tool_context("call_client")
        broker = _bound_client_tool_broker("call_client", output_timeout_secs=5.0, context=context)
        serializer.set_client_tool_broker(broker)
        gate = _install_client_tool_gate(serializer, controller, context)
        results: list[Any] = []

        output_frame = await asyncio.wait_for(
            serializer.deserialize(
                json.dumps(
                    {
                        "type": "conversation.item.create",
                        "item": {
                            "type": "function_call_output",
                            "call_id": "call_client",
                            "output": '{"ok":true}',
                        },
                    }
                )
            ),
            timeout=0.1,
        )
        self.assertIsInstance(output_frame, RealtimeClientToolOutputFrame)
        response_b_frame = await asyncio.wait_for(
            serializer.deserialize(json.dumps({"type": "response.create"})),
            timeout=0.1,
        )
        self.assertIsInstance(response_b_frame, RealtimeDeferredResponseCreateFrame)

        async def process_client_frames_in_pipeline_order() -> None:
            await gate.process_frame(output_frame, FrameDirection.DOWNSTREAM)
            await gate.process_frame(response_b_frame, FrameDirection.DOWNSTREAM)

        pipeline_task = asyncio.create_task(process_client_frames_in_pipeline_order())
        await asyncio.sleep(0)
        self.assertFalse(pipeline_task.done())
        self.assertEqual(recorder.events, [])

        await recorder.emit_batch(controller.finish_response(status="completed"))
        serializer.notify_response_done_published(response_a_id)
        await _wait_until(lambda: recorder.types[-2:] == ["conversation.item.added", "conversation.item.done"])
        self.assertFalse(pipeline_task.done())
        self.assertNotIn("response.created", recorder.types)

        # A staged output remains valid even if Pipecat registers its handler
        # only after the originating response has reached the wire.
        handler_task = asyncio.create_task(broker.handle(_client_tool_params(results)))
        await asyncio.wait_for(asyncio.gather(pipeline_task, handler_task), timeout=1)
        self.assertEqual(gate.push_frame.await_count, 2)
        self.assertIsInstance(gate.push_frame.await_args_list[0].args[0], LLMConfigureOutputFrame)
        self.assertFalse(gate.push_frame.await_args_list[0].args[0].skip_tts)

        response_b_id = controller.active_response_id
        await recorder.emit_batch(controller.finish_response(status="completed"))
        serializer.notify_response_done_published(response_b_id)
        self.assertEqual(
            recorder.types,
            [
                "response.done",
                "conversation.item.added",
                "conversation.item.done",
                "response.created",
                "response.done",
            ],
        )
        self.assertEqual(results, [{"ok": True}])
        self.assertNotIn("error", recorder.types)

    async def test_deferred_response_input_reference_is_frozen_when_request_is_accepted(self) -> None:
        recorder = _Recorder()
        controller = _controller(tool_owner="client", output_kind="text")
        controller.start_function_call(call_id="call_client", name="lookup", arguments={"key": "value"})
        serializer = _serializer(recorder, controller)
        referenced_context = {"role": "user", "content": "original"}
        context = _client_tool_context("call_client")
        context.add_message(referenced_context)
        gate = _install_client_tool_gate(serializer, controller, context)
        referenced_events = controller.create_conversation_item(
            {
                "id": "item_reference_snapshot",
                "type": "message",
                "role": "user",
                "status": "completed",
                "content": [{"type": "input_text", "text": "original"}],
            }
        )
        self.assertEqual(referenced_events[0]["item"]["id"], "item_reference_snapshot")
        serializer.bind_conversation_context_message("item_reference_snapshot", referenced_context)
        serializer.set_client_tool_broker(
            SimpleNamespace(
                pending_context_call_ids=lambda: (),
                outputs_staged=AsyncMock(return_value=True),
            )
        )

        frame = await serializer.deserialize(
            json.dumps(
                {
                    "type": "response.create",
                    "response": {"input": [{"type": "item_reference", "id": "item_reference_snapshot"}]},
                }
            )
        )
        self.assertIsInstance(frame, RealtimeDeferredResponseCreateFrame)
        self.assertEqual(frame.input_messages, [{"role": "user", "content": "original"}])

        referenced_context["content"] = "mutated later"
        controller.delete_item_event("item_reference_snapshot")
        self.assertEqual(frame.input_messages, [{"role": "user", "content": "original"}])
        gate.reset()

    async def test_duplicate_early_client_output_is_rejected_without_releasing_first(self) -> None:
        recorder = _Recorder()
        controller = _controller(tool_owner="client")
        controller.start_function_call(call_id="call_client", name="lookup", arguments={})
        response_id = controller.tool_call("call_client").response_id
        serializer = _serializer(recorder, controller)
        context = _client_tool_context("call_client")
        broker = _bound_client_tool_broker("call_client", output_timeout_secs=5.0, context=context)
        serializer.set_client_tool_broker(broker)
        gate = _install_client_tool_gate(serializer, controller, context)
        results: list[Any] = []
        handler_task = asyncio.create_task(broker.handle(_client_tool_params(results)))
        output_event = json.dumps(
            {
                "type": "conversation.item.create",
                "item": {
                    "type": "function_call_output",
                    "call_id": "call_client",
                    "output": "first",
                },
            }
        )

        first_frame = await asyncio.wait_for(serializer.deserialize(output_event), timeout=0.1)
        self.assertIsInstance(first_frame, RealtimeClientToolOutputFrame)
        first_task = asyncio.create_task(gate.process_frame(first_frame, FrameDirection.DOWNSTREAM))
        await asyncio.sleep(0)
        self.assertFalse(first_task.done())
        duplicate_frame = await asyncio.wait_for(serializer.deserialize(output_event), timeout=0.1)
        self.assertIsNone(duplicate_frame)
        self.assertEqual(recorder.events[-1]["error"]["code"], "duplicate_tool_output")
        self.assertFalse(handler_task.done())

        await recorder.emit_batch(controller.finish_response(status="completed"))
        serializer.notify_response_done_published(response_id)
        await asyncio.wait_for(first_task, timeout=1)
        await asyncio.wait_for(handler_task, timeout=1)
        self.assertEqual(results, ["first"])

    async def test_early_client_output_is_discarded_for_noncompleted_response(self) -> None:
        terminal_cases = (
            ("cancelled", "client_cancelled"),
            ("failed", "llm_provider_error"),
            ("incomplete", "max_output_tokens"),
        )
        for status, reason in terminal_cases:
            with self.subTest(status=status):
                recorder = _Recorder()
                controller = _controller(tool_owner="client")
                controller.start_function_call(call_id="call_client", name="lookup", arguments={})
                response_id = controller.tool_call("call_client").response_id
                serializer = _serializer(recorder, controller)
                context = _client_tool_context("call_client")
                broker = _bound_client_tool_broker(
                    "call_client",
                    output_timeout_secs=5.0,
                    context=context,
                )
                serializer.set_client_tool_broker(broker)
                gate = _install_client_tool_gate(serializer, controller, context)
                results: list[Any] = []
                output_frame = await asyncio.wait_for(
                    serializer.deserialize(
                        json.dumps(
                            {
                                "type": "conversation.item.create",
                                "item": {
                                    "type": "function_call_output",
                                    "call_id": "call_client",
                                    "output": '{"must_not":"reach_context"}',
                                },
                            }
                        )
                    ),
                    timeout=0.1,
                )
                self.assertIsInstance(output_frame, RealtimeClientToolOutputFrame)
                output_task = asyncio.create_task(gate.process_frame(output_frame, FrameDirection.DOWNSTREAM))
                handler_task = asyncio.create_task(broker.handle(_client_tool_params(results)))
                await asyncio.sleep(0)
                self.assertFalse(output_task.done())

                await recorder.emit_batch(controller.finish_response(status=status, reason=reason))
                serializer.notify_response_done_published(response_id)
                await asyncio.wait_for(asyncio.gather(output_task, handler_task), timeout=1)

                self.assertEqual(recorder.types, ["response.done", "error"])
                self.assertEqual(recorder.events[-1]["error"]["code"], "tool_call_not_active")
                self.assertEqual(len(results), 1)
                self.assertIsInstance(results[0], ClientToolCancelledResult)
                self.assertEqual(results[0]["error"]["code"], "client_tool_cancelled")
                self.assertFalse(controller.tool_call("call_client").completed)

    async def test_connection_close_cancels_an_outstanding_client_tool_handler(self) -> None:
        recorder = _Recorder()
        serializer = _serializer(recorder)
        broker = _bound_client_tool_broker("call_client", output_timeout_secs=30.0)
        serializer.set_client_tool_broker(broker)
        results: list[Any] = []
        handler_task = asyncio.create_task(broker.handle(_client_tool_params(results)))
        await asyncio.sleep(0)

        serializer.notify_connection_closed()
        [handler_result] = await asyncio.wait_for(
            asyncio.gather(handler_task, return_exceptions=True),
            timeout=1,
        )

        self.assertIsInstance(handler_result, asyncio.CancelledError)
        self.assertEqual(len(results), 1)
        self.assertIsInstance(results[0], ClientToolCancelledResult)
        self.assertEqual(results[0]["error"]["code"], "client_tool_cancelled")

    async def test_result_callback_failure_retires_client_tool_and_wakes_context_waiter(self) -> None:
        broker = _bound_client_tool_broker("call_failed", output_timeout_secs=5.0)

        async def result_callback(_result: Any, *, properties: Any) -> None:  # noqa: ARG001
            raise RuntimeError("context write failed")

        params = SimpleNamespace(
            tool_call_id="call_failed",
            function_name="lookup",
            result_callback=result_callback,
        )
        handler = asyncio.create_task(broker.handle(params))
        await asyncio.sleep(0)
        await broker.stage_output(call_id="call_failed", name="lookup", output='{"ok":true}')
        context_wait = asyncio.create_task(broker.wait_context_applied("call_failed", timeout=10))
        await broker.release_output(call_id="call_failed", name="lookup")

        with self.assertRaisesRegex(RuntimeError, "context write failed"):
            await handler
        with self.assertRaises(RealtimeProtocolError) as context_error:
            await asyncio.wait_for(context_wait, timeout=0.5)
        self.assertEqual(context_error.exception.code, "client_tool_context_unavailable")
        self.assertEqual(broker.pending_context_call_ids(), ("call_failed",))
        with self.assertRaises(RealtimeProtocolError) as caught:
            await broker.stage_output(call_id="call_failed", name="lookup", output='{"duplicate":true}')
        self.assertEqual(caught.exception.code, "client_tool_handler_unavailable")

    async def test_context_callback_failure_closes_after_publishing_client_tool_output(self) -> None:
        for failure_mode in ("result_callback", "context_update"):
            with self.subTest(failure_mode=failure_mode):
                recorder = _Recorder()
                controller = _controller(tool_owner="client")
                controller.start_function_call(call_id="call_client", name="lookup", arguments={})
                response_a_id = controller.tool_call("call_client").response_id
                controller.finish_response(status="completed")
                serializer = _serializer(recorder, controller)
                context = _client_tool_context("call_client") if failure_mode == "result_callback" else LLMContext([])
                broker = _bound_client_tool_broker(
                    "call_client",
                    output_timeout_secs=5.0,
                    context=context,
                )
                serializer.set_client_tool_broker(broker)
                _install_client_tool_gate(serializer, controller, context)
                close_connection = AsyncMock()
                serializer.set_connection_failure_handler(close_connection)
                serializer.notify_response_done_published(response_a_id)

                async def result_callback(
                    _result: Any,
                    *,
                    properties: Any,
                    _failure_mode: str = failure_mode,
                ) -> None:
                    if _failure_mode == "result_callback":
                        raise RuntimeError("result callback failed")
                    await properties.on_context_updated()

                params = SimpleNamespace(
                    tool_call_id="call_client",
                    function_name="lookup",
                    result_callback=result_callback,
                )
                handler = asyncio.create_task(broker.handle(params))
                await asyncio.sleep(0)
                output_frame = await serializer.deserialize(
                    json.dumps(
                        {
                            "type": "conversation.item.create",
                            "event_id": "client_output_failed",
                            "item": {
                                "type": "function_call_output",
                                "call_id": "call_client",
                                "output": '{"ok":true}',
                            },
                        }
                    )
                )
                self.assertIsInstance(output_frame, RealtimeClientToolOutputFrame)

                await asyncio.wait_for(
                    serializer._finalize_client_tool_output(output_frame),
                    timeout=0.5,
                )
                [handler_result] = await asyncio.gather(handler, return_exceptions=True)

                self.assertIsInstance(handler_result, RuntimeError)
                self.assertEqual(
                    recorder.types,
                    ["conversation.item.added", "conversation.item.done", "error"],
                )
                self.assertEqual(recorder.events[-1]["error"]["code"], "client_tool_context_unavailable")
                self.assertEqual(recorder.events[-1]["error"]["event_id"], "client_output_failed")
                self.assertTrue(serializer.connection_closed)
                close_connection.assert_awaited_once_with("client tool context failed")
                self.assertIsNone(await serializer.deserialize(json.dumps({"type": "response.create"})))
                self.assertIsNone(controller.active_response_id)

    async def test_context_failure_retires_before_blocked_error_publication(self) -> None:
        recorder = _Recorder()
        controller = _controller(tool_owner="client")
        controller.start_function_call(call_id="call_client", name="lookup", arguments={})
        response_a_id = controller.tool_call("call_client").response_id
        controller.finish_response(status="completed")
        serializer = _serializer(recorder, controller)
        context = _client_tool_context("call_client")
        broker = _bound_client_tool_broker(
            "call_client",
            output_timeout_secs=5.0,
            context=context,
        )
        serializer.set_client_tool_broker(broker)
        _install_client_tool_gate(serializer, controller, context)
        serializer.notify_response_done_published(response_a_id)
        error_emit_started = asyncio.Event()
        release_error_emit = asyncio.Event()

        async def emit_batch(events: list[dict[str, Any]]) -> None:
            if any(event["type"] == "error" for event in events):
                error_emit_started.set()
                await release_error_emit.wait()
            recorder.events.extend(events)

        serializer.set_emit(recorder.emit, emit_batch)
        close_connection = AsyncMock()
        serializer.set_connection_failure_handler(close_connection)

        async def result_callback(_result: Any, *, properties: Any) -> None:  # noqa: ARG001
            raise RuntimeError("result callback failed")

        handler = asyncio.create_task(
            broker.handle(
                SimpleNamespace(
                    tool_call_id="call_client",
                    function_name="lookup",
                    result_callback=result_callback,
                )
            )
        )
        await asyncio.sleep(0)
        output_frame = await serializer.deserialize(
            json.dumps(
                {
                    "type": "conversation.item.create",
                    "item": {
                        "type": "function_call_output",
                        "call_id": "call_client",
                        "output": '{"ok":true}',
                    },
                }
            )
        )
        self.assertIsInstance(output_frame, RealtimeClientToolOutputFrame)
        finalize = asyncio.create_task(serializer._finalize_client_tool_output(output_frame))

        await asyncio.wait_for(error_emit_started.wait(), timeout=0.5)
        self.assertTrue(serializer.connection_closed)
        self.assertIsNone(await serializer.deserialize(json.dumps({"type": "response.create"})))
        self.assertIsNone(controller.active_response_id)
        close_connection.assert_not_awaited()

        release_error_emit.set()
        await asyncio.wait_for(finalize, timeout=0.5)
        [handler_result] = await asyncio.gather(handler, return_exceptions=True)
        self.assertIsInstance(handler_result, RuntimeError)
        self.assertEqual(recorder.types[-1], "error")
        close_connection.assert_awaited_once_with("client tool context failed")

    async def test_staged_output_wins_deadline_race_but_late_output_is_rejected(self) -> None:
        broker = _bound_client_tool_broker(
            "call_client",
            "call_late",
            output_timeout_secs=0.01,
        )
        staged_results: list[Any] = []
        staged_handler = asyncio.create_task(broker.handle(_client_tool_params(staged_results)))
        await asyncio.sleep(0)
        await broker.stage_output(call_id="call_client", name="lookup", output='{"ok":true}')
        await asyncio.sleep(0.03)
        self.assertFalse(staged_handler.done())
        await broker.release_output(call_id="call_client", name="lookup")
        await asyncio.wait_for(staged_handler, timeout=1)
        self.assertEqual(staged_results, [{"ok": True}])

        cancelled_broker = _bound_client_tool_broker(
            "call_cancelled",
            output_timeout_secs=0.01,
        )
        cancelled_results: list[Any] = []
        cancelled_handler = asyncio.create_task(
            cancelled_broker.handle(_client_tool_params(cancelled_results, call_id="call_cancelled"))
        )
        await asyncio.sleep(0)
        await cancelled_broker.stage_output(
            call_id="call_cancelled",
            name="lookup",
            output="received before deadline",
        )
        await asyncio.sleep(0.03)
        await cancelled_broker.discard_staged_output(call_id="call_cancelled", name="lookup")
        await asyncio.wait_for(cancelled_handler, timeout=1)
        self.assertEqual(len(cancelled_results), 1)
        self.assertIsInstance(cancelled_results[0], ClientToolCancelledResult)

        late_results: list[Any] = []
        late_handler = asyncio.create_task(broker.handle(_client_tool_params(late_results, call_id="call_late")))
        await asyncio.wait_for(late_handler, timeout=1)
        self.assertEqual(len(late_results), 1)
        self.assertIsInstance(late_results[0], ClientToolTimeoutResult)
        with self.assertRaises(RealtimeProtocolError) as caught:
            await broker.stage_output(call_id="call_late", name="lookup", output="too late")
        self.assertEqual(caught.exception.code, "client_tool_timeout")

    async def test_output_arriving_after_absolute_deadline_loses_scheduler_race(self) -> None:
        broker = _bound_client_tool_broker(
            "call_blocked_loop",
            output_timeout_secs=0.01,
        )
        results: list[Any] = []
        handler = asyncio.create_task(broker.handle(_client_tool_params(results, call_id="call_blocked_loop")))
        await asyncio.sleep(0)

        # Keep this task running across the deadline. Before the broker used an
        # absolute arrival timestamp, this coroutine could stage the output
        # before asyncio's already-due timeout callback resumed.
        time.sleep(0.03)
        with self.assertRaises(RealtimeProtocolError) as caught:
            await broker.stage_output(
                call_id="call_blocked_loop",
                name="lookup",
                output="arrived after deadline",
            )

        self.assertEqual(caught.exception.code, "client_tool_timeout")
        await asyncio.wait_for(handler, timeout=1)
        self.assertEqual(len(results), 1)
        self.assertIsInstance(results[0], ClientToolTimeoutResult)
        self.assertEqual(results[0]["error"]["code"], "client_tool_timeout")

    async def test_client_text_item_delete_removes_exact_duplicate_from_context(self) -> None:
        recorder = _Recorder()
        controller = _controller()
        serializer = _serializer(recorder, controller)
        context = LLMContext([])
        gate = _install_client_tool_gate(serializer, controller, context)
        frames = []
        item_ids = []
        for item_id in ("item_duplicate_1", "item_duplicate_2"):
            frame = await serializer.deserialize(
                json.dumps(
                    {
                        "type": "conversation.item.create",
                        "item": {
                            "id": item_id,
                            "type": "message",
                            "role": "user",
                            "content": [{"type": "input_text", "text": "same text"}],
                        },
                    }
                )
            )
            self.assertIsInstance(frame, RealtimeConversationAppendFrame)
            await gate.process_frame(frame, FrameDirection.DOWNSTREAM)
            frames.append(frame)
            item_ids.append(item_id)

        await serializer.deserialize(json.dumps({"type": "conversation.item.delete", "item_id": item_ids[0]}))

        self.assertEqual(recorder.types[-1], "conversation.item.deleted")
        self.assertEqual(recorder.events[-1]["item_id"], item_ids[0])
        self.assertEqual(len(context.get_messages()), 1)
        self.assertIs(context.get_messages()[0], frames[1].context_message)
        with self.assertRaises(RealtimeProtocolError):
            controller.conversation.item(item_ids[0])
        self.assertEqual(controller.conversation.item(item_ids[1])["id"], item_ids[1])

    async def test_client_text_create_ack_and_immediate_delete_wait_for_context_commit(self) -> None:
        recorder = _Recorder()
        controller = _controller()
        serializer = _serializer(recorder, controller)
        context = LLMContext([])
        gate = _install_client_tool_gate(serializer, controller, context)

        append = await serializer.deserialize(
            json.dumps(
                {
                    "type": "conversation.item.create",
                    "item": {
                        "id": "item_ordered",
                        "type": "message",
                        "role": "user",
                        "content": [{"type": "input_text", "text": "ordered"}],
                    },
                }
            )
        )
        self.assertIsInstance(append, RealtimeConversationAppendFrame)
        self.assertEqual(recorder.events, [])

        delete = asyncio.create_task(
            serializer.deserialize(json.dumps({"type": "conversation.item.delete", "item_id": "item_ordered"}))
        )
        await asyncio.sleep(0)
        self.assertFalse(delete.done())

        await gate.process_frame(append, FrameDirection.DOWNSTREAM)
        await asyncio.wait_for(delete, timeout=1)
        self.assertEqual(
            recorder.types,
            ["conversation.item.added", "conversation.item.done", "conversation.item.deleted"],
        )
        self.assertEqual(context.get_messages(), [])
        self.assertNotIn("item_ordered", serializer._context_applied_events)

    async def test_assistant_delete_after_response_done_waits_for_aggregator_context(self) -> None:
        recorder = _Recorder()
        controller = _controller(output_kind="text")
        serializer = _serializer(recorder, controller)
        context = LLMContext([])
        serializer.bind_context(context)
        controller.start_response()
        controller.append_assistant_text("answer")
        response_id = controller.active_response_id
        controller.finish_response(status="completed")
        self.assertIsNotNone(response_id)
        item_id = controller.last_assistant_item_id
        self.assertIsNotNone(item_id)
        serializer.notify_response_done_published(response_id)

        delete = asyncio.create_task(
            serializer.deserialize(json.dumps({"type": "conversation.item.delete", "item_id": item_id}))
        )
        await asyncio.sleep(0)
        self.assertFalse(delete.done())

        context.add_message({"role": "assistant", "content": "answer"})
        self.assertTrue(serializer.bind_latest_assistant_context_message())
        await asyncio.wait_for(delete, timeout=1)
        self.assertEqual(recorder.types, ["conversation.item.deleted"])
        self.assertEqual(context.get_messages(), [])

    async def test_item_delete_rejects_unmapped_audio_without_text_matching(self) -> None:
        recorder = _Recorder()
        controller = _controller()
        serializer = _serializer(recorder, controller)
        context = LLMContext([{"role": "user", "content": "same transcript"}])
        serializer.bind_context(context)
        item_id, _audio_start_ms = controller.begin_user_turn()
        controller.set_user_transcript("same transcript", item_id=item_id)
        stopped_item_id, _audio_end_ms, _events = controller.stop_user_turn()
        self.assertEqual(stopped_item_id, item_id)

        frame = await serializer.deserialize(
            json.dumps({"type": "conversation.item.delete", "item_id": item_id, "event_id": "delete_audio"})
        )

        self.assertIsNone(frame)
        self.assertEqual(recorder.events[-1]["error"]["code"], "conversation_item_context_unavailable")
        self.assertEqual(recorder.events[-1]["error"]["event_id"], "delete_audio")
        self.assertEqual(controller.conversation.item(item_id)["id"], item_id)
        self.assertEqual(context.get_messages(), [{"role": "user", "content": "same transcript"}])

    async def test_zero_truncate_clears_exact_assistant_audio_context(self) -> None:
        recorder = _Recorder()
        controller = _controller()
        serializer = _serializer(recorder, controller)
        context_message = {"role": "assistant", "content": "heard only by the server"}
        context = LLMContext([context_message])
        serializer.bind_context(context)
        controller.start_response()
        controller.append_assistant_text(context_message["content"])
        controller.finish_response(status="completed")
        item_id = controller.last_assistant_item_id
        self.assertIsNotNone(item_id)
        self.assertTrue(serializer.bind_latest_assistant_context_message())

        frame = await serializer.deserialize(
            json.dumps(
                {
                    "type": "conversation.item.truncate",
                    "item_id": item_id,
                    "content_index": 0,
                    "audio_end_ms": 0,
                }
            )
        )

        self.assertIsNone(frame)
        self.assertEqual(recorder.types[-1], "conversation.item.truncated")
        self.assertEqual(context.get_messages(), [])
        item = controller.conversation.item(item_id)
        self.assertEqual(item["content"][0]["transcript"], "")

        await serializer.deserialize(json.dumps({"type": "conversation.item.delete", "item_id": item_id}))
        self.assertEqual(recorder.types[-1], "conversation.item.deleted")

    async def test_positive_item_truncate_uses_inclusive_exact_checkpoint(self) -> None:
        recorder = _Recorder()
        controller = _controller()
        item_id = _finish_playout_aligned_assistant(controller)
        context_message = {"role": "assistant", "content": "Hello there"}
        context = LLMContext([context_message])
        serializer = _serializer(recorder, controller)
        serializer.bind_context(context)
        self.assertTrue(serializer.bind_latest_assistant_context_message())

        frame = await serializer.deserialize(
            json.dumps(
                {
                    "type": "conversation.item.truncate",
                    "item_id": item_id,
                    "content_index": 0,
                    "audio_end_ms": 100,
                }
            )
        )

        self.assertIsNone(frame)
        self.assertEqual(recorder.types[-1], "conversation.item.truncated")
        self.assertEqual(recorder.events[-1]["audio_end_ms"], 100)
        self.assertEqual(controller.conversation.item(item_id)["content"][0]["transcript"], "Hello")
        self.assertEqual(context.get_messages(), [{"role": "assistant", "content": "Hello"}])
        self.assertIsNot(context.get_messages()[0], context_message)

        await serializer.deserialize(
            json.dumps(
                {
                    "type": "conversation.item.truncate",
                    "item_id": item_id,
                    "content_index": 0,
                    "audio_end_ms": 101,
                }
            )
        )
        self.assertEqual(recorder.events[-1]["error"]["code"], "invalid_value")
        self.assertEqual(controller.conversation.item(item_id)["content"][0]["transcript"], "Hello")
        self.assertEqual(context.get_messages(), [{"role": "assistant", "content": "Hello"}])

    async def test_positive_item_truncate_between_checkpoints_keeps_only_known_played_text(self) -> None:
        recorder = _Recorder()
        controller = _controller()
        item_id = _finish_playout_aligned_assistant(controller)
        context_message = {"role": "assistant", "content": "Hello there"}
        context = LLMContext([context_message])
        serializer = _serializer(recorder, controller)
        serializer.bind_context(context)
        self.assertTrue(serializer.bind_latest_assistant_context_message())

        await serializer.deserialize(
            json.dumps(
                {
                    "type": "conversation.item.truncate",
                    "item_id": item_id,
                    "content_index": 0,
                    "audio_end_ms": 150,
                }
            )
        )

        self.assertEqual(controller.conversation.item(item_id)["content"][0]["transcript"], "Hello")
        self.assertEqual(context.get_messages(), [{"role": "assistant", "content": "Hello"}])

    async def test_positive_item_truncate_uses_raw_text_for_model_context_projection(self) -> None:
        recorder = _Recorder()
        controller = _controller()
        item_id = _finish_playout_aligned_assistant(
            controller,
            transcript_fragments=("Hello", "there"),
            context_fragments=("Hello,", "friend!"),
        )
        context = LLMContext([{"role": "assistant", "content": "Hello, friend!"}])
        serializer = _serializer(recorder, controller)
        serializer.bind_context(context)
        self.assertTrue(serializer.bind_latest_assistant_context_message())

        await serializer.deserialize(
            json.dumps(
                {
                    "type": "conversation.item.truncate",
                    "item_id": item_id,
                    "content_index": 0,
                    "audio_end_ms": 150,
                }
            )
        )

        self.assertEqual(controller.conversation.item(item_id)["content"][0]["transcript"], "Hello")
        self.assertEqual(context.get_messages(), [{"role": "assistant", "content": "Hello,"}])

    async def test_positive_item_truncate_rejects_boundary_beyond_wire_audio_atomically(self) -> None:
        recorder = _Recorder()
        controller = _controller()
        item_id = _finish_playout_aligned_assistant(controller)
        context_message = {"role": "assistant", "content": "Hello there"}
        context = LLMContext([context_message])
        serializer = _serializer(recorder, controller)
        serializer.bind_context(context)
        self.assertTrue(serializer.bind_latest_assistant_context_message())

        await serializer.deserialize(
            json.dumps(
                {
                    "type": "conversation.item.truncate",
                    "item_id": item_id,
                    "content_index": 0,
                    "audio_end_ms": 301,
                    "event_id": "truncate_past_end",
                }
            )
        )

        error = recorder.events[-1]["error"]
        self.assertEqual(error["code"], "invalid_value")
        self.assertEqual(error["param"], "audio_end_ms")
        self.assertEqual(error["event_id"], "truncate_past_end")
        self.assertEqual(controller.conversation.item(item_id)["content"][0]["transcript"], "Hello there")
        self.assertEqual(context.get_messages(), [context_message])

    async def test_positive_item_truncate_rejects_missing_playout_alignment_atomically(self) -> None:
        recorder = _Recorder()
        controller = _controller()
        context_message = {"role": "assistant", "content": "complete transcript"}
        context = LLMContext([context_message])
        serializer = _serializer(recorder, controller)
        serializer.bind_context(context)
        controller.start_response()
        controller.append_assistant_text(context_message["content"])
        controller.finish_response(status="completed")
        item_id = controller.last_assistant_item_id
        self.assertTrue(serializer.bind_latest_assistant_context_message())

        await serializer.deserialize(
            json.dumps(
                {
                    "type": "conversation.item.truncate",
                    "item_id": item_id,
                    "content_index": 0,
                    "audio_end_ms": 1,
                }
            )
        )

        self.assertEqual(recorder.events[-1]["error"]["code"], "invalid_truncate")
        self.assertEqual(controller.conversation.item(item_id)["content"][0]["transcript"], "complete transcript")
        self.assertEqual(context.get_messages(), [context_message])

    async def test_positive_item_truncate_rejects_model_context_mismatch_atomically(self) -> None:
        recorder = _Recorder()
        controller = _controller()
        item_id = _finish_playout_aligned_assistant(controller)
        context_message = {"role": "assistant", "content": "different text"}
        context = LLMContext([context_message])
        serializer = _serializer(recorder, controller)
        serializer.bind_context(context)
        self.assertTrue(serializer.bind_latest_assistant_context_message())

        await serializer.deserialize(
            json.dumps(
                {
                    "type": "conversation.item.truncate",
                    "item_id": item_id,
                    "content_index": 0,
                    "audio_end_ms": 150,
                }
            )
        )

        self.assertEqual(recorder.events[-1]["error"]["code"], "conversation_item_context_mismatch")
        self.assertEqual(controller.conversation.item(item_id)["content"][0]["transcript"], "Hello there")
        self.assertEqual(context.get_messages(), [context_message])

    async def test_item_truncate_rejects_invalid_item_content_and_state(self) -> None:
        cases: list[tuple[str, RealtimeSessionController, str, int, str]] = []

        missing = _controller()
        cases.append(("missing", missing, "item_missing", 0, "item_not_found"))

        user = _controller()
        user_event = user.conversation.add_item(
            {
                "type": "message",
                "role": "user",
                "status": "completed",
                "content": [{"type": "input_text", "text": "hello"}],
            }
        )
        cases.append(("user", user, user_event["item"]["id"], 0, "invalid_truncate"))

        text = _controller(output_kind="text")
        text.start_response()
        text.append_assistant_text("hello")
        text.finish_response(status="completed")
        cases.append(("text_part", text, text.last_assistant_item_id, 0, "invalid_truncate"))

        in_progress = _controller()
        progress_event = in_progress.conversation.add_item(
            {
                "type": "message",
                "role": "assistant",
                "status": "in_progress",
                "content": [{"type": "output_audio", "transcript": ""}],
            }
        )
        cases.append(("in_progress", in_progress, progress_event["item"]["id"], 0, "invalid_truncate"))

        wrong_index = _controller()
        wrong_index_item = _finish_playout_aligned_assistant(wrong_index)
        cases.append(("content_index", wrong_index, wrong_index_item, 1, "invalid_truncate"))

        for name, controller, item_id, content_index, expected_code in cases:
            with self.subTest(name=name):
                recorder = _Recorder()
                serializer = _serializer(recorder, controller)
                await serializer.deserialize(
                    json.dumps(
                        {
                            "type": "conversation.item.truncate",
                            "item_id": item_id,
                            "content_index": content_index,
                            "audio_end_ms": 0,
                        }
                    )
                )
                self.assertEqual(recorder.events[-1]["error"]["code"], expected_code)

    async def test_item_truncate_rejects_invalid_numeric_fields(self) -> None:
        invalid_cases = (
            (True, 0, "content_index", "invalid_type"),
            (0.0, 0, "content_index", "invalid_type"),
            (-1, 0, "content_index", "invalid_value"),
            (0, True, "audio_end_ms", "invalid_type"),
            (0, 1.5, "audio_end_ms", "invalid_type"),
            (0, -1, "audio_end_ms", "invalid_value"),
        )
        for content_index, audio_end_ms, param, code in invalid_cases:
            with self.subTest(content_index=content_index, audio_end_ms=audio_end_ms):
                recorder = _Recorder()
                serializer = _serializer(recorder)
                await serializer.deserialize(
                    json.dumps(
                        {
                            "type": "conversation.item.truncate",
                            "item_id": "item_any",
                            "content_index": content_index,
                            "audio_end_ms": audio_end_ms,
                        }
                    )
                )
                self.assertEqual(recorder.events[-1]["error"]["code"], code)
                self.assertEqual(recorder.events[-1]["error"]["param"], param)

    async def test_item_truncate_rejects_active_assistant_item(self) -> None:
        recorder = _Recorder()
        controller = _controller()
        controller.start_response()
        controller.output_audio_delta_event(
            "encoded-audio",
            sample_count=2400,
            sample_rate=24000,
        )
        item_id = controller.assistant_item_id
        serializer = _serializer(recorder, controller)

        await serializer.deserialize(
            json.dumps(
                {
                    "type": "conversation.item.truncate",
                    "item_id": item_id,
                    "content_index": 0,
                    "audio_end_ms": 50,
                }
            )
        )

        self.assertEqual(recorder.events[-1]["error"]["code"], "conversation_item_in_use")

    async def test_output_audio_buffer_clear_remains_websocket_unsupported(self) -> None:
        recorder = _Recorder()
        serializer = _serializer(recorder)

        frame = await serializer.deserialize(
            json.dumps({"type": "output_audio_buffer.clear", "event_id": "clear_output"})
        )

        self.assertIsNone(frame)
        self.assertEqual(recorder.events[-1]["error"]["code"], "unsupported_capability")
        self.assertEqual(recorder.events[-1]["error"]["event_id"], "clear_output")


class SessionStartHandlerTests(unittest.IsolatedAsyncioTestCase):
    async def test_realtime_start_configures_output_without_unsolicited_intro(self) -> None:
        from examples.shared.pipeline_utils import register_session_start_handlers

        controller = _controller(output_kind="text")
        task = SimpleNamespace(queue_frames=AsyncMock())
        context = SimpleNamespace(add_message=MagicMock())

        class _Transport:
            def __init__(self) -> None:
                self.handlers: dict[str, Any] = {}

            def event_handler(self, name: str):
                def decorator(fn):
                    self.handlers[name] = fn
                    return fn

                return decorator

        transport = _Transport()
        runner_args = SimpleNamespace(
            body={
                "protocol": "realtime",
                "realtime_controller": controller,
            }
        )
        with patch("realtime.transport.realtime_controller", return_value=controller):
            register_session_start_handlers(
                transport=transport,
                task=task,
                context=context,
                runner_args=runner_args,
                welcome_enabled=True,
            )
            await transport.handlers["on_client_connected"](transport, object())

        queued = task.queue_frames.await_args.args[0]
        self.assertEqual(len(queued), 1)
        self.assertIsInstance(queued[0], LLMConfigureOutputFrame)
        self.assertTrue(queued[0].skip_tts)
        context.add_message.assert_not_called()


if __name__ == "__main__":
    unittest.main()
