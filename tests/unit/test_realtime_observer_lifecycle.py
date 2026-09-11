# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""Observable Realtime response lifecycles at the Pipecat observer boundary."""

# ruff: noqa: D102

from __future__ import annotations

import json
import unittest
from typing import Any
from unittest.mock import AsyncMock, MagicMock

from pipecat.frames.frames import (
    FunctionCallResultFrame,
    FunctionCallsStartedFrame,
    InterruptionFrame,
    LLMFullResponseEndFrame,
    LLMTextFrame,
    MetricsFrame,
    OutputAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
    TTSTextFrame,
)
from pipecat.metrics.metrics import LLMTokenUsage, LLMUsageMetricsData, ProcessingMetricsData, TTFBMetricsData
from pipecat.observers.base_observer import FramePushed
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.llm_service import FunctionCallFromLLM, FunctionCallResultProperties
from pipecat.services.tts_service import TTSService
from pipecat.transports.base_output import BaseOutputTransport

from realtime.controller import RealtimeSessionController
from realtime.frames import RealtimeOwnedLLMFullResponseStartFrame
from realtime.observer import RealtimeLifecycleObserver, _serialize_tool_result
from realtime.serializer import RealtimeFrameSerializer
from realtime.session import RealtimeSessionCapabilities
from realtime.transport import RealtimeToolResultPolicy


def _tool(name: str) -> dict[str, Any]:
    return {
        "type": "function",
        "name": name,
        "description": f"Run {name}",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
            "additionalProperties": False,
        },
    }


def _controller(*, output_kind: str = "audio", server_tools: tuple[str, ...] = ()) -> RealtimeSessionController:
    schemas = [_tool(name) for name in server_tools]
    capabilities = RealtimeSessionCapabilities(
        trusted_function_tools=frozenset(server_tools),
    )
    controller = RealtimeSessionController(
        model="test-model",
        voice="test-voice",
        runtime_config={},
        server_tools=list(server_tools),
        trusted_tool_schemas=schemas,
        capabilities=capabilities,
    )
    if output_kind == "text":
        controller.apply_session_update({"output_modalities": ["text"]})
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


async def _push(
    observer: RealtimeLifecycleObserver,
    frame: Any,
    *,
    source: Any,
    destination: Any,
) -> None:
    await observer.on_push_frame(
        FramePushed(
            source=source,
            destination=destination,
            frame=frame,
            direction=FrameDirection.DOWNSTREAM,
            timestamp=0,
        )
    )


def _owned_start(controller: RealtimeSessionController, owner_id: str) -> RealtimeOwnedLLMFullResponseStartFrame:
    generation = controller.interruption_generation
    controller.prepare_pipeline_response(
        owner_id=owner_id,
        interruption_generation=generation,
    )
    return RealtimeOwnedLLMFullResponseStartFrame(
        run_owner_id=owner_id,
        activation_generation=generation,
    )


class ObserverAudioLifecycleTests(unittest.IsolatedAsyncioTestCase):
    """Verify public audio and text response terminal boundaries."""

    async def asyncSetUp(self) -> None:
        self.llm = FrameProcessor(name="lifecycle-llm")
        self.tts = MagicMock(spec=TTSService)
        self.output = MagicMock(spec=BaseOutputTransport)
        self.destination = FrameProcessor(name="lifecycle-destination")

    async def test_audio_response_waits_for_every_tts_context_and_final_playout(self) -> None:
        recorder = _Recorder()
        controller = _controller()
        flush_observations: list[list[str]] = []

        async def flush_output_audio() -> None:
            flush_observations.append(list(recorder.types))

        observer = RealtimeLifecycleObserver(
            emit=recorder.emit,
            emit_batch=recorder.emit_batch,
            controller=controller,
            flush_output_audio=flush_output_audio,
        )
        start = _owned_start(controller, "audio-run")
        await _push(observer, start, source=self.llm, destination=self.destination)
        response_id = next(event["response"]["id"] for event in recorder.events if event["type"] == "response.created")
        await _push(observer, start, source=self.tts, destination=self.destination)

        await _push(observer, TTSStartedFrame(context_id="tts-1"), source=self.tts, destination=self.destination)
        await _push(observer, LLMTextFrame(text="Hello "), source=self.llm, destination=self.destination)
        await _push(
            observer,
            TTSTextFrame(text="Hello", raw_text="Hello", aggregated_by="sentence", context_id="tts-1"),
            source=self.output,
            destination=self.destination,
        )
        await _push(observer, TTSStartedFrame(context_id="tts-2"), source=self.tts, destination=self.destination)
        await _push(observer, LLMTextFrame(text="world."), source=self.llm, destination=self.destination)
        await _push(
            observer,
            TTSTextFrame(text="world.", raw_text="world.", aggregated_by="sentence", context_id="tts-2"),
            source=self.output,
            destination=self.destination,
        )

        end = LLMFullResponseEndFrame()
        await _push(observer, end, source=self.llm, destination=self.destination)
        await _push(observer, end, source=self.tts, destination=self.destination)
        self.assertNotIn("response.done", recorder.types)

        await _push(observer, TTSStoppedFrame(context_id="tts-1"), source=self.output, destination=self.destination)
        self.assertNotIn("response.done", recorder.types)
        self.assertEqual(flush_observations, [])

        await _push(observer, TTSStoppedFrame(context_id="tts-2"), source=self.output, destination=self.destination)

        self.assertEqual(len(flush_observations), 1)
        self.assertNotIn("response.done", flush_observations[0])
        self.assertEqual(recorder.types.count("response.created"), 1)
        self.assertEqual(recorder.types.count("response.done"), 1)
        transcript_deltas = [
            event["delta"] for event in recorder.events if event["type"] == "response.output_audio_transcript.delta"
        ]
        self.assertEqual(transcript_deltas, ["Hello", " world."])
        done = next(event for event in recorder.events if event["type"] == "response.done")
        self.assertEqual(done["response"]["id"], response_id)
        self.assertEqual(done["response"]["status"], "completed")
        self.assertEqual(
            done["response"]["output"][0]["content"],
            [{"type": "output_audio", "transcript": "Hello world."}],
        )
        observer.shutdown()

    async def test_text_output_completes_at_llm_output_edge_without_tts(self) -> None:
        recorder = _Recorder()
        controller = _controller(output_kind="text")
        flush_output_audio = AsyncMock()
        observer = RealtimeLifecycleObserver(
            emit=recorder.emit,
            emit_batch=recorder.emit_batch,
            controller=controller,
            flush_output_audio=flush_output_audio,
        )
        start = _owned_start(controller, "text-run")
        await _push(observer, start, source=self.llm, destination=self.destination)
        await _push(observer, LLMTextFrame(text="Text only."), source=self.llm, destination=self.destination)

        end = LLMFullResponseEndFrame()
        await _push(observer, end, source=self.llm, destination=self.destination)
        self.assertNotIn("response.done", recorder.types)
        await _push(observer, end, source=self.output, destination=self.destination)

        done = next(event for event in recorder.events if event["type"] == "response.done")
        self.assertEqual(done["response"]["status"], "completed")
        self.assertEqual(done["response"]["output"][0]["content"], [{"type": "output_text", "text": "Text only."}])
        self.assertFalse(any(event_type.startswith("response.output_audio") for event_type in recorder.types))
        flush_output_audio.assert_not_awaited()
        observer.shutdown()

    async def test_llm_metrics_project_to_extension_and_standard_usage(self) -> None:
        recorder = _Recorder()
        controller = _controller(output_kind="text")
        observer = RealtimeLifecycleObserver(
            emit=recorder.emit,
            emit_batch=recorder.emit_batch,
            controller=controller,
        )
        await _push(observer, _owned_start(controller, "metrics-run"), source=self.llm, destination=self.destination)
        await _push(
            observer,
            MetricsFrame(
                data=[
                    TTFBMetricsData(processor=self.llm.name, model="test-model", value=0.25),
                    ProcessingMetricsData(processor=self.llm.name, model="test-model", value=0.5),
                    LLMUsageMetricsData(
                        processor=self.llm.name,
                        model="test-model",
                        value=LLMTokenUsage(prompt_tokens=20, completion_tokens=10, total_tokens=30),
                    ),
                ]
            ),
            source=self.llm,
            destination=self.destination,
        )

        self.assertEqual(
            [event["metrics"] for event in recorder.events if event["type"] == "nvidia.metrics.updated"],
            [{"llm_ttft": 0.25}, {"llm_processing_time": 0.5}, {"llm_tokens_per_sec": 20.0}],
        )
        done = controller.finish_response(status="completed")[-1]
        self.assertEqual(done["response"]["usage"]["total_tokens"], 30)
        self.assertEqual(done["response"]["usage"]["input_tokens"], 20)
        self.assertEqual(done["response"]["usage"]["output_tokens"], 10)
        observer.shutdown()

    async def test_cancelled_response_ignores_late_tts_without_reopening(self) -> None:
        recorder = _Recorder()
        controller = _controller()
        observer = RealtimeLifecycleObserver(
            emit=recorder.emit,
            emit_batch=recorder.emit_batch,
            controller=controller,
        )
        start = _owned_start(controller, "cancelled-audio-run")
        await _push(observer, start, source=self.llm, destination=self.destination)
        await _push(observer, start, source=self.tts, destination=self.destination)
        await _push(
            observer,
            TTSStartedFrame(context_id="tts-before-cancel"),
            source=self.tts,
            destination=self.destination,
        )

        await _push(observer, InterruptionFrame(), source=self.llm, destination=self.destination)
        done = next(event for event in recorder.events if event["type"] == "response.done")
        self.assertEqual(done["response"]["status"], "cancelled")
        terminal_event_count = len(recorder.events)

        await _push(
            observer,
            TTSTextFrame(
                text="stale",
                raw_text="stale",
                aggregated_by="sentence",
                context_id="tts-before-cancel",
            ),
            source=self.output,
            destination=self.destination,
        )
        await _push(
            observer,
            TTSStoppedFrame(context_id="tts-before-cancel"),
            source=self.output,
            destination=self.destination,
        )
        await _push(
            observer,
            TTSStartedFrame(context_id="tts-after-cancel"),
            source=self.tts,
            destination=self.destination,
        )
        await _push(
            observer,
            TTSTextFrame(
                text="still stale",
                raw_text="still stale",
                aggregated_by="sentence",
                context_id="tts-after-cancel",
            ),
            source=self.output,
            destination=self.destination,
        )
        await _push(
            observer,
            TTSStoppedFrame(context_id="tts-after-cancel"),
            source=self.output,
            destination=self.destination,
        )
        end = LLMFullResponseEndFrame()
        await _push(observer, end, source=self.llm, destination=self.destination)
        await _push(observer, end, source=self.tts, destination=self.destination)

        self.assertEqual(recorder.events[terminal_event_count:], [])
        self.assertEqual(recorder.types.count("response.created"), 1)
        self.assertEqual(recorder.types.count("response.done"), 1)
        self.assertIsNone(controller.active_response_id)
        self.assertEqual(observer._tts_context_response, {})
        self.assertEqual(observer._pending_tts_contexts, {})
        observer.shutdown()

    async def test_output_audio_conversion_failure_fails_its_owned_response_once(self) -> None:
        recorder = _Recorder()
        controller = _controller()
        serializer = RealtimeFrameSerializer(controller=controller)
        serializer.set_emit(recorder.emit, recorder.emit_batch)
        observer = RealtimeLifecycleObserver(
            emit=recorder.emit,
            emit_batch=recorder.emit_batch,
            controller=controller,
            flush_output_audio=serializer.flush_output_audio,
            reset_output_audio=serializer.reset_output_audio,
        )
        serializer.set_output_audio_failure_handler(observer.on_output_audio_failure)

        start = _owned_start(controller, "audio-conversion-run")
        await _push(observer, start, source=self.llm, destination=self.destination)
        response_id = controller.active_response_id
        await _push(observer, start, source=self.tts, destination=self.destination)
        await _push(
            observer,
            TTSStartedFrame(context_id="tts-conversion"),
            source=self.tts,
            destination=self.destination,
        )
        failure_event_index = len(recorder.events)
        serializer._resampler.from_pipeline = AsyncMock(side_effect=ValueError("provider sample format mismatch"))

        audio = OutputAudioRawFrame(audio=b"\x00\x00" * 160, sample_rate=22050, num_channels=1)
        await serializer.serialize(audio)

        self.assertEqual(
            recorder.types[failure_event_index:],
            [
                "error",
                "response.output_audio.done",
                "response.output_audio_transcript.done",
                "response.content_part.done",
                "conversation.item.done",
                "response.output_item.done",
                "response.done",
            ],
        )
        error = recorder.events[failure_event_index]
        self.assertEqual(error["error"]["type"], "server_error")
        self.assertEqual(error["error"]["code"], "output_audio_conversion_error")
        self.assertEqual(error["error"]["param"], "response.audio.output.format")
        self.assertNotIn("provider sample format mismatch", error["error"]["message"])
        done = recorder.events[-1]
        self.assertEqual(done["response"]["id"], response_id)
        self.assertEqual(done["response"]["status"], "failed")
        self.assertEqual(
            done["response"]["status_details"],
            {
                "type": "failed",
                "error": {
                    "type": "server_error",
                    "code": "output_audio_conversion_error",
                },
            },
        )

        terminal_event_count = len(recorder.events)
        await serializer.serialize(audio)
        self.assertEqual(len(recorder.events), terminal_event_count)
        self.assertIsNone(controller.active_response_id)
        observer.shutdown()


class ObserverServerToolLifecycleTests(unittest.IsolatedAsyncioTestCase):
    """Verify canonical server-tool events and separate response lifecycles."""

    def test_non_json_tool_results_become_structured_failures(self) -> None:
        for result in (object(), b"not-json"):
            with self.subTest(result_type=type(result).__name__):
                output, failure = _serialize_tool_result(result)
                self.assertEqual(
                    failure,
                    ("tool_result_serialization_error", "Tool result could not be serialized as finite JSON"),
                )
                self.assertEqual(json.loads(output)["error"]["code"], "tool_result_serialization_error")

    async def test_parallel_server_results_finish_response_a_before_one_response_b(self) -> None:
        recorder = _Recorder()
        controller = _controller(output_kind="text", server_tools=("lookup", "reserve"))
        observer = RealtimeLifecycleObserver(
            emit=recorder.emit,
            emit_batch=recorder.emit_batch,
            controller=controller,
        )
        llm = FrameProcessor(name="tool-lifecycle-llm")
        output = MagicMock(spec=BaseOutputTransport)
        destination = FrameProcessor(name="tool-lifecycle-destination")
        context = LLMContext([])

        response_a_start = _owned_start(controller, "tool-response-a")
        await _push(observer, response_a_start, source=llm, destination=destination)
        calls = (
            FunctionCallFromLLM(
                function_name="lookup",
                tool_call_id="call-lookup",
                arguments={"query": "weather"},
                context=context,
            ),
            FunctionCallFromLLM(
                function_name="reserve",
                tool_call_id="call-reserve",
                arguments={"query": "seat 12A"},
                context=context,
            ),
        )
        await _push(
            observer,
            FunctionCallsStartedFrame(function_calls=calls),
            source=llm,
            destination=destination,
        )

        for frame in (
            FunctionCallResultFrame(
                function_name="reserve",
                tool_call_id="call-reserve",
                arguments={"query": "seat 12A"},
                result={"ok": False, "error": {"code": "seat_unavailable", "message": "Seat is occupied"}},
                properties=FunctionCallResultProperties(run_llm=True),
            ),
            FunctionCallResultFrame(
                function_name="lookup",
                tool_call_id="call-lookup",
                arguments={"query": "weather"},
                result={"ok": True, "temperature_c": 24},
                properties=FunctionCallResultProperties(run_llm=True),
            ),
        ):
            await _push(observer, frame, source=llm, destination=destination)

        self.assertEqual(recorder.types.count("response.created"), 1)
        self.assertNotIn("function_call_output", [event.get("item", {}).get("type") for event in recorder.events])
        response_a_end = LLMFullResponseEndFrame()
        await _push(observer, response_a_end, source=llm, destination=destination)
        await _push(observer, response_a_end, source=output, destination=destination)

        created = [event for event in recorder.events if event["type"] == "response.created"]
        done = [event for event in recorder.events if event["type"] == "response.done"]
        self.assertEqual(len(created), 1)
        self.assertEqual(len(done), 1)
        response_a_id = created[0]["response"]["id"]
        self.assertEqual(done[0]["response"]["id"], response_a_id)
        self.assertEqual(done[0]["response"]["status"], "completed")
        self.assertEqual(
            {(item["call_id"], item["name"]) for item in done[0]["response"]["output"]},
            {("call-lookup", "lookup"), ("call-reserve", "reserve")},
        )

        argument_events = [
            event for event in recorder.events if event["type"] == "response.function_call_arguments.done"
        ]
        self.assertEqual({event["call_id"] for event in argument_events}, {"call-lookup", "call-reserve"})
        self.assertEqual(len(argument_events), 2)
        output_items = [
            event["item"]
            for event in recorder.events
            if event["type"] == "conversation.item.added"
            and event.get("item", {}).get("type") == "function_call_output"
        ]
        self.assertEqual({item["call_id"] for item in output_items}, {"call-lookup", "call-reserve"})
        decoded_outputs = {item["call_id"]: json.loads(item["output"]) for item in output_items}
        self.assertEqual(decoded_outputs["call-lookup"], {"ok": True, "temperature_c": 24})
        self.assertEqual(decoded_outputs["call-reserve"]["error"]["code"], "seat_unavailable")
        tool_error = next(event for event in recorder.events if event["type"] == "error")
        self.assertEqual(tool_error["error"]["code"], "seat_unavailable")

        response_b_start = _owned_start(controller, "tool-response-b")
        await _push(observer, response_b_start, source=llm, destination=destination)
        await _push(
            observer,
            LLMTextFrame(text="The lookup succeeded, but seat 12A is unavailable."),
            source=llm,
            destination=destination,
        )
        response_b_end = LLMFullResponseEndFrame()
        await _push(observer, response_b_end, source=llm, destination=destination)
        await _push(observer, response_b_end, source=output, destination=destination)

        created = [event for event in recorder.events if event["type"] == "response.created"]
        done = [event for event in recorder.events if event["type"] == "response.done"]
        self.assertEqual(len(created), 2)
        self.assertEqual(len(done), 2)
        response_b_id = created[1]["response"]["id"]
        self.assertNotEqual(response_b_id, response_a_id)
        self.assertEqual(done[1]["response"]["id"], response_b_id)
        self.assertEqual(
            done[1]["response"]["output"][0]["content"],
            [{"type": "output_text", "text": "The lookup succeeded, but seat 12A is unavailable."}],
        )
        first_output_index = next(
            index
            for index, event in enumerate(recorder.events)
            if event["type"] == "conversation.item.added"
            and event.get("item", {}).get("type") == "function_call_output"
        )
        response_b_created_index = next(
            index
            for index, event in enumerate(recorder.events)
            if event["type"] == "response.created" and event["response"]["id"] == response_b_id
        )
        self.assertLess(first_output_index, response_b_created_index)
        observer.shutdown()

    async def test_mixed_result_policy_preserves_observer_frame_identity(self) -> None:
        recorder = _Recorder()
        controller = _controller(output_kind="text", server_tools=("lookup",))
        client_tool = _tool("client_lookup")
        controller.apply_session_update({"tools": [_tool("lookup"), client_tool]})
        controller.bind_session_tool_projection(
            client_tool_bindings={"client_lookup": "client_lookup"},
            mcp_pipeline_names=frozenset(),
        )
        observer = RealtimeLifecycleObserver(
            emit=recorder.emit,
            emit_batch=recorder.emit_batch,
            controller=controller,
        )
        llm = FrameProcessor(name="mixed-tool-llm")
        policy = RealtimeToolResultPolicy(controller=controller)
        destination = FrameProcessor(name="mixed-tool-destination")
        output = MagicMock(spec=BaseOutputTransport)
        context = LLMContext([])

        start = _owned_start(controller, "mixed-tool-response")
        await _push(observer, start, source=llm, destination=policy)
        calls_started = FunctionCallsStartedFrame(
            function_calls=(
                FunctionCallFromLLM(
                    function_name="lookup",
                    tool_call_id="call-server",
                    arguments={"query": "x"},
                    context=context,
                ),
                FunctionCallFromLLM(
                    function_name="client_lookup",
                    tool_call_id="call-client",
                    arguments={"query": "y"},
                    context=context,
                ),
            )
        )
        await _push(observer, calls_started, source=llm, destination=policy)
        policy.push_frame = AsyncMock()
        await policy.process_frame(calls_started, FrameDirection.DOWNSTREAM)
        result = FunctionCallResultFrame("lookup", "call-server", {"query": "x"}, {"ok": True})
        await _push(observer, result, source=llm, destination=policy)
        policy.push_frame.reset_mock()
        await policy.process_frame(result, FrameDirection.DOWNSTREAM)
        forwarded = policy.push_frame.await_args.args[0]
        await _push(observer, forwarded, source=policy, destination=destination)
        end = LLMFullResponseEndFrame()
        await _push(observer, end, source=llm, destination=destination)
        await _push(observer, end, source=output, destination=destination)

        self.assertIs(forwarded, result)
        self.assertFalse(forwarded.properties.run_llm)
        self.assertEqual(
            sum(
                event.get("item", {}).get("type") == "function_call_output"
                and event.get("item", {}).get("call_id") == "call-server"
                for event in recorder.events
            ),
            2,
        )
        self.assertFalse(
            any(event.get("error", {}).get("code") == "duplicate_tool_output" for event in recorder.events)
        )
        observer.shutdown()


if __name__ == "__main__":
    unittest.main()
