# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D100, D101, D102

import asyncio
import json
import os
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.evals.fastapi_transport import EvalFastAPIWebsocketTransport
from pipecat.frames.frames import InterruptionFrame, OutputTransportMessageFrame
from pipecat.runner.types import WebSocketRunnerArguments

from examples.generic import pipeline as generic_pipeline
from examples.shared.pipeline_utils import (
    WebClientFrameSerializer,
    build_user_aggregator_params,
    create_transport,
    latency_metric_group,
    turn_taking_latency_secs,
)


class GenericASRSelectionTest(unittest.IsolatedAsyncioTestCase):
    async def test_catalog_type_selects_stt_and_vad_independently_of_http_url(self):
        transcribe = {"asr_type": "transcribe"}
        prompted = {**transcribe, "transcription_prompt": "Produce a verbatim transcript of the audio."}
        spanish = {**transcribe, "language_code": "es-ES"}
        english = {"language_code": "en-US"}
        for asr_id, default, selected, expected, prompt, language, expected_language in (
            ("self-hosted:transcribe", {}, transcribe, True, None, "en-US", "en-US"),
            ("self-hosted:transcribe", {}, prompted, True, prompted["transcription_prompt"], "en-US", "en-US"),
            ("", transcribe, {}, True, None, "en-US", "en-US"),
            ("self-hosted:streaming", transcribe, {}, False, None, "en-US", "en-US"),
            ("", {}, {}, False, None, "en-US", "en-US"),
            ("self-hosted:transcribe", english, spanish, True, None, "", "es-ES"),
            ("self-hosted:transcribe", english, spanish, True, None, "en-US", "en-US"),
            ("self-hosted:transcribe", english, transcribe, True, None, "", None),
            ("", spanish, {}, True, None, "", "es-ES"),
            ("self-hosted:streaming", english, {}, False, None, "", "en-US"),
        ):
            body = {
                "asr_server": "http://speech.invalid:9000/v1",
                "asr_model": "test-asr",
                "asr_language_code": language,
                "asr_id": asr_id,
            }
            with (
                self.subTest(asr_id=asr_id, default=default, selected=selected, language=language),
                patch.object(generic_pipeline, "create_transport", new_callable=AsyncMock),
                patch.object(generic_pipeline, "resolve_prompt", return_value=("test", "test prompt")),
                patch.object(generic_pipeline, "load_prompt_catalog", return_value={}),
                patch.object(
                    generic_pipeline,
                    "load_service_entry",
                    side_effect=lambda slot, _, default=default: default if slot == "asr" else {},
                ),
                patch.object(generic_pipeline, "load_service_entry_by_id", return_value=selected) as lookup,
                patch.object(generic_pipeline, "nvidia_api_key", return_value=""),
                patch.object(generic_pipeline, "NvidiaSTTService") as streaming_stt,
                patch.object(generic_pipeline, "NvidiaTranscribeSTTService") as transcribe_stt,
                patch.object(generic_pipeline, "NvidiaLLMService"),
                patch.object(generic_pipeline, "NvidiaTTSService"),
                patch.object(generic_pipeline, "build_tools_schema", return_value=(None, [])),
                patch.object(
                    generic_pipeline, "build_user_aggregator_params", side_effect=RuntimeError("params built")
                ) as params,
                self.assertRaisesRegex(RuntimeError, "params built"),
            ):
                await generic_pipeline.bot(SimpleNamespace(body=body))
            self.assertEqual(transcribe_stt.call_count, int(expected))
            self.assertEqual(streaming_stt.call_count, int(not expected))
            params.assert_called_once_with(True, vad_only=expected)
            if expected:
                transcribe_stt.Settings.assert_called_once_with(
                    model="test-asr", language=expected_language, prompt=prompt
                )
                transcribe_stt.assert_called_once_with(
                    base_url=body["asr_server"], settings=transcribe_stt.Settings.return_value
                )
            else:
                self.assertEqual(streaming_stt.call_args.kwargs["settings"].language, expected_language)
            if asr_id:
                lookup.assert_called_once_with("asr", asr_id)
            else:
                lookup.assert_not_called()


class UserAggregatorParamsTest(unittest.TestCase):
    def test_vad_only_forces_the_pure_vad_branch(self):
        for override, expected in (("", 0.5), ("0.8", 0.8)):
            with (
                self.subTest(override=override),
                patch.dict(os.environ, {"SILERO_VAD_STOP_SECS": override, "USE_SILERO_VAD_TURN_DETECTION": "false"}),
            ):
                params = build_user_aggregator_params(False, vad_only=True)
                self.assertEqual(params.vad_analyzer.params.start_secs, VADParams().start_secs)
                self.assertEqual(params.vad_analyzer.params.stop_secs, expected)
                self.assertEqual(build_user_aggregator_params(False).vad_analyzer.params.stop_secs, 0.2)


def _breakdown(**overrides):
    fields = {
        "user_turn_secs": 0.2,
        "ttfb": [SimpleNamespace(processor="NvidiaLLMService#1", duration_secs=0.0375)],
        "text_aggregation": None,
        "contributions": [
            SimpleNamespace(key="endpointing_wait", start_time=10.0, duration_secs=0.2),
            SimpleNamespace(key="transcription", start_time=10.2, duration_secs=0.3),
            SimpleNamespace(key="turn_detection", start_time=10.5, duration_secs=0.25),
            SimpleNamespace(key="llm_inference", start_time=10.75, duration_secs=0.4),
        ],
    }
    return SimpleNamespace(**(fields | overrides))


class LatencyMetricGroupTest(unittest.TestCase):
    def test_group_names_the_turn_and_reports_milliseconds(self):
        group = latency_metric_group(2, 0.5, _breakdown())

        self.assertEqual((group["type"], group["group_id"], group["group_label"]), ("metric-group", "turn-2", "Turn 2"))
        self.assertEqual(
            [(m["key"], m["value"]) for m in group["metrics"]],
            [
                ("total_latency_ms", 500.0),
                ("user_turn_ms", 200.0),
                ("turn_taking_latency_ms", 550.0),
                ("ttfb_0_NvidiaLLMService_1", 37.5),
            ],
        )

    def test_group_skips_missing_user_turn(self):
        group = latency_metric_group(1, 0.25, _breakdown(user_turn_secs=None))
        self.assertNotIn("user_turn_ms", [m["key"] for m in group["metrics"]])

    def test_turn_taking_latency_is_vad_stop_to_llm_request(self):
        self.assertAlmostEqual(turn_taking_latency_secs(_breakdown()), 0.55)

    def test_turn_taking_latency_requires_vad_and_llm_boundaries(self):
        self.assertIsNone(turn_taking_latency_secs(_breakdown(contributions=[])))


class WebClientFrameSerializerTest(unittest.TestCase):
    def test_skips_interruptions_and_keeps_messages(self):
        serializer = WebClientFrameSerializer()
        self.assertIsNone(asyncio.run(serializer.serialize(InterruptionFrame())))
        self.assertTrue(asyncio.run(serializer.serialize(OutputTransportMessageFrame(message={"label": "rtvi-ai"}))))


class EvalTransportFactoryTest(unittest.IsolatedAsyncioTestCase):
    async def test_accepted_eval_websocket_uses_native_eval_transport(self):
        class _WebSocket:
            def iter_text(self):
                async def messages():
                    yield json.dumps(
                        {
                            "label": "rtvi-ai",
                            "type": "client-message",
                            "data": {"t": "eval-connect", "d": {"skip_tts": True}},
                        }
                    )

                return messages()

        args = WebSocketRunnerArguments(websocket=_WebSocket())
        transport = await create_transport(args)

        self.assertIsInstance(transport, EvalFastAPIWebsocketTransport)
        self.assertEqual(args.transport_type, "eval")


if __name__ == "__main__":
    unittest.main()
