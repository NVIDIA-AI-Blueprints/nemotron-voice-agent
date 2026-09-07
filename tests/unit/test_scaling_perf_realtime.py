# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

# ruff: noqa: D100,D101,D102

import asyncio
import json
import sys
import tempfile
import unittest
from dataclasses import fields
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

SCALING_PERF_DIR = Path(__file__).resolve().parents[2] / "benchmarking_tools" / "scaling-perf"
sys.path.insert(0, str(SCALING_PERF_DIR))

from benchmark import resolve_protocol  # noqa: E402
from benchmark_aggregation import aggregate_run_dir  # noqa: E402
from benchmark_core import (  # noqa: E402
    SERVER_METRIC_KEYS,
    ClientResult,
    EndOfBotUtterance,
    FailedBotUtterance,
    PerfClient,
    ProtocolClient,
    ProtocolConnectionClosed,
)
from realtime_client import RealtimeClient, RealtimeFrameAdapter  # noqa: E402
from realtime_transport import RealtimeProtocolError, RealtimeTransport  # noqa: E402
from rtvi_client import RTVIClient, categorize_processor  # noqa: E402
from rtvi_transport import RTVITransport  # noqa: E402
from websockets.exceptions import ConnectionClosed  # noqa: E402


class FakeWebSocket:
    def __init__(self, *events: object):
        """Create a fake socket that yields serialized Realtime events."""
        self.events = iter(json.dumps(event) for event in events)
        self.sent: list[dict] = []

    async def recv(self) -> str:
        return next(self.events)

    async def send(self, payload: str) -> None:
        self.sent.append(json.loads(payload))


class RealtimeSocketTests(unittest.IsolatedAsyncioTestCase):
    def realtime_client(self) -> RealtimeClient:
        return RealtimeClient(
            stream_id="client",
            ws_url="wss://example.test/v1/realtime",
            logger=AsyncMock(),
            connect_timeout=1,
        )

    def perf_client(self) -> PerfClient:
        return PerfClient(
            stream_id="client",
            protocol_client=self.realtime_client(),
            audio_files=[],
            start_delay=0,
            metrics_start_time=None,
            session_end_time=None,
            test_duration=1,
            reverse_barge_in_threshold=0.4,
            audio_output_path=None,
            logger=AsyncMock(),
            drain_bot_intro=False,
        )

    async def test_recv_event_rejects_non_object_json(self) -> None:
        socket = RealtimeTransport("wss://example.test")
        socket.ws = FakeWebSocket(["not", "an", "object"])

        with self.assertRaisesRegex(RealtimeProtocolError, "expected a JSON object"):
            await socket.recv_event()

    async def test_rejects_api_key_over_unencrypted_websocket(self) -> None:
        socket = RealtimeTransport("ws://example.test", api_key="secret")

        with (
            patch("realtime_transport.websockets.connect", new_callable=AsyncMock) as connect,
            self.assertRaisesRegex(RuntimeError, "unencrypted WebSocket"),
        ):
            await socket.connect()

        connect.assert_not_awaited()

    async def test_configure_waits_for_session_updated(self) -> None:
        socket = RealtimeTransport("wss://example.test")
        websocket = FakeWebSocket(
            {"type": "session.created", "session": {}},
            {"type": "session.updated", "session": {"audio": {"output": {"format": {"rate": 16_000}}}}},
        )
        socket.ws = websocket

        event = await socket.configure_input(24_000)

        self.assertEqual(event["type"], "session.updated")
        self.assertEqual(socket.output_sample_rate, 16_000)
        self.assertEqual(websocket.sent[0]["type"], "session.update")

    async def test_item_failure_details_are_retained_for_turn_metrics(self) -> None:
        socket = RealtimeTransport("wss://example.test")
        socket.ws = FakeWebSocket(
            {
                "type": "conversation.item.input_audio_transcription.failed",
                "error": {"message": "transcription unavailable"},
            }
        )

        await socket.recv_event()

        self.assertEqual(socket.events[0]["error"], "transcription unavailable")

    async def test_noncompleted_response_is_terminal_turn_failure(self) -> None:
        socket = RealtimeTransport("wss://example.test")
        socket.ws = FakeWebSocket(
            {
                "type": "response.done",
                "response": {
                    "status": "failed",
                    "status_details": {"error": {"message": "generation failed"}},
                },
            }
        )

        adapter = RealtimeFrameAdapter(socket)
        with self.assertRaises(FailedBotUtterance) as raised:
            await adapter.recv_audio()

        self.assertTrue(raised.exception.terminal)
        self.assertIn("generation failed", str(raised.exception))

    async def test_realtime_disconnect_is_not_response_completion(self) -> None:
        socket = RealtimeTransport("wss://example.test")
        socket.recv_event = AsyncMock(side_effect=ConnectionClosed(None, None))
        adapter = RealtimeFrameAdapter(socket)

        with self.assertRaises(ProtocolConnectionClosed):
            await adapter.recv_audio()

    async def test_realtime_session_failure_terminates_without_turn_recovery(self) -> None:
        socket = RealtimeTransport("wss://example.test")
        socket.ws = FakeWebSocket(
            {
                "type": "session.update.failed",
                "error": {"message": "session configuration rejected"},
            }
        )
        adapter = RealtimeFrameAdapter(socket)

        with self.assertRaisesRegex(RealtimeProtocolError, "session configuration rejected"):
            await adapter.recv_audio()

    async def test_response_usage_and_metadata_are_retained(self) -> None:
        socket = RealtimeTransport("wss://example.test")
        socket.ws = FakeWebSocket(
            {
                "type": "response.done",
                "response": {
                    "status": "completed",
                    "usage": {"total_tokens": 3, "input_tokens": 2, "output_tokens": 1},
                    "metadata": {"nvidia_metrics": '{"asr_ttfb":0.4}'},
                },
            }
        )

        adapter = RealtimeFrameAdapter(socket)
        with self.assertRaises(EndOfBotUtterance):
            await adapter.recv_audio()

        self.assertEqual(socket.events[0]["usage"]["total_tokens"], 3)
        self.assertEqual(socket.events[0]["metadata"]["nvidia_metrics"], '{"asr_ttfb":0.4}')

    def test_realtime_client_extracts_nvidia_response_metadata(self) -> None:
        client = self.realtime_client()
        client.collecting_metrics = True

        metrics = client.process_response_metrics(
            {
                "metadata": {
                    "nvidia_metrics": (
                        '{"asr_ttfb":[0.4,0.6],"llm_ttft":[0.2],"tts_ttfb":[0.1],'
                        '"server_e2e":[0.9],"vad_smart_turn":[0.5],'
                        '"llm_processing_time":[0.6],"llm_tokens_per_sec":[20.0]}'
                    )
                }
            }
        )

        self.assertEqual(metrics["llm_ttft"], [0.2])
        self.assertEqual(client.server_metric_samples["asr_ttfb"], [0.4, 0.6])
        self.assertEqual(client.server_metric_samples["llm_tokens_per_sec"], [20.0])

    def test_realtime_client_drains_welcome_by_default(self) -> None:
        protocol = self.realtime_client()
        client = PerfClient(
            stream_id="client",
            protocol_client=protocol,
            audio_files=[],
            start_delay=0,
            metrics_start_time=None,
            session_end_time=None,
            test_duration=1,
            reverse_barge_in_threshold=0.4,
            audio_output_path=None,
            logger=AsyncMock(),
        )

        self.assertTrue(client.drain_bot_intro)
        self.assertIsInstance(protocol, ProtocolClient)

    async def test_realtime_dataset_validation_returns_failed_result(self) -> None:
        client = PerfClient(
            stream_id="client",
            protocol_client=self.realtime_client(),
            audio_files=[],
            start_delay=0,
            metrics_start_time=None,
            session_end_time=None,
            test_duration=1,
            reverse_barge_in_threshold=0.4,
            audio_output_path=None,
            logger=AsyncMock(),
        )

        result = await client.run()

        self.assertIn("Realtime requires one input sample rate", result.error or "")
        self.assertEqual(result.num_turns, 0)

    def test_processor_classifier_prefers_stt_over_overlapping_tts(self) -> None:
        self.assertEqual(categorize_processor("NvidiaSTTService#1"), "asr")

    def test_aggregate_metadata_count_is_bounded(self) -> None:
        client = self.realtime_client()
        client.collecting_metrics = True

        metrics = client.process_response_metrics(
            {"metadata": {"nvidia_metrics": '{"asr_ttfb":{"average":0.4,"count":1000000000}}'}}
        )

        self.assertEqual(metrics, {})
        self.assertEqual(client.server_metric_samples["asr_ttfb"], [])

    def test_raw_metadata_array_count_is_bounded(self) -> None:
        client = self.realtime_client()
        client.collecting_metrics = True
        encoded = json.dumps({"asr_ttfb": [0.4] * 4097})

        metrics = client.process_response_metrics({"metadata": {"nvidia_metrics": encoded}})

        self.assertEqual(metrics, {})
        self.assertEqual(client.server_metric_samples["asr_ttfb"], [])

    def test_non_finite_metadata_metrics_are_rejected(self) -> None:
        client = self.realtime_client()
        client.collecting_metrics = True
        encoded = json.dumps(
            {
                "asr_ttfb": [float("nan"), float("inf"), 0.4],
                "llm_ttft": {"average": float("-inf"), "count": 2},
            }
        )

        metrics = client.process_response_metrics({"metadata": {"nvidia_metrics": encoded}})

        self.assertEqual(metrics, {"asr_ttfb": [0.4]})
        self.assertEqual(client.server_metric_samples["llm_ttft"], [])

    async def test_rtvi_client_parses_symmetric_server_metrics(self) -> None:
        client = RTVIClient(
            stream_id="client",
            host="example.test",
            port=443,
            logger=AsyncMock(),
            connect_timeout=1,
        )
        client.set_collecting_metrics(True)

        await client.process_server_message(
            {
                "type": "metrics",
                "data": {
                    "ttfb": [
                        {"processor": "NvidiaSTTService#1", "value": 0.4},
                        {"processor": "LLMService#1", "value": 0.2},
                        {"processor": "TTSService#1", "value": 0.1},
                    ]
                },
            }
        )

        self.assertIsInstance(client, ProtocolClient)
        self.assertEqual(client.server_metric_samples["asr_ttfb"], [0.4])
        self.assertEqual(client.server_metric_samples["llm_ttft"], [0.2])
        self.assertEqual(client.server_metric_samples["tts_ttfb"], [0.1])
        self.assertEqual(len(client.rtvi_messages), 1)

    async def test_protocol_clients_populate_the_same_metric_schema(self) -> None:
        realtime = self.realtime_client()
        realtime.set_collecting_metrics(True)
        realtime.process_response_metrics(
            {"metadata": {"nvidia_metrics": '{"asr_ttfb":[0.4],"llm_ttft":[0.2],"tts_ttfb":[0.1]}'}}
        )
        rtvi = RTVIClient(
            stream_id="client",
            host="example.test",
            port=443,
            logger=AsyncMock(),
            connect_timeout=1,
        )
        rtvi.set_collecting_metrics(True)
        await rtvi.process_server_message(
            {
                "type": "metrics",
                "data": {
                    "ttfb": [
                        {"processor": "NvidiaSTTService#1", "value": 0.4},
                        {"processor": "LLMService#1", "value": 0.2},
                        {"processor": "TTSService#1", "value": 0.1},
                    ]
                },
            }
        )

        self.assertEqual(rtvi.server_metric_samples, realtime.server_metric_samples)

    async def test_rtvi_throughput_pairs_usage_with_final_processing_metric(self) -> None:
        client = RTVIClient(
            stream_id="client",
            host="example.test",
            port=443,
            logger=AsyncMock(),
            connect_timeout=1,
        )
        client.set_collecting_metrics(True)
        messages = (
            {"type": "metrics", "data": {"processing": [{"processor": "LLMService", "value": 0.1}]}},
            {"type": "metrics", "data": {"tokens": [{"completion_tokens": 10}]}},
            {"type": "metrics", "data": {"processing": [{"processor": "LLMService", "value": 0.5}]}},
        )
        for message in messages:
            await client.process_server_message(message)
        self.assertEqual(client.server_metric_samples["llm_tokens_per_sec"], [20.0])

    async def test_rtvi_throughput_pairs_tokens_from_the_same_message(self) -> None:
        client = RTVIClient(
            stream_id="client",
            host="example.test",
            port=443,
            logger=AsyncMock(),
            connect_timeout=1,
        )
        client.set_collecting_metrics(True)

        await client.process_server_message(
            {
                "type": "metrics",
                "data": {
                    "tokens": [{"completion_tokens": 10}],
                    "processing": [{"processor": "LLMService", "value": 0.5}],
                },
            }
        )

        self.assertEqual(client.server_metric_samples["llm_tokens_per_sec"], [20.0])

    def test_rtvi_tls_verification_is_configurable(self) -> None:
        client = RTVIClient(
            stream_id="client",
            host="example.test",
            port=443,
            logger=AsyncMock(),
            connect_timeout=1,
            verify_tls=False,
        )

        self.assertFalse(client.transport.verify_tls)

    def test_rtvi_protocol_rejects_realtime_url(self) -> None:
        with self.assertRaisesRegex(ValueError, "conflicts"):
            resolve_protocol(
                protocol="rtvi",
                ws_url="wss://example.test/v1/realtime",
                explicit_ws_url="wss://example.test/v1/realtime",
            )

    def test_rtvi_protocol_ignores_realtime_url_from_environment(self) -> None:
        self.assertEqual(
            resolve_protocol(protocol="rtvi", ws_url="wss://example.test/v1/realtime"),
            "rtvi",
        )

    async def test_realtime_cancellation_preserves_connection_closed_classification(self) -> None:
        client = self.realtime_client()
        client.realtime = AsyncMock()
        client.realtime.send_event.side_effect = ConnectionClosed(None, None)

        with self.assertRaises(ProtocolConnectionClosed):
            await client._cancel_active_response(1)

    async def test_session_failure_cancels_in_progress_audio_sender(self) -> None:
        client = self.perf_client()
        sender_started = asyncio.Event()
        sender_cancelled = asyncio.Event()

        async def blocked_sender(_audio_file: Path) -> None:
            sender_started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                sender_cancelled.set()
                raise

        client._send_audio_file = blocked_sender
        client.protocol_client.recv_audio = AsyncMock(side_effect=ProtocolConnectionClosed)

        with self.assertRaises(ProtocolConnectionClosed):
            await client._process_conversation_turn(Path("input.wav"), None)

        self.assertTrue(sender_started.is_set())
        self.assertTrue(sender_cancelled.is_set())

    async def test_inner_timeout_is_not_labeled_as_hard_deadline(self) -> None:
        client = self.perf_client()
        client._run_connected_session = AsyncMock(side_effect=TimeoutError("inner operation"))

        with self.assertRaisesRegex(RuntimeError, "session operation timed out"):
            await client._run_with_hard_deadline(1)

    async def test_outer_timeout_is_labeled_as_hard_deadline(self) -> None:
        client = self.perf_client()

        async def slow_session() -> None:
            await asyncio.sleep(1)

        client._run_connected_session = slow_session
        with self.assertRaisesRegex(RuntimeError, "Hard deadline reached"):
            await client._run_with_hard_deadline(0.001)

    async def test_silence_teardown_does_not_mask_session_error(self) -> None:
        client = self.perf_client()

        async def fail_session(_wf):
            await asyncio.sleep(0)
            raise ValueError("primary session failure")

        async def fail_silence() -> None:
            raise RuntimeError("secondary silence failure")

        client._continuous_audio_loop = fail_session
        client._silence_sender_loop = fail_silence

        with self.assertRaisesRegex(ValueError, "primary session failure"):
            await client._run_connected_session()

    def test_aggregation_handles_missing_stream_id(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            for index in range(2):
                client_dir = run_dir / f"client_unknown_{index}"
                client_dir.mkdir()
                (client_dir / f"result_unknown_{index}.json").write_text(
                    json.dumps({"error": "worker failed", "num_valid_turns": 0}),
                    encoding="utf-8",
                )

            summary_path = aggregate_run_dir(run_dir, num_clients=2)
            summary = json.loads(summary_path.read_text(encoding="utf-8"))

        self.assertEqual(summary["results"]["runtime_error_client_ids"], ["(unknown)", "(unknown)"])
        self.assertEqual(summary["results"]["failed_client_ids"], ["(unknown)"])
        self.assertEqual(summary["results"]["failed_clients"], 2)
        self.assertEqual(summary["results"]["successful_clients"], 0)

    def test_client_result_archive_schema_is_stable(self) -> None:
        names = {field.name for field in fields(ClientResult)}
        self.assertIn("rtvi_messages", names)
        self.assertIn("realtime_turn_metrics", names)
        self.assertEqual(set(SERVER_METRIC_KEYS), set(self.realtime_client().server_metric_samples))

    async def test_rtvi_readiness_failure_closes_connected_socket(self) -> None:
        client = RTVIClient(
            stream_id="client",
            host="example.test",
            port=443,
            logger=AsyncMock(),
            connect_timeout=1,
        )
        websocket = MagicMock()
        websocket.send = AsyncMock(side_effect=RuntimeError("readiness failed"))
        websocket.close = AsyncMock()

        with (
            patch("rtvi_transport.websockets.connect", AsyncMock(return_value=websocket)),
            self.assertRaisesRegex(RuntimeError, "readiness failed"),
        ):
            await client.__aenter__()

        websocket.close.assert_awaited_once()
        self.assertIsNone(client.transport.websocket)
        self.assertIsInstance(client.transport, RTVITransport)


if __name__ == "__main__":
    unittest.main()
