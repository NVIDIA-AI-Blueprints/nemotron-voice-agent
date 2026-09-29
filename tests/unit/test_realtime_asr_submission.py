# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""Deterministic lifecycle tests for turn-scoped Realtime NVIDIA ASR."""

from __future__ import annotations

import asyncio
import threading
import unittest
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

from pipecat.frames.frames import CancelFrame, EndFrame, Frame, InputAudioRawFrame, StartFrame, TranscriptionFrame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.stt_service import STTService
from riva.client.proto import riva_asr_pb2 as rasr

from examples.shared.frames import USER_TRANSCRIPT_TURN_FRAME_ID_METADATA
from realtime.asr import (
    RealtimeASROwnershipError,
    RealtimeNvidiaSTTService,
    _ProviderEvent,
    _RealtimeASRTurn,
    _RealtimeAudioChunk,
    _RealtimeAudioIterator,
    _streaming_requests,
)
from realtime.frames import (
    RealtimeASRTurnEndedFrame,
    RealtimeInputTranscriptionErrorFrame,
    RealtimeManualUserStoppedSpeakingFrame,
)
from realtime.transport import (
    realtime_input_transcription_publication_timeout_secs,
    realtime_input_transcription_timeout_secs,
)
from realtime.turns import RealtimeASRInputSequencer
from realtime.vad import (
    RealtimeVADConfigurationFrame,
    RealtimeVADUserStartedSpeakingFrame,
    RealtimeVADUserStoppedSpeakingFrame,
)


def _turn(
    *,
    token: int = 101,
    request_id: str = "request-101",
    boundary_kind: str = "vad",
) -> _RealtimeASRTurn:
    return _RealtimeASRTurn(
        token=token,
        request_id=request_id,
        chunks=_RealtimeAudioIterator(),
        boundary_kind=boundary_kind,
    )


def _response(request_id: str, text: str, *, final: bool = True) -> SimpleNamespace:
    result = SimpleNamespace(
        is_final=final,
        alternatives=[SimpleNamespace(transcript=text)],
    )
    return SimpleNamespace(id=SimpleNamespace(value=request_id), results=[result])


async def _wait_for_thread_event(event: threading.Event, timeout: float = 1.0) -> bool:
    deadline = asyncio.get_running_loop().time() + timeout
    while not event.is_set() and asyncio.get_running_loop().time() < deadline:
        await asyncio.sleep(0)
    return event.is_set()


class _DirectFrameRecorder(FrameProcessor):
    """Record frames at the real downstream queue boundary."""

    def __init__(self) -> None:
        super().__init__(enable_direct_mode=True)
        self.frames: list[Frame] = []

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        if direction == FrameDirection.DOWNSTREAM:
            self.frames.append(frame)


class RealtimeASRSubmissionTests(unittest.IsolatedAsyncioTestCase):
    """Verify native request, ownership, cancellation, and shutdown invariants."""

    def test_native_provider_deadline_precedes_publication_watchdog(self) -> None:
        """Reserve deterministic time for terminal publication after provider expiry."""
        with patch.dict("os.environ", {"REALTIME_INPUT_TRANSCRIPTION_TIMEOUT_SECONDS": "3.25"}):
            provider_deadline = realtime_input_transcription_timeout_secs()
            publication_deadline = realtime_input_transcription_publication_timeout_secs()
        service = RealtimeNvidiaSTTService(
            server="localhost:50051",
            use_ssl=False,
            turn_drain_timeout_secs=provider_deadline,
        )

        self.assertEqual(service._realtime_turn_drain_timeout_secs, 3.25)
        self.assertLess(service._realtime_turn_drain_timeout_secs, publication_deadline)

    def test_native_requests_carry_request_id_and_end_at_iterator_eof(self) -> None:
        """Put ownership on the config request and terminate on native EOF."""
        chunks = _RealtimeAudioIterator()
        chunks.put(_RealtimeAudioChunk(b"first"))
        chunks.put(_RealtimeAudioChunk(b"last"))
        chunks.close()

        requests = list(
            _streaming_requests(
                chunks,
                rasr.StreamingRecognitionConfig(),
                "turn-request",
            )
        )

        self.assertEqual(requests[0].id.value, "turn-request")
        self.assertTrue(requests[0].HasField("streaming_config"))
        self.assertEqual([request.audio_content for request in requests[1:]], [b"first", b"last"])
        self.assertEqual([dict(request.runtime_config) for request in requests[1:]], [{}, {}])
        self.assertFalse(requests[1].HasField("id"))
        self.assertFalse(requests[2].HasField("id"))
        with self.assertRaises(StopIteration):
            next(chunks)

    def test_pending_audio_budgets_are_independent_and_reusable(self) -> None:
        """Bound resident prefix and stream bytes without capping turn duration."""
        service = RealtimeNvidiaSTTService(server="localhost:50051", use_ssl=False)
        service._sample_rate = 10
        service._audio_channel_count = 1
        turn = _RealtimeASRTurn(
            token=1,
            request_id="bounded",
            chunks=_RealtimeAudioIterator(max_bootstrap_bytes=6, max_stream_pending_bytes=4),
        )

        service._queue_turn_audio(turn, b"123456", bootstrap=True)
        service._queue_turn_audio(turn, b"7890")
        with self.assertRaises(RealtimeASROwnershipError) as raised:
            service._queue_turn_audio(turn, b"xx")
        self.assertEqual(raised.exception.code, "input_buffer_overflow")
        self.assertEqual(turn.audio_bytes, 10)

        self.assertEqual(b"".join(next(turn.chunks).audio for _ in range(5)), b"1234567890")
        service._queue_turn_audio(turn, b"abcd")
        self.assertEqual(turn.audio_bytes, 14)

    async def test_maximum_vad_lookback_is_accepted_by_asr(self) -> None:
        """Accept a full negotiated prefix plus its exact detector confirmation."""
        service = RealtimeNvidiaSTTService(server="localhost:50051", use_ssl=False)
        service._sample_rate = 16_000
        frame = RealtimeVADConfigurationFrame(
            prefix_padding_samples=960_000,
            start_confirmation_samples=3_072,
            sample_rate=16_000,
        )

        with patch.object(STTService, "process_frame", AsyncMock()):
            await service.process_frame(frame, FrameDirection.DOWNSTREAM)

        self.assertEqual(service._realtime_prefix_samples, 963_072)

    async def test_direct_sequencer_and_asr_publish_boundaries_before_native_lifecycle(self) -> None:
        """Prevent audio/start/stop overtaking with the real direct queue path."""
        sequencer = RealtimeASRInputSequencer()
        service = RealtimeNvidiaSTTService(
            server="localhost:50051",
            use_ssl=False,
            audio_passthrough=True,
        )
        recorder = _DirectFrameRecorder()
        sequencer.link(service)
        service.link(recorder)
        service._sample_rate = 16_000
        service._audio_channel_count = 1
        start = RealtimeVADUserStartedSpeakingFrame(audio_start_sample=0, sample_rate=16_000)
        audio = InputAudioRawFrame(audio=b"\x01\x00" * 160, sample_rate=16_000, num_channels=1)
        stop = RealtimeVADUserStoppedSpeakingFrame(audio_stop_sample=160, sample_rate=16_000)
        launched: list[_RealtimeASRTurn] = []

        def launch_after_start(turn: _RealtimeASRTurn) -> None:
            self.assertIs(recorder.frames[-1], start)
            launched.append(turn)

        original_close = service._close_active_turn

        def close_after_stop(**kwargs):
            self.assertIs(recorder.frames[-1], stop)
            return original_close(**kwargs)

        def create_task(coroutine, name=None):
            return asyncio.create_task(coroutine, name=name)

        with (
            patch.object(service, "start", AsyncMock()),
            patch.object(service, "_launch_turn", side_effect=launch_after_start),
            patch.object(service, "_close_active_turn", side_effect=close_after_stop),
            patch.object(service, "create_task", side_effect=create_task),
        ):
            await sequencer.queue_frame(StartFrame(audio_in_sample_rate=16_000))
            await sequencer.queue_frame(start)
            await sequencer.queue_frame(audio)
            await sequencer.queue_frame(stop)

        observed_input = [
            frame
            for frame in recorder.frames
            if isinstance(
                frame, (RealtimeVADUserStartedSpeakingFrame, InputAudioRawFrame, RealtimeVADUserStoppedSpeakingFrame)
            )
        ]
        self.assertEqual(observed_input, [start, audio, stop])
        self.assertEqual(len(launched), 1)
        turn = launched[0]
        requests = list(_streaming_requests(turn.chunks, rasr.StreamingRecognitionConfig(), turn.request_id))[1:]
        self.assertEqual(b"".join(request.audio_content for request in requests), audio.audio)
        turn.done.set()
        assert turn.deadline_task is not None
        await turn.deadline_task

    async def test_absolute_boundary_selects_exact_preroll_when_asr_cursor_is_ahead(self) -> None:
        """Slice retained PCM by absolute sample position, not receipt timing."""
        service = RealtimeNvidiaSTTService(
            server="localhost:50051",
            use_ssl=False,
            vad_prefix_padding_secs=0.8,
        )
        service._sample_rate = 100
        service._audio_channel_count = 1
        service._realtime_prefix_samples = 80
        source = b"".join(sample.to_bytes(2, "little") for sample in range(100))

        with patch.object(service, "start_processing_metrics", AsyncMock()):
            self.assertEqual([frame async for frame in service.run_stt(source)], [None])

        async def no_op(_turn: _RealtimeASRTurn) -> None:
            return None

        def create_task(coroutine, name=None):
            return asyncio.create_task(coroutine, name=name)

        boundary = RealtimeVADUserStartedSpeakingFrame(audio_start_sample=40, sample_rate=100)
        with (
            patch.object(service, "_publish_provider_events", side_effect=no_op),
            patch.object(service, "_run_turn_worker", side_effect=no_op),
            patch.object(service, "create_task", side_effect=create_task),
        ):
            created = await service._start_turn(boundary)
            assert created is not None
            service._launch_turn(created)

        turn = service._realtime_active_turn
        self.assertIsNotNone(turn)
        assert turn is not None
        await asyncio.gather(turn.publisher_task, turn.worker_task)
        expected = b"".join(sample.to_bytes(2, "little") for sample in range(40, 100))
        turn.chunks.close()
        queued = b"".join(chunk.audio for chunk in turn.chunks)
        self.assertEqual(service._realtime_audio_cursor, 100)
        self.assertEqual(queued, expected)
        self.assertEqual(turn.token, boundary.id)

    async def test_only_provider_terminal_marks_last_finalized_with_immutable_owner(self) -> None:
        """Do not mistake a segment final for the end of the native turn."""
        service = RealtimeNvidiaSTTService(
            server="localhost:50051",
            use_ssl=False,
        )
        completed = _turn(token=202, request_id="completed-request")
        service._realtime_active_turn = _turn(token=303, request_id="new-request")
        pushed: list[Any] = []

        async def record(frame, *_args, **_kwargs) -> None:
            pushed.append(frame)

        with (
            patch.object(service, "push_frame", side_effect=record),
            patch.object(service, "stop_processing_metrics", AsyncMock()),
            patch.object(service, "emit_stt_usage_metrics", AsyncMock()),
            patch.object(service, "_handle_transcription", AsyncMock()),
        ):
            await service._handle_turn_response(
                completed,
                _response(completed.request_id, "Hi, "),
            )
            await service._handle_turn_response(
                completed,
                _response(completed.request_id, "I want to book a flight."),
            )
            self.assertEqual([frame.text for frame in pushed], ["Hi, "])
            self.assertFalse(pushed[0].finalized)
            await service._finish_turn(completed)

        self.assertEqual([frame.text for frame in pushed], ["Hi, ", "I want to book a flight."])
        self.assertTrue(all(isinstance(frame, TranscriptionFrame) for frame in pushed))
        self.assertEqual([frame.finalized for frame in pushed], [False, True])
        self.assertEqual(
            [frame.metadata[USER_TRANSCRIPT_TURN_FRAME_ID_METADATA] for frame in pushed],
            [202, 202],
        )

    async def test_provider_terminal_orders_final_transcript_before_turn_terminal(self) -> None:
        """Publish the final text before releasing the same owner upstream."""
        service = RealtimeNvidiaSTTService(
            server="localhost:50051",
            use_ssl=False,
        )
        service._sample_rate = 16_000
        service._audio_channel_count = 1
        turn = _turn()
        turn.stop_token = 102
        turn.audio_bytes = 3_200
        pushed: list[tuple[Any, FrameDirection]] = []

        async def record(frame, direction=FrameDirection.DOWNSTREAM) -> None:
            pushed.append((frame, direction))

        with (
            patch.object(service, "push_frame", side_effect=record),
            patch.object(service, "stop_processing_metrics", AsyncMock()) as stop_metrics,
            patch.object(service, "emit_stt_usage_metrics", AsyncMock()) as usage_metrics,
            patch.object(service, "_handle_transcription", AsyncMock()),
        ):
            service._enqueue_provider_event(
                turn,
                _ProviderEvent("response", _response(turn.request_id, "complete thought")),
            )
            service._enqueue_provider_event(turn, _ProviderEvent("completed"))
            await service._publish_provider_events(turn)
            await service._emit_turn_terminal(turn, status="completed")

        self.assertEqual(len(pushed), 3)
        transcript = pushed[0][0]
        self.assertIsInstance(transcript, TranscriptionFrame)
        self.assertTrue(transcript.finalized)
        terminals = [frame for frame, _ in pushed[1:]]
        self.assertTrue(all(isinstance(frame, RealtimeASRTurnEndedFrame) for frame in terminals))
        self.assertIs(terminals[0], terminals[1])
        self.assertEqual(
            [direction for _, direction in pushed],
            [FrameDirection.DOWNSTREAM] * 2 + [FrameDirection.UPSTREAM],
        )
        self.assertIsNone(turn.pending_final)
        self.assertEqual(service._stt_usage_pending_seconds, 0.1)
        stop_metrics.assert_awaited_once()
        usage_metrics.assert_awaited_once()

    async def test_failures_preserve_final_response_already_accepted_from_provider(self) -> None:
        """Publish accepted final text before timeout and overflow terminals."""
        for failure, expected_code in (
            ("timeout", "asr_provider_timeout"),
            ("overflow", "asr_provider_backpressure"),
        ):
            with self.subTest(failure=failure):
                service = RealtimeNvidiaSTTService(
                    server="localhost:50051",
                    use_ssl=False,
                    max_pending_provider_events=1,
                )
                service._sample_rate = 16_000
                service._audio_channel_count = 1
                turn = _turn()
                turn.stop_token = 102
                turn.audio_bytes = 3_200
                pushed: list[tuple[Any, FrameDirection]] = []

                async def record(
                    frame,
                    direction=FrameDirection.DOWNSTREAM,
                    _pushed=pushed,
                ) -> None:
                    _pushed.append((frame, direction))

                with (
                    patch.object(service, "push_frame", side_effect=record),
                    patch.object(service, "stop_processing_metrics", AsyncMock()) as stop_metrics,
                    patch.object(service, "emit_stt_usage_metrics", AsyncMock()) as usage_metrics,
                    patch.object(service, "_handle_transcription", AsyncMock()),
                ):
                    service._enqueue_provider_event(
                        turn,
                        _ProviderEvent("response", _response(turn.request_id, "usable final")),
                    )
                    if failure == "timeout":
                        service._queue_turn_failure(
                            turn,
                            code=expected_code,
                            message="provider timed out after its final response",
                        )
                    else:
                        service._enqueue_provider_event(
                            turn,
                            _ProviderEvent("response", _response(turn.request_id, "over capacity")),
                        )
                    await service._publish_provider_events(turn)

                self.assertEqual([event.kind for event in turn.provider_events], [])
                self.assertTrue(turn.discard_provider_output)
                transcript = pushed[0][0]
                self.assertIsInstance(transcript, TranscriptionFrame)
                self.assertEqual(transcript.text, "usable final")
                self.assertTrue(transcript.finalized)
                terminals = [frame for frame, _ in pushed[1:]]
                self.assertTrue(all(isinstance(frame, RealtimeASRTurnEndedFrame) for frame in terminals))
                self.assertIs(terminals[0], terminals[1])
                self.assertEqual(terminals[0].status, "failed")
                self.assertEqual(terminals[0].code, expected_code)
                self.assertEqual(
                    [direction for _, direction in pushed],
                    [FrameDirection.DOWNSTREAM] * 2 + [FrameDirection.UPSTREAM],
                )
                self.assertIsNone(turn.pending_final)
                stop_metrics.assert_awaited_once()
                usage_metrics.assert_awaited_once()

    async def test_explicit_cancellation_discards_accepted_provider_response(self) -> None:
        """Keep shutdown cancellation distinct from failure-result draining."""
        service = RealtimeNvidiaSTTService(server="localhost:50051", use_ssl=False)
        turn = _turn()
        turn.stop_token = 102
        service._enqueue_provider_event(
            turn,
            _ProviderEvent("response", _response(turn.request_id, "late text")),
        )
        service._request_turn_cancel(turn, suppress_errors=True)
        service._enqueue_provider_event(turn, _ProviderEvent("cancelled"))

        with (
            patch.object(service, "push_frame", AsyncMock()) as push_frame,
            patch.object(service, "stop_processing_metrics", AsyncMock()),
            patch.object(service, "emit_stt_usage_metrics", AsyncMock()),
        ):
            await service._publish_provider_events(turn)

        push_frame.assert_not_awaited()
        self.assertTrue(turn.discard_provider_output)
        self.assertTrue(turn.retired)

    async def test_explicit_cancellation_discards_staged_final_response(self) -> None:
        """Do not flush a pending final while an explicit shutdown wins."""
        service = RealtimeNvidiaSTTService(server="localhost:50051", use_ssl=False)
        turn = _turn()
        turn.stop_token = 102
        await service._handle_turn_response(
            turn,
            _response(turn.request_id, "staged final"),
        )
        service._request_turn_cancel(turn, suppress_errors=True)
        service._enqueue_provider_event(
            turn,
            _ProviderEvent("error", ("asr_provider_error", "late provider failure")),
        )

        with (
            patch.object(service, "push_frame", AsyncMock()) as push_frame,
            patch.object(service, "stop_processing_metrics", AsyncMock()),
            patch.object(service, "emit_stt_usage_metrics", AsyncMock()),
        ):
            await service._publish_provider_events(turn)

        push_frame.assert_not_awaited()
        self.assertIsNone(turn.pending_final)
        self.assertTrue(turn.retired)

    async def test_idle_audio_is_not_reported_as_native_asr_usage(self) -> None:
        """Exclude prefix and trailing audio that never enters a native turn."""
        service = RealtimeNvidiaSTTService(server="localhost:50051", use_ssl=False)
        service._sample_rate = 16_000
        idle_audio = b"\x00\x00" * 16_000

        service._record_stt_audio_usage(idle_audio)
        with patch.object(service, "start_stt_usage_metrics", AsyncMock()) as start_usage:
            await service.emit_stt_usage_metrics()

        self.assertEqual(service._stt_usage_pending_seconds, 0.0)
        start_usage.assert_not_awaited()

    def test_first_provider_terminal_intent_wins(self) -> None:
        """Drop every response and terminal intent after the first terminal."""
        service = RealtimeNvidiaSTTService(server="localhost:50051", use_ssl=False)
        turn = _turn()

        service._enqueue_provider_event(turn, _ProviderEvent("completed"))
        service._enqueue_provider_event(turn, _ProviderEvent("error", ("late", "late")))
        service._enqueue_provider_event(turn, _ProviderEvent("response", _response(turn.request_id, "late")))

        self.assertTrue(turn.provider_terminal_queued)
        self.assertEqual([event.kind for event in turn.provider_events], ["completed"])

    async def test_failed_then_successful_turns_close_fresh_exact_metrics(self) -> None:
        """Close each turn once and account only for that turn's real PCM."""
        service = RealtimeNvidiaSTTService(server="localhost:50051", use_ssl=False)
        service._sample_rate = 16_000
        service._audio_channel_count = 1
        usage_seconds: list[float] = []
        pushed: list[Any] = []

        async def record_usage() -> None:
            usage_seconds.append(service._stt_usage_pending_seconds)

        async def record(frame, *_args, **_kwargs) -> None:
            pushed.append(frame)

        def create_task(coroutine, name=None):
            return asyncio.create_task(coroutine, name=name)

        async def run_turn(source: bytes, transcript: str | None) -> _RealtimeASRTurn:
            start_sample = service._realtime_audio_cursor
            start = RealtimeVADUserStartedSpeakingFrame(
                audio_start_sample=start_sample,
                sample_rate=16_000,
            )
            turn = await service._start_turn(start)
            assert turn is not None
            service._queue_turn_audio(turn, source)
            service._realtime_audio_cursor += len(source) // 2
            stop = RealtimeVADUserStoppedSpeakingFrame(
                audio_stop_sample=service._realtime_audio_cursor,
                sample_rate=16_000,
            )
            service._close_active_turn(frame=stop, required=True)
            requests = list(_streaming_requests(turn.chunks, rasr.StreamingRecognitionConfig(), turn.request_id))[1:]
            self.assertEqual(b"".join(request.audio_content for request in requests), source)
            if transcript is not None:
                service._enqueue_provider_event(
                    turn,
                    _ProviderEvent("response", _response(turn.request_id, transcript)),
                )
            service._enqueue_provider_event(turn, _ProviderEvent("completed"))
            await service._publish_provider_events(turn)
            assert turn.deadline_task is not None
            await asyncio.gather(turn.deadline_task, return_exceptions=True)
            return turn

        with (
            patch.object(service, "create_task", side_effect=create_task),
            patch.object(service, "start_processing_metrics", AsyncMock()) as start_metrics,
            patch.object(service, "stop_processing_metrics", AsyncMock()) as stop_metrics,
            patch.object(service, "emit_stt_usage_metrics", side_effect=record_usage) as usage_metrics,
            patch.object(service, "push_frame", side_effect=record),
            patch.object(service, "_handle_transcription", AsyncMock()),
        ):
            failed = await run_turn(b"\x01\x00" * 1_600, None)
            succeeded = await run_turn(b"\x02\x00" * 3_200, "next turn works")

        self.assertEqual((failed.audio_bytes, succeeded.audio_bytes), (3_200, 6_400))
        self.assertEqual(usage_seconds, [0.1, 0.2])
        self.assertEqual(start_metrics.await_count, 2)
        self.assertEqual(stop_metrics.await_count, 2)
        self.assertEqual(usage_metrics.await_count, 2)
        terminals = [frame for frame in pushed if isinstance(frame, RealtimeASRTurnEndedFrame)]
        self.assertEqual([frame.status for frame in terminals[::2]], ["empty", "completed"])
        self.assertIsNone(terminals[0].code)

    async def test_second_turn_prefix_uses_post_commit_audio_coordinates(self) -> None:
        """Retain only audio written after the prior turn closed."""
        service = RealtimeNvidiaSTTService(
            server="localhost:50051",
            use_ssl=False,
            vad_prefix_padding_secs=0.2,
        )
        service._sample_rate = 100
        service._audio_channel_count = 1
        service._realtime_prefix_samples = 20
        first = _turn(request_id="first")
        service._realtime_active_turn = first
        service._realtime_turns[first.request_id] = first
        service._realtime_audio_cursor = 30

        def create_task(coroutine, name=None):
            return asyncio.create_task(coroutine, name=name)

        with patch.object(service, "create_task", side_effect=create_task):
            service._close_active_turn(
                frame=RealtimeVADUserStoppedSpeakingFrame(audio_stop_sample=30, sample_rate=100),
                required=True,
            )
        first.done.set()
        assert first.deadline_task is not None
        await first.deadline_task

        source = b"".join(sample.to_bytes(2, "little") for sample in range(30, 40))
        with patch.object(service, "start_processing_metrics", AsyncMock()):
            self.assertEqual([frame async for frame in service.run_stt(source)], [None])

        async def no_op(_turn: _RealtimeASRTurn) -> None:
            return None

        boundary = RealtimeVADUserStartedSpeakingFrame(audio_start_sample=35, sample_rate=100)
        with (
            patch.object(service, "_publish_provider_events", side_effect=no_op),
            patch.object(service, "_run_turn_worker", side_effect=no_op),
            patch.object(service, "create_task", side_effect=create_task),
        ):
            created = await service._start_turn(boundary)
            assert created is not None
            service._launch_turn(created)

        second = service._realtime_active_turn
        assert second is not None
        await asyncio.gather(second.publisher_task, second.worker_task)
        second.chunks.close()
        self.assertEqual(b"".join(chunk.audio for chunk in second.chunks), source[10:])

    async def test_manual_audio_is_losslessly_split_into_streaming_requests(self) -> None:
        """Bound a full manual commit without changing any PCM bytes."""
        service = RealtimeNvidiaSTTService(server="localhost:50051", use_ssl=False)
        service._sample_rate = 100
        service._audio_channel_count = 1
        source = b"".join(sample.to_bytes(2, "little") for sample in range(25))
        turn = _turn(boundary_kind="manual")
        service._realtime_active_turn = turn
        service._realtime_turns[turn.request_id] = turn
        service._queue_turn_audio(turn, source)
        service._realtime_audio_cursor = 25

        def create_task(coroutine, name=None):
            return asyncio.create_task(coroutine, name=name)

        with patch.object(service, "create_task", side_effect=create_task):
            service._close_active_turn(
                frame=RealtimeManualUserStoppedSpeakingFrame(
                    audio_sample_cursor=25,
                    sample_rate=100,
                ),
                required=True,
            )
        requests = list(_streaming_requests(turn.chunks, rasr.StreamingRecognitionConfig(), turn.request_id))[1:]
        self.assertEqual([len(request.audio_content) for request in requests], [20, 20, 10])
        self.assertTrue(all(len(request.audio_content) <= 20 for request in requests))
        self.assertEqual(b"".join(request.audio_content for request in requests), source)
        self.assertEqual([dict(request.runtime_config) for request in requests], [{}, {}, {}])
        turn.done.set()
        assert turn.deadline_task is not None
        await turn.deadline_task

    async def test_stop_closes_request_without_waiting_for_provider_drain(self) -> None:
        """Close the native request at EOF without changing the accepted PCM."""
        service = RealtimeNvidiaSTTService(server="localhost:50051", use_ssl=False)
        service._sample_rate = 100
        service._audio_channel_count = 1
        turn = _turn()
        source = b"".join(sample.to_bytes(2, "little") for sample in range(5))
        service._queue_turn_audio(turn, source)
        service._realtime_audio_cursor = 5
        service._realtime_active_turn = turn
        service._realtime_turns[turn.request_id] = turn

        def create_task(coroutine, name=None):
            return asyncio.create_task(coroutine, name=name)

        with patch.object(service, "create_task", side_effect=create_task):
            stopped = service._close_active_turn(
                frame=RealtimeVADUserStoppedSpeakingFrame(audio_stop_sample=5, sample_rate=100),
                required=True,
            )

        self.assertIs(stopped, turn)
        self.assertTrue(turn.request_closed)
        self.assertFalse(turn.done.is_set())
        requests = list(_streaming_requests(turn.chunks, rasr.StreamingRecognitionConfig(), turn.request_id))[1:]
        self.assertEqual(b"".join(request.audio_content for request in requests), source)
        self.assertTrue(all(dict(request.runtime_config) == {} for request in requests))
        turn.done.set()
        assert turn.deadline_task is not None
        await turn.deadline_task

    async def test_shutdown_during_call_assignment_waits_for_real_worker_exit(self) -> None:
        """Wait for the synchronous worker when cancellation races call creation."""
        service = RealtimeNvidiaSTTService(
            server="localhost:50051",
            use_ssl=False,
            worker_cancel_timeout_secs=1.0,
        )
        call = MagicMock()
        entered = threading.Event()
        release = threading.Event()

        def assign_call(*_args, **_kwargs):
            entered.set()
            release.wait(timeout=2)
            return call

        service._asr_service = SimpleNamespace(
            stub=SimpleNamespace(StreamingRecognize=assign_call),
            auth=SimpleNamespace(get_auth_metadata=MagicMock(return_value=[])),
        )
        service._config = rasr.StreamingRecognitionConfig()
        turn = _turn(token=404, request_id="assignment-race")
        service._realtime_active_turn = turn
        service._realtime_turns[turn.request_id] = turn
        with patch.object(service, "get_event_loop", return_value=asyncio.get_running_loop()):
            turn.worker_task = asyncio.create_task(service._run_turn_worker(turn))
            self.assertTrue(await _wait_for_thread_event(entered))
            shutdown = asyncio.create_task(service._shutdown_turns())
            await asyncio.sleep(0)
            self.assertTrue(turn.cancel_requested)
            self.assertFalse(turn.worker_exited.is_set())
            self.assertFalse(shutdown.done())

            release.set()
            await asyncio.wait_for(shutdown, timeout=2)

        self.assertTrue(turn.worker_exited.is_set())
        self.assertTrue(turn.done.is_set())
        self.assertTrue(turn.worker_task.done())
        call.cancel.assert_called_once_with()

    async def test_cancellation_bypasses_stalled_downstream_publication(self) -> None:
        """Cancel the native call even while frame publication is blocked."""
        service = RealtimeNvidiaSTTService(server="localhost:50051", use_ssl=False)
        call = MagicMock()
        turn = _turn(token=505, request_id="stalled-publication")
        turn.call = call
        publication_started = asyncio.Event()
        release_publication = asyncio.Event()

        async def stall(_frame, *_args, **_kwargs) -> None:
            publication_started.set()
            await release_publication.wait()

        with (
            patch.object(service, "push_frame", side_effect=stall),
            patch.object(service, "stop_processing_metrics", AsyncMock()),
            patch.object(service, "emit_stt_usage_metrics", AsyncMock()),
            patch.object(service, "_handle_transcription", AsyncMock()),
        ):
            service._enqueue_provider_event(
                turn,
                _ProviderEvent("response", _response(turn.request_id, "published later")),
            )
            service._enqueue_provider_event(turn, _ProviderEvent("completed"))
            publisher = asyncio.create_task(service._publish_provider_events(turn))
            await publication_started.wait()

            service._request_turn_cancel(turn)
            call.cancel.assert_called_once_with()
            self.assertFalse(publisher.done())

            release_publication.set()
            publisher.cancel()
            await asyncio.gather(publisher, return_exceptions=True)

    async def test_shutdown_closes_metrics_once_and_suppresses_late_failure(self) -> None:
        """Account real PCM once while suppressing provider output after shutdown."""
        service = RealtimeNvidiaSTTService(server="localhost:50051", use_ssl=False)
        service._sample_rate = 16_000
        service._audio_channel_count = 1
        turn = _turn(token=606, request_id="shutdown-request")
        turn.stop_token = 607
        service._queue_turn_audio(turn, b"\x01\x00" * 1_600)
        turn.worker_exited.set()
        service._realtime_active_turn = turn
        service._realtime_turns[turn.request_id] = turn
        usage_seconds: list[float] = []

        async def record_usage() -> None:
            usage_seconds.append(service._stt_usage_pending_seconds)

        with (
            patch.object(service, "push_frame", AsyncMock()) as push_frame,
            patch.object(service, "stop_processing_metrics", AsyncMock()) as stop_metrics,
            patch.object(service, "emit_stt_usage_metrics", side_effect=record_usage) as usage_metrics,
        ):
            await service._shutdown_turns()
            await service._shutdown_turns()
            await service._fail_turn(
                turn,
                code="asr_provider_error",
                message="late provider failure",
            )
            await service._emit_turn_terminal(
                turn,
                status="failed",
                code="asr_provider_error",
                message="late provider failure",
            )
            await service._retire_turn(turn)

        push_frame.assert_not_awaited()
        stop_metrics.assert_awaited_once()
        usage_metrics.assert_awaited_once()
        self.assertEqual(usage_seconds, [0.1])
        self.assertTrue(turn.suppress_errors)
        self.assertTrue(turn.retired)

    async def test_provider_failure_before_stop_closes_exact_metrics_once(self) -> None:
        """Retire a failed open stream without synthetic audio or duplicate metrics."""
        service = RealtimeNvidiaSTTService(server="localhost:50051", use_ssl=False)
        service._sample_rate = 16_000
        service._audio_channel_count = 1
        source = b"\x02\x00" * 3_200
        usage_seconds: list[float] = []
        pushed: list[Frame] = []

        async def record_usage() -> None:
            usage_seconds.append(service._stt_usage_pending_seconds)

        async def record(frame: Frame, *_args, **_kwargs) -> None:
            pushed.append(frame)

        with (
            patch.object(service, "start_processing_metrics", AsyncMock()) as start_metrics,
            patch.object(service, "stop_processing_metrics", AsyncMock()) as stop_metrics,
            patch.object(service, "emit_stt_usage_metrics", side_effect=record_usage) as usage_metrics,
            patch.object(service, "push_frame", side_effect=record),
        ):
            turn = await service._start_turn(
                RealtimeVADUserStartedSpeakingFrame(audio_start_sample=0, sample_rate=16_000)
            )
            assert turn is not None
            service._queue_turn_audio(turn, source)
            service._enqueue_provider_event(
                turn,
                _ProviderEvent("error", ("asr_provider_error", "provider failed before stop")),
            )
            await service._publish_provider_events(turn)
            await service._retire_turn(turn)
            service._enqueue_provider_event(turn, _ProviderEvent("completed"))

        self.assertEqual(turn.audio_bytes, len(source))
        self.assertIsNone(turn.stop_token)
        self.assertTrue(turn.retired)
        self.assertIsNone(service._realtime_active_turn)
        self.assertEqual(usage_seconds, [0.2])
        start_metrics.assert_awaited_once()
        stop_metrics.assert_awaited_once()
        usage_metrics.assert_awaited_once()
        errors = [frame for frame in pushed if isinstance(frame, RealtimeInputTranscriptionErrorFrame)]
        self.assertEqual([frame.code for frame in errors], ["asr_provider_error"])

    async def test_normal_stop_or_cancel_suppresses_running_turn_output(self) -> None:
        """Emit no late terminal or error while End/Cancel retires native work."""
        for method_name, frame in (("stop", EndFrame()), ("cancel", CancelFrame())):
            with self.subTest(method=method_name):
                service = RealtimeNvidiaSTTService(server="localhost:50051", use_ssl=False)
                turn = _turn(token=808, request_id=f"running-{method_name}")
                turn.stop_token = 809
                worker_started = asyncio.Event()
                release_worker = asyncio.Event()

                async def worker(
                    _started=worker_started,
                    _release=release_worker,
                    _turn=turn,
                ) -> None:
                    _started.set()
                    await _release.wait()
                    _turn.worker_exited.set()

                call = MagicMock()
                call.cancel.side_effect = release_worker.set
                turn.call = call
                turn.worker_task = asyncio.create_task(worker())
                turn.publisher_task = asyncio.create_task(service._publish_provider_events(turn))
                service._realtime_active_turn = turn
                service._realtime_turns[turn.request_id] = turn
                await worker_started.wait()

                async def cancel_task(task, *_args, **_kwargs) -> None:
                    task.cancel()
                    await asyncio.gather(task, return_exceptions=True)

                with (
                    patch.object(service, "push_frame", AsyncMock()) as push_frame,
                    patch.object(service, "cancel_task", side_effect=cancel_task),
                    patch.object(STTService, method_name, AsyncMock()),
                ):
                    await getattr(service, method_name)(frame)

                push_frame.assert_not_awaited()
                call.cancel.assert_called_once_with()
                self.assertTrue(turn.suppress_errors)
                self.assertTrue(turn.retired)
                self.assertTrue(turn.worker_task.done())
                self.assertTrue(turn.publisher_task.done())

    async def test_publication_failure_always_retires_turn(self) -> None:
        """Release ownership even when transcript and error publication both fail."""
        service = RealtimeNvidiaSTTService(
            server="localhost:50051",
            use_ssl=False,
        )
        turn = _turn(request_id="publication-failure")
        service._realtime_turns[turn.request_id] = turn
        service._enqueue_provider_event(turn, _ProviderEvent("response", _response(turn.request_id, "text")))
        service._enqueue_provider_event(turn, _ProviderEvent("completed"))
        with (
            patch.object(service, "push_frame", AsyncMock(side_effect=RuntimeError("closed"))),
            patch.object(service, "stop_processing_metrics", AsyncMock()),
            patch.object(service, "emit_stt_usage_metrics", AsyncMock()),
        ):
            await service._publish_provider_events(turn)

        self.assertTrue(turn.retired)
        self.assertTrue(turn.done.is_set())
        self.assertNotIn(turn.request_id, service._realtime_turns)

    async def test_deadline_retires_and_wakes_idle_publisher(self) -> None:
        """Leave no publisher task parked after a timed-out native stream."""
        service = RealtimeNvidiaSTTService(
            server="localhost:50051",
            use_ssl=False,
            turn_drain_timeout_secs=0.001,
            worker_cancel_timeout_secs=0.001,
        )
        turn = _turn(request_id="deadline")
        service._realtime_active_turn = turn
        service._realtime_turns[turn.request_id] = turn
        publisher = asyncio.create_task(service._publish_provider_events(turn))
        turn.publisher_task = publisher
        with patch.object(service, "push_frame", AsyncMock()):
            await service._enforce_turn_deadline(turn)
        await asyncio.wait_for(publisher, timeout=1)

        self.assertTrue(turn.retired)
        self.assertTrue(turn.done.is_set())

    async def test_response_id_mismatch_fails_closed(self) -> None:
        """Reject mismatched provider ownership without leaking transcript text."""
        service = RealtimeNvidiaSTTService(server="localhost:50051", use_ssl=False)
        turn = _turn(token=707, request_id="expected-request")
        turn.call = MagicMock()

        with patch.object(service, "push_frame", AsyncMock()) as push_frame:
            await service._handle_turn_response(
                turn,
                _response("different-request", "must not escape"),
            )
            await service._handle_turn_response(
                turn,
                _response(turn.request_id, "must still not escape"),
            )

        push_frame.assert_awaited_once()
        error = push_frame.await_args.args[0]
        self.assertIsInstance(error, RealtimeInputTranscriptionErrorFrame)
        self.assertEqual(error.code, "asr_turn_owner_mismatch")
        self.assertTrue(turn.cancel_requested)
        self.assertTrue(turn.error_emitted)
        self.assertTrue(turn.call.cancel.called)
