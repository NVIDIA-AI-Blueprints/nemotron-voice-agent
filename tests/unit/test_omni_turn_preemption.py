# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

# ruff: noqa: D100, D101, D102, D105

import asyncio
import unittest
from collections.abc import Callable
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from pipecat.frames.frames import (
    InputAudioRawFrame,
    LLMConfigureOutputFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
    LLMThoughtEndFrame,
    LLMThoughtStartFrame,
    LLMThoughtTextFrame,
    TranscriptionFrame,
    TTSUpdateSettingsFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.nvidia.llm import NvidiaLLMService as PipecatNvidiaLLMService
from pipecat.turns.user_turn_completion_mixin import UserTurnCompletionLLMServiceMixin

from examples.omni_assistant.nvidia_omni_multimodal_service import (
    MAX_FUSED_USER_AUDIO_SECS,
    NvidiaOmniLLMService,
    NvidiaOmniSettings,
    _TranscriptResponseExtractor,
    audio_message_part,
)
from examples.omni_assistant_subagents.subagents.speaker.agent import SubagentsSpeakerOmniService
from examples.shared.frames import UserTranscriptProducerEndedFrame
from realtime.frames import (
    RealtimeOwnedLLMFullResponseStartFrame,
    RealtimeResponseContextFrame,
    RealtimeResponseLLMContext,
    RealtimeResponseOrigin,
)


def _chunk(content=None, *, tool_calls=None):
    delta = SimpleNamespace(content=content, tool_calls=tool_calls)
    return SimpleNamespace(choices=[SimpleNamespace(delta=delta)], usage=None)


def _provider_chunk(content=None, *, finish_reason=None, usage=None):
    choices = []
    if content is not None or finish_reason is not None:
        delta = SimpleNamespace(content=content, reasoning_content=None)
        choices = [SimpleNamespace(delta=delta, finish_reason=finish_reason)]
    return SimpleNamespace(choices=choices, usage=usage)


async def _stream(*chunks):
    for chunk in chunks:
        yield chunk


class _ClosableStream:
    def __init__(self, *chunks) -> None:
        self._chunks = iter(chunks)
        self.closed = False

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            return next(self._chunks)
        except StopIteration as exc:
            raise StopAsyncIteration from exc

    async def aclose(self) -> None:
        self.closed = True


def _context(messages):
    return SimpleNamespace(get_messages=lambda: messages)


def _user_context():
    """A context carrying one unanswered user turn, as an aggregator pushes."""
    return _context([{"role": "user", "content": "hi"}])


class _FakeTurn:
    """Stand-in for one in-flight Omni turn that can be released or cancelled."""

    def __init__(self) -> None:
        self.release = asyncio.Event()
        self.cancelled = False
        self.completed = False
        self.args: tuple = ()
        self.kwargs: dict = {}


class OmniOutOfPipelineInferenceTests(unittest.IsolatedAsyncioTestCase):
    @staticmethod
    def _service(*chunks):
        service = NvidiaOmniLLMService(api_key="not-needed", base_url="http://localhost:8000/v1")
        service._out_of_band_request_kwargs = Mock(return_value={})
        response_stream = _ClosableStream(*chunks)
        create = AsyncMock(return_value=response_stream)
        service._client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        return service, response_stream, create

    async def test_stream_retains_terminal_usage_only_chunk_and_real_timings(self) -> None:
        usage = SimpleNamespace(
            prompt_tokens=12,
            completion_tokens=5,
            total_tokens=17,
            prompt_tokens_details=SimpleNamespace(
                cached_tokens=3,
                cache_write_tokens=2,
                audio_tokens=4,
            ),
            completion_tokens_details=SimpleNamespace(reasoning_tokens=2, audio_tokens=0),
        )
        service, response_stream, create = self._service(
            _provider_chunk("answer", finish_reason="stop"),
            _provider_chunk(usage=usage),
        )

        with patch(
            "examples.omni_assistant.nvidia_omni_multimodal_service.time.monotonic",
            side_effect=[10.0, 10.4, 11.2],
        ):
            result = await service.run_multimodal_inference(LLMContext(messages=[]), stream=True)

        self.assertEqual(result.text, "answer")
        self.assertAlmostEqual(result.ttfb_seconds, 0.4)
        self.assertAlmostEqual(result.processing_seconds, 1.2)
        self.assertEqual(result.usage.prompt_tokens, 12)
        self.assertEqual(result.usage.completion_tokens, 5)
        self.assertEqual(result.usage.total_tokens, 17)
        self.assertEqual(result.usage.cache_read_input_tokens, 3)
        self.assertEqual(result.usage.cache_creation_input_tokens, 2)
        self.assertEqual(result.usage.reasoning_tokens, 2)
        self.assertEqual(result.usage.input_audio_tokens, 4)
        self.assertEqual(result.usage.output_audio_tokens, 0)
        self.assertEqual(create.await_args.kwargs["stream_options"], {"include_usage": True})
        self.assertTrue(response_stream.closed)

    async def test_stream_without_provider_terminal_is_rejected_and_closed(self) -> None:
        service, response_stream, _ = self._service(_provider_chunk("partial"))

        with self.assertRaisesRegex(ValueError, "without a finish_reason"):
            await service.run_multimodal_inference(LLMContext(messages=[]), stream=True)

        self.assertTrue(response_stream.closed)

    async def test_stream_rejects_choices_after_terminal_and_closes(self) -> None:
        service, response_stream, _ = self._service(
            _provider_chunk("complete", finish_reason="stop"),
            _provider_chunk("late"),
        )

        with self.assertRaisesRegex(ValueError, "choices after its terminal chunk"):
            await service.run_multimodal_inference(LLMContext(messages=[]), stream=True)

        self.assertTrue(response_stream.closed)


class OmniOrdinaryPipelineDelegationTests(unittest.IsolatedAsyncioTestCase):
    async def test_non_realtime_context_uses_pipecats_nvidia_stream_path_unchanged(self) -> None:
        service = NvidiaOmniLLMService(api_key="not-needed", base_url="http://localhost:8000/v1")
        context = LLMContext([{"role": "user", "content": "hello"}])
        provider_stream = object()

        with patch.object(
            PipecatNvidiaLLMService,
            "get_chat_completions",
            new=AsyncMock(return_value=provider_stream),
        ) as stock_get_chat_completions:
            result = await service.get_chat_completions(context)

        self.assertIs(result, provider_stream)
        stock_get_chat_completions.assert_awaited_once_with(context)

    async def test_ordinary_transcription_has_no_realtime_producer_terminal(self) -> None:
        service = NvidiaOmniLLMService(
            api_key="not-needed",
            base_url="http://localhost:8000/v1",
            settings=NvidiaOmniSettings(emit_transcriptions=True),
        )
        pushed = AsyncMock()
        with patch.object(UserTurnCompletionLLMServiceMixin, "push_frame", new=pushed):
            await service._handle_user_stopped(UserStoppedSpeakingFrame())
            await service._emit_user_transcript("ordinary transcript")

        self.assertEqual(pushed.await_count, 1)
        self.assertIsInstance(pushed.await_args.args[0], TranscriptionFrame)
        self.assertEqual(pushed.await_args.args[0].text, "ordinary transcript")

    async def test_failed_response_start_does_not_emit_unmatched_end_boundary(self) -> None:
        service = NvidiaOmniLLMService(api_key="not-needed", base_url="http://localhost:8000/v1")
        attempted_frames = []

        async def fail_response_start(_processor, frame, _direction=FrameDirection.DOWNSTREAM) -> None:
            attempted_frames.append(frame)
            if isinstance(frame, LLMFullResponseStartFrame):
                raise RuntimeError("response start failed")

        service.push_error = AsyncMock()
        service.start_processing_metrics = AsyncMock()
        service._process_context = AsyncMock()
        with patch.object(UserTurnCompletionLLMServiceMixin, "push_frame", new=fail_response_start):
            await service._run_turn(LLMContext([{"role": "user", "content": "hello"}]))

        self.assertEqual([type(frame) for frame in attempted_frames], [LLMFullResponseStartFrame])
        service.push_error.assert_awaited_once()
        service.start_processing_metrics.assert_not_awaited()
        service._process_context.assert_not_awaited()

    async def test_realtime_binding_enables_transcript_producer_terminal(self) -> None:
        service = NvidiaOmniLLMService(
            api_key="not-needed",
            base_url="http://localhost:8000/v1",
            settings=NvidiaOmniSettings(emit_transcriptions=True),
        )
        service.bind_realtime_response_snapshot(AsyncMock())
        pushed = AsyncMock()
        stopped = UserStoppedSpeakingFrame()
        with patch.object(UserTurnCompletionLLMServiceMixin, "push_frame", new=pushed):
            await service._handle_user_stopped(stopped)

        pushed.assert_awaited_once()
        terminal = pushed.await_args.args[0]
        self.assertIsInstance(terminal, UserTranscriptProducerEndedFrame)
        self.assertEqual(terminal.turn_frame_id, stopped.id)
        self.assertEqual(terminal.status, "skipped")
        self.assertEqual(pushed.await_args.args[1], FrameDirection.UPSTREAM)


class OmniRealtimeResponseSnapshotTests(unittest.IsolatedAsyncioTestCase):
    """Fused turns must enter the same immutable Realtime response lifecycle."""

    @staticmethod
    def _services() -> tuple[NvidiaOmniLLMService, SubagentsSpeakerOmniService]:
        return (
            NvidiaOmniLLMService(api_key="not-needed", base_url="http://localhost:8000/v1"),
            SubagentsSpeakerOmniService(
                api_key="not-needed",
                base_url="http://localhost:8000/v1",
                audio_response_instruction="Return the required JSON action envelope.",
            ),
        )

    async def test_service_started_turn_applies_snapshot_before_output_for_single_and_speaker(self) -> None:
        for service in self._services():
            with self.subTest(service=type(service).__name__):
                canonical = LLMContext([{"role": "user", "content": "hello"}])
                snapshot = RealtimeResponseLLMContext(
                    [{"role": "user", "content": "hello"}],
                    max_output_tokens=73,
                    parallel_tool_calls=False,
                )
                tts = Mock(name="response-tts")
                setup = (
                    LLMConfigureOutputFrame(skip_tts=True),
                    TTSUpdateSettingsFrame(settings={"voice": "voice-b"}, service=tts),
                )
                abort = Mock()
                activate = AsyncMock(return_value="resp_fused")
                hook = AsyncMock(return_value=(snapshot, setup, activate, abort))
                service.bind_realtime_response_snapshot(hook)
                service.start_processing_metrics = AsyncMock()
                service.stop_processing_metrics = AsyncMock()
                seen_contexts: list[RealtimeResponseLLMContext] = []
                frames = []

                async def process_context(context, _service=service, _seen=seen_contexts) -> None:
                    _seen.append(context)
                    self.assertTrue(_service._skip_tts)
                    await _service.push_frame(LLMTextFrame("response text"))

                async def collect_frame(
                    _processor,
                    frame,
                    _direction=FrameDirection.DOWNSTREAM,
                    _frames=frames,
                ) -> None:
                    _frames.append(frame)

                service._process_context = AsyncMock(side_effect=process_context)
                with patch.object(UserTurnCompletionLLMServiceMixin, "push_frame", new=collect_frame):
                    await service._run_turn(canonical)

                hook.assert_awaited_once_with(canonical, None, RealtimeResponseOrigin.SERVICE_INITIATED)
                activate.assert_awaited_once_with()
                abort.assert_not_called()
                self.assertEqual(seen_contexts, [snapshot])
                self.assertEqual(snapshot.max_output_tokens, 73)
                self.assertIs(snapshot.parallel_tool_calls, False)
                self.assertEqual(
                    [type(frame) for frame in frames],
                    [
                        LLMConfigureOutputFrame,
                        TTSUpdateSettingsFrame,
                        RealtimeOwnedLLMFullResponseStartFrame,
                        LLMTextFrame,
                        LLMFullResponseEndFrame,
                    ],
                )
                self.assertIs(frames[1].service, tts)
                self.assertTrue(frames[2].skip_tts)
                self.assertTrue(frames[3].skip_tts)
                self.assertTrue(frames[4].skip_tts)
                self.assertIsNone(service._skip_tts)
                self.assertEqual(canonical.get_messages(), [{"role": "user", "content": "hello"}])

    async def test_cancel_during_response_start_still_emits_matching_end_boundary(self) -> None:
        service = NvidiaOmniLLMService(api_key="not-needed", base_url="http://localhost:8000/v1")
        canonical = LLMContext([{"role": "user", "content": "hello"}])
        snapshot = RealtimeResponseLLMContext(list(canonical.get_messages()))
        abort = Mock()
        service.bind_realtime_response_snapshot(
            AsyncMock(return_value=(snapshot, (), AsyncMock(return_value="resp_cancelled"), abort))
        )
        frames = []

        async def cancel_at_start(_processor, frame, _direction=FrameDirection.DOWNSTREAM) -> None:
            frames.append(frame)
            if isinstance(frame, LLMFullResponseStartFrame):
                raise asyncio.CancelledError

        with (
            patch.object(UserTurnCompletionLLMServiceMixin, "push_frame", new=cancel_at_start),
            self.assertRaises(asyncio.CancelledError),
        ):
            await service._run_turn(canonical)

        self.assertEqual(
            [type(frame) for frame in frames],
            [RealtimeOwnedLLMFullResponseStartFrame, LLMFullResponseEndFrame],
        )
        abort.assert_not_called()

    async def test_explicit_snapshot_is_not_frozen_twice_or_retained_as_canonical(self) -> None:
        service = NvidiaOmniLLMService(api_key="not-needed", base_url="http://localhost:8000/v1")
        canonical = LLMContext([{"role": "user", "content": "canonical"}])
        snapshot = RealtimeResponseLLMContext(
            [{"role": "user", "content": "explicit"}],
            max_output_tokens=19,
            parallel_tool_calls=True,
        )
        hook = Mock(side_effect=AssertionError("explicit response was snapshotted twice"))
        service.bind_realtime_response_snapshot(hook)
        service._context = canonical
        service._process_context = AsyncMock()
        service.start_processing_metrics = AsyncMock()
        service.stop_processing_metrics = AsyncMock()

        async def discard_frame(_processor, _frame, _direction=FrameDirection.DOWNSTREAM) -> None:
            return None

        with patch.object(UserTurnCompletionLLMServiceMixin, "push_frame", new=discard_frame):
            await service._run_turn(snapshot)

        hook.assert_not_called()
        service._process_context.assert_awaited_once_with(snapshot)

        service._maybe_run_text_turn = AsyncMock()
        owned = RealtimeResponseContextFrame(
            context=snapshot,
            response_id="resp_explicit",
            canonical_context=canonical,
        )
        with patch("pipecat.services.llm_service.LLMService.process_frame", new=AsyncMock()):
            await service.process_frame(owned, FrameDirection.DOWNSTREAM)
        self.assertIs(service._context, canonical)
        service._maybe_run_text_turn.assert_awaited_once_with(snapshot)


class OmniAudioBufferBoundTests(unittest.IsolatedAsyncioTestCase):
    @staticmethod
    def _service(*, max_secs: float = 0.01, pre_secs: float = 0.0) -> NvidiaOmniLLMService:
        service = NvidiaOmniLLMService(
            api_key="not-needed",
            base_url="http://localhost:8000/v1",
            settings=NvidiaOmniSettings(
                emit_transcriptions=True,
                min_user_audio_secs=0.0,
                max_user_audio_secs=max_secs,
                pre_speech_buffer_secs=pre_secs,
            ),
        )
        service.bind_realtime_response_snapshot(
            AsyncMock(),
            reserve_audio_response=Mock(),
            release_audio_response=Mock(),
        )
        return service

    async def test_rtvi_does_not_apply_the_realtime_audio_ceiling(self) -> None:
        service = NvidiaOmniLLMService(
            api_key="not-needed",
            base_url="http://localhost:8000/v1",
            settings=NvidiaOmniSettings(
                emit_transcriptions=True,
                min_user_audio_secs=0.0,
                max_user_audio_secs=0.01,
                pre_speech_buffer_secs=0.0,
            ),
        )
        await service._handle_user_started()
        audio = InputAudioRawFrame(audio=b"\x00" * 640, sample_rate=16000, num_channels=1)

        service._handle_audio_frame(audio)

        self.assertFalse(service._audio_buffer_overflowed)
        self.assertEqual(service._audio_buffer_bytes, 640)

    async def test_cumulative_limit_releases_turn_memory_and_ignores_remainder(self) -> None:
        service = self._service()
        service._end_audio_transcript_producer = AsyncMock()
        await service._handle_user_started()
        five_ms = InputAudioRawFrame(audio=b"\x00" * 160, sample_rate=16000, num_channels=1)

        service._handle_audio_frame(five_ms)
        service._handle_audio_frame(five_ms)
        self.assertEqual(service._audio_buffer_bytes, 320)
        self.assertEqual(len(service._audio_buffer), 2)

        service._handle_audio_frame(InputAudioRawFrame(audio=b"\x00\x00", sample_rate=16000, num_channels=1))
        self.assertTrue(service._audio_buffer_overflowed)
        self.assertEqual(service._audio_buffer_bytes, 0)
        self.assertEqual(service._audio_buffer, [])

        service._handle_audio_frame(five_ms)
        self.assertEqual(service._audio_buffer_bytes, 0)
        self.assertEqual(service._audio_buffer, [])

        stopped = UserStoppedSpeakingFrame()
        await service._handle_user_stopped(stopped)
        terminal = service._end_audio_transcript_producer.await_args.args[0]
        self.assertEqual(terminal.turn_frame_id, stopped.id)
        self.assertEqual(service._end_audio_transcript_producer.await_args.kwargs["status"], "overflowed")
        self.assertFalse(service._audio_buffer_overflowed)
        self.assertIsNone(service._pending_request)

        await service._handle_user_started()
        service._handle_audio_frame(five_ms)
        self.assertEqual(service._audio_buffer_bytes, 160)
        self.assertFalse(service._audio_buffer_overflowed)

    async def test_pre_speech_audio_counts_toward_the_same_cumulative_limit(self) -> None:
        service = self._service(max_secs=0.01, pre_secs=0.005)
        five_ms = InputAudioRawFrame(audio=b"\x00" * 160, sample_rate=16000, num_channels=1)
        service._handle_audio_frame(five_ms)
        self.assertEqual(service._pre_speech_buffer_bytes, 160)

        await service._handle_user_started()
        self.assertEqual(service._audio_buffer_bytes, 160)
        service._handle_audio_frame(five_ms)
        self.assertEqual(service._audio_buffer_bytes, 320)
        service._handle_audio_frame(InputAudioRawFrame(audio=b"\x00\x00", sample_rate=16000, num_channels=1))

        self.assertTrue(service._audio_buffer_overflowed)
        self.assertEqual(service._audio_buffer_bytes, 0)
        self.assertEqual(service._audio_buffer, [])

    async def test_lower_runtime_limit_drops_an_already_over_limit_turn(self) -> None:
        service = self._service(max_secs=1.0)
        await service._handle_user_started()
        service._handle_audio_frame(InputAudioRawFrame(audio=b"\x00" * 24000, sample_rate=16000, num_channels=1))

        await service._update_settings(NvidiaOmniSettings(max_user_audio_secs=0.5))

        self.assertEqual(service._settings.max_user_audio_secs, 0.5)
        self.assertTrue(service._audio_buffer_overflowed)
        self.assertEqual(service._audio_buffer_bytes, 0)
        self.assertEqual(service._audio_buffer, [])

    def test_audio_duration_settings_are_finite_and_positive_for_every_transport(self) -> None:
        default_service = NvidiaOmniLLMService(
            api_key="not-needed",
            base_url="http://localhost:8000/v1",
        )
        self.assertEqual(default_service._settings.max_user_audio_secs, MAX_FUSED_USER_AUDIO_SECS)

        invalid_settings = (
            NvidiaOmniSettings(max_user_audio_secs=0),
            NvidiaOmniSettings(max_user_audio_secs=True),
            NvidiaOmniSettings(max_user_audio_secs=float("nan")),
            NvidiaOmniSettings(max_user_audio_secs=float("inf")),
            NvidiaOmniSettings(min_user_audio_secs=-0.1),
            NvidiaOmniSettings(pre_speech_buffer_secs=-0.1),
        )
        for settings in invalid_settings:
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                NvidiaOmniLLMService(
                    api_key="not-needed",
                    base_url="http://localhost:8000/v1",
                    settings=settings,
                )

    def test_realtime_audio_ceiling_is_validated_only_when_realtime_is_bound(self) -> None:
        realtime_invalid_settings = (
            NvidiaOmniSettings(max_user_audio_secs=MAX_FUSED_USER_AUDIO_SECS + 1),
            NvidiaOmniSettings(min_user_audio_secs=2.0, max_user_audio_secs=1.0),
            NvidiaOmniSettings(pre_speech_buffer_secs=2.0, max_user_audio_secs=1.0),
        )
        for settings in realtime_invalid_settings:
            with self.subTest(settings=settings):
                service = NvidiaOmniLLMService(
                    api_key="not-needed",
                    base_url="http://localhost:8000/v1",
                    settings=settings,
                )
                with self.assertRaises(ValueError):
                    service.bind_realtime_response_snapshot(AsyncMock())
                self.assertIsNone(service._realtime_response_snapshot_hook)


class OmniRealtimeAudioResponseSnapshotTests(unittest.IsolatedAsyncioTestCase):
    async def test_audio_and_tool_followup_get_distinct_service_snapshots(self) -> None:
        service = NvidiaOmniLLMService(api_key="not-needed", base_url="http://localhost:8000/v1")
        service.create_task = lambda coro, name=None: asyncio.create_task(coro, name=name)
        service.start_processing_metrics = AsyncMock()
        service.stop_processing_metrics = AsyncMock()
        service.stop_all_metrics = AsyncMock()
        canonical = LLMContext([{"role": "user", "content": "weather?"}])
        service._context = canonical
        response_a = RealtimeResponseLLMContext(
            list(canonical.get_messages()),
            max_output_tokens=31,
            parallel_tool_calls=False,
        )
        tool_context = LLMContext(
            [
                {"role": "user", "content": "weather?"},
                {"role": "assistant", "tool_calls": [{"id": "call_1"}]},
                {"role": "tool", "tool_call_id": "call_1", "content": "sunny"},
            ]
        )
        response_b = RealtimeResponseLLMContext(
            list(tool_context.get_messages()),
            max_output_tokens=47,
            parallel_tool_calls=True,
        )
        hook = AsyncMock(
            side_effect=[
                (
                    response_a,
                    (LLMConfigureOutputFrame(skip_tts=False),),
                    AsyncMock(return_value="resp_a"),
                    Mock(),
                ),
                (
                    response_b,
                    (LLMConfigureOutputFrame(skip_tts=False),),
                    AsyncMock(return_value="resp_b"),
                    Mock(),
                ),
            ]
        )
        service.bind_realtime_response_snapshot(hook)
        service._process_context = AsyncMock()
        service._audio_buffer = [b"\x00" * (service._sample_rate * service._channels * 2)]

        async def discard_frame(_processor, _frame, _direction=FrameDirection.DOWNSTREAM) -> None:
            return None

        with patch.object(UserTurnCompletionLLMServiceMixin, "push_frame", new=discard_frame):
            await service._maybe_run_audio_turn(transcript_turn_frame_id=501)
            await service._pending_request
            await service._maybe_run_text_turn(tool_context)
            await service._pending_request

        self.assertEqual(
            hook.await_args_list,
            [
                unittest.mock.call(canonical, None, RealtimeResponseOrigin.AUTOMATIC_USER_TURN),
                unittest.mock.call(tool_context, None, RealtimeResponseOrigin.INTERNAL_TOOL_CONTINUATION),
            ],
        )
        self.assertEqual(
            [call.args[0] for call in service._process_context.await_args_list],
            [response_a, response_b],
        )

    async def test_short_audio_and_transcript_echo_do_not_reserve_phantom_responses(self) -> None:
        service = NvidiaOmniLLMService(api_key="not-needed", base_url="http://localhost:8000/v1")
        hook = Mock()
        service.bind_realtime_response_snapshot(hook)
        service._context = LLMContext([{"role": "user", "content": "too short"}])
        service._audio_buffer = [b"\x00" * 100]

        await service._maybe_run_audio_turn(transcript_turn_frame_id=601)
        self.assertIsNone(service._pending_request)

        service._answered_transcript = "already answered"
        echo = LLMContext([{"role": "user", "content": "already answered"}])
        await service._maybe_run_text_turn(echo)
        self.assertIsNone(service._pending_request)
        hook.assert_not_called()


class OmniTurnPreemptionTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.service = NvidiaOmniLLMService(api_key="not-needed", base_url="http://localhost:8000/v1")

        # Run turns as plain asyncio tasks and bypass the pipecat task-manager /
        # metrics machinery, which would otherwise require a full pipeline setup.
        self.service.create_task = lambda coro, name=None: asyncio.create_task(coro, name=name)
        self.service.stop_all_metrics = AsyncMock()

        self.turns: list[_FakeTurn] = []

        async def fake_run_turn(*args, **kwargs) -> None:
            turn = _FakeTurn()
            turn.args = args
            turn.kwargs = kwargs
            self.turns.append(turn)
            try:
                await turn.release.wait()
                turn.completed = True
            except asyncio.CancelledError:
                turn.cancelled = True
                raise

        self.service._run_turn = fake_run_turn

    async def asyncTearDown(self) -> None:
        for turn in self.turns:
            turn.release.set()
        await self.service._cancel_pending_request()

    async def _wait_for(self, predicate: Callable[[], bool], timeout: float = 1.0) -> None:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        while not predicate():
            if loop.time() > deadline:
                self.fail("condition was not met within the timeout")
            await asyncio.sleep(0.005)

    def _fill_audio(self, seconds: float = 1.0) -> None:
        # PCM16 mono payload comfortably above the min_user_audio_secs gate (0.3s).
        nbytes = int(self.service._sample_rate * self.service._channels * 2 * seconds)
        self.service._audio_buffer = [b"\x00" * nbytes]

    async def test_audio_turn_preempts_in_flight_turn(self) -> None:
        self._fill_audio()
        await self.service._maybe_run_audio_turn(transcript_turn_frame_id=101)
        first_task = self.service._pending_request
        self.assertIsNotNone(first_task)
        await self._wait_for(lambda: len(self.turns) == 1)
        self.assertFalse(first_task.done())

        self._fill_audio()
        await self.service._maybe_run_audio_turn(transcript_turn_frame_id=102)
        second_task = self.service._pending_request

        # The previous turn must be preempted (cancelled), not skipped...
        self.assertTrue(first_task.cancelled())
        self.assertTrue(self.turns[0].cancelled)
        self.service.stop_all_metrics.assert_awaited()

        # ...and a brand-new turn must have started in its place.
        self.assertIsNotNone(second_task)
        self.assertIsNot(second_task, first_task)
        await self._wait_for(lambda: len(self.turns) == 2)
        self.assertFalse(second_task.done())

    async def test_text_turn_preempts_in_flight_turn(self) -> None:
        await self.service._maybe_run_text_turn(_user_context(), force=True)
        first_task = self.service._pending_request
        self.assertIsNotNone(first_task)
        await self._wait_for(lambda: len(self.turns) == 1)

        await self.service._maybe_run_text_turn(_user_context(), force=True)
        second_task = self.service._pending_request

        self.assertTrue(first_task.cancelled())
        self.assertTrue(self.turns[0].cancelled)
        self.assertIsNot(second_task, first_task)
        await self._wait_for(lambda: len(self.turns) == 2)
        self.assertFalse(second_task.done())

    async def test_context_turn_yields_to_in_flight_audio_turn(self) -> None:
        self._fill_audio()
        await self.service._maybe_run_audio_turn(transcript_turn_frame_id=201)
        audio_task = self.service._pending_request
        await self._wait_for(lambda: len(self.turns) == 1)

        # Context/run echo for the same spoken turn must yield, not preempt.
        await self.service._maybe_run_text_turn(_user_context(), force=True)

        self.assertIs(self.service._pending_request, audio_task)
        self.assertFalse(audio_task.cancelled())
        self.assertEqual(len(self.turns), 1)
        self.service.stop_all_metrics.assert_not_awaited()

    async def test_audio_turn_below_min_duration_does_not_preempt(self) -> None:
        self._fill_audio()
        await self.service._maybe_run_audio_turn(transcript_turn_frame_id=301)
        first_task = self.service._pending_request
        await self._wait_for(lambda: len(self.turns) == 1)

        self._fill_audio(seconds=0.05)
        await self.service._maybe_run_audio_turn(transcript_turn_frame_id=302)

        self.assertIs(self.service._pending_request, first_task)
        self.assertFalse(first_task.cancelled())
        self.assertEqual(len(self.turns), 1)

    async def test_text_turn_is_skipped_when_text_modality_is_disabled(self) -> None:
        self.service._settings.input_modalities = ("audio",)
        await self.service._maybe_run_text_turn(_user_context(), force=True)
        self.assertIsNone(self.service._pending_request)

    async def test_an_utterance_arriving_before_any_context_is_still_answered(self) -> None:
        # An audio-only pipeline may never send a context frame, and the user is
        # waiting either way, so the turn runs without history behind it.
        self.service._context = None
        self._fill_audio()

        await self.service._maybe_run_audio_turn(transcript_turn_frame_id=401)
        await self._wait_for(lambda: len(self.turns) == 1)

        self.assertEqual(self.turns[0].args[0].get_messages(), [])

    async def test_a_completion_asked_for_before_any_context_does_not_run(self) -> None:
        await self.service._maybe_run_text_turn(None, force=True)
        self.assertIsNone(self.service._pending_request)

    async def test_tool_result_is_answered_even_in_an_audio_only_pipeline(self) -> None:
        # The completion that asked for the call can only be finished by another
        # one, so an audio-only pipeline must not leave the result unspoken.
        self.service._settings.input_modalities = ("audio",)
        context = _context(
            [
                {"role": "user", "content": "weather?"},
                {"role": "assistant", "tool_calls": [{"id": "call_1"}]},
                {"role": "tool", "tool_call_id": "call_1", "content": "sunny"},
            ]
        )

        await self.service._maybe_run_text_turn(context)

        self.assertIsNotNone(self.service._pending_request)
        await self._wait_for(lambda: len(self.turns) == 1)

    async def test_text_only_pipeline_answers_every_context_frame(self) -> None:
        # With no audio path to echo, this behaves like any other LLM service:
        # the aggregator asks for a completion and it runs.
        self.service._settings.input_modalities = ("text",)
        self.service._answered_transcript = "where is the tower?"
        context = _context([{"role": "user", "content": "where is the tower?"}, {"role": "assistant", "content": "hi"}])

        await self.service._maybe_run_text_turn(context)

        self.assertIsNotNone(self.service._pending_request)
        await self._wait_for(lambda: len(self.turns) == 1)

    async def test_context_without_pending_turn_does_not_run(self) -> None:
        context = _context([{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}])
        await self.service._maybe_run_text_turn(context)
        self.assertIsNone(self.service._pending_request)

    async def test_transcript_echo_does_not_answer_the_same_turn_twice(self) -> None:
        self.service._answered_transcript = "where is the tower?"
        context = _context([{"role": "user", "content": "where is the tower?"}])

        await self.service._maybe_run_text_turn(context)

        self.assertIsNone(self.service._pending_request)

    async def test_tool_result_runs_the_follow_up_completion(self) -> None:
        context = _context(
            [
                {"role": "user", "content": "weather?"},
                {"role": "assistant", "tool_calls": [{"id": "call_1"}]},
                {"role": "tool", "tool_call_id": "call_1", "content": "sunny"},
            ]
        )
        await self.service._maybe_run_text_turn(context)
        self.assertIsNotNone(self.service._pending_request)
        await self._wait_for(lambda: len(self.turns) == 1)

    async def test_tool_follow_up_carries_a_spoken_turn_the_context_lacks(self) -> None:
        # The aggregator writes a spoken turn once the assistant response starts,
        # which can be later than the follow-up a tool result asks for, so the
        # follow-up carries the request the user spoke itself.
        self.service._answered_transcript = "what is the weather?"
        context = _context(
            [
                {"role": "assistant", "tool_calls": [{"id": "call_1"}]},
                {"role": "tool", "tool_call_id": "call_1", "content": "sunny"},
            ]
        )

        await self.service._maybe_run_text_turn(context)
        await self._wait_for(lambda: len(self.turns) == 1)

        self.assertEqual(self.turns[0].kwargs["turn_parts"], [{"type": "text", "text": "what is the weather?"}])

    async def test_tool_follow_up_leaves_a_written_spoken_turn_alone(self) -> None:
        self.service._answered_transcript = "what is the weather?"
        context = _context(
            [
                {"role": "user", "content": "what is the weather?"},
                {"role": "assistant", "tool_calls": [{"id": "call_1"}]},
                {"role": "tool", "tool_call_id": "call_1", "content": "sunny"},
            ]
        )

        await self.service._maybe_run_text_turn(context)
        await self._wait_for(lambda: len(self.turns) == 1)

        self.assertIsNone(self.turns[0].kwargs["turn_parts"])

    async def test_tool_result_waits_for_the_turn_that_requested_it(self) -> None:
        # The tool-calling turn is still wrapping up when the follow-up arrives.
        await self.service._maybe_run_text_turn(_user_context(), force=True)
        tool_call_task = self.service._pending_request
        await self._wait_for(lambda: len(self.turns) == 1)

        context = _context(
            [
                {"role": "user", "content": "weather?"},
                {"role": "assistant", "tool_calls": [{"id": "call_1"}]},
                {"role": "tool", "tool_call_id": "call_1", "content": "sunny"},
            ]
        )
        follow_up = asyncio.create_task(self.service._maybe_run_text_turn(context))
        await asyncio.sleep(0)

        # It must wait rather than cancel the turn that issued the tool call.
        self.assertFalse(follow_up.done())
        self.assertFalse(tool_call_task.cancelled())

        self.turns[0].release.set()
        await follow_up

        self.assertTrue(self.turns[0].completed)
        self.assertIsNot(self.service._pending_request, tool_call_task)
        await self._wait_for(lambda: len(self.turns) == 2)


class OmniStreamHandlingTests(unittest.IsolatedAsyncioTestCase):
    """Reasoning comes from NvidiaLLMService; these cover what Omni adds on top."""

    async def asyncSetUp(self) -> None:
        self.service = NvidiaOmniLLMService(api_key="not-needed", base_url="http://localhost:8000/v1")
        self.frames: list = []
        self.service.push_frame = AsyncMock(side_effect=lambda frame, *_a, **_kw: self.frames.append(frame))

    def _begin_turn(self, *, expect_transcript: bool) -> None:
        self.service._reset_response_state()
        self.service._transcript_extractor = _TranscriptResponseExtractor() if expect_transcript else None
        self.service._transcript_emitted = False

    async def _drain(self, *chunks) -> list:
        """Run chunks through the inherited wrapper and the base loop's text push."""
        out = []
        async for chunk in self.service._handle_reasoning_content(_stream(*chunks)):
            out.append(chunk)
            delta = chunk.choices[0].delta if chunk.choices else None
            if delta is not None and delta.content:
                await self.service._push_llm_text(delta.content)
        return out

    def _spoken(self) -> str:
        return "".join(f.text for f in self.frames if isinstance(f, LLMTextFrame))

    async def test_transcript_tags_split_user_speech_from_spoken_reply(self) -> None:
        self._begin_turn(expect_transcript=True)
        await self._drain(
            _chunk("<transcript>Where is the "),
            _chunk("Eiffel Tower?</transcript>"),
            _chunk("<response>It is in Paris."),
            _chunk("</response>"),
        )

        transcripts = [f for f in self.frames if isinstance(f, TranscriptionFrame)]
        self.assertEqual([f.text for f in transcripts], ["Where is the Eiffel Tower?"])
        self.assertEqual(self._spoken(), "It is in Paris.")

    async def test_untagged_response_still_reaches_tts(self) -> None:
        self._begin_turn(expect_transcript=True)
        await self._drain(_chunk("It is in Paris."))

        self.assertEqual([f for f in self.frames if isinstance(f, TranscriptionFrame)], [])
        self.assertEqual(self._spoken(), "It is in Paris.")

    async def test_response_held_at_stream_end_is_still_spoken(self) -> None:
        # The last chunk ends mid-response, so the split is holding text back
        # against a possible closing tag when the stream ends.
        self._begin_turn(expect_transcript=True)
        await self._drain(
            _chunk("<transcript>Hi</transcript><response>Hello the"),
            _chunk("re."),
        )

        self.assertEqual(self._spoken(), "Hello there.")

    async def test_reasoning_and_transcript_sections_compose(self) -> None:
        self._begin_turn(expect_transcript=True)
        await self._drain(
            _chunk("<think>They asked for a city.</think>"),
            _chunk("<transcript>Where is it?</transcript>"),
            _chunk("<response>In Paris.</response>"),
        )

        thoughts = "".join(f.text for f in self.frames if isinstance(f, LLMThoughtTextFrame))
        self.assertEqual(thoughts, "They asked for a city.")
        self.assertIsInstance(self.frames[0], LLMThoughtStartFrame)
        self.assertTrue(any(isinstance(f, LLMThoughtEndFrame) for f in self.frames))
        self.assertEqual([f.text for f in self.frames if isinstance(f, TranscriptionFrame)], ["Where is it?"])
        self.assertEqual(self._spoken(), "In Paris.")

    async def test_reasoning_content_field_stays_out_of_the_transcript_split(self) -> None:
        self._begin_turn(expect_transcript=True)
        reasoning_chunk = _chunk(None)
        reasoning_chunk.choices[0].delta.reasoning_content = "Thinking hard."
        await self._drain(reasoning_chunk, _chunk("<transcript>Hi</transcript><response>Hello.</response>"))

        self.assertIsInstance(self.frames[0], LLMThoughtStartFrame)
        self.assertEqual(self.frames[1].text, "Thinking hard.")
        self.assertEqual(self._spoken(), "Hello.")

    async def test_plain_content_is_not_buffered_or_rewritten(self) -> None:
        self._begin_turn(expect_transcript=False)
        out = await self._drain(_chunk("Hello"), _chunk(" there."))

        self.assertEqual([c.choices[0].delta.content for c in out], ["Hello", " there."])
        self.assertEqual(self._spoken(), "Hello there.")

    async def test_tool_call_chunks_pass_through_untouched(self) -> None:
        self._begin_turn(expect_transcript=True)
        tool_call = SimpleNamespace(index=0, id="call_1", function=SimpleNamespace(name="get_weather", arguments=""))
        out = await self._drain(_chunk(None, tool_calls=[tool_call]))

        self.assertEqual(out[0].choices[0].delta.tool_calls, [tool_call])
        self.assertEqual([f for f in self.frames if isinstance(f, TranscriptionFrame)], [])


class OmniTranscriptOwnershipTests(unittest.IsolatedAsyncioTestCase):
    """Who writes a spoken turn into the conversation, and who only reports it."""

    async def asyncSetUp(self) -> None:
        self.service = NvidiaOmniLLMService(api_key="not-needed", base_url="http://localhost:8000/v1")
        self.context = LLMContext([{"role": "system", "content": "You are helpful."}])
        self.service._context = self.context
        self.pushed: list[tuple] = []
        self.service.push_frame = AsyncMock(
            side_effect=lambda frame, direction=None: self.pushed.append((frame, direction))
        )

    async def test_a_spoken_turn_is_reported_for_the_user_aggregator_to_write(self) -> None:
        # The aggregator upstream writes what this frame carries, so writing it
        # here as well would leave the same turn in the conversation twice.
        await self.service._emit_user_transcript("where is the tower?")

        self.assertEqual([m["role"] for m in self.context.get_messages()], ["system"])
        frame, direction = self.pushed[-1]
        self.assertIsInstance(frame, TranscriptionFrame)
        self.assertEqual(frame.text, "where is the tower?")
        self.assertEqual(direction, FrameDirection.UPSTREAM)

    async def test_an_audio_pipeline_is_announced_as_a_realtime_service(self) -> None:
        # Realtime mode is what moves the aggregator's write late enough for a
        # transcript that only exists once the model has answered.
        self.assertTrue(self.service.service_metadata_frame().is_realtime_service)

    async def test_a_text_only_pipeline_is_announced_as_a_plain_service(self) -> None:
        self.service._settings.input_modalities = ("text",)

        self.assertFalse(self.service.service_metadata_frame().is_realtime_service)


class OmniRequestBuildingTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.service = NvidiaOmniLLMService(api_key="not-needed", base_url="http://localhost:8000/v1")

    async def test_audio_turn_parts_are_appended_without_mutating_context(self) -> None:
        params_from_context = {"messages": [{"role": "user", "content": "hi"}]}
        self.service._active_turn_parts = [{"type": "text", "text": "listen"}]

        params = self.service.build_chat_completion_params(params_from_context)

        self.assertEqual(len(params["messages"]), 2)
        self.assertEqual(params["messages"][-1]["content"], [{"type": "text", "text": "listen"}])
        self.assertEqual(len(params_from_context["messages"]), 1)

    async def test_text_turns_send_context_messages_unchanged(self) -> None:
        params_from_context = {"messages": [{"role": "user", "content": "hi"}]}

        params = self.service.build_chat_completion_params(params_from_context)

        self.assertEqual(params["messages"], [{"role": "user", "content": "hi"}])

    async def test_a_universal_audio_message_is_named_the_way_the_endpoint_reads_it(self) -> None:
        # Callers build audio the standard way; the adapter renames it for NIM.
        message = {"role": "user", "content": [audio_message_part(b"\x00\x00", 16000, 1)]}
        context = LLMContext([message])

        params = self.service.get_llm_adapter().get_llm_invocation_params(
            context, system_instruction=None, convert_developer_to_user=False
        )

        part = params["messages"][0]["content"][0]
        self.assertEqual(part["type"], "audio_url")
        self.assertTrue(part["audio_url"]["url"].startswith("data:audio/wav;base64,"))
        # The context keeps the universal shape: only the request is rewritten.
        self.assertEqual(message["content"][0]["type"], "input_audio")

    async def test_a_buffered_utterance_is_named_the_same_way(self) -> None:
        self.service._active_turn_parts = [audio_message_part(b"\x00\x00", 16000, 1)]

        params = self.service.build_chat_completion_params({"messages": []})

        self.assertEqual(params["messages"][-1]["content"][0]["type"], "audio_url")

    async def test_only_the_configured_token_limit_reaches_the_endpoint(self) -> None:
        # Both fields at once leaves the endpoint free to honour either.
        service = NvidiaOmniLLMService(
            api_key="not-needed",
            base_url="http://localhost:8000/v1",
            settings=NvidiaOmniSettings(max_tokens=8192),
        )
        sent: dict = {}

        async def fake_create(**kwargs):
            sent.update(kwargs)
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))])

        service._client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=fake_create)))

        await service.run_inference(LLMContext([{"role": "user", "content": "hi"}]), max_tokens=2048)

        self.assertEqual(sent["max_tokens"], 2048)
        self.assertNotIn("max_completion_tokens", sent)

    async def test_no_token_limit_is_sent_unless_one_is_configured(self) -> None:
        params = self.service.build_chat_completion_params({"messages": []})

        self.assertNotIn("max_tokens", params)
        self.assertNotIn("max_completion_tokens", params)

    async def test_media_modalities_are_rejected_for_pipeline_input(self) -> None:
        with self.assertRaises(ValueError):
            NvidiaOmniLLMService(
                api_key="not-needed",
                base_url="http://localhost:8000/v1",
                settings=NvidiaOmniSettings(input_modalities=("video",)),
            )


if __name__ == "__main__":
    unittest.main()
