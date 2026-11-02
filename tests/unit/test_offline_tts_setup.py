# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D102, D107

"""Exercise real pinned setup/queues with only the remote unary stub replaced."""

import asyncio
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from grpc import FutureCancelledError, FutureTimeoutError, RpcError, StatusCode
from pipecat.clocks.system_clock import SystemClock
from pipecat.frames.frames import MetricsFrame
from pipecat.metrics.metrics import ProcessingMetricsData, TTFATMetricsData, TTFBMetricsData
from pipecat.processors.frame_processor import FrameProcessor, FrameProcessorSetup
from pipecat.services.nvidia.tts import NvidiaTTSSettings
from pipecat.utils.asyncio.task_manager import TaskManager
from test_offline_native_response import ScriptedLLM

from examples.generic.offline.native_response import (
    CapturingNativeNvidiaTTSService,
    NativeResponseSession,
    native_metric_values,
)
from examples.shared.nvidia_tts_explicit_eof import ExplicitEOFNvidiaTTSService


class Recorder:
    """Record bounded synthetic scalar observations from both threads."""

    def __init__(self):
        self.events = []
        self.lock = threading.Lock()

    def record(self, name, **fields):
        with self.lock:
            self.events.append((name, fields))


class LocalRpcError(RpcError):
    """Keep native gRPC error conversion active without a service."""

    def code(self):
        return StatusCode.UNAVAILABLE

    def details(self):
        return "controlled configuration failure"


class ControlledFuture:
    """Use the real future contract, with controlled completion/cancellation."""

    def __init__(self, *, ignore_cancel=False, error=None):
        self.started = threading.Event()
        self.release = threading.Event()
        self.finished = threading.Event()
        self.ignore_cancel = ignore_cancel
        self.error = error
        self.cancelled = False
        self.value = object()
        self.thread = None
        self.rpc_deadline = None
        self.requests = []

    def future(self, request, *, timeout):
        self.requests.append(request.SerializeToString())
        self.thread = threading.get_ident()
        self.rpc_deadline = timeout
        self.started.set()
        return self

    def result(self, *, timeout):
        try:
            if not self.release.wait(None if self.ignore_cancel else timeout):
                raise FutureTimeoutError()
            if self.cancelled and not self.ignore_cancel:
                raise FutureCancelledError()
            if self.error is not None:
                raise self.error
            return self.value
        finally:
            self.finished.set()

    def cancel(self):
        self.cancelled = True
        if not self.ignore_cancel:
            self.release.set()
        return True


class OfflineTTSSetupTests(unittest.IsolatedAsyncioTestCase):
    """Keep native init/helper/setup/cleanup, replacing only the unary RPC."""

    async def asyncSetUp(self):
        self.services = []
        self.calls = []
        self.auths = []
        self.recorder = Recorder()
        self.loop_thread = threading.get_ident()
        self.clock = SystemClock()
        self.clock.start()
        self.setup = FrameProcessorSetup(
            clock=self.clock,
            task_manager=TaskManager(),
            pipeline_worker=None,
            audio_in_sample_rate=16000,
            audio_out_sample_rate=22050,
            enable_metrics=True,
        )

        def auth(*args):
            result = SimpleNamespace(channel=SimpleNamespace(close=Mock()))
            self.auths.append(result)
            return result

        def service(auth):
            call = self.calls[len(self.auths) - 1]
            return SimpleNamespace(auth=auth, stub=SimpleNamespace(GetRivaSynthesisConfig=call))

        prefix = "pipecat.services.nvidia.tts.riva.client."
        self.enterContext(patch(prefix + "Auth", side_effect=auth))
        self.enterContext(patch(prefix + "SpeechSynthesisService", side_effect=service))

    async def asyncTearDown(self):
        for call in self.calls:
            call.release.set()
        await asyncio.gather(*(service.cleanup() for service in self.services))

    def make_service(self, *, call=None, timeout=1, diagnostics=True):
        call = call or ControlledFuture()
        self.calls.append(call)
        service = CapturingNativeNvidiaTTSService(
            capture=AsyncMock(),
            use_ssl=False,
            settings=NvidiaTTSSettings(synthesis_mode="stitched"),
            config_timeout_s=timeout,
            diagnostics=self.recorder if diagnostics else None,
        )
        self.services.append(service)
        return service, call

    async def wait_for(self, condition):
        async with asyncio.timeout(2):
            while not condition():
                await asyncio.sleep(0)

    async def test_six_native_setups_overlap_without_blocking_loop_or_duplicating_rpc(self):
        pairs = [self.make_service() for _ in range(6)]
        setup_threads = []
        native_setup = FrameProcessor.setup

        async def observe_native_setup(processor, setup):
            setup_threads.append(threading.get_ident())
            return await native_setup(processor, setup)

        with patch.object(FrameProcessor, "setup", observe_native_setup):
            tasks = [asyncio.create_task(service.setup(self.setup)) for service, _ in pairs]
            await self.wait_for(lambda: all(call.started.is_set() for _, call in pairs))
            # All six blocking unary waits coexist while this loop progresses.
            self.assertTrue(all(not task.done() for task in tasks))
            self.assertTrue(all(call.thread != self.loop_thread for _, call in pairs))
            for _, call in pairs:
                call.release.set()
            await asyncio.gather(*tasks)
        self.assertEqual(set(setup_threads), {self.loop_thread})
        self.assertEqual(len(self.auths), 6)
        for service, call in pairs:
            self.assertIs(service._config, call.value)
            self.assertIs(service._service.stub.GetRivaSynthesisConfig, call)
            self.assertEqual(call.requests, [b""])
            self.assertGreater(call.rpc_deadline, 0)
            self.assertLessEqual(call.rpc_deadline, 1)
            self.assertEqual(service.sample_rate, 22050)
            self.assertIs(service._setup, self.setup)
        self.assertEqual(sum(name == "tts_config_rpc_start" for name, _ in self.recorder.events), 6)
        self.assertEqual(sum(name == "tts_native_setup_start" for name, _ in self.recorder.events), 6)

    async def test_cancellation_cancels_only_owned_call_and_prevents_config_assignment(self):
        service, call = self.make_service()
        task = asyncio.create_task(service.setup(self.setup))
        await self.wait_for(call.started.is_set)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        await self.wait_for(call.finished.is_set)
        self.assertTrue(call.cancelled)
        self.assertIsNone(service._config)
        self.assertIsNone(service._service)
        self.auths[0].channel.close.assert_called_once()

    async def test_cancelling_one_session_preserves_another_sessions_rpc_and_client(self):
        first, first_call = self.make_service()
        second, second_call = self.make_service()
        tasks = [asyncio.create_task(service.setup(self.setup)) for service in (first, second)]
        await self.wait_for(lambda: first_call.started.is_set() and second_call.started.is_set())
        tasks[0].cancel()
        with self.assertRaises(asyncio.CancelledError):
            await tasks[0]
        self.assertTrue(first_call.cancelled)
        self.assertFalse(second_call.cancelled)
        self.auths[1].channel.close.assert_not_called()
        self.assertFalse(tasks[1].done())
        second_call.release.set()
        await tasks[1]
        self.assertIs(second._config, second_call.value)
        self.assertIsNotNone(second._service)

    async def test_cancelled_response_session_start_cleans_partially_initialized_native_services(self):
        instances = {}

        def llm_factory(capture):
            instances["llm"] = ScriptedLLM(capture)
            return instances["llm"]

        def tts_factory(capture):
            service, call = self.make_service()
            service._native_capture = capture
            instances.update(tts=service, call=call)
            return service

        session = NativeResponseSession(llm_factory, tts_factory, sample_rate=22050)
        task = asyncio.create_task(session.start())
        await self.wait_for(lambda: "call" in instances and instances["call"].started.is_set())
        task.cancel()
        with self.assertRaises(asyncio.CancelledError) as raised:
            await task
        self.assertEqual(getattr(raised.exception, "__notes__", []), [])
        self.assertTrue(instances["call"].cancelled)
        self.assertIsNone(instances["tts"]._service)
        self.assertTrue(instances["llm"]._client.is_closed())
        self.assertFalse(session._manager.current_tasks())

    async def test_timeout_retires_noncooperative_late_result(self):
        service, call = self.make_service(call=ControlledFuture(ignore_cancel=True), timeout=0.02)
        with self.assertRaises(TimeoutError):
            await service.setup(self.setup)
        self.assertTrue(call.cancelled)
        self.assertIsNone(service._config)
        call.release.set()
        await self.wait_for(call.finished.is_set)
        await asyncio.sleep(0)
        self.assertIsNone(service._config)
        self.assertIsNone(service._service)

    async def test_explicit_future_deadline_failure_remains_failure(self):
        service, call = self.make_service(timeout=0.02)
        with self.assertRaises(TimeoutError):
            await service.setup(self.setup)
        self.assertTrue(call.cancelled)
        self.assertIsNone(service._config)

    async def test_native_grpc_error_conversion_is_preserved(self):
        error = LocalRpcError()
        service, call = self.make_service(call=ControlledFuture(error=error))
        call.release.set()
        with self.assertRaisesRegex(RuntimeError, "startup failed while fetching synthesis config") as raised:
            await service.setup(self.setup)
        self.assertIs(raised.exception.__cause__, error)
        self.assertIsNone(service._service)
        self.auths[0].channel.close.assert_called_once()

    async def test_cancel_while_worker_is_queued_does_not_start_an_rpc(self):
        executor = ThreadPoolExecutor(max_workers=1)
        asyncio.get_running_loop().set_default_executor(executor)
        blocker_started, blocker_release = threading.Event(), threading.Event()

        def block():
            blocker_started.set()
            blocker_release.wait(2)

        blocker = asyncio.create_task(asyncio.to_thread(block))
        await self.wait_for(blocker_started.is_set)
        try:
            service, call = self.make_service()
            task = asyncio.create_task(service.setup(self.setup))
            await self.wait_for(lambda: len(self.auths) == 1)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
        finally:
            blocker_release.set()
            await blocker
        await asyncio.sleep(0)
        self.assertFalse(call.started.is_set())
        self.assertIsNone(service._service)

    async def test_diagnostic_sink_failure_does_not_change_native_result(self):
        service, call = self.make_service()
        service._diagnostics = SimpleNamespace(record=Mock(side_effect=RuntimeError("diagnostics unavailable")))
        call.release.set()
        await service.setup(self.setup)
        self.assertIs(service._config, call.value)

    async def test_version_guard_fails_before_client_creation(self):
        service, _ = self.make_service()
        with (
            patch("examples.generic.offline.native_response.version", return_value="1.13.0"),
            self.assertRaisesRegex(RuntimeError, "exactly Pipecat 1.12.0"),
        ):
            await service.setup(self.setup)
        self.assertEqual(self.auths, [])

    def test_offline_only_deadline_defaults_and_bounds(self):
        offline, _ = self.make_service()
        online = ExplicitEOFNvidiaTTSService(
            settings=NvidiaTTSSettings(synthesis_mode="stitched"), stop_frame_timeout_s=None
        )
        self.assertEqual(offline._tts_provider_idle_timeout_s, 60)
        self.assertEqual(online._tts_provider_idle_timeout_s, 600)
        for timeout in (0, -1, 61, True, float("nan"), float("inf")):
            with self.subTest(timeout=timeout), self.assertRaises(ValueError):
                self.make_service(timeout=timeout)

    def test_numeric_metrics_exclude_nonlatency_private_or_nonfinite_values(self):
        frame = MetricsFrame(
            data=[
                TTFBMetricsData(processor="NativeLLM#1", value=0.125),
                TTFATMetricsData(processor="NativeLLM#1", ttfat=0.2, ttfb=0.125, thinking_time=0.075),
                ProcessingMetricsData(processor="NativeTTS#2", value=0.25),
                TTFBMetricsData(processor="private label with whitespace", value=0.5),
                ProcessingMetricsData(processor="NativeTTS#2", value=float("nan")),
                ProcessingMetricsData(processor="NativeTTS#2", value=1e308),
            ]
        )
        values = native_metric_values(frame)
        self.assertEqual([value["value_ms"] for value in values], [125, 200, 250, 500])
        self.assertEqual(
            [value["processor"] for value in values], ["NativeLLM#1", "NativeLLM#1", "NativeTTS#2", "other"]
        )
        self.assertTrue(all(set(value) == {"metric_type", "processor", "value_ms"} for value in values))


if __name__ == "__main__":
    unittest.main()
