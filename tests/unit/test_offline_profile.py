# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D102, D107

"""Passive profiles preserve native work and distinguish measured wait boundaries."""

import asyncio
import json
import os
import tempfile
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams

from examples.generic.offline.detection import SampleTimedSileroMixin
from examples.generic.offline.diagnostics import (
    ProfiledExecutor,
    ServerDiagnostics,
    bind_diagnostics,
    profile_vad_window,
    reset_diagnostics,
)
from examples.generic.offline.models import AudioSource
from examples.generic.offline.recognition import ChunkProgressASR
from examples.generic.offline.websocket import serve_offline_session


class ObservedVAD(SampleTimedSileroMixin, SileroVADAnalyzer):
    """Use the real pinned analyzer, replacing only the model boundary."""


class Model:
    """A controlled scalar classifier with no neural model or provider calls."""

    def __init__(self, *, delay=0, entered=None, release=None):
        self.delay, self.entered, self.release = delay, entered, release

    def __call__(self, audio, sample_rate):
        if self.entered:
            self.entered.set()
        if self.release:
            self.release.wait(2)
        if self.delay:
            time.sleep(self.delay)
        return np.array([[0.9]])

    def reset_states(self):
        pass


class PassiveProfileTests(unittest.IsolatedAsyncioTestCase):
    """Controlled thread/loop delays prove each scope without a production model."""

    def analyzer(self, sink, model):
        token = bind_diagnostics(sink)
        try:
            with patch("pipecat.audio.vad.silero.SileroOnnxModel", return_value=model):
                analyzer = ObservedVAD(sample_rate=16000, params=VADParams(min_volume=0))
        finally:
            reset_diagnostics(token)
        analyzer.set_sample_rate(16000)
        self.addAsyncCleanup(analyzer.cleanup)
        return analyzer

    async def until(self, condition):
        async with asyncio.timeout(2):
            while not condition():
                await asyncio.sleep(0.001)

    async def test_native_states_are_unchanged_and_profile_is_bounded(self):
        sink = ServerDiagnostics(profiling=True)
        observed = self.analyzer(sink, Model())
        with patch("pipecat.audio.vad.silero.SileroOnnxModel", return_value=Model()):
            native = SileroVADAnalyzer(sample_rate=16000, params=VADParams(min_volume=0))
        native.set_sample_rate(16000)
        self.addAsyncCleanup(native.cleanup)
        pcm = b"\x10\x20" * 512
        for index in range(24):
            observed.set_source_window_end((index + 1) * 512)
            self.assertEqual(await observed.analyze_audio(pcm), await native.analyze_audio(pcm))
        value = sink.snapshot()
        profile = value["profiling"]
        self.assertEqual(profile["total_vad_windows"], 24)
        self.assertEqual(len(profile["slow_vad_windows"]), 16)
        self.assertEqual(profile["inflight_vad_windows"], [])
        self.assertEqual(profile["dropped_vad_records"], 0)
        self.assertEqual(value["duration_summaries"]["vad_executor_queue"]["count"], 24)
        self.assertEqual(value["duration_summaries"]["vad_model_thread_cpu"]["count"], 24)
        self.assertEqual(value["duration_summaries"]["vad_executor_thread_cpu"]["count"], 24)
        for window in profile["slow_vad_windows"]:
            self.assertLessEqual(window["submitted_elapsed_ns"], window["executor_started_elapsed_ns"])
            self.assertLessEqual(window["executor_finished_elapsed_ns"], window["event_loop_resumed_elapsed_ns"])
            self.assertEqual(window["outcome"], "returned")
            self.assertEqual(window["model_calls"], 1)
        self.assertNotIn("profiling", ServerDiagnostics().snapshot())

    async def test_real_executor_queue_is_separate_from_model_wall_and_calling_thread_CPU(self):
        sink = ServerDiagnostics(profiling=True)
        analyzer = self.analyzer(sink, Model(delay=0.01))
        self.assertIsInstance(analyzer._executor, ProfiledExecutor)
        entered, release = threading.Event(), threading.Event()

        def blocker():
            entered.set()
            release.wait(2)

        blocker_future = analyzer._executor.submit(blocker)
        await asyncio.to_thread(entered.wait, 1)
        analyzer.set_source_window_end(512)
        task = asyncio.create_task(analyzer.analyze_audio(b"\x10\x20" * 512))
        try:
            await self.until(lambda: bool(sink.snapshot()["profiling"]["inflight_vad_windows"]))
            await asyncio.sleep(0.03)
            release.set()
            await task
        finally:
            release.set()
            await asyncio.wrap_future(blocker_future)
        value = sink.snapshot()
        self.assertGreater(value["duration_summaries"]["vad_executor_queue"]["sum_ms"], 20)
        window = value["profiling"]["slow_vad_windows"][0]
        self.assertGreater(window["model_wall_ns"], 8_000_000)
        self.assertLess(window["model_thread_cpu_ns"], window["model_wall_ns"])

    async def test_cancelled_awaiter_does_not_claim_executor_finished_or_join_native_work(self):
        sink = ServerDiagnostics(profiling=True)
        entered, release = threading.Event(), threading.Event()
        analyzer = self.analyzer(sink, Model(entered=entered, release=release))
        analyzer.set_source_window_end(512)
        task = asyncio.create_task(analyzer.analyze_audio(b"\x10\x20" * 512))
        await asyncio.to_thread(entered.wait, 1)
        try:
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await asyncio.wait_for(task, 1)
            value = sink.snapshot()["profiling"]
            self.assertEqual(len(value["inflight_vad_windows"]), 1)
            window = value["inflight_vad_windows"][0]
            self.assertTrue(window["awaiter_cancelled"])
            self.assertIsNone(window["executor_finished_elapsed_ns"])
            self.assertIsNone(window["executor_thread_cpu_ns"])
            self.assertNotIn("vad_executor_finish_to_loop_resume", sink.snapshot()["duration_summaries"])
        finally:
            release.set()
            await self.until(lambda: not sink.snapshot()["profiling"]["inflight_vad_windows"])
        finished = sink.snapshot()["profiling"]["slow_vad_windows"][0]
        self.assertEqual(finished["outcome"], "cancelled")
        self.assertIsNotNone(finished["executor_finished_elapsed_ns"])

    async def test_completion_to_loop_resume_and_loop_lag_are_observed_without_catchup(self):
        sink = ServerDiagnostics(profiling=True)
        entered, release = threading.Event(), threading.Event()
        analyzer = self.analyzer(sink, Model(entered=entered, release=release))
        sink.start_loop_observer(interval_s=0.005)
        self.addAsyncCleanup(sink.stop_loop_observer)
        analyzer.set_source_window_end(512)
        task = asyncio.create_task(analyzer.analyze_audio(b"\x10\x20" * 512))
        await asyncio.to_thread(entered.wait, 1)
        timer = threading.Timer(0.005, release.set)
        timer.start()
        try:
            # Deliberately block only this fixture's loop while its worker finishes.
            time.sleep(0.04)
            await task
            await asyncio.sleep(0.01)
        finally:
            release.set()
            timer.join(1)
            await sink.stop_loop_observer()
        value = sink.snapshot()
        self.assertGreater(value["duration_summaries"]["vad_executor_finish_to_loop_resume"]["max_ms"], 20)
        self.assertGreater(value["duration_summaries"]["event_loop_lag"]["max_ms"], 20)
        self.assertLess(value["duration_summaries"]["event_loop_lag"]["count"], 10)
        self.assertTrue(value["profiling"]["loop_observer_stopped"])

    async def test_submission_failure_preserves_original_error_and_profile_context_is_reset(self):
        sink = ServerDiagnostics(profiling=True)
        executor = ProfiledExecutor(ThreadPoolExecutor(max_workers=1))
        executor.shutdown(wait=True)
        with self.assertRaises(RuntimeError), profile_vad_window(sink, 512):
            executor.submit(lambda: 1)
        profile = sink.snapshot()["profiling"]
        self.assertFalse(profile["inflight_vad_windows"])
        self.assertEqual(profile["slow_vad_windows"][0]["outcome"], "error")
        self.assertNotIn("private", json.dumps(profile))

    async def test_trusted_pool_join_and_persistence_scope_are_separate_from_handler_return(self):
        holder = {"lease_sha256": "a" * 64, "handler_started_monotonic_ns": time.monotonic_ns()}
        socket = SimpleNamespace(scope={"nva.offline.pool": holder})

        async def native(*args, **kwargs):
            return None

        with tempfile.TemporaryDirectory() as root:
            with (
                patch.dict(os.environ, {"NVA_OFFLINE_SERVER_METRICS_DIR": root, "NVA_OFFLINE_PROFILE": "1"}),
                patch("examples.generic.offline.websocket._serve_offline_session", native),
            ):
                await serve_offline_session(socket, None, root)
            evidence = json.loads((Path(root) / (holder["server_metrics_session_id"] + ".json")).read_text())
        events = evidence["events"]
        joined = next(x for x in events if x["name"] == "pool_session_bound")
        self.assertEqual(joined["lease_sha256"], holder["lease_sha256"])
        self.assertEqual(next(x for x in events if x["name"] == "session_returned")["boundary"], "pre_persistence")
        self.assertTrue(any(x["name"] == "session_finalizer_entered" for x in events))
        self.assertTrue(any(x["name"] == "diagnostics_persistence_started" for x in events))
        self.assertFalse(any(x["name"] == "diagnostics_persistence_completed" for x in events))
        self.assertGreaterEqual(
            holder["diagnostics_persistence_completed_monotonic_ns"],
            holder["diagnostics_persistence_started_monotonic_ns"],
        )
        self.assertTrue(evidence["profiling"]["loop_observer_stopped"])
        self.assertNotIn("lease_id", json.dumps(evidence))

    async def test_ASR_profile_keeps_native_progress_tape_and_records_blank_results_without_bodies(self):
        from test_offline_recognition import ManualClock, response, result

        async def execute(sink):
            clock = ManualClock()

            def stream(audio):
                list(audio)
                clock.advance(1000)
                yield response(result(512, "fixture", final=False))
                clock.advance(1000)
                yield response(result(1024, "fixture", final=True), result(1536, "", final=False))

            token = bind_diagnostics(sink)
            try:
                recognizer = ChunkProgressASR(stream, clock=clock)
            finally:
                reset_diagnostics(token)
            return await recognizer.recognize(AudioSource(b"\0\0" * 1536))

        plain = await execute(None)
        sink = ServerDiagnostics(profiling=True)
        observed = await execute(sink)
        self.assertEqual(observed.events, plain.events)
        self.assertEqual(observed.metrics, plain.metrics)
        evidence = sink.snapshot()
        phases = [x["name"] for x in evidence["events"]]
        self.assertIn("asr_thread_dispatched", phases)
        self.assertIn("asr_thread_entered", phases)
        self.assertIn("asr_thread_finished", phases)
        self.assertIn("asr_first_rpc_response_seen", phases)
        progress = [x for x in evidence["events"] if x["name"] == "asr_result_progress"]
        self.assertEqual([x["processed_sample"] for x in progress], [512, 1024, 1536])
        self.assertTrue(all(x["last_supplied_sample"] == 1536 for x in progress))
        self.assertEqual(evidence["duration_summaries"]["asr_callback_to_actor_lock"]["count"], 2)
        self.assertNotIn("fixture", json.dumps(evidence))
