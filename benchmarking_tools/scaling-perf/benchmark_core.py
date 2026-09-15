# SPDX-FileCopyrightText: Copyright (c) 2024–2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""Protocol-neutral orchestration for the scaling performance benchmark."""

# ruff: noqa: D101,D102,D103,D105,D107

from __future__ import annotations

import asyncio
import contextlib
import datetime as dt
import sys
import time
import wave
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

CHUNK_DURATION_MS = 32
BOT_INTRO_TIMEOUT = 5
END_OF_RESPONSE_TIMEOUT = 3.0
HARD_DEADLINE_BUFFER = 60
REALTIME_TURN_RESPONSE_TIMEOUT = 45.0
TURN_RESPONSE_TIMEOUT = 10.0
SERVER_METRIC_KEYS = (
    "llm_ttft",
    "tts_ttfb",
    "asr_ttfb",
    "server_e2e",
    "vad_smart_turn",
    "llm_processing_time",
    "llm_tokens_per_sec",
)

_SHUTDOWN_REQUESTED = False


def request_shutdown(signum: int) -> None:
    global _SHUTDOWN_REQUESTED
    _SHUTDOWN_REQUESTED = True
    print(f"\nReceived signal {signum}, shutting down gracefully...")


def log_error(msg: str) -> None:
    timestamp = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[ERROR] {timestamp} - {msg}", file=sys.stderr, flush=True)


def round3(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"{value:.3f}"


def average_or_none(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


class RunLogger:
    """Append-only timestamped log writer for one client."""

    def __init__(self, path: Path):
        self.path = path
        self._lock = asyncio.Lock()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("", encoding="utf-8")

    async def log(self, message: str) -> None:
        timestamp = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        async with self._lock:
            with self.path.open("a", encoding="utf-8") as f:
                f.write(f"[{timestamp}] {message}\n")

    async def log_table(self, title: str, rows: list[tuple[str, str]]) -> None:
        await self.log(title)
        if not rows:
            await self.log("  (no rows)")
            return
        width = max(len(label) for label, _ in rows)
        for label, value in rows:
            await self.log(f"  {label.ljust(width)} : {value}")


@dataclass
class ClientResult:
    """Stable per-client JSON result schema."""

    stream_id: str
    average_latency: float | None
    individual_latencies: list[float]
    valid_latencies: list[float]
    num_turns: int
    num_valid_turns: int
    failed_turns: int
    reverse_barge_ins_count: int
    glitch_detected: bool
    reverse_barge_in_threshold: float
    turn_response_timeout: float
    metrics_start_time: float | None
    test_duration: float
    server_metrics: dict
    rtvi_messages: list[dict[str, Any]]
    realtime_turn_metrics: list[dict[str, Any]]
    timestamp: str
    error: str | None = None


@dataclass(frozen=True)
class AudioFrame:
    audio: bytes
    sample_rate: int
    num_channels: int


class EndOfBotUtterance(Exception):
    """A response completed before producing more audio."""


class FailedBotUtterance(Exception):
    """A response failed without necessarily ending the session."""

    def __init__(self, message: str, *, terminal: bool):
        super().__init__(message)
        self.terminal = terminal


class ProtocolConnectionClosed(Exception):
    """The protocol transport closed unexpectedly."""


class ProtocolClient(ABC):
    """Small transport contract consumed by the shared turn loop."""

    name: str
    uri: str
    server_metric_samples: dict[str, list[float]]
    connect_timeout: float
    requires_response_done = False

    @abstractmethod
    async def __aenter__(self) -> ProtocolClient: ...

    @abstractmethod
    async def __aexit__(self, exc_type, exc, tb) -> None: ...

    @abstractmethod
    async def send_audio(self, frame: AudioFrame) -> None: ...

    @abstractmethod
    async def recv_audio(self, timeout: float | None = None) -> AudioFrame: ...

    @abstractmethod
    async def recover_turn(
        self,
        *,
        timeout: float,
        failure: FailedBotUtterance | None = None,
        intro_timeout: bool = False,
    ) -> None: ...

    @abstractmethod
    def set_collecting_metrics(self, collecting: bool) -> None: ...

    def begin_turn_observation(self) -> Any:
        """Return protocol state needed to diagnose one turn."""
        return None

    async def record_turn_observation(
        self,
        audio_file: Path,
        marker: Any,
        input_end: float | None,
        *,
        error: str | None = None,
    ) -> None:
        """Record optional protocol-specific diagnostics for one turn."""
        del audio_file, marker, input_end, error

    @property
    def ready_log(self) -> str | None:
        """Return optional connection details for the client log."""
        return None

    def result_artifacts(self) -> dict[str, list[dict[str, Any]]]:
        """Return protocol-specific diagnostic collections for the result."""
        return {}


class PerfClient:
    """One synthetic client with a protocol-neutral shared turn loop."""

    def __init__(
        self,
        *,
        stream_id: str,
        protocol_client: ProtocolClient,
        audio_files: list[Path],
        start_delay: float,
        metrics_start_time: float | None,
        session_end_time: float | None,
        test_duration: float,
        reverse_barge_in_threshold: float,
        audio_output_path: Path | None,
        logger: RunLogger,
        turn_response_timeout: float = TURN_RESPONSE_TIMEOUT,
        drain_bot_intro: bool | None = None,
        bot_intro_timeout: float = BOT_INTRO_TIMEOUT,
    ):
        self.stream_id = stream_id
        self.protocol_client = protocol_client
        self.audio_files = audio_files
        self.start_delay = start_delay
        self.metrics_start_time = metrics_start_time
        self.session_end_time = session_end_time
        self.test_duration = test_duration
        self.reverse_barge_in_threshold = reverse_barge_in_threshold
        self.turn_response_timeout = turn_response_timeout
        self.audio_output_path = audio_output_path
        self.logger = logger
        self.drain_bot_intro = drain_bot_intro is not False
        self.bot_intro_timeout = max(0.0, bot_intro_timeout)

        self.latency_values: list[float] = []
        self.valid_latency_values: list[float] = []
        self.failed_turns = 0
        self.input_audio_file_end: dt.datetime | None = None
        self.input_audio_file_end_monotonic: float | None = None
        self.glitch_detected = False
        self.total_reverse_barge_ins = 0
        self.running = True
        self.collecting_metrics = False
        self.silence_running = False
        self.silence_event: asyncio.Event | None = None
        self.audio_params: tuple[int, int, int] | None = None

    def _set_collecting_metrics(self, collecting: bool) -> None:
        self.collecting_metrics = collecting
        self.protocol_client.set_collecting_metrics(collecting)

    def _write_audio_to_wav(self, frame: AudioFrame, wf, *, create_new_file: bool = False):
        sample_rate = frame.sample_rate
        num_channels = frame.num_channels
        audio_data = frame.audio

        if self.audio_output_path is not None and create_new_file and wf is None:
            try:
                self.audio_output_path.parent.mkdir(parents=True, exist_ok=True)
                wf = wave.open(str(self.audio_output_path), "wb")  # noqa: SIM115
                wf.setnchannels(num_channels)
                wf.setsampwidth(2)
                wf.setframerate(sample_rate)
            except Exception as exc:
                log_error(f"Failed to create WAV file {self.audio_output_path}: {exc}")
                return None, None, None, None

        if self.audio_output_path is not None and wf is not None:
            try:
                wf.writeframes(audio_data)
            except Exception as exc:
                log_error(f"Failed to write audio data: {exc}")
                return None, None, None, None
        return wf, sample_rate, num_channels, audio_data

    async def _send_audio_file(self, file_path: Path) -> None:
        assert self.silence_event is not None
        self.silence_event.set()
        await self.logger.log(f"{self.stream_id} sending input audio: {file_path.name}")
        try:
            with wave.open(str(file_path), "rb") as wav_file:
                channels = wav_file.getnchannels()
                rate = wav_file.getframerate()
                width = wav_file.getsampwidth()
                chunk_size = int((rate * channels * CHUNK_DURATION_MS) / 1000) * width
                self.audio_params = (rate, channels, chunk_size)
                while True:
                    chunk = wav_file.readframes(chunk_size // width)
                    if not chunk:
                        break
                    await self.protocol_client.send_audio(AudioFrame(chunk, rate, channels))
                    await asyncio.sleep(CHUNK_DURATION_MS / 1000)
        finally:
            self.input_audio_file_end = dt.datetime.now()
            self.input_audio_file_end_monotonic = time.monotonic()
            await self.logger.log(
                f"{self.stream_id} input audio finished at {self.input_audio_file_end.strftime('%H:%M:%S.%f')[:-3]}"
            )
            self.silence_event.clear()

    async def _silence_sender_loop(self) -> None:
        self.silence_running = True
        try:
            while self.silence_running:
                if _SHUTDOWN_REQUESTED:
                    return
                if self.silence_event is None or self.silence_event.is_set() or self.audio_params is None:
                    await asyncio.sleep(0.1)
                    continue
                rate, channels, chunk_size = self.audio_params
                await self.protocol_client.send_audio(AudioFrame(b"\x00" * chunk_size, rate, channels))
                await asyncio.sleep(CHUNK_DURATION_MS / 1000)
        except ProtocolConnectionClosed:
            return

    async def _drain_utterance(
        self,
        first_frame: AudioFrame,
        wf,
        *,
        detect_glitches: bool,
        drain_timeout: float = END_OF_RESPONSE_TIMEOUT,
    ):
        playback_buffer_duration = 0.0
        last_update_time = None
        chunk_count = 0
        frame = first_frame
        while True:
            wf, sample_rate, channels, audio_data = self._write_audio_to_wav(frame, wf, create_new_file=(wf is None))
            if audio_data and sample_rate and channels:
                current_time = time.time()
                chunk_count += 1
                chunk_duration = (len(audio_data) // (channels * 2)) / sample_rate
                if detect_glitches:
                    if last_update_time is not None:
                        playback_buffer_duration -= current_time - last_update_time
                        if playback_buffer_duration < -0.020:
                            self.glitch_detected = True
                            await self.logger.log(
                                f"{self.stream_id} audio glitch detected: "
                                f"buffer underrun {(-playback_buffer_duration) * 1000:.1f}ms"
                            )
                            playback_buffer_duration = 0
                    playback_buffer_duration += chunk_duration
                    last_update_time = current_time
            try:
                frame = await self.protocol_client.recv_audio(timeout=drain_timeout)
            except EndOfBotUtterance:
                return wf, chunk_count
            except TimeoutError:
                if self.protocol_client.requires_response_done:
                    raise FailedBotUtterance("response timed out before response.done", terminal=False) from None
                return wf, chunk_count

    async def _receive_initial_bot_intro(self, wf):
        try:
            frame = await self.protocol_client.recv_audio(timeout=self.bot_intro_timeout)
        except FailedBotUtterance as exc:
            await self.protocol_client.recover_turn(timeout=self.turn_response_timeout, failure=exc)
            await self.logger.log(f"{self.stream_id} initial bot intro failed: {exc}")
            return wf
        except EndOfBotUtterance:
            await self.logger.log(f"{self.stream_id} initial bot intro completed without audio")
            return wf
        except TimeoutError:
            await self.protocol_client.recover_turn(timeout=self.turn_response_timeout, intro_timeout=True)
            await self.logger.log(f"{self.stream_id} no initial bot intro within {self.bot_intro_timeout:g}s")
            return wf
        await self.logger.log(f"{self.stream_id} received initial bot intro")
        try:
            wf, _ = await self._drain_utterance(frame, wf, detect_glitches=False)
        except FailedBotUtterance as exc:
            await self.protocol_client.recover_turn(timeout=self.turn_response_timeout, failure=exc)
            await self.logger.log(f"{self.stream_id} initial bot intro failed: {exc}")
        return wf

    async def _await_first_bot_frame(
        self, send_task: asyncio.Task, recv_task: asyncio.Task
    ) -> tuple[AudioFrame | None, dt.datetime | None]:
        frame = None
        try:
            done, _ = await asyncio.wait({send_task, recv_task}, return_when=asyncio.FIRST_COMPLETED)
            if recv_task in done:
                return recv_task.result(), dt.datetime.now()
            await send_task
            try:
                frame = await asyncio.wait_for(recv_task, timeout=self.turn_response_timeout)
                return frame, dt.datetime.now()
            except TimeoutError:
                if self.collecting_metrics:
                    self.failed_turns += 1
                    await self.logger.log(
                        f"{self.stream_id} turn timed out: no bot response within "
                        f"{self.turn_response_timeout:.1f}s after input audio finished"
                    )
                return None, None
        finally:
            if frame is None and not recv_task.done():
                recv_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await recv_task

    async def _record_failed_turn(self, audio_file: Path, marker: Any, error: str) -> None:
        if self.collecting_metrics:
            self.failed_turns += 1
        await self.logger.log(f"{self.stream_id} turn failed: {error}")
        await self.protocol_client.record_turn_observation(
            audio_file,
            marker,
            self.input_audio_file_end_monotonic,
            error=error,
        )

    async def _process_conversation_turn(self, audio_file: Path, wf):
        self.input_audio_file_end = None
        self.input_audio_file_end_monotonic = None
        marker = self.protocol_client.begin_turn_observation()
        await self.logger.log(f"{self.stream_id} turn start using {audio_file.name}")
        send_task = asyncio.create_task(self._send_audio_file(audio_file))
        recv_task = asyncio.create_task(self.protocol_client.recv_audio(timeout=None))
        try:
            return await self._run_conversation_turn(audio_file, wf, marker, send_task, recv_task)
        finally:
            if not send_task.done():
                send_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await send_task

    async def _run_conversation_turn(
        self,
        audio_file: Path,
        wf,
        marker: Any,
        send_task: asyncio.Task,
        recv_task: asyncio.Task,
    ):
        try:
            frame, utterance_start = await self._await_first_bot_frame(send_task, recv_task)
        except FailedBotUtterance as exc:
            await send_task
            await self.protocol_client.recover_turn(timeout=self.turn_response_timeout, failure=exc)
            await self._record_failed_turn(audio_file, marker, str(exc))
            return wf
        except EndOfBotUtterance:
            await send_task
            await self._record_failed_turn(audio_file, marker, "response completed without bot audio")
            return wf

        if frame is None or utterance_start is None:
            try:
                await self.protocol_client.recover_turn(timeout=self.turn_response_timeout)
            except Exception as exc:
                await self.protocol_client.record_turn_observation(
                    audio_file,
                    marker,
                    self.input_audio_file_end_monotonic,
                    error=f"unable to synchronize timed-out response: {exc}",
                )
                raise
            await self.protocol_client.record_turn_observation(
                audio_file,
                marker,
                self.input_audio_file_end_monotonic,
                error=f"no bot response within {self.turn_response_timeout:.1f}s",
            )
            return wf

        try:
            wf, _ = await self._drain_utterance(frame, wf, detect_glitches=True)
        except FailedBotUtterance as exc:
            await send_task
            await self.protocol_client.recover_turn(timeout=self.turn_response_timeout, failure=exc)
            await self._record_failed_turn(audio_file, marker, str(exc))
            return wf
        await send_task
        await self.protocol_client.record_turn_observation(audio_file, marker, self.input_audio_file_end_monotonic)

        latency = (
            (utterance_start - self.input_audio_file_end).total_seconds()
            if self.input_audio_file_end is not None
            else None
        )
        if self.collecting_metrics and latency is not None:
            self.latency_values.append(latency)
            if latency < self.reverse_barge_in_threshold:
                self.total_reverse_barge_ins += 1
                await self.logger.log(
                    f"{self.stream_id} turn complete (barge-in) latency={latency:.3f}s "
                    f"< threshold={self.reverse_barge_in_threshold:.3f}s"
                )
            else:
                self.valid_latency_values.append(latency)
                await self.logger.log(f"{self.stream_id} turn complete latency={latency:.3f}s")
        return wf

    async def _continuous_audio_loop(self, wf):
        turn_index = 0
        while self.running and not _SHUTDOWN_REQUESTED:
            now = time.time()
            if self.session_end_time and now >= self.session_end_time:
                self._set_collecting_metrics(False)
                self.running = False
                await self.logger.log(f"{self.stream_id} session window closed")
                break
            if self.metrics_start_time and now >= self.metrics_start_time and not self.collecting_metrics:
                self._set_collecting_metrics(True)
                await self.logger.log(f"{self.stream_id} metrics collection started")
            if (
                self.metrics_start_time
                and self.collecting_metrics
                and now >= self.metrics_start_time + self.test_duration
            ):
                self._set_collecting_metrics(False)
                self.running = False
                await self.logger.log(f"{self.stream_id} metrics collection stopped")
                break
            wf = await self._process_conversation_turn(self.audio_files[turn_index % len(self.audio_files)], wf)
            turn_index += 1
            await asyncio.sleep(0.1)
        return wf

    async def _run_connected_session(self) -> None:
        wf = None
        if self.drain_bot_intro:
            wf = await self._receive_initial_bot_intro(wf)
        if self.metrics_start_time is None:
            self.metrics_start_time = time.time()
            if self.session_end_time is None:
                self.session_end_time = self.metrics_start_time + self.test_duration
        self.silence_event = asyncio.Event()
        self.silence_event.set()
        silence_task = asyncio.create_task(self._silence_sender_loop())
        try:
            wf = await self._continuous_audio_loop(wf)
        finally:
            self.silence_running = False
            self.silence_event.set()
            silence_task.cancel()
            try:
                await silence_task
            except asyncio.CancelledError:
                pass
            except Exception as exc:
                log_error(f"{self.stream_id} silence sender failed during teardown: {exc}")
        if wf is not None:
            wf.close()

    def _hard_deadline(self) -> float:
        fallback = self.test_duration + HARD_DEADLINE_BUFFER
        if self.session_end_time:
            return max(self.session_end_time - time.time() + HARD_DEADLINE_BUFFER, HARD_DEADLINE_BUFFER)
        if self.metrics_start_time:
            remaining = self.metrics_start_time + self.test_duration + HARD_DEADLINE_BUFFER - time.time()
            return max(remaining, 60) if remaining > 0 else fallback
        return fallback

    async def _run_with_hard_deadline(self, hard_deadline: float) -> None:
        deadline = asyncio.timeout(hard_deadline)
        try:
            async with deadline:
                await self._run_connected_session()
        except TimeoutError as exc:
            if deadline.expired():
                raise RuntimeError(f"Hard deadline reached ({hard_deadline:.0f}s)") from exc
            raise RuntimeError("Benchmark session operation timed out") from exc

    async def run(self) -> ClientResult:
        await self.logger.log(f"{self.stream_id} starting client uri={self.protocol_client.uri.partition('?')[0]}")
        if self.start_delay > 0:
            await self.logger.log(f"{self.stream_id} waiting start_delay={self.start_delay:.2f}s")
            await asyncio.sleep(self.start_delay)

        error = None
        hard_deadline = self._hard_deadline()
        try:
            async with self.protocol_client:
                if self.protocol_client.ready_log:
                    await self.logger.log(f"{self.stream_id} {self.protocol_client.ready_log}")
                await self._run_with_hard_deadline(hard_deadline)
        except TimeoutError:
            error = f"WebSocket/session setup timed out ({self.protocol_client.connect_timeout:.0f}s)"
        except ProtocolConnectionClosed:
            error = "WebSocket connection closed"
        except Exception as exc:
            error = str(exc) or exc.__class__.__name__

        if error:
            await self.logger.log(f"{self.stream_id} finished with error: {error}")
        elif not self.valid_latency_values:
            error = "No valid benchmark turns completed"
            await self.logger.log(f"{self.stream_id} finished with error: {error}")
        else:
            await self.logger.log(f"{self.stream_id} finished successfully")

        samples = self.protocol_client.server_metric_samples
        averages = {key: average_or_none(values) for key, values in samples.items()}
        counts = {key: len(values) for key, values in samples.items()}
        artifacts = self.protocol_client.result_artifacts()
        result = ClientResult(
            stream_id=self.stream_id,
            average_latency=average_or_none(self.valid_latency_values),
            individual_latencies=self.latency_values,
            valid_latencies=self.valid_latency_values,
            num_turns=len(self.latency_values),
            num_valid_turns=len(self.valid_latency_values),
            failed_turns=self.failed_turns,
            reverse_barge_ins_count=self.total_reverse_barge_ins,
            glitch_detected=self.glitch_detected,
            reverse_barge_in_threshold=self.reverse_barge_in_threshold,
            turn_response_timeout=self.turn_response_timeout,
            metrics_start_time=self.metrics_start_time,
            test_duration=self.test_duration,
            server_metrics={"samples": samples, "average": averages, "sample_counts": counts},
            rtvi_messages=artifacts.get("rtvi_messages", []),
            realtime_turn_metrics=artifacts.get("realtime_turn_metrics", []),
            timestamp=dt.datetime.now().isoformat(),
            error=error,
        )
        await self.logger.log_table(
            f"{self.stream_id} latency breakdown summary",
            [
                ("client_avg_latency", round3(result.average_latency)),
                ("client_valid_turns", str(result.num_valid_turns)),
                ("client_total_turns", str(result.num_turns)),
                ("client_barge_ins", str(result.reverse_barge_ins_count)),
                ("client_failed_turns", str(result.failed_turns)),
                ("glitch_detected", str(result.glitch_detected)),
                *((key, round3(averages.get(key))) for key in SERVER_METRIC_KEYS),
            ],
        )
        return result
