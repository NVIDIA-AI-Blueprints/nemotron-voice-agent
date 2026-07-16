# SPDX-FileCopyrightText: Copyright (c) 2024–2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""Voice-agent benchmark runner.

Single-client voice-agent benchmark + result aggregator.

What this script does
-----------------

Default mode (no ``--aggregate-*`` flag): act as **one** synthetic voice
client against a running Nemotron Voice Agent server.

The flow per turn:
  1. Open a wss WebSocket to ``wss://<host>:<port>/api/ws?session_id=...``
     (TLS verification disabled, since the server uses a self-signed cert).
  2. Receive and discard the bot's initial intro utterance.
  3. Stream a WAV file from ``--dataset-dir`` to the server in 32 ms chunks
     (the simulated user "speaking").
  4. After the WAV ends, send PCM silence so VAD sees a clean turn boundary.
  5. Read incoming audio frames; the first chunk that arrives more than
     ``--reverse-barge-in-threshold`` seconds after the user finished is
     counted as the *real* response. Anything earlier is treated as a
     reverse barge-in (the server racing the end of the user's
     utterance) and is drained but not timed.
  6. Server-side timing breakdowns arrive as RTVI ``message`` frames on the
     same WebSocket; they're parsed alongside the audio.

When the metric window (``--metrics-start-time`` → +``--test-duration``)
expires, the client closes the connection, writes its result JSON to
``--result-path``, and exits.

Concurrency is **not handled here.** ``simulate_concurrency.sh`` spawns N
parallel copies with synchronized ``--metrics-start-time`` /
``--session-end-time`` so every worker measures over the same wall-clock
window.

Aggregation modes
-----------------

* ``--aggregate-run-dir DIR`` folds the ``client_*/result_*.json`` files
  under ``DIR`` into a single ``benchmark_summary.json``.
* ``--aggregate-suite-dir DIR`` folds every ``run_<N>_clients/benchmark_summary.json``
  under ``DIR`` into ``results.{tsv,txt,json}`` for the whole sweep.

These modes keep the orchestrator script free of embedded Python; the shell
only has to spawn/wait on processes.
"""

# ruff: noqa: D101,D102,D103,D107

from __future__ import annotations

import argparse
import asyncio
import contextlib
import datetime as dt
import io
import json
import math
import signal
import ssl
import struct
import sys
import time
import wave
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import websockets
from pipecat.frames.protobufs import frames_pb2
from websockets.exceptions import ConnectionClosed

CHUNK_DURATION_MS = 32
WS_CONNECT_TIMEOUT = 30
BOT_INTRO_TIMEOUT = 5
END_OF_RESPONSE_TIMEOUT = 3.0
HARD_DEADLINE_BUFFER = 60
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


def _signal_handler(signum, frame):
    del frame
    global _SHUTDOWN_REQUESTED
    _SHUTDOWN_REQUESTED = True
    print(f"\nReceived signal {signum}, shutting down gracefully...")


signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)


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


def categorize_processor(processor: str) -> str:
    name = processor.lower()
    if "asr" in name or "stt" in name:
        return "asr"
    if "tts" in name:
        return "tts"
    if "llm" in name:
        return "llm"
    return ""


class RunLogger:
    """Append-only timestamped log writer for a single client run.

    Used as a side channel for turn-by-turn diagnostics so the result JSON
    can stay focused on metrics. Safe to call concurrently from multiple
    coroutines (a per-instance asyncio lock serializes writes).
    """

    def __init__(self, path: Path):
        self.path = path
        self._lock = asyncio.Lock()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("", encoding="utf-8")

    async def log(self, message: str) -> None:
        timestamp = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        line = f"[{timestamp}] {message}\n"
        async with self._lock:
            with self.path.open("a", encoding="utf-8") as f:
                f.write(line)

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
    """Per-client metrics produced by one ``PerfClient.run()``.

    Serialized to ``--result-path`` as JSON. ``--aggregate-run-dir`` reads
    these files back to build the per-run ``benchmark_summary.json``.

    Attributes:
        stream_id: WebSocket session identifier (matches the ``client_<i>_<id>``
            output directory name).
        average_latency: Mean of *valid* response latencies (above the reverse
            barge-in threshold), or ``None`` when no valid turn completed
            inside the metric window. Used by the aggregator for headline
            averages and p95.
        individual_latencies: One entry per timed turn (seconds), including
            barge-in latencies. Useful for post-hoc analysis.
        valid_latencies: Subset of ``individual_latencies`` >=
            ``reverse_barge_in_threshold``; what averages and p95 are computed
            over.
        num_turns: Total timed turns recorded during the metric window
            (``len(individual_latencies)``).
        num_valid_turns: ``len(valid_latencies)``. The aggregator uses
            ``num_valid_turns > 0`` as the per-client success gate.
        failed_turns: Turns whose first bot frame did not arrive inside
            ``--turn-response-timeout`` after input audio finished and were
            abandoned without recording a latency.
        reverse_barge_ins_count: Turns counted as reverse barge-ins
            (latency below the threshold) inside the metric window.
        glitch_detected: ``True`` when at least one output buffer underrun
            was seen during a timed turn.
        reverse_barge_in_threshold: Echoed-back configuration value (seconds).
        turn_response_timeout: Echoed-back configuration value (seconds).
        metrics_start_time: Unix epoch when the metric window opened.
        test_duration: Configured length of the metric window (seconds).
        server_metrics: ``{"samples": {...}, "average": {...}, "sample_counts": {...}}``
            for each ``SERVER_METRIC_KEYS`` entry.
        rtvi_messages: Every RTVI message received during the session,
            preserved for offline post-mortem analysis.
        timestamp: ISO-8601 wall-clock time when the result was written.
        error: Human-readable failure reason, or ``None`` on success.
    """

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
    timestamp: str
    error: str | None = None


type FirstBotFrame = tuple[bytes | None, dt.datetime | None]


class PerfClient:
    """One synthetic voice client driving a single WebSocket session.

    Owns the full lifecycle for one simulated user:

    * connects to ``wss://<host>:<port>/api/ws``,
    * cycles through ``audio_files`` as user utterances (one per turn),
    * pads gaps with silence so the server's VAD detects clean turn ends,
    * times every bot response and flags reverse barge-ins post-hoc
      (latencies below ``reverse_barge_in_threshold`` are recorded but
      excluded from valid-latency averages),
    * abandons turns where no bot frame arrives within
      ``turn_response_timeout`` after input audio finished and continues to the
      next turn,
    * records RTVI server-side metric samples (``server_metric_samples``),
    * detects audio glitches (output buffer underruns),
    * returns a :class:`ClientResult` summarizing the session.

    Designed to be instantiated once per session by ``async_main`` (default
    mode) — the orchestrator script provides per-client uniqueness by
    spawning N processes, not by reusing this class within one process.
    """

    def __init__(
        self,
        *,
        stream_id: str,
        host: str,
        port: int,
        audio_files: list[Path],
        start_delay: float,
        metrics_start_time: float | None,
        session_end_time: float | None,
        test_duration: float,
        reverse_barge_in_threshold: float,
        audio_output_path: Path | None,
        logger: RunLogger,
        turn_response_timeout: float = TURN_RESPONSE_TIMEOUT,
    ):
        self.stream_id = stream_id
        self.host = host
        self.port = port
        self.audio_files = audio_files
        self.start_delay = start_delay
        self.metrics_start_time = metrics_start_time
        self.session_end_time = session_end_time
        self.test_duration = test_duration
        self.reverse_barge_in_threshold = reverse_barge_in_threshold
        self.turn_response_timeout = turn_response_timeout
        self.audio_output_path = audio_output_path
        self.logger = logger

        self.latency_values: list[float] = []
        self.valid_latency_values: list[float] = []
        self.failed_turns = 0
        self.timestamps: dict[str, dt.datetime | None] = {"input_audio_file_end": None}
        self.glitch_detected = False
        self.total_reverse_barge_ins = 0
        self.server_metric_samples: dict[str, list[float]] = {key: [] for key in SERVER_METRIC_KEYS}
        self.rtvi_messages: list[dict[str, Any]] = []
        self.running = True
        self.collecting_metrics = False
        self.silence_running = False
        self.silence_event: asyncio.Event | None = None
        self.audio_params: tuple[int, int, int] | None = None
        self._pending_llm_completion_tokens: list[float] = []

    @property
    def uri(self) -> str:
        return f"wss://{self.host}:{self.port}/api/ws?session_id={self.stream_id}"

    async def _process_server_message(self, message: dict) -> None:
        if not isinstance(message, dict):
            return

        message_type = str(message.get("type", "unknown"))
        self.rtvi_messages.append(
            {
                "timestamp": dt.datetime.now().isoformat(),
                "type": message_type,
                "data": message,
            }
        )
        await self.logger.log(f"{self.stream_id} RTVI message: {json.dumps(message, sort_keys=True)}")

        if message_type == "server-message" and isinstance(message.get("data"), dict):
            nested_message = message["data"]
            nested_type = nested_message.get("type")
            if isinstance(nested_type, str):
                message = nested_message
                message_type = nested_type

        if message_type == "user-bot-latency":
            latency = message.get("latency")
            is_first = bool(message.get("first", False))
            if self.collecting_metrics and isinstance(latency, (int, float)) and not is_first:
                self.server_metric_samples["server_e2e"].append(float(latency))
            return

        if message_type == "latency-breakdown":
            vad_smart_turn = message.get("vad_smart_turn")
            if self.collecting_metrics and isinstance(vad_smart_turn, (int, float)):
                self.server_metric_samples["vad_smart_turn"].append(float(vad_smart_turn))
            return

        if message_type != "metrics":
            return

        metrics = message.get("data", {})
        if not isinstance(metrics, dict) or not self.collecting_metrics:
            return

        for item in metrics.get("ttfb", []):
            if not isinstance(item, dict):
                continue
            value = item.get("value")
            processor = str(item.get("processor", ""))
            if not isinstance(value, (int, float)):
                continue
            category = categorize_processor(processor)
            if category == "llm":
                self.server_metric_samples["llm_ttft"].append(float(value))
            elif category == "tts":
                self.server_metric_samples["tts_ttfb"].append(float(value))
            elif category == "asr":
                self.server_metric_samples["asr_ttfb"].append(float(value))

        for item in metrics.get("processing", []):
            if not isinstance(item, dict):
                continue
            value = item.get("value")
            processor = str(item.get("processor", ""))
            if not isinstance(value, (int, float)):
                continue
            if categorize_processor(processor) == "llm":
                processing_time = float(value)
                self.server_metric_samples["llm_processing_time"].append(processing_time)
                if self._pending_llm_completion_tokens:
                    completion_tokens = self._pending_llm_completion_tokens.pop(0)
                    if processing_time > 0:
                        self.server_metric_samples["llm_tokens_per_sec"].append(completion_tokens / processing_time)

        for item in metrics.get("tokens", []):
            if not isinstance(item, dict):
                continue
            completion_tokens = item.get("completion_tokens")
            if isinstance(completion_tokens, (int, float)):
                self._pending_llm_completion_tokens.append(float(completion_tokens))

    async def _recv_audio_frame(self, websocket, timeout: float | None = None) -> bytes:
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            if _SHUTDOWN_REQUESTED:
                raise asyncio.CancelledError

            wait_timeout = None
            if deadline is not None:
                wait_timeout = max(deadline - time.monotonic(), 0)
                if wait_timeout == 0:
                    raise TimeoutError

            data = await asyncio.wait_for(websocket.recv(), timeout=wait_timeout)
            try:
                proto = frames_pb2.Frame.FromString(data)
            except Exception as exc:
                log_error(f"Failed to parse protobuf frame: {exc}")
                continue

            which = proto.WhichOneof("frame")
            if which == "audio":
                return data
            if which == "message":
                try:
                    await self._process_server_message(json.loads(proto.message.data))
                except Exception as exc:
                    log_error(f"Failed to parse message frame payload: {exc}")

    def _write_audio_to_wav(self, data: bytes, wf, *, create_new_file: bool = False):
        try:
            proto = frames_pb2.Frame.FromString(data)
            which = proto.WhichOneof("frame")
            if which is None:
                return wf, None, None, None
        except Exception as exc:
            log_error(f"Failed to parse protobuf frame: {exc}")
            return wf, None, None, None

        args = getattr(proto, which)
        sample_rate = getattr(args, "sample_rate", 16000)
        num_channels = getattr(args, "num_channels", 1)
        audio_data = getattr(args, "audio", None)
        if audio_data is None:
            return wf, None, None, None

        try:
            with io.BytesIO(audio_data) as buffer, wave.open(buffer, "rb") as wav_file:
                audio_data = wav_file.readframes(wav_file.getnframes())
                sample_rate = wav_file.getframerate()
                num_channels = wav_file.getnchannels()
        except (wave.Error, EOFError, struct.error):
            pass

        if self.audio_output_path is not None and create_new_file and wf is None:
            output_wav = self.audio_output_path
            try:
                output_wav.parent.mkdir(parents=True, exist_ok=True)
                wf = wave.open(str(output_wav), "wb")  # noqa: SIM115
                wf.setnchannels(num_channels)
                wf.setsampwidth(2)
                wf.setframerate(sample_rate)
            except Exception as exc:
                log_error(f"Failed to create WAV file {output_wav}: {exc}")
                return None, None, None, None

        if self.audio_output_path is not None and wf is not None:
            try:
                wf.writeframes(audio_data)
            except Exception as exc:
                log_error(f"Failed to write audio data: {exc}")
                return None, None, None, None

        return wf, sample_rate, num_channels, audio_data

    async def _send_audio_file(self, websocket, file_path: Path) -> None:
        assert self.silence_event is not None
        self.silence_event.set()
        await self.logger.log(f"{self.stream_id} sending input audio: {file_path.name}")
        try:
            with wave.open(str(file_path), "rb") as wav_file:
                n_channels = wav_file.getnchannels()
                frame_rate = wav_file.getframerate()
                sample_width = wav_file.getsampwidth()
                chunk_size = int((frame_rate * n_channels * CHUNK_DURATION_MS) / 1000) * sample_width
                self.audio_params = (frame_rate, n_channels, chunk_size)

                while True:
                    chunk = wav_file.readframes(chunk_size // sample_width)
                    if not chunk:
                        break
                    samples_in_chunk = len(chunk) / (sample_width * n_channels)
                    chunk_duration_ms = (samples_in_chunk / frame_rate) * 1000
                    await self.logger.log(
                        f"{self.stream_id} input chunk size={len(chunk)}B "
                        f"duration_ms={chunk_duration_ms:.1f} "
                        f"sample_rate={frame_rate} channels={n_channels}"
                    )
                    audio_frame = frames_pb2.AudioRawFrame(
                        audio=chunk,
                        sample_rate=frame_rate,
                        num_channels=n_channels,
                    )
                    await websocket.send(frames_pb2.Frame(audio=audio_frame).SerializeToString())
                    await asyncio.sleep(CHUNK_DURATION_MS / 1000)
        finally:
            self.timestamps["input_audio_file_end"] = dt.datetime.now()
            await self.logger.log(
                f"{self.stream_id} input audio finished at "
                f"{self.timestamps['input_audio_file_end'].strftime('%H:%M:%S.%f')[:-3]}"
            )
            self.silence_event.clear()

    async def _silence_sender_loop(self, websocket) -> None:
        self.silence_running = True
        try:
            while self.silence_running:
                if _SHUTDOWN_REQUESTED:
                    return
                if self.silence_event is None or self.silence_event.is_set() or self.audio_params is None:
                    await asyncio.sleep(0.1)
                    continue
                frame_rate, n_channels, chunk_size = self.audio_params
                silent_chunk = b"\x00" * chunk_size
                audio_frame = frames_pb2.AudioRawFrame(
                    audio=silent_chunk,
                    sample_rate=frame_rate,
                    num_channels=n_channels,
                )
                await websocket.send(frames_pb2.Frame(audio=audio_frame).SerializeToString())
                await asyncio.sleep(CHUNK_DURATION_MS / 1000)
        except ConnectionClosed:
            return

    async def _receive_initial_bot_intro(self, websocket, wf):
        try:
            data = await self._recv_audio_frame(websocket, timeout=BOT_INTRO_TIMEOUT)
        except TimeoutError:
            await self.logger.log(f"{self.stream_id} no initial bot intro within {BOT_INTRO_TIMEOUT}s")
            return wf
        await self.logger.log(f"{self.stream_id} received initial bot intro")
        wf, _ = await self._drain_utterance(websocket, data, wf, detect_glitches=False)
        return wf

    async def _drain_utterance(self, websocket, first_data: bytes, wf, *, detect_glitches: bool, drain_timeout=None):
        if drain_timeout is None:
            drain_timeout = END_OF_RESPONSE_TIMEOUT

        playback_buffer_duration = 0.0
        last_update_time = None
        chunk_count = 0
        data = first_data

        while True:
            wf, sample_rate, num_channels, audio_data = self._write_audio_to_wav(data, wf, create_new_file=(wf is None))
            if audio_data and sample_rate and num_channels:
                current_time = time.time()
                chunk_count += 1
                samples_in_chunk = len(audio_data) // (num_channels * 2)
                chunk_duration_seconds = samples_in_chunk / sample_rate
                await self.logger.log(
                    f"{self.stream_id} output chunk #{chunk_count} size={len(audio_data)}B "
                    f"duration_ms={chunk_duration_seconds * 1000:.1f}"
                )

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
                    playback_buffer_duration += chunk_duration_seconds
                    last_update_time = current_time

            try:
                data = await self._recv_audio_frame(websocket, timeout=drain_timeout)
            except TimeoutError:
                return wf, chunk_count

    async def _await_first_bot_frame(self, send_task: asyncio.Task, recv_task: asyncio.Task) -> FirstBotFrame:
        """Wait for the first bot audio frame for a turn.

        Returns ``(None, None)`` when the bot never replies within
        ``turn_response_timeout`` after the input audio finishes.
        """
        data = None
        utterance_start = None
        try:
            done, _ = await asyncio.wait({send_task, recv_task}, return_when=asyncio.FIRST_COMPLETED)
            if recv_task in done:
                data = recv_task.result()
                utterance_start = dt.datetime.now()
                return data, utterance_start

            await send_task
            try:
                data = await asyncio.wait_for(recv_task, timeout=self.turn_response_timeout)
                utterance_start = dt.datetime.now()
                return data, utterance_start
            except TimeoutError:
                if self.collecting_metrics:
                    self.failed_turns += 1
                    await self.logger.log(
                        f"{self.stream_id} turn timed out: no bot response within "
                        f"{self.turn_response_timeout:.1f}s after input audio finished"
                    )
                return None, None
        finally:
            if data is None and not recv_task.done():
                recv_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await recv_task

    async def _process_conversation_turn(self, websocket, audio_file_path: Path, wf):
        # Single-shot turn flow: send user audio in the background, keep
        # consuming server messages, then wait up to ``turn_response_timeout``
        # after end-of-user-audio for the bot's first frame. Barge-in
        # classification is post-hoc (latency vs.
        # ``reverse_barge_in_threshold``); only valid latencies (>= threshold)
        # feed the per-client average and p95.
        self.timestamps["input_audio_file_end"] = None
        await self.logger.log(f"{self.stream_id} turn start using {audio_file_path.name}")
        send_task = asyncio.create_task(self._send_audio_file(websocket, audio_file_path))
        recv_task = asyncio.create_task(self._recv_audio_frame(websocket, timeout=None))
        data, utterance_start = await self._await_first_bot_frame(send_task, recv_task)
        if data is None or utterance_start is None:
            return wf

        wf, _ = await self._drain_utterance(websocket, data, wf, detect_glitches=True)
        await send_task

        input_end = self.timestamps["input_audio_file_end"]
        latency = (utterance_start - input_end).total_seconds() if input_end is not None else None

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

    async def _continuous_audio_loop(self, websocket, wf):
        turn_index = 0
        while self.running and not _SHUTDOWN_REQUESTED:
            now = time.time()
            if self.session_end_time and now >= self.session_end_time:
                self.collecting_metrics = False
                self.running = False
                await self.logger.log(f"{self.stream_id} session window closed")
                break
            if self.metrics_start_time and now >= self.metrics_start_time and not self.collecting_metrics:
                self.collecting_metrics = True
                await self.logger.log(f"{self.stream_id} metrics collection started")
            if (
                self.metrics_start_time
                and self.collecting_metrics
                and now >= self.metrics_start_time + self.test_duration
            ):
                self.collecting_metrics = False
                self.running = False
                await self.logger.log(f"{self.stream_id} metrics collection stopped")
                break

            audio_file = self.audio_files[turn_index % len(self.audio_files)]
            wf = await self._process_conversation_turn(websocket, audio_file, wf)
            turn_index += 1
            await asyncio.sleep(0.1)
        return wf

    async def run(self) -> ClientResult:
        await self.logger.log(f"{self.stream_id} starting client uri={self.uri}")
        if self.start_delay > 0:
            await self.logger.log(f"{self.stream_id} waiting start_delay={self.start_delay:.2f}s")
            await asyncio.sleep(self.start_delay)

        self.silence_event = asyncio.Event()
        self.silence_event.set()

        fallback_deadline = self.test_duration + HARD_DEADLINE_BUFFER
        if self.session_end_time:
            remaining = self.session_end_time - time.time()
            hard_deadline = max(remaining + 10, 30) if remaining > 0 else 30
        elif self.metrics_start_time:
            remaining = (self.metrics_start_time + self.test_duration + HARD_DEADLINE_BUFFER) - time.time()
            hard_deadline = max(remaining, 60) if remaining > 0 else fallback_deadline
        else:
            hard_deadline = fallback_deadline

        error = None
        try:
            ssl_ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
            ssl_ctx.check_hostname = False
            ssl_ctx.verify_mode = ssl.CERT_NONE
            async with websockets.connect(self.uri, open_timeout=WS_CONNECT_TIMEOUT, ssl=ssl_ctx) as websocket:
                await self.logger.log(f"{self.stream_id} websocket connected")
                ready_payload = {
                    "label": "rtvi-ai",
                    "type": "client-ready",
                    "id": f"{self.stream_id}-client-ready",
                    "data": {
                        "version": "0.1.0",
                        "about": {
                            "name": "scaling-perf-benchmark",
                        },
                    },
                }
                ready_message = frames_pb2.MessageFrame(data=json.dumps(ready_payload))
                await websocket.send(frames_pb2.Frame(message=ready_message).SerializeToString())
                await self.logger.log(f"{self.stream_id} sent RTVI client-ready")

                async def _run_session():
                    wf = await self._receive_initial_bot_intro(websocket, wf=None)
                    silence_task = asyncio.create_task(self._silence_sender_loop(websocket))
                    try:
                        wf = await self._continuous_audio_loop(websocket, wf)
                    finally:
                        self.silence_running = False
                        self.silence_event.set()
                        silence_task.cancel()
                        with contextlib.suppress(asyncio.CancelledError):
                            await silence_task
                    if wf is not None:
                        wf.close()

                await asyncio.wait_for(_run_session(), timeout=hard_deadline)
        except TimeoutError:
            error = f"Hard deadline reached ({hard_deadline:.0f}s)"
        except ConnectionClosed:
            error = "WebSocket connection closed"
        except Exception as exc:
            error = str(exc) or exc.__class__.__name__

        if error:
            await self.logger.log(f"{self.stream_id} finished with error: {error}")
        else:
            await self.logger.log(f"{self.stream_id} finished successfully")

        server_metric_average = {key: average_or_none(values) for key, values in self.server_metric_samples.items()}
        server_metric_counts = {key: len(values) for key, values in self.server_metric_samples.items()}

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
            server_metrics={
                "samples": self.server_metric_samples,
                "average": server_metric_average,
                "sample_counts": server_metric_counts,
            },
            rtvi_messages=self.rtvi_messages,
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
                ("llm_ttft", round3(server_metric_average.get("llm_ttft"))),
                ("tts_ttfb", round3(server_metric_average.get("tts_ttfb"))),
                ("asr_ttfb", round3(server_metric_average.get("asr_ttfb"))),
                ("server_e2e", round3(server_metric_average.get("server_e2e"))),
                ("vad_smart_turn", round3(server_metric_average.get("vad_smart_turn"))),
                ("llm_processing_time", round3(server_metric_average.get("llm_processing_time"))),
                ("llm_tokens_per_sec", round3(server_metric_average.get("llm_tokens_per_sec"))),
            ],
        )
        return result


def _resolve_paths(args: argparse.Namespace) -> tuple[Path, Path, Path | None]:
    """Compute final result/log/audio paths, falling back to ``--output-dir``."""
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)

    result_path = Path(args.result_path).resolve() if args.result_path else out / f"result_{args.stream_id}.json"
    logger_path = Path(args.logger_path).resolve() if args.logger_path else out / f"benchmark_{args.stream_id}.log"

    if args.audio_output_path:
        audio_output_path: Path | None = Path(args.audio_output_path).resolve()
    elif args.save_audio:
        audio_output_path = out / f"audio_output_{args.stream_id}.wav"
    else:
        audio_output_path = None
    return result_path, logger_path, audio_output_path


def build_arg_parser() -> argparse.ArgumentParser:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Single-client voice-agent benchmark. Use simulate_concurrency.sh for parallel runs."
    )
    parser.add_argument("--host", default="localhost", help="WebSocket host")
    parser.add_argument("--port", type=int, default=7860, help="WebSocket port")
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=script_dir / "dataset",
        help="Directory containing 16 kHz mono WAV files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=script_dir,
        help="Default parent directory for result/log/audio outputs",
    )
    parser.add_argument("--stream-id", default="", help="Stream id; auto-generated when empty")
    parser.add_argument("--start-delay", type=float, default=0.0, help="Seconds to wait before connecting")
    parser.add_argument(
        "--metrics-start-time",
        type=float,
        help="Unix epoch seconds when metric collection should begin (defaults to now+start_delay)",
    )
    parser.add_argument(
        "--session-end-time",
        type=float,
        help="Unix epoch seconds when this client should stop (defaults to metrics_start+test_duration)",
    )
    parser.add_argument("--test-duration", type=float, default=300.0, help="Metric collection window in seconds")
    parser.add_argument(
        "--reverse-barge-in-threshold",
        type=float,
        default=0.4,
        help=(
            "Latency (seconds) below which a turn is classified as a reverse "
            "barge-in and excluded from average/p95 reporting. Turns are still "
            "recorded in individual_latencies for analysis."
        ),
    )
    parser.add_argument(
        "--turn-response-timeout",
        type=float,
        default=TURN_RESPONSE_TIMEOUT,
        help=(
            "Per-turn timeout (seconds) waiting for the bot's first audio frame "
            "after the input audio file finishes sending. On timeout the turn "
            "is recorded as a failed_turn and the loop moves on to the next turn."
        ),
    )
    parser.add_argument("--result-path", type=Path, help="Write client result JSON here")
    parser.add_argument("--logger-path", type=Path, help="Write client log here")
    parser.add_argument("--audio-output-path", type=Path, help="Write client output WAV here")
    parser.set_defaults(save_audio=True)
    parser.add_argument(
        "--no-save-audio",
        dest="save_audio",
        action="store_false",
        help="Disable writing the default per-client output WAV",
    )

    # Aggregation modes — used by simulate_concurrency.sh after a run completes.
    aggregate = parser.add_argument_group("aggregation modes (post-run)")
    aggregate.add_argument(
        "--aggregate-run-dir",
        type=Path,
        help="Fold a directory of client_*/result_*.json into benchmark_summary.json",
    )
    aggregate.add_argument(
        "--aggregate-suite-dir",
        type=Path,
        help="Fold run_*_clients/benchmark_summary.json files into results.{tsv,txt,json}",
    )
    aggregate.add_argument(
        "--num-clients",
        type=int,
        help="Configured client count for --aggregate-run-dir (defaults to file count)",
    )
    return parser


async def async_main(args: argparse.Namespace) -> int:
    args.output_dir = args.output_dir.resolve()
    args.dataset_dir = args.dataset_dir.resolve()
    if not args.dataset_dir.is_dir():
        print(f"Dataset directory not found: {args.dataset_dir}", file=sys.stderr)
        return 2

    audio_files = sorted(p for p in args.dataset_dir.iterdir() if p.suffix.lower() == ".wav")
    if not audio_files:
        print(f"No .wav files found in {args.dataset_dir}", file=sys.stderr)
        return 2

    if not args.stream_id:
        args.stream_id = f"client_1_{str(time.time_ns())[:13]}"

    if args.metrics_start_time is None:
        args.metrics_start_time = time.time() + max(0.0, args.start_delay)
    if args.session_end_time is None:
        args.session_end_time = args.metrics_start_time + args.test_duration

    result_path, logger_path, audio_output_path = _resolve_paths(args)

    logger = RunLogger(logger_path)
    client = PerfClient(
        stream_id=args.stream_id,
        host=args.host,
        port=int(args.port),
        audio_files=audio_files,
        start_delay=float(args.start_delay),
        metrics_start_time=float(args.metrics_start_time),
        session_end_time=float(args.session_end_time),
        test_duration=float(args.test_duration),
        reverse_barge_in_threshold=float(args.reverse_barge_in_threshold),
        turn_response_timeout=float(args.turn_response_timeout),
        audio_output_path=audio_output_path,
        logger=logger,
    )
    result = await client.run()
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(asdict(result), indent=2), encoding="utf-8")
    print(
        f"client={args.stream_id} turns={result.num_turns} "
        f"avg_latency={round3(result.average_latency)}s "
        f"glitch={result.glitch_detected} "
        f"result={result_path}"
    )
    return 0


# ---------------------------------------------------------------------------
# Aggregation helpers (used by --aggregate-run-dir / --aggregate-suite-dir)
# ---------------------------------------------------------------------------

_SUITE_HEADERS = (
    "Parallel Streams",
    "Successful",
    "Failures",
    "No Response",
    "Avg Latency",
    "P95 Latency",
    "Min Latency",
    "Max Latency",
    "LLM TTFT",
    "TTS TTFB",
    "ASR TTFB",
    "Server E2E",
    "VAD+Smart Turn",
    "LLM Proc Time",
    "LLM Tok/s",
    "Glitches",
)

_CLIENT_HEADERS = ("Client", *_SUITE_HEADERS[1:])


def _calculate_p95(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    idx = math.ceil(0.95 * len(ordered)) - 1
    return ordered[max(0, min(idx, len(ordered) - 1))]


def _client_int(client: dict, key: str, default: int = 0) -> int:
    try:
        return int(client.get(key, default) or default)
    except (TypeError, ValueError):
        return default


def _client_valid_turns(client: dict) -> int:
    if "num_valid_turns" in client:
        return _client_int(client, "num_valid_turns")
    if client.get("average_latency") is not None:
        return _client_int(client, "num_turns")
    return 0


def _client_has_valid_response(client: dict) -> bool:
    return client.get("average_latency") is not None and _client_valid_turns(client) > 0


def _is_hard_deadline_client(client: dict) -> bool:
    return str(client.get("error") or "").startswith("Hard deadline reached")


def _client_has_core_server_metric(client: dict) -> bool:
    """Return true when the client observed any core server metric sample."""
    server_metrics = client.get("server_metrics") or {}
    sample_counts = server_metrics.get("sample_counts") or {}
    for key in ("asr_ttfb", "llm_ttft", "tts_ttfb"):
        try:
            if int(sample_counts.get(key, 0) or 0) > 0:
                return True
        except (TypeError, ValueError):
            continue
    return False


def _client_has_runtime_error(client: dict) -> bool:
    return bool(client.get("error")) and not _is_hard_deadline_client(client)


def _client_is_failed(client: dict) -> bool:
    return _client_has_runtime_error(client)


def _client_is_no_response(client: dict) -> bool:
    return not _client_is_failed(client) and not _client_has_valid_response(client)


def _client_is_successful(client: dict) -> bool:
    return not _client_is_failed(client) and _client_has_valid_response(client)


def _client_valid_latency_values(client: dict) -> list[float]:
    values = client.get("valid_latencies")
    if isinstance(values, list):
        out = []
        for value in values:
            if isinstance(value, (int, float)):
                out.append(float(value))
        return out

    average_latency = client.get("average_latency")
    if average_latency is not None and _client_valid_turns(client) > 0:
        return [float(average_latency)]
    return []


def _client_latency_summary(client: dict) -> dict[str, float | None]:
    latencies = _client_valid_latency_values(client)
    return {
        "avg_latency": average_or_none(latencies),
        "p95_latency": _calculate_p95(latencies),
        "min_latency": min(latencies) if latencies else None,
        "max_latency": max(latencies) if latencies else None,
    }


def _average_latency_columns(clients: Iterable[dict]) -> dict[str, float | None]:
    summaries = [_client_latency_summary(client) for client in clients]
    return {
        key: average_or_none([summary[key] for summary in summaries if summary.get(key) is not None])
        for key in ("avg_latency", "p95_latency", "min_latency", "max_latency")
    }


def _client_server_metric_average(client: dict, key: str) -> float | None:
    server_metrics = client.get("server_metrics") or {}
    value = (server_metrics.get("average") or {}).get(key)
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _client_sort_key(client: dict) -> tuple[int, str]:
    stream_id = str(client.get("stream_id") or "")
    parts = stream_id.split("_")
    if len(parts) > 1:
        try:
            return int(parts[1]), stream_id
        except ValueError:
            pass
    return sys.maxsize, stream_id


def _weighted_avg(per_client: list[dict], key: str) -> tuple[float | None, int]:
    weighted = 0.0
    total = 0
    for c in per_client:
        sm = c.get("server_metrics") or {}
        avg = (sm.get("average") or {}).get(key)
        cnt = (sm.get("sample_counts") or {}).get(key, 0)
        if avg is not None and cnt:
            weighted += float(avg) * int(cnt)
            total += int(cnt)
    return ((weighted / total) if total else None), total


def _format_table_lines(headers: Iterable[str], rows: list[list[str]]) -> list[str]:
    """Column-aligned text table; numeric columns right-justified."""
    headers = list(headers)
    widths = [len(h) for h in headers]
    for row in rows:
        for i, val in enumerate(row):
            widths[i] = max(widths[i], len(val))

    def is_num(v: str) -> bool:
        v = v.strip()
        if not v:
            return False
        try:
            float(v)
        except ValueError:
            return False
        return True

    right = [bool(rows) and i > 0 and all(is_num(row[i]) for row in rows) for i in range(len(headers))]

    def render(values: list[str]) -> str:
        return "  ".join(
            (values[i].rjust(widths[i]) if right[i] else values[i].ljust(widths[i])) for i in range(len(headers))
        )

    out = [render(headers), "  ".join("-" * w for w in widths)]
    for row in rows:
        out.append(render(row))
    return out


def _aggregate_run_dir(run_dir: Path, num_clients: int | None) -> Path:
    """Collapse all client_*/result_*.json into a benchmark_summary.json."""
    result_files = sorted(run_dir.glob("client_*/result_*.json"))
    clients: list[dict] = [json.loads(p.read_text(encoding="utf-8")) for p in result_files]

    if num_clients is None:
        num_clients = len(clients)

    hard_deadline = [c for c in clients if _is_hard_deadline_client(c)]

    # Top-level buckets are mutually exclusive:
    # * failure: real client/runtime error or missing result file,
    # * no response: client stayed alive but completed no valid in-window turn,
    # * success: at least one valid completed turn and no real runtime error.
    # Hard deadlines remain diagnostics; they only affect the top-level bucket
    # when the client also has a real runtime error or no valid turn.
    latency_clients = [c for c in clients if _client_has_valid_response(c)]
    latency_columns = _average_latency_columns(latency_clients)

    barge_in_only = [c for c in clients if _client_valid_turns(c) == 0 and _client_int(c, "num_turns") > 0]
    hard_deadline_with_valid_response = [c for c in hard_deadline if _client_has_valid_response(c)]
    hard_deadline_without_valid_response = [c for c in hard_deadline if not _client_has_valid_response(c)]
    runtime_errors = [c for c in clients if _client_has_runtime_error(c)]
    no_response = [c for c in clients if _client_is_no_response(c)]
    metric_only_no_response = [
        c for c in no_response if not _client_has_valid_response(c) and _client_has_core_server_metric(c)
    ]
    failed_client_ids = {str(c.get("stream_id")) for c in runtime_errors if c.get("stream_id")}
    missing_clients = max(0, num_clients - len(clients))
    failed_clients = len(failed_client_ids) + missing_clients
    successful_clients = max(0, num_clients - failed_clients - len(no_response))

    hard_deadline_with_success_signal = hard_deadline_with_valid_response
    hard_deadline_without_success_signal = hard_deadline_without_valid_response

    server_avg: dict[str, float | None] = {}
    server_counts: dict[str, int] = {}
    for key in SERVER_METRIC_KEYS:
        avg, total = _weighted_avg(clients, key)
        server_avg[key] = avg
        server_counts[key] = total

    summary = {
        "timestamp": dt.datetime.now().isoformat(),
        "config": {
            "num_clients": num_clients,
            "test_duration": (clients[0].get("test_duration") if clients else None),
            "metrics_start_time": (clients[0].get("metrics_start_time") if clients else None),
            "reverse_barge_in_threshold": (clients[0].get("reverse_barge_in_threshold") if clients else None),
            "turn_response_timeout": (clients[0].get("turn_response_timeout") if clients else None),
            "concurrency_mode": "process-per-client",
        },
        "results": {
            "configured_clients": num_clients,
            "successful_clients": successful_clients,
            "failed_clients": failed_clients,
            "failed_client_ids": sorted(failed_client_ids),
            "runtime_error_clients": len(runtime_errors),
            "runtime_error_client_ids": [c["stream_id"] for c in runtime_errors],
            "missing_clients": missing_clients,
            "latency_sample_clients": len(latency_clients),
            "metric_only_success_clients": 0,
            "metric_only_no_response_clients": len(metric_only_no_response),
            "barge_in_only_clients": len(barge_in_only),
            "no_response_clients": len(no_response),
            "no_response_client_ids": [c["stream_id"] for c in no_response],
            "hard_deadline_clients": len(hard_deadline),
            "hard_deadline_with_valid_response_clients": len(hard_deadline_with_valid_response),
            "hard_deadline_with_success_signal_clients": len(hard_deadline_with_success_signal),
            "hard_deadline_successful_clients": len(hard_deadline_with_success_signal),
            "hard_deadline_no_success_signal_clients": len(hard_deadline_without_success_signal),
            "hard_deadline_no_valid_response_clients": len(hard_deadline_without_valid_response),
            "hard_deadline_client_ids": [c["stream_id"] for c in hard_deadline],
            "hard_deadline_no_valid_response_client_ids": [
                c["stream_id"] for c in hard_deadline_without_valid_response
            ],
            "total_turns": sum(_client_int(c, "num_turns") for c in clients),
            "total_valid_turns": sum(_client_valid_turns(c) for c in latency_clients),
            "total_barge_ins": sum(_client_int(c, "reverse_barge_ins_count") for c in clients),
            "total_failed_turns": sum(_client_int(c, "failed_turns") for c in clients),
            "aggregate_average_latency": latency_columns["avg_latency"],
            "p95_client_latency": latency_columns["p95_latency"],
            "min_client_latency": latency_columns["min_latency"],
            "max_client_latency": latency_columns["max_latency"],
            "server_metrics": {"average": server_avg, "sample_counts": server_counts},
            "glitch_detection": {
                "clients_with_glitches": sum(1 for c in latency_clients if c.get("glitch_detected")),
                "total_clients": len(latency_clients),
                "affected_client_ids": [c["stream_id"] for c in latency_clients if c.get("glitch_detected")],
            },
            "error_detection": {
                "total_clients": len(clients),
                "clients_with_errors": len(runtime_errors),
                "runtime_error_clients": len(runtime_errors),
                "runtime_error_client_ids": [c["stream_id"] for c in runtime_errors],
                "hard_deadline_clients": len(hard_deadline),
                "hard_deadline_successful_clients": len(hard_deadline_with_success_signal),
                "hard_deadline_with_success_signal_clients": len(hard_deadline_with_success_signal),
                "hard_deadline_no_success_signal_clients": len(hard_deadline_without_success_signal),
                "hard_deadline_no_valid_response_clients": len(hard_deadline_without_valid_response),
                "hard_deadline_client_ids": [c["stream_id"] for c in hard_deadline],
                "client_error_counts": {c["stream_id"]: 1 for c in clients if c.get("error")},
            },
        },
        "clients": clients,
    }

    out = run_dir / "benchmark_summary.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return out


def _row_from_summary(summary: dict[str, Any], num_clients: int) -> dict[str, Any]:
    r = summary["results"]
    sa = r["server_metrics"]["average"]
    return {
        "parallel_streams": num_clients,
        "successful_streams": r["successful_clients"],
        "configured_streams": r["configured_clients"],
        "failed_streams": r.get("failed_clients", 0),
        "no_response_streams": r.get("no_response_clients", 0),
        "avg_latency": r["aggregate_average_latency"],
        "p95_latency": r["p95_client_latency"],
        "min_latency": r["min_client_latency"],
        "max_latency": r["max_client_latency"],
        "llm_ttft": sa.get("llm_ttft"),
        "tts_ttfb": sa.get("tts_ttfb"),
        "asr_ttfb": sa.get("asr_ttfb"),
        "server_e2e": sa.get("server_e2e"),
        "vad_smart_turn": sa.get("vad_smart_turn"),
        "llm_processing_time": sa.get("llm_processing_time"),
        "llm_tokens_per_sec": sa.get("llm_tokens_per_sec"),
        "audio_glitches": r["glitch_detection"]["clients_with_glitches"],
    }


def _client_row_from_result(client: dict[str, Any]) -> dict[str, Any]:
    failed = _client_is_failed(client)
    no_response = _client_is_no_response(client)
    successful = _client_is_successful(client)
    latency_summary = _client_latency_summary(client)
    return {
        "client": str(client.get("stream_id") or "(unknown)"),
        "successful_streams": 1 if successful else 0,
        "configured_streams": 1,
        "failed_streams": 1 if failed else 0,
        "no_response_streams": 1 if no_response else 0,
        **latency_summary,
        "llm_ttft": _client_server_metric_average(client, "llm_ttft"),
        "tts_ttfb": _client_server_metric_average(client, "tts_ttfb"),
        "asr_ttfb": _client_server_metric_average(client, "asr_ttfb"),
        "server_e2e": _client_server_metric_average(client, "server_e2e"),
        "vad_smart_turn": _client_server_metric_average(client, "vad_smart_turn"),
        "llm_processing_time": _client_server_metric_average(client, "llm_processing_time"),
        "llm_tokens_per_sec": _client_server_metric_average(client, "llm_tokens_per_sec"),
        "audio_glitches": 1 if client.get("glitch_detected") else 0,
    }


def _client_rows_from_summary(summary: dict[str, Any], num_clients: int) -> list[dict[str, Any]]:
    rows = [_client_row_from_result(client) for client in sorted(summary.get("clients", []), key=_client_sort_key)]
    average = _row_from_summary(summary, num_clients)
    average["client"] = "AVERAGE"
    latency_columns = _average_latency_columns(summary.get("clients", []))
    average.update(latency_columns)
    rows.append(average)
    return rows


def _metric_row_to_strings(row: dict[str, Any], label_key: str) -> list[str]:
    return [
        str(row[label_key]),
        f"{row['successful_streams']}/{row['configured_streams']}",
        str(row["failed_streams"]),
        str(row["no_response_streams"]),
        round3(row["avg_latency"]),
        round3(row["p95_latency"]),
        round3(row["min_latency"]),
        round3(row["max_latency"]),
        round3(row["llm_ttft"]),
        round3(row["tts_ttfb"]),
        round3(row["asr_ttfb"]),
        round3(row["server_e2e"]),
        round3(row["vad_smart_turn"]),
        round3(row["llm_processing_time"]),
        round3(row["llm_tokens_per_sec"]),
        str(row["audio_glitches"]),
    ]


def _row_to_strings(row: dict[str, Any]) -> list[str]:
    return _metric_row_to_strings(row, "parallel_streams")


def _client_row_to_strings(row: dict[str, Any]) -> list[str]:
    return _metric_row_to_strings(row, "client")


def _aggregate_suite_dir(suite_dir: Path) -> tuple[Path, Path, Path]:
    """Emit results.{tsv,txt,json} for a sweep dir or a single-level run dir.

    Sweep layout (one row per ``run_<N>_clients`` subdir):
        suite_dir/run_<N>_clients/benchmark_summary.json

    Single-level layout (one row, taken from the summary at the top level):
        suite_dir/benchmark_summary.json
    """
    run_dirs = sorted(
        (p for p in suite_dir.iterdir() if p.is_dir() and p.name.startswith("run_")),
        key=lambda p: int(p.name.split("_")[1]) if p.name.split("_")[1].isdigit() else 0,
    )

    rows: list[dict[str, Any]] = []
    headers = _SUITE_HEADERS
    stringify_row = _row_to_strings
    if run_dirs:
        for run_dir in run_dirs:
            try:
                num_clients = int(run_dir.name.split("_")[1])
            except (IndexError, ValueError):
                continue
            summary_path = run_dir / "benchmark_summary.json"
            if not summary_path.exists():
                continue
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            rows.append(_row_from_summary(summary, num_clients))
    else:
        summary_path = suite_dir / "benchmark_summary.json"
        if summary_path.exists():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            num_clients = int(summary.get("config", {}).get("num_clients") or summary["results"]["configured_clients"])
            rows = _client_rows_from_summary(summary, num_clients)
            headers = _CLIENT_HEADERS
            stringify_row = _client_row_to_strings

    tsv = suite_dir / "results.tsv"
    txt = suite_dir / "results.txt"
    js = suite_dir / "results.json"

    str_rows = [stringify_row(r) for r in rows]
    with tsv.open("w", encoding="utf-8") as f:
        f.write("\t".join(headers) + "\n")
        for row in str_rows:
            f.write("\t".join(row) + "\n")

    txt.write_text("\n".join(_format_table_lines(headers, str_rows)) + "\n", encoding="utf-8")
    js.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return tsv, txt, js


def _run_aggregate_run(run_dir: Path, num_clients: int | None) -> int:
    run_dir = run_dir.resolve()
    if not run_dir.is_dir():
        print(f"--aggregate-run-dir not found: {run_dir}", file=sys.stderr)
        return 2
    summary_path = _aggregate_run_dir(run_dir, num_clients=num_clients)
    r = json.loads(summary_path.read_text(encoding="utf-8"))["results"]
    print(
        f"run aggregated: {summary_path}  "
        f"successful={r['successful_clients']}/{r['configured_clients']}  "
        f"failures={r.get('failed_clients', 0)}  "
        f"no_response={r.get('no_response_clients', 0)}  "
        f"avg_latency={round3(r['aggregate_average_latency'])}s  "
        f"p95={round3(r['p95_client_latency'])}s"
    )
    return 0


def _run_aggregate_suite(suite_dir: Path) -> int:
    suite_dir = suite_dir.resolve()
    if not suite_dir.is_dir():
        print(f"--aggregate-suite-dir not found: {suite_dir}", file=sys.stderr)
        return 2
    tsv, txt, js = _aggregate_suite_dir(suite_dir)
    print(f"suite aggregated: {suite_dir}")
    print(f"  TXT:  {txt}")
    print(f"  TSV:  {tsv}")
    print(f"  JSON: {js}")
    return 0


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------


def main() -> int:
    args = build_arg_parser().parse_args()

    if args.aggregate_run_dir is not None:
        return _run_aggregate_run(args.aggregate_run_dir, args.num_clients)
    if args.aggregate_suite_dir is not None:
        return _run_aggregate_suite(args.aggregate_suite_dir)

    try:
        return asyncio.run(async_main(args))
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
