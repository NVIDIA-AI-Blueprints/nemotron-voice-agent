# SPDX-FileCopyrightText: Copyright (c) 2024–2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""OpenAI Realtime protocol adapter for scaling-perf."""

# ruff: noqa: D101,D102,D103,D105,D107

from __future__ import annotations

import asyncio
import json
import math
import wave
from pathlib import Path
from typing import Any

import numpy as np
from benchmark_core import (
    SERVER_METRIC_KEYS,
    AudioFrame,
    EndOfBotUtterance,
    FailedBotUtterance,
    ProtocolClient,
    ProtocolConnectionClosed,
    RunLogger,
)
from realtime_transport import (
    DEFAULT_AUTH_SCHEME,
    DEFAULT_OUTPUT_RATE,
    RealtimeProtocolError,
    RealtimeTransport,
    error_code,
    error_message,
    is_error_event,
    is_response_done,
    parse_output_audio,
)
from websockets.exceptions import ConnectionClosed

REALTIME_INPUT_RATE = 24_000
MAX_AGGREGATED_METADATA_SAMPLES = 4096


def _is_finite_number(value: Any) -> bool:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False
    try:
        return math.isfinite(float(value))
    except (OverflowError, ValueError):
        return False


def resample_pcm16(pcm: bytes, source_rate: int, target_rate: int) -> bytes:
    """Resample mono little-endian PCM16 while preserving chunk duration."""
    if source_rate == target_rate or not pcm:
        return pcm
    if source_rate <= 0 or target_rate <= 0 or len(pcm) % 2:
        raise ValueError("invalid PCM16 audio frame")
    samples = np.frombuffer(pcm, dtype="<i2")
    output_count = round(len(samples) * target_rate / source_rate)
    source_positions = np.arange(output_count, dtype=np.float64) * source_rate / target_rate
    output = np.interp(source_positions, np.arange(len(samples)), samples)
    return np.rint(output).clip(-32768, 32767).astype("<i2").tobytes()


class RealtimeFrameAdapter:
    """Translate benchmark audio frames to and from Realtime JSON audio."""

    def __init__(self, realtime: RealtimeTransport, input_sample_rate: int = REALTIME_INPUT_RATE):
        self.realtime = realtime
        self.input_sample_rate = input_sample_rate

    async def send_audio(self, frame: AudioFrame) -> None:
        if frame.num_channels != 1:
            raise ValueError("Realtime input must be mono")
        pcm = resample_pcm16(frame.audio, frame.sample_rate, self.input_sample_rate)
        await self.realtime.send_pcm(pcm)

    async def recv_audio(self, timeout: float | None = None) -> AudioFrame:
        deadline = None if timeout is None else asyncio.get_running_loop().time() + timeout
        while True:
            wait = None if deadline is None else max(0.0, deadline - asyncio.get_running_loop().time())
            if wait == 0:
                raise TimeoutError
            try:
                event = await self.realtime.recv_event(timeout=wait)
            except ConnectionClosed as exc:
                raise ProtocolConnectionClosed from exc
            if str(event.get("type") or "") == "error":
                raise FailedBotUtterance(error_message(event), terminal=False)
            if is_error_event(event):
                raise RealtimeProtocolError(error_message(event))
            if is_response_done(event):
                response = event.get("response")
                if isinstance(response, dict):
                    status = response.get("status")
                    if status and status != "completed":
                        raise FailedBotUtterance(error_message(event), terminal=True)
                raise EndOfBotUtterance
            pcm = parse_output_audio(event)
            if pcm:
                return AudioFrame(pcm, self.realtime.output_sample_rate, 1)


class RealtimeClient(ProtocolClient):
    """Realtime connection, synchronization, events, and metadata metrics."""

    name = "realtime"
    requires_response_done = True

    def __init__(
        self,
        *,
        stream_id: str,
        ws_url: str,
        logger: RunLogger,
        api_key: str = "",
        auth_scheme: str = DEFAULT_AUTH_SCHEME,
        connect_timeout: float,
        verify_tls: bool = True,
        audio_files: list[Path] | None = None,
    ):
        self.stream_id = stream_id
        self.uri = ws_url
        self.logger = logger
        self.api_key = api_key
        self.auth_scheme = auth_scheme
        self.connect_timeout = connect_timeout
        self.verify_tls = verify_tls
        self.realtime: RealtimeTransport | None = None
        self.adapter: RealtimeFrameAdapter | None = None
        self.source_input_rate: int | None = None
        self.server_metric_samples = {key: [] for key in SERVER_METRIC_KEYS}
        self.realtime_turn_metrics: list[dict[str, Any]] = []
        self.collecting_metrics = False
        self._ready_type = ""
        self.audio_files = audio_files or []

    @property
    def ready_log(self) -> str:
        assert self.realtime is not None and self.source_input_rate is not None
        return (
            f"realtime session ready type={self._ready_type} "
            f"input_rate={self.source_input_rate}->{REALTIME_INPUT_RATE} "
            f"output_rate={self.realtime.output_sample_rate}"
        )

    async def __aenter__(self) -> RealtimeClient:
        self._validate_audio_files(self.audio_files)
        self.realtime = RealtimeTransport(
            self.uri,
            api_key=self.api_key,
            auth_scheme=self.auth_scheme,
            connect_timeout=self.connect_timeout,
            input_sample_rate=REALTIME_INPUT_RATE,
            output_sample_rate=DEFAULT_OUTPUT_RATE,
            verify_tls=self.verify_tls,
        )
        try:
            await self.realtime.connect()
            ready = await self.realtime.configure_input(REALTIME_INPUT_RATE, timeout=self.connect_timeout)
        except ConnectionClosed as exc:
            await self.realtime.close()
            self.realtime = None
            raise ProtocolConnectionClosed from exc
        except Exception:
            await self.realtime.close()
            self.realtime = None
            raise
        self._ready_type = str(ready.get("type") or "")
        self.adapter = RealtimeFrameAdapter(self.realtime)
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        del exc_type, exc, tb
        if self.realtime is not None:
            await self.realtime.close()
        self.adapter = None
        self.realtime = None

    async def send_audio(self, frame: AudioFrame) -> None:
        assert self.adapter is not None
        try:
            await self.adapter.send_audio(frame)
        except ConnectionClosed as exc:
            raise ProtocolConnectionClosed from exc

    async def recv_audio(self, timeout: float | None = None) -> AudioFrame:
        assert self.adapter is not None
        return await self.adapter.recv_audio(timeout)

    async def recover_turn(
        self,
        *,
        timeout: float,
        failure: FailedBotUtterance | None = None,
        intro_timeout: bool = False,
    ) -> None:
        del intro_timeout
        if failure is None or not failure.terminal:
            await self._cancel_active_response(timeout)

    async def _cancel_active_response(self, timeout: float) -> None:
        assert self.realtime is not None
        try:
            await self.realtime.send_event({"type": "response.cancel"})
            deadline = asyncio.get_running_loop().time() + timeout
            while True:
                remaining = deadline - asyncio.get_running_loop().time()
                if remaining <= 0:
                    raise TimeoutError("timed out waiting for cancelled response.done")
                event = await self.realtime.recv_event(timeout=remaining)
                if str(event.get("type") or "") == "error" and error_code(event) == "response_cancel_not_active":
                    return
                if is_error_event(event):
                    raise RuntimeError(error_message(event))
                if is_response_done(event):
                    return
        except ConnectionClosed as exc:
            raise ProtocolConnectionClosed from exc
        except Exception as exc:
            raise RuntimeError("Realtime response cancellation barrier failed") from exc

    def set_collecting_metrics(self, collecting: bool) -> None:
        self.collecting_metrics = collecting

    def result_artifacts(self) -> dict[str, list[dict[str, Any]]]:
        return {"realtime_turn_metrics": self.realtime_turn_metrics}

    def begin_turn_observation(self) -> int:
        return len(self.realtime.events) if self.realtime is not None else 0

    def _validate_audio_files(self, audio_files: list[Path]) -> None:
        input_rates: set[int] = set()
        for audio_file in audio_files:
            with wave.open(str(audio_file), "rb") as wav_file:
                if wav_file.getnchannels() != 1:
                    raise ValueError(f"Realtime requires mono WAV input: {audio_file}")
                if wav_file.getsampwidth() != 2 or wav_file.getcomptype() != "NONE":
                    raise ValueError(f"Realtime requires uncompressed PCM16 WAV input: {audio_file}")
                input_rates.add(wav_file.getframerate())
        if len(input_rates) != 1:
            raise ValueError(f"Realtime requires one input sample rate; dataset contains {sorted(input_rates)}")
        self.source_input_rate = input_rates.pop()

    def process_response_metrics(self, response_done: dict[str, Any]) -> dict[str, list[float]]:
        metadata = response_done.get("metadata")
        encoded = metadata.get("nvidia_metrics") if isinstance(metadata, dict) else None
        if not isinstance(encoded, str):
            return {}
        try:
            decoded = json.loads(encoded)
        except json.JSONDecodeError:
            return {}
        if not isinstance(decoded, dict):
            return {}
        metrics: dict[str, list[float]] = {}
        for key in SERVER_METRIC_KEYS:
            value = decoded.get(key)
            samples: list[float] = []
            if isinstance(value, list) and len(value) <= MAX_AGGREGATED_METADATA_SAMPLES:
                samples = [float(sample) for sample in value if _is_finite_number(sample)]
            elif isinstance(value, dict):
                average = value.get("average")
                count = value.get("count")
                if (
                    _is_finite_number(average)
                    and isinstance(count, int)
                    and not isinstance(count, bool)
                    and 0 < count <= MAX_AGGREGATED_METADATA_SAMPLES
                ):
                    samples = [float(average)] * count
            if samples:
                metrics[key] = samples
                if self.collecting_metrics:
                    self.server_metric_samples[key].extend(samples)
        return metrics

    async def record_turn_observation(
        self,
        audio_file: Path,
        marker: Any,
        input_end: float | None,
        *,
        error: str | None = None,
    ) -> None:
        if self.realtime is None or input_end is None:
            return
        lifecycle_types = (
            "input_audio_buffer.speech_stopped",
            "conversation.item.input_audio_transcription.completed",
            "response.created",
            "response.output_audio_transcript.delta",
            "response.output_audio.delta",
            "response.done",
            "error",
        )
        first_by_type: dict[str, dict[str, Any]] = {}
        first_item_failure = None
        for event in self.realtime.events[int(marker) :]:
            kind = str(event.get("type") or "")
            if kind in lifecycle_types and kind not in first_by_type:
                first_by_type[kind] = event
            if first_item_failure is None and kind.endswith(".failed"):
                first_item_failure = event
        offsets = {
            kind: (
                float(event["received_at"]) - input_end
                if (event := first_by_type.get(kind)) and isinstance(event.get("received_at"), (int, float))
                else None
            )
            for kind in lifecycle_types
        }

        def stage_delta(later: str, earlier: str) -> float | None:
            later_value, earlier_value = offsets.get(later), offsets.get(earlier)
            return later_value - earlier_value if later_value is not None and earlier_value is not None else None

        done = first_by_type.get("response.done") or {}
        metrics = {
            "audio_file": audio_file.name,
            "input_end_to_speech_stopped": offsets["input_audio_buffer.speech_stopped"],
            "from_speech_stopped": {
                "asr_transcription_completed": stage_delta(
                    "conversation.item.input_audio_transcription.completed",
                    "input_audio_buffer.speech_stopped",
                ),
                "response_created": stage_delta("response.created", "input_audio_buffer.speech_stopped"),
                "first_output_transcript": stage_delta(
                    "response.output_audio_transcript.delta", "input_audio_buffer.speech_stopped"
                ),
                "first_output_audio": stage_delta("response.output_audio.delta", "input_audio_buffer.speech_stopped"),
                "response_done": stage_delta("response.done", "input_audio_buffer.speech_stopped"),
            },
            "stage_deltas": {
                "response_after_asr": stage_delta(
                    "response.created", "conversation.item.input_audio_transcription.completed"
                ),
                "first_text_after_response_created": stage_delta(
                    "response.output_audio_transcript.delta", "response.created"
                ),
                "first_audio_after_first_text": stage_delta(
                    "response.output_audio.delta", "response.output_audio_transcript.delta"
                ),
            },
            "response_status": done.get("status"),
            "usage": done.get("usage"),
            "server_metrics": self.process_response_metrics(done),
            "error": (
                error
                or (first_by_type.get("error") or {}).get("error")
                or (first_item_failure or {}).get("error")
                or done.get("error")
            ),
        }
        self.realtime_turn_metrics.append(metrics)
        self.realtime.events.clear()
        await self.logger.log(
            f"{self.stream_id} realtime lifecycle {audio_file.name}: {json.dumps(metrics, sort_keys=True)}"
        )
