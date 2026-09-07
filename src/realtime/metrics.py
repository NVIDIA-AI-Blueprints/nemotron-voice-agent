# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""Collect Pipecat metrics for an OpenAI Realtime response."""

from __future__ import annotations

import json
import math
from collections import deque
from dataclasses import dataclass
from typing import Any

from pipecat.frames.frames import MetricsFrame
from pipecat.metrics.metrics import LLMUsageMetricsData, ProcessingMetricsData, TTFBMetricsData
from pipecat.processors.frameworks.rtvi.frames import RTVIServerMessageFrame

_METADATA_KEY = "nvidia_metrics"
_METADATA_VALUE_MAX_CHARS = 512


@dataclass(frozen=True)
class RealtimeResponseTelemetry:
    """Standard response usage plus schema-compatible NVIDIA metadata."""

    usage: dict[str, Any] | None = None
    metadata: dict[str, str] | None = None


def _processor_category(processor: str) -> str | None:
    value = processor.lower()
    if "stt" in value or "asr" in value or "transcription" in value:
        return "asr"
    if "tts" in value or "texttospeech" in value:
        return "tts"
    if "llm" in value:
        return "llm"
    return None


def _average(values: list[float]) -> float | None:
    if not values:
        return None
    return round(sum(values) / len(values), 6)


def _metric_value(value: Any) -> float | None:
    """Return a finite numeric metric value, rejecting booleans and malformed data."""
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


class RealtimeMetricsAccumulator:
    """Passively accumulate metrics observed between response completion events."""

    def __init__(self) -> None:
        """Create an empty stream accumulator."""
        self._samples: dict[str, list[float]] = {
            "asr_ttfb": [],
            "llm_ttft": [],
            "tts_ttfb": [],
            "server_e2e": [],
            "vad_smart_turn": [],
            "llm_processing_time": [],
        }
        self._token_totals: dict[str, int] = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "cache_read_input_tokens": 0,
            "input_audio_tokens": 0,
            "output_audio_tokens": 0,
            "cache_read_input_audio_tokens": 0,
        }
        self._token_fields_seen: set[str] = set()
        self._pending_completion_tokens: deque[int] = deque()
        self._pending_processing_times: deque[float] = deque()
        self._throughput_samples: list[float] = []

    def consume(self, frame: MetricsFrame | RTVIServerMessageFrame) -> None:
        """Consume one supported frame without affecting Realtime lifecycle state."""
        if isinstance(frame, MetricsFrame):
            self._consume_metrics_frame(frame)
        elif isinstance(frame, RTVIServerMessageFrame):
            self._consume_server_message(frame.data)

    def finish(self) -> RealtimeResponseTelemetry:
        """Snapshot accumulated telemetry and reset after a ``response.done``."""
        usage = self._build_usage()
        metrics = {key: [round(value, 6) for value in values] for key, values in self._samples.items() if values}
        if self._throughput_samples:
            metrics["llm_tokens_per_sec"] = [round(value, 6) for value in self._throughput_samples]

        metadata = None
        if metrics:
            encoded = json.dumps(metrics, separators=(",", ":"), sort_keys=True)
            if len(encoded) > _METADATA_VALUE_MAX_CHARS:
                aggregated = {
                    key: {"average": _average(values), "count": len(values)} for key, values in metrics.items()
                }
                encoded = json.dumps(aggregated, separators=(",", ":"), sort_keys=True)
            if len(encoded) <= _METADATA_VALUE_MAX_CHARS:
                metadata = {_METADATA_KEY: encoded}

        telemetry = RealtimeResponseTelemetry(usage=usage, metadata=metadata)
        self.clear()
        return telemetry

    def clear(self) -> None:
        """Discard all accumulated samples."""
        for values in self._samples.values():
            values.clear()
        for key in self._token_totals:
            self._token_totals[key] = 0
        self._token_fields_seen.clear()
        self._pending_completion_tokens.clear()
        self._pending_processing_times.clear()
        self._throughput_samples.clear()

    def _consume_metrics_frame(self, frame: MetricsFrame) -> None:
        for item in frame.data:
            if isinstance(item, TTFBMetricsData):
                category = _processor_category(item.processor)
                key = {"asr": "asr_ttfb", "llm": "llm_ttft", "tts": "tts_ttfb"}.get(category or "")
                value = _metric_value(item.value)
                if key and value is not None:
                    self._samples[key].append(value)
            elif isinstance(item, ProcessingMetricsData):
                if _processor_category(item.processor) == "llm":
                    processing_time = _metric_value(item.value)
                    if processing_time is None:
                        continue
                    self._samples["llm_processing_time"].append(processing_time)
                    self._pending_processing_times.append(processing_time)
                    self._pair_throughput_samples()
            elif isinstance(item, LLMUsageMetricsData):
                usage = item.value
                for key in self._token_totals:
                    value = getattr(usage, key, None)
                    if isinstance(value, int) and not isinstance(value, bool):
                        self._token_totals[key] += value
                        self._token_fields_seen.add(key)
                completion_tokens = getattr(usage, "completion_tokens", None)
                if isinstance(completion_tokens, int) and not isinstance(completion_tokens, bool):
                    self._pending_completion_tokens.append(completion_tokens)
                    self._pair_throughput_samples()

    def _pair_throughput_samples(self) -> None:
        while self._pending_completion_tokens and self._pending_processing_times:
            completion_tokens = self._pending_completion_tokens.popleft()
            processing_time = self._pending_processing_times.popleft()
            if processing_time > 0:
                self._throughput_samples.append(completion_tokens / processing_time)

    def _consume_server_message(self, raw: Any) -> None:
        if not isinstance(raw, dict):
            return
        message = raw
        if message.get("type") == "server-message" and isinstance(message.get("data"), dict):
            message = message["data"]

        message_type = message.get("type")
        if message_type == "user-bot-latency" and not bool(message.get("first", False)):
            latency = _metric_value(message.get("latency"))
            if latency is not None:
                self._samples["server_e2e"].append(latency)
        elif message_type == "latency-breakdown":
            latency = _metric_value(message.get("vad_smart_turn"))
            if latency is not None:
                self._samples["vad_smart_turn"].append(latency)

    def _build_usage(self) -> dict[str, Any] | None:
        if not self._token_fields_seen:
            return None

        completion_tokens = self._token_totals["completion_tokens"]
        reported_total = self._token_totals["total_tokens"]
        prompt_tokens = self._token_totals["prompt_tokens"]
        total_tokens = reported_total or prompt_tokens + completion_tokens
        input_tokens = max(total_tokens - completion_tokens, 0) if reported_total else prompt_tokens
        usage: dict[str, Any] = {
            "total_tokens": total_tokens,
            "input_tokens": input_tokens,
            "output_tokens": completion_tokens,
        }

        input_details: dict[str, Any] = {}
        input_audio_tokens = self._token_totals["input_audio_tokens"]
        if "input_audio_tokens" in self._token_fields_seen:
            input_details["audio_tokens"] = input_audio_tokens
            input_details["text_tokens"] = max(input_tokens - input_audio_tokens, 0)
        if "cache_read_input_tokens" in self._token_fields_seen:
            cached_tokens = self._token_totals["cache_read_input_tokens"]
            input_details["cached_tokens"] = cached_tokens
            cached_audio_tokens = self._token_totals["cache_read_input_audio_tokens"]
            cached_details = {"text_tokens": max(cached_tokens - cached_audio_tokens, 0)}
            if "cache_read_input_audio_tokens" in self._token_fields_seen:
                cached_details["audio_tokens"] = cached_audio_tokens
            input_details["cached_tokens_details"] = cached_details
        if input_details:
            usage["input_token_details"] = input_details

        output_details: dict[str, int] = {}
        output_audio_tokens = self._token_totals["output_audio_tokens"]
        if "output_audio_tokens" in self._token_fields_seen:
            output_details["audio_tokens"] = output_audio_tokens
            output_details["text_tokens"] = max(completion_tokens - output_audio_tokens, 0)
        if output_details:
            usage["output_token_details"] = output_details

        return usage
