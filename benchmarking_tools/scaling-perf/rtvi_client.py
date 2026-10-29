# SPDX-FileCopyrightText: Copyright (c) 2024–2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""RTVI protocol client for scaling-perf."""

# ruff: noqa: D101,D102,D103,D105,D107

from __future__ import annotations

import asyncio
import datetime as dt
import json
import math
from typing import Any

from benchmark_core import (
    SERVER_METRIC_KEYS,
    AudioFrame,
    FailedBotUtterance,
    ProtocolClient,
    ProtocolConnectionClosed,
    RunLogger,
)
from rtvi_transport import RTVIAudioFrame, RTVITransport, RTVITransportClosed


def categorize_processor(processor: str) -> str:
    name = processor.lower()
    if "asr" in name or "stt" in name or "transcription" in name:
        return "asr"
    if "tts" in name or "texttospeech" in name:
        return "tts"
    if "llm" in name:
        return "llm"
    return ""


def _finite_number(value: Any) -> float | None:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _positive_number(value: Any) -> float | None:
    number = _finite_number(value)
    return number if number is not None and number > 0 else None


class RTVIClient(ProtocolClient):
    """RTVI WebSocket connection, framing, readiness, and metric parsing."""

    name = "rtvi"

    def __init__(
        self,
        *,
        stream_id: str,
        host: str,
        port: int,
        logger: RunLogger,
        connect_timeout: float,
        verify_tls: bool = True,
    ):
        self.stream_id = stream_id
        self.logger = logger
        self.connect_timeout = connect_timeout
        self.transport = RTVITransport(
            stream_id=stream_id,
            host=host,
            port=port,
            connect_timeout=connect_timeout,
            verify_tls=verify_tls,
        )
        self.uri = self.transport.uri
        self.server_metric_samples = {key: [] for key in SERVER_METRIC_KEYS}
        self.rtvi_messages: list[dict[str, Any]] = []
        self.collecting_metrics = False
        self._pending_llm_completion_tokens: list[float] = []

    @property
    def ready_log(self) -> None:
        return None

    async def __aenter__(self) -> RTVIClient:
        try:
            await self.transport.__aenter__()
        except RTVITransportClosed as exc:
            raise ProtocolConnectionClosed from exc
        await self.logger.log(f"{self.stream_id} websocket connected")
        await self.logger.log(f"{self.stream_id} sent RTVI client-ready")
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.transport.__aexit__(exc_type, exc, tb)

    async def send_audio(self, frame: AudioFrame) -> None:
        try:
            await self.transport.send_audio(frame.audio, frame.sample_rate, frame.num_channels)
        except RTVITransportClosed as exc:
            raise ProtocolConnectionClosed from exc

    async def recv_audio(self, timeout: float | None = None) -> AudioFrame:
        deadline = None if timeout is None else asyncio.get_running_loop().time() + timeout
        while True:
            wait = None if deadline is None else max(0.0, deadline - asyncio.get_running_loop().time())
            if wait == 0:
                raise TimeoutError
            try:
                payload = await self.transport.recv(timeout=wait)
            except RTVITransportClosed as exc:
                raise ProtocolConnectionClosed from exc
            if isinstance(payload, RTVIAudioFrame):
                return AudioFrame(payload.audio, payload.sample_rate, payload.num_channels)
            await self.process_server_message(payload)

    async def process_server_message(self, message: dict) -> None:
        if not isinstance(message, dict):
            return
        message_type = str(message.get("type", "unknown"))
        self.rtvi_messages.append({"timestamp": dt.datetime.now().isoformat(), "type": message_type, "data": message})
        await self.logger.log(f"{self.stream_id} RTVI message: {json.dumps(message, sort_keys=True)}")
        if message_type == "server-message" and isinstance(message.get("data"), dict):
            nested = message["data"]
            if isinstance(nested.get("type"), str):
                message = nested
                message_type = nested["type"]
        if message_type == "user-bot-latency":
            value = _positive_number(message.get("latency"))
            if self.collecting_metrics and value is not None and not message.get("first", False):
                self.server_metric_samples["server_e2e"].append(value)
            return
        if message_type == "metric-group":
            for item in message.get("metrics", []):
                if not isinstance(item, dict):
                    continue
                metric_key = {
                    "text_aggregation_ms": "text_aggregation",
                    "turn_taking_latency_ms": "turn_taking_latency",
                }.get(item.get("key"))
                if metric_key is None:
                    continue
                value = _positive_number(item.get("value"))
                if self.collecting_metrics and value is not None:
                    self.server_metric_samples[metric_key].append(value / 1000)
            return
        metrics = message.get("data", {})
        if message_type != "metrics" or not isinstance(metrics, dict) or not self.collecting_metrics:
            return
        for item in metrics.get("ttfb", []):
            if not isinstance(item, dict):
                continue
            value = _positive_number(item.get("value"))
            category = categorize_processor(str(item.get("processor", "")))
            if value is not None and category:
                key = {"asr": "asr_ttfb", "llm": "llm_ttfb", "tts": "tts_ttfb"}[category]
                self.server_metric_samples[key].append(value)
        for item in metrics.get("ttfa", []):
            if not isinstance(item, dict):
                continue
            if categorize_processor(str(item.get("processor", ""))) != "tts":
                continue
            value = _positive_number(item.get("ttfa"))
            if value is not None:
                self.server_metric_samples["tts_ttfa"].append(value)
            leading_silence = _finite_number(item.get("leading_silence"))
            if leading_silence is not None:
                self.server_metric_samples["tts_leading_silence"].append(leading_silence)
        for item in metrics.get("ttfat", []):
            if not isinstance(item, dict):
                continue
            if categorize_processor(str(item.get("processor", ""))) != "llm":
                continue
            value = _positive_number(item.get("ttfat"))
            if value is not None:
                self.server_metric_samples["llm_ttfat"].append(value)
            thinking_time = _finite_number(item.get("thinking_time"))
            if thinking_time is not None:
                self.server_metric_samples["llm_thinking_time"].append(thinking_time)
        for item in metrics.get("tokens", []):
            if not isinstance(item, dict):
                continue
            completion_tokens = _positive_number(item.get("completion_tokens"))
            if completion_tokens is not None:
                self._pending_llm_completion_tokens.append(completion_tokens)
        for item in metrics.get("processing", []):
            if not isinstance(item, dict):
                continue
            value = _positive_number(item.get("value"))
            if value is not None and categorize_processor(str(item.get("processor", ""))) == "llm":
                processing_time = value
                self.server_metric_samples["llm_processing_time"].append(processing_time)
                if self._pending_llm_completion_tokens and processing_time > 0:
                    completion_tokens = self._pending_llm_completion_tokens.pop(0)
                    self.server_metric_samples["llm_tokens_per_sec"].append(completion_tokens / processing_time)

    async def recover_turn(
        self,
        *,
        timeout: float,
        failure: FailedBotUtterance | None = None,
        intro_timeout: bool = False,
    ) -> None:
        del timeout
        if failure is not None or intro_timeout:
            return
        raise RuntimeError(
            "RTVI turn timed out without a synchronization event; reconnect before starting another turn"
        )

    def set_collecting_metrics(self, collecting: bool) -> None:
        self.collecting_metrics = collecting

    def result_artifacts(self) -> dict[str, list[dict[str, Any]]]:
        return {"rtvi_messages": self.rtvi_messages}
