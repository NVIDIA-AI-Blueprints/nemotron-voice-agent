# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Replay canonical WAVs at 1x through the live gateway and retain paired evidence.

The live recorder models a zero-prebuffer client playback queue: each PCM packet
starts at its observed receive offset or after already queued PCM, whichever is
later. It never moves audio earlier than receipt. This client timeline is not a
server sample clock. Input is unchanged 16 kHz mono PCM16, including silence and
the final short packet. No extra silence, response.create, or manual commit is
sent. To test natural EOF turn detection, append the same explicit silence to
both online and offline source files before running either client.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import contextlib
import csv
import hashlib
import importlib.util
import json
import math
import os
import sys
import time
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import websockets
from websockets.asyncio.client import connect as WebSocketConnect
from websockets.exceptions import SecurityError

_SPEC = importlib.util.spec_from_file_location(
    "nva_live_comparison_runner", Path(__file__).with_name("run_fdb_samples.py")
)
assert _SPEC is not None and _SPEC.loader is not None
runner = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = runner
_SPEC.loader.exec_module(runner)
RATE = runner.client.RATE
LATENCY_KEYS = (
    "asr_ttfb",
    "llm_ttft",
    "tts_ttfb",
    "server_e2e",
    "vad_smart_turn",
    "llm_processing_time",
)

DEFINITIONS = {
    "units": "All duration fields end in _ms and contain milliseconds. PCM offsets are integer 16 kHz samples.",
    "playback": (
        "Client zero-prebuffer PCM queue: max(packet receipt offset, queued playback end). "
        "User speech-start/cancel events clip queued playback at their observed receipt time. "
        "No network-delay correction, acoustic onset detection, normalization, or time shifting."
    ),
    "server_metrics": (
        "response.done.metadata.nvidia_metrics from the normal live pipeline, converted from seconds to milliseconds. "
        "Samples accumulate between response completions and may not map one-to-one to user turns. "
        "The scopes are Pipecat metrics, not the offline adapters' request timers."
    ),
    "server_e2e_ms": (
        "Live UserBotLatencyObserver measurement as emitted by the server; never a client arrival estimate."
    ),
    "asr_ttfb_ms": (
        "Live STT timer: observed user speech end to transcript finalization, unlike offline stream-start TTFB."
    ),
    "llm_ttft_ms": "Live Pipecat LLM timer may end at a role-only choices delta; offline TTFT requires nonempty text.",
    "client_stop_to_first_audio_ms": (
        "Observed speech_stopped event receipt to first PCM receipt for the subsequent response. "
        "Not last acoustic user sample to audible bot onset, and not server E2E."
    ),
    "gateway_stop_to_playback_ms": (
        "First queued playback sample minus the gateway speech_stopped.audio_end_ms received-input offset. "
        "Gateway stop is semantic turn release, not the last acoustic speech sample."
    ),
    "eof": "Source ends without a synthetic EOF flush. Collection uses bounded post-input idle and absolute deadlines.",
    "missing": "Null means not reported or not applicable; never substitute zero.",
    "alias_handling": "Only GA response.output_audio.delta is consumed; beta response.audio.delta aliases are ignored.",
}


class _CredentialSafeConnect(WebSocketConnect):
    """Prevent a credentialed WebSocket handshake from redirecting to another origin."""

    def process_redirect(self, exc: Exception) -> Exception | str:
        """Return the original failure instead of allowing a redirect with credentials."""
        redirect = super().process_redirect(exc)
        return SecurityError("credentialed WebSocket redirects are disabled") if isinstance(redirect, str) else redirect


def write_json(path: Path, value: Any) -> None:
    """Write JSON that rejects non-finite numbers rather than hiding invalid timers."""
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if math.isfinite(value) else None


def server_metrics_ms(response: dict[str, Any]) -> dict[str, Any]:
    """Convert only documented server latency keys, retaining compact counts."""
    result: dict[str, Any] = {f"{key}_ms": None for key in LATENCY_KEYS}
    encoded = (response.get("metadata") or {}).get("nvidia_metrics")
    if not isinstance(encoded, str):
        return result
    try:
        raw = json.loads(encoded)
    except (ValueError, TypeError):
        return result
    if not isinstance(raw, dict):
        return result
    for key in LATENCY_KEYS:
        value = raw.get(key)
        if isinstance(value, list):
            numbers = [_number(item) for item in value]
            result[f"{key}_ms"] = [item * 1000 for item in numbers if item is not None] or None
        elif isinstance(value, dict):
            average, count = _number(value.get("average")), value.get("count")
            if average is not None and isinstance(count, int) and not isinstance(count, bool) and count > 0:
                result[f"{key}_ms"] = {"average_ms": average * 1000, "count": count}
    return result


@dataclass
class Span:
    """One received PCM packet placed in the simulated client playback queue."""

    response_id: str
    start_sample: int
    pcm: bytes
    received_ms: float

    @property
    def end_sample(self) -> int:
        """Return the exclusive PCM end."""
        return self.start_sample + len(self.pcm) // 2


class PlaybackCapture:
    """Retain actual receive clocks separately from causal PCM playback positions."""

    def __init__(self) -> None:
        """Initialize an empty recording and response ledger."""
        self.spans: list[Span] = []
        self.responses: dict[str, dict[str, Any]] = {}
        self.transcripts: list[dict[str, Any]] = []
        self.cancelled: set[str] = set()
        self.active_response_id: str | None = None
        self.last_stop: dict[str, Any] | None = None
        self.active_user_speech = False
        self.awaiting_response = False

    def clip(self, offset_ms: float, response_id: str | None = None) -> None:
        """Drop only future queued PCM; preserve samples already played."""
        cutoff = max(0, round(offset_ms * RATE / 1000))
        kept = []
        for span in self.spans:
            if response_id is not None and span.response_id != response_id:
                kept.append(span)
                continue
            if span.start_sample >= cutoff:
                continue
            count = min(len(span.pcm) // 2, cutoff - span.start_sample)
            kept.append(Span(span.response_id, span.start_sample, span.pcm[: count * 2], span.received_ms))
        self.spans = kept

    def consume(self, event: dict[str, Any], offset_ms: float) -> None:
        """Observe a GA event; intentionally ignore dual-emitted beta audio aliases."""
        kind = event.get("type")
        if kind == "input_audio_buffer.speech_started":
            self.active_user_speech = True
            self.clip(offset_ms)
            if self.active_response_id:
                self.cancelled.add(self.active_response_id)
        elif kind == "input_audio_buffer.speech_stopped":
            self.active_user_speech = False
            self.awaiting_response = True
            self.last_stop = {
                "item_id": event.get("item_id"),
                "gateway_audio_end_ms": event.get("audio_end_ms"),
                "received_ms": offset_ms,
            }
        elif kind == "response.created":
            response_id = event["response"]["id"]
            self.active_response_id = response_id
            self.awaiting_response = False
            self.responses.setdefault(
                response_id,
                {
                    "response_id": response_id,
                    "created_receive_ms": offset_ms,
                    "preceding_speech_stop": dict(self.last_stop) if self.last_stop else None,
                    "first_audio_receive_ms": None,
                    "status": "in_progress",
                    **server_metrics_ms({}),
                },
            )
        elif kind == "response.output_audio.delta":
            response_id = event.get("response_id")
            if response_id in self.cancelled:
                return
            if not isinstance(response_id, str) or response_id not in self.responses:
                raise ValueError("Received PCM without a preceding response.created")
            pcm = base64.b64decode(event["delta"], validate=True)
            if len(pcm) % 2:
                raise ValueError("Server returned a partial PCM16 sample")
            if not pcm:
                return
            start = max(0, round(offset_ms * RATE / 1000))
            if self.spans:
                start = max(start, self.spans[-1].end_sample)
            self.spans.append(Span(response_id, start, pcm, offset_ms))
            row = self.responses[response_id]
            if row["first_audio_receive_ms"] is None:
                row["first_audio_receive_ms"] = offset_ms
            row["last_audio_receive_ms"] = offset_ms
        elif kind == "response.done":
            response = event["response"]
            response_id = response["id"]
            if response_id not in self.responses:
                raise ValueError("Received response.done without response.created")
            row = self.responses[response_id]
            row.update(server_metrics_ms(response))
            row.update(status=response.get("status"), done_receive_ms=offset_ms, usage=response.get("usage"))
            if response.get("status") == "cancelled":
                self.cancelled.add(response_id)
                self.clip(offset_ms, response_id)
            if self.active_response_id == response_id:
                self.active_response_id = None
        if kind in {
            "conversation.item.input_audio_transcription.completed",
            "response.output_audio_transcript.done",
        }:
            self.transcripts.append({"received_ms": offset_ms, **event})

    def render(self, input_samples: int) -> bytes:
        """Pad silence around committed playback and preserve a final response tail."""
        count = max([input_samples, *(span.end_sample for span in self.spans)])
        result = bytearray(count * 2)
        for span in self.spans:
            result[span.start_sample * 2 : span.end_sample * 2] = span.pcm
        return bytes(result)

    def response_rows(self) -> list[dict[str, Any]]:
        """Add rendered spans without assigning metrics to unplayed cancelled audio."""
        rows = []
        for response_id, original in self.responses.items():
            row = dict(original)
            spans = [span for span in self.spans if span.response_id == response_id]
            row["committed_audio_samples"] = sum(len(span.pcm) // 2 for span in spans)
            row["first_playback_ms"] = spans[0].start_sample * 1000 / RATE if spans else None
            row["last_playback_end_ms"] = spans[-1].end_sample * 1000 / RATE if spans else None
            stop, first = row["preceding_speech_stop"], row["first_audio_receive_ms"]
            row["client_stop_to_first_audio_ms"] = first - stop["received_ms"] if stop and first is not None else None
            gateway_end = _number(stop.get("gateway_audio_end_ms")) if stop else None
            row["gateway_stop_to_playback_ms"] = (
                row["first_playback_ms"] - gateway_end if spans and gateway_end is not None else None
            )
            rows.append(row)
        return rows


def session_patch(config: dict[str, Any]) -> dict[str, Any]:
    """Request native 16 kHz PCM; do not silently resample or change the model config."""

    def validate_public(value: Any) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                normalized = key.lower().replace("-", "_")
                if normalized in {
                    "api_key",
                    "authorization",
                    "password",
                    "secret",
                    "access_token",
                    "refresh_token",
                } or normalized.endswith("_api_key"):
                    raise ValueError("Credentials must come from the environment, never a saved session configuration")
                validate_public(item)
        elif isinstance(value, list):
            for item in value:
                validate_public(item)

    validate_public(config)
    patch = json.loads(json.dumps(config))
    audio = patch.setdefault("audio", {})
    for side in ("input", "output"):
        settings = audio.setdefault(side, {})
        existing = settings.get("format")
        wanted = {"type": "audio/pcm", "rate": RATE}
        if existing is not None and existing != wanted:
            raise ValueError("Paired comparison requires explicit 16 kHz PCM input and output")
        settings["format"] = wanted
    audio["input"].setdefault("turn_detection", {"type": "server_vad"})
    return patch


def flat_to_session(config: dict[str, Any]) -> dict[str, Any]:
    """Map a flat offline config to catalog selectors, excluding internal endpoints.

    The live gateway resolves ASR/LLM/TTS endpoints from its server-side catalog.
    Endpoint fields must never be forwarded as session.nvidia overrides.
    """
    allowed = {
        "pipeline_mode",
        "llm_id",
        "model_id",
        "extra_params",
        "llm_settings",
        "llm_streaming",
        "asr_id",
        "asr_model",
        "asr_language_code",
        "asr_automatic_punctuation",
        "asr_settings",
        "tts_id",
        "tts_model",
        "tts_synthesis_mode",
        "tts_language_code",
        "tts_settings",
    }
    result: dict[str, Any] = {"nvidia": {key: value for key, value in config.items() if key in allowed}}
    if config.get("prompt_key"):
        result["prompt"] = {"id": config["prompt_key"]}
    if config.get("system_prompt"):
        raise ValueError("Realtime cannot represent a system_prompt override; configure the matching server catalog")
    instructions = config.get("prompt_content")
    if instructions:
        result["instructions"] = instructions
    for flat, live in (
        ("temperature", "temperature"),
        ("max_tokens", "max_output_tokens"),
        ("tool_choice", "tool_choice"),
    ):
        if flat in config:
            result[live] = config[flat]
    if config.get("tts_voice_id"):
        result["audio"] = {"output": {"voice": config["tts_voice_id"]}}
    return result


def prepare_paired_sources(cases: list[Any], output: Path, trailing_silence_ms: int = 4000) -> dict[str, Any]:
    """Make shared, provenance-recorded EOF padding for BOTH execution modes."""
    if isinstance(trailing_silence_ms, bool) or not isinstance(trailing_silence_ms, int) or trailing_silence_ms < 0:
        raise ValueError("Trailing source silence must be a nonnegative whole number of milliseconds")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Prepared sources must use a new or empty directory")
    sources = [(case, runner.client.read_pcm(case.input_path)) for case in cases]
    if any(not pcm for _, pcm in sources):
        raise ValueError("Source audio must be nonempty")
    padding_samples = trailing_silence_ms * RATE // 1000
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for case, original in sources:
        directory = output / case.case_id
        directory.mkdir()
        prepared = original + bytes(padding_samples * 2)
        runner.client.write_pcm(directory / "input.wav", prepared)
        record = {
            "id": case.case_id,
            "input": f"{case.case_id}/input.wav",
            "provenance": {
                **case.provenance,
                "paired_source": {
                    "original_input": str(case.input_path),
                    "original_file_sha256": hashlib.sha256(case.input_path.read_bytes()).hexdigest(),
                    "original_pcm_sha256": hashlib.sha256(original).hexdigest(),
                    "prepared_pcm_sha256": hashlib.sha256(prepared).hexdigest(),
                    "original_samples": len(original) // 2,
                    "prepared_samples": len(prepared) // 2,
                    "trailing_silence_samples": padding_samples,
                    "trailing_silence_ms": trailing_silence_ms,
                    "note": (
                        "Identical explicit padded source is replayed in both modes; no unrecorded audio is appended."
                    ),
                },
            },
        }
        records.append(record)
    result = {"cases": records}
    write_json(output / "cases.json", result)
    return result


def write_mode_comparison(offline_pcm: bytes, online_pcm: bytes, target: Path) -> None:
    """Write offline-left / online-right without shifting either recording."""
    count = max(len(offline_pcm), len(online_pcm)) // 2
    left, right = offline_pcm.ljust(count * 2, b"\0"), online_pcm.ljust(count * 2, b"\0")
    stereo = bytearray(count * 4)
    stereo[0::4], stereo[1::4] = left[0::2], left[1::2]
    stereo[2::4], stereo[3::4] = right[0::2], right[1::2]
    with wave.open(str(target), "wb") as stream:
        stream.setnchannels(2)
        stream.setsampwidth(2)
        stream.setframerate(RATE)
        stream.writeframes(stereo)


def _trace_event(event: dict[str, Any], offset_ms: float) -> dict[str, Any]:
    """Keep event contents while replacing bulky audio with exact hashes and sizes."""
    value = dict(event)
    if value.get("type") in {"response.output_audio.delta", "response.audio.delta"}:
        pcm = base64.b64decode(value.pop("delta"), validate=True)
        value.update(pcm_bytes=len(pcm), pcm_sha256=hashlib.sha256(pcm).hexdigest())
    return {"receive_offset_ms": offset_ms, "event": value}


async def run_live(
    url: str,
    pcm: bytes,
    output: Path,
    config: dict[str, Any],
    *,
    api_key: str = "",
    chunk_samples: int = 512,
    input_speed: float = 1.0,
    setup_timeout_ms: float = 120_000,
    post_input_idle_ms: float = 5_000,
    max_post_input_ms: float = 60_000,
) -> dict[str, Any]:
    """Capture native behavior; zero input speed diagnoses unpaced ingestion.

    Unpaced ingestion retains native wall-clock policies and output pacing.
    Its capture uses client elapsed time, not offline source-offset placement.
    """
    if not pcm or len(pcm) % 2 or not 0 < chunk_samples <= RATE:
        raise ValueError("Provide nonempty whole PCM16 samples and a positive chunk at most 1000 ms")
    if post_input_idle_ms <= 0 or max_post_input_ms < post_input_idle_ms or setup_timeout_ms <= 0:
        raise ValueError("Timeouts must be positive and absolute collection bound must cover idle bound")
    if (
        isinstance(input_speed, bool)
        or not isinstance(input_speed, (int, float))
        or not math.isfinite(input_speed)
        or input_speed < 0
    ):
        raise ValueError("Input speed must be finite and nonnegative; zero means unpaced diagnostic ingestion")
    if api_key and not url.lower().startswith("wss://"):
        raise ValueError("Refusing to send a Realtime API key over an unencrypted WebSocket")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite nonempty output directory {output}")
    patch = session_patch(config)
    output.mkdir(parents=True, exist_ok=True)
    runner.client.write_pcm(output / "input.wav", pcm)
    write_json(output / "requested_session.json", patch)
    capture = PlaybackCapture()
    trace: list[dict[str, Any]] = []
    sends: list[dict[str, Any]] = []
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    started = time.monotonic_ns()
    origin: int | None = None
    send_done_ns: int | None = None
    accepted_session = None
    reason = "absolute_post_input_deadline"
    # No redirects with credentials: changing origins could disclose the bearer.
    connect_args = {"additional_headers": headers, "max_size": 8 * 1024 * 1024, "open_timeout": setup_timeout_ms / 1000}
    connector = _CredentialSafeConnect if headers else websockets.connect
    async with connector(url, **connect_args) as ws:
        await ws.send(json.dumps({"type": "session.update", "session": patch}))
        setup_deadline = time.monotonic_ns() + round(setup_timeout_ms * 1_000_000)
        while accepted_session is None:
            remaining = (setup_deadline - time.monotonic_ns()) / 1e9
            if remaining <= 0:
                raise TimeoutError("Timed out waiting for session.updated")
            event = json.loads(await asyncio.wait_for(ws.recv(), remaining))
            if event.get("type") == "error":
                raise RuntimeError(f"Live session configuration failed: {event.get('error', {}).get('code')}")
            if event.get("type") == "session.updated":
                accepted_session = event.get("session")
        write_json(output / "accepted_session.json", accepted_session)

        async def send_source() -> None:
            nonlocal origin, send_done_ns
            origin = time.monotonic_ns()
            for start in range(0, len(pcm) // 2, chunk_samples):
                if input_speed:
                    deadline = origin + round(start * 1e9 / RATE / input_speed)
                    await asyncio.sleep(max(0, (deadline - time.monotonic_ns()) / 1e9))
                packet = pcm[start * 2 : (start + chunk_samples) * 2]
                before = time.monotonic_ns()
                await asyncio.wait_for(
                    ws.send(
                        json.dumps({"type": "input_audio_buffer.append", "audio": base64.b64encode(packet).decode()})
                    ),
                    timeout=30,
                )
                after = time.monotonic_ns()
                sends.append(
                    {
                        "start_sample": start,
                        "sample_count": len(packet) // 2,
                        "send_start_ms": (before - origin) / 1e6,
                        "send_complete_ms": (after - origin) / 1e6,
                    }
                )
            # Include the last short packet in the requested source pacing.
            if input_speed:
                end_deadline = origin + round(len(pcm) / 2 * 1e9 / RATE / input_speed)
                await asyncio.sleep(max(0, (end_deadline - time.monotonic_ns()) / 1e9))
            send_done_ns = time.monotonic_ns()

        sender = asyncio.create_task(send_source())
        last_activity_ns: int | None = None
        try:
            while True:
                if sender.done():
                    sender.result()
                try:
                    event = json.loads(await asyncio.wait_for(ws.recv(), 0.1))
                    now = time.monotonic_ns()
                    if origin is None:
                        raise RuntimeError("Server emitted output before source transmission began; disable welcome")
                    offset_ms = (now - origin) / 1e6
                    trace.append(_trace_event(event, offset_ms))
                    if event.get("type") == "error":
                        raise RuntimeError(f"Live pipeline error: {event.get('error', {}).get('code')}")
                    capture.consume(event, offset_ms)
                    if event.get("type") in {
                        "response.output_audio.delta",
                        "response.created",
                        "response.done",
                        "input_audio_buffer.speech_started",
                        "input_audio_buffer.speech_stopped",
                        "conversation.item.input_audio_transcription.completed",
                    }:
                        last_activity_ns = now
                except TimeoutError:
                    now = time.monotonic_ns()
                if send_done_ns is None:
                    continue
                if now - send_done_ns >= max_post_input_ms * 1e6:
                    break
                idle_from = max(send_done_ns, last_activity_ns or send_done_ns)
                if (
                    capture.active_response_id is None
                    and not capture.awaiting_response
                    and not capture.active_user_speech
                    and now - idle_from >= post_input_idle_ms * 1e6
                ):
                    reason = "post_input_idle"
                    break
        finally:
            if not sender.done():
                sender.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await sender
            # Preserve wire evidence even when collection fails its completion
            # guard. A failed run must not lose already received benchmark PCM
            # or conceal which gateway state kept it awaiting completion.
            runner.client.write_pcm(output / "received_output.wav", capture.render(len(pcm) // 2))
            write_json(
                output / "capture_state.json",
                {
                    "active_response_id": capture.active_response_id,
                    "awaiting_response": capture.awaiting_response,
                    "active_user_speech": capture.active_user_speech,
                    "cancelled_response_ids": sorted(capture.cancelled),
                    "response_statuses": {key: value.get("status") for key, value in capture.responses.items()},
                    "source_send_completed": send_done_ns is not None,
                    "collection_reason": reason,
                },
            )
            write_json(output / "transcripts.json", capture.transcripts)
            write_json(output / "send_trace.json", sends)
            (output / "events.jsonl").write_text(
                "".join(json.dumps(item, allow_nan=False) + "\n" for item in trace), encoding="utf-8"
            )
    assert origin is not None and send_done_ns is not None
    if reason != "post_input_idle":
        raise TimeoutError("Live output did not settle before the absolute post-input bound")
    full_pcm = capture.render(len(pcm) // 2)
    runner.client.write_pcm(output / "full_output.wav", full_pcm)
    runner.client.write_pcm(output / "output.wav", full_pcm[: len(pcm)])
    runner.write_comparison(output / "input.wav", output / "full_output.wav", output / "comparison.wav")
    rows = capture.response_rows()
    for index, row in enumerate(rows, 1):
        row["response_index"] = index
    result = {
        "mode": "online_1x_normal_realtime_pipeline" if input_speed == 1 else "native_pipeline_pacing_diagnostic",
        "input_speed": input_speed,
        "input_chunk_samples": chunk_samples,
        "recording_clock": "client_elapsed_time",
        "input_pcm_sha256": hashlib.sha256(pcm).hexdigest(),
        "input_samples": len(pcm) // 2,
        "input_duration_ms": len(pcm) / 2 * 1000 / RATE,
        "output_samples": len(full_pcm) // 2,
        "output_duration_ms": len(full_pcm) / 2 * 1000 / RATE,
        "client_setup_ms": (origin - started) / 1e6,
        "client_input_send_ms": (send_done_ns - origin) / 1e6,
        "client_total_wall_ms": (time.monotonic_ns() - started) / 1e6,
        "completion_reason": reason,
        "responses": rows,
        "playback_spans": [
            {
                "response_id": span.response_id,
                "start_sample": span.start_sample,
                "end_sample": span.end_sample,
                "received_ms": span.received_ms,
                "pcm_sha256": hashlib.sha256(span.pcm).hexdigest(),
            }
            for span in capture.spans
        ],
        "definitions": DEFINITIONS,
    }
    write_json(output / "result.json", result)
    write_json(output / "transcripts.json", capture.transcripts)
    write_json(output / "send_trace.json", sends)
    (output / "events.jsonl").write_text(
        "".join(json.dumps(item, allow_nan=False) + "\n" for item in trace), encoding="utf-8"
    )
    _write_response_csv(output / "responses_ms.csv", rows)
    return result


def _write_response_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "response_index",
        "response_id",
        "status",
        "committed_audio_samples",
        "first_audio_receive_ms",
        "first_playback_ms",
        "last_playback_end_ms",
        "client_stop_to_first_audio_ms",
        "gateway_stop_to_playback_ms",
        *(f"{key}_ms" for key in LATENCY_KEYS),
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {key: json.dumps(value) if isinstance(value, (list, dict)) else value for key, value in row.items()}
            )


async def run_cases(args: argparse.Namespace) -> dict[str, Any]:
    """Run independent sessions sequentially with source hashes and paired checks."""
    cases = runner.load_cases(manifest=args.manifest)
    if len(cases) > args.max_cases:
        raise ValueError("Case count exceeds explicit bound")
    sources = [(case, runner.client.read_pcm(case.input_path)) for case in cases]
    if any(len(pcm) / 2 / RATE > args.max_input_seconds for _, pcm in sources):
        raise ValueError("Input exceeds explicit duration bound; never truncate it")
    if args.prepare_only:
        return prepare_paired_sources(cases, args.output_dir, args.trailing_silence_ms)
    config_path = args.session_config or args.config
    if config_path is None or not args.live_url:
        raise ValueError("Live replay requires --live-url and either --session-config or --config")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("Live session config must be a JSON object")
    if args.config:
        config = flat_to_session(config)
    if args.offline_results_dir:
        for case, pcm in sources:
            offline = args.offline_results_dir / case.case_id
            if runner.client.read_pcm(offline / "input.wav") != pcm:
                raise ValueError("Paired online/offline source PCM hashes differ")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError("Output directory must be new or empty")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {"definitions": DEFINITIONS, "cases": []}
    for case, pcm in sources:
        target = args.output_dir / case.case_id
        result = await run_live(
            args.live_url,
            pcm,
            target,
            config,
            api_key=os.environ.get(args.api_key_env, ""),
            chunk_samples=args.chunk_samples,
            setup_timeout_ms=args.setup_timeout_ms,
            post_input_idle_ms=args.post_input_idle_ms,
            max_post_input_ms=args.max_post_input_ms,
        )
        entry = {
            "id": case.case_id,
            "input_pcm_sha256": result["input_pcm_sha256"],
            "response_count": len(result["responses"]),
            "committed_response_count": sum(row["committed_audio_samples"] > 0 for row in result["responses"]),
            "client_total_wall_ms": result["client_total_wall_ms"],
        }
        if args.offline_results_dir:
            offline = args.offline_results_dir / case.case_id
            entry["offline_directory"] = str(offline)
            entry["paired_input_pcm_equal"] = True
            write_mode_comparison(
                runner.client.read_pcm(offline / "full_output.wav"),
                runner.client.read_pcm(target / "full_output.wav"),
                target / "offline_left_online_right.wav",
            )
        summary["cases"].append(entry)
        write_json(args.output_dir / "summary.json", summary)
        print(f"Completed {case.case_id}: {entry['committed_response_count']} played responses", flush=True)
    return summary


def main() -> None:
    """Run live replay against the standard gateway, optionally checking offline pairs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--live-url")
    config = parser.add_mutually_exclusive_group()
    config.add_argument("--session-config", type=Path, help="Realtime session.update.session JSON; no credentials")
    config.add_argument(
        "--config", type=Path, help="Flat offline config: map catalog selectors; resolve endpoints on server"
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--offline-results-dir", type=Path)
    parser.add_argument(
        "--prepare-only", action="store_true", help="Write shared padded source WAVs and cases.json; no requests"
    )
    parser.add_argument(
        "--trailing-silence-ms", type=int, default=4000, help="Explicit shared source padding for --prepare-only"
    )
    parser.add_argument("--api-key-env", default="REALTIME_API_KEY")
    parser.add_argument("--chunk-samples", type=int, default=512)
    parser.add_argument("--setup-timeout-ms", type=float, default=120_000)
    parser.add_argument("--post-input-idle-ms", type=float, default=5_000)
    parser.add_argument("--max-post-input-ms", type=float, default=60_000)
    parser.add_argument("--max-cases", type=int, default=5)
    parser.add_argument("--max-input-seconds", type=float, default=300)
    asyncio.run(run_cases(parser.parse_args()))


if __name__ == "__main__":
    main()
