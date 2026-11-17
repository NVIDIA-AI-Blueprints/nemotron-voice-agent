# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Opt-in, bounded metadata capture for post-test playback analysis.

Offsets describe PCM queued for output, not forced alignment of words to sound.
No audio, text, tool arguments, or tool results are copied into the event stream.
"""

import asyncio
import contextlib
import time
import uuid
from collections import OrderedDict
from collections.abc import Awaitable, Callable

from loguru import logger
from pipecat.frames.frames import (
    AggregatedTextFrame,
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    CancelFrame,
    EndFrame,
    ErrorFrame,
    FunctionCallInProgressFrame,
    FunctionCallResultFrame,
    InterruptionFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    OutputAudioRawFrame,
    OutputTransportMessageUrgentFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.observers.base_observer import BaseObserver, FramePushed
from pipecat.processors.frame_processor import FrameDirection
from pipecat.utils.security.redaction import redact_secrets

DIAGNOSTICS_TYPE = "nva.perf.diagnostics"


def component_name(processor: str) -> str:
    """Identify a provider family without treating an unknown source as a model."""
    name = processor.lower()
    if any(part in name for part in ("asr", "stt", "transcri")):
        return "asr"
    if "tts" in name:
        return "tts"
    if "llm" in name:
        return "llm"
    return "pipecat"


class PerfDiagnosticsObserver(BaseObserver):
    """Observe semantic boundaries and errors without changing frame processing."""

    def __init__(self, output, emit: Callable[[dict], Awaitable[None]], *, max_pending: int = 512):
        """Bind one observer to a session's output and metadata emitter."""
        super().__init__()
        self.output = output
        self.emit = emit
        self.session_id = uuid.uuid4().hex
        self.pending: list[dict] = []
        self.max_pending = max_pending
        self.dropped = 0
        self.sequence = 0
        self.audio_sequence = 0
        self.output_audio_ms = 0.0
        self.epochs: OrderedDict[str, dict] = OrderedDict()
        self.markers: OrderedDict[int, dict] = OrderedDict()
        self.input_epoch = None
        self.output_epoch = None
        self.tool_results: list[str] = []
        self.followup_tools: list[str] = []
        self.wake = asyncio.Event()
        self.sender: asyncio.Task | None = None

    def _epoch(self, context: str | None, frame_id: int) -> dict:
        key = context or (self.input_epoch or {}).get("id") or f"frame-{frame_id}"
        if key not in self.epochs:
            self.epochs[key] = {"id": key, "input_audio_ms": 0.0, "tool_call_ids": list(self.followup_tools)}
            if len(self.epochs) > 1024:
                self.epochs.popitem(last=False)
                self.dropped += 1
        return self.epochs[key]

    def _record(self, kind: str, data: FramePushed | None = None, **fields) -> None:
        self.sequence += 1
        event = {
            "seq": self.sequence,
            "kind": kind,
            "server_time_ns": time.time_ns(),
            "pipeline_time_ns": data.timestamp if data else None,
            "audio_sequence": self.audio_sequence,
            "output_audio_ms": self.output_audio_ms,
            **fields,
        }
        if len(self.pending) >= self.max_pending:
            self.dropped += 1
            return
        self.pending.append(event)
        self.wake.set()

    async def on_pipeline_started(self):
        """Start metadata delivery only after the transport is ready."""
        self._record("session_start")
        self.sender = asyncio.create_task(self._send_loop())

    async def _send_loop(self):
        while True:
            await self.wake.wait()
            await asyncio.sleep(0.05)
            await self.flush()

    async def flush(self):
        """Send a bounded batch; network pressure never blocks the audio callback."""
        if not self.pending:
            self.wake.clear()
            return
        events, self.pending = self.pending, []
        self.wake.clear()
        payload = {
            "type": DIAGNOSTICS_TYPE,
            "schema_version": 1,
            "session_id": self.session_id,
            "events": events,
            "dropped_events": self.dropped,
            "last_sequence": self.sequence,
        }
        try:
            await asyncio.wait_for(self.emit(payload), timeout=0.5)
        except Exception:
            self.dropped += len(events)
            logger.debug("Performance diagnostics batch could not be delivered")

    async def cleanup(self):
        """Drain pending metadata while the worker still owns its transport."""
        if self.sender:
            self.sender.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self.sender
        await self.flush()
        await super().cleanup()

    async def on_push_frame(self, data: FramePushed):
        """Record first-push lifecycle events and output-local audio offsets."""
        frame = data.frame
        downstream = data.direction == FrameDirection.DOWNSTREAM
        if data.first_push:
            if isinstance(frame, ErrorFrame):
                origin = frame.processor or data.source
                name = getattr(origin, "name", type(origin).__name__)
                self._record(
                    "error",
                    data,
                    frame_id=frame.id,
                    component=component_name(name),
                    processor=name,
                    message=redact_secrets(str(frame.error)),
                    fatal=bool(frame.fatal),
                    category=getattr(frame.category, "value", None),
                    exception_type=type(frame.exception).__name__ if frame.exception else None,
                )
            elif isinstance(frame, FunctionCallInProgressFrame):
                self._record("tool_start", data, tool_call_id=frame.tool_call_id, function_name=frame.function_name)
            elif isinstance(frame, FunctionCallResultFrame):
                self.tool_results.append(frame.tool_call_id)
                self.tool_results = self.tool_results[-64:]
                self._record(
                    "tool_end",
                    data,
                    tool_call_id=frame.tool_call_id,
                    function_name=frame.function_name,
                    error=redact_secrets(frame.error) if frame.error else None,
                )
            elif isinstance(frame, LLMFullResponseStartFrame):
                self.followup_tools, self.tool_results = self.tool_results, []
                self._record("llm_start", data, tool_call_ids=list(self.followup_tools))
            elif isinstance(frame, LLMFullResponseEndFrame):
                self._record("llm_end", data)
            elif isinstance(frame, InterruptionFrame):
                self._record("interruption", data)
            elif isinstance(frame, (EndFrame, CancelFrame)):
                self._record("session_end", data, reason=type(frame).__name__)
        if not downstream:
            return
        if data.destination is self.output:
            context = getattr(frame, "context_id", None)
            if isinstance(frame, TTSStartedFrame):
                self.input_epoch = self._epoch(context or f"frame-{frame.id}", frame.id)
                self.markers[frame.id] = dict(self.input_epoch)
            elif isinstance(frame, OutputAudioRawFrame) and self.input_epoch:
                duration = len(frame.audio) * 1000 / (2 * frame.num_channels * frame.sample_rate)
                epoch = self._epoch(context, frame.id) if context else self.input_epoch
                epoch["input_audio_ms"] += duration
            elif isinstance(frame, AggregatedTextFrame) and frame.will_be_spoken:
                epoch = self._epoch(context, frame.id)
                # Token-streaming text is not necessarily aligned with already-produced PCM.
                aligned = (
                    getattr(data.source, "_is_streaming_tokens", None) is False and str(frame.text_type) == "sentence"
                )
                self.markers[frame.id] = {
                    "id": epoch["id"],
                    "source_audio_ms": epoch["input_audio_ms"],
                    "sentence_aligned": aligned and str(frame.text_type) == "sentence",
                }
                self._record(
                    "text_unit_queued",
                    data,
                    epoch=epoch["id"],
                    frame_id=frame.id,
                    source_audio_ms=epoch["input_audio_ms"],
                    sentence_aligned=aligned,
                    text_type=str(frame.text_type),
                    characters=len(frame.text),
                )
            if len(self.markers) > 4096:
                self.markers.popitem(last=False)
                self.dropped += 1
        if data.source is not self.output:
            return
        if isinstance(frame, TTSStartedFrame):
            epoch = self.markers.pop(frame.id, self.input_epoch or {})
            self.output_epoch = epoch.get("id")
            self._record(
                "audio_epoch_start", data, epoch=self.output_epoch, tool_call_ids=epoch.get("tool_call_ids", [])
            )
        elif isinstance(frame, OutputAudioRawFrame):
            self.audio_sequence += 1
            self.output_audio_ms += len(frame.audio) * 1000 / (2 * frame.num_channels * frame.sample_rate)
        elif isinstance(frame, AggregatedTextFrame) and frame.will_be_spoken:
            marker = self.markers.pop(frame.id, None)
            if marker:
                self._record(
                    "sentence_boundary",
                    data,
                    epoch=marker["id"],
                    source_audio_ms=marker["source_audio_ms"],
                    sentence_aligned=marker["sentence_aligned"],
                    frame_id=frame.id,
                )
        elif isinstance(frame, TTSStoppedFrame):
            self._record("audio_epoch_end", data, epoch=self.output_epoch)
        elif isinstance(frame, (BotStartedSpeakingFrame, BotStoppedSpeakingFrame)):
            self._record(
                "speech_start" if isinstance(frame, BotStartedSpeakingFrame) else "speech_stop",
                data,
                epoch=self.output_epoch,
            )


def make_perf_diagnostics_observer(transport) -> PerfDiagnosticsObserver:
    """Build the correct metadata emitter for RTVI or the local Realtime gateway."""
    output = transport.output()
    realtime = getattr(transport, "_realtime_serializer", None)

    async def emit(payload):
        if realtime is not None and realtime.emit is not None:
            await realtime.emit(payload)
        else:
            await output.send_message(
                OutputTransportMessageUrgentFrame(
                    message={
                        "label": "rtvi-ai",
                        "type": "server-message",
                        "data": payload,
                    }
                )
            )

    return PerfDiagnosticsObserver(output, emit)
