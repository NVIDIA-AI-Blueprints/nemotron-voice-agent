# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Opt-in timing hooks for scaling experiments (``NVA_PERF_DIAGNOSTICS=1``). No PCM, text, credentials or model
parameters are recorded. Events go to ``$NVA_PERF_DIAGNOSTICS_DIR/events-<pid>.jsonl``."""

import asyncio
from contextvars import ContextVar
import json
import os
import threading
import time
import uuid
from pathlib import Path

_ENABLED = os.getenv("NVA_PERF_DIAGNOSTICS", "0") == "1"
_FILE = None
_MONITOR = None
_EVENTS = 0
_SESSION = "worker"
_AUDIO_QUEUE_FRAME = ContextVar("audio_queue_frame", default=None)


def record(kind, session=None, **data):
    global _FILE, _EVENTS
    if not _ENABLED:
        return
    if _FILE is None:
        root = Path(os.environ.get("NVA_PERF_DIAGNOSTICS_DIR", "/tmp/nva-diagnostics"))
        root.mkdir(parents=True, exist_ok=True)
        _FILE = (root / f"events-{os.getpid()}.jsonl").open("a", buffering=256 * 1024)
    _FILE.write(
        json.dumps(
            {
                "time_ns": time.time_ns(),
                "monotonic_ns": time.monotonic_ns(),
                "pid": os.getpid(),
                "probe_version": 2,
                "session": session or _SESSION,
                "kind": kind,
                **data,
            },
            separators=(",", ":"),
        )
        + "\n"
    )
    _EVENTS += 1


async def _monitor_loop():
    previous = time.monotonic_ns()
    previous_cpu = time.thread_time_ns()
    last_flush = previous
    max_lag = 0.0
    samples = 0
    try:
        while True:
            await asyncio.sleep(0.05)
            now = time.monotonic_ns()
            cpu = time.thread_time_ns()
            lag = max(0, now - previous - 50_000_000) / 1e6
            if lag > 10:
                record(
                    "event_loop_lag",
                    lag_ms=lag,
                    window_start_ns=previous,
                    sample_end_ns=now,
                    thread_cpu_ms=(cpu - previous_cpu) / 1e6,
                    native_tid=threading.get_native_id(),
                )
            previous = now
            previous_cpu = cpu
            max_lag = max(max_lag, lag)
            samples += 1
            if now - last_flush >= 1_000_000_000:
                record(
                    "event_loop_summary",
                    max_lag_ms=max_lag,
                    samples=samples,
                    events_written=_EVENTS,
                )
                if _FILE:
                    _FILE.flush()
                last_flush = now
                max_lag = 0.0
                samples = 0
    finally:
        if _FILE:
            _FILE.flush()


def ensure_monitor():
    global _MONITOR
    if not _ENABLED:
        return
    try:
        if _MONITOR is None:
            _MONITOR = asyncio.create_task(_monitor_loop())
    except RuntimeError:
        return


def client_audio(session, frame, buffer_seconds, last_update_time):
    ensure_monitor()
    record(
        "client_audio",
        session,
        nbytes=len(frame.audio),
        sample_rate=frame.sample_rate,
        channels=frame.num_channels,
        buffer_before_s=buffer_seconds,
        last_update_wall_s=last_update_time,
    )


def _probe_audio_queue(queue, session):
    if getattr(queue, "_perf_profiled", False):
        return
    queue._perf_profiled = True
    from pipecat.frames.frames import OutputAudioRawFrame

    original_put = queue.put_nowait
    original_get = queue.get_nowait
    original_reset = queue.reset
    resetting = False

    def observe(kind, frame):
        if isinstance(frame, OutputAudioRawFrame):
            record(
                kind,
                session,
                frame_id=frame.id,
                queue_frames=queue.qsize(),
                nbytes=len(frame.audio),
                sample_rate=frame.sample_rate,
                channels=frame.num_channels,
            )

    def put_nowait(frame):
        original_put(frame)
        if not resetting:
            observe("audio_queue_put", frame)

    def get_nowait():
        frame = original_get()
        observe("audio_queue_reset_pop" if resetting else "audio_queue_get", frame)
        return frame

    def reset():
        nonlocal resetting
        resetting = True
        try:
            original_reset()
        finally:
            resetting = False

    queue.put_nowait = put_nowait
    queue.get_nowait = get_nowait
    queue.reset = reset


def attach(tts, output, websocket=None, *, llm=None):
    if not _ENABLED:
        return
    ensure_monitor()
    peer = getattr(websocket, "client", None)
    session = f"{peer.host}:{peer.port}" if peer else uuid.uuid4().hex
    record(
        "session_attach",
        session,
        tts_name=str(tts),
        streaming=str(getattr(tts, "_streaming", "unknown")),
        native_tid=threading.get_native_id(),
        output_name=str(output),
    )
    if llm is not None:
        original_llm_push = llm.push_frame

        async def push_llm(frame, *args, **kwargs):
            name = type(frame).__name__
            if name in {"LLMTextFrame", "LLMFullResponseStartFrame", "LLMFullResponseEndFrame"}:
                record("llm_output", session, frame_type=name,
                       characters=len(getattr(frame, "text", "") or ""))
            return await original_llm_push(frame, *args, **kwargs)

        llm.push_frame = push_llm
    original_tts_run = tts.run_tts

    async def synthesize(text, context_id):
        record("tts_text_ready", session, context=context_id, characters=len(text))
        async for item in original_tts_run(text, context_id):
            yield item

    tts.run_tts = synthesize
    original_send_json = tts._send_json

    async def send_json(payload):
        kind = payload.get("type")
        record("tts_provider_input", session, context=getattr(tts, "_synth_context_id", None), event=kind)
        return await original_send_json(payload)

    tts._send_json = send_json
    original_emit = tts._emit_audio

    async def emit(chunk, context_id):
        record(
            "tts_pcm",
            session,
            context=context_id,
            nbytes=len(chunk),
            sample_rate=getattr(tts, "_output_sample_rate", None)
            or getattr(tts, "sample_rate", None),
        )
        return await original_emit(chunk, context_id)

    tts._emit_audio = emit
    original_tokens = tts._send_input_tokens

    async def tokens(ids):
        record(
            "tts_input_tokens",
            session,
            context=getattr(tts, "_synth_context_id", None),
            count=len(ids),
        )
        return await original_tokens(ids)

    tts._send_input_tokens = tokens
    original_control = tts._handle_control_event

    async def control(event, context_id):
        if event.get("type") in ["audio.start", "audio.done", "session.done", "error"]:
            record(
                "tts_control",
                session,
                context=context_id,
                event=event.get("type"),
                error=bool(event.get("error")),
                request_id=event.get("request_id"),
            )
        return await original_control(event, context_id)

    tts._handle_control_event = control
    original_process = output.process_frame
    original_ready = output.set_transport_ready

    def instrument_queues():
        for sender in output._media_senders.values():
            queue = getattr(sender, "_audio_queue", None)
            if queue is not None:
                _probe_audio_queue(queue, session)

    async def ready(frame):
        await original_ready(frame)
        instrument_queues()

    output.set_transport_ready = ready

    async def process(frame, direction):
        if type(frame).__name__ in [
            "InterruptionFrame",
            "TTSStartedFrame",
            "TTSStoppedFrame",
            "EndFrame",
            "CancelFrame",
        ]:
            record(
                "output_control",
                session,
                frame_type=type(frame).__name__,
                context=getattr(frame, "context_id", None),
            )
        if any(cls.__name__ == "OutputAudioRawFrame" for cls in type(frame).__mro__):
            instrument_queues()
            record(
                "output_enqueue",
                session,
                context=getattr(frame, "context_id", None),
                frame_id=getattr(frame, "id", None),
                frame_type=type(frame).__name__,
                nbytes=len(frame.audio),
                sample_rate=getattr(frame, "sample_rate", None),
                channels=getattr(frame, "num_channels", None),
            )
        return await original_process(frame, direction)

    output.process_frame = process
    original_audio = output.write_audio_frame

    async def write_audio(frame):
        token = _AUDIO_QUEUE_FRAME.set(frame.id)
        try:
            return await original_audio(frame)
        finally:
            _AUDIO_QUEUE_FRAME.reset(token)

    output.write_audio_frame = write_audio
    original_socket_send = output._client.send

    async def socket_send(data):
        frame_id = _AUDIO_QUEUE_FRAME.get()
        if frame_id is None:
            return await original_socket_send(data)
        start = time.monotonic_ns()
        record("audio_socket_send_start", session, queue_frame_id=frame_id)
        try:
            return await original_socket_send(data)
        finally:
            record(
                "audio_socket_send_end", session, queue_frame_id=frame_id,
                elapsed_ms=(time.monotonic_ns() - start) / 1e6,
            )

    output._client.send = socket_send
    original_write = output._write_frame

    async def write(frame):
        audio = hasattr(frame, "audio")
        start = time.monotonic_ns()
        if audio:
            rate = getattr(frame, "sample_rate", 0) or 0
            channels = getattr(frame, "num_channels", 1) or 1
            output._perf_last_audio_ms = (
                1000 * len(frame.audio) / (2 * channels * rate) if rate else None
            )
            record(
                "wire_send_start",
                session,
                frame_id=getattr(frame, "id", None),
                queue_frame_id=_AUDIO_QUEUE_FRAME.get(),
                nbytes=len(frame.audio),
                sample_rate=rate,
                channels=channels,
                audio_ms=output._perf_last_audio_ms,
            )
        result = await original_write(frame)
        if audio:
            record(
                "wire_send_end",
                session,
                frame_id=getattr(frame, "id", None),
                queue_frame_id=_AUDIO_QUEUE_FRAME.get(),
                elapsed_ms=(time.monotonic_ns() - start) / 1e6,
                success=result,
            )
        return result

    output._write_frame = write
    original_sleep = output._write_audio_sleep

    async def sleep():
        start = time.monotonic()
        expected = max(0, getattr(output, "_next_send_time", start) - start)
        await original_sleep()
        elapsed = time.monotonic() - start
        record(
            "transport_sleep",
            session,
            requested_ms=expected * 1000,
            actual_ms=elapsed * 1000,
            overshoot_ms=max(0, elapsed - expected) * 1000,
            configured_interval_ms=getattr(output, "_send_interval", 0) * 1000,
            previous_packet_audio_ms=getattr(output, "_perf_last_audio_ms", None),
        )

    output._write_audio_sleep = sleep
