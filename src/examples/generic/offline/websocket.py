# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""WebSocket boundary for finite evaluation, independent of paced transport.

The factory is invoked only after source validation and sealing. Watchdogs fail
stalled operations; they never invent silence, speech stops, or media offsets.
"""

import asyncio
import contextlib
import inspect
import json
import logging
import time
import traceback
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

from .models import RATE, AudioSource, SessionResult
from .protocol import (
    CAPABILITIES,
    PROTOCOL,
    InputUpload,
    OfflineProtocolError,
    SessionLimits,
    configuration_fingerprint,
    encode_packet,
    parse_control,
    redact_private,
    validate_result,
    validate_start,
)


class _ClientDisconnected(Exception):
    """A disconnect cancels evaluation and never acts as source EOF."""


async def _receive(websocket: Any) -> bytes | str:
    """Receive one ASGI payload while retaining disconnect as a distinct event."""
    message = await websocket.receive()
    if message.get("type") == "websocket.disconnect":
        raise _ClientDisconnected
    if message.get("type") != "websocket.receive":
        raise OfflineProtocolError("invalid_message", "Expected a WebSocket data message")
    if message.get("bytes") is not None and message.get("text") is None:
        return message["bytes"]
    if message.get("text") is not None and message.get("bytes") is None:
        return message["text"]
    raise OfflineProtocolError("invalid_message", "Expected exactly one text or binary payload")


async def _sealed_source(
    websocket: Any,
    limits: SessionLimits,
    *,
    milestones: dict[str, int] | None = None,
    clock: Callable[[], int] = time.monotonic_ns,
) -> tuple[dict[str, Any], AudioSource]:
    """Acknowledge capabilities, then receive a finite, hash-validated source."""
    message = await _receive(websocket)
    if milestones is not None:
        milestones["input_start_received"] = clock()
    if not isinstance(message, str):
        raise OfflineProtocolError("invalid_start", "Session start must be a JSON text message")
    start = validate_start(parse_control(message, limits), limits)
    await websocket.send_json(
        {
            "type": "offline.session.accepted",
            "protocol": PROTOCOL,
            "capabilities": list(CAPABILITIES),
            "output_sample_rate": RATE,
            "request_fingerprint": configuration_fingerprint(start.get("config", {})),
            "limits": {
                "max_input_samples": limits.max_input_samples,
                "max_output_samples": limits.max_output_samples,
                "max_packet_samples": limits.max_packet_samples,
            },
        }
    )
    if milestones is not None:
        milestones["upload_accepted"] = clock()
    upload = InputUpload(start["input_samples"], limits)
    while True:
        message = await _receive(websocket)
        if isinstance(message, bytes):
            upload.append(message)
        else:
            source = upload.seal(parse_control(message, limits))
            if milestones is not None:
                milestones["source_sealed"] = clock()
            return start, source


@contextlib.asynccontextmanager
async def _monitor_client(websocket: Any):
    """Monitor sealed-session misuse and disconnect through output publication."""

    async def reject_more_input():
        await _receive(websocket)
        raise OfflineProtocolError("input_sealed", "No messages are accepted after offline.input.complete")

    receiver = asyncio.create_task(reject_more_input())
    try:
        yield receiver
    finally:
        receiver.cancel()
        await asyncio.gather(receiver, return_exceptions=True)


def _check_client(receiver: asyncio.Task) -> None:
    """Surface a disconnect or forbidden payload that the monitor has received."""
    if receiver.done():
        receiver.result()


async def _execute(
    receiver: asyncio.Task,
    runtime_factory: Callable,
    start: dict[str, Any],
    source: AudioSource,
    *,
    milestones: dict[str, int] | None = None,
    clock: Callable[[], int] = time.monotonic_ns,
):
    """Cancel provider work if the client disconnects or sends data after EOF."""

    async def run():
        build_started = clock()
        runtime = runtime_factory(start)
        if inspect.isawaitable(runtime):
            runtime = await runtime
        run_started = clock()
        if milestones is not None:
            milestones["runtime_build_wall_ns"] = run_started - build_started
        result = await runtime.run(source)
        completed = clock()
        if milestones is not None:
            milestones["runtime_run_wall_ns"] = completed - run_started
            milestones["generation_complete"] = completed
        return runtime, result

    runner = asyncio.create_task(run())
    try:
        done, _ = await asyncio.wait((runner, receiver), return_when=asyncio.FIRST_COMPLETED)
        if receiver in done:
            receiver.result()
        return await runner
    finally:
        if not runner.done():
            runner.cancel()
        await asyncio.gather(runner, return_exceptions=True)


async def _send_result(
    websocket: Any,
    result: SessionResult,
    source: AudioSource,
    limits: SessionLimits,
    fingerprint: str,
    receiver: asyncio.Task,
    *,
    before_complete: Callable[[int | None], Awaitable[SessionResult]] | None = None,
    clock: Callable[[], int] = time.monotonic_ns,
):
    """Emit only committed spans; empty intervals are represented by offsets."""
    sequence = 0
    first_packet_sent_at = None
    for span in result.spans:
        count = len(span.pcm) // 2
        for position in range(0, count, limits.output_packet_samples):
            _check_client(receiver)
            pcm = span.pcm[position * 2 : min(position + limits.output_packet_samples, count) * 2]
            await websocket.send_bytes(encode_packet(b"NVAO", sequence, span.start_sample + position, pcm))
            if first_packet_sent_at is None:
                first_packet_sent_at = clock()
            sequence += 1
    for event in result.events:
        _check_client(receiver)
        await websocket.send_json({"type": "offline.event", "event": redact_private(event)})
    # Yield to the input monitor even when a socket implementation buffers sends
    # synchronously. This schedules tasks; it does not advance presentation time.
    await asyncio.sleep(0)
    _check_client(receiver)
    if before_complete is not None:
        result = await before_complete(first_packet_sent_at)
        _check_client(receiver)
    await websocket.send_json(
        {
            "type": "offline.session.complete",
            "protocol": PROTOCOL,
            "status": "ok",
            "input_samples": source.sample_count,
            "rendered_samples": result.rendered_samples,
            "output_packets": sequence,
            "configuration_fingerprint": fingerprint,
            "metrics": redact_private(result.metrics),
        }
    )


def _failure_record(directory: Path | None, code: str) -> None:
    """Preserve public failure information without provider exceptions or secrets."""
    if directory is not None:
        (directory / "failure.json").write_text(json.dumps({"status": "error", "code": code}) + "\n", encoding="utf-8")


async def serve_offline_session(
    websocket: Any,
    runtime_factory: Callable,
    record_root: Path | str,
    *,
    limits: SessionLimits | None = None,
    clock: Callable[[], int] = time.monotonic_ns,
) -> None:
    """Accept, seal, evaluate, record, and publish one finite WebSocket session.

    ``runtime_factory(start_message)`` may return a runtime or an awaitable. The
    runtime must expose ``async run(AudioSource) -> SessionResult`` and may
    expose a public ``configuration`` dict for the completion fingerprint.
    The route should call this function without accepting the socket first.

    ``metrics.server`` records observed monotonic wall execution time only;
    these measurements never advance conversation time. Upload acceptance is
    the completion of the capabilities send, and source sealing includes
    validation of the EOF extent and hash. Runtime build includes the complete
    factory await; runtime run includes its own cleanup. Output recording
    includes all initial recording and configuration-manifest writes. Output
    publication includes committed audio and event sends through the final
    input-monitor yield. Both E2E scopes stop at that same pre-completion
    boundary, excluding the final metrics-manifest rewrite, completion send
    and socket close. A first output packet is measured when its server send
    await completes, independently of its positioned audio offset, and is
    ``None`` if the result has no committed audio.
    """
    from .recorder import write_session, write_source

    limits = limits or SessionLimits()
    directory = None
    await websocket.accept()
    milestones = {"socket_accepted": clock()}
    try:
        async with asyncio.timeout(limits.upload_timeout_seconds):
            start, source = await _sealed_source(websocket, limits, milestones=milestones, clock=clock)
        directory = Path(record_root) / str(uuid.uuid4())
        directory.mkdir(parents=True, exist_ok=False)
        async with _monitor_client(websocket) as receiver:
            async with asyncio.timeout(limits.execution_timeout_seconds):
                recording_started = clock()
                await asyncio.to_thread(write_source, directory, source)
                milestones["source_recording_wall_ns"] = clock() - recording_started
                _check_client(receiver)
                runtime, result = await _execute(
                    receiver, runtime_factory, start, source, milestones=milestones, clock=clock
                )
            validate_result(result, source, limits)
            configuration = getattr(runtime, "configuration", start.get("config", {}))
            fingerprint = configuration_fingerprint(configuration)
            async with asyncio.timeout(limits.output_timeout_seconds):
                result = replace(result, metrics=redact_private(result.metrics))
                recording_started = clock()
                manifest = await asyncio.to_thread(write_session, directory, source, result)
                manifest["configuration"] = redact_private(configuration)
                manifest["configuration_fingerprint"] = fingerprint
                await asyncio.to_thread(
                    (directory / "manifest.json").write_text, json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
                )
                milestones["output_recording_wall_ns"] = clock() - recording_started
                _check_client(receiver)
                publication_started = clock()

                async def finalize_metrics(first_packet_sent_at: int | None) -> SessionResult:
                    # Freeze a single pre-completion boundary for both the wire
                    # and disk manifest. The metadata rewrite and complete-frame
                    # send happen afterwards, so neither can time itself.
                    precomplete = clock()
                    sealed = milestones["source_sealed"]
                    server = {
                        "clock": "monotonic_ns",
                        "units": "nanoseconds",
                        "socket_accept_to_source_sealed_wall_ns": sealed - milestones["socket_accepted"],
                        "input_start_to_source_sealed_wall_ns": sealed - milestones["input_start_received"],
                        "upload_accepted_to_source_sealed_wall_ns": sealed - milestones["upload_accepted"],
                        "source_recording_wall_ns": milestones["source_recording_wall_ns"],
                        "runtime_build_wall_ns": milestones["runtime_build_wall_ns"],
                        "runtime_run_wall_ns": milestones["runtime_run_wall_ns"],
                        "source_sealed_to_generation_complete_wall_ns": milestones["generation_complete"] - sealed,
                        "output_recording_wall_ns": milestones["output_recording_wall_ns"],
                        "source_sealed_to_first_output_packet_sent_wall_ns": (
                            first_packet_sent_at - sealed if first_packet_sent_at is not None else None
                        ),
                        "output_publication_wall_ns": precomplete - publication_started,
                        "socket_accept_to_precomplete_wall_ns": precomplete - milestones["socket_accepted"],
                        "source_sealed_to_precomplete_wall_ns": precomplete - sealed,
                        "precomplete_excludes": [
                            "final_metrics_manifest_rewrite",
                            "completion_message_send",
                            "socket_close",
                        ],
                        "first_output_packet_scope": "server send await completed; not client receive or playback",
                    }
                    final_result = replace(result, metrics={**result.metrics, "server": server})
                    manifest["metrics"] = final_result.metrics
                    await asyncio.to_thread(
                        (directory / "manifest.json").write_text,
                        json.dumps(manifest, indent=2) + "\n",
                        encoding="utf-8",
                    )
                    return final_result

                await _send_result(
                    websocket,
                    result,
                    source,
                    limits,
                    fingerprint,
                    receiver,
                    before_complete=finalize_metrics,
                    clock=clock,
                )
    except _ClientDisconnected:
        _failure_record(directory, "client_disconnected")
        return
    except asyncio.CancelledError:
        _failure_record(directory, "server_cancelled")
        raise
    except Exception as error:
        logging.getLogger(__name__).error(
            "Offline session %s failed: %s\n%s",
            directory.name if directory else "upload",
            type(error).__name__,
            "".join(traceback.format_tb(error.__traceback__)),
        )
        if isinstance(error, OfflineProtocolError):
            code, message = error.code, error.message
        elif isinstance(error, TimeoutError):
            code, message = "watchdog_timeout", "Session exceeded an execution watchdog"
        else:
            code, message = "session_failed", "Offline session failed; check server diagnostics"
        _failure_record(directory, code)
        with contextlib.suppress(Exception):
            await websocket.send_json({"type": "offline.session.error", "code": code, "message": message})
        with contextlib.suppress(Exception):
            await websocket.close(code=1008 if isinstance(error, OfflineProtocolError) else 1011)
        return
    with contextlib.suppress(Exception):
        await websocket.close(code=1000)
