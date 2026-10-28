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
import sys
import time
import traceback
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

from grpc import RpcError, StatusCode

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


class _LiveEventWriter:
    """Bound text-stream buffering without backpressuring the media scheduler."""

    def __init__(self, websocket, limits):
        self.websocket = websocket
        self.limits = limits
        self.queue = asyncio.Queue(maxsize=limits.max_pending_stream_events)
        self.pending_bytes = 0
        self.last_sample = 0

    def observe(self, event):
        """Snapshot a committed event or fail explicitly instead of dropping it."""
        sample = event.get("sample")
        if type(sample) is not int or sample < self.last_sample:
            raise OfflineProtocolError("event_order", "Incremental semantic event moved backwards")
        payload = {"type": "offline.event", "event": redact_private(event)}
        size = len(json.dumps(payload, allow_nan=False).encode("utf-8"))
        if self.queue.full() or self.pending_bytes + size > self.limits.max_pending_stream_bytes:
            raise OfflineProtocolError("event_queue_overflow", "Client cannot keep up with incremental events")
        self.queue.put_nowait((payload, size))
        self.pending_bytes += size
        self.last_sample = sample

    async def run(self):
        """Use one writer to preserve the exact observed append order."""
        while True:
            payload, size = await self.queue.get()
            try:
                await self.websocket.send_json(payload)
            finally:
                self.pending_bytes -= size
                self.queue.task_done()


async def _execute_text(websocket, receiver, runtime_factory, start, source, limits):
    """Publish raw event evidence while native providers continue, until disconnect."""
    runtime = None
    prepare = runner = writer = drain = None
    try:

        async def prepare_runtime():
            nonlocal runtime
            value = runtime_factory(start)
            runtime = await value if inspect.isawaitable(value) else value
            configuration = redact_private(getattr(runtime, "configuration", start.get("config", {})))
            _check_client(receiver)
            await websocket.send_json(
                {
                    "type": "offline.runtime.configuration",
                    "protocol": PROTOCOL,
                    "completion": "text",
                    "configuration": configuration,
                    "configuration_fingerprint": configuration_fingerprint(configuration),
                }
            )

        prepare = asyncio.create_task(prepare_runtime())
        await asyncio.wait((prepare, receiver), return_when=asyncio.FIRST_COMPLETED)
        if prepare.done():
            prepare.result()
        _check_client(receiver)
        await prepare
        stream = _LiveEventWriter(websocket, limits)
        writer = asyncio.create_task(stream.run())
        runner = asyncio.create_task(runtime.run(source, event_sink=stream.observe))

        async def guarded(operation):
            await asyncio.wait((operation, writer, receiver), return_when=asyncio.FIRST_COMPLETED)
            if operation.done():
                operation.result()
            _check_client(receiver)
            if writer.done():
                writer.result()
                raise RuntimeError("Incremental event writer stopped unexpectedly")
            return await operation

        # No server-side candidate choice: the same native client policy decides
        # which raw text epoch to score, then closes its connection immediately.
        await guarded(runner)

        async def finish_stream():
            await stream.queue.join()
            _check_client(receiver)
            await websocket.send_json({"type": "offline.events.complete", "protocol": PROTOCOL, "completion": "text"})

        drain = asyncio.create_task(finish_stream())
        await guarded(drain)
    finally:
        primary_error = sys.exception()
        for task in (prepare, runner, writer, drain):
            if task is not None and not task.done():
                task.cancel()
        tasks = [task for task in (prepare, runner, writer, drain) if task is not None]
        outcomes = await asyncio.gather(*tasks, return_exceptions=True)
        # Native cleanup can raise from a cancelled runner's finally block.
        # Preserve that real provider failure rather than reclassifying it as
        # a normal text-scope disconnect. Closed-socket writer failures are
        # already handled by guarded() and do not attest provider failure.
        cleanup_errors = [
            outcome
            for task, outcome in zip(tasks, outcomes, strict=True)
            if task in (prepare, runner)
            and isinstance(outcome, BaseException)
            and not isinstance(outcome, asyncio.CancelledError)
            and outcome is not primary_error
        ]
        if runtime is not None and hasattr(runtime, "aclose"):
            try:
                await runtime.aclose()
            except Exception as error:
                cleanup_errors.append(error)
        if cleanup_errors:
            if primary_error is None or isinstance(primary_error, (_ClientDisconnected, asyncio.CancelledError)):
                raise cleanup_errors[0]
            primary_error.add_note(
                "Text-stream cleanup also failed: " + ", ".join(type(e).__name__ for e in cleanup_errors)
            )


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
    configuration: dict[str, Any],
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
            "configuration": redact_private(configuration),
            "metrics": redact_private(result.metrics),
        }
    )


def _grpc_status_name(error: Exception) -> str | None:
    """Return only a typed gRPC status enum, excluding provider diagnostics."""
    if not isinstance(error, RpcError):
        return None
    try:
        status = error.code()
    except Exception:
        return None
    return status.name if isinstance(status, StatusCode) else None


def _public_failure_diagnostics(error: Exception) -> dict[str, str]:
    """Read typed diagnostics without replacing the original session failure."""
    diagnostics = {}
    grpc_status = _grpc_status_name(error)
    if grpc_status is not None:
        diagnostics["grpc_status"] = grpc_status
    try:
        # Optional enrichment must not load native providers during preflight,
        # or mask a failure if those providers could not be imported.
        from .native_response import get_native_provider_error_diagnostic

        diagnostics.update(get_native_provider_error_diagnostic(error))
    except Exception:
        pass
    return diagnostics


def _failure_record(directory: Path | None, code: str, *, diagnostics: dict[str, str] | None = None) -> None:
    """Preserve public failure information without provider exceptions or secrets."""
    if directory is not None:
        record = {"status": "error", "code": code}
        record.update(diagnostics or {})
        (directory / "failure.json").write_text(json.dumps(record) + "\n", encoding="utf-8")


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
    start = None
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
                if start.get("completion", "full_session") == "text":
                    await _execute_text(websocket, receiver, runtime_factory, start, source, limits)
                    with contextlib.suppress(Exception):
                        await websocket.close(code=1000)
                    return
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
                    configuration=configuration,
                    before_complete=finalize_metrics,
                    clock=clock,
                )
    except _ClientDisconnected:
        if start is not None and start.get("completion") == "text" and directory is not None:
            (directory / "cancellation.json").write_text(
                json.dumps({"status": "cancelled", "completion": "text", "reason": "client_disconnected"}) + "\n"
            )
        else:
            _failure_record(directory, "client_disconnected")
        return
    except asyncio.CancelledError:
        _failure_record(directory, "server_cancelled")
        raise
    except Exception as error:
        diagnostics = _public_failure_diagnostics(error)
        status_suffix = "".join(f" {name}={value}" for name, value in diagnostics.items())
        logging.getLogger(__name__).error(
            "Offline session %s failed: %s%s\n%s",
            directory.name if directory else "upload",
            type(error).__name__,
            status_suffix,
            "".join(traceback.format_tb(error.__traceback__)),
        )
        if isinstance(error, OfflineProtocolError):
            code, message = error.code, error.message
        elif isinstance(error, TimeoutError):
            code, message = "watchdog_timeout", "Session exceeded an execution watchdog"
        else:
            code, message = "session_failed", "Offline session failed; check server diagnostics"
        _failure_record(directory, code, diagnostics=diagnostics)
        record = {"type": "offline.session.error", "code": code, "message": message}
        record.update(diagnostics)
        with contextlib.suppress(Exception):
            await websocket.send_json(record)
        with contextlib.suppress(Exception):
            await websocket.close(code=1008 if isinstance(error, OfflineProtocolError) else 1011)
        return
    with contextlib.suppress(Exception):
        await websocket.close(code=1000)
