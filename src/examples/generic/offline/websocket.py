# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""WebSocket boundary for finite evaluation, independent of paced transport.

By default the factory follows source validation and sealing. Explicit text-only
upload processing can prepare providers and analyze validated prefixes earlier;
publication still requires final hash validation. Watchdogs never invent silence,
speech stops, source EOF, or media offsets.
"""

import asyncio
import contextlib
import hashlib
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

from .diagnostics import (
    bind_diagnostics,
    create_diagnostics,
    current_diagnostics,
    persist_diagnostics,
    record,
    reset_diagnostics,
)
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
from .recorder import create_private_directory, write_private_text


class _ClientDisconnected(Exception):
    """A disconnect cancels evaluation and never acts as source EOF."""


class _LiveEventWriter:
    """Bound text-stream buffering without backpressuring the media scheduler."""

    def __init__(self, websocket, limits, *, publication_ready=None):
        self.websocket = websocket
        self.limits = limits
        self.queue = asyncio.Queue(maxsize=limits.max_pending_stream_events)
        self.pending_bytes = 0
        self.last_sample = 0
        self._diagnostics = current_diagnostics()
        self._first_text_sent = set()
        self._publication_ready = publication_ready

    def observe(self, event):
        """Snapshot a committed event or fail explicitly instead of dropping it."""
        sample = event.get("sample")
        if type(sample) is not int or sample < self.last_sample:
            raise OfflineProtocolError("event_order", "Incremental semantic event moved backwards")
        payload = {"type": "offline.event", "event": redact_private(event)}
        size = len(json.dumps(payload, allow_nan=False).encode("utf-8"))
        if self.queue.full() or self.pending_bytes + size > self.limits.max_pending_stream_bytes:
            if self._publication_ready is not None and not self._publication_ready.is_set():
                raise OfflineProtocolError(
                    "provisional_event_overflow", "Pre-EOF event buffering exceeded the configured limit"
                )
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
                if self._diagnostics is None:
                    continue
                event = payload.get("event", {})
                kind = event.get("type")
                if kind == "asr_transcript" and event.get("final") is True:
                    record(
                        self._diagnostics,
                        "asr_final_sent",
                        sequence=event.get("sequence"),
                        processed_sample=event.get("processed_sample"),
                        sample=event.get("sample"),
                    )
                elif kind == "response_requested":
                    record(self._diagnostics, "llm_request_event_sent", response_id=event.get("response_id"))
                elif kind == "provider_observation" and event.get("source") == "llm":
                    identity = event.get("response_id")
                    if event.get("frame") == "LLMTextFrame" and event.get("text", "").strip():
                        if identity not in self._first_text_sent and len(self._first_text_sent) < 4096:
                            self._first_text_sent.add(identity)
                            record(self._diagnostics, "llm_first_text_sent", response_id=identity)
                    elif event.get("frame") == "LLMFullResponseEndFrame":
                        record(self._diagnostics, "llm_end_sent", response_id=identity)
            finally:
                self.pending_bytes -= size
                self.queue.task_done()


async def _execute_text(websocket, receiver, runtime_factory, start, source, limits, *, publication_ready=None):
    """Publish raw event evidence while native providers continue, until disconnect."""
    runtime = None
    prepare = runner = writer = drain = None
    try:

        async def prepare_runtime():
            nonlocal runtime
            value = runtime_factory(start)
            runtime = await value if inspect.isawaitable(value) else value
            if publication_ready is not None and hasattr(runtime, "prepare"):
                # Warmup is explicit in the opt-in path. The sealed default
                # retains native initialization inside its existing engine.
                await runtime.prepare()
            configuration = redact_private(getattr(runtime, "configuration", start.get("config", {})))
            _check_client(receiver)
            diagnostics = current_diagnostics()
            payload = {
                "type": "offline.runtime.configuration",
                "protocol": PROTOCOL,
                "completion": "text",
                "configuration": configuration,
                "configuration_fingerprint": configuration_fingerprint(configuration),
                **({"server_metrics_session_id": diagnostics.session_id} if diagnostics is not None else {}),
            }
            if publication_ready is None:
                await websocket.send_json(payload)
            return payload

        prepare = asyncio.create_task(prepare_runtime())
        await asyncio.wait((prepare, receiver), return_when=asyncio.FIRST_COMPLETED)
        if prepare.done():
            prepare.result()
        _check_client(receiver)
        configuration_payload = await prepare
        stream = _LiveEventWriter(websocket, limits, publication_ready=publication_ready)
        writer_ready = asyncio.Event()

        async def write_stream():
            if publication_ready is not None:
                # Pre-EOF provider work is provisional. Never publish text or
                # configuration as a validated-input result before hash seal.
                await publication_ready.wait()
                _check_client(receiver)
                await websocket.send_json(configuration_payload)
            writer_ready.set()
            await stream.run()

        writer = asyncio.create_task(write_stream())
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
            if publication_ready is not None:
                await publication_ready.wait()
            await writer_ready.wait()
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


async def _accepted_start(websocket, limits, *, milestones=None, clock=time.monotonic_ns):
    """Validate the finite request and acknowledge capabilities without providers."""
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
            "input_processing": start.get("input_processing", "sealed"),
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
    return start


async def _sealed_source(
    websocket: Any,
    limits: SessionLimits,
    *,
    milestones: dict[str, int] | None = None,
    clock: Callable[[], int] = time.monotonic_ns,
    start=None,
) -> tuple[dict[str, Any], AudioSource]:
    """Acknowledge capabilities, then receive a finite, hash-validated source."""
    if start is None:
        start = await _accepted_start(websocket, limits, milestones=milestones, clock=clock)
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


async def _execute_during_upload(websocket, runtime_factory, start, directory, limits, upload_deadline):
    """Overlap owned native setup/analysis with ingress, keeping EOF as a validation barrier."""
    from .models import StreamingAudioSource
    from .recorder import write_source

    source = StreamingAudioSource(start["input_samples"], max_samples=limits.max_input_samples)
    upload = InputUpload(start["input_samples"], limits, streaming_source=source)
    sealed = asyncio.Event()
    publication_ready = asyncio.Event()
    execution = pending_receive = None

    async def monitor_after_seal():
        # Exactly one coroutine reads the socket in each phase. A missing
        # packet or a disconnect during upload is never interpreted as EOF.
        await sealed.wait()
        await _receive(websocket)
        raise OfflineProtocolError("input_sealed", "No messages are accepted after offline.input.complete")

    receiver = asyncio.create_task(monitor_after_seal())

    async def run():
        async with asyncio.timeout(limits.execution_timeout_seconds):
            await _execute_text(
                websocket, receiver, runtime_factory, start, source, limits, publication_ready=publication_ready
            )

    try:
        async with asyncio.timeout_at(upload_deadline):
            while True:
                pending_receive = asyncio.create_task(_receive(websocket))
                if execution is not None:
                    await asyncio.wait((pending_receive, execution), return_when=asyncio.FIRST_COMPLETED)
                    if execution.done():
                        execution.result()
                        raise RuntimeError("Streaming execution ended before validated input EOF")
                message = await pending_receive
                pending_receive = None
                if isinstance(message, bytes):
                    upload.append(message)
                    if execution is None:
                        # Start only after a valid nonempty packet. ACK-only
                        # capability probes do not initialize native services.
                        execution = asyncio.create_task(run())
                else:
                    canonical = upload.seal(parse_control(message, limits))
                    record(
                        current_diagnostics(),
                        "source_sealed",
                        input_samples=canonical.sample_count,
                        source_sha256=hashlib.sha256(canonical.pcm).hexdigest(),
                    )
                    sealed.set()
                    await asyncio.to_thread(write_source, directory, canonical)
                    _check_client(receiver)
                    publication_ready.set()
                    break
        if execution is None:
            raise OfflineProtocolError("input_extent", "Streaming input requires a nonempty audio packet")
        await execution
    finally:
        # Abort before joining: the ASR transport can be blocked in a sync
        # chunk reader waiting for ingress. Cancellation must wake that reader.
        if not source.sealed:
            source.abort()
        tasks = [task for task in (execution, pending_receive, receiver) if task is not None]
        for task in tasks:
            if not task.done():
                task.cancel()
        outcomes = await asyncio.gather(*tasks, return_exceptions=True)
        primary = sys.exception()
        errors = [
            outcome
            for task, outcome in zip(tasks, outcomes, strict=True)
            if task is execution
            and isinstance(outcome, BaseException)
            and not isinstance(outcome, asyncio.CancelledError)
            and outcome is not primary
        ]
        if errors:
            if primary is None or isinstance(primary, (_ClientDisconnected, asyncio.CancelledError)):
                raise errors[0]
            primary.add_note("Streaming cleanup also failed: " + type(errors[0]).__name__)


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
        write_private_text(directory / "failure.json", json.dumps(record) + "\n")


async def serve_offline_session(
    websocket: Any,
    runtime_factory: Callable,
    record_root: Path | str,
    *,
    limits: SessionLimits | None = None,
    clock: Callable[[], int] = time.monotonic_ns,
) -> None:
    """Observe optional scalar timings without changing the offline wire policy."""
    diagnostics, root = create_diagnostics()
    if diagnostics is None:
        return await _serve_offline_session(websocket, runtime_factory, record_root, limits=limits, clock=clock)
    token = bind_diagnostics(diagnostics)
    record(diagnostics, "session_started")
    scope = getattr(websocket, "scope", None)
    pool = scope.get("nva.offline.pool") if isinstance(scope, dict) else None
    if not (
        isinstance(pool, dict)
        and isinstance(pool.get("lease_sha256"), str)
        and len(pool["lease_sha256"]) == 64
        and all(c in "0123456789abcdef" for c in pool["lease_sha256"])
        and type(pool.get("handler_started_monotonic_ns")) is int
        and pool["handler_started_monotonic_ns"] > 0
    ):
        pool = None
    if pool is not None:
        pool["server_metrics_session_id"] = diagnostics.session_id
        record(
            diagnostics,
            "pool_session_bound",
            lease_sha256=pool["lease_sha256"],
            handler_started_monotonic_ns=pool["handler_started_monotonic_ns"],
        )
    with contextlib.suppress(Exception):
        diagnostics.start_loop_observer()
    try:
        return await _serve_offline_session(websocket, runtime_factory, record_root, limits=limits, clock=clock)
    except BaseException as error:
        record(diagnostics, "session_error", error_type=type(error).__name__)
        raise
    finally:
        record(diagnostics, "session_finalizer_entered")
        # Kept for old readers: this has always preceded persistence, and is
        # not the enclosing ASGI handler's actual return boundary.
        record(diagnostics, "session_returned", boundary="pre_persistence")
        try:
            with contextlib.suppress(Exception):
                await diagnostics.stop_loop_observer()
        finally:
            record(diagnostics, "diagnostics_persistence_started")
            if pool is not None:
                pool["diagnostics_persistence_started_monotonic_ns"] = time.monotonic_ns()
            try:
                await asyncio.shield(asyncio.to_thread(persist_diagnostics, root, diagnostics))
            except asyncio.CancelledError:
                raise
            except Exception as error:
                logging.getLogger(__name__).warning("Offline scalar metrics unavailable: %s", type(error).__name__)
            else:
                # Completion cannot appear in the snapshot being written.
                # The trusted worker holder can retain it after native return.
                if pool is not None:
                    pool["diagnostics_persistence_completed_monotonic_ns"] = time.monotonic_ns()
            finally:
                reset_diagnostics(token)


async def _serve_offline_session(
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
        async with asyncio.timeout(limits.upload_timeout_seconds) as upload_timeout:
            start = await _accepted_start(websocket, limits, milestones=milestones, clock=clock)
            if start.get("input_processing", "sealed") == "during_upload":
                # This explicit opt-in keeps the scored sealed-input baseline
                # unchanged. Upload and execution have independent watchdogs.
                deadline = upload_timeout.when()
                upload_timeout.reschedule(None)
                directory = Path(record_root) / str(uuid.uuid4())
                create_private_directory(directory, exist_ok=False)
                await _execute_during_upload(websocket, runtime_factory, start, directory, limits, deadline)
                with contextlib.suppress(Exception):
                    await websocket.close(code=1000)
                return
            start, source = await _sealed_source(websocket, limits, milestones=milestones, clock=clock, start=start)
        if current_diagnostics() is not None:
            record(
                current_diagnostics(),
                "source_sealed",
                input_samples=source.sample_count,
                source_sha256=hashlib.sha256(source.pcm).hexdigest(),
            )
        directory = Path(record_root) / str(uuid.uuid4())
        create_private_directory(directory, exist_ok=False)
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
                    write_private_text, directory / "manifest.json", json.dumps(manifest, indent=2) + "\n"
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
                        write_private_text,
                        directory / "manifest.json",
                        json.dumps(manifest, indent=2) + "\n",
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
        record(current_diagnostics(), "session_cancelled", outcome="client_disconnected")
        if start is not None and start.get("completion") == "text" and directory is not None:
            write_private_text(
                directory / "cancellation.json",
                json.dumps({"status": "cancelled", "completion": "text", "reason": "client_disconnected"}) + "\n",
            )
        else:
            _failure_record(directory, "client_disconnected")
        return
    except asyncio.CancelledError:
        record(current_diagnostics(), "session_cancelled", outcome="server_cancelled")
        _failure_record(directory, "server_cancelled")
        raise
    except Exception as error:
        record(current_diagnostics(), "session_failed", error_type=type(error).__name__)
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
        error_record = {"type": "offline.session.error", "code": code, "message": message}
        error_record.update(diagnostics)
        with contextlib.suppress(Exception):
            await websocket.send_json(error_record)
        with contextlib.suppress(Exception):
            await websocket.close(code=1008 if isinstance(error, OfflineProtocolError) else 1011)
        return
    with contextlib.suppress(Exception):
        await websocket.close(code=1000)
