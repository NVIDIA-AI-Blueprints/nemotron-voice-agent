# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""Client for NemoClaw's response-only committed-turn HTTP contract.

This transport is a response-only bridge. This adapter therefore implements
:class:`EphemeralCommittedTurnPort`, not ``AgentInteractionPort``. It never
manufactures Work, replay, delivery, or durability semantics that the endpoint
does not provide.
"""

from __future__ import annotations

import asyncio
import http.client
import ipaddress
import json
import math
import re
import socket
import ssl
import threading
import time
from collections.abc import AsyncIterator, Callable, Iterator
from concurrent.futures import CancelledError as FutureCancelledError
from concurrent.futures import Future as ConcurrentFuture
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from typing import Any
from urllib.parse import quote, urlsplit

from voiceclaw.adapters.nemoclaw.security import (
    ManagedServiceSecurityError,
    validate_managed_http_origin,
    validate_managed_route_host,
)
from voiceclaw.adapters.result_envelope import (
    MAX_RESULT_DISPLAY_BYTES,
    MAX_RESULT_ENVELOPE_BYTES,
    MAX_RESULT_SPEECH_BYTES,
    ResultEnvelopeLimitError,
    ResultEnvelopeProtocolError,
    ResultEnvelopeStreamParser,
    build_result_envelope_prompt,
)
from voiceclaw.domain.models import (
    BackendCapabilities,
    BackendOperation,
    CapabilitySource,
    Durability,
    EventDelivery,
)
from voiceclaw.model_contracts import ModelContractCatalog, load_model_contract_catalog
from voiceclaw.ports.readiness import (
    SelectedAgentReadinessCode,
    SelectedAgentReadinessError,
)
from voiceclaw.ports.turns import (
    MAX_COMMITTED_TURN_GOAL_BYTES,
    CommittedTurnBackend,
    CommittedTurnCompleted,
    CommittedTurnDisplayDelta,
    CommittedTurnError,
    CommittedTurnEvent,
    CommittedTurnRequest,
    CommittedTurnResult,
)

_RUNTIME_VALUE = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9._:-]{0,126}[A-Za-z0-9])?$")
_ERROR_CODE = re.compile(r"^[a-z][a-z0-9_]{0,127}$")
# NemoClaw validates the final posted ``text`` value, after VoiceClaw adds its
# result envelope contract, against this UTF-8 limit.
_MAX_BACKEND_TURN_TEXT_BYTES = 48 * 1024
_MAX_STREAM_BYTES = 4 * 1024 * 1024
# The outer NDJSON encoder may escape each backslash or quote from the inner
# envelope once more. This wire bound must not narrow either inner limit.
_MAX_NDJSON_EVENT_BYTES = MAX_RESULT_ENVELOPE_BYTES * 2 + 4096
_MAX_EVENT_COUNT = 65_536
_MAX_SMALL_BODY_BYTES = 64 * 1024
_MAX_IDENTIFIER_BYTES = 256
_MIN_BEARER_BYTES = 32
_MAX_BEARER_BYTES = 4096
_ABANDONED_ADMISSION_RECOVERY_SECONDS = 4.0
_ABANDONED_SESSION_CLEANUP_SECONDS = 1.0
_ABANDONED_OPERATION_JOIN_SECONDS = 1.25
_ADAPTER_SHUTDOWN_SECONDS = _ABANDONED_ADMISSION_RECOVERY_SECONDS + _ABANDONED_SESSION_CLEANUP_SECONDS + 0.5
_STREAM_QUEUE_DEPTH = 16
_SESSION_LIFETIME = timedelta(minutes=5)
_SESSION_EXPIRY_TOLERANCE = timedelta(seconds=5)
DEFAULT_RESULT_DISPLAY_BUDGET_BYTES = 8 * 1024
_BASE_EVENT_FIELDS = {"type", "voiceSessionId", "turnId", "responseId"}
_FAILURE_REASONS = {
    "agent_failed",
    "agent_gateway_unavailable",
    "agent_protocol_error",
    "response_too_large",
    "session_closed",
    "session_expired",
    "turn_timeout",
}
_PRIVATE_NETWORKS = (
    ipaddress.ip_network("10.0.0.0/8"),
    ipaddress.ip_network("172.16.0.0/12"),
    ipaddress.ip_network("192.168.0.0/16"),
    ipaddress.ip_network("fc00::/7"),
)


class EndpointPolicy(StrEnum):
    """Destinations to which server-side bearer credentials may be sent."""

    LOOPBACK_ONLY = "loopback_only"
    PRIVATE_NETWORK = "private_network"
    MANAGED_PRIVATE_HTTP = "managed_private_http"


class DeploymentAuthorization(StrEnum):
    """Authentication schemes accepted by a specific NemoClaw transport."""

    BEARER = "bearer"
    NEMOCLAW_SCOPED = "nemoclaw_scoped"


class NemoClawEndpointError(ValueError):
    """A fail-closed endpoint or deployment-bearer configuration error."""


class NemoClawCommittedTurnError(CommittedTurnError):
    """Base class for safe runtime failures without bodies or bearer values."""

    def __init__(self, code: str) -> None:
        """Create a failure containing only a bounded machine-readable code."""
        safe_code = code if _ERROR_CODE.fullmatch(code) else "committed_turn_error"
        super().__init__(safe_code)


class NemoClawRequestValidationError(NemoClawCommittedTurnError):
    """The committed request cannot satisfy NemoClaw's input contract."""


class NemoClawTransportError(NemoClawCommittedTurnError):
    """The endpoint timed out or could not be reached."""


class NemoClawRequestRejected(NemoClawCommittedTurnError):
    """NemoClaw rejected an admission or turn before streaming."""

    def __init__(self, code: str, *, status: int) -> None:
        """Retain the HTTP status without retaining a response body."""
        self.status = status
        super().__init__(code)


class NemoClawProtocolError(NemoClawCommittedTurnError):
    """NemoClaw returned a malformed, oversized, or truncated response."""


class NemoClawBackendFailure(NemoClawCommittedTurnError):
    """The NDJSON stream ended in a canonical ``response.failed`` event."""

    def __init__(self, reason: str) -> None:
        """Expose only a reason from the protocol's closed failure vocabulary."""
        self.reason = reason
        super().__init__(reason)


@dataclass(frozen=True, slots=True)
class _Endpoint:
    scheme: str
    host: str
    port: int | None


@dataclass(frozen=True, slots=True)
class _CleanupAuthority:
    session_id: str
    grant: str
    expires_monotonic: float | None = None

    def expired(self) -> bool:
        return self.expires_monotonic is not None and time.monotonic() >= self.expires_monotonic


@dataclass(frozen=True, slots=True)
class _StreamOutcome:
    """Worker-thread outcome delivered after all provisional display events."""

    result: CommittedTurnResult | None = None
    error: BaseException | None = None

    def __post_init__(self) -> None:
        if (self.result is None) == (self.error is None):
            raise ValueError("stream outcome must contain exactly one result or error")


class _StreamBridge:
    """Bounded cancellation-aware channel from one worker into its event loop."""

    def __init__(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop
        self._queue: asyncio.Queue[CommittedTurnDisplayDelta | _StreamOutcome] = asyncio.Queue(
            maxsize=_STREAM_QUEUE_DEPTH
        )
        self._lock = threading.Lock()
        self._closed = False
        self._pending: ConcurrentFuture[None] | None = None

    async def get(self) -> CommittedTurnDisplayDelta | _StreamOutcome:
        """Wait for the next ordered worker message."""
        return await self._queue.get()

    def publish(self, item: CommittedTurnDisplayDelta | _StreamOutcome) -> None:
        """Apply bounded backpressure while remaining interruptible by ``close``."""
        with self._lock:
            if self._closed:
                raise _ExchangeAbandoned
        put = self._queue.put(item)
        try:
            future = asyncio.run_coroutine_threadsafe(put, self._loop)
        except RuntimeError as exc:
            put.close()
            raise _ExchangeAbandoned from exc
        with self._lock:
            if self._closed:
                future.cancel()
                raise _ExchangeAbandoned
            self._pending = future
        try:
            future.result()
        except (FutureCancelledError, RuntimeError) as exc:
            raise _ExchangeAbandoned from exc
        finally:
            with self._lock:
                if self._pending is future:
                    self._pending = None

    def close(self) -> None:
        """Stop future publications and unblock a producer waiting on backpressure."""
        with self._lock:
            self._closed = True
            pending = self._pending
        if pending is not None:
            pending.cancel()


class _ExchangeAbandoned(Exception):
    """Internal signal that the realtime client no longer consumes this exchange."""


async def _run_daemon[T](function: Callable[[], T], *, name: str) -> T:
    """Run one blocking adapter operation on a daemon thread."""
    completed = threading.Event()
    result: list[T] = []
    error: list[BaseException] = []

    def run() -> None:
        try:
            result.append(function())
        except BaseException as exc:
            error.append(exc)
        finally:
            completed.set()

    threading.Thread(target=run, name=name, daemon=True).start()
    while not completed.is_set():
        await asyncio.sleep(0.025)
    if error:
        raise error[0]
    return result[0]


class _ExchangeControl:
    """Thread-safe handle used to interrupt only the active client-side HTTP exchange."""

    def __init__(self) -> None:
        self._abandoned = threading.Event()
        self._lock = threading.Lock()
        self._abort: Callable[[], None] | None = None

    @property
    def abandoned(self) -> bool:
        """Return whether the owning realtime request has been abandoned."""
        return self._abandoned.is_set()

    def abandon(self) -> None:
        """Close the active HTTP socket without asserting backend Work cancellation."""
        self._abandoned.set()
        with self._lock:
            abort = self._abort
        if abort is not None:
            abort()

    def bind(self, abort: Callable[[], None]) -> None:
        """Register the currently interruptible socket closer."""
        with self._lock:
            self._abort = abort
            abandoned = self._abandoned.is_set()
        if abandoned:
            abort()

    def unbind(self, abort: Callable[[], None]) -> None:
        """Remove a socket closer only if it is still the active one."""
        with self._lock:
            if self._abort is abort:
                self._abort = None

    def raise_if_abandoned(self) -> None:
        """Stop the synchronous exchange at a safe boundary."""
        if self._abandoned.is_set():
            raise _ExchangeAbandoned


class _DuplicateKey(ValueError):
    pass


def _strict_object(raw: bytes) -> dict[str, Any]:
    def collect(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise _DuplicateKey
            value[key] = item
        return value

    try:
        parsed = json.loads(raw.decode("utf-8"), object_pairs_hook=collect)
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError, _DuplicateKey) as exc:
        raise NemoClawProtocolError("protocol_error") from exc
    if not isinstance(parsed, dict):
        raise NemoClawProtocolError("protocol_error")
    return parsed


def _bounded_string(value: Any, *, maximum: int) -> str:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise NemoClawProtocolError("protocol_error")
    try:
        size = len(value.encode("utf-8"))
    except UnicodeEncodeError as exc:
        raise NemoClawProtocolError("protocol_error") from exc
    if size > maximum:
        raise NemoClawProtocolError("response_limit")
    return value


def _validate_bearer(value: str) -> str:
    if not isinstance(value, str):
        raise NemoClawEndpointError("deployment bearer must be a string")
    try:
        encoded = value.encode("ascii")
    except UnicodeEncodeError as exc:
        raise NemoClawEndpointError("deployment bearer must contain visible ASCII only") from exc
    if (
        len(encoded) < _MIN_BEARER_BYTES
        or len(encoded) > _MAX_BEARER_BYTES
        or any(byte < 0x21 or byte > 0x7E for byte in encoded)
    ):
        raise NemoClawEndpointError("deployment bearer must contain 32..4096 visible ASCII bytes")
    return value


def _validate_grant(value: Any) -> str:
    try:
        grant = _bounded_string(value, maximum=_MAX_BEARER_BYTES)
        encoded = grant.encode("ascii")
    except (NemoClawProtocolError, UnicodeEncodeError) as exc:
        raise NemoClawProtocolError("protocol_error") from exc
    if any(byte < 0x21 or byte > 0x7E for byte in encoded):
        raise NemoClawProtocolError("protocol_error")
    return grant


def _validate_endpoint(origin: str, policy: EndpointPolicy) -> _Endpoint:
    if policy is EndpointPolicy.MANAGED_PRIVATE_HTTP:
        try:
            _canonical, host, port = validate_managed_http_origin(origin)
        except ManagedServiceSecurityError as exc:
            raise NemoClawEndpointError(str(exc)) from exc
        return _Endpoint(scheme="http", host=host, port=port)
    if not isinstance(origin, str) or not origin or origin != origin.strip():
        raise NemoClawEndpointError("endpoint origin must not be empty or padded")
    try:
        parsed = urlsplit(origin)
        port = parsed.port
    except ValueError as exc:
        raise NemoClawEndpointError("endpoint origin is malformed") from exc
    if parsed.scheme not in {"http", "https"}:
        raise NemoClawEndpointError("endpoint origin must use http or https")
    if parsed.username is not None or parsed.password is not None:
        raise NemoClawEndpointError("endpoint origin must not contain credentials")
    if parsed.path not in {"", "/"} or parsed.query or parsed.fragment:
        raise NemoClawEndpointError("endpoint origin must not contain a path, query, or fragment")
    if not parsed.hostname or "%" in parsed.hostname:
        raise NemoClawEndpointError("endpoint origin must contain a literal IP address")
    try:
        address = ipaddress.ip_address(parsed.hostname)
    except ValueError as exc:
        raise NemoClawEndpointError("endpoint origin must contain a literal IP address") from exc

    is_private = address.is_loopback or any(address in network for network in _PRIVATE_NETWORKS)
    if policy is EndpointPolicy.LOOPBACK_ONLY and not address.is_loopback:
        raise NemoClawEndpointError("endpoint policy requires a loopback address")
    if policy is EndpointPolicy.PRIVATE_NETWORK and not is_private:
        raise NemoClawEndpointError("endpoint policy requires a loopback or private address")
    if parsed.scheme == "http" and not address.is_loopback:
        raise NemoClawEndpointError("non-loopback NemoClaw endpoints require https")
    return _Endpoint(scheme=parsed.scheme, host=address.compressed, port=port)


def _validate_route_host(value: str | None) -> str | None:
    if value is None:
        return None
    try:
        route, _port = validate_managed_route_host(value)
    except ManagedServiceSecurityError as exc:
        raise NemoClawEndpointError(str(exc)) from exc
    return route


def _media_type(response: http.client.HTTPResponse) -> str:
    return response.getheader("Content-Type", "").split(";", 1)[0].strip().lower()


def _read_bounded(response: http.client.HTTPResponse, maximum: int) -> bytes:
    declared = response.getheader("Content-Length")
    if declared is not None:
        try:
            if int(declared) > maximum:
                raise NemoClawProtocolError("response_limit")
        except ValueError as exc:
            raise NemoClawProtocolError("protocol_error") from exc
    body = response.read(maximum + 1)
    if len(body) > maximum:
        raise NemoClawProtocolError("response_limit")
    return body


class NemoClawCommittedTurnAdapter:
    """Execute one fresh response-only NemoClaw session for each committed turn.

    The deployment bearer is held only by this server-side object. NemoClaw's
    returned session grant remains a local variable, is used only for the turn
    and DELETE requests, and is never returned to callers.
    """

    def __init__(
        self,
        *,
        origin: str,
        deployment_bearer: str | None = None,
        deployment_bearer_loader: Callable[[], str] | None = None,
        deployment_authorization: DeploymentAuthorization = DeploymentAuthorization.BEARER,
        route_host: str | None = None,
        target_ref: str = "backend-selected-per-request",
        endpoint_policy: EndpointPolicy = EndpointPolicy.LOOPBACK_ONLY,
        exchange_deadline_seconds: float = 125.0,
        result_display_budget_bytes: int = DEFAULT_RESULT_DISPLAY_BUDGET_BYTES,
        model_contracts: ModelContractCatalog | None = None,
    ) -> None:
        """Configure a literal-IP endpoint and an admission-and-response deadline."""
        try:
            policy = EndpointPolicy(endpoint_policy)
        except ValueError as exc:
            raise NemoClawEndpointError("unknown endpoint policy") from exc
        if (
            isinstance(exchange_deadline_seconds, bool)
            or not isinstance(exchange_deadline_seconds, (int, float))
            or not math.isfinite(exchange_deadline_seconds)
            or exchange_deadline_seconds <= 0
        ):
            raise NemoClawEndpointError("exchange_deadline_seconds must be a finite positive number")
        if (
            isinstance(result_display_budget_bytes, bool)
            or not isinstance(result_display_budget_bytes, int)
            or not 1 <= result_display_budget_bytes <= MAX_RESULT_DISPLAY_BYTES
        ):
            raise NemoClawEndpointError(
                f"result_display_budget_bytes must be an integer from 1 through {MAX_RESULT_DISPLAY_BYTES}"
            )
        if (deployment_bearer is None) == (deployment_bearer_loader is None):
            raise NemoClawEndpointError("exactly one deployment credential source is required")
        try:
            authorization = DeploymentAuthorization(deployment_authorization)
        except ValueError as exc:
            raise NemoClawEndpointError("unknown deployment authorization") from exc
        managed_route_host = _validate_route_host(route_host)
        if authorization is DeploymentAuthorization.NEMOCLAW_SCOPED:
            if policy is not EndpointPolicy.MANAGED_PRIVATE_HTTP or managed_route_host is None:
                raise NemoClawEndpointError("managed authorization requires its private endpoint and route host")
        elif managed_route_host is not None or policy is EndpointPolicy.MANAGED_PRIVATE_HTTP:
            raise NemoClawEndpointError("managed endpoint routing requires managed authorization")
        self._endpoint = _validate_endpoint(origin, policy)
        if managed_route_host is not None:
            _route, route_port = validate_managed_route_host(managed_route_host)
            if route_port != self._endpoint.port:
                raise NemoClawEndpointError("managed endpoint and route host ports must match")
        if deployment_bearer is not None:
            static_bearer = _validate_bearer(deployment_bearer)
            self._deployment_bearer_loader: Callable[[], str] | None = lambda: static_bearer
        else:
            assert deployment_bearer_loader is not None
            self._deployment_bearer_loader = deployment_bearer_loader
        self._deployment_authorization = authorization
        self._route_host = managed_route_host
        if _RUNTIME_VALUE.fullmatch(target_ref) is None:
            raise NemoClawEndpointError("target_ref is invalid")
        self._target_ref = target_ref
        self._exchange_deadline_seconds = float(exchange_deadline_seconds)
        self._result_display_budget_bytes = result_display_budget_bytes
        self._model_contracts = model_contracts or load_model_contract_catalog()
        self._closed = threading.Event()
        self._discard_authority = threading.Event()
        self._shutdown_started = threading.Event()
        self._shutdown_complete = threading.Event()
        self._shutdown_lock = threading.Lock()
        self._lifecycle_lock = threading.Lock()
        self._active_control: _ExchangeControl | None = None
        self._cleanup_degraded = threading.Event()
        self._cleanup_lock = threading.Lock()
        self._active_cleanup: _CleanupAuthority | None = None
        self._pending_cleanup: _CleanupAuthority | None = None
        # A blocking OS connect may not stop promptly after abandonment. Keep
        # the provider's single-active-session invariant until its worker exits.
        self._operation_gate = threading.Lock()

    def _deployment_bearer(self) -> str:
        with self._lifecycle_lock:
            loader = self._deployment_bearer_loader
        if loader is None:
            raise NemoClawTransportError("adapter_closed")
        try:
            return _validate_bearer(loader())
        except NemoClawEndpointError as exc:
            raise NemoClawTransportError("credential_unavailable") from exc
        except Exception as exc:
            raise NemoClawTransportError("credential_unavailable") from exc

    async def check_selected_agent(self) -> None:
        """Attest the installer-bound target without creating conversational work."""
        if self._closed.is_set():
            raise SelectedAgentReadinessError(SelectedAgentReadinessCode.UNAVAILABLE)
        if self._cleanup_degraded.is_set():
            recovered = await _run_daemon(
                lambda: self._retry_pending_cleanup(timeout_seconds=_ABANDONED_SESSION_CLEANUP_SECONDS),
                name="voiceclaw-nemoclaw-readiness-cleanup",
            )
            if not recovered:
                raise SelectedAgentReadinessError(SelectedAgentReadinessCode.UNAVAILABLE)
        if self._closed.is_set():
            raise SelectedAgentReadinessError(SelectedAgentReadinessCode.UNAVAILABLE)
        try:
            await _run_daemon(self._inspect_sync, name="voiceclaw-nemoclaw-readiness")
        except NemoClawRequestRejected as exc:
            if exc.status in {401, 403}:
                code = SelectedAgentReadinessCode.ACCESS_DENIED
            elif exc.status == 404:
                code = SelectedAgentReadinessCode.WRONG_AGENT
            else:
                code = SelectedAgentReadinessCode.UNAVAILABLE
            raise SelectedAgentReadinessError(code) from exc
        except NemoClawTransportError as exc:
            raise SelectedAgentReadinessError(SelectedAgentReadinessCode.ENDPOINT_UNAVAILABLE) from exc
        except NemoClawProtocolError as exc:
            raise SelectedAgentReadinessError(SelectedAgentReadinessCode.PROTOCOL_ERROR) from exc
        except NemoClawCommittedTurnError as exc:
            raise SelectedAgentReadinessError() from exc
        if self._closed.is_set() or self._cleanup_degraded.is_set():
            raise SelectedAgentReadinessError(SelectedAgentReadinessCode.UNAVAILABLE)

    async def inspect(self) -> CommittedTurnBackend:
        """Check only authenticated gateway health, not target-agent readiness."""
        if self._closed.is_set():
            raise NemoClawTransportError("adapter_closed")
        await _run_daemon(self._inspect_sync, name="voiceclaw-nemoclaw-inspect")
        if self._closed.is_set():
            raise NemoClawTransportError("adapter_closed")
        return CommittedTurnBackend(
            label="NemoClaw",
            target_ref=self._target_ref,
            mode="response_only",
            capabilities=BackendCapabilities(
                backend_kind="response_only",
                target_label="NemoClaw",
                revision="nemoclaw-committed-turn-v1",
                operations=frozenset({BackendOperation.SUBMIT}),
                durability=Durability.NONE,
                event_delivery=EventDelivery.RESPONSE_ONLY,
                sessionful=False,
                supports_parallel_work=False,
                max_parallel_work=1,
            ),
            capability_source=CapabilitySource.COMPATIBILITY_PROJECTION,
            capability_source_id="nemoclaw_committed_turn",
        )

    async def commit_turn(self, request: CommittedTurnRequest) -> CommittedTurnResult:
        """Consume the typed stream and return its one validated terminal result."""
        terminal: CommittedTurnResult | None = None
        async for event in self.stream_turn(request):
            if isinstance(event, CommittedTurnCompleted):
                terminal = event.result
        if terminal is None:
            raise NemoClawProtocolError("truncated_stream")
        return terminal

    async def stream_turn(self, request: CommittedTurnRequest) -> AsyncIterator[CommittedTurnEvent]:
        """Bridge blocking NDJSON into provisional display and terminal events."""
        if not isinstance(request, CommittedTurnRequest):
            raise NemoClawRequestValidationError("invalid_request")
        if self._closed.is_set():
            raise NemoClawTransportError("adapter_closed")
        if self._cleanup_degraded.is_set() and not await _run_daemon(
            self._retry_pending_cleanup,
            name="voiceclaw-nemoclaw-cleanup",
        ):
            raise NemoClawTransportError("cleanup_degraded")
        control = _ExchangeControl()
        with self._lifecycle_lock:
            if self._closed.is_set():
                raise NemoClawTransportError("adapter_closed")
            if not self._operation_gate.acquire(blocking=False):
                raise NemoClawTransportError("backend_busy")
            self._active_control = control
        bridge = _StreamBridge(asyncio.get_running_loop())

        def publish_display_delta(event: CommittedTurnDisplayDelta) -> None:
            control.raise_if_abandoned()
            bridge.publish(event)

        completed = threading.Event()

        def run_exchange() -> None:
            try:
                try:
                    result = self._commit_turn_sync(
                        request,
                        control,
                        on_display_delta=publish_display_delta,
                    )
                    outcome = _StreamOutcome(result=result)
                except BaseException as exc:
                    outcome = _StreamOutcome(error=exc)
                with suppress(_ExchangeAbandoned):
                    bridge.publish(outcome)
            finally:
                self._release_operation(control)
                completed.set()

        worker = threading.Thread(target=run_exchange, name="voiceclaw-nemoclaw-turn", daemon=True)
        try:
            worker.start()
        except BaseException:
            self._release_operation(control)
            completed.set()
            raise
        retired = False
        try:
            while True:
                item = await bridge.get()
                if isinstance(item, CommittedTurnDisplayDelta):
                    yield item
                    continue
                bridge.close()
                retired = True
                await self._retire_operation(completed, control, abandon=False)
                if item.error is not None:
                    raise item.error
                if item.result is None:
                    raise NemoClawProtocolError("truncated_stream")
                yield CommittedTurnCompleted(result=item.result)
                return
        finally:
            if not retired:
                retired = True
                bridge.close()
                await self._retire_operation(completed, control, abandon=True)

    async def _retire_operation(
        self,
        completed: threading.Event,
        control: _ExchangeControl,
        *,
        abandon: bool,
    ) -> None:
        """Bound stream teardown while the worker retains admission ownership."""
        if abandon:
            # Presentation abandonment is not backend Work cancellation. Close
            # only this client-side exchange and let the worker perform its
            # existing best-effort DELETE of the temporary backend session.
            control.abandon()
        cancelled = False
        cleanup_deadline = asyncio.get_running_loop().time() + _ABANDONED_OPERATION_JOIN_SECONDS
        while not completed.is_set() and asyncio.get_running_loop().time() < cleanup_deadline:
            try:
                await asyncio.sleep(min(0.025, max(0.0, cleanup_deadline - asyncio.get_running_loop().time())))
            except asyncio.CancelledError:
                cancelled = True
        if cancelled:
            raise asyncio.CancelledError

    def _release_operation(self, control: _ExchangeControl) -> None:
        """Forget one retired operation before reopening the single-operation gate."""
        with self._lifecycle_lock:
            if self._active_control is control:
                self._active_control = None
        self._operation_gate.release()

    async def shutdown(self) -> None:
        """Bound remote-session cleanup and discard retained bearer authority."""
        start_shutdown = False
        with self._lifecycle_lock:
            self._closed.set()
            control = self._active_control
            if not self._shutdown_started.is_set():
                self._shutdown_started.set()
                start_shutdown = True
        if control is not None:
            control.abandon()
        if start_shutdown:
            worker = threading.Thread(
                target=self._shutdown_sync,
                name="voiceclaw-nemoclaw-shutdown",
                daemon=True,
            )
            try:
                worker.start()
            except BaseException:
                self._clear_sensitive_state()
                self._shutdown_complete.set()
                raise

        cancelled = False
        deadline = asyncio.get_running_loop().time() + _ADAPTER_SHUTDOWN_SECONDS + 0.25
        while not self._shutdown_complete.is_set() and asyncio.get_running_loop().time() < deadline:
            try:
                await asyncio.sleep(0.025)
            except asyncio.CancelledError:
                cancelled = True
        if not self._shutdown_complete.is_set():
            self._clear_sensitive_state()
            self._shutdown_complete.set()
        if cancelled:
            raise asyncio.CancelledError

    def _shutdown_sync(self) -> None:
        """Serialize one fixed-budget shutdown without retaining failed authority."""
        if self._shutdown_complete.is_set():
            return
        deadline = time.monotonic() + _ADAPTER_SHUTDOWN_SECONDS
        if not self._shutdown_lock.acquire(timeout=_ADAPTER_SHUTDOWN_SECONDS):
            self._clear_sensitive_state()
            self._shutdown_complete.set()
            return
        try:
            if self._shutdown_complete.is_set():
                return
            join_seconds = max(
                0.0,
                deadline - time.monotonic() - _ABANDONED_SESSION_CLEANUP_SECONDS,
            )
            acquired = self._operation_gate.acquire(timeout=join_seconds)
            try:
                self._cleanup_for_shutdown(deadline)
            finally:
                if acquired:
                    self._operation_gate.release()
                self._clear_sensitive_state()
                self._shutdown_complete.set()
        finally:
            self._shutdown_lock.release()

    def _cleanup_for_shutdown(self, deadline: float) -> None:
        """Try each retained session authority once within the shutdown budget."""
        with self._cleanup_lock:
            authorities = tuple(
                dict.fromkeys(
                    authority for authority in (self._active_cleanup, self._pending_cleanup) if authority is not None
                )
            )
        for authority in authorities:
            remaining = min(_ABANDONED_SESSION_CLEANUP_SECONDS, deadline - time.monotonic())
            if remaining <= 0:
                break
            self._delete_once(authority, timeout_seconds=remaining)

    def _clear_sensitive_state(self) -> None:
        """Drop every adapter-owned reference to deployment or session authority."""
        self._discard_authority.set()
        with self._cleanup_lock:
            self._active_cleanup = None
            self._pending_cleanup = None
            self._cleanup_degraded.clear()
        with self._lifecycle_lock:
            self._deployment_bearer_loader = None
            self._active_control = None

    def _commit_turn_sync(
        self,
        request: CommittedTurnRequest,
        control: _ExchangeControl,
        *,
        on_display_delta: Callable[[CommittedTurnDisplayDelta], None] | None = None,
    ) -> CommittedTurnResult:
        turn_text = self._validate_request(
            request,
            display_budget_bytes=self._result_display_budget_bytes,
        )
        control.raise_if_abandoned()
        deadline = time.monotonic() + self._exchange_deadline_seconds
        authority: _CleanupAuthority | None = None
        try:
            authority = self._admit(request.runtime_conversation_id, deadline=deadline, control=control)
            control.raise_if_abandoned()
            return self._exchange(
                authority.session_id,
                authority.grant,
                request,
                turn_text=turn_text,
                deadline=deadline,
                control=control,
                on_display_delta=on_display_delta,
            )
        except NemoClawCommittedTurnError:
            raise
        except TimeoutError as exc:
            raise NemoClawTransportError("timeout") from exc
        except (OSError, http.client.HTTPException, ssl.SSLError) as exc:
            raise NemoClawTransportError("transport_error") from exc
        except _ExchangeAbandoned as exc:
            code = "adapter_closed" if self._closed.is_set() else "transport_error"
            raise NemoClawTransportError(code) from exc
        finally:
            if authority is not None:
                cleanup_seconds = (
                    _ABANDONED_SESSION_CLEANUP_SECONDS
                    if control.abandoned
                    else min(self._exchange_deadline_seconds, 5.0)
                )
                self._delete_best_effort(authority, timeout_seconds=cleanup_seconds)
            authority = None

    def _validate_request(
        self,
        request: CommittedTurnRequest,
        *,
        display_budget_bytes: int = MAX_RESULT_DISPLAY_BYTES,
    ) -> str:
        if _RUNTIME_VALUE.fullmatch(request.runtime_conversation_id) is None:
            raise NemoClawRequestValidationError("invalid_runtime_conversation_id")
        if _RUNTIME_VALUE.fullmatch(request.commit_id) is None:
            raise NemoClawRequestValidationError("invalid_commit_id")
        try:
            goal_bytes = request.text.encode("utf-8")
        except UnicodeEncodeError as exc:
            raise NemoClawRequestValidationError("invalid_text") from exc
        if len(goal_bytes) > MAX_COMMITTED_TURN_GOAL_BYTES:
            raise NemoClawRequestValidationError("goal_too_large")
        try:
            turn_text = build_result_envelope_prompt(
                request.text,
                display_budget_bytes=display_budget_bytes,
                contracts=self._model_contracts,
            )
            encoded = turn_text.encode("utf-8")
        except (ResultEnvelopeProtocolError, UnicodeEncodeError) as exc:
            raise NemoClawRequestValidationError("invalid_text") from exc
        if not encoded or b"\x00" in encoded:
            raise NemoClawRequestValidationError("invalid_text")
        if len(encoded) > _MAX_BACKEND_TURN_TEXT_BYTES:
            raise NemoClawRequestValidationError("turn_payload_too_large")
        return turn_text

    def _connection(self, *, timeout_seconds: float) -> http.client.HTTPConnection:
        connection_type = (
            http.client.HTTPSConnection if self._endpoint.scheme == "https" else http.client.HTTPConnection
        )
        return connection_type(
            self._endpoint.host,
            self._endpoint.port,
            timeout=timeout_seconds,
        )

    @staticmethod
    def _remaining(deadline: float) -> float:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError
        return remaining

    @contextmanager
    def _connection_until(
        self,
        deadline: float,
        *,
        control: _ExchangeControl | None = None,
        abandon_grace_seconds: float = 0.0,
        check_abandoned_on_exit: bool = True,
    ) -> Iterator[tuple[http.client.HTTPConnection, Callable[[http.client.HTTPResponse], None]]]:
        """Abort the socket at a wall-clock deadline, including slow trickles.

        A socket timeout only limits an individual period without bytes. A peer
        that continuously sends partial headers or event lines can otherwise
        keep a request alive forever. The watchdog closes the active socket at
        the absolute deadline so admission plus the NDJSON response has one
        bounded wall-clock budget.
        """
        if control is not None:
            control.raise_if_abandoned()
        remaining = self._remaining(deadline)
        connection = self._connection(timeout_seconds=remaining)
        expired = threading.Event()
        abandonment_expired = threading.Event()
        finished = threading.Event()
        active_response: list[http.client.HTTPResponse | None] = [None]
        abandoned_timer: list[threading.Timer | None] = [None]
        abandoned_timer_lock = threading.Lock()

        def close_active() -> None:
            sockets: list[Any] = [connection.sock]
            response = active_response[0]
            if response is not None and response.fp is not None:
                sockets.append(getattr(getattr(response.fp, "raw", None), "_sock", None))
            for active_socket in sockets:
                with suppress(Exception):
                    if active_socket is not None:
                        active_socket.shutdown(socket.SHUT_RDWR)
            with suppress(Exception):
                if response is not None:
                    response.close()
            with suppress(Exception):
                connection.close()

        def expire() -> None:
            expired.set()
            close_active()

        def close_after_abandonment() -> None:
            abandonment_expired.set()
            if not finished.is_set():
                close_active()

        def abandon_connection() -> None:
            if abandon_grace_seconds <= 0:
                close_active()
                return
            with abandoned_timer_lock:
                if finished.is_set() or abandoned_timer[0] is not None:
                    return
                delay = min(abandon_grace_seconds, max(0.0, deadline - time.monotonic()))
                timer = threading.Timer(delay, close_after_abandonment)
                timer.daemon = True
                abandoned_timer[0] = timer
                timer.start()

        def watch_response(response: http.client.HTTPResponse) -> None:
            active_response[0] = response
            if (
                expired.is_set()
                or abandonment_expired.is_set()
                or (control is not None and control.abandoned and abandon_grace_seconds <= 0)
            ):
                close_active()

        timer = threading.Timer(remaining, expire)
        timer.daemon = True
        if control is not None:
            control.bind(abandon_connection)
        timer.start()
        try:
            if control is not None:
                control.raise_if_abandoned()
            yield connection, watch_response
            if control is not None and check_abandoned_on_exit:
                control.raise_if_abandoned()
            if expired.is_set() or time.monotonic() >= deadline:
                raise TimeoutError
        except Exception as exc:
            if control is not None and control.abandoned:
                raise _ExchangeAbandoned from exc
            if not isinstance(exc, TimeoutError) and (expired.is_set() or time.monotonic() >= deadline):
                raise TimeoutError from exc
            raise
        finally:
            finished.set()
            timer.cancel()
            with abandoned_timer_lock:
                if abandoned_timer[0] is not None:
                    abandoned_timer[0].cancel()
            if control is not None:
                control.unbind(abandon_connection)
            connection.close()

    def _inspect_sync(self) -> None:
        control = _ExchangeControl()
        with self._lifecycle_lock:
            if self._closed.is_set():
                raise NemoClawTransportError("adapter_closed")
            if not self._operation_gate.acquire(blocking=False):
                raise NemoClawTransportError("backend_busy")
            self._active_control = control
        try:
            deadline = time.monotonic() + min(self._exchange_deadline_seconds, 5.0)
            with self._connection_until(deadline, control=control) as (connection, watch_response):
                response: http.client.HTTPResponse | None = None
                try:
                    headers = self._deployment_headers(accepts="application/json")
                    control.raise_if_abandoned()
                    connection.request(
                        "GET",
                        "/healthz",
                        headers=headers,
                    )
                    response = connection.getresponse()
                    watch_response(response)
                    body = _read_bounded(response, _MAX_SMALL_BODY_BYTES)
                    if response.status != 204:
                        self._raise_rejected(response.status, response, body)
                    if body or response.getheader("Content-Encoding", "") not in {"", "identity"}:
                        raise NemoClawProtocolError("protocol_error")
                finally:
                    if response is not None:
                        response.close()
        except NemoClawCommittedTurnError:
            raise
        except TimeoutError as exc:
            raise NemoClawTransportError("timeout") from exc
        except (OSError, http.client.HTTPException, ssl.SSLError) as exc:
            raise NemoClawTransportError("transport_error") from exc
        except _ExchangeAbandoned as exc:
            code = "adapter_closed" if self._closed.is_set() else "transport_error"
            raise NemoClawTransportError(code) from exc
        finally:
            self._release_operation(control)

    @staticmethod
    def _json_body(value: dict[str, str]) -> bytes:
        try:
            return json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        except UnicodeEncodeError as exc:
            raise NemoClawRequestValidationError("invalid_text") from exc

    def _headers(self, bearer: str, *, accepts: str) -> dict[str, str]:
        headers = {
            "Accept": accepts,
            "Connection": "close",
            "Content-Type": "application/json",
        }
        authorization_header = (
            "X-NemoClaw-Authorization"
            if self._deployment_authorization is DeploymentAuthorization.NEMOCLAW_SCOPED
            else "Authorization"
        )
        headers[authorization_header] = f"Bearer {bearer}"
        if self._route_host is not None:
            headers["Host"] = self._route_host
        return headers

    def _deployment_headers(self, *, accepts: str) -> dict[str, str]:
        bearer = self._deployment_bearer()
        headers = {
            "Accept": accepts,
            "Connection": "close",
            "Content-Type": "application/json",
        }
        if self._deployment_authorization is DeploymentAuthorization.NEMOCLAW_SCOPED:
            headers["X-NemoClaw-Authorization"] = f"Bearer {bearer}"
        else:
            headers["Authorization"] = f"Bearer {bearer}"
        if self._route_host is not None:
            headers["Host"] = self._route_host
        return headers

    def _admit(
        self,
        runtime_conversation_id: str,
        *,
        deadline: float,
        control: _ExchangeControl,
    ) -> _CleanupAuthority:
        session_id: str | None = None
        grant: str | None = None
        with self._connection_until(
            deadline,
            control=control,
            abandon_grace_seconds=_ABANDONED_ADMISSION_RECOVERY_SECONDS,
            check_abandoned_on_exit=False,
        ) as (connection, watch_response):
            response: http.client.HTTPResponse | None = None
            try:
                connection.request(
                    "POST",
                    "/v1/voice/sessions",
                    body=self._json_body({"runtimeConversationId": runtime_conversation_id}),
                    headers=self._deployment_headers(accepts="application/json"),
                )
                response = connection.getresponse()
                watch_response(response)
                body = _read_bounded(response, _MAX_SMALL_BODY_BYTES)
                if response.status != 201:
                    self._raise_rejected(response.status, response, body)
                payload = _strict_object(body)

                # Recover cleanup authority before validating non-authority fields.
                session_id = _bounded_string(payload.get("voiceSessionId"), maximum=_MAX_IDENTIFIER_BYTES)
                grant = _validate_grant(payload.get("grant"))
                authority = _CleanupAuthority(session_id=session_id, grant=grant)
                with self._cleanup_lock:
                    if not self._discard_authority.is_set():
                        self._active_cleanup = authority
                try:
                    if _media_type(response) != "application/json" or response.getheader(
                        "Content-Encoding", ""
                    ) not in {
                        "",
                        "identity",
                    }:
                        raise NemoClawProtocolError("protocol_error")
                    if set(payload) != {"voiceSessionId", "grant", "expiresAt"}:
                        raise NemoClawProtocolError("protocol_error")
                    expires_at = _bounded_string(payload.get("expiresAt"), maximum=128)
                    parsed_expiry = datetime.fromisoformat(expires_at.replace("Z", "+00:00"))
                    if parsed_expiry.tzinfo is None:
                        raise ValueError
                    remaining = parsed_expiry.astimezone(UTC) - datetime.now(UTC)
                    if remaining <= timedelta(0) or remaining > _SESSION_LIFETIME + _SESSION_EXPIRY_TOLERANCE:
                        raise ValueError
                    authority = _CleanupAuthority(
                        session_id=session_id,
                        grant=grant,
                        expires_monotonic=time.monotonic() + remaining.total_seconds(),
                    )
                    with self._cleanup_lock:
                        if self._active_cleanup == _CleanupAuthority(session_id=session_id, grant=grant):
                            self._active_cleanup = authority
                except (ValueError, NemoClawProtocolError) as exc:
                    self._delete_best_effort(authority)
                    raise NemoClawProtocolError("protocol_error") from exc
                return authority
            finally:
                if response is not None:
                    response.close()

    def _exchange(
        self,
        session_id: str,
        grant: str,
        request: CommittedTurnRequest,
        *,
        turn_text: str,
        deadline: float,
        control: _ExchangeControl,
        on_display_delta: Callable[[CommittedTurnDisplayDelta], None] | None,
    ) -> CommittedTurnResult:
        with self._connection_until(deadline, control=control) as (connection, watch_response):
            response: http.client.HTTPResponse | None = None
            try:
                path = f"/v1/voice/sessions/{quote(session_id, safe='')}/turns"
                connection.request(
                    "POST",
                    path,
                    body=self._json_body({"commitId": request.commit_id, "text": turn_text}),
                    headers=self._headers(grant, accepts="application/x-ndjson"),
                )
                response = connection.getresponse()
                watch_response(response)
                if response.status != 200:
                    body = _read_bounded(response, _MAX_SMALL_BODY_BYTES)
                    self._raise_rejected(response.status, response, body)
                if _media_type(response) != "application/x-ndjson" or response.getheader(
                    "Content-Encoding", ""
                ) not in {
                    "",
                    "identity",
                }:
                    raise NemoClawProtocolError("protocol_error")
                return self._consume_stream(
                    response,
                    session_id,
                    on_display_delta=on_display_delta,
                    maximum_display_bytes=self._result_display_budget_bytes,
                )
            finally:
                if response is not None:
                    response.close()

    @staticmethod
    def _consume_stream(
        response: http.client.HTTPResponse,
        expected_session_id: str,
        *,
        on_display_delta: Callable[[CommittedTurnDisplayDelta], None] | None,
        maximum_display_bytes: int,
    ) -> CommittedTurnResult:
        event_count = 0
        total_bytes = 0
        text_bytes = 0
        expected_sequence = 0
        started = False
        terminal: str | None = None
        failure: str | None = None
        turn_id: str | None = None
        response_id: str | None = None
        display_sequence = 0

        def publish_display_delta(delta: str) -> None:
            nonlocal display_sequence
            if on_display_delta is None:
                return
            if turn_id is None or response_id is None:
                raise NemoClawProtocolError("protocol_error")
            event = CommittedTurnDisplayDelta(
                backend_session_id=expected_session_id,
                turn_id=turn_id,
                response_id=response_id,
                sequence=display_sequence,
                delta=delta,
            )
            on_display_delta(event)
            display_sequence += 1

        parser = ResultEnvelopeStreamParser(
            publish_display_delta,
            maximum_bytes=MAX_RESULT_ENVELOPE_BYTES,
            maximum_display_bytes=maximum_display_bytes,
            maximum_speech_bytes=MAX_RESULT_SPEECH_BYTES,
        )

        while True:
            line = response.readline(_MAX_NDJSON_EVENT_BYTES + 2)
            if not line:
                break
            total_bytes += len(line)
            if total_bytes > _MAX_STREAM_BYTES or len(line) > _MAX_NDJSON_EVENT_BYTES + 1:
                raise NemoClawProtocolError("response_limit")
            if not line.endswith(b"\n") or line.endswith(b"\r\n") or line == b"\n" or terminal is not None:
                raise NemoClawProtocolError("protocol_error")
            event_count += 1
            if event_count > _MAX_EVENT_COUNT:
                raise NemoClawProtocolError("response_limit")
            event = _strict_object(line[:-1])
            event_type = event.get("type")
            if event_type == "response.text.delta":
                extras = {"sequence", "text"}
            elif event_type == "response.failed":
                extras = {"reason"}
            else:
                extras = set()
            if event_type not in {"response.started", "response.text.delta", "response.completed", "response.failed"}:
                raise NemoClawProtocolError("protocol_error")
            if set(event) != _BASE_EVENT_FIELDS | extras:
                raise NemoClawProtocolError("protocol_error")

            current_session_id = _bounded_string(event.get("voiceSessionId"), maximum=_MAX_IDENTIFIER_BYTES)
            current_turn_id = _bounded_string(event.get("turnId"), maximum=_MAX_IDENTIFIER_BYTES)
            current_response_id = _bounded_string(event.get("responseId"), maximum=_MAX_IDENTIFIER_BYTES)
            if current_session_id != expected_session_id:
                raise NemoClawProtocolError("protocol_error")
            if turn_id is None:
                turn_id, response_id = current_turn_id, current_response_id
            elif turn_id != current_turn_id or response_id != current_response_id:
                raise NemoClawProtocolError("protocol_error")

            if event_type == "response.started":
                if started or event_count != 1:
                    raise NemoClawProtocolError("protocol_error")
                started = True
            elif event_type == "response.text.delta":
                sequence = event.get("sequence")
                text = event.get("text")
                if (
                    not started
                    or isinstance(sequence, bool)
                    or sequence != expected_sequence
                    or not isinstance(text, str)
                ):
                    raise NemoClawProtocolError("protocol_error")
                expected_sequence += 1
                try:
                    text_bytes += len(text.encode("utf-8"))
                except UnicodeEncodeError as exc:
                    raise NemoClawProtocolError("protocol_error") from exc
                if text_bytes > MAX_RESULT_ENVELOPE_BYTES:
                    raise NemoClawProtocolError("response_limit")
                try:
                    parser.push(text)
                except ResultEnvelopeLimitError as exc:
                    raise NemoClawProtocolError("response_limit") from exc
                except ResultEnvelopeProtocolError as exc:
                    raise NemoClawProtocolError("protocol_error") from exc
            elif event_type == "response.completed":
                if not started:
                    raise NemoClawProtocolError("protocol_error")
                terminal = event_type
            else:
                reason = event.get("reason")
                if reason not in _FAILURE_REASONS or (not started and event_count != 1):
                    raise NemoClawProtocolError("protocol_error")
                terminal = event_type
                failure = reason

        if terminal is None or turn_id is None or response_id is None:
            raise NemoClawProtocolError("truncated_stream")
        if failure is not None:
            raise NemoClawBackendFailure(failure)
        try:
            result = parser.finish()
        except ResultEnvelopeLimitError as exc:
            raise NemoClawProtocolError("response_limit") from exc
        except ResultEnvelopeProtocolError as exc:
            raise NemoClawProtocolError("protocol_error") from exc
        return CommittedTurnResult(
            backend_session_id=expected_session_id,
            turn_id=turn_id,
            response_id=response_id,
            display_text=result.display,
            speak_text=result.speech,
        )

    @staticmethod
    def _raise_rejected(status: int, response: http.client.HTTPResponse, body: bytes) -> None:
        code = "http_error"
        if _media_type(response) == "application/json":
            try:
                payload = _strict_object(body)
                candidate = payload.get("error")
                if set(payload) == {"error"} and isinstance(candidate, str) and _ERROR_CODE.fullmatch(candidate):
                    code = candidate
            except NemoClawProtocolError:
                pass
        raise NemoClawRequestRejected(code, status=status)

    def _delete_once(self, authority: _CleanupAuthority, *, timeout_seconds: float | None = None) -> bool:
        if authority.expired():
            return True
        try:
            cleanup_timeout = min(self._exchange_deadline_seconds, 5.0) if timeout_seconds is None else timeout_seconds
            cleanup_deadline = time.monotonic() + cleanup_timeout
            with self._connection_until(cleanup_deadline) as (connection, watch_response):
                response: http.client.HTTPResponse | None = None
                try:
                    connection.request(
                        "DELETE",
                        f"/v1/voice/sessions/{quote(authority.session_id, safe='')}",
                        headers=self._headers(authority.grant, accepts="application/json"),
                    )
                    response = connection.getresponse()
                    watch_response(response)
                    body = _read_bounded(response, _MAX_SMALL_BODY_BYTES)
                    return (
                        response.status == 204
                        and not body
                        and response.getheader("Content-Encoding", "") in {"", "identity"}
                    )
                finally:
                    if response is not None:
                        response.close()
        except Exception:
            return False

    def _delete_best_effort(
        self,
        authority: _CleanupAuthority,
        *,
        timeout_seconds: float | None = None,
    ) -> None:
        succeeded = self._delete_once(authority, timeout_seconds=timeout_seconds) or authority.expired()
        with self._cleanup_lock:
            if succeeded:
                if self._active_cleanup == authority:
                    self._active_cleanup = None
                if self._pending_cleanup == authority:
                    self._pending_cleanup = None
                if self._pending_cleanup is None:
                    self._cleanup_degraded.clear()
                return
            if self._closed.is_set():
                return
            if self._active_cleanup == authority:
                self._active_cleanup = None
            if self._pending_cleanup is None:
                self._pending_cleanup = authority
            self._cleanup_degraded.set()

    def _retry_pending_cleanup(self, *, timeout_seconds: float | None = None) -> bool:
        if not self._operation_gate.acquire(blocking=False):
            return False
        try:
            with self._cleanup_lock:
                pending = self._pending_cleanup
            if pending is None:
                self._cleanup_degraded.clear()
                return True
            if not self._delete_once(pending, timeout_seconds=timeout_seconds) and not pending.expired():
                return False
            with self._cleanup_lock:
                if self._pending_cleanup == pending:
                    self._pending_cleanup = None
                recovered = self._pending_cleanup is None
                if recovered:
                    self._cleanup_degraded.clear()
                return recovered
        finally:
            self._operation_gate.release()


__all__ = [
    "DeploymentAuthorization",
    "EndpointPolicy",
    "DEFAULT_RESULT_DISPLAY_BUDGET_BYTES",
    "NemoClawBackendFailure",
    "NemoClawCommittedTurnAdapter",
    "NemoClawCommittedTurnError",
    "NemoClawEndpointError",
    "NemoClawProtocolError",
    "NemoClawRequestRejected",
    "NemoClawRequestValidationError",
    "NemoClawTransportError",
]
