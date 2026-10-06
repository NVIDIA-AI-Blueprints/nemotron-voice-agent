# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""Response-only committed turns through OpenShell and the Fabric bridge."""

from __future__ import annotations

import asyncio
import math
import re
import threading
import time
from collections.abc import AsyncIterator, Callable

from voiceclaw.adapters.openshell_fabric.client import (
    OpenShellClientError,
    OpenShellFailureCode,
    SandboxExecutor,
)
from voiceclaw.adapters.openshell_fabric.protocol import (
    FabricEffects,
    FabricFirstTurnQualificationError,
    FabricProtocolError,
    FabricStatus,
    adapter_codec,
    decode_response,
    parse_json_pointer,
    resolve_json_pointer,
    successful_fabric_result,
)
from voiceclaw.adapters.result_envelope import (
    DEFAULT_RESULT_SPEECH_BUDGET_BYTES,
    MAX_RESULT_DISPLAY_BYTES,
    MAX_RESULT_SPEECH_BYTES,
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
from voiceclaw.domain.response_only import ResponseOnlyContextContinuity, ResponseOnlyTargetAvailability
from voiceclaw.model_contracts import ModelContractCatalog, load_model_contract_catalog
from voiceclaw.ports.readiness import SelectedAgentReadinessCode, SelectedAgentReadinessError
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

FABRIC_AGENT_BINARY = "/usr/local/bin/fabric-agent"
DEFAULT_INVOCATION_TIMEOUT_SECONDS = 330
DEFAULT_CHECK_TIMEOUT_SECONDS = 30
DEFAULT_READINESS_CACHE_TTL_SECONDS = 30.0
_IDENTITY = re.compile(r"^[^\x00\r\n]{1,512}$")


def validate_fabric_binding_identity(value: str, *, name: str, maximum_characters: int = 512) -> str:
    """Validate one operator-selected Fabric/OpenShell identity."""
    if not isinstance(value, str) or _IDENTITY.fullmatch(value) is None or len(value) > maximum_characters:
        raise ValueError(f"{name} is invalid")
    try:
        value.encode("utf-8")
    except UnicodeEncodeError as error:
        raise ValueError(f"{name} is invalid") from error
    return value


async def _run_daemon[T](function: Callable[[], T], *, name: str) -> T:
    completed = threading.Event()
    result: list[T] = []
    failures: list[BaseException] = []

    def run() -> None:
        try:
            result.append(function())
        except BaseException as error:
            failures.append(error)
        finally:
            completed.set()

    threading.Thread(target=run, name=name, daemon=True).start()
    while not completed.is_set():
        await asyncio.sleep(0.01)
    if failures:
        raise failures[0]
    return result[0]


class OpenShellFabricError(CommittedTurnError):
    """Base class for safe OpenShell/Fabric committed-turn failures."""


class OpenShellFabricRejected(OpenShellFabricError):
    """The bridge confirmed that execution did not begin."""


class OpenShellFabricAmbiguous(OpenShellFabricError):
    """Invocation may have executed and must not be retried automatically."""


class OpenShellFabricProtocolError(OpenShellFabricError):
    """A bridge or selected adapter response violated its public contract."""


class OpenShellFabricAdapter:
    """Invoke one deployed Fabric agent without making durability claims."""

    def __init__(
        self,
        *,
        executor: SandboxExecutor,
        workspace: str,
        sandbox: str,
        fabric_agent: str,
        adapter_id: str,
        native_agent: str | None = None,
        label: str = "OpenShell / Fabric",
        invoke_timeout_seconds: int = DEFAULT_INVOCATION_TIMEOUT_SECONDS,
        check_timeout_seconds: int = DEFAULT_CHECK_TIMEOUT_SECONDS,
        readiness_cache_ttl_seconds: float = DEFAULT_READINESS_CACHE_TTL_SECONDS,
        result_pointer: str | None = None,
        result_display_budget_bytes: int = MAX_RESULT_DISPLAY_BYTES,
        result_speech_budget_bytes: int = DEFAULT_RESULT_SPEECH_BUDGET_BYTES,
        model_contracts: ModelContractCatalog | None = None,
    ) -> None:
        """Bind one service identity to one deployed agent target."""
        if not isinstance(executor, SandboxExecutor):
            raise TypeError("executor must implement SandboxExecutor")
        for name, value in (
            ("workspace", workspace),
            ("sandbox", sandbox),
            ("fabric_agent", fabric_agent),
            ("adapter_id", adapter_id),
        ):
            validate_fabric_binding_identity(value, name=name)
        validate_fabric_binding_identity(label, name="label", maximum_characters=128)
        try:
            codec = adapter_codec(adapter_id)
        except FabricProtocolError as error:
            raise ValueError("adapter_id has no installed codec") from error
        selected_native_agent = fabric_agent if native_agent is None else native_agent
        validate_fabric_binding_identity(selected_native_agent, name="native_agent")
        for name, value in (
            ("invoke_timeout_seconds", invoke_timeout_seconds),
            ("check_timeout_seconds", check_timeout_seconds),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 3600:
                raise ValueError(f"{name} must be an integer from 1 through 3600")
        if (
            isinstance(readiness_cache_ttl_seconds, bool)
            or not isinstance(readiness_cache_ttl_seconds, (int, float))
            or not math.isfinite(readiness_cache_ttl_seconds)
            or not 0 <= readiness_cache_ttl_seconds <= 300
        ):
            raise ValueError("readiness_cache_ttl_seconds must be in [0, 300]")
        if (
            isinstance(result_display_budget_bytes, bool)
            or not isinstance(result_display_budget_bytes, int)
            or not 1 <= result_display_budget_bytes <= MAX_RESULT_DISPLAY_BYTES
        ):
            raise ValueError("result_display_budget_bytes is invalid")
        if (
            isinstance(result_speech_budget_bytes, bool)
            or not isinstance(result_speech_budget_bytes, int)
            or not 1 <= result_speech_budget_bytes <= MAX_RESULT_SPEECH_BYTES
        ):
            raise ValueError("result_speech_budget_bytes is invalid")
        if result_pointer is not None:
            parse_json_pointer(result_pointer)
        self._executor = executor
        self._workspace = workspace
        self._sandbox = sandbox
        self._fabric_agent = fabric_agent
        self._codec = codec
        self._native_agent = selected_native_agent
        self._label = label
        self._invoke_timeout_seconds = invoke_timeout_seconds
        self._check_timeout_seconds = check_timeout_seconds
        self._readiness_cache_ttl_seconds = float(readiness_cache_ttl_seconds)
        self._result_pointer = result_pointer
        self._result_display_budget_bytes = result_display_budget_bytes
        self._result_speech_budget_bytes = result_speech_budget_bytes
        self._model_contracts = model_contracts or load_model_contract_catalog()
        self._invoke_gate = threading.Lock()
        self._invocation_state_lock = threading.Lock()
        self._invocation_state = ResponseOnlyTargetAvailability.AVAILABLE
        self._lifecycle_lock = threading.Lock()
        self._readiness_condition = threading.Condition(self._lifecycle_lock)
        self._readiness_inflight = False
        self._readiness_generation = 0
        self._readiness_failure: str | None = None
        self._closed = False
        self._client_closed = False
        self._binding_verified = False
        self._binding_verified_at: float | None = None
        self._binding_health_supported = False

    async def inspect(self) -> CommittedTurnBackend:
        """Check the runtime binding without generating an agent response."""
        availability = self._target_availability()
        if availability is ResponseOnlyTargetAvailability.AVAILABLE:
            await _run_daemon(self._check_sync, name="voiceclaw-fabric-inspect")
            availability = self._target_availability()
        return self._backend_descriptor(availability=availability)

    def _backend_descriptor(self, *, availability: ResponseOnlyTargetAvailability) -> CommittedTurnBackend:
        """Project the verified binding and its current one-shot availability."""
        return CommittedTurnBackend(
            label=self._label,
            target_ref=f"{self._workspace}/{self._sandbox}/{self._fabric_agent}",
            mode="response_only",
            capabilities=BackendCapabilities(
                backend_kind="response_only",
                target_label=self._label,
                revision=f"openshell-fabric-{self._codec.revision}",
                operations=(
                    frozenset({BackendOperation.SUBMIT})
                    if availability is ResponseOnlyTargetAvailability.AVAILABLE
                    else frozenset()
                ),
                durability=Durability.NONE,
                event_delivery=EventDelivery.RESPONSE_ONLY,
                agent_targets=(),
                sessionful=False,
                supports_parallel_work=False,
                max_parallel_work=1,
            ),
            capability_source=CapabilitySource.OPERATOR_CONFIGURED,
            capability_source_id="openshell_fabric",
            context_continuity=ResponseOnlyContextContinuity.UNQUALIFIED,
            target_availability=availability,
        )

    def _target_availability(self) -> ResponseOnlyTargetAvailability:
        """Read the one-shot target state without waiting for invocation."""
        with self._invocation_state_lock:
            return self._invocation_state

    def _claim_target(self) -> None:
        """Atomically claim the target before any potentially blocking work."""
        with self._invocation_state_lock:
            if self._invocation_state is not ResponseOnlyTargetAvailability.AVAILABLE:
                raise OpenShellFabricRejected("target_context_consumed")
            self._invocation_state = ResponseOnlyTargetAvailability.IN_FLIGHT

    def _release_target(self) -> None:
        """Return a conclusively unused target to the available state."""
        with self._invocation_state_lock:
            if self._invocation_state is ResponseOnlyTargetAvailability.IN_FLIGHT:
                self._invocation_state = ResponseOnlyTargetAvailability.AVAILABLE

    def _consume_target(self) -> None:
        """Fence a target whose invocation ran or may have run."""
        with self._invocation_state_lock:
            self._invocation_state = ResponseOnlyTargetAvailability.CONSUMED

    def _consume_target_if_in_flight(self) -> None:
        """Fence an invocation that escaped through an unexpected failure."""
        with self._invocation_state_lock:
            if self._invocation_state is ResponseOnlyTargetAvailability.IN_FLIGHT:
                self._invocation_state = ResponseOnlyTargetAvailability.CONSUMED

    async def check_selected_agent(self) -> None:
        """Verify only the selected bridge/runtime binding, not inference health."""
        try:
            await _run_daemon(self._check_sync, name="voiceclaw-fabric-readiness")
        except OpenShellFabricError as error:
            mapping = {
                "auth_denied": SelectedAgentReadinessCode.ACCESS_DENIED,
                "bridge_protocol_error": SelectedAgentReadinessCode.PROTOCOL_ERROR,
                "openshell_unavailable": SelectedAgentReadinessCode.ENDPOINT_UNAVAILABLE,
                "target_missing": SelectedAgentReadinessCode.WRONG_AGENT,
                "timeout": SelectedAgentReadinessCode.ENDPOINT_UNAVAILABLE,
            }
            raise SelectedAgentReadinessError(
                mapping.get(error.code, SelectedAgentReadinessCode.UNAVAILABLE)
            ) from error

    async def commit_turn(self, request: CommittedTurnRequest) -> CommittedTurnResult:
        """Execute one invocation exactly once and return one terminal result."""
        if not isinstance(request, CommittedTurnRequest):
            raise OpenShellFabricError("invalid_request")
        return await _run_daemon(
            lambda: self._commit_sync(request),
            name="voiceclaw-fabric-invoke",
        )

    async def stream_turn(self, request: CommittedTurnRequest) -> AsyncIterator[CommittedTurnEvent]:
        """Expose the terminal-only bridge through the committed-turn stream port."""
        result = await self.commit_turn(request)
        yield CommittedTurnDisplayDelta(
            backend_session_id=result.backend_session_id,
            turn_id=result.turn_id,
            response_id=result.response_id,
            sequence=0,
            delta=result.display_text,
        )
        yield CommittedTurnCompleted(result=result)

    async def shutdown(self) -> None:
        """Reject new calls and release the OpenShell transport when idle."""
        self.close()

    def close(self) -> None:
        """Reject new calls and close the transport once active calls finish."""
        with self._readiness_condition:
            self._closed = True
            self._readiness_condition.notify_all()
        self._close_executor_if_idle()

    def _check_sync(self) -> bool:
        with self._readiness_condition:
            if self._closed:
                raise OpenShellFabricError("adapter_closed")
            cached = self._cached_readiness_locked()
            if cached is not None:
                return cached
            observed_generation = self._readiness_generation
            if self._readiness_inflight:
                while (
                    self._readiness_inflight and self._readiness_generation == observed_generation and not self._closed
                ):
                    self._readiness_condition.wait()
                if self._closed:
                    raise OpenShellFabricError("adapter_closed")
                if self._readiness_generation != observed_generation and self._readiness_failure is not None:
                    raise OpenShellFabricError(self._readiness_failure)
                cached = self._cached_readiness_locked()
                if cached is not None:
                    return cached
                if self._readiness_generation != observed_generation and self._binding_verified:
                    return self._binding_health_supported
            self._readiness_inflight = True
        try:
            execution = self._execute(
                (
                    FABRIC_AGENT_BINARY,
                    "check",
                    "--agent",
                    self._fabric_agent,
                    "--live",
                ),
                stdin=None,
                timeout_seconds=self._check_timeout_seconds,
            )
            try:
                response = decode_response(
                    operation="check",
                    exit_code=execution.exit_code,
                    stdout=execution.stdout,
                )
            except FabricProtocolError as error:
                raise OpenShellFabricProtocolError("bridge_protocol_error") from error
            if response.status is FabricStatus.SUCCEEDED and self._running_snapshot(response.result):
                return self._publish_readiness_success(health_supported=True)
            if (
                response.status is FabricStatus.UNSUPPORTED
                and response.error is not None
                and response.error.code == "fabric_health_unsupported"
                and self._running_snapshot(response.result, unsupported=True)
            ):
                return self._publish_readiness_success(health_supported=False)
            self._raise_bridge_failure(response.error.code if response.error is not None else "")
            raise OpenShellFabricError("target_unavailable")
        except OpenShellFabricError as error:
            self._publish_readiness_failure(error.code)
            raise
        except BaseException:
            self._publish_readiness_failure("openshell_unavailable")
            raise

    def _commit_sync(self, request: CommittedTurnRequest) -> CommittedTurnResult:
        if not self._acquire_operation(self._invoke_gate):
            raise OpenShellFabricError("backend_busy")
        dispatched = False
        try:
            self._claim_target()
            try:
                request_bytes = request.text.encode("utf-8")
            except UnicodeEncodeError as error:
                raise OpenShellFabricError("invalid_text") from error
            if len(request_bytes) > MAX_COMMITTED_TURN_GOAL_BYTES:
                raise OpenShellFabricError("goal_too_large")
            try:
                prompt = build_result_envelope_prompt(
                    request.text,
                    display_budget_bytes=self._result_display_budget_bytes,
                    speech_budget_bytes=self._result_speech_budget_bytes,
                    contracts=self._model_contracts,
                )
                payload = self._codec.encode_input(
                    native_agent=self._native_agent,
                    prompt=prompt,
                )
            except FabricProtocolError as error:
                raise OpenShellFabricProtocolError("incompatible_result") from error
            except ResultEnvelopeProtocolError as error:
                raise OpenShellFabricProtocolError("invalid_result_contract") from error
            try:
                dispatched = True
                execution = self._execute(
                    (
                        FABRIC_AGENT_BINARY,
                        "invoke",
                        "--agent",
                        self._fabric_agent,
                        "--input",
                        "-",
                    ),
                    stdin=payload,
                    timeout_seconds=self._invoke_timeout_seconds,
                    invocation=True,
                )
            except OpenShellFabricRejected:
                self._release_target()
                raise
            except OpenShellFabricAmbiguous:
                self._consume_target()
                raise
            try:
                response = decode_response(
                    operation="invoke",
                    exit_code=execution.exit_code,
                    stdout=execution.stdout,
                )
            except FabricProtocolError as error:
                self._consume_target()
                raise OpenShellFabricAmbiguous("invocation_outcome_unknown") from error
            if response.status is not FabricStatus.SUCCEEDED:
                assert response.error is not None
                if response.error.effects is FabricEffects.NONE:
                    self._release_target()
                    if response.error.code == "wrong_agent":
                        raise OpenShellFabricRejected("target_missing")
                    raise OpenShellFabricRejected("invocation_rejected")
                self._consume_target()
                raise OpenShellFabricAmbiguous("invocation_outcome_unknown")
            try:
                runtime_id, fabric_result = successful_fabric_result(response)
                turn_id, response_id = self._validate_correlation(runtime_id, fabric_result)
                backend_session_id = self._codec.validate_first_turn_result(
                    fabric_result=fabric_result,
                    native_agent=self._native_agent,
                    runtime_id=runtime_id,
                    prompt=prompt,
                )
                pointer = self._result_pointer or self._codec.default_result_pointer
                text = resolve_json_pointer(fabric_result, pointer)
                if not isinstance(text, str):
                    raise FabricProtocolError("selected result is not text")
                parser = ResultEnvelopeStreamParser(
                    lambda _delta: None,
                    maximum_display_bytes=self._result_display_budget_bytes,
                    maximum_speech_bytes=self._result_speech_budget_bytes,
                )
                parser.push(text)
                envelope = parser.finish()
            except FabricFirstTurnQualificationError as error:
                self._consume_target()
                raise OpenShellFabricAmbiguous("target_context_reused") from error
            except (FabricProtocolError, ResultEnvelopeProtocolError, TypeError, ValueError) as error:
                self._consume_target()
                raise OpenShellFabricAmbiguous("invocation_result_unusable") from error
            self._consume_target()
            return CommittedTurnResult(
                backend_session_id=backend_session_id,
                turn_id=turn_id,
                response_id=response_id,
                display_text=envelope.display,
                speak_text=envelope.speech,
            )
        finally:
            if not dispatched:
                self._release_target()
            else:
                self._consume_target_if_in_flight()
            self._release_operation(self._invoke_gate)

    def _validate_correlation(self, runtime_id: str, fabric_result: dict[str, object]) -> tuple[str, str]:
        required = (
            "agent_name",
            "harness",
            "adapter_kind",
            "runtime_id",
            "invocation_id",
            "request_id",
        )
        identities = {name: fabric_result.get(name) for name in required}
        for value in identities.values():
            if not isinstance(value, str) or _IDENTITY.fullmatch(value) is None:
                raise FabricProtocolError("Fabric result identity is invalid")
        adapter_id = fabric_result.get("adapter_id")
        if (
            not isinstance(adapter_id, str)
            or _IDENTITY.fullmatch(adapter_id) is None
            or adapter_id != self._codec.adapter_id
        ):
            raise FabricProtocolError("Fabric adapter identity mismatch")
        if (
            identities["runtime_id"] != runtime_id
            or identities["agent_name"] != self._fabric_agent
            or identities["harness"] != self._codec.expected_harness
            or identities["adapter_kind"] != self._codec.expected_adapter_kind
        ):
            raise FabricProtocolError("Fabric result identity mismatch")
        return identities["request_id"], identities["invocation_id"]

    def _execute(
        self,
        command: tuple[str, ...],
        *,
        stdin: bytes | None,
        timeout_seconds: int,
        invocation: bool = False,
    ):
        try:
            return self._executor.execute(command, stdin=stdin, timeout_seconds=timeout_seconds)
        except OpenShellClientError as error:
            if error.code is OpenShellFailureCode.AUTH_DENIED:
                failure = OpenShellFabricRejected if invocation else OpenShellFabricError
                raise failure("auth_denied") from error
            if error.code is OpenShellFailureCode.TARGET_MISSING:
                failure = OpenShellFabricRejected if invocation else OpenShellFabricError
                raise failure("target_missing") from error
            if error.code is OpenShellFailureCode.TIMEOUT:
                code = "invocation_outcome_unknown" if invocation else "timeout"
                failure = OpenShellFabricAmbiguous if invocation else OpenShellFabricError
                raise failure(code) from error
            code = "invocation_outcome_unknown" if invocation else "openshell_unavailable"
            failure = OpenShellFabricAmbiguous if invocation else OpenShellFabricError
            raise failure(code) from error

    @staticmethod
    def _raise_bridge_failure(code: str) -> None:
        if code == "wrong_agent":
            raise OpenShellFabricError("target_missing")
        if code in {"runtime_unavailable", "host_stopping"}:
            raise OpenShellFabricError("target_unavailable")
        raise OpenShellFabricError("target_check_failed")

    def _running_snapshot(self, result: dict[str, object] | None, *, unsupported: bool = False) -> bool:
        if not isinstance(result, dict):
            return False
        required = {"runtime_id", "runtime_state", "generation", "applied_config", "health"}
        if not required.issubset(result):
            return False
        runtime_id = result["runtime_id"]
        generation = result["generation"]
        applied_config = result["applied_config"]
        if (
            result["runtime_state"] != "running"
            or not isinstance(runtime_id, str)
            or _IDENTITY.fullmatch(runtime_id) is None
            or not isinstance(generation, str)
            or _IDENTITY.fullmatch(generation) is None
            or not isinstance(applied_config, dict)
        ):
            return False
        metadata = applied_config.get("metadata")
        harness = applied_config.get("harness")
        if (
            not isinstance(metadata, dict)
            or metadata.get("name") != self._fabric_agent
            or not isinstance(harness, dict)
            or harness.get("adapter_id") != self._codec.adapter_id
            or not self._native_binding_matches(harness)
        ):
            return False
        health = result["health"]
        return health is None if unsupported else isinstance(health, dict)

    def _native_binding_matches(self, harness: dict[str, object]) -> bool:
        if self._codec.adapter_id != "nvidia.fabric.openclaw":
            return True
        settings = harness.get("settings", {})
        if not isinstance(settings, dict):
            return False
        return settings.get("agent_name", "main") == self._native_agent

    def _cached_readiness_locked(self) -> bool | None:
        if not self._binding_verified:
            return None
        if self._invoke_gate.locked():
            return self._binding_health_supported
        verified_at = self._binding_verified_at
        if (
            verified_at is not None
            and self._readiness_cache_ttl_seconds > 0
            and time.monotonic() - verified_at <= self._readiness_cache_ttl_seconds
        ):
            return self._binding_health_supported
        return None

    def _publish_readiness_success(self, *, health_supported: bool) -> bool:
        with self._readiness_condition:
            self._binding_verified = True
            self._binding_verified_at = time.monotonic()
            self._binding_health_supported = health_supported
            self._readiness_failure = None
            self._readiness_generation += 1
            self._readiness_inflight = False
            self._readiness_condition.notify_all()
        self._close_executor_if_idle()
        return health_supported

    def _publish_readiness_failure(self, code: str) -> None:
        with self._readiness_condition:
            self._binding_verified = False
            self._binding_verified_at = None
            self._binding_health_supported = False
            self._readiness_failure = code
            self._readiness_generation += 1
            self._readiness_inflight = False
            self._readiness_condition.notify_all()
        self._close_executor_if_idle()

    def _acquire_operation(self, gate: threading.Lock) -> bool:
        with self._lifecycle_lock:
            if self._closed:
                raise OpenShellFabricError("adapter_closed")
            return gate.acquire(blocking=False)

    def _release_operation(self, gate: threading.Lock) -> None:
        gate.release()
        self._close_executor_if_idle()

    def _close_executor_if_idle(self) -> None:
        with self._lifecycle_lock:
            if not self._closed or self._client_closed or self._invoke_gate.locked() or self._readiness_inflight:
                return
            self._client_closed = True
        try:
            self._executor.close()
        except OpenShellClientError:
            return


__all__ = [
    "DEFAULT_CHECK_TIMEOUT_SECONDS",
    "DEFAULT_INVOCATION_TIMEOUT_SECONDS",
    "DEFAULT_READINESS_CACHE_TTL_SECONDS",
    "FABRIC_AGENT_BINARY",
    "OpenShellFabricAdapter",
    "OpenShellFabricAmbiguous",
    "OpenShellFabricError",
    "OpenShellFabricProtocolError",
    "OpenShellFabricRejected",
    "validate_fabric_binding_identity",
]
