# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""Non-durable committed turns through OpenShell and a Fabric 0.5 runtime."""

from __future__ import annotations

import asyncio
import re
import threading
import uuid
from collections.abc import AsyncIterator, Callable

from voiceclaw.adapters.openshell_fabric.client import (
    OpenShellClientError,
    OpenShellFailureCode,
    OpenShellFailurePhase,
    SandboxExecutor,
)
from voiceclaw.adapters.openshell_fabric.protocol import (
    FABRIC_ADAPTER_KINDS,
    FabricEffects,
    FabricProtocolError,
    FabricRunStatus,
    FabricStatus,
    decode_response,
    encode_run_request,
    parse_json_pointer,
    parse_run_result,
    resolve_json_pointer,
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

FABRIC_AGENT_BINARY = "/opt/fabric/bin/fabric-agent-ctl"
DEFAULT_INVOCATION_TIMEOUT_SECONDS = 330
MAX_VOLATILE_SEQUENTIAL_TURNS = 4096
_IDENTITY = re.compile(r"^[^\x00\r\n]{1,512}$")
_VOICECLAW_RELAY_NAMESPACE = uuid.uuid5(
    uuid.NAMESPACE_URL,
    "https://github.com/NVIDIA-AI-Blueprints/nemotron-voice-agent/voiceclaw",
)


def validate_fabric_binding_identity(value: str, *, name: str, maximum_characters: int = 512) -> str:
    """Validate one operator-selected OpenShell/Fabric binding identity."""
    if not isinstance(value, str) or _IDENTITY.fullmatch(value) is None:
        raise ValueError(f"{name} is invalid")
    try:
        encoded = value.encode("utf-8")
    except UnicodeEncodeError as error:
        raise ValueError(f"{name} is invalid") from error
    if len(encoded) > maximum_characters or any(ord(character) < 0x20 or ord(character) == 0x7F for character in value):
        raise ValueError(f"{name} is invalid")
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
    """The invocation was rejected before backend execution began."""


class OpenShellFabricAmbiguous(OpenShellFabricError):
    """The invocation may have executed and its runtime is fenced."""


class OpenShellFabricProtocolError(OpenShellFabricError):
    """The configured binding or request violated the Fabric contract."""


class OpenShellFabricAdapter:
    """Invoke one NemoClaw-provisioned Fabric runtime without owning its lifecycle."""

    def __init__(
        self,
        *,
        executor: SandboxExecutor,
        workspace: str,
        sandbox: str,
        fabric_agent: str,
        runtime_id: str,
        adapter_id: str,
        harness: str,
        adapter_kind: str,
        result_pointer: str,
        label: str = "OpenShell / Fabric",
        invoke_timeout_seconds: int = DEFAULT_INVOCATION_TIMEOUT_SECONDS,
        result_display_budget_bytes: int = MAX_RESULT_DISPLAY_BYTES,
        result_speech_budget_bytes: int = DEFAULT_RESULT_SPEECH_BUDGET_BYTES,
        model_contracts: ModelContractCatalog | None = None,
    ) -> None:
        """Bind to one already-started runtime owned by NemoClaw."""
        if not isinstance(executor, SandboxExecutor):
            raise TypeError("executor must implement SandboxExecutor")
        for name, value in (
            ("workspace", workspace),
            ("sandbox", sandbox),
            ("fabric_agent", fabric_agent),
            ("runtime_id", runtime_id),
            ("adapter_id", adapter_id),
            ("harness", harness),
            ("adapter_kind", adapter_kind),
        ):
            validate_fabric_binding_identity(value, name=name)
        if adapter_kind not in FABRIC_ADAPTER_KINDS:
            raise ValueError("adapter_kind is not supported by Fabric 0.5")
        validate_fabric_binding_identity(label, name="label", maximum_characters=128)
        if isinstance(invoke_timeout_seconds, bool) or not isinstance(invoke_timeout_seconds, int):
            raise ValueError("invoke_timeout_seconds must be an integer")
        if not 1 <= invoke_timeout_seconds <= 3600:
            raise ValueError("invoke_timeout_seconds must be from 1 through 3600")
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
        parse_json_pointer(result_pointer)
        self._executor = executor
        self._workspace = workspace
        self._sandbox = sandbox
        self._fabric_agent = fabric_agent
        self._runtime_id = runtime_id
        self._adapter_id = adapter_id
        self._harness = harness
        self._adapter_kind = adapter_kind
        self._result_pointer = result_pointer
        self._label = label
        self._invoke_timeout_seconds = invoke_timeout_seconds
        self._result_display_budget_bytes = result_display_budget_bytes
        self._result_speech_budget_bytes = result_speech_budget_bytes
        self._model_contracts = model_contracts or load_model_contract_catalog()
        self._invoke_gate = threading.Lock()
        self._state_lock = threading.Lock()
        self._lifecycle_lock = threading.Lock()
        self._invocation_state = ResponseOnlyTargetAvailability.AVAILABLE
        self._bound_conversation_id: str | None = None
        self._request_ids: set[str] = set()
        self._invocation_ids: set[str] = set()
        self._maximum_sequential_turns = MAX_VOLATILE_SEQUENTIAL_TURNS
        self._closed = False
        self._client_closed = False

    async def inspect(self) -> CommittedTurnBackend:
        """Return configured capability evidence without inventing a health probe."""
        return self._backend_descriptor(availability=self._target_availability())

    def _backend_descriptor(self, *, availability: ResponseOnlyTargetAvailability) -> CommittedTurnBackend:
        operations = (
            frozenset({BackendOperation.SUBMIT})
            if availability is not ResponseOnlyTargetAvailability.CONSUMED
            else frozenset()
        )
        return CommittedTurnBackend(
            label=self._label,
            target_ref=f"{self._workspace}/{self._sandbox}/{self._fabric_agent}/{self._runtime_id}",
            mode="response_only",
            capabilities=BackendCapabilities(
                backend_kind="response_only",
                target_label=self._label,
                revision="openshell-fabric-0.5",
                operations=operations,
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
        with self._state_lock:
            return self._invocation_state

    def _claim_target(self, *, conversation_id: str, request_id: str) -> None:
        with self._state_lock:
            if self._invocation_state is not ResponseOnlyTargetAvailability.AVAILABLE:
                raise OpenShellFabricRejected("target_context_consumed")
            if self._bound_conversation_id not in (None, conversation_id):
                raise OpenShellFabricRejected("target_context_mismatch")
            if request_id in self._request_ids:
                raise OpenShellFabricRejected("duplicate_request")
            if len(self._request_ids) >= self._maximum_sequential_turns:
                self._invocation_state = ResponseOnlyTargetAvailability.CONSUMED
                raise OpenShellFabricRejected("target_context_consumed")
            self._invocation_state = ResponseOnlyTargetAvailability.IN_FLIGHT

    def _record_dispatch(self, *, conversation_id: str, request_id: str) -> None:
        with self._state_lock:
            if (
                self._invocation_state is not ResponseOnlyTargetAvailability.IN_FLIGHT
                or self._bound_conversation_id not in (None, conversation_id)
                or request_id in self._request_ids
                or len(self._request_ids) >= self._maximum_sequential_turns
            ):
                self._invocation_state = ResponseOnlyTargetAvailability.CONSUMED
                raise OpenShellFabricAmbiguous("invocation_outcome_unknown")
            self._bound_conversation_id = conversation_id
            self._request_ids.add(request_id)

    def _release_target(self) -> None:
        with self._state_lock:
            if self._invocation_state is ResponseOnlyTargetAvailability.IN_FLIGHT:
                self._invocation_state = (
                    ResponseOnlyTargetAvailability.CONSUMED
                    if len(self._request_ids) >= self._maximum_sequential_turns
                    else ResponseOnlyTargetAvailability.AVAILABLE
                )

    def _consume_target(self) -> None:
        with self._state_lock:
            self._invocation_state = ResponseOnlyTargetAvailability.CONSUMED

    async def commit_turn(self, request: CommittedTurnRequest) -> CommittedTurnResult:
        """Execute one ordered invocation against the provisioned runtime."""
        if not isinstance(request, CommittedTurnRequest):
            raise OpenShellFabricError("invalid_request")
        dispatched = threading.Event()
        cancelled = threading.Event()
        dispatch_gate = threading.Lock()
        try:
            return await _run_daemon(
                lambda: self._commit_sync(
                    request,
                    dispatched=dispatched,
                    cancelled=cancelled,
                    dispatch_gate=dispatch_gate,
                ),
                name="voiceclaw-fabric-invoke",
            )
        except asyncio.CancelledError:
            with dispatch_gate:
                cancelled.set()
                if dispatched.is_set():
                    self._consume_target()
            raise

    async def stream_turn(self, request: CommittedTurnRequest) -> AsyncIterator[CommittedTurnEvent]:
        """Expose Fabric's terminal result through the committed-turn stream port."""
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
        """Release only the OpenShell transport; NemoClaw retains runtime ownership."""
        self.close()

    def close(self) -> None:
        """Reject new calls and close the OpenShell client once invokes finish."""
        with self._lifecycle_lock:
            self._closed = True
        self._close_executor_if_idle()

    def _commit_sync(
        self,
        request: CommittedTurnRequest,
        *,
        dispatched: threading.Event,
        cancelled: threading.Event,
        dispatch_gate: threading.Lock,
    ) -> CommittedTurnResult:
        self._acquire_operation()
        claimed = False
        invocation_dispatched = False
        terminal_known = False
        try:
            if cancelled.is_set():
                raise OpenShellFabricRejected("invocation_cancelled")
            self._validate_request(request)
            self._claim_target(
                conversation_id=request.runtime_conversation_id,
                request_id=request.commit_id,
            )
            claimed = True
            prompt = self._build_prompt(request.text)
            try:
                payload = encode_run_request(
                    input_text=prompt,
                    request_id=request.commit_id,
                    relay_session_root=str(uuid.uuid5(_VOICECLAW_RELAY_NAMESPACE, request.runtime_conversation_id)),
                    context={
                        "voiceclaw_conversation_id": request.runtime_conversation_id,
                        "voiceclaw_commit_id": request.commit_id,
                    },
                )
            except FabricProtocolError as error:
                raise OpenShellFabricProtocolError("invalid_fabric_request") from error
            try:
                with dispatch_gate:
                    if cancelled.is_set():
                        raise OpenShellFabricRejected("invocation_cancelled")
                    if not request.dispatch_marker.mark():
                        raise OpenShellFabricRejected("invocation_cancelled")
                    self._record_dispatch(
                        conversation_id=request.runtime_conversation_id,
                        request_id=request.commit_id,
                    )
                    invocation_dispatched = True
                    dispatched.set()
                execution = self._execute(
                    (
                        FABRIC_AGENT_BINARY,
                        "invoke",
                        "--agent",
                        self._fabric_agent,
                        "--runtime-id",
                        self._runtime_id,
                        "--request",
                        "-",
                    ),
                    stdin=payload,
                    timeout_seconds=self._invoke_timeout_seconds,
                )
            except OpenShellFabricRejected:
                terminal_known = True
                self._release_target()
                raise
            except OpenShellFabricAmbiguous:
                self._consume_target()
                raise
            try:
                response = decode_response(operation="invoke", exit_code=execution.exit_code, stdout=execution.stdout)
            except FabricProtocolError as error:
                self._consume_target()
                raise OpenShellFabricAmbiguous("invocation_outcome_unknown") from error
            if response.status is not FabricStatus.SUCCEEDED:
                assert response.error is not None
                if response.error.effects is FabricEffects.NONE:
                    terminal_known = True
                    self._release_target()
                    if response.error.code == "wrong_agent":
                        raise OpenShellFabricRejected("target_missing")
                    raise OpenShellFabricRejected("invocation_rejected")
                self._consume_target()
                raise OpenShellFabricAmbiguous("invocation_outcome_unknown")
            try:
                result = parse_run_result(
                    response,
                    agent_name=self._fabric_agent,
                    runtime_id=self._runtime_id,
                    adapter_id=self._adapter_id,
                    harness=self._harness,
                    adapter_kind=self._adapter_kind,
                    request_id=request.commit_id,
                )
                with self._state_lock:
                    if result.invocation_id in self._invocation_ids:
                        raise FabricProtocolError("Fabric invocation identity was reused")
                    self._invocation_ids.add(result.invocation_id)
                if result.status is FabricRunStatus.FAILED:
                    terminal_known = True
                    self._release_target()
                    raise OpenShellFabricError("invocation_failed")
                if result.status is FabricRunStatus.CANCELLED:
                    terminal_known = True
                    self._release_target()
                    raise OpenShellFabricError("invocation_cancelled")
                text = resolve_json_pointer(result.value, self._result_pointer)
                if not isinstance(text, str):
                    raise FabricProtocolError("selected Fabric result is not text")
                parser = ResultEnvelopeStreamParser(
                    lambda _delta: None,
                    maximum_display_bytes=self._result_display_budget_bytes,
                    maximum_speech_bytes=self._result_speech_budget_bytes,
                )
                parser.push(text)
                envelope = parser.finish()
            except OpenShellFabricError:
                raise
            except (FabricProtocolError, ResultEnvelopeProtocolError, TypeError, ValueError) as error:
                self._consume_target()
                raise OpenShellFabricAmbiguous("invocation_result_unusable") from error
            terminal_known = True
            self._release_target()
            return CommittedTurnResult(
                backend_session_id=self._runtime_id,
                turn_id=result.request_id,
                response_id=result.invocation_id,
                display_text=envelope.display,
                speak_text=envelope.speech,
            )
        finally:
            if claimed and not invocation_dispatched:
                self._release_target()
            elif claimed and invocation_dispatched and not terminal_known:
                self._consume_target()
            self._release_operation()

    @staticmethod
    def _validate_request(request: CommittedTurnRequest) -> None:
        try:
            validate_fabric_binding_identity(request.runtime_conversation_id, name="runtime_conversation_id")
            validate_fabric_binding_identity(request.commit_id, name="commit_id")
        except ValueError as error:
            raise OpenShellFabricProtocolError("invalid_fabric_correlation") from error
        try:
            encoded = request.text.encode("utf-8")
        except UnicodeEncodeError as error:
            raise OpenShellFabricError("invalid_text") from error
        if len(encoded) > MAX_COMMITTED_TURN_GOAL_BYTES:
            raise OpenShellFabricError("goal_too_large")

    def _build_prompt(self, text: str) -> str:
        try:
            return build_result_envelope_prompt(
                text,
                display_budget_bytes=self._result_display_budget_bytes,
                speech_budget_bytes=self._result_speech_budget_bytes,
                contracts=self._model_contracts,
            )
        except ResultEnvelopeProtocolError as error:
            raise OpenShellFabricProtocolError("invalid_result_contract") from error

    def _execute(self, command: tuple[str, ...], *, stdin: bytes, timeout_seconds: int):
        try:
            return self._executor.execute(command, stdin=stdin, timeout_seconds=timeout_seconds)
        except OpenShellClientError as error:
            if error.phase is OpenShellFailurePhase.BEFORE_ADMISSION and error.code is OpenShellFailureCode.AUTH_DENIED:
                raise OpenShellFabricRejected("auth_denied") from error
            if (
                error.phase is OpenShellFailurePhase.BEFORE_ADMISSION
                and error.code is OpenShellFailureCode.TARGET_MISSING
            ):
                raise OpenShellFabricRejected("target_missing") from error
            raise OpenShellFabricAmbiguous("invocation_outcome_unknown") from error

    def _acquire_operation(self) -> None:
        with self._lifecycle_lock:
            if self._closed:
                raise OpenShellFabricError("adapter_closed")
        self._invoke_gate.acquire()
        closed = False
        with self._lifecycle_lock:
            if self._closed:
                self._invoke_gate.release()
                closed = True
        if closed:
            self._close_executor_if_idle()
            raise OpenShellFabricError("adapter_closed")

    def _release_operation(self) -> None:
        self._invoke_gate.release()
        self._close_executor_if_idle()

    def _close_executor_if_idle(self) -> None:
        with self._lifecycle_lock:
            if not self._closed or self._client_closed or self._invoke_gate.locked():
                return
            self._client_closed = True
        try:
            self._executor.close()
        except OpenShellClientError:
            return


__all__ = [
    "DEFAULT_INVOCATION_TIMEOUT_SECONDS",
    "FABRIC_AGENT_BINARY",
    "MAX_VOLATILE_SEQUENTIAL_TURNS",
    "OpenShellFabricAdapter",
    "OpenShellFabricAmbiguous",
    "OpenShellFabricError",
    "OpenShellFabricProtocolError",
    "OpenShellFabricRejected",
    "validate_fabric_binding_identity",
]
