# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

# ruff: noqa: D100, D101, D102, D103

from __future__ import annotations

import asyncio
import json
import threading
import uuid
from collections.abc import Sequence
from dataclasses import dataclass, field

import pytest

import voiceclaw.adapters.openshell_fabric.committed_turn as committed_turn_module
from voiceclaw.adapters.openshell_fabric.client import (
    OpenShellClientError,
    OpenShellFailureCode,
    OpenShellFailurePhase,
    SandboxExecution,
)
from voiceclaw.adapters.openshell_fabric.committed_turn import (
    FABRIC_AGENT_BINARY,
    OpenShellFabricAdapter,
    OpenShellFabricAmbiguous,
    OpenShellFabricError,
    OpenShellFabricRejected,
)
from voiceclaw.adapters.openshell_fabric.factory import build_openshell_fabric_backend
from voiceclaw.adapters.result_envelope import DEFAULT_RESULT_SPEECH_BUDGET_BYTES
from voiceclaw.config import BackendProfile, ConfigurationError
from voiceclaw.domain.models import BackendOperation, Durability, EventDelivery
from voiceclaw.domain.response_only import (
    RESULT_ENVELOPE_SCHEMA,
    ResponseOnlyContextContinuity,
    ResponseOnlyTargetAvailability,
)
from voiceclaw.ports.turns import (
    CommittedTurnCompleted,
    CommittedTurnDispatchMarker,
    CommittedTurnDisplayDelta,
    CommittedTurnRequest,
)


def _wire(value: object) -> str:
    return json.dumps(value, separators=(",", ":")) + "\n"


def _result_text(*, speech: str | None = "The result is ready.", display: str = "## Result\n\nDone.") -> str:
    return json.dumps(
        {"schema": RESULT_ENVELOPE_SCHEMA, "speech": speech, "display": display},
        separators=(",", ":"),
    )


def _invoke_success(
    *,
    commit_id: str = "commit-1",
    invocation_id: str = "invocation-1",
    result_text: str | None = None,
    **overrides: object,
) -> SandboxExecution:
    result: dict[str, object] = {
        "agent_name": "researcher",
        "harness": "nvidia.fabric.hermes",
        "adapter_kind": "python",
        "adapter_id": "nvidia.fabric.hermes",
        "runtime_id": "runtime-1",
        "invocation_id": invocation_id,
        "request_id": commit_id,
        "status": "succeeded",
        "output": {"response": result_text or _result_text()},
        "error": None,
        "usage": None,
        "artifacts": {"root": None, "artifacts": []},
        "telemetry": [],
        "events": [],
        "metadata": {},
    }
    result.update(overrides)
    return SandboxExecution(
        0,
        _wire(
            {
                "operation": "invoke",
                "status": "succeeded",
                "changed": None,
                "result": result,
                "error": None,
            }
        ),
    )


def _bridge_failure(*, effects: str, code: str = "invoke_failed") -> SandboxExecution:
    return SandboxExecution(
        1,
        _wire(
            {
                "operation": "invoke",
                "status": "failed",
                "changed": False if effects == "none" else None,
                "result": None,
                "error": {"code": code, "stage": "invoke", "message": "private", "effects": effects},
            }
        ),
    )


@dataclass
class _Executor:
    responses: list[SandboxExecution | BaseException]
    calls: list[tuple[tuple[str, ...], bytes | None, int]] = field(default_factory=list)
    closed: bool = False
    started: threading.Event | None = None
    release: threading.Event | None = None

    def execute(
        self,
        command: Sequence[str],
        *,
        stdin: bytes | None,
        timeout_seconds: int,
    ) -> SandboxExecution:
        self.calls.append((tuple(command), stdin, timeout_seconds))
        outcome = self.responses.pop(0)
        if self.started is not None:
            self.started.set()
            assert self.release is not None
            self.release.wait(timeout=5)
            self.started = None
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    def close(self) -> None:
        self.closed = True


def _adapter(executor: _Executor, **overrides: object) -> OpenShellFabricAdapter:
    values = {
        "executor": executor,
        "workspace": "voice",
        "sandbox": "deployed-agent",
        "fabric_agent": "researcher",
        "runtime_id": "runtime-1",
        "adapter_id": "nvidia.fabric.hermes",
        "harness": "nvidia.fabric.hermes",
        "adapter_kind": "python",
        "result_pointer": "/output/response",
    }
    values.update(overrides)
    return OpenShellFabricAdapter(**values)


def _request(
    commit_id: str = "commit-1",
    text: str = "Compare two search tree implementations.",
    conversation_id: str = "conversation-1",
    dispatch_marker: CommittedTurnDispatchMarker | None = None,
) -> CommittedTurnRequest:
    if dispatch_marker is None:
        return CommittedTurnRequest(conversation_id, commit_id, text)
    return CommittedTurnRequest(conversation_id, commit_id, text, dispatch_marker)


def _profile(**settings: object) -> BackendProfile:
    values: dict[str, object] = {
        "endpoint": "127.0.0.1:8080",
        "authentication": "anonymous",
        "workspace": "voice",
        "sandbox": "deployed-agent",
        "fabric_agent": "researcher",
        "runtime_id": "runtime-1",
        "adapter_id": "nvidia.fabric.hermes",
        "harness": "nvidia.fabric.hermes",
        "adapter_kind": "python",
        "result_pointer": "/output/response",
        "tls": False,
    }
    values.update(settings)
    return BackendProfile(kind="openshell_fabric", settings=values)


def test_dispatch_marker_is_idempotent_across_threads() -> None:
    callback_calls = 0
    callback_lock = threading.Lock()

    def accept() -> bool:
        nonlocal callback_calls
        with callback_lock:
            callback_calls += 1
        return True

    marker = CommittedTurnDispatchMarker(accept)
    outcomes: list[bool] = []
    threads = [threading.Thread(target=lambda: outcomes.append(marker.mark())) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert outcomes == [True] * 8
    assert callback_calls == 1
    assert marker.marked is True


def test_commit_invokes_the_exact_nemoclaw_owned_runtime_with_a_run_request() -> None:
    executor = _Executor([_invoke_success()])
    result = asyncio.run(_adapter(executor).commit_turn(_request()))

    assert result.backend_session_id == "runtime-1"
    assert result.turn_id == "commit-1"
    assert result.response_id == "invocation-1"
    assert result.display_text == "## Result\n\nDone."
    assert result.speak_text == "The result is ready."
    command, stdin, timeout = executor.calls[0]
    assert command == (
        FABRIC_AGENT_BINARY,
        "invoke",
        "--agent",
        "researcher",
        "--runtime-id",
        "runtime-1",
        "--request",
        "-",
    )
    assert timeout == 330
    assert stdin is not None
    payload = json.loads(stdin)
    assert payload["request_id"] == "commit-1"
    assert str(uuid.UUID(payload["relay_session_root"])) == payload["relay_session_root"]
    assert payload["context"] == {
        "voiceclaw_conversation_id": "conversation-1",
        "voiceclaw_commit_id": "commit-1",
    }
    assert "Compare two search tree implementations." in payload["input"]
    assert RESULT_ENVELOPE_SCHEMA in payload["input"]
    assert f"at most {DEFAULT_RESULT_SPEECH_BUDGET_BYTES} UTF-8 bytes" in payload["input"]


def test_ordered_turns_reuse_one_runtime_for_one_frontend_conversation() -> None:
    executor = _Executor(
        [
            _invoke_success(),
            _invoke_success(
                commit_id="commit-2",
                invocation_id="invocation-2",
                result_text=_result_text(display="Next"),
            ),
        ]
    )
    adapter = _adapter(executor)

    first = asyncio.run(adapter.commit_turn(_request()))
    second = asyncio.run(adapter.commit_turn(_request("commit-2", "Follow up.")))
    descriptor = asyncio.run(adapter.inspect())

    assert first.backend_session_id == second.backend_session_id == "runtime-1"
    assert second.display_text == "Next"
    assert descriptor.context_continuity is ResponseOnlyContextContinuity.UNQUALIFIED
    assert descriptor.target_availability is ResponseOnlyTargetAvailability.AVAILABLE
    assert descriptor.capabilities.sessionful is False
    assert [call[0] for call in executor.calls] == [
        (
            FABRIC_AGENT_BINARY,
            "invoke",
            "--agent",
            "researcher",
            "--runtime-id",
            "runtime-1",
            "--request",
            "-",
        )
    ] * 2
    first_payload = executor.calls[0][1]
    second_payload = executor.calls[1][1]
    assert first_payload is not None and second_payload is not None
    first_request = json.loads(first_payload)
    second_request = json.loads(second_payload)
    assert first_request["relay_session_root"] == second_request["relay_session_root"]
    assert first_request["request_id"] != second_request["request_id"]


def test_runtime_binding_rejects_another_frontend_conversation_without_dispatch() -> None:
    executor = _Executor(
        [
            _invoke_success(),
            _invoke_success(commit_id="commit-2", invocation_id="invocation-2"),
        ]
    )
    adapter = _adapter(executor)
    asyncio.run(adapter.commit_turn(_request()))

    with pytest.raises(OpenShellFabricRejected, match="target_context_mismatch"):
        asyncio.run(adapter.commit_turn(_request("other", conversation_id="conversation-2")))

    asyncio.run(adapter.commit_turn(_request("commit-2", "Continue.")))
    assert len(executor.calls) == 2


@pytest.mark.parametrize(
    "turn_request",
    [
        _request(commit_id="x" * 513),
        _request(conversation_id="line\nbreak"),
    ],
)
def test_invalid_correlation_identity_is_rejected_before_dispatch(turn_request: CommittedTurnRequest) -> None:
    executor = _Executor([])

    with pytest.raises(OpenShellFabricError, match="invalid_fabric_correlation"):
        asyncio.run(_adapter(executor).commit_turn(turn_request))

    assert executor.calls == []


def test_concurrent_turns_are_serialized_in_admission_order() -> None:
    started = threading.Event()
    release = threading.Event()
    executor = _Executor(
        [
            _invoke_success(),
            _invoke_success(commit_id="commit-2", invocation_id="invocation-2"),
        ],
        started=started,
        release=release,
    )
    adapter = _adapter(executor)

    async def exercise() -> None:
        first = asyncio.create_task(adapter.commit_turn(_request()))
        while not started.is_set():
            await asyncio.sleep(0.01)
        descriptor = await adapter.inspect()
        assert descriptor.target_availability is ResponseOnlyTargetAvailability.IN_FLIGHT
        assert descriptor.capabilities.operations == frozenset({BackendOperation.SUBMIT})
        second = asyncio.create_task(adapter.commit_turn(_request("commit-2", "Continue.")))
        await asyncio.sleep(0.05)
        assert len(executor.calls) == 1
        release.set()
        first_result, second_result = await asyncio.gather(first, second)
        assert first_result.response_id == "invocation-1"
        assert second_result.response_id == "invocation-2"

    asyncio.run(exercise())


def test_cancelled_dispatched_invocation_fences_late_completion() -> None:
    started = threading.Event()
    release = threading.Event()
    executor = _Executor([_invoke_success()], started=started, release=release)
    adapter = _adapter(executor)
    effects = threading.Event()
    marker = CommittedTurnDispatchMarker(lambda: effects.set() or True)

    async def exercise() -> None:
        stream = adapter.stream_turn(_request(dispatch_marker=marker))
        invocation = asyncio.create_task(anext(stream))
        while not started.is_set():
            await asyncio.sleep(0.01)
        assert effects.is_set()
        assert marker.marked is True
        invocation.cancel()
        with pytest.raises(asyncio.CancelledError):
            await invocation
        assert adapter._target_availability() is ResponseOnlyTargetAvailability.CONSUMED
        release.set()
        while adapter._invoke_gate.locked():
            await asyncio.sleep(0.01)
        with pytest.raises(OpenShellFabricRejected, match="target_context_consumed"):
            await adapter.commit_turn(_request("commit-2", "Continue."))

    asyncio.run(exercise())
    assert len(executor.calls) == 1


def test_stream_cancelled_before_dispatch_gate_never_marks_backend_effects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executor = _Executor([_invoke_success()])
    adapter = _adapter(executor)
    entered = threading.Event()
    release = threading.Event()
    effects = threading.Event()
    marker = CommittedTurnDispatchMarker(lambda: effects.set() or True)
    build_prompt = adapter._build_prompt

    def paused_build_prompt(text: str) -> str:
        entered.set()
        release.wait(timeout=5)
        return build_prompt(text)

    monkeypatch.setattr(adapter, "_build_prompt", paused_build_prompt)

    async def exercise() -> None:
        stream = adapter.stream_turn(_request(dispatch_marker=marker))
        first = asyncio.create_task(anext(stream))
        while not entered.is_set():
            await asyncio.sleep(0.01)
        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first
        assert effects.is_set() is False
        assert marker.marked is False
        assert executor.calls == []
        release.set()
        while adapter._invoke_gate.locked():
            await asyncio.sleep(0.01)
        assert effects.is_set() is False
        assert marker.marked is False
        assert adapter._target_availability() is ResponseOnlyTargetAvailability.AVAILABLE

    try:
        asyncio.run(exercise())
    finally:
        release.set()


def test_cancelled_queued_invocation_never_dispatches_or_fences_the_runtime() -> None:
    started = threading.Event()
    release = threading.Event()
    executor = _Executor(
        [
            _invoke_success(),
            _invoke_success(commit_id="commit-3", invocation_id="invocation-3"),
        ],
        started=started,
        release=release,
    )
    adapter = _adapter(executor)
    queued_marker = CommittedTurnDispatchMarker()

    async def exercise() -> None:
        first = asyncio.create_task(adapter.commit_turn(_request()))
        while not started.is_set():
            await asyncio.sleep(0.01)
        queued = asyncio.create_task(
            adapter.commit_turn(_request("commit-2", "Queued.", dispatch_marker=queued_marker))
        )
        await asyncio.sleep(0.05)
        queued.cancel()
        with pytest.raises(asyncio.CancelledError):
            await queued
        assert queued_marker.marked is False
        release.set()
        await first
        while adapter._invoke_gate.locked():
            await asyncio.sleep(0.01)
        result = await adapter.commit_turn(_request("commit-3", "Continue."))
        assert result.response_id == "invocation-3"

    asyncio.run(exercise())
    assert len(executor.calls) == 2


def test_dispatch_marker_veto_rejects_without_execution_and_does_not_burn_commit_id() -> None:
    executor = _Executor([_invoke_success(commit_id="commit-2")])
    adapter = _adapter(executor)
    vetoed = CommittedTurnDispatchMarker(lambda: False)

    with pytest.raises(OpenShellFabricRejected, match="invocation_cancelled"):
        asyncio.run(adapter.commit_turn(_request(dispatch_marker=vetoed)))

    assert vetoed.marked is False
    assert executor.calls == []
    assert adapter._target_availability() is ResponseOnlyTargetAvailability.AVAILABLE
    assert adapter._bound_conversation_id is None
    assert (
        asyncio.run(adapter.commit_turn(_request("commit-2", conversation_id="conversation-2"))).response_id
        == "invocation-1"
    )
    assert adapter._bound_conversation_id == "conversation-2"


def test_volatile_replay_history_is_bounded_without_forgetting_identities(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(committed_turn_module, "MAX_VOLATILE_SEQUENTIAL_TURNS", 2)
    executor = _Executor(
        [
            _invoke_success(),
            _invoke_success(commit_id="commit-2", invocation_id="invocation-2"),
        ]
    )
    adapter = _adapter(executor)

    asyncio.run(adapter.commit_turn(_request()))
    asyncio.run(adapter.commit_turn(_request("commit-2", "Continue.")))

    assert adapter._request_ids == {"commit-1", "commit-2"}
    assert adapter._invocation_ids == {"invocation-1", "invocation-2"}
    assert asyncio.run(adapter.inspect()).target_availability is ResponseOnlyTargetAvailability.CONSUMED
    with pytest.raises(OpenShellFabricRejected, match="target_context_consumed"):
        asyncio.run(adapter.commit_turn(_request("commit-3", "Continue again.")))
    assert len(executor.calls) == 2


def test_replay_identity_is_rejected_before_the_volatile_bound_is_reached(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(committed_turn_module, "MAX_VOLATILE_SEQUENTIAL_TURNS", 3)
    executor = _Executor(
        [
            _invoke_success(),
            _invoke_success(commit_id="commit-2", invocation_id="invocation-2"),
        ]
    )
    adapter = _adapter(executor)

    asyncio.run(adapter.commit_turn(_request()))
    asyncio.run(adapter.commit_turn(_request("commit-2", "Continue.")))

    with pytest.raises(OpenShellFabricRejected, match="duplicate_request"):
        asyncio.run(adapter.commit_turn(_request()))
    assert len(executor.calls) == 2
    assert asyncio.run(adapter.inspect()).target_availability is ResponseOnlyTargetAvailability.AVAILABLE


@pytest.mark.parametrize(
    "outcome",
    [
        OpenShellClientError(OpenShellFailureCode.TIMEOUT),
        SandboxExecution(0, "not-json\n"),
        _bridge_failure(effects="unknown"),
    ],
)
def test_ambiguous_post_dispatch_failure_fences_the_runtime(outcome: SandboxExecution | BaseException) -> None:
    executor = _Executor([outcome])
    adapter = _adapter(executor)

    with pytest.raises(OpenShellFabricAmbiguous, match="invocation_outcome_unknown"):
        asyncio.run(adapter.commit_turn(_request()))
    assert asyncio.run(adapter.inspect()).target_availability is ResponseOnlyTargetAvailability.CONSUMED


@pytest.mark.parametrize(
    "code",
    [OpenShellFailureCode.AUTH_DENIED, OpenShellFailureCode.TARGET_MISSING],
)
def test_post_admission_sdk_rejection_is_ambiguous_and_fences_the_runtime(code: OpenShellFailureCode) -> None:
    error = OpenShellClientError(code, phase=OpenShellFailurePhase.ADMITTED_STREAM)
    adapter = _adapter(_Executor([error]))

    with pytest.raises(OpenShellFabricAmbiguous, match="invocation_outcome_unknown"):
        asyncio.run(adapter.commit_turn(_request()))
    assert asyncio.run(adapter.inspect()).target_availability is ResponseOnlyTargetAvailability.CONSUMED


@pytest.mark.parametrize(
    ("code", "message"),
    [
        (OpenShellFailureCode.AUTH_DENIED, "auth_denied"),
        (OpenShellFailureCode.TARGET_MISSING, "target_missing"),
    ],
)
def test_pre_admission_sdk_rejection_releases_the_runtime(
    code: OpenShellFailureCode,
    message: str,
) -> None:
    executor = _Executor(
        [
            OpenShellClientError(code, phase=OpenShellFailurePhase.BEFORE_ADMISSION),
            _invoke_success(commit_id="commit-2", invocation_id="invocation-2"),
        ]
    )
    adapter = _adapter(executor)

    with pytest.raises(OpenShellFabricRejected, match=message):
        asyncio.run(adapter.commit_turn(_request()))
    assert asyncio.run(adapter.commit_turn(_request("commit-2"))).response_id == "invocation-2"


def test_confirmed_bridge_rejection_does_not_fence_the_runtime() -> None:
    executor = _Executor(
        [
            _bridge_failure(effects="none"),
            _invoke_success(commit_id="commit-2", invocation_id="invocation-2"),
        ]
    )
    adapter = _adapter(executor)

    with pytest.raises(OpenShellFabricRejected, match="invocation_rejected"):
        asyncio.run(adapter.commit_turn(_request()))
    result = asyncio.run(adapter.commit_turn(_request("commit-2")))
    assert result.response_id == "invocation-2"


@pytest.mark.parametrize(
    ("status", "error", "expected_code"),
    [
        (
            "failed",
            {
                "stage": "invoke",
                "code": "harness_failed",
                "message": "private",
                "retryable": False,
                "metadata": {},
            },
            "invocation_failed",
        ),
        ("cancelled", None, "invocation_cancelled"),
    ],
)
def test_terminal_harness_outcome_is_known_and_runtime_remains_usable(
    status: str, error: dict[str, object] | None, expected_code: str
) -> None:
    executor = _Executor(
        [
            _invoke_success(status=status, error=error),
            _invoke_success(commit_id="commit-2", invocation_id="invocation-2"),
        ]
    )
    adapter = _adapter(executor)

    with pytest.raises(OpenShellFabricError, match=expected_code):
        asyncio.run(adapter.commit_turn(_request()))
    assert asyncio.run(adapter.commit_turn(_request("commit-2"))).response_id == "invocation-2"


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value["result"].__setitem__("request_id", "wrong"),
        lambda value: value["result"].__setitem__("runtime_id", "wrong"),
        lambda value: value["result"].__setitem__("adapter_id", "wrong"),
        lambda value: value["result"].__setitem__("harness", "wrong"),
        lambda value: value["result"].__setitem__("adapter_kind", "wrong"),
    ],
)
def test_result_binding_mismatch_fences_the_runtime(mutation) -> None:
    value = json.loads(_invoke_success().stdout)
    mutation(value)
    adapter = _adapter(_Executor([SandboxExecution(0, _wire(value))]))

    with pytest.raises(OpenShellFabricAmbiguous, match="invocation_result_unusable"):
        asyncio.run(adapter.commit_turn(_request()))
    assert asyncio.run(adapter.inspect()).target_availability is ResponseOnlyTargetAvailability.CONSUMED


def test_reused_invocation_identity_fences_the_runtime() -> None:
    executor = _Executor([_invoke_success(), _invoke_success(commit_id="commit-2")])
    adapter = _adapter(executor)
    asyncio.run(adapter.commit_turn(_request()))

    with pytest.raises(OpenShellFabricAmbiguous, match="invocation_result_unusable"):
        asyncio.run(adapter.commit_turn(_request("commit-2")))


def test_stream_turn_keeps_display_and_speech_separate() -> None:
    adapter = _adapter(_Executor([_invoke_success()]))

    async def collect() -> list[CommittedTurnDisplayDelta | CommittedTurnCompleted]:
        return [event async for event in adapter.stream_turn(_request())]

    events = asyncio.run(collect())
    assert isinstance(events[0], CommittedTurnDisplayDelta)
    assert events[0].delta == "## Result\n\nDone."
    assert isinstance(events[1], CommittedTurnCompleted)
    assert events[1].result.speak_text == "The result is ready."


def test_inspect_reports_only_configured_non_durable_capabilities_without_remote_probe() -> None:
    executor = _Executor([])
    descriptor = asyncio.run(_adapter(executor).inspect())

    assert executor.calls == []
    assert descriptor.capabilities.operations == frozenset({BackendOperation.SUBMIT})
    assert descriptor.capabilities.durability is Durability.NONE
    assert descriptor.capabilities.event_delivery is EventDelivery.RESPONSE_ONLY
    assert descriptor.capabilities.supports_parallel_work is False
    assert descriptor.context_continuity is ResponseOnlyContextContinuity.UNQUALIFIED


def test_factory_requires_typed_runtime_and_adapter_binding(monkeypatch) -> None:
    captured: dict[str, object] = {}
    executor = _Executor([])

    def build_executor(**kwargs):
        captured.update(kwargs)
        return executor

    monkeypatch.setattr("voiceclaw.adapters.openshell_fabric.factory.SdkSandboxExecutor", build_executor)
    composition = build_openshell_fabric_backend(_profile(), {})

    assert composition.turn_backend is not None
    assert composition.selected_agent_readiness is None
    assert composition.turn_backend._runtime_id == "runtime-1"
    assert composition.turn_backend._adapter_id == "nvidia.fabric.hermes"
    assert composition.turn_backend._harness == "nvidia.fabric.hermes"
    assert composition.turn_backend._adapter_kind == "python"
    assert composition.turn_backend._result_pointer == "/output/response"
    assert captured["authentication"] == "anonymous"


@pytest.mark.parametrize("adapter_kind", ["http", "native_plugin", "process", "python"])
def test_factory_accepts_each_fabric_05_adapter_kind(monkeypatch, adapter_kind: str) -> None:
    monkeypatch.setattr(
        "voiceclaw.adapters.openshell_fabric.factory.SdkSandboxExecutor",
        lambda **_kwargs: _Executor([]),
    )

    composition = build_openshell_fabric_backend(_profile(adapter_kind=adapter_kind), {})

    assert composition.turn_backend is not None
    assert composition.turn_backend._adapter_kind == adapter_kind


def test_factory_rejects_unknown_fabric_adapter_kind_before_constructing_sdk(monkeypatch) -> None:
    constructed = False

    def build_executor(**_kwargs):
        nonlocal constructed
        constructed = True
        return _Executor([])

    monkeypatch.setattr("voiceclaw.adapters.openshell_fabric.factory.SdkSandboxExecutor", build_executor)

    with pytest.raises(ConfigurationError, match="adapter_kind must be one of"):
        build_openshell_fabric_backend(_profile(adapter_kind="shell"), {})

    assert constructed is False


def test_adapter_rejects_unknown_fabric_adapter_kind() -> None:
    with pytest.raises(ValueError, match="adapter_kind is not supported"):
        _adapter(_Executor([]), adapter_kind="shell")


@pytest.mark.parametrize("setting", ["runtime_id", "adapter_id", "harness", "adapter_kind", "result_pointer"])
def test_factory_rejects_missing_typed_binding_setting(setting: str) -> None:
    settings = dict(_profile().settings)
    settings.pop(setting)
    profile = BackendProfile(kind="openshell_fabric", settings=settings)

    with pytest.raises(ConfigurationError, match=setting):
        build_openshell_fabric_backend(profile, {})


def test_close_never_stops_the_nemoclaw_owned_runtime() -> None:
    executor = _Executor([])
    adapter = _adapter(executor)
    adapter.close()

    assert executor.closed is True
    assert executor.calls == []
