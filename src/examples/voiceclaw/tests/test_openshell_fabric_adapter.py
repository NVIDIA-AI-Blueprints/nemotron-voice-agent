# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

# ruff: noqa: D100, D101, D102, D103

from __future__ import annotations

import asyncio
import json
import threading
from collections.abc import Sequence
from dataclasses import dataclass, field

import pytest

from voiceclaw.adapters.openshell_fabric import security
from voiceclaw.adapters.openshell_fabric.client import (
    OpenShellClientError,
    OpenShellFailureCode,
    SandboxExecution,
    SdkSandboxExecutor,
    _classify_sdk_error,
)
from voiceclaw.adapters.openshell_fabric.committed_turn import (
    FABRIC_AGENT_BINARY,
    OpenShellFabricAdapter,
    OpenShellFabricAmbiguous,
    OpenShellFabricError,
    OpenShellFabricRejected,
)
from voiceclaw.adapters.openshell_fabric.factory import _endpoint, _issuer, build_openshell_fabric_backend
from voiceclaw.adapters.result_envelope import (
    DEFAULT_RESULT_SPEECH_BUDGET_BYTES,
    MAX_RESULT_SPEECH_BYTES,
    build_result_envelope_prompt,
)
from voiceclaw.config import BackendProfile, ConfigurationError
from voiceclaw.domain.models import BackendOperation, Durability, EventDelivery
from voiceclaw.domain.response_only import (
    RESULT_ENVELOPE_SCHEMA,
    ResponseOnlyContextContinuity,
    ResponseOnlyTargetAvailability,
)
from voiceclaw.ports.readiness import SelectedAgentReadinessError
from voiceclaw.ports.turns import CommittedTurnCompleted, CommittedTurnDisplayDelta, CommittedTurnRequest


def _wire(value: object) -> str:
    return json.dumps(value, separators=(",", ":")) + "\n"


def _result_text(*, speech: str | None = "The result is ready.", display: str = "## Result\n\nDone.") -> str:
    return json.dumps(
        {"schema": RESULT_ENVELOPE_SCHEMA, "speech": speech, "display": display},
        separators=(",", ":"),
    )


def _invoke_success(
    *,
    result_text: str | None = None,
    adapter_id: object = "nvidia.fabric.openclaw",
    display_budget_bytes: int | None = None,
    speech_budget_bytes: int | None = None,
) -> SandboxExecution:
    prompt_budgets = {}
    if display_budget_bytes is not None:
        prompt_budgets["display_budget_bytes"] = display_budget_bytes
    if speech_budget_bytes is not None:
        prompt_budgets["speech_budget_bytes"] = speech_budget_bytes
    prompt = build_result_envelope_prompt(
        "Compare two search tree implementations.",
        **prompt_budgets,
    )
    response_text = result_text or _result_text()
    fabric_result: dict[str, object] = {
        "agent_name": "researcher",
        "harness": "nvidia.fabric.openclaw",
        "adapter_kind": "python",
        "runtime_id": "runtime-1",
        "invocation_id": "invocation-1",
        "request_id": "request-1",
        "status": "succeeded",
        "output": {
            "harness": "openclaw",
            "response": response_text,
            "session_key": "agent:default:fabric-runtime-1",
            "messages": [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": response_text, "tool_calls": []},
            ],
        },
    }
    fabric_result["adapter_id"] = adapter_id
    return SandboxExecution(
        0,
        _wire(
            {
                "operation": "invoke",
                "status": "succeeded",
                "changed": None,
                "result": {"runtime_id": "runtime-1", "fabric_result": fabric_result},
                "error": None,
            }
        ),
    )


def _check_unsupported() -> SandboxExecution:
    return SandboxExecution(
        1,
        _wire(
            {
                "operation": "check",
                "status": "unsupported",
                "changed": False,
                "result": {
                    "runtime_id": "runtime-1",
                    "runtime_state": "running",
                    "generation": "generation-1",
                    "applied_config": {
                        "metadata": {"name": "researcher"},
                        "harness": {
                            "adapter_id": "nvidia.fabric.openclaw",
                            "settings": {"agent_name": "default"},
                        },
                    },
                    "health": None,
                },
                "error": {
                    "code": "fabric_health_unsupported",
                    "stage": "check",
                    "message": "private detail",
                    "effects": "none",
                },
            }
        ),
    )


def _oauth_profile(*, credential_env: str = "SECRET", **settings: object) -> BackendProfile:
    defaults: dict[str, object] = {
        "endpoint": "127.0.0.1:8080",
        "workspace": "voice",
        "sandbox": "deployed-agent",
        "fabric_agent": "researcher",
        "adapter_id": "nvidia.fabric.openclaw",
        "issuer": "http://127.0.0.1:8081",
        "client_id": "voiceclaw",
        "tls": False,
    }
    return BackendProfile(
        kind="openshell_fabric",
        credential_env=credential_env,
        settings={**defaults, **settings},
    )


@dataclass
class _Executor:
    responses: list[SandboxExecution | BaseException]
    calls: list[tuple[tuple[str, ...], bytes | None, int]] = field(default_factory=list)
    closed: bool = False
    started: threading.Event | None = None
    release: threading.Event | None = None
    blocked_operation: str = "invoke"

    def execute(
        self,
        command: Sequence[str],
        *,
        stdin: bytes | None,
        timeout_seconds: int,
    ) -> SandboxExecution:
        self.calls.append((tuple(command), stdin, timeout_seconds))
        outcome = self.responses.pop(0)
        if self.started is not None and command[1] == self.blocked_operation:
            self.started.set()
            assert self.release is not None
            self.release.wait(timeout=5)
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
        "adapter_id": "nvidia.fabric.openclaw",
        "native_agent": "default",
        "readiness_cache_ttl_seconds": 0,
    }
    values.update(overrides)
    return OpenShellFabricAdapter(**values)


def _request() -> CommittedTurnRequest:
    return CommittedTurnRequest(
        runtime_conversation_id="conversation-1",
        commit_id="commit-1",
        text="Compare two search tree implementations.",
    )


def test_commit_uses_one_stdin_invoke_and_keeps_speech_separate_from_display() -> None:
    executor = _Executor([_invoke_success()])
    adapter = _adapter(executor)
    result = asyncio.run(adapter.commit_turn(_request()))
    assert result.backend_session_id == "agent:default:fabric-runtime-1"
    assert result.turn_id == "request-1"
    assert result.response_id == "invocation-1"
    assert result.display_text == "## Result\n\nDone."
    assert result.speak_text == "The result is ready."
    assert len(executor.calls) == 1
    command, stdin, timeout = executor.calls[0]
    assert command == (
        FABRIC_AGENT_BINARY,
        "invoke",
        "--agent",
        "researcher",
        "--input",
        "-",
    )
    assert timeout == 330
    assert stdin is not None
    payload = json.loads(stdin)
    assert payload["agent"] == "default"
    assert "Compare two search tree implementations." in payload["message"]
    assert RESULT_ENVELOPE_SCHEMA in payload["message"]
    assert f"at most {DEFAULT_RESULT_SPEECH_BUDGET_BYTES} UTF-8 bytes containing only facts" in payload["message"]


def test_response_only_target_is_claimed_by_exactly_one_invocation() -> None:
    executor = _Executor([_invoke_success()])
    adapter = _adapter(executor)

    first = asyncio.run(adapter.commit_turn(_request()))
    assert first.display_text == "## Result\n\nDone."
    backend = asyncio.run(adapter.inspect())
    assert backend.capabilities.operations == frozenset()
    assert backend.target_availability is ResponseOnlyTargetAvailability.CONSUMED
    with pytest.raises(OpenShellFabricRejected, match="target_context_consumed"):
        asyncio.run(adapter.commit_turn(_request()))

    assert [call[0][1] for call in executor.calls] == ["invoke"]


def test_ambiguous_invocation_permanently_consumes_the_response_only_target() -> None:
    executor = _Executor([SandboxExecution(0, "not-json\n")])
    adapter = _adapter(executor)

    with pytest.raises(OpenShellFabricAmbiguous, match="invocation_outcome_unknown"):
        asyncio.run(adapter.commit_turn(_request()))
    with pytest.raises(OpenShellFabricRejected, match="target_context_consumed"):
        asyncio.run(adapter.commit_turn(_request()))

    assert [call[0][1] for call in executor.calls] == ["invoke"]


def test_unavailable_invocation_is_ambiguous_not_retried_and_consumes_the_target() -> None:
    executor = _Executor([OpenShellClientError(OpenShellFailureCode.UNAVAILABLE)])
    adapter = _adapter(executor)

    with pytest.raises(OpenShellFabricAmbiguous, match="invocation_outcome_unknown"):
        asyncio.run(adapter.commit_turn(_request()))
    with pytest.raises(OpenShellFabricRejected, match="target_context_consumed"):
        asyncio.run(adapter.commit_turn(_request()))

    assert [call[0][1] for call in executor.calls] == ["invoke"]


def test_confirmed_no_effect_rejection_does_not_consume_the_target() -> None:
    rejection = SandboxExecution(
        1,
        _wire(
            {
                "operation": "invoke",
                "status": "failed",
                "changed": False,
                "result": None,
                "error": {
                    "code": "bad_input",
                    "stage": "request",
                    "message": "private detail",
                    "effects": "none",
                },
            }
        ),
    )
    executor = _Executor([rejection, _invoke_success()])
    adapter = _adapter(executor)

    with pytest.raises(OpenShellFabricRejected, match="invocation_rejected"):
        asyncio.run(adapter.commit_turn(_request()))
    result = asyncio.run(adapter.commit_turn(_request()))

    assert result.display_text == "## Result\n\nDone."
    assert [call[0][1] for call in executor.calls] == ["invoke", "invoke"]


def test_concurrent_invocation_never_reaches_the_claimed_target() -> None:
    started = threading.Event()
    release = threading.Event()
    executor = _Executor([_invoke_success()], started=started, release=release)
    adapter = _adapter(executor)

    async def exercise() -> None:
        first = asyncio.create_task(adapter.commit_turn(_request()))
        for _ in range(200):
            if started.is_set():
                break
            await asyncio.sleep(0.01)
        assert started.is_set()
        backend = await asyncio.wait_for(adapter.inspect(), timeout=0.25)
        assert backend.capabilities.operations == frozenset()
        assert backend.target_availability is ResponseOnlyTargetAvailability.IN_FLIGHT
        with pytest.raises(OpenShellFabricError, match="backend_busy"):
            await adapter.commit_turn(_request())
        release.set()
        await first

    asyncio.run(exercise())
    assert [call[0][1] for call in executor.calls] == ["invoke"]


def test_terminal_bridge_projects_one_display_delta_before_completion() -> None:
    adapter = _adapter(_Executor([_invoke_success()]))

    async def collect() -> list[CommittedTurnDisplayDelta | CommittedTurnCompleted]:
        return [event async for event in adapter.stream_turn(_request())]

    events = asyncio.run(collect())
    assert len(events) == 2
    assert isinstance(events[0], CommittedTurnDisplayDelta)
    assert events[0].sequence == 0
    assert events[0].delta == "## Result\n\nDone."
    assert isinstance(events[1], CommittedTurnCompleted)
    assert events[1].result.display_text == events[0].delta


@pytest.mark.parametrize("speech", [None, "The result is ready."])
def test_history_outer_whitespace_preserves_result_channels_and_single_invocation(speech: str | None) -> None:
    execution = _invoke_success(result_text=_result_text(speech=speech))
    wire = json.loads(execution.stdout)
    output = wire["result"]["fabric_result"]["output"]
    output["messages"][1]["content"] = "\n" + output["response"] + "\n"
    executor = _Executor([SandboxExecution(0, _wire(wire))])
    adapter = _adapter(executor)

    async def collect() -> list[CommittedTurnDisplayDelta | CommittedTurnCompleted]:
        return [event async for event in adapter.stream_turn(_request())]

    events = asyncio.run(collect())
    assert len(events) == 2
    assert isinstance(events[0], CommittedTurnDisplayDelta)
    assert events[0].sequence == 0
    assert events[0].delta == "## Result\n\nDone."
    assert isinstance(events[1], CommittedTurnCompleted)
    assert events[1].result.display_text == events[0].delta
    assert events[1].result.speak_text == speech
    assert asyncio.run(adapter.inspect()).target_availability is ResponseOnlyTargetAvailability.CONSUMED
    with pytest.raises(OpenShellFabricRejected, match="target_context_consumed"):
        asyncio.run(adapter.commit_turn(_request()))
    assert [call[0][1] for call in executor.calls] == ["invoke"]


def test_configured_display_budget_is_enforced_on_the_returned_result() -> None:
    executor = _Executor([_invoke_success(result_text=_result_text(display="12345"), display_budget_bytes=4)])
    adapter = _adapter(executor, result_display_budget_bytes=4)
    with pytest.raises(OpenShellFabricAmbiguous, match="invocation_result_unusable"):
        asyncio.run(adapter.commit_turn(_request()))


@pytest.mark.parametrize(
    ("speech_bytes", "speech_available"),
    [
        (DEFAULT_RESULT_SPEECH_BUDGET_BYTES, True),
        (DEFAULT_RESULT_SPEECH_BUDGET_BYTES + 1, False),
    ],
)
def test_default_speech_budget_discards_only_oversized_optional_speech(
    speech_bytes: int, speech_available: bool
) -> None:
    executor = _Executor([_invoke_success(result_text=_result_text(speech="x" * speech_bytes))])
    adapter = _adapter(executor)

    result = asyncio.run(adapter.commit_turn(_request()))

    assert result.display_text == "## Result\n\nDone."
    assert (result.speak_text is not None) is speech_available
    if speech_available:
        assert len(result.speak_text or "") == speech_bytes


def test_configured_speech_budget_is_declared_and_discards_only_oversized_speech() -> None:
    executor = _Executor(
        [
            _invoke_success(
                result_text=_result_text(speech="12345"),
                speech_budget_bytes=4,
            )
        ]
    )
    adapter = _adapter(executor, result_speech_budget_bytes=4)

    result = asyncio.run(adapter.commit_turn(_request()))

    assert result.display_text == "## Result\n\nDone."
    assert result.speak_text is None
    assert executor.calls[0][1] is not None
    payload = json.loads(executor.calls[0][1])
    assert "at most 4 UTF-8 bytes containing only facts" in payload["message"]


@pytest.mark.parametrize("value", [True, 0, MAX_RESULT_SPEECH_BYTES + 1, "512"])
def test_adapter_rejects_invalid_result_speech_budget(value: object) -> None:
    with pytest.raises(ValueError, match="result_speech_budget_bytes is invalid"):
        _adapter(_Executor([]), result_speech_budget_bytes=value)


def test_inspect_accepts_unsupported_check_without_invoking_the_agent() -> None:
    executor = _Executor([_check_unsupported(), _check_unsupported()])
    adapter = _adapter(executor)
    backend = asyncio.run(adapter.inspect())
    asyncio.run(adapter.check_selected_agent())
    assert backend.capabilities.operations == frozenset({BackendOperation.SUBMIT})
    assert backend.target_availability is ResponseOnlyTargetAvailability.AVAILABLE
    assert backend.capabilities.durability is Durability.NONE
    assert backend.capabilities.event_delivery is EventDelivery.RESPONSE_ONLY
    assert backend.context_continuity is ResponseOnlyContextContinuity.UNQUALIFIED
    assert all(call[0] == (FABRIC_AGENT_BINARY, "check", "--agent", "researcher", "--live") for call in executor.calls)


def test_readiness_rejects_an_unsupported_health_code_near_miss() -> None:
    value = json.loads(_check_unsupported().stdout)
    value["error"]["code"] = "fabric_health_unsupported_v2"
    executor = _Executor([SandboxExecution(1, _wire(value))])

    with pytest.raises(SelectedAgentReadinessError, match="selected_agent_unavailable"):
        asyncio.run(_adapter(executor).check_selected_agent())


def test_readiness_rejects_unsupported_status_with_non_null_health() -> None:
    value = json.loads(_check_unsupported().stdout)
    value["result"]["health"] = {"status": "unknown"}
    executor = _Executor([SandboxExecution(1, _wire(value))])

    with pytest.raises(SelectedAgentReadinessError, match="selected_agent_unavailable"):
        asyncio.run(_adapter(executor).check_selected_agent())


def test_readiness_rejects_succeeded_status_with_null_health() -> None:
    value = json.loads(_check_unsupported().stdout)
    value["status"] = "succeeded"
    value["error"] = None
    executor = _Executor([SandboxExecution(0, _wire(value))])

    with pytest.raises(SelectedAgentReadinessError, match="selected_agent_unavailable"):
        asyncio.run(_adapter(executor).check_selected_agent())


def test_verified_readiness_is_cached_for_admission_then_session_bootstrap() -> None:
    executor = _Executor([_check_unsupported()])
    adapter = _adapter(executor, readiness_cache_ttl_seconds=30)

    async def exercise() -> None:
        await adapter.check_selected_agent()
        backend = await adapter.inspect()
        assert backend.capabilities.operations == frozenset({BackendOperation.SUBMIT})

    asyncio.run(exercise())
    assert [call[0][1] for call in executor.calls] == ["check"]


def test_concurrent_readiness_checks_share_one_remote_probe() -> None:
    started = threading.Event()
    release = threading.Event()
    executor = _Executor(
        [_check_unsupported()],
        started=started,
        release=release,
        blocked_operation="check",
    )
    adapter = _adapter(executor, readiness_cache_ttl_seconds=30)

    async def exercise() -> None:
        first = asyncio.create_task(adapter.check_selected_agent())
        for _ in range(200):
            if started.is_set():
                break
            await asyncio.sleep(0.01)
        assert started.is_set()
        second = asyncio.create_task(adapter.check_selected_agent())
        await asyncio.sleep(0.05)
        assert not second.done()
        release.set()
        await asyncio.gather(first, second)

    asyncio.run(exercise())
    assert [call[0][1] for call in executor.calls] == ["check"]


def test_verified_binding_does_not_flap_while_an_invoke_holds_the_gate() -> None:
    started = threading.Event()
    release = threading.Event()
    executor = _Executor([_check_unsupported(), _invoke_success()], started=started, release=release)
    adapter = _adapter(executor)

    async def exercise() -> None:
        await adapter.inspect()
        task = asyncio.create_task(adapter.commit_turn(_request()))
        for _ in range(200):
            if started.is_set():
                break
            await asyncio.sleep(0.01)
        assert started.is_set()
        await adapter.check_selected_agent()
        release.set()
        await task

    asyncio.run(exercise())
    assert len(executor.calls) == 2


def test_failed_revalidation_clears_cached_binding_before_an_invoke() -> None:
    started = threading.Event()
    release = threading.Event()
    executor = _Executor(
        [_check_unsupported(), SandboxExecution(0, "not-json\n"), _invoke_success(), _check_unsupported()],
        started=started,
        release=release,
    )
    adapter = _adapter(executor)

    async def exercise() -> None:
        await adapter.check_selected_agent()
        with pytest.raises(SelectedAgentReadinessError, match="selected_agent_protocol_error"):
            await adapter.check_selected_agent()
        task = asyncio.create_task(adapter.commit_turn(_request()))
        for _ in range(200):
            if started.is_set():
                break
            await asyncio.sleep(0.01)
        assert started.is_set()
        await adapter.check_selected_agent()
        release.set()
        await task

    asyncio.run(exercise())
    assert [call[0][1] for call in executor.calls] == ["check", "check", "invoke", "check"]


def test_concurrent_revalidation_does_not_reuse_stale_binding() -> None:
    invalid = json.loads(_check_unsupported().stdout)
    invalid["result"]["runtime_state"] = "stopped"
    executor = _Executor([_check_unsupported(), SandboxExecution(1, _wire(invalid))])
    adapter = _adapter(executor)

    async def exercise() -> None:
        await adapter.check_selected_agent()
        executor.started = threading.Event()
        executor.release = threading.Event()
        executor.blocked_operation = "check"
        revalidation = asyncio.create_task(adapter.check_selected_agent())
        for _ in range(200):
            if executor.started.is_set():
                break
            await asyncio.sleep(0.01)
        assert executor.started.is_set()
        joined = asyncio.create_task(adapter.check_selected_agent())
        await asyncio.sleep(0.05)
        assert not joined.done()
        executor.release.set()
        outcomes = await asyncio.gather(revalidation, joined, return_exceptions=True)
        assert all(
            isinstance(outcome, SelectedAgentReadinessError) and outcome.code == "selected_agent_unavailable"
            for outcome in outcomes
        )

    asyncio.run(exercise())
    assert [call[0][1] for call in executor.calls] == ["check", "check"]


def test_readiness_check_does_not_reject_an_invocation() -> None:
    started = threading.Event()
    release = threading.Event()
    executor = _Executor(
        [_check_unsupported(), _invoke_success()],
        started=started,
        release=release,
        blocked_operation="check",
    )
    adapter = _adapter(executor)

    async def exercise() -> None:
        readiness = asyncio.create_task(adapter.check_selected_agent())
        for _ in range(200):
            if started.is_set():
                break
            await asyncio.sleep(0.01)
        assert started.is_set()
        result = await asyncio.wait_for(adapter.commit_turn(_request()), timeout=1)
        assert result.display_text == "## Result\n\nDone."
        release.set()
        await readiness

    asyncio.run(exercise())
    assert [call[0][1] for call in executor.calls] == ["check", "invoke"]


@pytest.mark.parametrize("runtime_state", ["stopped", "unknown", "malformed"])
def test_readiness_requires_a_coherent_running_snapshot(runtime_state: str) -> None:
    value = json.loads(_check_unsupported().stdout)
    if runtime_state == "malformed":
        value["result"].pop("generation")
    else:
        value["result"]["runtime_state"] = runtime_state
    executor = _Executor([SandboxExecution(1, _wire(value))])
    with pytest.raises(SelectedAgentReadinessError, match="selected_agent_unavailable"):
        asyncio.run(_adapter(executor).check_selected_agent())


@pytest.mark.parametrize(
    "mutator",
    [
        lambda value: value["result"]["applied_config"]["metadata"].__setitem__("name", "wrong"),
        lambda value: value["result"]["applied_config"]["harness"].__setitem__("adapter_id", "nvidia.fabric.hermes"),
        lambda value: value["result"]["applied_config"]["harness"]["settings"].__setitem__("agent_name", "wrong"),
        lambda value: value["result"].__setitem__("applied_config", {}),
    ],
)
def test_readiness_rejects_the_wrong_agent_or_adapter_binding(mutator) -> None:
    value = json.loads(_check_unsupported().stdout)
    mutator(value)
    executor = _Executor([SandboxExecution(1, _wire(value))])
    with pytest.raises(SelectedAgentReadinessError, match="selected_agent_unavailable"):
        asyncio.run(_adapter(executor).check_selected_agent())


def test_openclaw_readiness_accepts_its_default_native_agent() -> None:
    value = json.loads(_check_unsupported().stdout)
    value["result"]["applied_config"]["harness"]["settings"] = {}
    executor = _Executor([SandboxExecution(1, _wire(value))])
    asyncio.run(_adapter(executor, native_agent="main").check_selected_agent())


@pytest.mark.parametrize(
    "execution,error_type,code",
    [
        (
            SandboxExecution(
                1,
                _wire(
                    {
                        "operation": "invoke",
                        "status": "failed",
                        "changed": False,
                        "result": None,
                        "error": {"code": "bad_input", "stage": "request", "message": "x", "effects": "none"},
                    }
                ),
            ),
            OpenShellFabricRejected,
            "invocation_rejected",
        ),
        (
            SandboxExecution(
                1,
                _wire(
                    {
                        "operation": "invoke",
                        "status": "failed",
                        "changed": None,
                        "result": None,
                        "error": {"code": "failed", "stage": "invoke", "message": "x", "effects": "unknown"},
                    }
                ),
            ),
            OpenShellFabricAmbiguous,
            "invocation_outcome_unknown",
        ),
        (SandboxExecution(0, "not-json\n"), OpenShellFabricAmbiguous, "invocation_outcome_unknown"),
        (
            SandboxExecution(
                0,
                '{"operation":"invoke","status":"succeeded","changed":null,"result":{"value":'
                + ("9" * 5_000)
                + '},"error":null}\n',
            ),
            OpenShellFabricAmbiguous,
            "invocation_outcome_unknown",
        ),
        (
            OpenShellClientError(OpenShellFailureCode.TIMEOUT),
            OpenShellFabricAmbiguous,
            "invocation_outcome_unknown",
        ),
    ],
)
def test_invoke_is_never_retried(execution: SandboxExecution | BaseException, error_type: type, code: str) -> None:
    executor = _Executor([execution])
    adapter = _adapter(executor)
    with pytest.raises(error_type) as captured:
        asyncio.run(adapter.commit_turn(_request()))
    assert captured.value.code == code
    assert len(executor.calls) == 1


def test_unknown_codec_fails_before_contacting_the_agent() -> None:
    executor = _Executor([])
    with pytest.raises(ValueError, match="installed codec"):
        _adapter(executor, adapter_id="vendor.unknown")
    assert executor.calls == []


@pytest.mark.parametrize(
    "mutator",
    [
        lambda value: value["result"]["fabric_result"].pop("request_id"),
        lambda value: value["result"]["fabric_result"].pop("adapter_id"),
        lambda value: value["result"]["fabric_result"].__setitem__("runtime_id", "wrong"),
        lambda value: value["result"]["fabric_result"].__setitem__("agent_name", "wrong"),
        lambda value: value["result"]["fabric_result"].__setitem__("harness", "wrong"),
        lambda value: value["result"]["fabric_result"].__setitem__("adapter_kind", "wrong"),
        lambda value: value["result"]["fabric_result"].__setitem__("adapter_id", "vendor.wrong"),
    ],
)
def test_fabric_correlation_identities_fail_closed(mutator) -> None:
    value = json.loads(_invoke_success(adapter_id="nvidia.fabric.openclaw").stdout)
    mutator(value)
    executor = _Executor([SandboxExecution(0, _wire(value))])
    with pytest.raises(OpenShellFabricAmbiguous, match="invocation_result_unusable"):
        asyncio.run(_adapter(executor).commit_turn(_request()))


def test_retained_fabric_history_is_reported_as_target_context_reuse() -> None:
    value = json.loads(_invoke_success().stdout)
    messages = value["result"]["fabric_result"]["output"]["messages"]
    messages[:0] = [
        {"role": "user", "content": "an earlier request"},
        {"role": "assistant", "content": "an earlier result", "tool_calls": []},
    ]
    executor = _Executor([SandboxExecution(0, _wire(value))])

    with pytest.raises(OpenShellFabricAmbiguous, match="target_context_reused"):
        asyncio.run(_adapter(executor).commit_turn(_request()))


def test_factory_validates_configuration_and_retains_renewable_secret(monkeypatch) -> None:
    captured: dict[str, object] = {}
    executor = _Executor([_check_unsupported()])

    def build_executor(**kwargs):
        captured.update(kwargs)
        return executor

    monkeypatch.setattr(
        "voiceclaw.adapters.openshell_fabric.factory.SdkSandboxExecutor",
        build_executor,
    )
    profile = _oauth_profile(
        credential_env="OPENSHELL_CLIENT_SECRET",
        issuer="http://127.0.0.1:8081/realms/voice",
        scopes=["sandbox:write"],
        result_speech_budget_bytes=256,
    )
    composition = build_openshell_fabric_backend(
        profile,
        {"OPENSHELL_CLIENT_SECRET": "secret-value"},
    )
    assert composition.turn_backend is not None
    assert composition.selected_agent_readiness is composition.turn_backend
    assert captured["scopes"] == ("sandbox:write",)
    assert captured["authentication"] == "client_credentials"
    assert captured["client_secret"]() == "secret-value"
    assert composition.turn_backend._result_speech_budget_bytes == 256


@pytest.mark.parametrize("value", [True, 0, MAX_RESULT_SPEECH_BYTES + 1, "512"])
def test_factory_rejects_invalid_result_speech_budget(value: object) -> None:
    profile = _oauth_profile(result_speech_budget_bytes=value)

    with pytest.raises(ConfigurationError, match="result_speech_budget_bytes"):
        build_openshell_fabric_backend(profile, {"SECRET": "secret-value"})


def test_factory_accepts_nemoclaw_applied_connection_url_and_workspace(monkeypatch) -> None:
    captured: dict[str, object] = {}
    executor = _Executor([_check_unsupported()])

    def build_executor(**kwargs):
        captured.update(kwargs)
        return executor

    monkeypatch.setattr(
        "voiceclaw.adapters.openshell_fabric.factory.SdkSandboxExecutor",
        build_executor,
    )
    profile = BackendProfile(
        kind="openshell_fabric",
        settings={
            "endpoint": "http://127.0.0.1:17681",
            "authentication": "anonymous",
            "workspace": "nc-00e7486fde94e5e0",
            "sandbox": "deployed-agent",
            "fabric_agent": "researcher",
            "adapter_id": "nvidia.fabric.openclaw",
        },
    )

    composition = build_openshell_fabric_backend(profile, {})

    assert composition.turn_backend is not None
    assert captured["endpoint"] == "127.0.0.1:17681"
    assert captured["workspace"] == "nc-00e7486fde94e5e0"
    assert captured["tls"] is False
    assert captured["authentication"] == "anonymous"


def test_factory_derives_remote_tls_from_nemoclaw_gateway_url(monkeypatch) -> None:
    captured: dict[str, object] = {}
    executor = _Executor([_check_unsupported()])

    def build_executor(**kwargs):
        captured.update(kwargs)
        return executor

    monkeypatch.setattr(
        "voiceclaw.adapters.openshell_fabric.factory.SdkSandboxExecutor",
        build_executor,
    )
    settings = dict(
        _oauth_profile(
            endpoint="https://Gateway.Example.com:8443/",
            issuer="https://issuer.example.com",
        ).settings
    )
    settings.pop("tls")
    profile = BackendProfile(
        kind="openshell_fabric",
        credential_env="SECRET",
        settings=settings,
    )

    composition = build_openshell_fabric_backend(profile, {"SECRET": "secret-value"})

    assert composition.turn_backend is not None
    assert captured["endpoint"] == "gateway.example.com:8443"
    assert captured["tls"] is True
    assert captured["authentication"] == "client_credentials"


@pytest.mark.parametrize(
    ("value", "configured_tls", "expected_endpoint", "expected_tls"),
    [
        ("http://127.0.0.1", None, "127.0.0.1:80", False),
        ("https://gateway.example.com/", None, "gateway.example.com:443", True),
        ("http://[::1]", False, "[::1]:80", False),
        ("https://[2001:db8::1]:8443", True, "[2001:db8::1]:8443", True),
        ("gateway.example.com:7443", None, "gateway.example.com:7443", True),
        ("[::1]:17681", False, "[::1]:17681", False),
    ],
)
def test_endpoint_normalizes_urls_and_preserves_legacy_authorities(
    value: str,
    configured_tls: bool | None,
    expected_endpoint: str,
    expected_tls: bool,
) -> None:
    assert _endpoint(value, configured_tls=configured_tls) == (expected_endpoint, expected_tls)


@pytest.mark.parametrize(
    ("value", "configured_tls"),
    [
        ("http://127.0.0.1:17681", True),
        ("https://gateway.example.com:443", False),
    ],
)
def test_endpoint_rejects_tls_settings_that_conflict_with_url_scheme(
    value: str,
    configured_tls: bool,
) -> None:
    with pytest.raises(ConfigurationError, match="conflicts with endpoint scheme"):
        _endpoint(value, configured_tls=configured_tls)


@pytest.mark.parametrize(
    "value",
    [
        "ftp://gateway.example.com:21",
        "ws://gateway.example.com:80",
        "https://user@gateway.example.com:443",
        "https://gateway.example.com:443/v1",
        "https://gateway.example.com:443?",
        "https://gateway.example.com:443#",
        "https://gateway.example.com:",
        "https://gateway.example.com:0",
        "https://gateway.example.com:65536",
        "https://[::1",
        "http://127.0.0.1%2f.example.com:80",
        "https://gateway example.com:443",
        "gateway.example.com",
    ],
)
def test_endpoint_rejects_malformed_or_ambiguous_values(value: str) -> None:
    with pytest.raises(ConfigurationError, match="endpoint"):
        _endpoint(value, configured_tls=None)


def test_factory_allows_anonymous_plaintext_only_on_loopback(monkeypatch) -> None:
    captured: dict[str, object] = {}
    executor = _Executor([_check_unsupported()])

    def build_executor(**kwargs):
        captured.update(kwargs)
        return executor

    monkeypatch.setattr(
        "voiceclaw.adapters.openshell_fabric.factory.SdkSandboxExecutor",
        build_executor,
    )
    settings = {
        "endpoint": "127.0.0.1:8080",
        "authentication": "anonymous",
        "workspace": "voice",
        "sandbox": "deployed-agent",
        "fabric_agent": "researcher",
        "adapter_id": "nvidia.fabric.openclaw",
        "tls": False,
    }
    composition = build_openshell_fabric_backend(
        BackendProfile(kind="openshell_fabric", settings=settings),
        {},
    )
    assert composition.turn_backend is not None
    assert captured["authentication"] == "anonymous"
    assert captured["client_secret"] is None
    assert captured["issuer"] is None
    assert captured["client_id"] is None

    for invalid_settings in (
        {**settings, "endpoint": "gateway.example.com:443", "tls": True},
        {**settings, "tls": True},
        {**settings, "issuer": "https://issuer.example.com"},
    ):
        with pytest.raises(ConfigurationError, match="anonymous authentication"):
            build_openshell_fabric_backend(
                BackendProfile(kind="openshell_fabric", settings=invalid_settings),
                {},
            )

    with pytest.raises(ConfigurationError, match="cannot include a credential"):
        build_openshell_fabric_backend(
            BackendProfile(kind="openshell_fabric", credential_env="SECRET", settings=settings),
            {"SECRET": "secret-value"},
        )


@pytest.mark.parametrize(
    "issuer",
    ["http://127.0.0.1:8081/realms/voice", "https://issuer.example.com"],
)
def test_sdk_executor_never_disables_oauth_tls_verification(
    monkeypatch,
    issuer: str,
) -> None:
    import openshell

    captured: dict[str, object] = {}

    def credentials(**kwargs):
        captured.update(kwargs)
        return object()

    class Client:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        def close(self) -> None:
            pass

    monkeypatch.setattr(openshell, "ClientCredentialsAuth", credentials)
    monkeypatch.setattr(openshell, "SandboxClient", Client)
    SdkSandboxExecutor(
        endpoint="127.0.0.1:8080",
        workspace="voice",
        sandbox="deployed-agent",
        client_secret=lambda: "secret-value",
        issuer=issuer,
        client_id="voiceclaw",
        tls=False,
    ).close()
    assert captured["insecure"] is False


def test_plaintext_oauth_issuer_accepts_only_loopback_hosts() -> None:
    assert _issuer("http://[::1]:8081/realms/voice") == "http://[::1]:8081/realms/voice"
    with pytest.raises(ConfigurationError, match="HTTPS except on loopback"):
        _issuer("http://issuer.example.com/realms/voice")


@pytest.mark.parametrize("scope", ["bad scope", 'bad"scope', r"bad\scope", "déploiement"])
def test_factory_rejects_invalid_oauth_scope_tokens(scope: str) -> None:
    profile = _oauth_profile(scopes=[scope])
    with pytest.raises(ConfigurationError, match="invalid OAuth scope"):
        build_openshell_fabric_backend(profile, {"SECRET": "secret-value"})


@pytest.mark.parametrize("status", [400, 401, 403])
def test_sdk_error_classification_limits_oauth_denial_to_token_exchange(status: int) -> None:
    import openshell

    error = openshell.SandboxError(f"OAuth client credentials exchange failed with HTTP {status}")
    assert _classify_sdk_error(error) is OpenShellFailureCode.AUTH_DENIED


@pytest.mark.parametrize(
    "message",
    [
        "OAuth discovery failed with HTTP 503",
        "OAuth discovery issuer mismatch",
        "OAuth client credentials request failed",
        "OAuth client credentials response is missing access_token",
    ],
)
def test_sdk_error_classification_does_not_mislabel_oauth_transport_failures(message: str) -> None:
    import openshell

    assert _classify_sdk_error(openshell.SandboxError(message)) is OpenShellFailureCode.UNAVAILABLE


def test_factory_rejects_remote_plaintext_and_invalid_timeouts() -> None:
    profile = _oauth_profile(endpoint="gateway.example.com:443", issuer="https://issuer.example.com")
    with pytest.raises(ConfigurationError, match="plaintext"):
        build_openshell_fabric_backend(profile, {"SECRET": "secret-value"})
    profile = _oauth_profile(
        endpoint="http://gateway.example.com:8080",
        issuer="https://issuer.example.com",
        tls=False,
    )
    with pytest.raises(ConfigurationError, match="plaintext"):
        build_openshell_fabric_backend(profile, {"SECRET": "secret-value"})
    profile = _oauth_profile(tls=None)
    with pytest.raises(ConfigurationError, match="must be a boolean"):
        build_openshell_fabric_backend(profile, {"SECRET": "secret-value"})
    profile = _oauth_profile(issuer="https://issuer.example.com", invoke_timeout_seconds=0)
    with pytest.raises(ConfigurationError, match="from 1 through 3600"):
        build_openshell_fabric_backend(profile, {"SECRET": "secret-value"})
    profile = _oauth_profile(issuer="https://issuer.example.com", rpc_timeout_seconds=31)
    with pytest.raises(ConfigurationError, match="30"):
        build_openshell_fabric_backend(profile, {"SECRET": "secret-value"})


def test_factory_fails_fast_without_an_installed_adapter_codec() -> None:
    profile = _oauth_profile(adapter_id="vendor.unknown")
    with pytest.raises(ConfigurationError, match="installed codec"):
        build_openshell_fabric_backend(profile, {"SECRET": "secret-value"})


def test_factory_validates_binding_identity_before_opening_sdk(monkeypatch) -> None:
    opened = False

    def build_executor(**_kwargs):
        nonlocal opened
        opened = True
        return _Executor([])

    monkeypatch.setattr(
        "voiceclaw.adapters.openshell_fabric.factory.SdkSandboxExecutor",
        build_executor,
    )
    profile = _oauth_profile(workspace="voice\ninvalid")

    with pytest.raises(ConfigurationError, match="workspace is invalid"):
        build_openshell_fabric_backend(profile, {"SECRET": "secret-value"})
    assert opened is False


@pytest.mark.parametrize("endpoint", ["127.0.0.1:0", "[::1]:0"])
def test_factory_rejects_zero_port(endpoint: str) -> None:
    profile = _oauth_profile(endpoint=endpoint)
    with pytest.raises(ConfigurationError, match="endpoint"):
        build_openshell_fabric_backend(profile, {"SECRET": "secret-value"})


def test_factory_closes_executor_when_adapter_construction_fails(monkeypatch) -> None:
    executor = _Executor([])
    monkeypatch.setattr(
        "voiceclaw.adapters.openshell_fabric.factory.SdkSandboxExecutor",
        lambda **_kwargs: executor,
    )

    def reject_adapter(**_kwargs):
        raise ValueError("invalid adapter")

    monkeypatch.setattr(
        "voiceclaw.adapters.openshell_fabric.factory.OpenShellFabricAdapter",
        reject_adapter,
    )
    profile = _oauth_profile()
    with pytest.raises(ConfigurationError, match="adapter settings are invalid"):
        build_openshell_fabric_backend(profile, {"SECRET": "secret-value"})
    assert executor.closed is True


def test_secret_file_reader_handles_short_reads(tmp_path, monkeypatch) -> None:
    secret_file = tmp_path / "client-secret"
    secret_file.write_text("secret-value\n", encoding="utf-8")
    secret_file.chmod(0o600)
    real_read = security.os.read

    def short_read(descriptor: int, count: int) -> bytes:
        return real_read(descriptor, min(count, 2))

    monkeypatch.setattr(security.os, "read", short_read)
    assert security._read_secret(secret_file) == "secret-value"


def test_shutdown_closes_an_idle_executor() -> None:
    executor = _Executor([])
    adapter = _adapter(executor)
    asyncio.run(adapter.shutdown())
    assert executor.closed
    with pytest.raises(OpenShellFabricError, match="adapter_closed"):
        asyncio.run(adapter.commit_turn(_request()))
    with pytest.raises(SelectedAgentReadinessError, match="selected_agent_unavailable"):
        asyncio.run(adapter.check_selected_agent())
