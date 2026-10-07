# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause
# ruff: noqa: D100, D101, D102, D103, D107

import asyncio
import json
import threading
from dataclasses import replace

import pytest
from test_installer_inputs import connection, connection_value
from test_openshell_fabric_adapter import _check_unsupported, _invoke_success, _request

from voiceclaw.adapters.openshell_fabric.client import OpenShellClientError, OpenShellFailureCode, SandboxExecutor
from voiceclaw.adapters.openshell_fabric.committed_turn import (
    FABRIC_AGENT_BINARY,
    OpenShellFabricAdapter,
    OpenShellFabricRejected,
)
from voiceclaw.domain.response_only import ResponseOnlyTargetAvailability
from voiceclaw.installer_executor import InstallerSandboxExecutor, compose_installer_backend
from voiceclaw.installer_transport import ExecutionResult, TransportFailure
from voiceclaw.model_contracts import load_model_contract_catalog


class Transport:
    def __init__(self, results):
        self.results = results
        self.calls = []
        self.closed = False

    async def execute(self, binding, argv, stdin, deadline):
        self.calls.append((binding, argv, stdin, deadline))
        result = self.results.pop(0)
        if isinstance(result, BaseException):
            raise result
        return result

    async def shutdown(self):
        self.closed = True


@pytest.mark.parametrize("no_auth", [False, True])
def test_executor_uses_upstream_protocol_and_preserves_binding(no_auth):
    binding = connection(connection_value(no_auth=no_auth))
    transport = Transport([ExecutionResult(1, b'{"status":"unsupported"}', b"")])
    executor = InstallerSandboxExecutor(binding, transport_factory=lambda: transport)
    assert isinstance(executor, SandboxExecutor)
    result = executor.execute(
        (FABRIC_AGENT_BINARY, "check", "--agent", binding.agent, "--live"),
        stdin=None,
        timeout_seconds=binding.health_seconds,
    )
    assert result.exit_code == 1 and result.stdout == '{"status":"unsupported"}'
    assert transport.calls[0][:3] == (
        binding,
        [FABRIC_AGENT_BINARY, "check", "--agent", binding.agent, "--live"],
        b"",
    )
    assert transport.closed
    executor.close()
    with pytest.raises(OpenShellClientError):
        executor.execute(
            (FABRIC_AGENT_BINARY, "check", "--agent", binding.agent, "--live"),
            stdin=None,
            timeout_seconds=1,
        )


@pytest.mark.parametrize(
    "failure,expected",
    [
        ("access_denied", OpenShellFailureCode.AUTH_DENIED),
        ("credential_invalid", OpenShellFailureCode.AUTH_DENIED),
        ("target_replaced", OpenShellFailureCode.TARGET_MISSING),
        ("outcome_unconfirmed", OpenShellFailureCode.TIMEOUT),
        ("output_limit", OpenShellFailureCode.UNAVAILABLE),
        ("protocol_error", OpenShellFailureCode.UNAVAILABLE),
    ],
)
def test_executor_maps_safe_failures_without_retries(failure, expected):
    binding = connection()
    transport = Transport([TransportFailure(failure)])
    executor = InstallerSandboxExecutor(binding, transport_factory=lambda: transport)
    with pytest.raises(OpenShellClientError) as error:
        executor.execute(
            (FABRIC_AGENT_BINARY, "invoke", "--agent", binding.agent, "--input", "-"),
            stdin=b"{}",
            timeout_seconds=1,
        )
    assert error.value.code is expected
    assert len(transport.calls) == 1 and transport.closed


@pytest.mark.parametrize("fault", ["shell", "wrong-agent", "ready", "check-input", "timeout", "bool-timeout"])
def test_executor_rejects_commands_outside_the_installer_binding(fault):
    binding = connection()
    argv = [FABRIC_AGENT_BINARY, "check", "--agent", binding.agent, "--live"]
    stdin = None
    seconds = 1
    if fault == "shell":
        argv = ["sh", "-c", "true"]
    elif fault == "wrong-agent":
        argv[3] = "another"
    elif fault == "ready":
        argv[-1] = "--ready"
    elif fault == "check-input":
        stdin = b"{}"
    elif fault == "timeout":
        seconds = binding.health_seconds + 1
    else:
        seconds = True
    transport = Transport([])
    executor = InstallerSandboxExecutor(binding, transport_factory=lambda: transport)
    with pytest.raises(OpenShellClientError):
        executor.execute(argv, stdin=stdin, timeout_seconds=seconds)
    assert transport.calls == []


def test_installer_composes_the_upstream_adapter_and_keeps_one_shot_results(monkeypatch):
    binding = replace(connection(), agent="researcher")
    check = json.loads(_check_unsupported().stdout)
    check["result"]["applied_config"]["harness"]["settings"]["agent_name"] = "researcher"
    invoke = json.loads(_invoke_success().stdout)
    invoke["result"]["fabric_result"]["output"]["session_key"] = "agent:researcher:fabric-runtime-1"
    transport = Transport(
        [
            ExecutionResult(1, (json.dumps(check) + "\n").encode(), b""),
            ExecutionResult(0, (json.dumps(invoke) + "\n").encode(), b""),
        ]
    )
    monkeypatch.setattr(
        "voiceclaw.installer_executor.InstallerSandboxExecutor",
        lambda selected: InstallerSandboxExecutor(selected, transport_factory=lambda: transport),
    )
    composition = compose_installer_backend(binding, load_model_contract_catalog())
    assert type(composition.turn_backend) is OpenShellFabricAdapter
    backend = asyncio.run(composition.turn_backend.inspect())
    assert backend.target_availability is ResponseOnlyTargetAvailability.AVAILABLE
    asyncio.run(composition.check_selected_agent_readiness())
    assert composition.turn_backend._binding_health_supported is False
    result = asyncio.run(composition.turn_backend.commit_turn(_request()))
    assert result.display_text == "## Result\n\nDone."
    assert result.speak_text == "The result is ready."
    with pytest.raises(OpenShellFabricRejected, match="target_context_consumed"):
        asyncio.run(composition.turn_backend.commit_turn(_request()))
    assert [call[1][1] for call in transport.calls] == ["check", "invoke"]
    assert json.loads(transport.calls[-1][2])["agent"] == "researcher"
    asyncio.run(composition.shutdown())


def test_executor_close_cancels_local_helper_without_waiting_for_remote_stop():
    started = threading.Event()
    finished = threading.Event()

    class WaitingTransport(Transport):
        async def execute(self, *_args):
            started.set()
            await asyncio.Event().wait()

        async def shutdown(self):
            finished.set()

    executor = InstallerSandboxExecutor(connection(), transport_factory=lambda: WaitingTransport([]))
    errors = []

    def run():
        try:
            executor.execute(
                (FABRIC_AGENT_BINARY, "invoke", "--agent", "assistant", "--input", "-"),
                stdin=b"{}",
                timeout_seconds=1,
            )
        except OpenShellClientError as error:
            errors.append(error.code)

    worker = threading.Thread(target=run, daemon=True)
    worker.start()
    assert started.wait(2)
    executor.close()
    assert executor.wait_closed(2)
    worker.join(2)
    assert not worker.is_alive() and finished.is_set()
    assert errors == [OpenShellFailureCode.UNAVAILABLE]
