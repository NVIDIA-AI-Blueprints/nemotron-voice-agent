# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""Bind the container environment settings to the upstream Fabric adapter."""

from __future__ import annotations

import asyncio
import threading
import time
from collections.abc import Callable, Sequence
from contextlib import suppress

from voiceclaw.adapters.openshell_fabric.client import (
    OpenShellClientError,
    OpenShellFailureCode,
    SandboxExecution,
)
from voiceclaw.adapters.openshell_fabric.committed_turn import FABRIC_AGENT_BINARY, OpenShellFabricAdapter
from voiceclaw.backends import BackendComposition
from voiceclaw.installer_inputs import AgentConnection
from voiceclaw.installer_transport import SDKExecutionTransport, TransportFailure
from voiceclaw.model_contracts import ModelContractCatalog

_CLOSE_POLL_SECONDS = 0.05


class InstallerSandboxExecutor:
    """Supply the upstream executor protocol without operator credential discovery.

    The SDK helper preserves the configured static bearer or explicit private
    HTTP mode and verifies the physical sandbox identity before every command.
    Upstream owns bridge decoding, readiness, context, and invocation semantics.
    """

    def __init__(
        self,
        connection: AgentConnection,
        *,
        transport_factory: Callable[[], SDKExecutionTransport] = SDKExecutionTransport,
    ) -> None:
        """Bind one connection and track only local client tasks for cleanup."""
        self._connection = connection
        self._transport_factory = transport_factory
        self._lock = threading.Lock()
        self._closed = False
        self._active: set[tuple[asyncio.AbstractEventLoop, asyncio.Task]] = set()
        self._idle = threading.Event()
        self._idle.set()

    def execute(
        self,
        command: Sequence[str],
        *,
        stdin: bytes | None,
        timeout_seconds: int,
    ) -> SandboxExecution:
        """Adapt one bounded helper exchange on the upstream worker thread."""
        invoke = (
            FABRIC_AGENT_BINARY,
            "invoke",
            "--agent",
            self._connection.agent,
            "--runtime-id",
            self._connection.fabric_runtime_id,
            "--request",
            "-",
        )
        argv = tuple(command)
        if (
            argv != invoke
            or type(timeout_seconds) is not int
            or not 1 <= timeout_seconds <= self._connection.invoke_seconds
            or not isinstance(stdin, bytes)
        ):
            raise OpenShellClientError(OpenShellFailureCode.UNAVAILABLE)

        async def exchange():
            loop = asyncio.get_running_loop()
            task = asyncio.current_task()
            assert task is not None
            with self._lock:
                if self._closed:
                    raise TransportFailure("adapter_closed")
                self._active.add((loop, task))
                self._idle.clear()
            transport = self._transport_factory()
            operation: asyncio.Task | None = None
            try:
                operation = asyncio.create_task(
                    transport.execute(self._connection, list(argv), stdin or b"", time.monotonic() + timeout_seconds)
                )
                while True:
                    done, _ = await asyncio.wait({operation}, timeout=_CLOSE_POLL_SECONDS)
                    if operation in done:
                        return await operation
                    with self._lock:
                        if self._closed:
                            raise asyncio.CancelledError
            finally:
                try:
                    if operation is not None and not operation.done():
                        operation.cancel()
                        await asyncio.gather(operation, return_exceptions=True)
                    await transport.shutdown()
                finally:
                    with self._lock:
                        self._active.discard((loop, task))
                        if not self._active:
                            self._idle.set()

        try:
            result = asyncio.run(exchange())
            return SandboxExecution(exit_code=result.exit_code, stdout=result.stdout.decode("utf-8", errors="strict"))
        except TransportFailure as error:
            code = {
                "access_denied": OpenShellFailureCode.AUTH_DENIED,
                "target_replaced": OpenShellFailureCode.TARGET_MISSING,
                "outcome_unconfirmed": OpenShellFailureCode.TIMEOUT,
            }.get(error.code, OpenShellFailureCode.UNAVAILABLE)
            raise OpenShellClientError(code) from None
        except (asyncio.CancelledError, UnicodeError, ValueError, OSError):
            raise OpenShellClientError(OpenShellFailureCode.UNAVAILABLE) from None

    def close(self) -> None:
        """Cancel local helper tasks; cancellation does not confirm remote stop."""
        with self._lock:
            self._closed = True
            active = tuple(self._active)
        for loop, task in active:
            with suppress(RuntimeError):
                loop.call_soon_threadsafe(task.cancel)

    def wait_closed(self, timeout_seconds: float = 2) -> bool:
        """Wait a bounded interval for cancelled local helpers to be reaped."""
        return self._idle.wait(timeout_seconds)


def compose_installer_backend(
    connection: AgentConnection,
    contracts: ModelContractCatalog,
    *,
    executor: InstallerSandboxExecutor | None = None,
) -> BackendComposition:
    """Bind VoiceClaw to the exact runtime NemoClaw already provisioned."""
    executor = executor or InstallerSandboxExecutor(connection)
    try:
        adapter = OpenShellFabricAdapter(
            executor=executor,
            workspace=connection.workspace,
            sandbox=connection.sandbox,
            fabric_agent=connection.agent,
            runtime_id=connection.fabric_runtime_id,
            adapter_id=connection.fabric_adapter_id,
            harness=connection.fabric_harness,
            adapter_kind=connection.fabric_adapter_kind,
            result_pointer=connection.fabric_result_pointer,
            invoke_timeout_seconds=connection.invoke_seconds,
            model_contracts=contracts,
        )
        return BackendComposition(turn_backend=adapter, turn_status="response_only")
    except BaseException:
        executor.close()
        raise
