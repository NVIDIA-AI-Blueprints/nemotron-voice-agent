# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""Bounded application-internal execution seam for the image-owned SDK client."""

from __future__ import annotations

import asyncio
import base64
import binascii
import json
import time
from contextlib import suppress
from dataclasses import dataclass, field
from typing import Protocol

from voiceclaw.installer_inputs import (
    AgentConnection,
    ContainerInputError,
    read_protected,
    strict_json,
    validate_bearer,
)

INPUT_LIMIT = 512 * 1024
OUTPUT_LIMIT = 4 * 1024 * 1024
HELPER_LIMIT = ((OUTPUT_LIMIT + 2) // 3) * 4 + 1024
CLIENT = "/usr/local/bin/voiceclaw-openshell-exec"
OPEN_SHELL_REVISION = "6648bd0c290efbc41ba131ee9831ee45cd431f94"
_FAILURES = frozenset(
    {
        "access_denied",
        "credential_invalid",
        "target_replaced",
        "transport_unavailable",
        "outcome_unconfirmed",
        "output_limit",
        "protocol_error",
        "adapter_closed",
        "credential_disclosure",
    }
)


class TransportFailure(RuntimeError):
    """One allowlisted failure; no raw SDK diagnostic or remote output."""

    def __init__(self, code: str) -> None:
        """Preserve only the classified transport outcome."""
        self.code = code if code in _FAILURES else "transport_unavailable"
        super().__init__(self.code)


@dataclass(frozen=True, slots=True)
class ExecutionResult:
    """Completed exit status and bounded byte channels, with private repr."""

    exit_code: int
    stdout: bytes = field(repr=False)
    stderr: bytes = field(repr=False)


class ExecutionTransport(Protocol):
    """Execute argv using explicit workspace/target, stdin and total deadline."""

    async def execute(
        self, connection: AgentConnection, argv: list[str], stdin: bytes, deadline: float
    ) -> ExecutionResult:
        """Return completed execution or a classified failure without retry."""
        ...

    async def shutdown(self) -> None:
        """Close and reap local client children, without remote lifecycle calls."""
        ...


async def _read_bound(stream: asyncio.StreamReader, maximum: int) -> bytes:
    chunks = []
    size = 0
    while True:
        chunk = await stream.read(min(65536, maximum + 1 - size))
        if not chunk:
            return b"".join(chunks)
        size += len(chunk)
        if size > maximum:
            raise TransportFailure("output_limit")
        chunks.append(chunk)


async def _reap(process: asyncio.subprocess.Process) -> None:
    if process.returncode is None:
        with suppress(ProcessLookupError):
            process.kill()
    await process.wait()


def _credential_echo(payload: bytes, bearer: str) -> bool:
    if bearer.encode("ascii") in payload:
        return True
    # Bridge JSON can escape otherwise visible credential characters. Reject
    # semantic echoes before any result mapper or browser receives the bytes.
    try:
        values = [json.loads(payload.decode("utf-8"))]
    except (ValueError, UnicodeError, RecursionError):
        return False
    budget = 4 * len(payload)
    while values:
        value = values.pop()
        if isinstance(value, str):
            budget -= len(value)
            if budget < 0 or bearer in value:
                return True
            if value.lstrip().startswith(("{", "[", '"')):
                with suppress(ValueError, RecursionError):
                    values.append(json.loads(value))
        if isinstance(value, dict):
            values.extend(value.keys())
            values.extend(value.values())
        elif isinstance(value, list):
            values.extend(value)
    return False


class SDKExecutionTransport:
    """Launch the pinned upstream-SDK client with a private stdin control pipe.

    Credentials enter only the trusted child stdin, never argv/environment.
    Cancelling this await kills the local helper, not the remote invocation.
    """

    def __init__(self, *, client: str = CLIENT) -> None:
        """Use one image-owned executable; a test may supply a fixture client."""
        self._client = client
        self._children: set[asyncio.subprocess.Process] = set()
        self._tasks: set[asyncio.Task] = set()
        self._closed = False

    async def execute(
        self, connection: AgentConnection, argv: list[str], stdin: bytes, deadline: float
    ) -> ExecutionResult:
        """Bound input, stream output and the entire client exchange."""
        invoke = ["/usr/local/bin/fabric-agent", "invoke", "--agent", connection.agent, "--input", "-"]
        check = ["/usr/local/bin/fabric-agent", "check", "--agent", connection.agent, "--live"]
        if argv not in (invoke, check) or not isinstance(stdin, bytes) or len(stdin) > INPUT_LIMIT:
            raise TransportFailure("protocol_error")
        if argv == check and stdin:
            raise TransportFailure("protocol_error")
        if argv == invoke:
            try:
                strict_json(stdin, INPUT_LIMIT)
            except ContainerInputError:
                raise TransportFailure("protocol_error") from None
        if self._closed:
            raise TransportFailure("adapter_closed")
        task = asyncio.current_task()
        assert task is not None
        self._tasks.add(task)
        process = None
        io_tasks = []
        try:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TransportFailure("transport_unavailable")
            async with asyncio.timeout(remaining):
                bearer = None
                if connection.authentication_mode == "oidcBearer" and connection.credential_file is not None:
                    try:
                        bearer = validate_bearer(
                            await asyncio.to_thread(read_protected, str(connection.credential_file))
                        )
                    except ContainerInputError:
                        raise TransportFailure("credential_invalid") from None
                elif connection.authentication_mode != "none" or connection.credential_file is not None:
                    raise TransportFailure("protocol_error")
                control = json.dumps(
                    {
                        "endpoint": connection.endpoint,
                        "workspace": connection.workspace,
                        "sandbox": connection.sandbox,
                        "sandboxId": connection.sandbox_id,
                        "argv": argv,
                        "stdin": base64.b64encode(stdin).decode("ascii"),
                        "authenticationMode": connection.authentication_mode,
                        "bearer": bearer,
                        "seconds": max(0.001, deadline - time.monotonic()),
                    },
                    separators=(",", ":"),
                ).encode("utf-8")
                process = await asyncio.create_subprocess_exec(
                    self._client,
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    env={"LANG": "C.UTF-8"},
                    limit=65536,
                )
                self._children.add(process)
                if self._closed:
                    raise TransportFailure("adapter_closed")
                assert process.stdin is not None and process.stdout is not None and process.stderr is not None

                async def feed() -> None:
                    process.stdin.write(control)
                    await process.stdin.drain()
                    process.stdin.close()
                    await process.stdin.wait_closed()

                io_tasks = [
                    asyncio.create_task(feed()),
                    asyncio.create_task(_read_bound(process.stdout, HELPER_LIMIT)),
                    asyncio.create_task(_read_bound(process.stderr, 1024)),
                ]
                _, raw, diagnostics = await asyncio.gather(*io_tasks)
                await process.wait()
                # The helper emits only the fixed protocol. Unexpected stderr
                # fails closed; it is never logged or returned to the browser.
                if process.returncode != 0 or diagnostics:
                    raise TransportFailure("transport_unavailable")
                value = strict_json(raw, HELPER_LIMIT)
                if set(value) == {"failure"}:
                    raise TransportFailure(value["failure"])
                if set(value) != {"exitCode", "stdout", "stderr"} or type(value["exitCode"]) is not int:
                    raise TransportFailure("protocol_error")
                output = base64.b64decode(value["stdout"], validate=True)
                errors = base64.b64decode(value["stderr"], validate=True)
                if len(output) + len(errors) > OUTPUT_LIMIT:
                    raise TransportFailure("output_limit")
                if bearer is not None and (_credential_echo(output, bearer) or _credential_echo(errors, bearer)):
                    raise TransportFailure("credential_disclosure")
                return ExecutionResult(value["exitCode"], output, errors)
        except TransportFailure:
            raise
        except TimeoutError:
            raise TransportFailure("outcome_unconfirmed" if process is not None else "transport_unavailable") from None
        except asyncio.CancelledError:
            raise
        except (ContainerInputError, ValueError, TypeError, binascii.Error):
            raise TransportFailure("protocol_error") from None
        except Exception:
            raise TransportFailure("outcome_unconfirmed" if process is not None else "transport_unavailable") from None
        finally:
            for pending in io_tasks:
                if not pending.done():
                    pending.cancel()
            if io_tasks:
                await asyncio.gather(*io_tasks, return_exceptions=True)
            if process is not None:
                await _reap(process)
                self._children.discard(process)
            self._tasks.discard(task)

    async def shutdown(self) -> None:
        """Cancel local exchanges and reap helpers; do not assert remote stop."""
        self._closed = True
        tasks = tuple(task for task in self._tasks if task is not asyncio.current_task())
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        for child in tuple(self._children):
            await _reap(child)
