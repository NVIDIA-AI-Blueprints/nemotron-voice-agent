# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""Small injectable boundary around the OpenShell Python SDK."""

from __future__ import annotations

import codecs
import re
from collections.abc import Callable, Sequence
from contextlib import suppress
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

_OAUTH_DENIED = re.compile(r"OAuth client credentials exchange failed with HTTP (?:400|401|403)")
MAX_SANDBOX_STDOUT_BYTES = 4 * 1024 * 1024
MAX_SANDBOX_STDERR_BYTES = 1024 * 1024
_MAX_SANDBOX_OUTPUT_CHUNKS = 16_384


class _SandboxOutputError(ValueError):
    """The sandbox stream violated its bounded output contract."""


class OpenShellFailureCode(StrEnum):
    """Safe failure classes exposed by the SDK boundary."""

    AUTH_DENIED = "auth_denied"
    TARGET_MISSING = "target_missing"
    TIMEOUT = "timeout"
    UNAVAILABLE = "openshell_unavailable"


class OpenShellAuthenticationMode(StrEnum):
    """Authentication modes supported by the OpenShell SDK boundary."""

    ANONYMOUS = "anonymous"
    CLIENT_CREDENTIALS = "client_credentials"


class OpenShellClientError(RuntimeError):
    """An OpenShell call failed without exposing provider diagnostics."""

    def __init__(self, code: OpenShellFailureCode | str) -> None:
        """Retain one allowlisted failure code."""
        try:
            self.code = OpenShellFailureCode(code)
        except (TypeError, ValueError):
            self.code = OpenShellFailureCode.UNAVAILABLE
        super().__init__(self.code.value)


@dataclass(frozen=True, slots=True)
class SandboxExecution:
    """Bounded information returned from one sandbox command."""

    exit_code: int
    stdout: str

    def __post_init__(self) -> None:
        """Reject malformed SDK results."""
        if isinstance(self.exit_code, bool) or not isinstance(self.exit_code, int):
            raise TypeError("exit_code must be an integer")
        if not isinstance(self.stdout, str):
            raise TypeError("stdout must be a string")


@runtime_checkable
class SandboxExecutor(Protocol):
    """Execute fixed argument vectors in one configured sandbox."""

    def execute(
        self,
        command: Sequence[str],
        *,
        stdin: bytes | None,
        timeout_seconds: int,
    ) -> SandboxExecution:
        """Execute one command without a login shell."""
        ...

    def close(self) -> None:
        """Release the underlying transport."""
        ...


class SdkSandboxExecutor:
    """OpenShell SDK v0.1.2 executor bound to one workspace and sandbox."""

    def __init__(
        self,
        *,
        endpoint: str,
        workspace: str,
        sandbox: str,
        authentication: OpenShellAuthenticationMode | str = OpenShellAuthenticationMode.CLIENT_CREDENTIALS,
        client_secret: Callable[[], str] | None = None,
        issuer: str | None = None,
        client_id: str | None = None,
        scopes: Sequence[str] = (),
        audience: str | None = None,
        tls: bool = True,
        tls_ca_file: Path | None = None,
        rpc_timeout_seconds: float = 30.0,
    ) -> None:
        """Create the SDK client with one explicitly selected authentication mode."""
        try:
            from openshell import ClientCredentialsAuth, ExecChunk, ExecResult, SandboxClient, TlsConfig
        except ImportError as error:
            raise OpenShellClientError(OpenShellFailureCode.UNAVAILABLE) from error
        try:
            mode = OpenShellAuthenticationMode(authentication)
            if mode is OpenShellAuthenticationMode.CLIENT_CREDENTIALS:
                if client_secret is None or issuer is None or client_id is None:
                    raise ValueError("client credentials authentication is incomplete")
                credentials = ClientCredentialsAuth(
                    client_secret=client_secret,
                    issuer=issuer,
                    client_id=client_id,
                    scopes=tuple(scopes) or None,
                    audience=audience,
                    insecure=False,
                    timeout=rpc_timeout_seconds,
                )
            else:
                if (
                    client_secret is not None
                    or issuer is not None
                    or client_id is not None
                    or scopes
                    or audience is not None
                ):
                    raise ValueError("anonymous authentication cannot include OAuth settings")
                credentials = None
            tls_config = TlsConfig(ca_path=tls_ca_file) if tls else None
            self._client: Any = SandboxClient(
                endpoint,
                tls=tls_config,
                client_credentials=credentials,
                timeout=rpc_timeout_seconds,
            )
        except Exception as error:
            raise OpenShellClientError(_classify_sdk_error(error)) from error
        self._exec_chunk_type = ExecChunk
        self._exec_result_type = ExecResult
        self._workspace = workspace
        self._sandbox = sandbox

    def execute(
        self,
        command: Sequence[str],
        *,
        stdin: bytes | None,
        timeout_seconds: int,
    ) -> SandboxExecution:
        """Run one literal argv through ``ExecSandbox``."""
        if not command or any(not isinstance(item, str) or not item or "\x00" in item for item in command):
            raise ValueError("command must contain non-empty strings without NUL")
        try:
            stream = self._client.exec_stream(
                self._sandbox,
                tuple(command),
                workspace=self._workspace,
                stdin=stdin,
                timeout_seconds=timeout_seconds,
                no_login_shell=True,
            )
            return self._consume_stream(stream)
        except OpenShellClientError:
            raise
        except Exception as error:
            raise OpenShellClientError(_classify_sdk_error(error)) from error

    def _consume_stream(self, stream: Any) -> SandboxExecution:
        """Decode one bounded SDK stream without trusting its lossy aggregate."""
        stdout_decoder = codecs.getincrementaldecoder("utf-8")(errors="strict")
        stderr_decoder = codecs.getincrementaldecoder("utf-8")(errors="strict")
        stdout_parts: list[str] = []
        stdout_bytes = 0
        stderr_bytes = 0
        output_chunks = 0
        exit_code: int | None = None
        try:
            for item in stream:
                if exit_code is not None:
                    raise _SandboxOutputError
                if isinstance(item, self._exec_chunk_type):
                    if not isinstance(item.data, bytes):
                        raise _SandboxOutputError
                    output_chunks += 1
                    if output_chunks > _MAX_SANDBOX_OUTPUT_CHUNKS:
                        raise _SandboxOutputError
                    if item.stream == "stdout":
                        stdout_bytes += len(item.data)
                        if stdout_bytes > MAX_SANDBOX_STDOUT_BYTES:
                            raise _SandboxOutputError
                        decoded = stdout_decoder.decode(item.data, final=False)
                        if decoded:
                            stdout_parts.append(decoded)
                    elif item.stream == "stderr":
                        stderr_bytes += len(item.data)
                        if stderr_bytes > MAX_SANDBOX_STDERR_BYTES:
                            raise _SandboxOutputError
                        stderr_decoder.decode(item.data, final=False)
                    else:
                        raise _SandboxOutputError
                    continue
                if not isinstance(item, self._exec_result_type):
                    raise _SandboxOutputError
                exit_code = item.exit_code
            if exit_code is None:
                raise _SandboxOutputError
            stdout_parts.append(stdout_decoder.decode(b"", final=True))
            stderr_decoder.decode(b"", final=True)
            return SandboxExecution(exit_code=exit_code, stdout="".join(stdout_parts))
        except (UnicodeDecodeError, _SandboxOutputError) as error:
            raise OpenShellClientError(OpenShellFailureCode.UNAVAILABLE) from error
        finally:
            close = getattr(stream, "close", None)
            if callable(close):
                with suppress(Exception):
                    close()

    def close(self) -> None:
        """Close the SDK channel without surfacing provider details."""
        try:
            self._client.close()
        except Exception as error:
            raise OpenShellClientError(_classify_sdk_error(error)) from error


def _classify_sdk_error(error: BaseException) -> OpenShellFailureCode:
    if isinstance(error, TimeoutError):
        return OpenShellFailureCode.TIMEOUT
    code_method = getattr(error, "code", None)
    try:
        code = code_method() if callable(code_method) else None
    except Exception:
        code = None
    name = getattr(code, "name", "")
    if name in {"UNAUTHENTICATED", "PERMISSION_DENIED"}:
        return OpenShellFailureCode.AUTH_DENIED
    if name == "NOT_FOUND":
        return OpenShellFailureCode.TARGET_MISSING
    if name == "DEADLINE_EXCEEDED":
        return OpenShellFailureCode.TIMEOUT
    if type(error).__name__ == "SandboxError" and _OAUTH_DENIED.fullmatch(str(error)) is not None:
        return OpenShellFailureCode.AUTH_DENIED
    return OpenShellFailureCode.UNAVAILABLE


__all__ = [
    "MAX_SANDBOX_STDERR_BYTES",
    "MAX_SANDBOX_STDOUT_BYTES",
    "OpenShellAuthenticationMode",
    "OpenShellClientError",
    "OpenShellFailureCode",
    "SandboxExecution",
    "SandboxExecutor",
    "SdkSandboxExecutor",
]
