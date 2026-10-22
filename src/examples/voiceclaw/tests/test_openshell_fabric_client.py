# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

# ruff: noqa: D100, D101, D102, D103

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from itertools import chain, repeat

import openshell
import pytest

from voiceclaw.adapters.openshell_fabric.client import (
    MAX_SANDBOX_STDERR_BYTES,
    MAX_SANDBOX_STDOUT_BYTES,
    OpenShellAuthenticationMode,
    OpenShellClientError,
    OpenShellFailureCode,
    OpenShellFailurePhase,
    SdkSandboxExecutor,
)


@dataclass
class _CapturedClient:
    events: Iterable[object]
    calls: list[tuple[tuple[object, ...], dict[str, object]]] = field(default_factory=list)
    closed: bool = False

    def exec_stream(self, *args: object, **kwargs: object) -> Iterable[object]:
        self.calls.append((args, kwargs))
        return self.events

    def close(self) -> None:
        self.closed = True


def _executor(
    monkeypatch: pytest.MonkeyPatch,
    events: Iterable[object],
) -> tuple[SdkSandboxExecutor, _CapturedClient]:
    captured = _CapturedClient(events)

    def client(*_args: object, **_kwargs: object) -> _CapturedClient:
        return captured

    monkeypatch.setattr(openshell, "SandboxClient", client)
    executor = SdkSandboxExecutor(
        endpoint="127.0.0.1:8080",
        workspace="voice",
        sandbox="deployed-agent",
        client_secret=lambda: "secret-value",
        issuer="http://127.0.0.1:8081/realms/voice",
        client_id="voiceclaw",
        tls=False,
    )
    return executor, captured


def _result(exit_code: int = 0) -> openshell.ExecResult:
    return openshell.ExecResult(exit_code=exit_code, stdout="ignored", stderr="ignored")


class _SdkError(RuntimeError):
    def __init__(self, code_name: str) -> None:
        self._code_name = code_name

    def code(self):
        return type("StatusCode", (), {"name": self._code_name})()


def test_anonymous_executor_omits_client_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def client(*_args: object, **kwargs: object) -> _CapturedClient:
        captured.update(kwargs)
        return _CapturedClient(())

    monkeypatch.setattr(openshell, "SandboxClient", client)
    executor = SdkSandboxExecutor(
        endpoint="127.0.0.1:8080",
        workspace="voice",
        sandbox="deployed-agent",
        authentication=OpenShellAuthenticationMode.ANONYMOUS,
        tls=False,
    )
    executor.close()
    assert captured["client_credentials"] is None


def test_exec_stream_preserves_exact_stdout_and_split_utf8(monkeypatch: pytest.MonkeyPatch) -> None:
    executor, client = _executor(
        monkeypatch,
        iter(
            [
                openshell.ExecChunk("stdout", b'{"value":"\xe2'),
                openshell.ExecChunk("stderr", b"diagnostic"),
                openshell.ExecChunk("stdout", b'\x82\xac"}\n'),
                _result(),
            ]
        ),
    )

    command = (
        "/opt/fabric/bin/fabric-agent-ctl",
        "invoke",
        "--agent",
        "researcher",
        "--runtime-id",
        "runtime-1",
        "--request",
        "-",
    )
    payload = b'{"input":"hello","request_id":"commit-1"}'
    execution = executor.execute(command, stdin=payload, timeout_seconds=330)

    assert execution.exit_code == 0
    assert execution.stdout == '{"value":"€"}\n'
    assert client.calls == [
        (
            ("deployed-agent", command),
            {
                "workspace": "voice",
                "stdin": payload,
                "timeout_seconds": 330,
                "no_login_shell": True,
            },
        )
    ]


@pytest.mark.parametrize(
    ("stream_name", "payload"),
    [
        ("stdout", b"\xff"),
        ("stdout", b"\xe2\x82"),
        ("stderr", b"\xff"),
        ("stderr", b"\xe2\x82"),
    ],
)
def test_exec_stream_rejects_malformed_or_incomplete_utf8(
    monkeypatch: pytest.MonkeyPatch,
    stream_name: str,
    payload: bytes,
) -> None:
    executor, _client = _executor(
        monkeypatch,
        iter([openshell.ExecChunk(stream_name, payload), _result()]),
    )

    with pytest.raises(OpenShellClientError) as raised:
        executor.execute(("bridge", "invoke"), stdin=None, timeout_seconds=330)

    assert raised.value.code is OpenShellFailureCode.UNAVAILABLE


@pytest.mark.parametrize(
    ("stream_name", "limit"),
    [
        ("stdout", MAX_SANDBOX_STDOUT_BYTES),
        ("stderr", MAX_SANDBOX_STDERR_BYTES),
    ],
)
def test_exec_stream_accepts_each_exact_output_limit(
    monkeypatch: pytest.MonkeyPatch,
    stream_name: str,
    limit: int,
) -> None:
    executor, _client = _executor(
        monkeypatch,
        iter([openshell.ExecChunk(stream_name, b"x" * limit), _result()]),
    )

    execution = executor.execute(("bridge", "invoke"), stdin=None, timeout_seconds=330)

    assert execution.stdout == ("x" * limit if stream_name == "stdout" else "")


@pytest.mark.parametrize(
    ("stream_name", "limit"),
    [
        ("stdout", MAX_SANDBOX_STDOUT_BYTES),
        ("stderr", MAX_SANDBOX_STDERR_BYTES),
    ],
)
def test_exec_stream_rejects_output_over_each_limit_and_closes_the_iterator(
    monkeypatch: pytest.MonkeyPatch,
    stream_name: str,
    limit: int,
) -> None:
    closed = False

    def events():
        nonlocal closed
        try:
            yield openshell.ExecChunk(stream_name, b"x" * limit)
            yield openshell.ExecChunk(stream_name, b"x")
            yield _result()
        finally:
            closed = True

    executor, _client = _executor(monkeypatch, events())

    with pytest.raises(OpenShellClientError) as raised:
        executor.execute(("bridge", "invoke"), stdin=None, timeout_seconds=330)

    assert raised.value.code is OpenShellFailureCode.UNAVAILABLE
    assert closed is True


def test_exec_stream_bounds_empty_output_chunks(monkeypatch: pytest.MonkeyPatch) -> None:
    chunks = repeat(openshell.ExecChunk("stdout", b""), 16_385)
    executor, _client = _executor(monkeypatch, chain(chunks, [_result()]))

    with pytest.raises(OpenShellClientError) as raised:
        executor.execute(("bridge", "invoke"), stdin=None, timeout_seconds=330)

    assert raised.value.code is OpenShellFailureCode.UNAVAILABLE


@pytest.mark.parametrize(
    "events",
    [
        [],
        [openshell.ExecChunk("other", b"value"), _result()],
        [openshell.ExecChunk("stdout", "not-bytes"), _result()],
        [object(), _result()],
        [_result(), openshell.ExecChunk("stdout", b"late")],
        [_result(), _result()],
        [openshell.ExecResult(exit_code=True, stdout="", stderr="")],
        [openshell.ExecResult(exit_code="0", stdout="", stderr="")],
    ],
)
def test_exec_stream_rejects_malformed_event_sequences(
    monkeypatch: pytest.MonkeyPatch,
    events: list[object],
) -> None:
    executor, _client = _executor(monkeypatch, iter(events))

    with pytest.raises(OpenShellClientError) as raised:
        executor.execute(("bridge", "invoke"), stdin=None, timeout_seconds=330)

    assert raised.value.code is OpenShellFailureCode.UNAVAILABLE


def test_exec_stream_preserves_sdk_timeout_classification(monkeypatch: pytest.MonkeyPatch) -> None:
    def events():
        raise TimeoutError
        yield

    executor, _client = _executor(monkeypatch, events())

    with pytest.raises(OpenShellClientError) as raised:
        executor.execute(("bridge", "invoke"), stdin=None, timeout_seconds=330)

    assert raised.value.code is OpenShellFailureCode.TIMEOUT
    assert raised.value.phase is OpenShellFailurePhase.ADMITTED_STREAM


@pytest.mark.parametrize(
    ("code_name", "expected"),
    [
        ("PERMISSION_DENIED", OpenShellFailureCode.AUTH_DENIED),
        ("NOT_FOUND", OpenShellFailureCode.TARGET_MISSING),
    ],
)
def test_exec_stream_classifies_synchronous_rejection_before_admission(
    monkeypatch: pytest.MonkeyPatch,
    code_name: str,
    expected: OpenShellFailureCode,
) -> None:
    class RejectingClient(_CapturedClient):
        def exec_stream(self, *args: object, **kwargs: object) -> Iterable[object]:
            raise _SdkError(code_name)

    monkeypatch.setattr(openshell, "SandboxClient", lambda *_args, **_kwargs: RejectingClient(()))
    executor = SdkSandboxExecutor(
        endpoint="127.0.0.1:8080",
        workspace="voice",
        sandbox="deployed-agent",
        client_secret=lambda: "secret-value",
        issuer="http://127.0.0.1:8081/realms/voice",
        client_id="voiceclaw",
        tls=False,
    )

    with pytest.raises(OpenShellClientError) as raised:
        executor.execute(("bridge", "invoke"), stdin=None, timeout_seconds=330)

    assert raised.value.code is expected
    assert raised.value.phase is OpenShellFailurePhase.BEFORE_ADMISSION


@pytest.mark.parametrize(
    ("code_name", "expected"),
    [
        ("PERMISSION_DENIED", OpenShellFailureCode.AUTH_DENIED),
        ("NOT_FOUND", OpenShellFailureCode.TARGET_MISSING),
    ],
)
def test_exec_stream_marks_failures_while_consuming_the_stream_as_admitted(
    monkeypatch: pytest.MonkeyPatch,
    code_name: str,
    expected: OpenShellFailureCode,
) -> None:
    def events():
        raise _SdkError(code_name)
        yield

    executor, _client = _executor(monkeypatch, events())

    with pytest.raises(OpenShellClientError) as raised:
        executor.execute(("bridge", "invoke"), stdin=None, timeout_seconds=330)

    assert raised.value.code is expected
    assert raised.value.phase is OpenShellFailurePhase.ADMITTED_STREAM
