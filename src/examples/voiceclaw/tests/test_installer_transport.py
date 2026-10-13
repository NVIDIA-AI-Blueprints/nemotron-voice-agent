# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause
# ruff: noqa: D100, D101, D102, D103, D107

import asyncio
import sys
import time
from pathlib import Path

import pytest
from test_installer_inputs import connection, connection_value

from voiceclaw import installer_transport as transport
from voiceclaw.installer_transport import SDKExecutionTransport, TransportFailure


def helper(tmp_path: Path, code: str) -> str:
    path = tmp_path / "fixture-client"
    path.write_text(f"#!{sys.executable}\nimport sys, json, base64, os, time\ncontrol=json.load(sys.stdin)\n{code}\n")
    path.chmod(0o700)
    return str(path)


@pytest.fixture(autouse=True)
def fixture_credential(monkeypatch):
    monkeypatch.setattr(transport, "read_protected", lambda _: b"fixture.bearer")


def execute(client, *, seconds=1, no_auth=False):
    c = connection(connection_value(no_auth=no_auth))
    return client.execute(
        c,
        ["/usr/local/bin/fabric-agent", "invoke", "--agent", c.agent, "--input", "-"],
        b"{}",
        time.monotonic() + seconds,
    )


def test_private_pipe_carries_explicit_connection_without_inheriting_secrets(tmp_path, monkeypatch):
    monkeypatch.setenv("NVIDIA_API_KEY", "unrelated-secret-sentinel")
    monkeypatch.setenv("SSL_CERT_FILE", "/operator/unsupported.pem")
    client = SDKExecutionTransport(
        client=helper(
            tmp_path,
            """
assert len(sys.argv) == 1
assert 'NVIDIA_API_KEY' not in os.environ and 'SSL_CERT_FILE' not in os.environ
assert control['bearer'] == 'fixture.bearer'
assert control['authenticationMode'] == 'oidcBearer'
assert control['workspace'] == 'explicit-workspace'
assert control['sandboxId'] == '893ed65e-1972-48ba-b5b7-764119c24d07'
assert control['endpoint'] == 'https://openshell.example.com:8443'
assert control['argv'] == ['/usr/local/bin/fabric-agent','invoke','--agent','assistant','--input','-']
assert base64.b64decode(control['stdin']) == b'{}'
assert 0 < control['seconds'] <= 1
print(json.dumps({'exitCode':0,'stdout':base64.b64encode(b'fixture output').decode(),'stderr':''}))
""",
        )
    )
    result = asyncio.run(execute(client))
    assert result.exit_code == 0 and result.stdout == b"fixture output"
    assert "fixture output" not in repr(result)
    assert not client._children


def test_no_auth_sends_no_token_and_never_reads_openshell_credentials(tmp_path, monkeypatch):
    def forbidden_read(_):
        raise AssertionError("no OpenShell credential may be read")

    monkeypatch.setattr(transport, "read_protected", forbidden_read)
    monkeypatch.setenv("OPENSHELL_TOKEN", "unrelated-operator-sentinel")
    monkeypatch.setenv("SSL_CERT_FILE", "/operator/unsupported.pem")
    client = SDKExecutionTransport(
        client=helper(
            tmp_path,
            """
assert control['authenticationMode'] == 'none'
assert control['bearer'] is None
assert control['endpoint'] == 'http://172.18.0.2:8080'
assert control['workspace'] == 'explicit-workspace'
assert control['sandboxId'] == '893ed65e-1972-48ba-b5b7-764119c24d07'
assert 'OPENSHELL_TOKEN' not in os.environ and 'SSL_CERT_FILE' not in os.environ
assert 'credentialFile' not in control
print(json.dumps({'exitCode':0,'stdout':base64.b64encode(b'fixture output').decode(),'stderr':''}))
""",
        )
    )
    result = asyncio.run(execute(client, no_auth=True))
    assert result.exit_code == 0 and result.stdout == b"fixture output"
    assert not client._children


@pytest.mark.parametrize("no_auth", [False, True])
@pytest.mark.parametrize("code", ["access_denied", "target_replaced", "transport_unavailable", "outcome_unconfirmed"])
def test_helper_failure_is_classified_without_diagnostic_leak(tmp_path, code, capsys, no_auth):
    client = SDKExecutionTransport(client=helper(tmp_path, f"print(json.dumps({{'failure':{code!r}}}))"))
    with pytest.raises(TransportFailure, match=f"^{code}$"):
        asyncio.run(execute(client, no_auth=no_auth))
    assert capsys.readouterr() == ("", "")
    assert not client._children


@pytest.mark.parametrize(
    "reply,category",
    [
        ("print('not-json fixture.bearer')", "protocol_error"),
        ("print(json.dumps({'failure':'private-sentinel'}))", "transport_unavailable"),
        ("print(json.dumps({'exitCode':True,'stdout':'','stderr':''}))", "protocol_error"),
        ("print(json.dumps({'exitCode':0,'stdout':'%%%','stderr':''}))", "protocol_error"),
        (
            "print(json.dumps({'exitCode':0,'stdout':base64.b64encode(b'fixture.bearer').decode(),'stderr':''}))",
            "credential_disclosure",
        ),
        ("sys.stderr.write('private-sentinel')", "transport_unavailable"),
    ],
)
def test_malformed_output_and_secret_echo_cannot_become_success(tmp_path, reply, category, capsys):
    client = SDKExecutionTransport(client=helper(tmp_path, reply))
    with pytest.raises(TransportFailure, match=f"^{category}$") as error:
        asyncio.run(execute(client))
    assert "fixture.bearer" not in str(error.value) and "private-sentinel" not in str(error.value)
    assert capsys.readouterr() == ("", "")


def test_pipe_bound_stops_reading_before_end_of_unbounded_source():
    class Stream:
        consumed = 0

        async def read(self, requested):
            assert self.consumed < 101
            self.consumed += requested
            return b"a" * requested

    stream = Stream()
    with pytest.raises(TransportFailure, match="output_limit"):
        asyncio.run(transport._read_bound(stream, 100))
    assert stream.consumed == 101


def test_total_timeout_reaps_local_helper_and_preserves_uncertain_outcome(tmp_path):
    client = SDKExecutionTransport(client=helper(tmp_path, "time.sleep(30)"))
    with pytest.raises(TransportFailure, match="outcome_unconfirmed"):
        asyncio.run(execute(client, seconds=0.05))
    assert not client._children and not client._tasks


def test_shutdown_cancels_only_local_execution_and_rejects_further_calls(tmp_path):
    client = SDKExecutionTransport(client=helper(tmp_path, "time.sleep(30)"))

    async def scenario():
        active = asyncio.create_task(execute(client, seconds=30))
        async with asyncio.timeout(1):
            while not client._children:
                await asyncio.sleep(0.001)
        child = next(iter(client._children))
        await client.shutdown()
        assert active.cancelled() and child.returncode is not None
        with pytest.raises(TransportFailure, match="adapter_closed"):
            await execute(client)

    asyncio.run(scenario())
    assert not client._children


def test_credential_failure_prevents_child_launch(monkeypatch):
    def invalid(_):
        raise transport.ContainerInputError

    monkeypatch.setattr(transport, "read_protected", invalid)
    client = SDKExecutionTransport(client="/never-launched")
    with pytest.raises(TransportFailure, match="credential_invalid"):
        asyncio.run(execute(client))
    assert not client._children


@pytest.mark.parametrize("nested", [False, True])
def test_json_escaped_credential_echo_cannot_reach_result_mapper(tmp_path, nested):
    client = SDKExecutionTransport(
        client=helper(
            tmp_path,
            """
escaped=''.join('\\\\u%04x' % ord(c) for c in control['bearer'])
raw=('{"response":"'+escaped+'"}').encode()
"""
            + ("raw=json.dumps({'response':raw.decode()}).encode()\n" if nested else "")
            + """
print(json.dumps({'exitCode':0,'stdout':base64.b64encode(raw).decode(),'stderr':''}))
""",
        )
    )
    with pytest.raises(TransportFailure, match="credential_disclosure"):
        asyncio.run(execute(client))
