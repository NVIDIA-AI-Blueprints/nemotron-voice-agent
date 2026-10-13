# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause
# ruff: noqa: D100, D101, D102, D103, D107

import errno
import json
import os
from pathlib import Path

import pytest

from voiceclaw.installer_inputs import (
    ContainerInputError,
    ProfileInputs,
    parse_connection,
    protected_directory,
    read_protected,
    validate_bearer,
)


def connection_value(*, no_auth=False):
    value = {
        "schemaVersion": "nemoclaw.agent-connection.v1",
        "deploymentUid": "3f3bcfca-27c7-4ed6-9d7e-2cc3918b1252",
        "service": "voice",
        "gateway": {"endpoint": "https://openshell.example.com:8443", "tls": {"trust": "system", "caFile": None}},
        "authentication": {
            "mode": "oidcBearer",
            "credentialFile": "/var/lib/voiceclaw/credentials/openshell",
            "refreshMode": "none",
        },
        "target": {
            "workspace": "explicit-workspace",
            "sandbox": "assistant",
            "sandboxId": "893ed65e-1972-48ba-b5b7-764119c24d07",
            "agent": "assistant",
        },
        "bridge": {"interfaceVersion": 1, "command": "fabric-agent"},
        "timeouts": {"healthSeconds": 12, "invokeSeconds": 120},
    }

    if no_auth:
        value["gateway"] = {"endpoint": "http://172.18.0.2:8080", "tls": {"trust": "none", "caFile": None}}
        value["authentication"] = {"mode": "none", "credentialFile": None, "refreshMode": "none"}
    return value


def connection(value=None):
    return parse_connection(json.dumps(connection_value() if value is None else value).encode())


def test_valid_exact_connection_and_profile():
    c = connection()
    assert c.workspace == "explicit-workspace"
    assert c.sandbox_id == connection_value()["target"]["sandboxId"]
    assert c.health_seconds == 12 and c.invoke_seconds == 120
    p = ProfileInputs.from_environment(
        {
            "VOICECLAW_RUNTIME_PROFILE": "nemoclaw-container-v1",
            "VOICECLAW_INSTALL_CONTRACT": "voiceclaw.nemoclaw.container.v1",
            "VOICECLAW_AGENT_CONNECTION_FILE": "/var/lib/voiceclaw/config/agent-connection.json",
            "VOICECLAW_SPEECH_CREDENTIAL_FILE": "/var/lib/voiceclaw/credentials/speech",
            "VOICECLAW_STATE_PATH": "/var/lib/voiceclaw/state/state.db",
        }
    )
    assert p.state == Path("/var/lib/voiceclaw/state/state.db")
    with pytest.raises(ContainerInputError):
        ProfileInputs.from_environment({"VOICECLAW_RUNTIME_PROFILE": "nemoclaw-container-v1"})


@pytest.mark.parametrize("no_auth", [False, True])
def test_explicit_transport_and_credential_selection(no_auth):
    c = connection(connection_value(no_auth=no_auth))
    assert c.authentication_mode == ("none" if no_auth else "oidcBearer")
    assert c.credential_file == (None if no_auth else Path("/var/lib/voiceclaw/credentials/openshell"))
    assert c.workspace == "explicit-workspace" and c.agent == "assistant"


@pytest.mark.parametrize(
    "no_auth,path,bad",
    [
        (True, ("authentication", "mode"), "oidcBearer"),
        (True, ("authentication", "credentialFile"), "/var/lib/voiceclaw/credentials/openshell"),
        (True, ("authentication", "refreshMode"), "automatic"),
        (True, ("gateway", "tls", "trust"), "system"),
        (True, ("gateway", "tls", "caFile"), "/var/lib/voiceclaw/config/ca.pem"),
        (True, ("gateway", "endpoint"), "https://172.18.0.2:8080"),
        (False, ("authentication", "mode"), "none"),
        (False, ("authentication", "credentialFile"), None),
        (False, ("gateway", "tls", "trust"), "none"),
        (False, ("gateway", "endpoint"), "http://172.18.0.2:8080"),
    ],
)
def test_rejects_inconsistent_authentication_and_transport(no_auth, path, bad):
    value = connection_value(no_auth=no_auth)
    nested = value
    for key in path[:-1]:
        nested = nested[key]
    nested[path[-1]] = bad
    with pytest.raises(ContainerInputError, match="^container-input-invalid$"):
        connection(value)


@pytest.mark.parametrize(
    "endpoint",
    [
        "http://8.8.8.8:8080",
        "http://100.64.0.1:8080",
        "http://198.18.0.1:8080",
        "http://169.254.0.1:8080",
        "http://127.0.0.1:8080",
        "http://0.0.0.0:8080",
        "http://224.0.0.1:8080",
        "http://localhost:8080",
        "http://openshell.example.com:8080",
        "http://gateway:8080",
        "http://[::1]:8080",
        "http://[fd00::1]:8080",
        "http://[::ffff:172.18.0.2]:8080",
        "http://0xac120002:8080",
        "http://172.18.2:8080",
        "http://172.018.0.2:8080",
        "http://172.18.0.2",
        "http://172.18.0.2:0",
        "http://172.18.0.2:080",
        "HTTP://172.18.0.2:8080",
        "http://172.18.0.2:65536",
        "http://172.18.0.2:8080/path",
        "http://172.18.0.2:8080?token=x",
        "http://172.18.0.2:8080#x",
        "http://user:secret@172.18.0.2:8080",
        "http://172.18.0.2:8080\n",
    ],
)
def test_no_auth_rejects_nonprivate_or_ambiguous_endpoints(endpoint):
    value = connection_value(no_auth=True)
    value["gateway"]["endpoint"] = endpoint
    with pytest.raises(ContainerInputError):
        connection(value)


@pytest.mark.parametrize("endpoint", ["http://10.23.1.2:8080", "http://172.18.0.2:8080", "http://192.168.50.2:8080/"])
def test_no_auth_accepts_canonical_rfc1918_literals(endpoint):
    value = connection_value(no_auth=True)
    value["gateway"]["endpoint"] = endpoint
    assert connection(value).endpoint == endpoint


@pytest.mark.parametrize(
    "path,bad",
    [
        (("schemaVersion",), "unknown"),
        (("deploymentUid",), "<deployment UUID>"),
        (("service",), ""),
        (("target", "workspace"), ""),
        (("target", "sandboxId"), "<sandbox UUID>"),
        (("target", "agent"), "../assistant"),
        (("target", "sandbox"), "a/b"),
        (("bridge", "interfaceVersion"), True),
        (("bridge", "command"), "sh"),
        (("authentication", "mode"), "mTLS"),
        (("authentication", "refreshMode"), "automatic"),
        (("authentication", "credentialFile"), "/var/lib/voiceclaw/credentials/../openshell"),
        (("authentication", "credentialFile"), "/etc/openshell"),
        (("gateway", "tls", "trust"), "insecure"),
        (("gateway", "tls", "trust"), "privateCa"),
        (("gateway", "tls", "caFile"), "/etc/ssl/ca.pem"),
        (("timeouts", "healthSeconds"), 13),
        (("timeouts", "invokeSeconds"), True),
        (("timeouts", "invokeSeconds"), 121),
    ],
)
def test_rejects_invalid_nested_fields(path, bad):
    value = connection_value()
    nested = value
    for key in path[:-1]:
        nested = nested[key]
    nested[path[-1]] = bad
    with pytest.raises(ContainerInputError, match="^container-input-invalid$"):
        connection(value)


@pytest.mark.parametrize(
    "url",
    [
        "http://openshell.example.com:8443",
        "https://user:secret@openshell.example.com:8443",
        "https://localhost:8443",
        "https://127.0.0.1:8443",
        "https://[::1]:8443",
        "https://[::ffff:127.0.0.1]:8443",
        "https://[::ffff:0.0.0.0]:8443",
        "https://0x7f000001:8443",
        "https://0x7f.0.0.1:8443",
        "https://0.0.0.0:8443",
        "https://openshell.example.com:8443/#secret",
        "https://openshell.example.com:8443?token=secret",
        "https://openshell.example.com:8443/path",
        "https://openshell.example.com:8443\n",
        "https://<host>:8443",
    ],
)
def test_rejects_unsafe_endpoint(url):
    value = connection_value()
    value["gateway"]["endpoint"] = url
    with pytest.raises(ContainerInputError):
        connection(value)


@pytest.mark.parametrize("raw", [b"[]", b"{} {}", b"\xff", b'{"x":NaN}', b" " * 65537])
def test_rejects_invalid_descriptor_bytes(raw):
    with pytest.raises(ContainerInputError):
        parse_connection(raw)


@pytest.mark.parametrize("field", list(connection_value()))
def test_rejects_missing_or_unknown_fields(field):
    value = connection_value()
    del value[field]
    with pytest.raises(ContainerInputError):
        connection(value)
    value = connection_value()
    value["unexpected"] = "unknown"
    with pytest.raises(ContainerInputError):
        connection(value)


def test_rejects_duplicate_keys_and_legacy():
    raw = json.dumps(connection_value()).replace(
        '"interfaceVersion": 1', '"interfaceVersion": 1, "interfaceVersion": 1'
    )
    with pytest.raises(ContainerInputError):
        parse_connection(raw.encode())
    value = connection_value()
    value["target"]["unexpected"] = "secret-sentinel"
    with pytest.raises(ContainerInputError) as error:
        connection(value)
    assert "secret-sentinel" not in str(error.value)
    with pytest.raises(ContainerInputError):
        connection({"integration": "voice", "agentRouteHost": "legacy.localhost"})


def no_acl(*_args):
    raise OSError(errno.ENODATA, "no ACL")


@pytest.fixture
def protected(tmp_path, monkeypatch):
    monkeypatch.setattr(os, "getxattr", no_acl, raising=False)
    root = tmp_path / "voiceclaw"
    root.mkdir(mode=0o700)
    leaf = root / "credentials" / "openshell"
    leaf.parent.mkdir(mode=0o700)
    leaf.write_bytes(b"short-valid-bearer")
    leaf.chmod(0o600)
    return root, leaf


def read_test(root, leaf):
    return read_protected(str(leaf), root=root, uid=os.geteuid(), gid=os.getegid(), maximum=65536)


def test_protected_read_and_protocol_specific_validation(protected):
    root, leaf = protected
    assert read_test(root, leaf) == b"short-valid-bearer"
    assert validate_bearer(b"ab") == "ab"
    for value in (b"a\rb", b"a\nb", b"a b", b"\x00", b"", b"\xff"):
        with pytest.raises(ContainerInputError):
            validate_bearer(value)


@pytest.mark.parametrize(
    "fault",
    ["leaf_mode", "parent_mode", "symlink", "directory", "oversized", "owner", "acl", "unstable", "acl_unavailable"],
)
def test_protected_file_rejections(protected, monkeypatch, fault):
    root, leaf = protected
    if fault == "leaf_mode":
        leaf.chmod(0o644)
    elif fault == "parent_mode":
        leaf.parent.chmod(0o755)
    elif fault == "symlink":
        leaf.unlink()
        leaf.symlink_to(root / "other")
    elif fault == "directory":
        leaf.unlink()
        leaf.mkdir()
    elif fault == "oversized":
        leaf.write_bytes(b"x" * 65537)
    elif fault == "owner":
        with pytest.raises(ContainerInputError):
            read_protected(str(leaf), root=root, uid=os.geteuid() + 1, gid=os.getegid(), maximum=65536)
        return
    elif fault == "acl":
        monkeypatch.setattr(os, "getxattr", lambda *_args: b"acl")
    elif fault == "acl_unavailable":
        monkeypatch.delattr(os, "getxattr")
    else:
        original = os.read

        def unstable(fd, size):
            result = original(fd, size)
            leaf.write_bytes(b"changed-sentinel")
            return result

        monkeypatch.setattr(os, "read", unstable)
    with pytest.raises(ContainerInputError):
        read_test(root, leaf)


def test_bound_short_reads_and_traversal(protected, monkeypatch):
    root, leaf = protected
    original = os.read
    monkeypatch.setattr(os, "read", lambda fd, size: original(fd, min(3, size)))
    assert read_test(root, leaf) == b"short-valid-bearer"
    with pytest.raises(ContainerInputError):
        read_test(root, str(leaf.parent) + "/../credentials/openshell")


def test_unrelated_ancestor_content_changes_do_not_replace_protected_binding(protected):
    root, leaf = protected
    with protected_directory(root, root=root, uid=os.geteuid(), gid=os.getegid()):
        unrelated = root.parent / "unrelated-sibling"
        unrelated.write_text("fixture")
    assert read_test(root, leaf) == b"short-valid-bearer"


@pytest.mark.parametrize("mutable", [False, True])
def test_ancestor_path_replacement_is_rejected_even_when_contents_may_change(protected, mutable):
    root, _leaf = protected
    original = root.parent / "original-mount"
    with (
        pytest.raises(ContainerInputError),
        protected_directory(root, root=root, uid=os.geteuid(), gid=os.getegid(), mutable=mutable),
    ):
        root.rename(original)
        root.mkdir(mode=0o700)
