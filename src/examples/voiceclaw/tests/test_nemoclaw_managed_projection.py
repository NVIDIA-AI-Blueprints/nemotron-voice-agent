# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

# ruff: noqa: D100, D103

from __future__ import annotations

import errno
import json
import os
from pathlib import Path
from typing import Any

import pytest
import yaml

import voiceclaw.adapters.nemoclaw.managed_projection as managed_projection_module
import voiceclaw.adapters.nemoclaw.security as managed_security_module
from voiceclaw.adapters.nemoclaw.managed_projection import (
    MANAGED_LISTENER_PORT,
    ManagedProjectionError,
    ManagedProjectionPaths,
    load_managed_projection,
    materialize_managed_config,
)
from voiceclaw.adapters.nemoclaw.security import (
    ManagedServiceSecurityError,
    read_managed_credential,
    validate_managed_http_origin,
    validate_managed_route_host,
)

_SPEECH_CREDENTIAL = "nvapi-speech-" + "s" * 40
_AGENT_CREDENTIAL = "nemoclaw-agent-" + "a" * 40


def _projection_value(paths: ManagedProjectionPaths) -> dict[str, Any]:
    return {
        "integration": "voiceclaw",
        "sandbox": "sandbox-one",
        "agent": "research-agent",
        "port": MANAGED_LISTENER_PORT,
        "speechProvider": "nvidia",
        "speechCredentialPath": str(paths.speech_credential),
        "agentCredentialPath": str(paths.agent_credential),
        "agentEndpoint": "http://10.42.0.17:8652",
        "agentRouteHost": "research-agent.sandbox-one.openshell.localhost:8652",
    }


def _write_file(path: Path, payload: str | bytes, *, mode: int = 0o600) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    path.parent.chmod(0o700)
    path.write_bytes(payload.encode() if isinstance(payload, str) else payload)
    path.chmod(mode)


def _write_projection(
    tmp_path: Path,
    *,
    value: dict[str, Any] | None = None,
    raw_configuration: str | None = None,
) -> tuple[ManagedProjectionPaths, dict[str, Any]]:
    root = tmp_path / "voiceclaw"
    root.mkdir(mode=0o700, parents=True)
    root.chmod(0o700)
    paths = ManagedProjectionPaths(root=root)
    selected = _projection_value(paths) if value is None else value
    configuration = json.dumps(selected, separators=(",", ":")) if raw_configuration is None else raw_configuration
    _write_file(paths.configuration, configuration)
    _write_file(paths.speech_credential, _SPEECH_CREDENTIAL)
    _write_file(paths.agent_credential, _AGENT_CREDENTIAL)
    return paths, selected


def _assert_invalid(paths: ManagedProjectionPaths) -> None:
    with pytest.raises(ManagedProjectionError, match="^projection-invalid$"):
        load_managed_projection(paths)


def test_loads_exact_projection_and_credentials_stably(tmp_path: Path) -> None:
    paths, _value = _write_projection(tmp_path)

    first = load_managed_projection(paths)
    second = load_managed_projection(paths)

    assert first == second
    assert first.integration == "voiceclaw"
    assert first.sandbox == "sandbox-one"
    assert first.agent == "research-agent"
    assert first.agent_endpoint == "http://10.42.0.17:8652"
    assert first.agent_route_host == "research-agent.sandbox-one.openshell.localhost:8652"
    assert len(first.projection_digest) == 64
    assert read_managed_credential(paths.speech_credential) == _SPEECH_CREDENTIAL
    assert read_managed_credential(paths.agent_credential) == _AGENT_CREDENTIAL


def test_projected_files_are_fully_read_when_the_os_returns_short_chunks(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    paths, _value = _write_projection(tmp_path)
    original_read = os.read
    interrupted: set[int] = set()

    def short_read(descriptor: int, maximum: int) -> bytes:
        if descriptor not in interrupted:
            interrupted.add(descriptor)
            raise InterruptedError
        return original_read(descriptor, min(maximum, 7))

    monkeypatch.setattr(managed_projection_module.os, "read", short_read)
    monkeypatch.setattr(managed_security_module.os, "read", short_read)

    projection = load_managed_projection(paths)

    assert projection.agent == "research-agent"
    assert read_managed_credential(paths.speech_credential) == _SPEECH_CREDENTIAL


def test_rejects_duplicate_unknown_and_missing_configuration_members(tmp_path: Path) -> None:
    paths, value = _write_projection(tmp_path)
    canonical = json.dumps(value, separators=(",", ":"))
    duplicate = canonical.replace(
        '"integration":"voiceclaw"',
        '"integration":"voiceclaw","integration":"other"',
        1,
    )
    _write_file(paths.configuration, duplicate)
    _assert_invalid(paths)

    value["unexpected"] = True
    _write_file(paths.configuration, json.dumps(value))
    _assert_invalid(paths)

    value.pop("unexpected")
    value.pop("agent")
    _write_file(paths.configuration, json.dumps(value))
    _assert_invalid(paths)


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("port", MANAGED_LISTENER_PORT + 1),
        ("port", float(MANAGED_LISTENER_PORT)),
        ("port", str(MANAGED_LISTENER_PORT)),
        ("speechProvider", "other"),
        ("speechCredentialPath", "/tmp/speech"),
        ("agentCredentialPath", "/tmp/agent"),
        ("integration", "Uppercase"),
        ("integration", "contains_underscore"),
        ("sandbox", "-leading-hyphen"),
        ("agent", "a" * 41),
        ("agent", ""),
    ],
)
def test_rejects_wrong_fixed_values_paths_and_names(
    tmp_path: Path,
    field: str,
    replacement: object,
) -> None:
    paths, value = _write_projection(tmp_path)
    value[field] = replacement
    _write_file(paths.configuration, json.dumps(value))

    _assert_invalid(paths)


def test_rejects_pathologically_nested_json_without_crashing_the_poller(tmp_path: Path) -> None:
    paths, _value = _write_projection(tmp_path)
    nested = '{"value":' * 1_100 + "null" + "}" * 1_100
    _write_file(paths.configuration, nested)

    _assert_invalid(paths)


@pytest.mark.parametrize(
    "endpoint",
    [
        "https://10.42.0.17:8652",
        "http://agent.internal:8652",
        "http://8.8.8.8:8652",
        "http://10.42.0.17:8652/path",
        "http://user:secret@10.42.0.17:8652",
        "http://10.42.0.17:8652?query=true",
        "http://10.42.0.17:80",
    ],
)
def test_rejects_unsafe_managed_endpoints(tmp_path: Path, endpoint: str) -> None:
    paths, value = _write_projection(tmp_path)
    value["agentEndpoint"] = endpoint
    _write_file(paths.configuration, json.dumps(value))

    _assert_invalid(paths)
    with pytest.raises(ManagedServiceSecurityError):
        validate_managed_http_origin(endpoint)


@pytest.mark.parametrize(
    "route_host",
    [
        "Research-Agent.sandbox-one.openshell.localhost:8652",
        "research-agent.example.test:8652",
        "research-agent.sandbox-one.openshell.localhost",
        "research-agent.sandbox-one.openshell.localhost:80",
        "research-agent.sandbox-one.openshell.localhost:70000",
        " research-agent.sandbox-one.openshell.localhost:8652",
    ],
)
def test_rejects_unsafe_managed_route_hosts(tmp_path: Path, route_host: str) -> None:
    paths, value = _write_projection(tmp_path)
    value["agentRouteHost"] = route_host
    _write_file(paths.configuration, json.dumps(value))

    _assert_invalid(paths)
    with pytest.raises(ManagedServiceSecurityError):
        validate_managed_route_host(route_host)


def test_rejects_endpoint_and_route_port_mismatch(tmp_path: Path) -> None:
    paths, value = _write_projection(tmp_path)
    value["agentRouteHost"] = "research-agent.sandbox-one.openshell.localhost:8653"
    _write_file(paths.configuration, json.dumps(value))

    _assert_invalid(paths)


def test_security_validators_canonicalize_supported_origins_and_routes() -> None:
    assert validate_managed_http_origin("http://127.0.0.1:8652") == (
        "http://127.0.0.1:8652",
        "127.0.0.1",
        8652,
    )
    assert validate_managed_http_origin("http://[::1]:8652") == ("http://[::1]:8652", "::1", 8652)
    assert validate_managed_route_host("agent.sandbox.openshell.localhost:8652") == (
        "agent.sandbox.openshell.localhost:8652",
        8652,
    )


def test_rejects_projection_root_and_leaf_symlinks(tmp_path: Path) -> None:
    real_paths, _value = _write_projection(tmp_path / "real")
    linked_root = tmp_path / "linked-root"
    linked_root.symlink_to(real_paths.root, target_is_directory=True)
    with pytest.raises(ManagedProjectionError, match="^projection-unavailable$"):
        load_managed_projection(ManagedProjectionPaths(root=linked_root))

    leaf_target = tmp_path / "config-target.json"
    leaf_target.write_bytes(real_paths.configuration.read_bytes())
    leaf_target.chmod(0o600)
    real_paths.configuration.unlink()
    real_paths.configuration.symlink_to(leaf_target)
    with pytest.raises(ManagedProjectionError, match="^projection-unavailable$"):
        load_managed_projection(real_paths)


def test_rejects_projected_directory_symlink(tmp_path: Path) -> None:
    paths, _value = _write_projection(tmp_path)
    real_credentials = tmp_path / "real-credentials"
    paths.root.joinpath("credentials").rename(real_credentials)
    paths.root.joinpath("credentials").symlink_to(real_credentials, target_is_directory=True)

    with pytest.raises(ManagedProjectionError, match="^projection-unavailable$"):
        load_managed_projection(paths)


@pytest.mark.parametrize(
    ("relative_path", "mode"),
    [
        ("runtime", 0o750),
        ("credentials", 0o750),
        ("runtime/config.json", 0o640),
        ("credentials/speech", 0o640),
        ("credentials/agent", 0o640),
    ],
)
def test_rejects_projection_with_unsafe_mode(tmp_path: Path, relative_path: str, mode: int) -> None:
    paths, _value = _write_projection(tmp_path)
    paths.root.joinpath(relative_path).chmod(mode)

    _assert_invalid(paths)


def test_rejects_projection_root_with_unsafe_mode(tmp_path: Path) -> None:
    paths, _value = _write_projection(tmp_path)
    paths.root.chmod(0o750)

    _assert_invalid(paths)


@pytest.mark.parametrize(("uid_offset", "gid_offset"), [(1, 0), (0, 1)])
def test_rejects_projection_owned_by_another_identity(
    tmp_path: Path,
    uid_offset: int,
    gid_offset: int,
) -> None:
    paths, _value = _write_projection(tmp_path)

    with pytest.raises(ManagedProjectionError, match="^projection-invalid$"):
        load_managed_projection(
            paths,
            expected_uid=os.geteuid() + uid_offset,
            expected_gid=os.getegid() + gid_offset,
        )


def test_rejects_access_acl(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    paths, _value = _write_projection(tmp_path)
    monkeypatch.setattr(managed_projection_module.os, "getxattr", lambda *_args: b"acl")

    _assert_invalid(paths)


def test_acl_inspection_failure_is_temporarily_unavailable(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    paths, _value = _write_projection(tmp_path)

    def unavailable(*_args: object) -> bytes:
        raise OSError(errno.EIO, "unavailable")

    monkeypatch.setattr(managed_projection_module.os, "getxattr", unavailable)

    with pytest.raises(ManagedProjectionError, match="^projection-unavailable$"):
        load_managed_projection(paths)


@pytest.mark.parametrize(
    ("relative_path", "payload"),
    [
        ("runtime/config.json", b""),
        ("runtime/config.json", b"x" * (32 * 1024 + 1)),
        ("credentials/speech", b""),
        ("credentials/speech", b"x" * 4097),
    ],
)
def test_rejects_empty_and_oversized_projected_files(
    tmp_path: Path,
    relative_path: str,
    payload: bytes,
) -> None:
    paths, _value = _write_projection(tmp_path)
    _write_file(paths.root / relative_path, payload)

    _assert_invalid(paths)


def test_rejects_an_absolute_projected_relative_path(tmp_path: Path) -> None:
    paths, _value = _write_projection(tmp_path)

    with pytest.raises(ManagedProjectionError, match="^projection-invalid$"):
        managed_projection_module._read_projected_file(
            paths,
            str(paths.configuration),
            uid=os.geteuid(),
            gid=os.getegid(),
            maximum=32 * 1024,
        )


def test_rejects_projection_that_changes_during_stable_read(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    paths, _value = _write_projection(tmp_path)
    first = (b"first", b"s" * 32, b"a" * 32)
    second = (b"second", b"s" * 32, b"a" * 32)
    snapshots = iter((first, second))
    monkeypatch.setattr(managed_projection_module, "_projection_payloads", lambda *_args, **_kwargs: next(snapshots))

    with pytest.raises(ManagedProjectionError, match="^projection-unavailable$"):
        load_managed_projection(paths)


def test_agent_credential_can_rotate_between_projection_snapshots(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    paths, _value = _write_projection(tmp_path)
    raw = paths.configuration.read_bytes()
    speech = paths.speech_credential.read_bytes()
    first = (raw, speech, b"a" * 32)
    second = (raw, speech, b"b" * 32)
    snapshots = iter((first, second))
    monkeypatch.setattr(managed_projection_module, "_projection_payloads", lambda *_args, **_kwargs: next(snapshots))

    projection = load_managed_projection(paths)

    assert projection.agent == "research-agent"


def test_rejects_speech_credential_rotation_between_projection_snapshots(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    paths, _value = _write_projection(tmp_path)
    raw = paths.configuration.read_bytes()
    agent = paths.agent_credential.read_bytes()
    snapshots = iter(((raw, b"s" * 32, agent), (raw, b"t" * 32, agent)))
    monkeypatch.setattr(managed_projection_module, "_projection_payloads", lambda *_args, **_kwargs: next(snapshots))

    with pytest.raises(ManagedProjectionError, match="^projection-unavailable$"):
        load_managed_projection(paths)


def test_projection_digest_tracks_frontend_generation_not_agent_rotation(tmp_path: Path) -> None:
    paths, _value = _write_projection(tmp_path)

    baseline = load_managed_projection(paths)
    _write_file(paths.agent_credential, "nemoclaw-agent-" + "b" * 40)
    agent_rotated = load_managed_projection(paths)
    _write_file(paths.speech_credential, "nvapi-speech-" + "t" * 40)
    speech_rotated = load_managed_projection(paths)

    assert agent_rotated.projection_digest == baseline.projection_digest
    assert speech_rotated.projection_digest != baseline.projection_digest


def test_rejects_a_projected_file_mutated_while_its_descriptor_is_read(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    paths, _value = _write_projection(tmp_path)
    original_read = os.read
    mutated = False

    def mutating_read(descriptor: int, maximum: int) -> bytes:
        nonlocal mutated
        chunk = original_read(descriptor, maximum)
        if chunk and not mutated:
            mutated = True
            paths.configuration.write_bytes(b"")
        return chunk

    monkeypatch.setattr(managed_projection_module.os, "read", mutating_read)

    with pytest.raises(ManagedProjectionError, match="^projection-unavailable$"):
        load_managed_projection(paths)


def test_rejects_same_size_credential_metadata_drift(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    paths, _value = _write_projection(tmp_path)
    original_read = os.read
    before = paths.speech_credential.stat()
    mutated = False

    def mutating_read(descriptor: int, maximum: int) -> bytes:
        nonlocal mutated
        chunk = original_read(descriptor, maximum)
        if chunk and not mutated:
            mutated = True
            os.utime(
                paths.speech_credential,
                ns=(before.st_atime_ns, before.st_mtime_ns + 1_000_000_000),
            )
        return chunk

    monkeypatch.setattr(managed_security_module.os, "read", mutating_read)

    with pytest.raises(ManagedServiceSecurityError, match="managed credential is unavailable"):
        read_managed_credential(paths.speech_credential)


def test_rejects_unsafe_credential_leaf_and_mode(tmp_path: Path) -> None:
    paths, _value = _write_projection(tmp_path)
    paths.speech_credential.chmod(0o640)
    with pytest.raises(ManagedServiceSecurityError, match="managed credential file is invalid"):
        read_managed_credential(paths.speech_credential)

    paths.speech_credential.unlink()
    credential_target = tmp_path / "credential-target"
    _write_file(credential_target, _SPEECH_CREDENTIAL)
    paths.speech_credential.symlink_to(credential_target)
    with pytest.raises(ManagedServiceSecurityError, match="managed credential is unavailable"):
        read_managed_credential(paths.speech_credential)


def test_materializes_secret_free_product_configuration(tmp_path: Path) -> None:
    paths, _value = _write_projection(tmp_path)
    projection = load_managed_projection(paths)
    destination = tmp_path / "runtime" / "voiceclaw.yaml"

    config = materialize_managed_config(projection, destination)
    rendered = destination.read_text(encoding="utf-8")
    document = yaml.safe_load(rendered)

    assert _SPEECH_CREDENTIAL not in rendered
    assert _AGENT_CREDENTIAL not in rendered
    assert config.server.port == MANAGED_LISTENER_PORT
    assert config.server.max_sessions == 1
    assert config.state.path == str(paths.state)
    assert config.backend_profiles["nemoclaw"].settings["endpoint"] == "http://10.42.0.17:8652"
    assert (
        config.backend_profiles["nemoclaw"].settings["agent_route_host"]
        == "research-agent.sandbox-one.openshell.localhost:8652"
    )
    assert config.backend_profiles["nemoclaw"].settings["target_ref"] == "voiceclaw:sandbox-one:research-agent"
    assert config.frontend_profiles["managed_nvidia"].services.llm.credential.file == str(paths.speech_credential)
    assert config.frontend_profiles["managed_nvidia"].services.asr.credential.file == str(paths.speech_credential)
    assert config.frontend_profiles["managed_nvidia"].services.tts.credential.file == str(paths.speech_credential)
    assert config.backend_profiles["nemoclaw"].credential_file == str(paths.agent_credential)
    assert document["server"]["max_sessions"] == 1
    assert document["state"]["path"] == str(paths.root / "runtime/state.db")
