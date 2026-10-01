# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""Strict adapter from NemoClaw's managed projection to VoiceClaw config."""

from __future__ import annotations

import errno
import hashlib
import json
import os
import re
import stat
import tempfile
from contextlib import suppress
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any

import yaml

from voiceclaw.adapters.nemoclaw.security import (
    ManagedServiceSecurityError,
    _read_stable_file_descriptor,
    _StableFileReadError,
    validate_managed_http_origin,
    validate_managed_route_host,
)
from voiceclaw.config import ConfigurationError, VoiceClawConfig, load_config

MANAGED_LISTENER_PORT = 18_790
_MAX_CONFIG_BYTES = 32 * 1024
_MAX_CREDENTIAL_BYTES = 4096
_NAME = re.compile(r"^[a-z][a-z0-9-]{0,39}$")
_EXPECTED_KEYS = frozenset(
    {
        "integration",
        "sandbox",
        "agent",
        "port",
        "speechProvider",
        "speechCredentialPath",
        "agentCredentialPath",
        "agentEndpoint",
        "agentRouteHost",
    }
)
_TEMPLATE = "nemoclaw_managed.yaml"


class ManagedProjectionError(RuntimeError):
    """A projected file is absent, unsafe, or incompatible."""

    def __init__(self, code: str) -> None:
        """Create an error carrying one stable, content-free reason code."""
        self.code = code
        super().__init__(code)


class _DuplicateKey(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class ManagedProjectionPaths:
    """Fixed installer-owned paths under one managed volume."""

    root: Path = Path("/var/lib/voiceclaw")

    @property
    def configuration(self) -> Path:
        """Return the installer-owned configuration path."""
        return self.root / "runtime/config.json"

    @property
    def speech_credential(self) -> Path:
        """Return the installer-owned NVIDIA credential path."""
        return self.root / "credentials/speech"

    @property
    def agent_credential(self) -> Path:
        """Return the installer-owned selected-agent credential path."""
        return self.root / "credentials/agent"

    @property
    def state(self) -> Path:
        """Return the writable VoiceClaw projection database path."""
        return self.root / "runtime/state.db"


@dataclass(frozen=True, slots=True)
class ManagedProjection:
    """Validated non-secret deployment settings selected by NemoClaw."""

    integration: str
    sandbox: str
    agent: str
    agent_endpoint: str
    agent_route_host: str
    paths: ManagedProjectionPaths
    projection_digest: str


def _strict_object(items: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in items:
        if key in value:
            raise _DuplicateKey
        value[key] = item
    return value


def _has_access_acl(descriptor: int) -> bool:
    getxattr = getattr(os, "getxattr", None)
    if getxattr is None:
        return False
    try:
        access_acl = getxattr(descriptor, "system.posix_acl_access")
    except OSError as error:
        allowed = {errno.ENODATA, errno.ENOTSUP, getattr(errno, "ENOATTR", errno.ENODATA)}
        if error.errno not in allowed:
            raise ManagedProjectionError("projection-unavailable") from error
    else:
        return bool(access_acl)
    return False


def _open_root(paths: ManagedProjectionPaths, *, uid: int, gid: int) -> int:
    if not paths.root.is_absolute() or not hasattr(os, "O_NOFOLLOW"):
        raise ManagedProjectionError("projection-invalid")
    components = paths.root.parts[1:]
    if not components or any(component in {"", ".", ".."} for component in components):
        raise ManagedProjectionError("projection-invalid")
    flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_DIRECTORY
    descriptor = -1
    try:
        descriptor = os.open("/", flags)
        for component in components:
            next_descriptor = os.open(component, flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = next_descriptor
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISDIR(metadata.st_mode)
            or metadata.st_uid != uid
            or metadata.st_gid != gid
            or stat.S_IMODE(metadata.st_mode) != 0o700
            or _has_access_acl(descriptor)
        ):
            raise ManagedProjectionError("projection-invalid")
        return descriptor
    except ManagedProjectionError:
        if descriptor >= 0:
            os.close(descriptor)
        raise
    except OSError as error:
        if descriptor >= 0:
            os.close(descriptor)
        raise ManagedProjectionError("projection-unavailable") from error


def _read_projected_file(
    paths: ManagedProjectionPaths,
    relative: str,
    *,
    uid: int,
    gid: int,
    maximum: int,
) -> bytes:
    directory = _open_root(paths, uid=uid, gid=gid)
    descriptor = -1
    try:
        relative_path = Path(relative)
        parts = relative_path.parts
        if relative_path.is_absolute() or not parts or any(part in {"", ".", ".."} for part in parts):
            raise ManagedProjectionError("projection-invalid")
        flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_DIRECTORY
        for component in parts[:-1]:
            next_directory = os.open(component, flags, dir_fd=directory)
            os.close(directory)
            directory = next_directory
            metadata = os.fstat(directory)
            if (
                not stat.S_ISDIR(metadata.st_mode)
                or metadata.st_uid != uid
                or metadata.st_gid != gid
                or stat.S_IMODE(metadata.st_mode) != 0o700
                or _has_access_acl(directory)
            ):
                raise ManagedProjectionError("projection-invalid")
        leaf_flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK
        descriptor = os.open(parts[-1], leaf_flags, dir_fd=directory)
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_uid != uid
            or before.st_gid != gid
            or stat.S_IMODE(before.st_mode) != 0o600
            or not 0 < before.st_size <= maximum
            or _has_access_acl(descriptor)
        ):
            raise ManagedProjectionError("projection-invalid")
        try:
            return _read_stable_file_descriptor(
                descriptor,
                before=before,
                maximum=maximum,
                has_access_acl=_has_access_acl,
            )
        except _StableFileReadError as error:
            raise ManagedProjectionError("projection-unavailable") from error
    except ManagedProjectionError:
        raise
    except OSError as error:
        raise ManagedProjectionError("projection-unavailable") from error
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        os.close(directory)


def _credential(payload: bytes) -> None:
    if not 32 <= len(payload) <= _MAX_CREDENTIAL_BYTES or any(byte < 0x21 or byte > 0x7E for byte in payload):
        raise ManagedProjectionError("projection-invalid")


def _name(value: object) -> str:
    if not isinstance(value, str) or _NAME.fullmatch(value) is None:
        raise ManagedProjectionError("projection-invalid")
    return value


def _projection_payloads(
    paths: ManagedProjectionPaths,
    *,
    uid: int,
    gid: int,
) -> tuple[bytes, bytes, bytes]:
    return (
        _read_projected_file(paths, "runtime/config.json", uid=uid, gid=gid, maximum=_MAX_CONFIG_BYTES),
        _read_projected_file(
            paths,
            "credentials/speech",
            uid=uid,
            gid=gid,
            maximum=_MAX_CREDENTIAL_BYTES,
        ),
        _read_projected_file(
            paths,
            "credentials/agent",
            uid=uid,
            gid=gid,
            maximum=_MAX_CREDENTIAL_BYTES,
        ),
    )


def load_managed_projection(
    paths: ManagedProjectionPaths | None = None,
    *,
    expected_uid: int | None = None,
    expected_gid: int | None = None,
) -> ManagedProjection:
    """Reload and validate one complete installer projection."""
    selected = paths or ManagedProjectionPaths()
    uid = os.geteuid() if expected_uid is None else expected_uid
    gid = os.getegid() if expected_gid is None else expected_gid
    first = _projection_payloads(selected, uid=uid, gid=gid)
    second = _projection_payloads(selected, uid=uid, gid=gid)
    if first[:2] != second[:2]:
        raise ManagedProjectionError("projection-unavailable")
    raw, speech, agent = second
    _credential(speech)
    _credential(agent)
    try:
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=_strict_object)
    except (UnicodeDecodeError, json.JSONDecodeError, _DuplicateKey, RecursionError) as error:
        raise ManagedProjectionError("projection-invalid") from error
    if not isinstance(value, dict) or set(value) != _EXPECTED_KEYS:
        raise ManagedProjectionError("projection-invalid")
    if type(value["port"]) is not int or value["port"] != MANAGED_LISTENER_PORT:
        raise ManagedProjectionError("projection-invalid")
    if value["speechProvider"] != "nvidia":
        raise ManagedProjectionError("projection-invalid")
    if value["speechCredentialPath"] != str(selected.speech_credential) or value["agentCredentialPath"] != str(
        selected.agent_credential
    ):
        raise ManagedProjectionError("projection-invalid")
    try:
        endpoint, _host, endpoint_port = validate_managed_http_origin(value["agentEndpoint"])
        route_host, route_port = validate_managed_route_host(value["agentRouteHost"])
    except ManagedServiceSecurityError as error:
        raise ManagedProjectionError("projection-invalid") from error
    if endpoint_port != route_port:
        raise ManagedProjectionError("projection-invalid")
    projection_digest = hashlib.sha256(raw + b"\0" + speech).hexdigest()
    return ManagedProjection(
        integration=_name(value["integration"]),
        sandbox=_name(value["sandbox"]),
        agent=_name(value["agent"]),
        agent_endpoint=endpoint,
        agent_route_host=route_host,
        paths=selected,
        projection_digest=projection_digest,
    )


def materialize_managed_config(projection: ManagedProjection, destination: Path) -> VoiceClawConfig:
    """Render the internal product configuration without copying secret values."""
    try:
        payload = resources.files("voiceclaw.resources").joinpath(_TEMPLATE).read_text(encoding="utf-8")
        raw = yaml.safe_load(payload)
        settings = raw["backend_profiles"]["nemoclaw"]["settings"]
        settings["endpoint"] = projection.agent_endpoint
        settings["agent_route_host"] = projection.agent_route_host
        settings["target_ref"] = f"{projection.integration}:{projection.sandbox}:{projection.agent}"
        raw["frontend_profiles"]["managed_nvidia"]["services"]["llm"]["credential"]["file"] = str(
            projection.paths.speech_credential
        )
        raw["frontend_profiles"]["managed_nvidia"]["services"]["asr"]["credential"]["file"] = str(
            projection.paths.speech_credential
        )
        raw["frontend_profiles"]["managed_nvidia"]["services"]["tts"]["credential"]["file"] = str(
            projection.paths.speech_credential
        )
        raw["backend_profiles"]["nemoclaw"]["credential"]["file"] = str(projection.paths.agent_credential)
        raw["state"]["path"] = str(projection.paths.state)
        rendered = yaml.safe_dump(raw, sort_keys=False, allow_unicode=True)
    except (KeyError, TypeError, yaml.YAMLError) as error:
        raise ManagedProjectionError("internal-profile-invalid") from error
    temporary_path: Path | None = None
    try:
        destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        metadata = destination.parent.stat(follow_symlinks=False)
        if (
            destination.parent.is_symlink()
            or metadata.st_uid != os.geteuid()
            or metadata.st_gid != os.getegid()
            or stat.S_IMODE(metadata.st_mode) != 0o700
        ):
            raise ManagedProjectionError("runtime-path-invalid")
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            temporary.write(rendered)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.chmod(temporary_path, 0o600)
        os.replace(temporary_path, destination)
    except ManagedProjectionError:
        raise
    except OSError as error:
        raise ManagedProjectionError("runtime-path-unavailable") from error
    finally:
        if temporary_path is not None and temporary_path.exists():
            with suppress(OSError):
                temporary_path.unlink()
    try:
        return load_config(destination, environ={})
    except (ConfigurationError, OSError, ValueError) as error:
        raise ManagedProjectionError("internal-profile-invalid") from error


__all__ = [
    "MANAGED_LISTENER_PORT",
    "ManagedProjection",
    "ManagedProjectionError",
    "ManagedProjectionPaths",
    "load_managed_projection",
    "materialize_managed_config",
]
