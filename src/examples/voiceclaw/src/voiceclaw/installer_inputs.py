# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""Explicit container-v1 inputs and descriptor-based protected reads."""

from __future__ import annotations

import errno
import ipaddress
import json
import os
import re
import stat
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit
from uuid import UUID

PROFILE = "nemoclaw-container-v1"
CONTRACT = "voiceclaw.nemoclaw.container.v1"
DATA_ROOT = Path("/var/lib/voiceclaw")
DESCRIPTOR_LIMIT = 64 * 1024
CREDENTIAL_LIMIT = 16 * 1024
_NAME = re.compile(r"[a-z][a-z0-9-]{0,62}\Z")
_WORKSPACE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}\Z")
_BEARER = re.compile(rb"[A-Za-z0-9._~+/-]+=*\Z")


class ContainerInputError(ValueError):
    """An input is absent, unsafe, unstable, or incompatible, without content."""

    def __init__(self) -> None:
        """Expose one fixed diagnostic without input paths or values."""
        super().__init__("container-input-invalid")


def _metadata_identity(metadata: os.stat_result) -> tuple[int, ...]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_uid,
        metadata.st_gid,
        stat.S_IFMT(metadata.st_mode),
        stat.S_IMODE(metadata.st_mode),
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _read_stable_file_descriptor(
    descriptor: int,
    *,
    before: os.stat_result,
    maximum: int,
    has_access_acl: Callable[[int], bool],
) -> bytes:
    chunks: list[bytes] = []
    remaining = maximum + 1
    try:
        while remaining:
            try:
                chunk = os.read(descriptor, min(remaining, 64 * 1024))
            except InterruptedError:
                continue
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        payload = b"".join(chunks)
        after = os.fstat(descriptor)
        if (
            len(payload) != before.st_size
            or len(payload) > maximum
            or _metadata_identity(after) != _metadata_identity(before)
            or has_access_acl(descriptor)
        ):
            raise ContainerInputError
    except OSError as error:
        raise ContainerInputError from error
    return payload


def _path(value: object, *, root: Path = DATA_ROOT, directory: str | None = None) -> Path:
    if not isinstance(value, str) or not value or any(ord(c) < 33 or ord(c) > 126 for c in value):
        raise ContainerInputError
    if "\\" in value or any(p in {"", ".", ".."} for p in value.split("/")[1:]):
        raise ContainerInputError
    candidate = Path(value)
    try:
        relative = candidate.relative_to(root)
    except ValueError:
        raise ContainerInputError from None
    if not candidate.is_absolute() or len(relative.parts) < 2:
        raise ContainerInputError
    if directory is not None and relative.parts[0] != directory:
        raise ContainerInputError
    return candidate


def _acl(fd: int) -> bool:
    getter = getattr(os, "getxattr", None)
    if getter is None:
        raise ContainerInputError
    try:
        return bool(getter(fd, "system.posix_acl_access")) or bool(getter(fd, "system.posix_acl_default"))
    except OSError as error:
        if error.errno in {errno.ENODATA, errno.ENOTSUP, getattr(errno, "ENOATTR", errno.ENODATA)}:
            # Inspect both ACL kinds independently: a missing access ACL does
            # not prove that a directory lacks an inheritable default ACL.
            for name in ("system.posix_acl_access", "system.posix_acl_default"):
                try:
                    if getter(fd, name):
                        return True
                except OSError as inner:
                    if inner.errno not in {errno.ENODATA, errno.ENOTSUP, getattr(errno, "ENOATTR", errno.ENODATA)}:
                        raise ContainerInputError from None
            return False
        raise ContainerInputError from None


def _owned_directory(fd: int, uid: int, gid: int) -> None:
    m = os.fstat(fd)
    if not stat.S_ISDIR(m.st_mode) or (m.st_uid, m.st_gid, stat.S_IMODE(m.st_mode)) != (uid, gid, 0o700) or _acl(fd):
        raise ContainerInputError


@contextmanager
def protected_directory(
    path: Path, *, root: Path = DATA_ROOT, uid: int = 65532, gid: int = 65532, mutable: bool = False
) -> Iterator[int]:
    """Hold and revalidate the no-follow directory chain at a trusted mount."""
    flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_DIRECTORY
    descriptors = []
    bindings = []
    try:
        relative = path.relative_to(root)
        fd = os.open("/", flags)
        descriptors.append(fd)
        for component in (*root.parts[1:], *relative.parts):
            parent = fd
            fd = os.open(component, flags, dir_fd=parent)
            descriptors.append(fd)
            before = os.fstat(fd)
            inside_root = len(descriptors) >= len(root.parts)
            bindings.append((parent, component, fd, before, inside_root))
            if len(descriptors) >= len(root.parts):
                _owned_directory(fd, uid, gid)
            elif root == DATA_ROOT and (before.st_uid != 0 or before.st_mode & 0o022):
                raise ContainerInputError
        _owned_directory(fd, uid, gid)
        yield fd
        for parent, name, child, before, inside_root in bindings:
            bound = os.stat(name, dir_fd=parent, follow_symlinks=False)
            # Ancestor contents are not installer inputs. Retain path identity,
            # owner and mode checks there, but allow unrelated sibling activity.
            stable_contents = inside_root and not mutable
            identity = _metadata_identity(before) if stable_contents else _metadata_identity(before)[:6]
            observed = _metadata_identity(bound) if stable_contents else _metadata_identity(bound)[:6]
            held = _metadata_identity(os.fstat(child)) if stable_contents else _metadata_identity(os.fstat(child))[:6]
            if observed != identity or held != identity:
                raise ContainerInputError
    except ContainerInputError:
        raise
    except Exception:
        raise ContainerInputError from None
    finally:
        for descriptor in reversed(descriptors):
            os.close(descriptor)


def read_protected(
    path: str,
    *,
    root: Path = DATA_ROOT,
    uid: int = 65532,
    gid: int = 65532,
    maximum: int = CREDENTIAL_LIMIT,
) -> bytes:
    """Read one stable owner-only file through a no-follow descriptor chain.

    The root is a trusted application mount boundary. Alternate roots and
    identities exist for deterministic tests, never installer descriptors.
    """
    candidate = _path(path, root=root)
    fd = None
    try:
        with protected_directory(candidate.parent, root=root, uid=uid, gid=gid) as directory:
            fd = os.open(candidate.name, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory)
            before = os.fstat(fd)
            if (
                not stat.S_ISREG(before.st_mode)
                or (before.st_uid, before.st_gid, stat.S_IMODE(before.st_mode)) != (uid, gid, 0o600)
                or not 0 < before.st_size <= maximum
                or _acl(fd)
            ):
                raise ContainerInputError
            payload = _read_stable_file_descriptor(fd, before=before, maximum=maximum, has_access_acl=_acl)
            os.lseek(fd, 0, os.SEEK_SET)
            again = _read_stable_file_descriptor(fd, before=before, maximum=maximum, has_access_acl=_acl)
            bound = os.stat(candidate.name, dir_fd=directory, follow_symlinks=False)
            if payload != again or _metadata_identity(bound) != _metadata_identity(before):
                raise ContainerInputError
        return payload
    except ContainerInputError:
        raise
    except Exception:
        raise ContainerInputError from None
    finally:
        if fd is not None:
            os.close(fd)


def validate_bearer(payload: bytes) -> str:
    """Accept a bounded RFC 6750 bearer without header-breaking characters."""
    if not 1 <= len(payload) <= CREDENTIAL_LIMIT or _BEARER.fullmatch(payload) is None:
        raise ContainerInputError
    return payload.decode("ascii")


def validate_speech_key(payload: bytes) -> str:
    """Validate one NVIDIA API key value, independently of bearer framing."""
    if not 1 <= len(payload) <= CREDENTIAL_LIMIT or any(c < 33 or c > 126 for c in payload):
        raise ContainerInputError
    return payload.decode("ascii")


def strict_json(payload: bytes, maximum: int) -> dict:
    """Decode exactly one bounded UTF-8 JSON object with no duplicate keys."""

    def collect(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ContainerInputError
            result[key] = value
        return result

    def invalid_constant(_value):
        raise ContainerInputError

    try:
        if not 0 < len(payload) <= maximum:
            raise ContainerInputError
        value = json.loads(payload.decode("utf-8"), object_pairs_hook=collect, parse_constant=invalid_constant)
        if not isinstance(value, dict):
            raise ContainerInputError
        return value
    except Exception:
        raise ContainerInputError from None


def _object(value: object, keys: set[str]) -> dict:
    if not isinstance(value, dict) or set(value) != keys:
        raise ContainerInputError
    return value


def _identity(value: object, pattern: re.Pattern) -> str:
    if not isinstance(value, str) or pattern.fullmatch(value) is None:
        raise ContainerInputError
    return value


def _uuid(value: object) -> str:
    try:
        if not isinstance(value, str) or str(UUID(value)) != value or UUID(value).int == 0:
            raise ContainerInputError
    except (ValueError, TypeError, AttributeError):
        raise ContainerInputError from None
    return value


def _endpoint(value: object, *, no_auth: bool = False) -> str:
    if not isinstance(value, str) or not value or any(ord(c) < 33 or ord(c) > 126 for c in value):
        raise ContainerInputError
    try:
        parsed = urlsplit(value)
        host = parsed.hostname
        port = parsed.port
        if (
            parsed.scheme != ("http" if no_auth else "https")
            or not host
            or parsed.username is not None
            or parsed.password is not None
            or parsed.fragment
            or parsed.query
            or parsed.path not in {"", "/"}
            or "#" in value
            or "?" in value
            or "%" in host
            or "\\" in value
            or port is None
            or not 1 <= port <= 65535
        ):
            raise ContainerInputError
        if no_auth:
            # A literal excludes DNS rebinding and alternate URL/IP spellings.
            # This is only an address filter: NemoClaw must attest the exact
            # Docker gateway binding and absence of LAN/public publication.
            address = ipaddress.IPv4Address(host)
            if value not in {f"http://{address}:{port}", f"http://{address}:{port}/"} or not any(
                address in ipaddress.IPv4Network(network)
                for network in ("10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16")
            ):
                raise ContainerInputError
            return value
        try:
            address = ipaddress.ip_address(host)
        except ValueError:
            labels = host.split(".")
            if (
                host.lower() == "localhost"
                or host.lower().endswith(".localhost")
                or len(host) > 253
                or not any(c.isalpha() for c in host)
                or all(re.fullmatch(r"(?:0[xX][0-9A-Fa-f]+|[0-9]+)", label) for label in labels)
                or any(
                    re.fullmatch(r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?", label) is None for label in labels
                )
            ):
                raise ContainerInputError from None
        else:
            if isinstance(address, ipaddress.IPv6Address) and address.ipv4_mapped is not None:
                address = address.ipv4_mapped
            if address.is_loopback or address.is_unspecified or address.is_multicast or address.is_link_local:
                raise ContainerInputError
    except (ValueError, TypeError):
        raise ContainerInputError from None
    return value


@dataclass(frozen=True, slots=True)
class AgentConnection:
    """One explicit target and auth mode; no inferred credentials or fallback."""

    deployment_uid: str
    service: str
    endpoint: str
    credential_file: Path | None
    authentication_mode: str
    workspace: str
    sandbox: str
    sandbox_id: str
    agent: str
    health_seconds: int
    invoke_seconds: int


def parse_connection(payload: bytes) -> AgentConnection:
    """Validate the exact v1 descriptor; no legacy or transport fallback."""
    value = _object(
        strict_json(payload, DESCRIPTOR_LIMIT),
        {
            "schemaVersion",
            "deploymentUid",
            "service",
            "gateway",
            "authentication",
            "target",
            "bridge",
            "timeouts",
        },
    )
    gateway = _object(value["gateway"], {"endpoint", "tls"})
    tls = _object(gateway["tls"], {"trust", "caFile"})
    auth = _object(value["authentication"], {"mode", "credentialFile", "refreshMode"})
    target = _object(value["target"], {"workspace", "sandbox", "sandboxId", "agent"})
    bridge = _object(value["bridge"], {"interfaceVersion", "command"})
    times = _object(value["timeouts"], {"healthSeconds", "invokeSeconds"})
    if (
        value["schemaVersion"] != "nemoclaw.agent-connection.v1"
        or auth["refreshMode"] != "none"
        or type(bridge["interfaceVersion"]) is not int
        or bridge != {"interfaceVersion": 1, "command": "fabric-agent"}
        or type(times["healthSeconds"]) is not int
        or not 1 <= times["healthSeconds"] <= 12
        or type(times["invokeSeconds"]) is not int
        or not 1 <= times["invokeSeconds"] <= 120
    ):
        raise ContainerInputError
    if auth["mode"] == "oidcBearer" and tls == {"trust": "system", "caFile": None}:
        credential = _path(auth["credentialFile"], directory="credentials")
        endpoint = _endpoint(gateway["endpoint"])
    elif auth["mode"] == "none" and tls == {"trust": "none", "caFile": None} and auth["credentialFile"] is None:
        credential = None
        endpoint = _endpoint(gateway["endpoint"], no_auth=True)
    else:
        raise ContainerInputError
    return AgentConnection(
        deployment_uid=_uuid(value["deploymentUid"]),
        service=_identity(value["service"], _NAME),
        endpoint=endpoint,
        credential_file=credential,
        authentication_mode=auth["mode"],
        workspace=_identity(target["workspace"], _WORKSPACE),
        sandbox=_identity(target["sandbox"], _NAME),
        sandbox_id=_uuid(target["sandboxId"]),
        agent=_identity(target["agent"], _NAME),
        health_seconds=times["healthSeconds"],
        invoke_seconds=times["invokeSeconds"],
    )


def load_connection(path: str) -> AgentConnection:
    """Read the protected connection descriptor without modifying inputs."""
    return parse_connection(read_protected(path, maximum=DESCRIPTOR_LIMIT))


@dataclass(frozen=True, slots=True)
class ProfileInputs:
    """The five complete, explicit nonsecret profile inputs."""

    connection_file: Path
    speech_file: Path
    state: Path

    @classmethod
    def from_environment(cls, environ: Mapping[str, str]) -> ProfileInputs:
        """Reject incomplete profile inputs before application activation."""
        if environ.get("VOICECLAW_RUNTIME_PROFILE") != PROFILE or environ.get("VOICECLAW_INSTALL_CONTRACT") != CONTRACT:
            raise ContainerInputError
        return cls(
            _path(environ.get("VOICECLAW_AGENT_CONNECTION_FILE"), directory="config"),
            _path(environ.get("VOICECLAW_SPEECH_CREDENTIAL_FILE"), directory="credentials"),
            _path(environ.get("VOICECLAW_STATE_PATH"), directory="state"),
        )
