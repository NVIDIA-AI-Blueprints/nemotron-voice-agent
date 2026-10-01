# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""Security primitives shared by the NemoClaw managed-service adapter."""

from __future__ import annotations

import errno
import ipaddress
import os
import re
import stat
from collections.abc import Callable
from pathlib import Path
from urllib.parse import urlsplit

_PRIVATE_IPV4_NETWORKS = (
    ipaddress.ip_network("10.0.0.0/8"),
    ipaddress.ip_network("172.16.0.0/12"),
    ipaddress.ip_network("192.168.0.0/16"),
)
_ROUTE_HOST = re.compile(
    r"^(?=.{1,253}:\d{1,5}$)(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+"
    r"localhost:[1-9]\d{0,4}$"
)
_MIN_CREDENTIAL_BYTES = 32
_MAX_CREDENTIAL_BYTES = 4096


class ManagedServiceSecurityError(ValueError):
    """A managed origin, route, or credential violates its fixed contract."""


class _StableFileReadError(RuntimeError):
    pass


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
            raise _StableFileReadError
    except OSError as error:
        raise _StableFileReadError from error
    return payload


def validate_managed_http_origin(value: object) -> tuple[str, str, int]:
    """Return the canonical private HTTP origin, literal address, and port."""
    if not isinstance(value, str) or not value or value != value.strip():
        raise ManagedServiceSecurityError("managed endpoint is invalid")
    try:
        parsed = urlsplit(value)
        port = parsed.port
    except ValueError as error:
        raise ManagedServiceSecurityError("managed endpoint is invalid") from error
    if (
        parsed.scheme != "http"
        or parsed.username is not None
        or parsed.password is not None
        or parsed.path not in {"", "/"}
        or parsed.query
        or parsed.fragment
        or parsed.hostname is None
        or "%" in parsed.hostname
        or port is None
        or not 1024 <= port <= 65_535
    ):
        raise ManagedServiceSecurityError("managed endpoint is invalid")
    try:
        address = ipaddress.ip_address(parsed.hostname)
    except ValueError as error:
        raise ManagedServiceSecurityError("managed endpoint must contain a literal IP address") from error
    private_ipv4 = address.version == 4 and any(address in network for network in _PRIVATE_IPV4_NETWORKS)
    if not (address.is_loopback or private_ipv4):
        raise ManagedServiceSecurityError("managed endpoint must be loopback or private")
    host = f"[{address.compressed}]" if address.version == 6 else address.compressed
    return f"http://{host}:{port}", address.compressed, port


def validate_managed_route_host(value: object) -> tuple[str, int]:
    """Return one canonical OpenShell route authority and its explicit port."""
    if not isinstance(value, str) or value != value.strip() or _ROUTE_HOST.fullmatch(value) is None:
        raise ManagedServiceSecurityError("managed route host is invalid")
    port = int(value.rsplit(":", 1)[1])
    if not 1024 <= port <= 65_535:
        raise ManagedServiceSecurityError("managed route host is invalid")
    return value, port


def read_managed_credential(
    path: str | os.PathLike[str],
    *,
    expected_uid: int | None = None,
    expected_gid: int | None = None,
) -> str:
    """Securely reopen one owner-only projected credential."""
    candidate = Path(path)
    if not candidate.is_absolute() or not hasattr(os, "O_NOFOLLOW"):
        raise ManagedServiceSecurityError("managed credential path is invalid")
    uid = os.geteuid() if expected_uid is None else expected_uid
    gid = os.getegid() if expected_gid is None else expected_gid
    parts = candidate.parts
    if len(parts) < 3 or any(part in {"", ".", ".."} for part in parts[1:]):
        raise ManagedServiceSecurityError("managed credential path is invalid")
    directory_flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_DIRECTORY
    leaf_flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK
    directory = -1
    descriptor = -1
    try:
        directory = os.open("/", directory_flags)
        for index, component in enumerate(parts[1:-1], start=1):
            next_directory = os.open(component, directory_flags, dir_fd=directory)
            os.close(directory)
            directory = next_directory
            if index >= len(parts) - 3:
                metadata = os.fstat(directory)
                if (
                    not stat.S_ISDIR(metadata.st_mode)
                    or metadata.st_uid != uid
                    or metadata.st_gid != gid
                    or stat.S_IMODE(metadata.st_mode) != 0o700
                    or _has_access_acl(directory, "managed credential directory")
                ):
                    raise ManagedServiceSecurityError("managed credential directory is invalid")
        descriptor = os.open(parts[-1], leaf_flags, dir_fd=directory)
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_uid != uid
            or before.st_gid != gid
            or stat.S_IMODE(before.st_mode) != 0o600
            or not _MIN_CREDENTIAL_BYTES <= before.st_size <= _MAX_CREDENTIAL_BYTES
            or _has_access_acl(descriptor, "managed credential file")
        ):
            raise ManagedServiceSecurityError("managed credential file is invalid")
        try:
            payload = _read_stable_file_descriptor(
                descriptor,
                before=before,
                maximum=_MAX_CREDENTIAL_BYTES,
                has_access_acl=lambda value: _has_access_acl(value, "managed credential file"),
            )
        except _StableFileReadError as error:
            raise ManagedServiceSecurityError("managed credential is unavailable") from error
    except OSError as error:
        raise ManagedServiceSecurityError("managed credential is unavailable") from error
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if directory >= 0:
            os.close(directory)
    try:
        value = payload.decode("ascii")
    except UnicodeDecodeError as error:
        raise ManagedServiceSecurityError("managed credential is invalid") from error
    if not _MIN_CREDENTIAL_BYTES <= len(payload) <= _MAX_CREDENTIAL_BYTES or any(
        byte < 0x21 or byte > 0x7E for byte in payload
    ):
        raise ManagedServiceSecurityError("managed credential is invalid")
    return value


def _has_access_acl(descriptor: int, label: str) -> bool:
    getxattr = getattr(os, "getxattr", None)
    if getxattr is None:
        return False
    try:
        access_acl = getxattr(descriptor, "system.posix_acl_access")
    except OSError as error:
        no_acl_errors = {
            errno.ENODATA,
            errno.ENOTSUP,
            getattr(errno, "ENOATTR", errno.ENODATA),
        }
        if error.errno not in no_acl_errors:
            raise ManagedServiceSecurityError(f"{label} ACL is unavailable") from error
    else:
        return bool(access_acl)
    return False


__all__ = [
    "ManagedServiceSecurityError",
    "read_managed_credential",
    "validate_managed_http_origin",
    "validate_managed_route_host",
]
