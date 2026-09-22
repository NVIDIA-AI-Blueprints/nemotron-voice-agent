# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""Container supervisor for the bundled NVA Realtime frontend and VoiceClaw."""

from __future__ import annotations

import argparse
import errno
import http.client
import json
import os
import pwd
import secrets
import signal
import socket
import ssl
import stat
import subprocess
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from contextlib import suppress
from pathlib import Path

import yaml

from voiceclaw.config import (
    ConfigurationError,
    CredentialReference,
    VoiceClawConfig,
    configuration_environment_references,
    load_config,
)
from voiceclaw.frontend_runtime import (
    FrontendRuntimePlan,
    bind_nva_credential,
    materialize_frontend_runtime,
    resolve_credential_value,
)
from voiceclaw.model_contracts import ModelContractError, load_model_contract_catalog
from voiceclaw.server import _tls_listener_files, _with_listener_host

_DEFAULT_CONFIG = "/var/lib/voiceclaw/config/voiceclaw.yaml"
_DEFAULT_NVA_SERVER = "/app/src/realtime_server.py"
_DEFAULT_NVA_PYTHON = "/app/.venv/bin/python"
_DEFAULT_FRONTEND_RUNTIME_DIR = "/run/voiceclaw/frontend"
_DEFAULT_HEALTHCHECK_TARGET = "/run/voiceclaw/healthcheck.json"
_DEFAULT_CONFIG_SNAPSHOT = "/run/voiceclaw/voiceclaw.snapshot.yaml"
_DEFAULT_OPERATOR_FILES_DIR = "/run/voiceclaw/operator"
_DEFAULT_MANAGED_CREDENTIALS_DIR = "/var/lib/voiceclaw/credentials"
_CONTAINER_RUNTIME_MARKER = "/etc/voiceclaw-container-runtime"
_INTERNAL_REALTIME_PORT = 7861
_MAX_CONFIG_BYTES = 4 * 1024 * 1024

# These process-level values are needed by Python or its TLS/network stack and
# do not grant either child another component's application credentials.
_PROCESS_ENVIRONMENT = frozenset(
    {
        "CURL_CA_BUNDLE",
        "GRPC_DEFAULT_SSL_ROOTS_FILE_PATH",
        "LANG",
        "LC_ALL",
        "LC_CTYPE",
        "LD_LIBRARY_PATH",
        "LIBRARY_PATH",
        "PATH",
        "PYTHONDONTWRITEBYTECODE",
        "PYTHONUNBUFFERED",
        "REQUESTS_CA_BUNDLE",
        "SSL_CERT_DIR",
        "SSL_CERT_FILE",
        "TZ",
    }
)

# The private Realtime child gets only the cloud frontend credential and the
# documented knobs consumed by the bundled NVA Realtime runtime. In particular,
# provider, TURN, public-facade, and backend secrets from a shared .env do not
# cross this process boundary.
_NVA_ENVIRONMENT = frozenset(
    {
        "APP_RUNTIME",
        "ASR_PREWARM_RPC_TIMEOUT_SECS",
        "AUDIO_DUMP_PATH",
        "AUDIO_OUT_10MS_CHUNKS",
        "CHAT_HISTORY_RECENT_TURNS",
        "CONNECT_PREWARM_TIMEOUT_SECS",
        "CUDA_VISIBLE_DEVICES",
        "ENABLE_ASR_AUDIO_DUMP",
        "ENABLE_TRACING",
        "ENABLE_TTS_AUDIO_DUMP",
        "ENABLE_WELCOME_MESSAGE",
        "EXAMPLE_SELECTION",
        "HTTPS_PROXY",
        "HTTP_PROXY",
        "NLTK_DATA",
        "NVIDIA_API_KEY",
        "NVIDIA_DRIVER_CAPABILITIES",
        "NVIDIA_VISIBLE_DEVICES",
        "NVA_RUNTIME_CONFIG_DIR",
        "NO_PROXY",
        "OTEL_CONSOLE_EXPORT",
        "OTEL_EXPORTER_OTLP_ENDPOINT",
        "PIPELINE_IDLE_TIMEOUT_SECS",
        "PIPELINE_TLS",
        "REALTIME_API_KEY",
        "REALTIME_CLIENT_TOOL_CONTEXT_TIMEOUT_SECS",
        "REALTIME_CLIENT_TOOL_TIMEOUT_SECONDS",
        "REALTIME_CONTEXT_APPLY_TIMEOUT_SECONDS",
        "REALTIME_INITIAL_EVENT_TIMEOUT_SECS",
        "REALTIME_INPUT_TRANSCRIPTION_TIMEOUT_SECONDS",
        "REALTIME_MAX_REJECTED_EVENTS",
        "REALTIME_SERVICE_PLATFORM",
        "SERVICES_CLOUD_PATH",
        "SERVICES_LOCAL_PATH",
        "SILERO_VAD_STOP_SECS",
        "SMART_TURN_STOP_SECS",
        "TOOLS_FILE_PATH",
        "TRANSPORT_SELECTION",
        "TTS_IPA_FILE_PATH",
        "TTS_PREWARM_RPC_TIMEOUT_SECS",
        "USE_SILERO_VAD_TURN_DETECTION",
        "UVICORN_WORKERS",
        "http_proxy",
        "https_proxy",
        "no_proxy",
    }
)
_FACADE_ENVIRONMENT = frozenset(
    {
        "VOICECLAW_TLS_CERTFILE",
        "VOICECLAW_TLS_KEYFILE",
    }
)


def _require_container_runtime(marker: Path = Path(_CONTAINER_RUNTIME_MARKER)) -> None:
    """Reject the image supervisor when invoked from an ordinary wheel install."""
    if not marker.is_file():
        raise ConfigurationError(
            "voiceclaw.container is image-internal; use the 'voiceclaw' command for a package installation"
        )


def _listener_tls_environment(environ: Mapping[str, str]) -> tuple[str, str]:
    """Resolve explicit TLS paths or the optional conventional operator mount."""
    certificate = environ.get("VOICECLAW_TLS_CERTFILE", "").strip()
    private_key = environ.get("VOICECLAW_TLS_KEYFILE", "").strip()
    if certificate or private_key:
        return certificate, private_key
    operator_files = Path(environ.get("VOICECLAW_OPERATOR_FILES_DIR", _DEFAULT_OPERATOR_FILES_DIR))
    conventional_certificate = operator_files / "tls" / "cert.pem"
    conventional_private_key = operator_files / "tls" / "key.pem"
    if conventional_certificate.exists() or conventional_private_key.exists():
        return str(conventional_certificate), str(conventional_private_key)
    return "", ""


def _healthcheck_host(listener_host: str) -> str:
    """Return an address reachable from inside the container namespace."""
    host = listener_host.strip().strip("[]")
    if host == "0.0.0.0":
        return "127.0.0.1"
    if host == "::":
        return "::1"
    return host


def _write_healthcheck_target(
    destination: Path,
    *,
    listener_host: str,
    listener_port: int,
    tls_enabled: bool,
) -> None:
    """Atomically publish the effective non-secret listener coordinates."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}")
    payload = {
        "schema": "voiceclaw.healthcheck.v1",
        "scheme": "https" if tls_enabled else "http",
        "host": _healthcheck_host(listener_host),
        "port": listener_port,
    }
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(temporary, flags, 0o644)
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, separators=(",", ":"), sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        with suppress(FileNotFoundError):
            temporary.unlink()


def _snapshot_configuration(
    source: Path,
    destination: Path,
    *,
    owner_uid: int,
    facade_gid: int,
) -> Path:
    """Copy one config read into a root-owned immutable-per-run snapshot."""
    read_flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        source_descriptor = os.open(source, read_flags)
    except OSError as error:
        raise ConfigurationError("VoiceClaw configuration could not be snapshotted") from error
    try:
        metadata = os.fstat(source_descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise ConfigurationError("VoiceClaw configuration must be a regular file")
        with os.fdopen(source_descriptor, "rb") as stream:
            source_descriptor = -1
            payload = stream.read(_MAX_CONFIG_BYTES + 1)
    finally:
        if source_descriptor >= 0:
            os.close(source_descriptor)
    if len(payload) > _MAX_CONFIG_BYTES:
        raise ConfigurationError(f"VoiceClaw configuration must not exceed {_MAX_CONFIG_BYTES} bytes")

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}")
    write_flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    write_flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(temporary, write_flags, 0o640)
        try:
            os.fchmod(descriptor, 0o640)
            os.fchown(descriptor, owner_uid, facade_gid)
            with os.fdopen(descriptor, "wb") as stream:
                descriptor = -1
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
        finally:
            if descriptor >= 0:
                os.close(descriptor)
        os.replace(temporary, destination)
    except OSError as error:
        raise ConfigurationError("VoiceClaw configuration snapshot could not be created") from error
    finally:
        with suppress(FileNotFoundError):
            temporary.unlink()
    return destination


def _read_healthcheck_target(source: Path) -> tuple[str, str, int]:
    """Load and validate coordinates written by the image supervisor."""
    try:
        raw = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ConfigurationError("container healthcheck target is unavailable") from error
    if not isinstance(raw, dict) or raw.get("schema") != "voiceclaw.healthcheck.v1":
        raise ConfigurationError("container healthcheck target has an unsupported schema")
    scheme = raw.get("scheme")
    host = raw.get("host")
    port = raw.get("port")
    if scheme not in {"http", "https"}:
        raise ConfigurationError("container healthcheck scheme must be http or https")
    if not isinstance(host, str) or not host or any(character.isspace() for character in host):
        raise ConfigurationError("container healthcheck host is invalid")
    if isinstance(port, bool) or not isinstance(port, int) or not 1 <= port <= 65_535:
        raise ConfigurationError("container healthcheck port is invalid")
    return scheme, host, port


def _run_healthcheck(source: Path, *, timeout: float = 2.5) -> None:
    """Probe the effective facade listener without proxies or credentials."""
    scheme, host, port = _read_healthcheck_target(source)
    connection: http.client.HTTPConnection
    if scheme == "https":
        context = ssl.create_default_context()
        context.check_hostname = False
        context.verify_mode = ssl.CERT_NONE
        connection = http.client.HTTPSConnection(host, port, timeout=timeout, context=context)
    else:
        connection = http.client.HTTPConnection(host, port, timeout=timeout)
    try:
        connection.request("GET", "/livez", headers={"Connection": "close"})
        response = connection.getresponse()
        response.read()
        if response.status != 200:
            raise RuntimeError(f"facade liveness endpoint returned HTTP {response.status}")
    finally:
        connection.close()


def _prepare_environment(
    environ: Mapping[str, str],
    *,
    key_factory: Callable[[], str] | None = None,
) -> dict[str, str]:
    """Return child environment with one private key shared only on loopback."""
    prepared = dict(environ)
    upstream_key = prepared.get("REALTIME_UPSTREAM_API_KEY", "").strip()
    nva_key = prepared.get("REALTIME_API_KEY", "").strip()
    if upstream_key and nva_key and not secrets.compare_digest(upstream_key, nva_key):
        raise ConfigurationError("REALTIME_UPSTREAM_API_KEY and REALTIME_API_KEY must match")
    internal_key = upstream_key or nva_key or (key_factory or (lambda: secrets.token_urlsafe(32)))()
    if not internal_key:
        raise ConfigurationError("internal Realtime credential generation failed")
    prepared["REALTIME_UPSTREAM_API_KEY"] = internal_key
    prepared["REALTIME_API_KEY"] = internal_key
    prepared["PIPELINE_TLS"] = "false"
    prepared["UVICORN_WORKERS"] = "1"
    return prepared


def _frontend_credential_environment_names(config: VoiceClawConfig) -> frozenset[str]:
    """Return every env credential source declared by a frontend profile."""
    names: set[str] = set()
    for profile in config.frontend_profiles.values():
        if str(profile.kind) == "openai_realtime":
            credential = getattr(profile, "credential", None)
            if credential is not None and credential.env is not None:
                names.add(credential.env)
            continue
        services = getattr(profile, "services", None)
        if services is None:
            continue
        for service in (services.llm, services.asr, services.tts):
            credential = service.credential
            if credential is not None and credential.env is not None:
                names.add(credential.env)
    return frozenset(names)


def _selected_external_credential_environment_name(config: VoiceClawConfig) -> str | None:
    """Return the selected external Realtime credential env source, if any."""
    selected = config.selected_frontend
    if selected is None or str(selected.kind) != "openai_realtime":
        return None
    credential = getattr(selected, "credential", None)
    return None if credential is None else credential.env


def _nva_environment(environ: Mapping[str, str], config: VoiceClawConfig) -> dict[str, str]:
    """Build an allowlisted environment for the private Realtime child."""
    prepared = {name: environ[name] for name in _PROCESS_ENVIRONMENT | _NVA_ENVIRONMENT if name in environ}
    facade_only = {
        config.server.api_key_env,
        *(profile.credential_env for profile in config.backend_profiles.values()),
        "VOICECLAW_REALTIME_API_KEY",
        "REALTIME_UPSTREAM_API_KEY",
        "NEMOCLAW_VOICE_GATEWAY_BEARER",
        "NEMOCLAW_VOICE_GATEWAY_BEARER_FILE",
        "VOICECLAW_TLS_CERTFILE",
        "VOICECLAW_TLS_KEYFILE",
        *_frontend_credential_environment_names(config),
    }
    for name in facade_only:
        if name:
            prepared.pop(name, None)
    prepared["HOME"] = "/home/voiceclaw-nva"
    prepared["XDG_CACHE_HOME"] = "/var/cache/voiceclaw-nva"
    return prepared


def _facade_environment(
    environ: Mapping[str, str],
    config: VoiceClawConfig,
    config_path: Path,
) -> dict[str, str]:
    """Build an allowlisted environment for the public VoiceClaw child.

    Environment names explicitly referenced by the selected YAML are part of
    that operator-authored configuration. Everything else from a shared
    Compose ``env_file`` is excluded unless the facade itself requires it.
    """
    try:
        raw_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise ConfigurationError("VoiceClaw configuration could not be inspected") from error
    references = configuration_environment_references(raw_config)
    selected_backend = config.backend_profiles[config.default_backend]
    frontend_credential_names = _frontend_credential_environment_names(config)
    selected_external_credential = _selected_external_credential_environment_name(config)
    reserved_external_credential_names = (_PROCESS_ENVIRONMENT | _FACADE_ENVIRONMENT | _NVA_ENVIRONMENT) - {
        # These are ordinary provider-key names when an external Realtime
        # profile is selected; no private NVA child is launched in that mode.
        "NVIDIA_API_KEY",
        "REALTIME_API_KEY",
    }
    if selected_external_credential in reserved_external_credential_names:
        raise ConfigurationError(
            "the selected external Realtime credential env must not reuse a facade operational variable"
        )
    configured_credentials = {
        config.server.api_key_env,
        config.realtime.credential_env if config.realtime is not None else None,
        selected_backend.credential_env,
    }
    active_credential_names = configured_credentials | {selected_external_credential}
    allowed = (
        _PROCESS_ENVIRONMENT
        | _FACADE_ENVIRONMENT
        | references
        | {name for name in configured_credentials if name is not None}
    )
    prepared = {name: environ[name] for name in allowed if name in environ}
    for name in frontend_credential_names:
        if name not in active_credential_names:
            prepared.pop(name, None)
    prepared.update(
        {
            "HOME": "/home/voiceclaw",
            "VOICECLAW_CONFIG": str(config_path),
            "XDG_CACHE_HOME": "/var/cache/voiceclaw",
        }
    )
    return prepared


def _identity(name: str, *, extra_groups: tuple[int, ...] = ()) -> dict[str, object]:
    """Return an explicit least-privilege subprocess identity."""
    try:
        account = pwd.getpwnam(name)
    except KeyError as error:
        raise ConfigurationError(f"required container account is missing: {name}") from error
    return {"user": account.pw_uid, "group": account.pw_gid, "extra_groups": extra_groups}


def _prepare_managed_volume_layout(
    root: Path,
    facade_identity: Mapping[str, object],
    *,
    root_uid: int = 0,
    root_gid: int = 0,
) -> None:
    """Idempotently establish one volume's config, credential, and state boundaries."""

    def prepare(path: Path, *, uid: int, gid: int, mode: int) -> None:
        try:
            path.mkdir(mode=mode, parents=False, exist_ok=True)
            metadata = os.lstat(path)
            if not stat.S_ISDIR(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode):
                raise ConfigurationError(f"managed volume path must be a directory: {path}")
            current_mode = stat.S_IMODE(metadata.st_mode)
            if metadata.st_uid == uid and metadata.st_gid == gid and current_mode == mode:
                return
            # With the managed container's narrow capability set, UID 0 has
            # CAP_CHOWN but intentionally lacks CAP_FOWNER. Reclaim ownership
            # before changing mode, then assign the final child identity.
            if metadata.st_uid != root_uid or metadata.st_gid != root_gid:
                os.chown(path, root_uid, root_gid)
            if current_mode != mode:
                os.chmod(path, mode)
            if uid != root_uid or gid != root_gid:
                os.chown(path, uid, gid)
        except OSError as error:
            raise ConfigurationError(f"managed volume path could not be secured: {path}") from error

    prepare(root, uid=root_uid, gid=root_gid, mode=0o755)
    prepare(root / "config", uid=root_uid, gid=root_gid, mode=0o700)
    # The facade may traverse this directory to open a specifically configured
    # group-readable agent credential, but it cannot list, create, replace, or
    # remove entries. Speech credentials remain root-only regular files and are
    # copied into an ephemeral NVA-only location before either child starts.
    prepare(
        root / "credentials",
        uid=root_uid,
        gid=int(facade_identity["group"]),
        mode=0o710,
    )
    prepare(
        root / "state",
        uid=int(facade_identity["user"]),
        gid=int(facade_identity["group"]),
        mode=0o700,
    )


_STATE_ACCESS_CHECK = """
import os
import stat
import sys
from pathlib import Path

state_path = Path(sys.argv[1])
parent = state_path.parent
parent_metadata = parent.stat(follow_symlinks=False)
parent_mode = stat.S_IMODE(parent_metadata.st_mode)
usable = (
    parent.is_dir()
    and not parent.is_symlink()
    and parent_metadata.st_uid == os.geteuid()
    and not parent_mode & (stat.S_IRWXG | stat.S_IRWXO)
    and parent_mode & (stat.S_IWUSR | stat.S_IXUSR) == stat.S_IWUSR | stat.S_IXUSR
    and os.access(parent, os.W_OK | os.X_OK)
)
if state_path.exists() or state_path.is_symlink():
    metadata = state_path.stat(follow_symlinks=False)
    usable = (
        usable
        and state_path.is_file()
        and not state_path.is_symlink()
        and metadata.st_uid == os.geteuid()
        and not stat.S_IMODE(metadata.st_mode) & (stat.S_IRWXG | stat.S_IRWXO)
        and os.access(state_path, os.R_OK | os.W_OK)
    )
raise SystemExit(0 if usable else 1)
"""

_STATE_ISOLATION_CHECK = """
import os
import sys
from pathlib import Path

state_path = Path(sys.argv[1])
# An identity that cannot traverse the owner-only state directory is isolated.
# Do not probe the database below that boundary: ``Path.exists()`` may raise
# ``PermissionError`` (rather than returning ``False``) on supported Python
# versions, which must not be confused with successful access.
raise SystemExit(1 if os.access(state_path.parent, os.X_OK) else 0)
"""


def _assert_state_path_usable(
    state_path: str,
    identity: Mapping[str, object],
    *,
    forbidden_identity: Mapping[str, object] | None = None,
) -> None:
    """Require private SQLite storage writable only by the facade identity."""
    path = Path(state_path)
    if not path.is_absolute():
        raise ConfigurationError("container state.path must be absolute")
    if not path.parent.is_dir():
        raise ConfigurationError("container state.path parent directory does not exist")
    completed = subprocess.run(
        [sys.executable, "-c", _STATE_ACCESS_CHECK, str(path)],
        check=False,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env={"PYTHONDONTWRITEBYTECODE": "1"},
        user=int(identity["user"]),
        group=int(identity["group"]),
        extra_groups=tuple(int(group) for group in identity["extra_groups"]),
    )
    if completed.returncode != 0:
        raise ConfigurationError(
            "container state.path must be an owner-only regular SQLite file in an owner-only directory "
            "writable by the voiceclaw identity; use directory mode 0700 and file mode 0600"
        )
    if forbidden_identity is None:
        return
    isolated = subprocess.run(
        [sys.executable, "-c", _STATE_ISOLATION_CHECK, str(path)],
        check=False,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env={"PYTHONDONTWRITEBYTECODE": "1"},
        user=int(forbidden_identity["user"]),
        group=int(forbidden_identity["group"]),
        extra_groups=tuple(int(group) for group in forbidden_identity["extra_groups"]),
    )
    if isolated.returncode != 0:
        raise ConfigurationError("container state.path is accessible to the bundled realtime-model identity")


_FILE_READABILITY_CHECK = """
import os
import sys
from pathlib import Path

for raw_path in sys.argv[1:]:
    path = Path(raw_path)
    if not path.is_file() or not os.access(path, os.R_OK):
        raise SystemExit(1)
"""


_FILE_MUTABILITY_CHECK = """
import os
import sys
from pathlib import Path

path = Path(sys.argv[1])
mutable = os.access(path, os.W_OK) or os.access(path.parent, os.W_OK | os.X_OK)
raise SystemExit(1 if mutable else 0)
"""


def _assert_files_readable_by_identity(
    paths: list[str],
    identity: Mapping[str, object],
    *,
    identity_label: str,
) -> None:
    """Check mounted operational files as the child that consumes them."""
    selected = list(dict.fromkeys(path for path in paths if path))
    if not selected:
        return
    completed = subprocess.run(
        [sys.executable, "-c", _FILE_READABILITY_CHECK, *selected],
        check=False,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env={"PYTHONDONTWRITEBYTECODE": "1"},
        user=int(identity["user"]),
        group=int(identity["group"]),
        extra_groups=tuple(int(group) for group in identity["extra_groups"]),
    )
    if completed.returncode != 0:
        raise ConfigurationError(f"configured files must be regular and readable by the {identity_label}")


def _assert_not_readable_by_identity(
    path: str,
    identity: Mapping[str, object],
    label: str,
    *,
    identity_label: str = "realtime-model identity",
) -> None:
    """Reject a mounted secret whose mode would expose it to one child identity."""
    try:
        metadata = os.lstat(path)
    except OSError as error:
        raise ConfigurationError(f"{label} could not be inspected") from error
    if not stat.S_ISREG(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode):
        raise ConfigurationError(f"{label} must be a regular file")
    getxattr = getattr(os, "getxattr", None)
    if getxattr is not None:
        try:
            access_acl = getxattr(path, "system.posix_acl_access", follow_symlinks=False)
        except OSError as error:
            no_acl_errors = {
                errno.ENODATA,
                errno.ENOTSUP,
                getattr(errno, "ENOATTR", errno.ENODATA),
            }
            if error.errno not in no_acl_errors:
                raise ConfigurationError(f"{label} POSIX access ACL could not be inspected") from error
        else:
            if access_acl:
                raise ConfigurationError(f"{label} must not have a POSIX access ACL")
    uid = int(identity["user"])
    gids = {int(identity["group"]), *(int(group) for group in identity.get("extra_groups", ()))}
    readable = bool(metadata.st_mode & stat.S_IROTH)
    readable = readable or (metadata.st_uid == uid and bool(metadata.st_mode & stat.S_IRUSR))
    readable = readable or (metadata.st_gid in gids and bool(metadata.st_mode & stat.S_IRGRP))
    if readable:
        raise ConfigurationError(f"{label} is readable by the {identity_label}")


def _assert_not_mutable_by_identity(
    path: str,
    identity: Mapping[str, object],
    label: str,
    *,
    identity_label: str,
) -> None:
    """Reject a secret that a child can rewrite, replace, rename, or unlink."""
    uid = int(identity["user"])
    gid = int(identity["group"])
    extra_groups = tuple(int(group) for group in identity["extra_groups"])
    identity_arguments: dict[str, object] = {}
    if uid != os.geteuid() or gid != os.getegid():
        identity_arguments = {"user": uid, "group": gid, "extra_groups": extra_groups}
    completed = subprocess.run(
        [sys.executable, "-c", _FILE_MUTABILITY_CHECK, path],
        check=False,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env={"PYTHONDONTWRITEBYTECODE": "1"},
        **identity_arguments,
    )
    if completed.returncode != 0:
        raise ConfigurationError(f"{label} is mutable by the {identity_label}")


def _assert_managed_credential_ancestry(path: Path) -> None:
    """Require root custody for credentials projected into the managed volume."""
    protected_root = Path(_DEFAULT_MANAGED_CREDENTIALS_DIR)
    absolute = Path(os.path.abspath(path))
    if not absolute.is_relative_to(protected_root):
        return
    for directory in (absolute.parent, *absolute.parent.parents):
        if not directory.is_relative_to(protected_root) and directory != protected_root:
            break
        try:
            metadata = os.lstat(directory)
        except OSError as error:
            raise ConfigurationError("managed credential ancestry could not be inspected") from error
        mode = stat.S_IMODE(metadata.st_mode)
        if (
            not stat.S_ISDIR(metadata.st_mode)
            or stat.S_ISLNK(metadata.st_mode)
            or metadata.st_uid != 0
            or mode & (stat.S_IWGRP | stat.S_IWOTH)
        ):
            raise ConfigurationError("managed credential directories must be root-owned and non-writable")
        if directory == protected_root:
            break


def _assert_owner_only_secret(path: str, label: str) -> None:
    """Require a provider secret file to expose no group or other permissions."""
    try:
        metadata = os.lstat(path)
    except OSError as error:
        raise ConfigurationError(f"{label} could not be inspected") from error
    if not stat.S_ISREG(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode):
        raise ConfigurationError(f"{label} must be a regular file")
    if stat.S_IMODE(metadata.st_mode) & (stat.S_IRWXG | stat.S_IRWXO):
        raise ConfigurationError(f"{label} must have owner-only permissions")
    managed_path = Path(os.path.abspath(path))
    if managed_path.is_relative_to(Path(_DEFAULT_MANAGED_CREDENTIALS_DIR)):
        if metadata.st_uid != 0:
            raise ConfigurationError(f"{label} in the managed credential directory must be root-owned")
        _assert_managed_credential_ancestry(managed_path)


def _assert_managed_facade_secret(
    path: str,
    facade_identity: Mapping[str, object],
    label: str,
) -> None:
    """Require managed facade secrets to be root-owned and group-readable."""
    managed_path = Path(os.path.abspath(path))
    if not managed_path.is_relative_to(Path(_DEFAULT_MANAGED_CREDENTIALS_DIR)):
        return
    try:
        metadata = os.lstat(managed_path)
    except OSError as error:
        raise ConfigurationError(f"{label} could not be inspected") from error
    mode = stat.S_IMODE(metadata.st_mode)
    if (
        not stat.S_ISREG(metadata.st_mode)
        or stat.S_ISLNK(metadata.st_mode)
        or metadata.st_uid != 0
        or metadata.st_gid != int(facade_identity["group"])
        or not mode & stat.S_IRGRP
        or mode & (stat.S_IWGRP | stat.S_IXGRP | stat.S_IRWXO)
    ):
        raise ConfigurationError(
            f"{label} in the managed credential directory must be root-owned and facade-group-readable"
        )
    _assert_managed_credential_ancestry(managed_path)


def _configured_frontend_secret_files(config: VoiceClawConfig) -> tuple[set[str], set[str]]:
    """Return configured facade-role and bundled-provider credential files."""
    facade_files: set[str] = set()
    provider_files: set[str] = set()
    for profile in config.frontend_profiles.values():
        if str(profile.kind) == "openai_realtime":
            credential = getattr(profile, "credential", None)
            if credential is not None and credential.file is not None:
                facade_files.add(credential.file)
            continue
        services = getattr(profile, "services", None)
        if services is None:
            continue
        for service in (services.llm, services.asr, services.tts):
            credential = service.credential
            if credential is not None and credential.file is not None:
                provider_files.add(credential.file)
    return facade_files, provider_files


def _selected_facade_secret_files(config: VoiceClawConfig, private_key: str | None) -> set[str]:
    """Return secrets consumed by the selected public-facade composition."""
    selected_backend = config.backend_profiles[config.default_backend]
    selected = {
        selected_backend.credential_file,
        config.server.api_key_file,
        config.realtime.credential_file if config.realtime is not None else None,
        private_key,
    }
    selected_frontend = config.selected_frontend
    if selected_frontend is not None and str(selected_frontend.kind) == "openai_realtime":
        credential = getattr(selected_frontend, "credential", None)
        if credential is not None:
            selected.add(credential.file)
    return {path for path in selected if path is not None}


def _assert_provider_files_isolated(
    config: VoiceClawConfig,
    facade_identity: Mapping[str, object],
) -> tuple[str, ...]:
    """Enforce provider-file isolation even when bundled NVA is not selected."""
    _external_frontend_files, provider_files = _configured_frontend_secret_files(config)
    existing = tuple(sorted(path for path in provider_files if Path(path).exists()))
    for provider_file in existing:
        _assert_owner_only_secret(provider_file, "bundled frontend credential file")
        _assert_not_readable_by_identity(
            provider_file,
            facade_identity,
            "bundled frontend credential file",
            identity_label="facade identity",
        )
        _assert_not_mutable_by_identity(
            provider_file,
            facade_identity,
            "bundled frontend credential file",
            identity_label="facade identity",
        )
    return existing


def _stage_nva_file_credential(
    credential: CredentialReference,
    destination: Path,
    identity: Mapping[str, object],
) -> CredentialReference:
    """Copy one provider secret into an ephemeral NVA-only regular file."""
    if credential.file is None:
        return credential
    value = resolve_credential_value(
        credential,
        source_environment={},
        label="frontend service credential",
    )
    if value is None:  # pragma: no cover - a file reference always resolves or raises
        raise ConfigurationError("frontend service credential is unavailable")
    try:
        destination.parent.mkdir(mode=0o755, parents=True, exist_ok=True)
        if destination.parent.is_symlink() or not destination.parent.is_dir():
            raise OSError("credential destination is not a directory")
        temporary = destination.parent / f".{destination.name}.{secrets.token_hex(12)}"
        descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0),
            0o600,
        )
        try:
            payload = value.encode("utf-8")
            written = 0
            while written < len(payload):
                written += os.write(descriptor, payload[written:])
            os.fsync(descriptor)
            os.fchown(descriptor, int(identity["user"]), int(identity["group"]))
            os.fchmod(descriptor, 0o400)
        finally:
            os.close(descriptor)
        os.replace(temporary, destination)
    except OSError as error:
        with suppress(OSError, UnboundLocalError):
            temporary.unlink()
        raise ConfigurationError("frontend service credential could not be staged securely") from error
    return CredentialReference(file=str(destination))


def _facade_secret_files_for_nva(config: VoiceClawConfig, private_key: str | None) -> set[str]:
    """Select required or existing facade-role files visible in the shared image."""
    files = {
        profile.credential_file
        for name, profile in config.backend_profiles.items()
        if profile.credential_file is not None
        and (name == config.default_backend or Path(profile.credential_file).exists())
    }
    external_frontend_files, _provider_files = _configured_frontend_secret_files(config)
    files.update(path for path in external_frontend_files if Path(path).exists())
    if config.server.api_key_file is not None:
        files.add(config.server.api_key_file)
    if private_key is not None:
        files.add(private_key)
    return files


def _wait_for_listener(process: subprocess.Popen[bytes], host: str, port: int, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        exit_code = process.poll()
        if exit_code is not None:
            raise RuntimeError(f"NVA Realtime frontend exited before readiness (status {exit_code})")
        try:
            with socket.create_connection((host, port), timeout=0.5):
                return
        except OSError:
            time.sleep(0.1)
    raise RuntimeError("NVA Realtime frontend did not become ready before the deadline")


_NVA_PROFILE_PREFLIGHT = """
import sys
from pathlib import Path

sys.path.insert(0, str(Path(sys.argv[1]).resolve().parent))
import examples_registry
import yaml

examples_registry.resolve_realtime_model_profile(sys.argv[2])
prompt_catalog = yaml.safe_load(Path(sys.argv[3]).read_text(encoding="utf-8"))
prompt = prompt_catalog.get(sys.argv[4]) if isinstance(prompt_catalog, dict) else None
if (
    not isinstance(prompt, dict)
    or not isinstance(prompt.get("content"), str)
    or not prompt["content"].strip()
    or prompt.get("internal") is True
):
    raise RuntimeError("configured prompt_key is unavailable or non-public")
"""


_FACADE_CONFIGURATION_PREFLIGHT = """
import os
import sys
from pathlib import Path

from voiceclaw.config import load_config
from voiceclaw.server import _with_listener_host, validate_configuration

config = _with_listener_host(
    load_config(Path(sys.argv[1]), environ=os.environ),
    sys.argv[2] or None,
)
validate_configuration(
    config,
    environ=os.environ,
    bundled_nva_supervised=sys.argv[3] == "1",
)
"""


def _preflight_facade_configuration(
    config_path: Path,
    listener_host: str,
    environment: Mapping[str, str],
    identity: Mapping[str, object],
    *,
    bundled_nva_supervised: bool,
) -> None:
    """Validate adapters and facade semantics under the facade security boundary."""
    try:
        completed = subprocess.run(
            [
                sys.executable,
                "-c",
                _FACADE_CONFIGURATION_PREFLIGHT,
                str(config_path),
                listener_host,
                "1" if bundled_nva_supervised else "0",
            ],
            check=False,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env=dict(environment),
            timeout=20,
            user=int(identity["user"]),
            group=int(identity["group"]),
            extra_groups=tuple(int(group) for group in identity["extra_groups"]),
        )
    except subprocess.TimeoutExpired as error:
        raise ConfigurationError("facade configuration preflight timed out") from error
    if completed.returncode != 0:
        raise ConfigurationError("the facade identity rejected the selected VoiceClaw configuration")


def _preflight_nva_profile(
    nva_python: Path,
    nva_server: Path,
    upstream_model: str,
    prompt_key: str,
    environment: Mapping[str, str],
    identity: Mapping[str, object],
) -> None:
    """Resolve the generated model through NVA before opening listeners."""
    prompt_catalog_path = Path(
        environment.get("PROMPT_FILE_PATH", "").strip() or nva_server.parent / "examples" / "generic" / "prompts.yaml"
    )
    try:
        completed = subprocess.run(
            [
                str(nva_python),
                "-c",
                _NVA_PROFILE_PREFLIGHT,
                str(nva_server),
                upstream_model,
                str(prompt_catalog_path),
                prompt_key,
            ],
            check=False,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env=dict(environment),
            timeout=20,
            user=int(identity["user"]),
            group=int(identity["group"]),
            extra_groups=tuple(int(group) for group in identity["extra_groups"]),
        )
    except subprocess.TimeoutExpired as error:
        raise ConfigurationError("bundled NVA profile preflight timed out") from error
    if completed.returncode != 0:
        raise ConfigurationError(
            "bundled NVA rejected the generated model profile; verify service selectors and prompt_key"
        )


def _terminate(processes: list[subprocess.Popen[bytes]]) -> None:
    for process in processes:
        if process.poll() is None:
            process.terminate()
    deadline = time.monotonic() + 10
    for process in processes:
        if process.poll() is not None:
            continue
        remaining = max(0.0, deadline - time.monotonic())
        try:
            process.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            process.kill()
    for process in processes:
        with suppress(OSError, subprocess.SubprocessError):
            process.wait(timeout=2)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the image-internal VoiceClaw process supervisor")
    parser.add_argument(
        "--config",
        type=Path,
        help="VoiceClaw YAML configuration (defaults to VOICECLAW_CONFIG or the packaged example)",
    )
    parser.add_argument("--ui", action="store_true", help="serve the optional VoiceClaw browser UI")
    parser.add_argument("--healthcheck", action="store_true", help="probe the effective container listener")
    return parser


def _facade_command(config_path: Path, host: str, *, ui: bool) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "voiceclaw.server",
        "--config",
        str(config_path),
        "--host",
        host,
    ]
    if ui:
        command.append("--ui")
    return command


def _frontend_runtime_plan(
    config: VoiceClawConfig,
    environ: Mapping[str, str],
    *,
    internal_endpoint: str,
) -> FrontendRuntimePlan | None:
    """Materialize a v3 frontend or preserve the legacy v2 bundled path."""
    profile = config.selected_frontend
    if profile is None:
        if config.realtime is None or config.realtime.upstream_endpoint != internal_endpoint:
            raise ConfigurationError(f"container realtime.upstream_endpoint must be {internal_endpoint}")
        return None
    runtime_dir = Path(environ.get("VOICECLAW_FRONTEND_RUNTIME_DIR", _DEFAULT_FRONTEND_RUNTIME_DIR))
    if not runtime_dir.is_absolute():
        raise ConfigurationError("VOICECLAW_FRONTEND_RUNTIME_DIR must be an absolute path")
    try:
        model_contracts = load_model_contract_catalog(
            config.model_contracts.path,
            profile=config.model_contracts.profile,
        )
    except ModelContractError as error:
        raise ConfigurationError(str(error)) from error
    plan = materialize_frontend_runtime(
        profile,
        runtime_dir,
        internal_endpoint=internal_endpoint,
        model_contracts=model_contracts,
    )
    realtime = config.realtime
    if (
        realtime is None
        or realtime.upstream_endpoint != plan.upstream_endpoint
        or realtime.upstream_model != plan.upstream_model
        or realtime.public_model != plan.public_model
    ):
        raise ConfigurationError("selected frontend did not produce the canonical Realtime binding")
    return plan


def main(argv: Sequence[str] | None = None) -> None:
    """Run the selected Realtime frontend, when bundled, and VoiceClaw."""
    arguments = _parser().parse_args(argv)
    try:
        _require_container_runtime()
    except ConfigurationError as error:
        raise SystemExit(f"VoiceClaw container configuration failed: {error}") from error
    if arguments.healthcheck:
        try:
            _run_healthcheck(Path(_DEFAULT_HEALTHCHECK_TARGET))
            return
        except (ConfigurationError, OSError, RuntimeError, http.client.HTTPException) as error:
            raise SystemExit(f"VoiceClaw container healthcheck failed: {error}") from error
    try:
        if os.geteuid() != 0:
            raise ConfigurationError("the container supervisor must start as root to isolate child identities")
        source_environment = dict(os.environ)
        source_config_path = arguments.config or Path(source_environment.get("VOICECLAW_CONFIG", _DEFAULT_CONFIG))
        inherited_groups = tuple(group for group in os.getgroups() if group != 0)
        facade_identity = _identity("voiceclaw", extra_groups=inherited_groups)
        _prepare_managed_volume_layout(Path("/var/lib/voiceclaw"), facade_identity)
        config_path = _snapshot_configuration(
            source_config_path,
            Path(_DEFAULT_CONFIG_SNAPSHOT),
            owner_uid=0,
            facade_gid=int(facade_identity["group"]),
        )
        config = _with_listener_host(
            load_config(config_path, environ=source_environment),
            source_environment.get("VOICECLAW_SERVER_HOST") or None,
        )
        selected_frontend = config.selected_frontend
        launch_bundled_nva = selected_frontend is None or str(selected_frontend.kind) == "bundled_nva"
        child_environment = _prepare_environment(source_environment) if launch_bundled_nva else source_environment
        public_key_name = config.server.api_key_env
        if (
            config.server.auth_mode == "ephemeral"
            and public_key_name is not None
            and not child_environment.get(public_key_name, "").strip()
        ):
            raise ConfigurationError(
                "the container requires the configured public Realtime master key in ephemeral auth mode"
            )
        tls_certificate, tls_private_key = _listener_tls_environment(child_environment)
        certificate, private_key = _tls_listener_files(config, tls_certificate, tls_private_key)
        if certificate is not None and private_key is not None:
            child_environment["VOICECLAW_TLS_CERTFILE"] = certificate
            child_environment["VOICECLAW_TLS_KEYFILE"] = private_key
        else:
            child_environment.pop("VOICECLAW_TLS_CERTFILE", None)
            child_environment.pop("VOICECLAW_TLS_KEYFILE", None)
        internal_port = _INTERNAL_REALTIME_PORT
        expected_origin = f"ws://127.0.0.1:{internal_port}/v1/realtime"
        frontend_plan = _frontend_runtime_plan(config, child_environment, internal_endpoint=expected_origin)
        launch_bundled_nva = frontend_plan is None or frontend_plan.launch_bundled_nva
        nva_identity = _identity("voiceclaw-nva") if launch_bundled_nva else None
        _assert_state_path_usable(config.state.path, facade_identity, forbidden_identity=nva_identity)
        selected_backend = config.backend_profiles[config.default_backend]
        facade_secret_files = _selected_facade_secret_files(config, private_key)
        facade_files = [str(config_path)]
        facade_files.extend(path for path in (certificate, private_key) if path is not None)
        facade_files.extend(
            path
            for path in (
                selected_backend.credential_file,
                config.server.api_key_file,
                config.realtime.credential_file if config.realtime is not None else None,
                config.model_contracts.path,
                config.interaction_profiles.path,
            )
            if path is not None
        )
        facade_files.extend(sorted(facade_secret_files))
        _assert_files_readable_by_identity(
            facade_files,
            facade_identity,
            identity_label="voiceclaw facade identity",
        )
        for secret_file in sorted(facade_secret_files):
            _assert_managed_facade_secret(secret_file, facade_identity, "facade credential file")
            _assert_not_mutable_by_identity(
                secret_file,
                facade_identity,
                "facade credential file",
                identity_label="voiceclaw facade identity",
            )
        existing_provider_files = _assert_provider_files_isolated(config, facade_identity)
        facade_environment = _facade_environment(child_environment, config, config_path)
        nva_command: list[str] | None = None
        nva_child_environment: dict[str, str] | None = None
        ready_timeout = 0.0
        if launch_bundled_nva:
            nva_server = Path(child_environment.get("VOICECLAW_NVA_SERVER", _DEFAULT_NVA_SERVER))
            nva_python = Path(child_environment.get("VOICECLAW_NVA_PYTHON", _DEFAULT_NVA_PYTHON))
            if not nva_server.is_file() or not nva_python.is_file():
                raise ConfigurationError("the bundled NVA Realtime runtime is missing")
            ready_timeout = float(child_environment.get("VOICECLAW_INTERNAL_READY_TIMEOUT", "90"))
            if not 1 <= ready_timeout <= 300:
                raise ConfigurationError("VOICECLAW_INTERNAL_READY_TIMEOUT must be between 1 and 300 seconds")
            assert nva_identity is not None
            sensitive_files = _facade_secret_files_for_nva(config, private_key)
            sensitive_files.add(str(config_path))
            for sensitive_index, sensitive_file in enumerate(sorted(sensitive_files), start=1):
                _assert_not_readable_by_identity(
                    sensitive_file,
                    nva_identity,
                    f"facade-only file {sensitive_index}",
                )
            for provider_file in existing_provider_files:
                _assert_not_readable_by_identity(
                    provider_file,
                    nva_identity,
                    "bundled frontend credential file",
                )
                _assert_not_mutable_by_identity(
                    provider_file,
                    nva_identity,
                    "bundled frontend credential file",
                    identity_label="bundled NVA identity",
                )
            nva_command = [
                str(nva_python),
                str(nva_server),
                "--host",
                "127.0.0.1",
                "--port",
                str(internal_port),
                "--workers",
                "1",
            ]
            nva_child_environment = _nva_environment(child_environment, config)
            if frontend_plan is not None:
                nva_child_environment.update(frontend_plan.nva_environment)
                nva_credential = frontend_plan.nva_credential
                if nva_credential is not None and nva_credential.file is not None:
                    if frontend_plan.registry_path is None:  # pragma: no cover - bundled plans always materialize it
                        raise ConfigurationError("bundled frontend runtime path is unavailable")
                    nva_credential = _stage_nva_file_credential(
                        nva_credential,
                        frontend_plan.registry_path.parent / "nvidia-api-key",
                        nva_identity,
                    )
                    assert nva_credential.file is not None
                    _assert_files_readable_by_identity(
                        [nva_credential.file],
                        nva_identity,
                        identity_label="bundled NVA identity",
                    )
                    _assert_not_readable_by_identity(
                        nva_credential.file,
                        facade_identity,
                        "staged frontend credential file",
                        identity_label="facade identity",
                    )
                nva_child_environment = bind_nva_credential(
                    nva_child_environment,
                    nva_credential,
                    source_environment=child_environment,
                )
                _preflight_nva_profile(
                    nva_python,
                    nva_server,
                    frontend_plan.upstream_model,
                    str(frontend_plan.prompt_key),
                    nva_child_environment,
                    nva_identity,
                )
        # Run child-owned validation only after every provider credential has
        # been copied into its immutable per-run location. A facade plugin can
        # no longer race the source file used by the NVA child.
        _preflight_facade_configuration(
            config_path,
            config.server.host,
            facade_environment,
            facade_identity,
            bundled_nva_supervised=launch_bundled_nva,
        )
        _write_healthcheck_target(
            Path(_DEFAULT_HEALTHCHECK_TARGET),
            listener_host=config.server.host,
            listener_port=config.server.port,
            tls_enabled=certificate is not None,
        )
    except (ConfigurationError, OSError, ValueError) as error:
        raise SystemExit(f"VoiceClaw container configuration failed: {error}") from error

    processes: list[subprocess.Popen[bytes]] = []
    stopping = False

    def stop(_signum: int, _frame: object) -> None:
        nonlocal stopping
        stopping = True
        for process in processes:
            if process.poll() is None:
                process.terminate()

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    try:
        if nva_command is not None and nva_child_environment is not None and nva_identity is not None:
            nva = subprocess.Popen(nva_command, env=nva_child_environment, **nva_identity)
            processes.append(nva)
            _wait_for_listener(nva, "127.0.0.1", internal_port, ready_timeout)
        facade = subprocess.Popen(
            _facade_command(config_path, config.server.host, ui=arguments.ui),
            env=facade_environment,
            **facade_identity,
        )
        processes.append(facade)

        exit_code = 0
        while not stopping:
            for process in processes:
                status = process.poll()
                if status is not None:
                    exit_code = status or 1
                    stopping = True
                    break
            if not stopping:
                time.sleep(0.25)
    except (OSError, RuntimeError) as error:
        print(f"VoiceClaw container failed: {error}", file=sys.stderr)
        exit_code = 1
    finally:
        _terminate(processes)
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
