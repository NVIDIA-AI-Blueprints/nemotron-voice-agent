#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""Verify an exact VoiceClaw image contract and optional black-box startup."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import stat
import subprocess
import sys
import tempfile
import time
import uuid
import zipfile
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

_IMAGE_ID = re.compile(r"^sha256:[0-9a-f]{64}$")
_REVISION = re.compile(r"^[0-9a-f]{40}$")
_MAX_WHEEL_BYTES = 16 * 1024 * 1024
_MAX_EVIDENCE_BYTES = 2 * 1024 * 1024
_MAX_PACKAGE_MEMBER_BYTES = 8 * 1024 * 1024
_MAX_PACKAGE_BYTES = 32 * 1024 * 1024
_EXPECTED_ENTRYPOINT = ["/usr/local/bin/voiceclaw-runtime"]
_EXPECTED_COMMAND = ["serve"]
_EXPECTED_HEALTHCHECK = ["CMD", "/usr/local/bin/voiceclaw-runtime", "healthcheck"]
_OCI_LABELS = {
    "org.opencontainers.image.title": "VoiceClaw",
    "org.opencontainers.image.licenses": "BSD-2-Clause",
}
_SECRET_NAME_PARTS = ("API_KEY", "BEARER", "CREDENTIAL", "PASSWORD", "SECRET", "TOKEN")
_INPUT_FILES = (
    "LICENSE",
    "examples_registry.yaml",
    "pyproject.toml",
    "src/examples/voiceclaw/Dockerfile",
    "src/examples/voiceclaw/Dockerfile.dockerignore",
    "src/examples/voiceclaw/pyproject.toml",
    "src/examples/voiceclaw/uv.lock",
    "third_party_oss_license.txt",
    "uv.lock",
)
_SMOKE_CONFIG = """\
schema_version: voiceclaw.config.v3
server:
  host: 127.0.0.1
  port: 18790
  listener_security: loopback
  auth_mode: none
  max_sessions: 1
frontend_profiles:
  smoke:
    kind: openai_realtime
    endpoint: ws://127.0.0.1:9/v1/realtime
    model: smoke/not-contacted
default_frontend: smoke
backend_profiles:
  unavailable_agent:
    kind: openshell_fabric
    settings:
      endpoint: http://127.0.0.1:9
      authentication: anonymous
      workspace: smoke-workspace
      sandbox: smoke-sandbox
      adapter_id: nvidia.fabric.openclaw
      fabric_agent: main
      rpc_timeout_seconds: 1
      invoke_timeout_seconds: 1
      check_timeout_seconds: 1
    interaction:
      profile: stateless
      tool_copy: {}
default_backend: unavailable_agent
state:
  kind: sqlite
  path: /var/lib/voiceclaw/state/state.db
"""
_CONTAINER_HARNESS_DIRECTORY = "/app/src/examples/voiceclaw/scripts"
_HTTP_PROBE = """\
import http.client
import sys

connection = http.client.HTTPConnection("127.0.0.1", int(sys.argv[1]), timeout=2)
connection.request("GET", sys.argv[2])
response = connection.getresponse()
response.read()
print(response.status)
"""


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", help="exact local image reference to verify")
    parser.add_argument("--expect-version", required=True, help="required OCI package version")
    parser.add_argument("--expect-revision", required=True, help="required full lowercase Git revision")
    parser.add_argument("--expect-source", required=True, help="required OCI source URL")
    parser.add_argument("--expect-architecture", help="required Docker architecture, for example arm64")
    parser.add_argument("--repository-root", type=Path, help="include digests for the build inputs below this root")
    parser.add_argument("--wheel", type=Path, help="exact verified wheel whose package payload must match the image")
    parser.add_argument("--wheel-evidence", type=Path, help="evidence emitted by verify-wheel.py for --wheel")
    parser.add_argument("--smoke", action="store_true", help="start and probe the exact image without the optional UI")
    parser.add_argument("--smoke-ui", action="store_true", help="also start and probe the opt-in packaged UI")
    parser.add_argument("--timeout", type=float, default=30.0, help="startup timeout in seconds (default: %(default)s)")
    return parser


def _docker(*arguments: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            ["docker", *arguments],
            check=check,
            capture_output=True,
            text=True,
            timeout=60,
        )
    except FileNotFoundError as error:
        raise RuntimeError("docker CLI is not installed") from error
    except subprocess.TimeoutExpired as error:
        raise RuntimeError("docker command exceeded its bounded timeout") from error
    except subprocess.CalledProcessError as error:
        detail = error.stderr.strip().splitlines()
        suffix = f": {detail[-1]}" if detail else ""
        raise RuntimeError(f"docker command failed{suffix}") from error


def _bounded_file_bytes(path: Path, maximum: int, label: str) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} must be one regular file")
    size = path.stat().st_size
    if size < 1 or size > maximum:
        raise ValueError(f"{label} exceeds its bounded size")
    data = path.read_bytes()
    if len(data) != size:
        raise ValueError(f"{label} changed while it was being read")
    return data


def _json_object(data: bytes, label: str) -> dict[str, Any]:
    def reject_duplicate(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for name, item in pairs:
            if name in value:
                raise ValueError(f"{label} contains a duplicate key")
            value[name] = item
        return value

    try:
        value = json.loads(data, object_pairs_hook=reject_duplicate)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is not valid UTF-8 JSON") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _package_manifest_sha256(members: list[dict[str, object]]) -> str:
    encoded = json.dumps(
        {"schema": "voiceclaw.package_manifest.v1", "members": sorted(members, key=lambda item: str(item["path"]))},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _wheel_package_members(wheel_bytes: bytes) -> list[dict[str, object]]:
    members: list[dict[str, object]] = []
    total = 0
    try:
        with zipfile.ZipFile(io.BytesIO(wheel_bytes)) as archive:
            names: set[str] = set()
            for info in archive.infolist():
                name = info.filename
                if info.is_dir() or not name.startswith("voiceclaw/"):
                    continue
                parts = name.split("/")
                mode = (info.external_attr >> 16) & 0xFFFF
                if (
                    name in names
                    or name.startswith("/")
                    or "\\" in name
                    or any(part in {"", ".", ".."} for part in parts)
                    or stat.S_ISLNK(mode)
                    or info.file_size > _MAX_PACKAGE_MEMBER_BYTES
                ):
                    raise ValueError("wheel contains an unsafe package member")
                names.add(name)
                total += info.file_size
                if total > _MAX_PACKAGE_BYTES:
                    raise ValueError("wheel package payload exceeds its bounded size")
                data = archive.read(info)
                members.append({"path": name, "sha256": hashlib.sha256(data).hexdigest(), "size": len(data)})
    except zipfile.BadZipFile as error:
        raise ValueError("wheel is not a valid ZIP archive") from error
    if not members:
        raise ValueError("wheel does not contain the VoiceClaw package")
    return members


def _verify_wheel_binding(
    wheel: Path | None,
    evidence_path: Path | None,
    *,
    version: str,
    revision: str,
    source_tree: str | None,
) -> dict[str, object] | None:
    if (wheel is None) != (evidence_path is None):
        raise ValueError("--wheel and --wheel-evidence must be supplied together")
    if wheel is None or evidence_path is None:
        return None
    wheel_bytes = _bounded_file_bytes(wheel, _MAX_WHEEL_BYTES, "wheel")
    evidence_bytes = _bounded_file_bytes(evidence_path, _MAX_EVIDENCE_BYTES, "wheel evidence")
    evidence = _json_object(evidence_bytes, "wheel evidence")
    package_manifest = _package_manifest_sha256(_wheel_package_members(wheel_bytes))
    expected_revision = revision if revision != "development" else None
    if (
        evidence.get("schema") != "voiceclaw.wheel_evidence.v1"
        or evidence.get("filename") != wheel.name
        or evidence.get("version") != version
        or evidence.get("sha256") != hashlib.sha256(wheel_bytes).hexdigest()
        or evidence.get("size") != len(wheel_bytes)
        or evidence.get("package_manifest_sha256") != package_manifest
        or evidence.get("source_revision") != expected_revision
        or evidence.get("source_tree") != source_tree
    ):
        raise ValueError("wheel and wheel evidence do not match the requested image source contract")
    return {
        "filename": wheel.name,
        "sha256": evidence["sha256"],
        "evidence_sha256": hashlib.sha256(evidence_bytes).hexdigest(),
        "package_manifest_sha256": package_manifest,
    }


def _image_package_manifest(image: str) -> tuple[str, int]:
    container = f"voiceclaw-package-{uuid.uuid4().hex}"
    with tempfile.TemporaryDirectory(prefix="voiceclaw-package-image-") as directory:
        destination = Path(directory)
        try:
            _docker("create", "--name", container, "--entrypoint", "/bin/true", image)
            _docker(
                "cp",
                f"{container}:/app/src/examples/voiceclaw/src/voiceclaw",
                str(destination),
            )
        finally:
            _docker("rm", "--force", "--volumes", container, check=False)
        package = destination / "voiceclaw"
        if package.is_symlink() or not package.is_dir():
            raise RuntimeError("image does not contain the VoiceClaw package payload")
        members: list[dict[str, object]] = []
        total = 0
        for path in sorted(package.rglob("*")):
            metadata = path.lstat()
            if stat.S_ISDIR(metadata.st_mode):
                continue
            if not stat.S_ISREG(metadata.st_mode) or metadata.st_size > _MAX_PACKAGE_MEMBER_BYTES:
                raise RuntimeError("image contains an unsafe VoiceClaw package member")
            total += metadata.st_size
            if total > _MAX_PACKAGE_BYTES:
                raise RuntimeError("image VoiceClaw package payload exceeds its bounded size")
            data = path.read_bytes()
            relative = path.relative_to(package).as_posix()
            members.append(
                {
                    "path": f"voiceclaw/{relative}",
                    "sha256": hashlib.sha256(data).hexdigest(),
                    "size": len(data),
                }
            )
        if not members:
            raise RuntimeError("image VoiceClaw package payload is empty")
        return _package_manifest_sha256(members), len(members)


def _inspect(image: str) -> dict[str, Any]:
    result = _docker("image", "inspect", image)
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as error:
        raise ValueError("docker image inspect returned malformed JSON") from error
    if not isinstance(payload, list) or len(payload) != 1 or not isinstance(payload[0], dict):
        raise ValueError("docker image inspect did not return exactly one image")
    return payload[0]


def _mapping(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"image {label} is missing or malformed")
    return value


def _verify_contract(
    inspect: dict[str, Any],
    *,
    version: str,
    revision: str,
    source: str,
    architecture: str | None = None,
) -> dict[str, Any]:
    image_id = inspect.get("Id")
    if not isinstance(image_id, str) or not _IMAGE_ID.fullmatch(image_id):
        raise ValueError("image does not have one immutable sha256 ID")
    if revision != "development" and not _REVISION.fullmatch(revision):
        raise ValueError("expected revision must be development or one full lowercase Git revision")
    parsed_source = urlsplit(source)
    if (
        parsed_source.scheme != "https"
        or not parsed_source.hostname
        or parsed_source.username is not None
        or parsed_source.password is not None
        or parsed_source.query
        or parsed_source.fragment
        or source != source.strip()
    ):
        raise ValueError("expected source must be one unpadded https URL")
    if inspect.get("Os") != "linux" or not isinstance(inspect.get("Architecture"), str):
        raise ValueError("VoiceClaw image must be a Linux image with a declared architecture")
    if architecture is not None and inspect.get("Architecture") != architecture:
        raise ValueError("VoiceClaw image architecture does not match the requested artifact platform")

    config = _mapping(inspect.get("Config"), "Config")
    image_user = config.get("User") or ""
    if image_user not in {"", "0", "0:0", "root"}:
        raise ValueError("runtime image must start its split-identity supervisor as root")
    if config.get("Entrypoint") != _EXPECTED_ENTRYPOINT or config.get("Cmd") != _EXPECTED_COMMAND:
        raise ValueError("image entrypoint or default command does not match the runtime contract")
    exposed_ports = _mapping(config.get("ExposedPorts"), "exposed ports")
    volumes = _mapping(config.get("Volumes"), "volumes")
    if set(exposed_ports) != {"7860/tcp"}:
        raise ValueError("runtime image must expose only 7860/tcp")
    if set(volumes) != {"/var/lib/voiceclaw"}:
        raise ValueError("runtime image must declare only the VoiceClaw state volume")
    if config.get("StopSignal") != "SIGTERM":
        raise ValueError("image stop signal does not match the runtime contract")
    healthcheck = _mapping(config.get("Healthcheck"), "healthcheck")
    if healthcheck.get("Test") != _EXPECTED_HEALTHCHECK:
        raise ValueError("image healthcheck does not use the runtime entrypoint")

    labels = _mapping(config.get("Labels"), "labels")
    expected_labels = {
        **_OCI_LABELS,
        "org.opencontainers.image.version": version,
        "org.opencontainers.image.revision": revision,
        "org.opencontainers.image.source": source,
    }
    for name, expected in expected_labels.items():
        if labels.get(name) != expected:
            raise ValueError(f"image label {name} does not match the verified build input")

    for item in config.get("Env") or []:
        if not isinstance(item, str) or "=" not in item:
            raise ValueError("image contains a malformed environment entry")
        name = item.split("=", 1)[0].upper()
        if any(part in name for part in _SECRET_NAME_PARTS):
            raise ValueError(f"image bakes a secret-like environment name: {name}")

    return {
        "id": image_id,
        "os": inspect["Os"],
        "architecture": inspect["Architecture"],
        "user": image_user,
        "labels": {name: labels[name] for name in sorted(expected_labels)},
    }


def _probe(container: str, path: str, *, port: int = 18790) -> int:
    result = _docker(
        "exec",
        container,
        "/app/src/examples/voiceclaw/.venv/bin/python",
        "-c",
        _HTTP_PROBE,
        str(port),
        path,
    )
    try:
        return int(result.stdout.strip())
    except ValueError as error:
        raise RuntimeError("image HTTP probe returned a malformed status") from error


def _wait_for_status(container: str, path: str, expected: int, timeout: float, *, port: int = 18790) -> None:
    deadline = time.monotonic() + timeout
    last_status: int | None = None
    while time.monotonic() < deadline:
        state = _docker("inspect", "--format", "{{.State.Status}}", container, check=False).stdout.strip()
        if state in {"dead", "exited"}:
            raise RuntimeError("VoiceClaw container exited before readiness")
        try:
            last_status = _probe(container, path, port=port)
        except RuntimeError:
            time.sleep(0.25)
            continue
        if last_status == expected:
            return
        time.sleep(0.25)
    raise RuntimeError(f"VoiceClaw image did not return HTTP {expected} for {path}; last status={last_status}")


def _package_version(container: str) -> str:
    result = _docker(
        "exec",
        container,
        "/app/src/examples/voiceclaw/.venv/bin/python",
        "-c",
        "from importlib.metadata import version; print(version('nemotron-voiceclaw'))",
    )
    return result.stdout.strip()


def _assert_test_harness_absent(container: str) -> None:
    result = _docker(
        "exec",
        container,
        "/app/src/examples/voiceclaw/.venv/bin/python",
        "-c",
        "import pathlib,sys; sys.exit(pathlib.Path(sys.argv[1]).exists())",
        _CONTAINER_HARNESS_DIRECTORY,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError("VoiceClaw image contains source-only artifact harness scripts")


def _stop_cleanly(container: str) -> None:
    _docker("stop", "--timeout", "10", container)
    result = _docker("inspect", "--format", "{{json .State}}", container)
    try:
        state = json.loads(result.stdout)
    except json.JSONDecodeError as error:
        raise RuntimeError("docker returned malformed container state after shutdown") from error
    if not isinstance(state, dict) or state.get("Status") != "exited" or state.get("ExitCode") != 0:
        status = state.get("Status") if isinstance(state, dict) else None
        exit_code = state.get("ExitCode") if isinstance(state, dict) else None
        raise RuntimeError(f"VoiceClaw container did not stop cleanly (status={status!r}, exit_code={exit_code!r})")


def _smoke(image: str, *, version: str, ui: bool, timeout: float) -> dict[str, int | bool | str]:
    if timeout <= 0 or timeout > 300:
        raise ValueError("smoke timeout must be greater than zero and no more than 300 seconds")
    name = f"voiceclaw-artifact-{uuid.uuid4().hex}"
    with tempfile.TemporaryDirectory(prefix="voiceclaw-image-") as directory:
        config = Path(directory) / "voiceclaw.yaml"
        config.write_text(_SMOKE_CONFIG, encoding="utf-8")
        arguments = [
            "run",
            "--detach",
            "--name",
            name,
            "--cap-drop",
            "ALL",
            "--cap-add",
            "CHOWN",
            "--cap-add",
            "DAC_OVERRIDE",
            "--cap-add",
            "KILL",
            "--cap-add",
            "SETGID",
            "--cap-add",
            "SETUID",
            "--security-opt",
            "no-new-privileges:true",
            "--mount",
            f"type=bind,src={config.resolve()},dst=/run/voiceclaw/config.yaml,readonly",
            "--mount",
            "type=volume,dst=/var/lib/voiceclaw",
            image,
            "serve",
            "--config",
            "/run/voiceclaw/config.yaml",
        ]
        if ui:
            arguments.append("--ui")
        try:
            _docker(*arguments)
            _wait_for_status(name, "/livez", 200, timeout)
            _wait_for_status(name, "/health", 200, timeout)
            _wait_for_status(name, "/readyz", 503, timeout)
            installed_version = _package_version(name)
            if installed_version != version:
                raise RuntimeError("installed package version does not match the verified image label")
            _assert_test_harness_absent(name)
            root_status = _probe(name, "/")
            expected_root = 200 if ui else 404
            if root_status != expected_root:
                raise RuntimeError(f"VoiceClaw image returned HTTP {root_status} for /; expected {expected_root}")
            if ui and _probe(name, "/app.js") != 200:
                raise RuntimeError("VoiceClaw image did not serve its packaged UI asset")
            result: dict[str, int | bool | str] = {
                "livez": 200,
                "health": 200,
                "readyz": 503,
                "root": root_status,
                "ui": ui,
                "package_version": installed_version,
                "source_harness_absent": True,
            }
            _stop_cleanly(name)
            return result
        finally:
            _docker("rm", "--force", "--volumes", name, check=False)


def _input_digests(root: Path | None) -> dict[str, str]:
    if root is None:
        return {}
    root = root.resolve()
    digests: dict[str, str] = {}
    for relative in _INPUT_FILES:
        path = root / relative
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"required image input does not exist: {relative}")
        digests[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    return digests


def _git(root: Path, *arguments: str) -> str:
    try:
        completed = subprocess.run(
            ["git", "-C", str(root), *arguments],
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except FileNotFoundError as error:
        raise RuntimeError("git CLI is not installed") from error
    except subprocess.TimeoutExpired as error:
        raise RuntimeError("git source verification exceeded its bounded timeout") from error
    except subprocess.CalledProcessError as error:
        raise RuntimeError("repository source state could not be verified") from error
    return completed.stdout.strip()


def _source_tree(root: Path | None, revision: str) -> str | None:
    if root is None:
        return None
    if not _REVISION.fullmatch(revision):
        raise ValueError("repository-backed evidence requires one full lowercase Git revision")
    root = root.resolve()
    if Path(_git(root, "rev-parse", "--show-toplevel")).resolve() != root:
        raise ValueError("repository root does not name the Git worktree root")
    if _git(root, "rev-parse", "HEAD") != revision:
        raise ValueError("repository HEAD does not match the expected image revision")
    if _git(root, "status", "--porcelain=v1", "--untracked-files=all"):
        raise ValueError("repository contains uncommitted build inputs")
    tree = _git(root, "rev-parse", "HEAD^{tree}")
    if not _REVISION.fullmatch(tree):
        raise ValueError("repository tree identity is malformed")
    return tree


def _assert_harness_origin(root: Path | None) -> None:
    if root is None:
        return
    expected = root.resolve() / "src/examples/voiceclaw/scripts/verify-image.py"
    if expected.is_symlink() or not expected.is_file() or Path(__file__).resolve() != expected:
        raise ValueError("artifact harness does not originate from the authenticated repository")


def main() -> int:
    """Validate the requested image and print canonical evidence JSON."""
    arguments = _parser().parse_args()
    try:
        source_tree = _source_tree(arguments.repository_root, arguments.expect_revision)
        _assert_harness_origin(arguments.repository_root)
        wheel = _verify_wheel_binding(
            arguments.wheel,
            arguments.wheel_evidence,
            version=arguments.expect_version,
            revision=arguments.expect_revision,
            source_tree=source_tree,
        )
        build_inputs = _input_digests(arguments.repository_root)
        image = _verify_contract(
            _inspect(arguments.image),
            version=arguments.expect_version,
            revision=arguments.expect_revision,
            source=arguments.expect_source,
            architecture=arguments.expect_architecture,
        )
        immutable_image = image["id"]
        if not isinstance(immutable_image, str):  # pragma: no cover - established by _verify_contract
            raise ValueError("verified image ID is malformed")
        image_package: dict[str, object] | None = None
        if wheel is not None:
            manifest_sha256, member_count = _image_package_manifest(immutable_image)
            if manifest_sha256 != wheel["package_manifest_sha256"]:
                raise ValueError("image VoiceClaw package payload does not match the verified wheel")
            image_package = {"manifest_sha256": manifest_sha256, "member_count": member_count}
        smoke: dict[str, object] = {}
        if arguments.smoke or arguments.smoke_ui:
            smoke["headless"] = _smoke(
                immutable_image,
                version=arguments.expect_version,
                ui=False,
                timeout=arguments.timeout,
            )
        if arguments.smoke_ui:
            smoke["ui"] = _smoke(
                immutable_image,
                version=arguments.expect_version,
                ui=True,
                timeout=arguments.timeout,
            )
        evidence = {
            "schema": "voiceclaw.image_evidence.v1",
            "reference": arguments.image,
            "source_revision": arguments.expect_revision,
            "source_tree": source_tree,
            "build_inputs": build_inputs,
            "wheel": wheel,
            "image_package": image_package,
            "image": image,
            "smoke": smoke,
        }
    except (OSError, RuntimeError, ValueError) as error:
        print(f"verify-image: {error}", file=sys.stderr)
        return 1
    print(json.dumps(evidence, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
