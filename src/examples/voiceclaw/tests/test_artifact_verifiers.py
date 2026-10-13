# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""Focused rejection tests for VoiceClaw release artifact verifiers."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import zipfile
from pathlib import Path

import pytest

from voiceclaw.composition import compose_backends
from voiceclaw.config import load_config


def _load_script(name: str):
    path = Path(__file__).parents[1] / "scripts" / name
    module_name = f"voiceclaw_{name.removesuffix('.py').replace('-', '_')}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_IMAGE_VERIFIER = _load_script("verify-image.py")
_WHEEL_VERIFIER = _load_script("verify-wheel.py")


def test_image_smoke_composes_the_packaged_response_only_adapter(tmp_path: Path) -> None:
    """The black-box image smoke must exercise the shipped OpenShell extra."""
    config_path = tmp_path / "voiceclaw.yaml"
    config_path.write_text(_IMAGE_VERIFIER._SMOKE_CONFIG, encoding="utf-8")

    config = load_config(config_path, environ={})
    composition = compose_backends(config, environ={})
    try:
        assert composition.turn_status == "response_only"
        assert composition.selected_agent_readiness is not None
    finally:
        composition.close()


def _zip(tmp_path: Path, name: str, members: dict[str, bytes]) -> Path:
    path = tmp_path / name
    with zipfile.ZipFile(path, "w") as archive:
        for member, contents in members.items():
            archive.writestr(member, contents)
    return path


def _source_contract(package_files: dict[str, bytes]):
    return _WHEEL_VERIFIER.SourceContract(
        tree="a" * 40,
        package_files=package_files,
        license_bytes=b"tracked license\n",
    )


def _image_inspect(
    *,
    source: str,
    environment: list[str] | None = None,
    user: str = "",
) -> dict[str, object]:
    return {
        "Id": "sha256:" + ("a" * 64),
        "Os": "linux",
        "Architecture": "amd64",
        "Config": {
            "Entrypoint": ["/usr/local/bin/voiceclaw-runtime"],
            "Cmd": ["serve"],
            "ExposedPorts": {"7860/tcp": {}},
            "Volumes": {"/var/lib/voiceclaw": {}},
            "StopSignal": "SIGTERM",
            "Healthcheck": {"Test": ["CMD", "/usr/local/bin/voiceclaw-runtime", "healthcheck"]},
            "Labels": {
                "org.opencontainers.image.title": "VoiceClaw",
                "org.opencontainers.image.licenses": "BSD-2-Clause",
                "org.opencontainers.image.version": "0.1.0",
                "org.opencontainers.image.revision": "development",
                "org.opencontainers.image.source": source,
            },
            "Env": environment or ["PATH=/usr/local/bin:/usr/bin:/bin"],
            "User": user,
        },
    }


def test_wheel_rejects_parent_directory_member(tmp_path: Path) -> None:
    """A wheel member cannot escape the extraction root."""
    wheel = tmp_path / "unsafe.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("../operator/backend-bearer", "not-a-real-secret")

    with zipfile.ZipFile(wheel) as archive, pytest.raises(ValueError, match="unsafe or duplicate member"):
        _WHEEL_VERIFIER._archive_files(archive)


def test_wheel_rejects_forbidden_private_payload() -> None:
    """A structurally complete package cannot smuggle private operator material."""
    dist_info = "nemotron_voiceclaw-0.1.0.dist-info/"
    names = {
        *_WHEEL_VERIFIER._REQUIRED_PACKAGE_FILES,
        *(f"{dist_info}{name}" for name in _WHEEL_VERIFIER._EXPECTED_DIST_INFO_MEMBERS),
        "voiceclaw/operator/backend-bearer.pem",
    }
    files = {name: zipfile.ZipInfo(name) for name in names}

    with pytest.raises(ValueError, match="development or private files"):
        _WHEEL_VERIFIER._contents(files, dist_info)


@pytest.mark.parametrize(
    "retired_member",
    [
        "voiceclaw/managed_runtime.py",
        "voiceclaw/resources/nemoclaw_managed.yaml",
        "voiceclaw/adapters/nemoclaw/committed_turn.py",
    ],
)
def test_wheel_rejects_retired_backend_payload(retired_member: str) -> None:
    """A stale local build directory cannot restore a retired adapter."""
    dist_info = "nemotron_voiceclaw-0.1.0.dist-info/"
    names = {
        *_WHEEL_VERIFIER._REQUIRED_PACKAGE_FILES,
        *(f"{dist_info}{name}" for name in _WHEEL_VERIFIER._EXPECTED_DIST_INFO_MEMBERS),
        retired_member,
    }
    files = {name: zipfile.ZipInfo(name) for name in names}

    with pytest.raises(ValueError, match="development or private files"):
        _WHEEL_VERIFIER._contents(files, dist_info)


def test_wheel_dist_info_inventory_is_exact() -> None:
    """Generated metadata has one closed, reviewable member set."""
    dist_info = "nemotron_voiceclaw-0.1.0.dist-info/"
    names = {
        *_WHEEL_VERIFIER._REQUIRED_PACKAGE_FILES,
        *(f"{dist_info}{name}" for name in _WHEEL_VERIFIER._EXPECTED_DIST_INFO_MEMBERS),
        f"{dist_info}direct_url.json",
    }
    files = {name: zipfile.ZipInfo(name) for name in names}

    with pytest.raises(ValueError, match=r"dist-info inventory.*unexpected direct_url\.json"):
        _WHEEL_VERIFIER._contents(files, dist_info)


@pytest.mark.parametrize("missing", ["licenses/LICENSE", "top_level.txt"])
def test_wheel_dist_info_requires_license_and_top_level(missing: str) -> None:
    """License and top-level metadata are mandatory members, not optional extras."""
    dist_info = "nemotron_voiceclaw-0.1.0.dist-info/"
    names = {
        *_WHEEL_VERIFIER._REQUIRED_PACKAGE_FILES,
        *(f"{dist_info}{name}" for name in _WHEEL_VERIFIER._EXPECTED_DIST_INFO_MEMBERS if name != missing),
    }
    files = {name: zipfile.ZipInfo(name) for name in names}

    with pytest.raises(ValueError, match=f"missing {missing}"):
        _WHEEL_VERIFIER._contents(files, dist_info)


@pytest.mark.parametrize(
    ("package_files", "wheel_files", "message"),
    [
        (
            {"voiceclaw/__init__.py": b"tracked\n"},
            {"voiceclaw/__init__.py": b"tracked\n", "voiceclaw/extra.py": b"extra\n"},
            "unexpected voiceclaw/extra.py",
        ),
        (
            {"voiceclaw/__init__.py": b"tracked\n", "voiceclaw/required.py": b"required\n"},
            {"voiceclaw/__init__.py": b"tracked\n"},
            "missing voiceclaw/required.py",
        ),
    ],
)
def test_wheel_package_inventory_must_equal_tracked_source(
    tmp_path: Path,
    package_files: dict[str, bytes],
    wheel_files: dict[str, bytes],
    message: str,
) -> None:
    """A repository-qualified wheel cannot add or omit package members."""
    dist_info = "nemotron_voiceclaw-0.1.0.dist-info/"
    members = {**wheel_files, f"{dist_info}licenses/LICENSE": b"tracked license\n"}
    wheel = _zip(tmp_path, "inventory.whl", members)

    with zipfile.ZipFile(wheel) as archive:
        files = _WHEEL_VERIFIER._archive_files(archive)
        with pytest.raises(ValueError, match=message):
            _WHEEL_VERIFIER._tracked_contents(
                archive,
                files,
                dist_info,
                _source_contract(package_files),
            )


def test_wheel_package_bytes_must_equal_tracked_source(tmp_path: Path) -> None:
    """Matching paths cannot conceal changed package or license bytes."""
    dist_info = "nemotron_voiceclaw-0.1.0.dist-info/"
    wheel = _zip(
        tmp_path,
        "changed.whl",
        {
            "voiceclaw/__init__.py": b"changed\n",
            f"{dist_info}licenses/LICENSE": b"tracked license\n",
        },
    )

    with zipfile.ZipFile(wheel) as archive:
        files = _WHEEL_VERIFIER._archive_files(archive)
        with pytest.raises(ValueError, match="does not match tracked source bytes: voiceclaw/__init__.py"):
            _WHEEL_VERIFIER._tracked_contents(
                archive,
                files,
                dist_info,
                _source_contract({"voiceclaw/__init__.py": b"tracked\n"}),
            )


def test_wheel_license_bytes_must_equal_tracked_source(tmp_path: Path) -> None:
    """The generated license member must be the exact committed license blob."""
    dist_info = "nemotron_voiceclaw-0.1.0.dist-info/"
    wheel = _zip(
        tmp_path,
        "changed-license.whl",
        {
            "voiceclaw/__init__.py": b"tracked\n",
            f"{dist_info}licenses/LICENSE": b"changed license\n",
        },
    )

    with zipfile.ZipFile(wheel) as archive:
        files = _WHEEL_VERIFIER._archive_files(archive)
        with pytest.raises(ValueError, match="license does not match tracked source bytes"):
            _WHEEL_VERIFIER._tracked_contents(
                archive,
                files,
                dist_info,
                _source_contract({"voiceclaw/__init__.py": b"tracked\n"}),
            )


def test_wheel_manifest_digest_is_order_independent_and_content_sensitive(tmp_path: Path) -> None:
    """Evidence identifies the complete member manifest, not ZIP insertion order."""
    first = _zip(tmp_path, "first.whl", {"voiceclaw/a.py": b"a", "voiceclaw/b.py": b"b"})
    second = _zip(tmp_path, "second.whl", {"voiceclaw/b.py": b"b", "voiceclaw/a.py": b"a"})
    changed = _zip(tmp_path, "changed.whl", {"voiceclaw/a.py": b"a", "voiceclaw/b.py": b"changed"})

    def digest(path: Path) -> str:
        with zipfile.ZipFile(path) as archive:
            return _WHEEL_VERIFIER._manifest_sha256(archive, _WHEEL_VERIFIER._archive_files(archive))

    assert digest(first) == digest(second)
    assert digest(first) != digest(changed)


def test_wheel_package_manifest_matches_the_image_verifier_contract(tmp_path: Path) -> None:
    """Wheel and image verification share one canonical package-payload identity."""
    wheel = _zip(
        tmp_path,
        "package.whl",
        {
            "voiceclaw/a.py": b"a",
            "voiceclaw/resources/config.yaml": b"value: true\n",
            "nemotron_voiceclaw-0.1.0.dist-info/METADATA": b"ignored by package identity",
        },
    )
    wheel_bytes = wheel.read_bytes()
    with zipfile.ZipFile(wheel) as archive:
        files = _WHEEL_VERIFIER._archive_files(archive)
        expected = _WHEEL_VERIFIER._package_manifest_sha256(archive, files)

    actual = _IMAGE_VERIFIER._package_manifest_sha256(_IMAGE_VERIFIER._wheel_package_members(wheel_bytes))

    assert actual == expected


def test_image_wheel_binding_checks_bytes_evidence_and_package_identity(tmp_path: Path) -> None:
    """Image verification consumes the exact wheel and its prior evidence receipt."""
    wheel = _zip(tmp_path, "nemotron_voiceclaw-0.1.0-py3-none-any.whl", {"voiceclaw/a.py": b"a"})
    wheel_bytes = wheel.read_bytes()
    package_manifest = _IMAGE_VERIFIER._package_manifest_sha256(_IMAGE_VERIFIER._wheel_package_members(wheel_bytes))
    evidence = {
        "schema": "voiceclaw.wheel_evidence.v1",
        "filename": wheel.name,
        "version": "0.1.0",
        "sha256": hashlib.sha256(wheel_bytes).hexdigest(),
        "size": len(wheel_bytes),
        "package_manifest_sha256": package_manifest,
        "source_revision": None,
        "source_tree": None,
    }
    evidence_path = tmp_path / "wheel-evidence.json"
    evidence_path.write_text(json.dumps(evidence), encoding="utf-8")

    binding = _IMAGE_VERIFIER._verify_wheel_binding(
        wheel,
        evidence_path,
        version="0.1.0",
        revision="development",
        source_tree=None,
    )

    assert binding is not None
    assert binding["package_manifest_sha256"] == package_manifest


def test_wheel_api_rejects_source_identity_without_exact_source_contract(tmp_path: Path) -> None:
    """Direct callers cannot label arbitrary bytes as repository-qualified evidence."""
    wheel = tmp_path / "nemotron_voiceclaw-0.1.0-py3-none-any.whl"
    wheel.write_bytes(b"not reached")

    with pytest.raises(ValueError, match="exact source contract"):
        _WHEEL_VERIFIER.verify_wheel(
            wheel,
            source_revision="a" * 40,
            source_tree="b" * 40,
        )


def test_source_contract_uses_exact_tracked_package_and_license_bytes(tmp_path: Path) -> None:
    """Repository verification derives its package contract from committed blobs."""
    package = tmp_path / "src/examples/voiceclaw/src/voiceclaw"
    package.mkdir(parents=True)
    (package / "__init__.py").write_bytes(b"tracked package\n")
    license_path = tmp_path / "src/examples/voiceclaw/LICENSE"
    license_path.write_bytes(b"tracked license\n")
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "config", "user.email", "voiceclaw@example.invalid"], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "config", "user.name", "VoiceClaw Test"], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "add", "."], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "commit", "-qm", "fixture"], check=True)
    revision = subprocess.run(
        ["git", "-C", str(tmp_path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    contract = _WHEEL_VERIFIER._source_contract(tmp_path, revision)

    assert contract is not None
    assert contract.package_files == {"voiceclaw/__init__.py": b"tracked package\n"}
    assert contract.license_bytes == b"tracked license\n"
    assert (
        contract.tree
        == subprocess.run(
            ["git", "-C", str(tmp_path), "rev-parse", "HEAD^{tree}"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )


def test_image_rejects_baked_secret_environment_name() -> None:
    """An OCI image cannot bake credential-like environment entries."""
    source = "https://github.com/NVIDIA-AI-Blueprints/nemotron-voice-agent"
    inspect = _image_inspect(source=source, environment=["PATH=/usr/bin", "NVIDIA_API_KEY=not-a-real-secret"])

    with pytest.raises(ValueError, match="secret-like environment name: NVIDIA_API_KEY"):
        _IMAGE_VERIFIER._verify_contract(
            inspect,
            version="0.1.0",
            revision="development",
            source=source,
        )


def test_image_rejects_credential_bearing_source_url() -> None:
    """OCI source provenance cannot encode URL user information."""
    source = "https://build-user:not-a-real-secret@github.com/NVIDIA-AI-Blueprints/nemotron-voice-agent"
    inspect = _image_inspect(source=source)

    with pytest.raises(ValueError, match="unpadded https URL"):
        _IMAGE_VERIFIER._verify_contract(
            inspect,
            version="0.1.0",
            revision="development",
            source=source,
        )


def test_image_rejects_an_unexpected_artifact_architecture() -> None:
    """A retained DGX Spark image cannot silently be built for another platform."""
    source = "https://github.com/NVIDIA-AI-Blueprints/nemotron-voice-agent"

    with pytest.raises(ValueError, match="architecture"):
        _IMAGE_VERIFIER._verify_contract(
            _image_inspect(source=source),
            version="0.1.0",
            revision="development",
            source=source,
            architecture="arm64",
        )


def test_image_rejects_an_unexpected_stop_signal() -> None:
    """The retained image must preserve graceful runtime shutdown semantics."""
    source = "https://github.com/NVIDIA-AI-Blueprints/nemotron-voice-agent"
    inspect = _image_inspect(source=source)
    inspect["Config"]["StopSignal"] = "SIGKILL"

    with pytest.raises(ValueError, match="stop signal"):
        _IMAGE_VERIFIER._verify_contract(
            inspect,
            version="0.1.0",
            revision="development",
            source=source,
        )


def test_image_runtime_requires_the_split_identity_supervisor() -> None:
    """The generic runtime starts as root and drops each child identity."""
    source = "https://github.com/NVIDIA-AI-Blueprints/nemotron-voice-agent"

    for root_user in ("", "0", "0:0", "root"):
        runtime = _IMAGE_VERIFIER._verify_contract(
            _image_inspect(source=source, user=root_user),
            version="0.1.0",
            revision="development",
            source=source,
        )
        assert runtime["user"] == root_user

    with pytest.raises(ValueError, match="supervisor as root"):
        _IMAGE_VERIFIER._verify_contract(
            _image_inspect(source=source, user="65532:65532"),
            version="0.1.0",
            revision="development",
            source=source,
        )


@pytest.mark.parametrize("fault", [None, "uid", "port", "health", "sdk", "revision"])
def test_installer_image_has_an_explicit_nonroot_contract(fault: str | None) -> None:
    """The installer target cannot pass as the root-operated default image."""
    source = "https://github.com/NVIDIA-AI-Blueprints/nemotron-voice-agent"
    revision = "a" * 40
    inspected = _image_inspect(source=source, user="65532:65532")
    config = inspected["Config"]
    config["ExposedPorts"] = {"18790/tcp": {}}
    config["Env"] = ["VOICECLAW_RUNTIME_PROFILE=nemoclaw-container-v1"]
    config["Labels"].update(
        {
            "org.opencontainers.image.revision": revision,
            "com.nvidia.voiceclaw.install-contract": "voiceclaw.nemoclaw.container.v1",
            "com.nvidia.voiceclaw.openshell-revision": _IMAGE_VERIFIER._OPEN_SHELL_REVISION,
        }
    )
    config["Healthcheck"].update(
        Interval=10_000_000_000, Timeout=20_000_000_000, StartPeriod=180_000_000_000, Retries=3
    )
    if fault == "uid":
        config["User"] = "root"
    elif fault == "port":
        config["ExposedPorts"] = {"7860/tcp": {}}
    elif fault == "health":
        config["Healthcheck"]["Timeout"] = 240_000_000_000
    elif fault == "sdk":
        config["Labels"]["com.nvidia.voiceclaw.openshell-revision"] = "b" * 40
    elif fault == "revision":
        revision = "development"

    def verify():
        return _IMAGE_VERIFIER._verify_contract(
            inspected, version="0.1.0", revision=revision, source=source, runtime_profile="nemoclaw-container-v1"
        )

    if fault is not None:
        with pytest.raises(ValueError):
            verify()
    else:
        assert verify()["user"] == "65532:65532"


def test_installer_manifest_requires_selected_engine_resolution() -> None:
    """A tag or configuration ID cannot replace the manifest reference."""
    reference = "voiceclaw-test@sha256:" + "a" * 64
    assert _IMAGE_VERIFIER._manifest_reference(reference, {"RepoDigests": [reference]}) == reference
    with pytest.raises(ValueError):
        _IMAGE_VERIFIER._manifest_reference(reference, {"RepoDigests": []})
    with pytest.raises(ValueError):
        _IMAGE_VERIFIER._manifest_reference("voiceclaw-test:latest", {"RepoDigests": [reference]})


@pytest.mark.parametrize(
    ("field", "extra", "message"),
    [
        ("ExposedPorts", {"8080/tcp": {}}, "expose only 7860/tcp"),
        ("Volumes", {"/tmp/extra": {}}, "only the VoiceClaw state volume"),
    ],
)
def test_runtime_image_rejects_additional_ports_and_volumes(
    field: str,
    extra: dict[str, object],
    message: str,
) -> None:
    """The runtime artifact exposes only its fixed service surfaces."""
    source = "https://github.com/NVIDIA-AI-Blueprints/nemotron-voice-agent"
    inspect = _image_inspect(source=source)
    inspect["Config"][field].update(extra)

    with pytest.raises(ValueError, match=message):
        _IMAGE_VERIFIER._verify_contract(
            inspect,
            version="0.1.0",
            revision="development",
            source=source,
        )


def test_image_repository_evidence_binds_the_harness_origin() -> None:
    """Repository-backed evidence can only use the harness from that checkout."""
    repository_root = Path(__file__).parents[4]

    _IMAGE_VERIFIER._assert_harness_origin(repository_root)

    with pytest.raises(ValueError, match="authenticated repository"):
        _IMAGE_VERIFIER._assert_harness_origin(repository_root.parent)


def test_image_rejects_baked_source_harness(monkeypatch: pytest.MonkeyPatch) -> None:
    """The production image cannot contain the source-only artifact harness."""

    def docker(*arguments: str, check: bool = True) -> subprocess.CompletedProcess[str]:
        del check
        assert arguments[-1] == _IMAGE_VERIFIER._CONTAINER_HARNESS_DIRECTORY
        return subprocess.CompletedProcess(arguments, 1, "", "")

    monkeypatch.setattr(_IMAGE_VERIFIER, "_docker", docker)
    with pytest.raises(RuntimeError, match="source-only artifact harness"):
        _IMAGE_VERIFIER._assert_test_harness_absent("voiceclaw-test")
