# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""Contract tests for the verified VoiceClaw image archive exporter."""

from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import tarfile
from pathlib import Path

import pytest


def _load_exporter():
    path = Path(__file__).parents[1] / "scripts" / "export-image-archive.py"
    spec = importlib.util.spec_from_file_location("voiceclaw_export_image_archive", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_EXPORTER = _load_exporter()


def _add_file(archive: tarfile.TarFile, name: str, data: bytes) -> None:
    member = tarfile.TarInfo(name)
    member.size = len(data)
    member.mode = 0o644
    archive.addfile(member, io.BytesIO(data))


def _image_config(layer: bytes = b"layer-bytes") -> bytes:
    return json.dumps(
        {
            "architecture": "arm64",
            "rootfs": {"type": "layers", "diff_ids": ["sha256:" + hashlib.sha256(layer).hexdigest()]},
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()


def _docker_archive(
    path: Path,
    config: bytes,
    *,
    layer: bytes = b"layer-bytes",
    special_member: bool = False,
) -> str:
    image_id = "sha256:" + hashlib.sha256(config).hexdigest()
    config_name = image_id.removeprefix("sha256:") + ".json"
    layer_name = "layer/layer.tar"
    manifest = json.dumps(
        [{"Config": config_name, "RepoTags": None, "Layers": [layer_name]}],
        separators=(",", ":"),
    ).encode()
    with tarfile.open(path, "w") as archive:
        _add_file(archive, config_name, config)
        _add_file(archive, layer_name, layer)
        _add_file(archive, "manifest.json", manifest)
        if special_member:
            member = tarfile.TarInfo("unsafe-link")
            member.type = tarfile.SYMTYPE
            member.linkname = config_name
            archive.addfile(member)
    return image_id


def _image_evidence(path: Path, image_id: str) -> bytes:
    data = (
        json.dumps(
            {
                "schema": "voiceclaw.image_evidence.v1",
                "source_revision": "a" * 40,
                "image": {"id": image_id},
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode()
    path.write_bytes(data)
    return data


def test_archive_validation_binds_config_bytes_to_verified_image_id(tmp_path: Path) -> None:
    """The exported tar is accepted only for the image ID derived from its config bytes."""
    archive = tmp_path / "image.tar"
    image_id = _docker_archive(archive, _image_config())

    size, digest = _EXPORTER._validate_archive(archive, image_id)

    assert size == archive.stat().st_size
    assert digest == hashlib.sha256(archive.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="does not match the verified image ID"):
        _EXPORTER._validate_archive(archive, "sha256:" + "0" * 64)


def test_archive_validation_rejects_non_regular_members(tmp_path: Path) -> None:
    """Links and other special tar members cannot enter the release artifact."""
    archive = tmp_path / "linked.tar"
    image_id = _docker_archive(archive, _image_config(), special_member=True)

    with pytest.raises(ValueError, match="non-regular member"):
        _EXPORTER._validate_archive(archive, image_id)


def test_archive_validation_rejects_layer_bytes_that_do_not_match_rootfs_identity(tmp_path: Path) -> None:
    """Every layer is authenticated by the image config, not merely present in the tar."""
    archive = tmp_path / "changed-layer.tar"
    image_id = _docker_archive(archive, _image_config(), layer=b"substituted-layer")

    with pytest.raises(ValueError, match="does not match its rootfs diff ID"):
        _EXPORTER._validate_archive(archive, image_id)


def test_export_refuses_image_that_differs_from_prior_evidence(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A mutable local tag cannot be exported after it moves away from the verified image."""
    evidence = tmp_path / "image-evidence.json"
    _image_evidence(evidence, "sha256:" + "1" * 64)
    monkeypatch.setattr(_EXPORTER, "_inspect_image_id", lambda _image: "sha256:" + "2" * 64)

    with pytest.raises(ValueError, match="does not match the verified image evidence"):
        _EXPORTER.export_archive("voiceclaw:mutable", evidence, tmp_path / "output.tar")

    assert not (tmp_path / "output.tar").exists()


def test_export_publishes_archive_and_digest_bound_evidence(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The final evidence binds source evidence, image identity, filename, and archive bytes."""
    config = _image_config()
    image_id = "sha256:" + hashlib.sha256(config).hexdigest()
    image_evidence = tmp_path / "image-evidence.json"
    image_evidence_bytes = _image_evidence(image_evidence, image_id)
    output = tmp_path / "voiceclaw.docker.tar"
    monkeypatch.setattr(_EXPORTER, "_inspect_image_id", lambda _image: image_id)

    def save_image(saved_id: str, destination: Path) -> None:
        assert saved_id == image_id
        assert _docker_archive(destination, config) == image_id

    monkeypatch.setattr(_EXPORTER, "_save_image", save_image)

    result = _EXPORTER.export_archive("voiceclaw:verified", image_evidence, output)

    assert output.is_file()
    assert result == {
        "schema": "voiceclaw.image_archive_evidence.v1",
        "format": "docker-archive",
        "filename": output.name,
        "size": output.stat().st_size,
        "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "image_id": image_id,
        "source_revision": "a" * 40,
        "image_evidence_sha256": hashlib.sha256(image_evidence_bytes).hexdigest(),
    }

    archive_evidence = tmp_path / "archive-evidence.json"
    archive_evidence.write_text(json.dumps(result, sort_keys=True, separators=(",", ":")), encoding="utf-8")
    assert _EXPORTER.verify_archive(output, image_evidence, archive_evidence) == result


def test_ci_retains_the_archive_and_both_evidence_receipts() -> None:
    """The release lane must retain the actual image bytes, not evidence alone."""
    repository = Path(__file__).resolve().parents[4]
    workflow = (repository / ".github/workflows/voiceclaw.yml").read_text(encoding="utf-8")

    assert "export-image-archive.py" in workflow
    assert "runs-on: ubuntu-24.04-arm" in workflow
    assert "actions/download-artifact@d3f86a106a0bac45b974a628896c90dbdf5c8093" in workflow
    assert "--expect-architecture arm64" in workflow
    assert '--wheel "${wheels[0]}"' in workflow
    assert '--wheel-evidence "${wheel_evidence[0]}"' in workflow
    assert "voiceclaw-${GITHUB_SHA}-linux-arm64.docker.tar" in workflow
    assert "${{ runner.temp }}/voiceclaw-image/voiceclaw-${{ github.sha }}-linux-arm64.docker.tar" in workflow
    assert "Verify the retained archive as a consumer" in workflow
    assert "--verify-only" in workflow
    assert "${{ runner.temp }}/voiceclaw-image/image-evidence.json" in workflow
    assert "${{ runner.temp }}/voiceclaw-image/archive-evidence.json" in workflow
    assert "retention-days: 30" in workflow
