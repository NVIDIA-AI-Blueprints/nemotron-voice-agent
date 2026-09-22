#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""Export one verified VoiceClaw image as an associated Docker archive."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO

_IMAGE_ID = re.compile(r"^sha256:[0-9a-f]{64}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_MAX_EVIDENCE_BYTES = 2 * 1024 * 1024
_MAX_ARCHIVE_MEMBERS = 65_536
_MAX_METADATA_BYTES = 16 * 1024 * 1024


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", nargs="?", help="local image reference already covered by image evidence")
    parser.add_argument("--image-evidence", required=True, type=Path, help="verified image-evidence JSON")
    parser.add_argument("--archive", required=True, type=Path, help="Docker archive output or verification path")
    parser.add_argument("--archive-evidence", type=Path, help="archive evidence to validate with --verify-only")
    parser.add_argument("--verify-only", action="store_true", help="verify an existing archive without loading it")
    return parser


def _json_object(data: bytes, label: str) -> dict[str, Any]:
    def reject_duplicate(name_value_pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for name, item in name_value_pairs:
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


def _regular_file(path: Path, *, maximum: int | None = None) -> tuple[os.stat_result, BinaryIO]:
    if not hasattr(os, "O_NOFOLLOW"):
        raise RuntimeError("safe artifact-file opens are unsupported on this platform")
    flags = os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NONBLOCK", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ValueError(f"artifact input cannot be opened safely: {path.name}") from error
    metadata = os.fstat(descriptor)
    if (
        not stat.S_ISREG(metadata.st_mode)
        or metadata.st_size < 1
        or (maximum is not None and metadata.st_size > maximum)
    ):
        os.close(descriptor)
        raise ValueError(f"artifact input is not a bounded regular file: {path.name}")
    return metadata, os.fdopen(descriptor, "rb")


def _read_evidence(path: Path) -> tuple[dict[str, Any], bytes]:
    metadata, stream = _regular_file(path, maximum=_MAX_EVIDENCE_BYTES)
    with stream:
        data = stream.read(_MAX_EVIDENCE_BYTES + 1)
    if len(data) != metadata.st_size:
        raise ValueError("image evidence changed while it was being read")
    evidence = _json_object(data, "image evidence")
    image = evidence.get("image")
    image_id = image.get("id") if isinstance(image, dict) else None
    if (
        evidence.get("schema") != "voiceclaw.image_evidence.v1"
        or not isinstance(image_id, str)
        or not _IMAGE_ID.fullmatch(image_id)
    ):
        raise ValueError("image evidence does not contain one verified VoiceClaw image ID")
    revision = evidence.get("source_revision")
    if not isinstance(revision, str) or not revision:
        raise ValueError("image evidence does not contain a source revision")
    return evidence, data


def _inspect_image_id(image: str) -> str:
    try:
        completed = subprocess.run(
            ["docker", "image", "inspect", "--format", "{{.Id}}", image],
            check=True,
            capture_output=True,
            text=True,
            timeout=60,
        )
    except FileNotFoundError as error:
        raise RuntimeError("docker CLI is not installed") from error
    except subprocess.TimeoutExpired as error:
        raise RuntimeError("docker image inspection exceeded its bounded timeout") from error
    except subprocess.CalledProcessError as error:
        raise RuntimeError("docker image inspection failed") from error
    image_id = completed.stdout.strip()
    if not _IMAGE_ID.fullmatch(image_id):
        raise ValueError("docker returned a malformed image ID")
    return image_id


def _save_image(image_id: str, destination: Path) -> None:
    try:
        subprocess.run(
            ["docker", "image", "save", "--output", str(destination), image_id],
            check=True,
            capture_output=True,
            timeout=600,
        )
    except FileNotFoundError as error:
        raise RuntimeError("docker CLI is not installed") from error
    except subprocess.TimeoutExpired as error:
        raise RuntimeError("docker image export exceeded its bounded timeout") from error
    except subprocess.CalledProcessError as error:
        raise RuntimeError("docker image export failed") from error


def _safe_member_name(name: str) -> str:
    path = PurePosixPath(name)
    if (
        not name
        or name.startswith("/")
        or "\\" in name
        or name != path.as_posix()
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ValueError("Docker archive contains an unsafe member path")
    return name


def _archive_members(archive: tarfile.TarFile) -> dict[str, tarfile.TarInfo]:
    members: dict[str, tarfile.TarInfo] = {}
    folded: set[str] = set()
    for count, member in enumerate(archive, start=1):
        if count > _MAX_ARCHIVE_MEMBERS:
            raise ValueError("Docker archive has too many members")
        name = _safe_member_name(member.name.rstrip("/"))
        if name in members or name.casefold() in folded:
            raise ValueError("Docker archive contains a duplicate member")
        folded.add(name.casefold())
        if not member.isfile() and not member.isdir():
            raise ValueError("Docker archive contains a non-regular member")
        members[name] = member
    return members


def _member_bytes(archive: tarfile.TarFile, member: tarfile.TarInfo, label: str) -> bytes:
    if not member.isfile() or member.size < 1 or member.size > _MAX_METADATA_BYTES:
        raise ValueError(f"Docker archive {label} is not a bounded regular file")
    stream = archive.extractfile(member)
    if stream is None:
        raise ValueError(f"Docker archive {label} cannot be read")
    data = stream.read(_MAX_METADATA_BYTES + 1)
    if len(data) != member.size:
        raise ValueError(f"Docker archive {label} is truncated")
    return data


def _member_sha256(archive: tarfile.TarFile, member: tarfile.TarInfo, label: str) -> str:
    if not member.isfile() or member.size < 1:
        raise ValueError(f"Docker archive {label} is not a non-empty regular file")
    stream = archive.extractfile(member)
    if stream is None:
        raise ValueError(f"Docker archive {label} cannot be read")
    digest = hashlib.sha256()
    size = 0
    for block in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(block)
        size += len(block)
    if size != member.size:
        raise ValueError(f"Docker archive {label} is truncated")
    return digest.hexdigest()


def _validate_archive(path: Path, expected_image_id: str) -> tuple[int, str]:
    if not _IMAGE_ID.fullmatch(expected_image_id):
        raise ValueError("expected image ID is malformed")
    metadata, stream = _regular_file(path)
    digest = hashlib.sha256()
    with stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
        after_hash = os.fstat(stream.fileno())
        if (metadata.st_dev, metadata.st_ino, metadata.st_size) != (
            after_hash.st_dev,
            after_hash.st_ino,
            after_hash.st_size,
        ):
            raise ValueError("Docker archive changed while it was being hashed")
        stream.seek(0)
        try:
            with tarfile.open(fileobj=stream, mode="r:") as archive:
                members = _archive_members(archive)
                manifest_member = members.get("manifest.json")
                if manifest_member is None:
                    raise ValueError("Docker archive manifest.json is absent")
                manifest = json.loads(_member_bytes(archive, manifest_member, "manifest.json"))
                if not isinstance(manifest, list) or len(manifest) != 1 or not isinstance(manifest[0], dict):
                    raise ValueError("Docker archive must describe exactly one image")
                record = manifest[0]
                config_name = record.get("Config")
                layers = record.get("Layers")
                if not isinstance(config_name, str) or _safe_member_name(config_name) != config_name:
                    raise ValueError("Docker archive manifest has an invalid config path")
                if not isinstance(layers, list) or not layers or any(not isinstance(item, str) for item in layers):
                    raise ValueError("Docker archive manifest has an invalid layer list")
                config_member = members.get(config_name)
                if config_member is None:
                    raise ValueError("Docker archive image config is absent")
                config = _member_bytes(archive, config_member, "image config")
                if "sha256:" + hashlib.sha256(config).hexdigest() != expected_image_id:
                    raise ValueError("Docker archive image config does not match the verified image ID")
                config_object = _json_object(config, "Docker archive image config")
                rootfs = config_object.get("rootfs")
                diff_ids = rootfs.get("diff_ids") if isinstance(rootfs, dict) else None
                if (
                    not isinstance(rootfs, dict)
                    or rootfs.get("type") != "layers"
                    or not isinstance(diff_ids, list)
                    or len(diff_ids) != len(layers)
                    or any(not isinstance(item, str) or not _IMAGE_ID.fullmatch(item) for item in diff_ids)
                ):
                    raise ValueError("Docker archive image config has an invalid rootfs layer identity")
                for layer_name, diff_id in zip(layers, diff_ids, strict=True):
                    if (
                        _safe_member_name(layer_name) != layer_name
                        or layer_name not in members
                        or not members[layer_name].isfile()
                    ):
                        raise ValueError("Docker archive contains a missing or invalid image layer")
                    layer_sha256 = _member_sha256(archive, members[layer_name], "image layer")
                    if f"sha256:{layer_sha256}" != diff_id:
                        raise ValueError("Docker archive image layer does not match its rootfs diff ID")
                    blob_match = re.fullmatch(r"blobs/sha256/([0-9a-f]{64})", layer_name)
                    if blob_match is not None and blob_match.group(1) != layer_sha256:
                        raise ValueError("Docker archive image layer does not match its blob path")
        except tarfile.TarError as error:
            raise ValueError("Docker archive is not a valid uncompressed tar archive") from error
    return metadata.st_size, digest.hexdigest()


def export_archive(image: str, image_evidence: Path, archive: Path) -> dict[str, Any]:
    """Export and self-verify one immutable image archive."""
    evidence, evidence_bytes = _read_evidence(image_evidence)
    image_id = evidence["image"]["id"]
    inspected_id = _inspect_image_id(image)
    if inspected_id != image_id:
        raise ValueError("local image does not match the verified image evidence")

    archive = archive.resolve()
    archive.parent.mkdir(parents=True, exist_ok=True)
    if archive.exists() or archive.is_symlink():
        raise ValueError("archive output already exists")
    descriptor, pending_name = tempfile.mkstemp(prefix=f".{archive.name}.", dir=archive.parent)
    os.close(descriptor)
    pending = Path(pending_name)
    try:
        _save_image(image_id, pending)
        size, archive_sha256 = _validate_archive(pending, image_id)
        os.replace(pending, archive)
    finally:
        pending.unlink(missing_ok=True)

    return {
        "schema": "voiceclaw.image_archive_evidence.v1",
        "format": "docker-archive",
        "filename": archive.name,
        "size": size,
        "sha256": archive_sha256,
        "image_id": image_id,
        "source_revision": evidence["source_revision"],
        "image_evidence_sha256": hashlib.sha256(evidence_bytes).hexdigest(),
    }


def verify_archive(archive: Path, image_evidence: Path, archive_evidence: Path) -> dict[str, Any]:
    """Verify a downloaded archive against both producer evidence receipts."""
    image_receipt, image_evidence_bytes = _read_evidence(image_evidence)
    archive_evidence_bytes = _bounded_evidence_bytes(archive_evidence, "archive evidence")
    archive_receipt = _json_object(archive_evidence_bytes, "archive evidence")
    expected_keys = {
        "schema",
        "format",
        "filename",
        "size",
        "sha256",
        "image_id",
        "source_revision",
        "image_evidence_sha256",
    }
    if set(archive_receipt) != expected_keys:
        raise ValueError("archive evidence has an unexpected field inventory")
    image_id = image_receipt["image"]["id"]
    if (
        archive_receipt.get("schema") != "voiceclaw.image_archive_evidence.v1"
        or archive_receipt.get("format") != "docker-archive"
        or archive_receipt.get("filename") != archive.name
        or archive_receipt.get("image_id") != image_id
        or archive_receipt.get("source_revision") != image_receipt["source_revision"]
        or archive_receipt.get("image_evidence_sha256") != hashlib.sha256(image_evidence_bytes).hexdigest()
    ):
        raise ValueError("archive evidence does not match the image evidence or archive name")
    size, archive_sha256 = _validate_archive(archive, image_id)
    if archive_receipt.get("size") != size or archive_receipt.get("sha256") != archive_sha256:
        raise ValueError("archive bytes do not match the archive evidence")
    return archive_receipt


def _bounded_evidence_bytes(path: Path, label: str) -> bytes:
    metadata, stream = _regular_file(path, maximum=_MAX_EVIDENCE_BYTES)
    with stream:
        data = stream.read(_MAX_EVIDENCE_BYTES + 1)
    if len(data) != metadata.st_size:
        raise ValueError(f"{label} changed while it was being read")
    return data


def main() -> int:
    """Export the requested image and print canonical archive evidence."""
    arguments = _parser().parse_args()
    try:
        if arguments.verify_only:
            if arguments.image is not None or arguments.archive_evidence is None:
                raise ValueError("--verify-only requires --archive-evidence and no image argument")
            evidence = verify_archive(arguments.archive, arguments.image_evidence, arguments.archive_evidence)
        else:
            if arguments.image is None or arguments.archive_evidence is not None:
                raise ValueError("export requires one image argument and no --archive-evidence")
            evidence = export_archive(arguments.image, arguments.image_evidence, arguments.archive)
    except (OSError, RuntimeError, ValueError) as error:
        print(f"export-image-archive: {error}", file=sys.stderr)
        return 1
    print(json.dumps(evidence, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
