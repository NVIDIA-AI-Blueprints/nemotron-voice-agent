#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""Verify the exact built VoiceClaw wheel and emit reproducible evidence."""

from __future__ import annotations

import argparse
import base64
import configparser
import csv
import hashlib
import io
import json
import re
import stat
import subprocess
import sys
import zipfile
from email.parser import BytesParser
from pathlib import Path, PurePosixPath
from typing import NamedTuple

_DISTRIBUTION = "nemotron-voiceclaw"
_DIST_INFO = re.compile(r"^nemotron_voiceclaw-(?P<version>[^/]+)\.dist-info/$")
_REVISION = re.compile(r"^[0-9a-f]{40}$")
_MAX_WHEEL_BYTES = 16 * 1024 * 1024
_MAX_MEMBER_BYTES = 8 * 1024 * 1024
_MAX_UNCOMPRESSED_BYTES = 32 * 1024 * 1024
_MAX_MEMBERS = 2_048
_EXPECTED_ENTRY_POINTS = {
    "voiceclaw": "voiceclaw.server:main",
    "voiceclaw-client-secret": "voiceclaw.client_secret:main",
    "voiceclaw-runtime": "voiceclaw.runtime_cli:main",
}
_PACKAGE_SOURCE_PREFIX = "src/examples/voiceclaw/src/"
_LICENSE_SOURCE_PATH = "src/examples/voiceclaw/LICENSE"
_EXPECTED_DIST_INFO_MEMBERS = frozenset(
    {
        "METADATA",
        "WHEEL",
        "entry_points.txt",
        "top_level.txt",
        "licenses/LICENSE",
        "RECORD",
    }
)
_REQUIRED_PACKAGE_FILES = {
    "voiceclaw/py.typed",
    "voiceclaw/resources/model_contracts.v1.yaml",
    "voiceclaw/resources/interaction_profiles.v1.yaml",
    "voiceclaw/resources/interaction_profiles.v2.yaml",
    "voiceclaw/resources/voiceclaw.example.yaml",
    "voiceclaw/ui/index.html",
    "voiceclaw/ui/styles.css",
    "voiceclaw/ui/app.js",
    "voiceclaw/ui/marked.min.js",
    "voiceclaw/ui/marked.LICENSE.txt",
}
_FORBIDDEN_PARTS = {"__pycache__", "operator", "tests"}
_FORBIDDEN_SUFFIXES = (".crt", ".db", ".docx", ".key", ".p12", ".pdf", ".pem", ".pyc", ".pyo", ".sqlite")


class WheelEvidence(NamedTuple):
    """Validated facts emitted for one immutable wheel."""

    filename: str
    sha256: str
    size: int
    distribution: str
    version: str
    member_count: int
    manifest_sha256: str
    package_manifest_sha256: str
    source_revision: str | None
    source_tree: str | None


class SourceContract(NamedTuple):
    """Exact tracked package bytes used to qualify one wheel."""

    tree: str
    package_files: dict[str, bytes]
    license_bytes: bytes


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheel", type=Path, help="exact wheel artifact to verify")
    parser.add_argument("--expect-version", help="required package version")
    parser.add_argument("--expect-revision", help="required full lowercase Git revision")
    parser.add_argument("--repository-root", type=Path, help="verify the clean Git source tree for this artifact")
    return parser


def _git_bytes(root: Path, *arguments: str) -> bytes:
    try:
        completed = subprocess.run(
            ["git", "--no-replace-objects", "-C", str(root), *arguments],
            check=True,
            capture_output=True,
            timeout=15,
        )
    except FileNotFoundError as error:
        raise RuntimeError("git CLI is not installed") from error
    except subprocess.TimeoutExpired as error:
        raise RuntimeError("git source verification exceeded its bounded timeout") from error
    except subprocess.CalledProcessError as error:
        raise RuntimeError("repository source state could not be verified") from error
    return completed.stdout


def _git(root: Path, *arguments: str) -> str:
    try:
        return _git_bytes(root, *arguments).decode("utf-8").strip()
    except UnicodeDecodeError as error:
        raise RuntimeError("git source verification returned non-UTF-8 text") from error


def _source_contract(root: Path | None, revision: str | None) -> SourceContract | None:
    if (root is None) != (revision is None):
        raise ValueError("--repository-root and --expect-revision must be supplied together")
    if root is None or revision is None:
        return None
    if not _REVISION.fullmatch(revision):
        raise ValueError("expected revision must be one full lowercase Git revision")
    root = root.resolve()
    if Path(_git(root, "rev-parse", "--show-toplevel")).resolve() != root:
        raise ValueError("repository root does not name the Git worktree root")
    if _git(root, "rev-parse", "HEAD") != revision:
        raise ValueError("repository HEAD does not match the expected wheel revision")
    if _git(root, "status", "--porcelain=v1", "--untracked-files=all"):
        raise ValueError("repository contains uncommitted wheel inputs")
    tree = _git(root, "rev-parse", "HEAD^{tree}")
    if not _REVISION.fullmatch(tree):
        raise ValueError("repository tree identity is malformed")

    listing = _git_bytes(
        root,
        "ls-tree",
        "-r",
        "-z",
        "--full-tree",
        revision,
        "--",
        _PACKAGE_SOURCE_PREFIX,
    )
    package_files: dict[str, bytes] = {}
    for raw_entry in listing.split(b"\0"):
        if not raw_entry:
            continue
        try:
            metadata, raw_path = raw_entry.split(b"\t", 1)
            mode, object_type, _object_id = metadata.split(b" ", 2)
            source_path = raw_path.decode("utf-8")
        except (UnicodeDecodeError, ValueError) as error:
            raise ValueError("tracked package inventory is malformed") from error
        if mode not in {b"100644", b"100755"} or object_type != b"blob":
            raise ValueError(f"tracked package member is not a regular file: {source_path}")
        if not source_path.startswith(_PACKAGE_SOURCE_PREFIX):
            raise ValueError("tracked package inventory escaped its source prefix")
        archive_path = source_path.removeprefix(_PACKAGE_SOURCE_PREFIX)
        parsed = PurePosixPath(archive_path)
        if (
            not archive_path.startswith("voiceclaw/")
            or parsed.as_posix() != archive_path
            or any(part in {"", ".", ".."} for part in parsed.parts)
            or archive_path in package_files
        ):
            raise ValueError(f"tracked package member has an unsafe path: {source_path}")
        package_files[archive_path] = _git_bytes(root, "show", f"{revision}:{source_path}")
    if not package_files:
        raise ValueError("tracked VoiceClaw package inventory is empty")

    license_entry = _git(root, "ls-tree", revision, "--", _LICENSE_SOURCE_PATH)
    if not license_entry.startswith("100644 blob ") or not license_entry.endswith(f"\t{_LICENSE_SOURCE_PATH}"):
        raise ValueError("tracked VoiceClaw license is missing or is not a regular file")
    license_bytes = _git_bytes(root, "show", f"{revision}:{_LICENSE_SOURCE_PATH}")
    if not license_bytes:
        raise ValueError("tracked VoiceClaw license is empty")
    return SourceContract(tree=tree, package_files=package_files, license_bytes=license_bytes)


def _archive_files(archive: zipfile.ZipFile) -> dict[str, zipfile.ZipInfo]:
    files: dict[str, zipfile.ZipInfo] = {}
    uncompressed_bytes = 0
    for info in archive.infolist():
        if info.is_dir():
            continue
        name = info.filename
        raw_parts = name.split("/")
        if (
            not name
            or name.startswith("/")
            or "\\" in name
            or any(ord(character) < 32 or ord(character) == 127 for character in name)
            or any(part in {"", ".", ".."} for part in raw_parts)
            or name in files
            or info.flag_bits & 0x1
            or stat.S_ISLNK((info.external_attr >> 16) & 0xFFFF)
            or info.file_size > _MAX_MEMBER_BYTES
        ):
            raise ValueError(f"wheel contains an unsafe or duplicate member: {name!r}")
        files[name] = info
        uncompressed_bytes += info.file_size
        if len(files) > _MAX_MEMBERS or uncompressed_bytes > _MAX_UNCOMPRESSED_BYTES:
            raise ValueError("wheel exceeds the bounded member or uncompressed-size limit")
    if archive.testzip() is not None:
        raise ValueError("wheel contains a member with an invalid CRC")
    return files


def _dist_info(files: dict[str, zipfile.ZipInfo]) -> tuple[str, str]:
    directories = {name.split("/", 1)[0] + "/" for name in files if ".dist-info/" in name}
    matches = [
        (directory, match.group("version")) for directory in directories if (match := _DIST_INFO.fullmatch(directory))
    ]
    if len(directories) != 1 or len(matches) != 1:
        raise ValueError("wheel must contain exactly one nemotron_voiceclaw dist-info directory")
    return matches[0]


def _metadata(archive: zipfile.ZipFile, dist_info: str, version: str) -> None:
    metadata = BytesParser().parsebytes(archive.read(f"{dist_info}METADATA"))
    if metadata.get("Name") != _DISTRIBUTION:
        raise ValueError("wheel METADATA contains the wrong distribution name")
    if metadata.get("Version") != version:
        raise ValueError("wheel filename/dist-info version disagrees with METADATA")


def _wheel_metadata(archive: zipfile.ZipFile, dist_info: str) -> None:
    metadata = BytesParser().parsebytes(archive.read(f"{dist_info}WHEEL"))
    if (
        metadata.get("Wheel-Version") != "1.0"
        or metadata.get("Root-Is-Purelib") != "true"
        or metadata.get_all("Tag") != ["py3-none-any"]
    ):
        raise ValueError("wheel compatibility metadata does not match the pure-Python package contract")


def _entry_points(archive: zipfile.ZipFile, dist_info: str) -> None:
    parser = configparser.ConfigParser(interpolation=None)
    parser.optionxform = str
    parser.read_string(archive.read(f"{dist_info}entry_points.txt").decode("utf-8"))
    actual = dict(parser.items("console_scripts")) if parser.has_section("console_scripts") else {}
    if actual != _EXPECTED_ENTRY_POINTS:
        raise ValueError("wheel console entry points do not match the VoiceClaw package contract")


def _top_level(archive: zipfile.ZipFile, dist_info: str) -> None:
    if archive.read(f"{dist_info}top_level.txt") != b"voiceclaw\n":
        raise ValueError("wheel top-level package metadata does not match the VoiceClaw package contract")


def _record(archive: zipfile.ZipFile, files: dict[str, zipfile.ZipInfo], dist_info: str) -> None:
    record_name = f"{dist_info}RECORD"
    rows = list(csv.reader(io.StringIO(archive.read(record_name).decode("utf-8"))))
    recorded: dict[str, tuple[str, str]] = {}
    for row in rows:
        if len(row) != 3 or row[0] in recorded:
            raise ValueError("wheel RECORD is malformed or contains duplicate members")
        recorded[row[0]] = (row[1], row[2])
    if set(recorded) != set(files):
        raise ValueError("wheel RECORD does not cover exactly the archive members")

    for name, info in files.items():
        digest_field, size_field = recorded[name]
        if name == record_name:
            if digest_field or size_field:
                raise ValueError("wheel RECORD must leave its own digest and size empty")
            continue
        try:
            algorithm, encoded_digest = digest_field.split("=", 1)
            expected_size = int(size_field)
        except (TypeError, ValueError) as error:
            raise ValueError(f"wheel RECORD entry is malformed: {name}") from error
        if algorithm != "sha256" or expected_size != info.file_size:
            raise ValueError(f"wheel RECORD metadata is invalid: {name}")
        actual_digest = (
            base64.urlsafe_b64encode(hashlib.sha256(archive.read(name)).digest()).rstrip(b"=").decode("ascii")
        )
        if actual_digest != encoded_digest:
            raise ValueError(f"wheel RECORD digest does not match: {name}")


def _contents(files: dict[str, zipfile.ZipInfo], dist_info: str) -> None:
    missing = sorted(_REQUIRED_PACKAGE_FILES - set(files))
    if missing:
        raise ValueError("wheel is missing required package files: " + ", ".join(missing))
    dist_info_members = {name.removeprefix(dist_info) for name in files if name.startswith(dist_info)}
    if dist_info_members != _EXPECTED_DIST_INFO_MEMBERS:
        missing_metadata = sorted(_EXPECTED_DIST_INFO_MEMBERS - dist_info_members)
        extra_metadata = sorted(dist_info_members - _EXPECTED_DIST_INFO_MEMBERS)
        detail: list[str] = []
        if missing_metadata:
            detail.append("missing " + ", ".join(missing_metadata))
        if extra_metadata:
            detail.append("unexpected " + ", ".join(extra_metadata))
        raise ValueError("wheel dist-info inventory does not match the package contract: " + "; ".join(detail))
    if {name.split("/", 1)[0] for name in files} != {"voiceclaw", dist_info.removesuffix("/")}:
        raise ValueError("wheel contains an unexpected top-level package or distribution")

    forbidden: list[str] = []
    for name in files:
        path = PurePosixPath(name)
        lowered = name.lower()
        lowered_parts = {part.lower() for part in path.parts}
        if (
            _FORBIDDEN_PARTS.intersection(lowered_parts)
            or any(part == ".env" or part.startswith(".env.") for part in lowered_parts)
            or lowered.endswith(_FORBIDDEN_SUFFIXES)
            or ".egg-info/" in lowered
        ):
            forbidden.append(name)
    if forbidden:
        raise ValueError("wheel contains development or private files: " + ", ".join(sorted(forbidden)))


def _tracked_contents(
    archive: zipfile.ZipFile,
    files: dict[str, zipfile.ZipInfo],
    dist_info: str,
    source: SourceContract | None,
) -> None:
    if source is None:
        return
    actual_package_files = {name for name in files if name.startswith("voiceclaw/")}
    expected_package_files = set(source.package_files)
    if actual_package_files != expected_package_files:
        missing = sorted(expected_package_files - actual_package_files)
        extra = sorted(actual_package_files - expected_package_files)
        detail: list[str] = []
        if missing:
            detail.append("missing " + ", ".join(missing))
        if extra:
            detail.append("unexpected " + ", ".join(extra))
        raise ValueError("wheel package inventory does not match tracked source: " + "; ".join(detail))
    for name in sorted(expected_package_files):
        if archive.read(name) != source.package_files[name]:
            raise ValueError(f"wheel package member does not match tracked source bytes: {name}")
    if archive.read(f"{dist_info}licenses/LICENSE") != source.license_bytes:
        raise ValueError("wheel license does not match tracked source bytes")


def _selected_manifest_sha256(
    archive: zipfile.ZipFile,
    files: dict[str, zipfile.ZipInfo],
    names: list[str],
    schema: str,
) -> str:
    members = [
        {
            "path": name,
            "sha256": hashlib.sha256(archive.read(name)).hexdigest(),
            "size": files[name].file_size,
        }
        for name in sorted(names)
    ]
    encoded = json.dumps(
        {"schema": schema, "members": members},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _manifest_sha256(archive: zipfile.ZipFile, files: dict[str, zipfile.ZipInfo]) -> str:
    return _selected_manifest_sha256(archive, files, list(files), "voiceclaw.wheel_manifest.v1")


def _package_manifest_sha256(archive: zipfile.ZipFile, files: dict[str, zipfile.ZipInfo]) -> str:
    names = [name for name in files if name.startswith("voiceclaw/")]
    return _selected_manifest_sha256(archive, files, names, "voiceclaw.package_manifest.v1")


def verify_wheel(
    path: Path,
    *,
    expect_version: str | None = None,
    source_revision: str | None = None,
    source_tree: str | None = None,
    source_contract: SourceContract | None = None,
) -> WheelEvidence:
    """Verify one wheel without importing code from the source checkout."""
    if source_contract is None:
        if source_revision is not None or source_tree is not None:
            raise ValueError("source-labeled wheel evidence requires an exact source contract")
    elif source_revision is None or not _REVISION.fullmatch(source_revision) or source_tree != source_contract.tree:
        raise ValueError("wheel source identity does not match its exact source contract")
    if not path.is_file() or path.suffix != ".whl":
        raise ValueError("wheel must name one existing .whl file")
    wheel_bytes = path.read_bytes()
    if len(wheel_bytes) > _MAX_WHEEL_BYTES:
        raise ValueError("wheel exceeds the compressed-size limit")
    try:
        with zipfile.ZipFile(io.BytesIO(wheel_bytes)) as archive:
            files = _archive_files(archive)
            dist_info, version = _dist_info(files)
            if expect_version is not None and version != expect_version:
                raise ValueError(f"wheel version {version!r} does not match {expect_version!r}")
            if path.name != f"nemotron_voiceclaw-{version}-py3-none-any.whl":
                raise ValueError("wheel filename does not match its distribution, version, and compatibility tag")
            _contents(files, dist_info)
            _metadata(archive, dist_info, version)
            _wheel_metadata(archive, dist_info)
            _entry_points(archive, dist_info)
            _top_level(archive, dist_info)
            _record(archive, files, dist_info)
            _tracked_contents(archive, files, dist_info, source_contract)
            manifest_sha256 = _manifest_sha256(archive, files)
            package_manifest_sha256 = _package_manifest_sha256(archive, files)
    except zipfile.BadZipFile as error:
        raise ValueError("wheel is not a valid ZIP archive") from error
    return WheelEvidence(
        filename=path.name,
        sha256=hashlib.sha256(wheel_bytes).hexdigest(),
        size=len(wheel_bytes),
        distribution=_DISTRIBUTION,
        version=version,
        member_count=len(files),
        manifest_sha256=manifest_sha256,
        package_manifest_sha256=package_manifest_sha256,
        source_revision=source_revision,
        source_tree=source_tree,
    )


def main() -> int:
    """Validate the requested wheel and print canonical evidence JSON."""
    arguments = _parser().parse_args()
    try:
        source_contract = _source_contract(arguments.repository_root, arguments.expect_revision)
        evidence = verify_wheel(
            arguments.wheel,
            expect_version=arguments.expect_version,
            source_revision=arguments.expect_revision,
            source_tree=None if source_contract is None else source_contract.tree,
            source_contract=source_contract,
        )
    except (OSError, RuntimeError, UnicodeError, ValueError) as error:
        print(f"verify-wheel: {error}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {"schema": "voiceclaw.wheel_evidence.v1", **evidence._asdict()},
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
