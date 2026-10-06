#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Copy the voice Frontend/Backend prototype into ``src/examples/frontend_backend_verdict``.

The example runs the prototype's own agent, wire, engine, speech, and normalization
code. This script is the only way that code enters the repository: it copies each
file from a prototype checkout, applies the recorded rewrites (package prefix plus
the few listed configuration values), and writes ``provenance.json`` with the
SHA-256 of every original file. ``tests/unit/frontend_backend_verdict/
test_fbv_provenance.py`` reverses the rewrites and checks the hashes, so a hand
edit to a copied file fails CI.

Usage, from the repository root::

    python scripts/frontend_backend_verdict_sync.py \
        --prototype-root ../nemotron-voice-agent-smasurekar
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_DIR = REPO_ROOT / "src" / "examples" / "frontend_backend_verdict"
TEST_DIR = REPO_ROOT / "tests" / "unit" / "frontend_backend_verdict"
PROVENANCE_FILE = EXAMPLE_DIR / "provenance.json"

# Import rewrites applied to every copied text file, in order. Reversing them in the
# opposite order restores the original bytes (the test relies on this).
PACKAGE_REWRITES: tuple[tuple[str, str], ...] = (
    # Same length as the originals, so no copied line changes length or import order.
    ("prototypes.text_frontend_backend_agent", "examples.frontend_backend_verdict.text"),
    ("prototypes.voice_frontend_backend_agent", "examples.frontend_backend_verdict.voice"),
)

# Package trees copied into the example: (prototype dir, example dir, excluded relative paths).
TREES: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    (
        "src/prototypes/text_frontend_backend_agent",
        "text",
        # Interactive terminal UI (needs `rich`, not a dependency here) and the prototype README.
        ("cli", "README.md"),
    ),
    (
        "src/prototypes/voice_frontend_backend_agent",
        "voice",
        # tau3 gate scripts run with tau2's interpreter, not this package; README is replaced.
        ("cli/tau2_gates", "README.md"),
    ),
)

# Per-file value edits beyond the package prefix: (example-relative path, old, new, reason).
# Each one is a configuration value whose prototype meaning depends on the prototype's
# location in its repository. Behaviour is unchanged.
VALUE_EDITS: tuple[tuple[str, str, str, int, str], ...] = (
    (
        "voice/config/voice_agent.yaml",
        'config: "../../text_frontend_backend_agent/config/agent.yaml"   # [in-path]',
        'config: "../../text/config/agent.yaml"   # [in-path]',
        1,
        "the text agent package is named text in this example",
    ),
    (
        "voice/config/voice_agent.yaml",
        'example_dir: "../../../examples/frontend_backend_agent"   # [in-path]',
        'example_dir: "../.."   # [in-path]',
        2,
        "ASR and TTS catalogs: this example's services.*.yaml (the prototype, under src/prototypes/, read the "
        "frontend_backend_agent example's catalog; the keys and nemo-speech endpoints are the same)",
    ),
)

# Single files copied verbatim to another place in the example: (prototype path, example path).
# prompts.yaml is the registry-facing prompt catalog; the runtime still reads its own copy,
# and a test asserts the two are identical.
EXTRA_FILES: tuple[tuple[str, str], ...] = (
    ("src/prototypes/voice_frontend_backend_agent/config/prompts.voice.yaml", "prompts.yaml"),
)

# Ported unit tests: (prototype path, test path). Helper modules get an fbv prefix so their
# bare-name imports cannot collide with other test directories.
TEST_HELPER_RENAMES: tuple[tuple[str, str], ...] = (
    ("_fakes", "_fbv_fakes"),
    ("_voice_fakes", "_fbv_voice_fakes"),
)
TEXT_TEST_EXCLUDES = ("test_ui.py",)  # the excluded rich terminal UI

# Test-only rewrites: repository paths differ (the tests live one directory higher, the
# packages under src/examples/frontend_backend_verdict), and one packaging test checks
# the prototype repository's pyproject extras, which this repository does not have.
TEST_TEXT_REWRITES: tuple[tuple[str, str], ...] = (
    (
        '/ "src" / "prototypes" / "text_frontend_backend_agent"',
        '/ "src" / "examples" / "frontend_backend_verdict" / "text"',
    ),
    (
        '/ "src" / "prototypes" / "voice_frontend_backend_agent"',
        '/ "src" / "examples" / "frontend_backend_verdict" / "voice"',
    ),
    (
        '    / "src"\n    / "prototypes"\n    / "text_frontend_backend_agent"\n',
        '    / "src"\n    / "examples"\n    / "frontend_backend_verdict"\n    / "text"\n',
    ),
    ("REPO_ROOT = Path(__file__).resolve().parents[4]", "REPO_ROOT = Path(__file__).resolve().parents[3]"),
    (
        'REPO_ROOT / "misc" / "prototypes" / "voice" / "verdict_cases.jsonl"',
        'REPO_ROOT / "tests" / "unit" / "frontend_backend_verdict" / "fixtures" / "verdict_cases.jsonl"',
    ),
    (
        "    def test_declared_extra_lists_the_direct_dependencies(self) -> None:\n",
        '    @unittest.skip("prototype-repository packaging: this repository declares the dependencies itself")\n'
        "    def test_declared_extra_lists_the_direct_dependencies(self) -> None:\n",
    ),
)
# Extra fixture files copied from outside tests/: (prototype path, fixture name).
EXTRA_FIXTURES: tuple[tuple[str, str], ...] = (("misc/prototypes/voice/verdict_cases.jsonl", "verdict_cases.jsonl"),)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def rewrite(text: str) -> str:
    """Apply the package rewrites (forward)."""
    for old, new in PACKAGE_REWRITES:
        text = text.replace(old, new)
    return text


def unrewrite(text: str) -> str:
    """Reverse :func:`rewrite`."""
    for old, new in reversed(PACKAGE_REWRITES):
        text = text.replace(new, old)
    return text


def _is_text(path: Path) -> bool:
    return path.suffix in {".py", ".yaml", ".yml", ".json", ".jsonl", ".md", ".html", ".txt"}


def _copy_tree(src: Path, dst: Path, excludes: tuple[str, ...], prefix: str, files: dict[str, dict]) -> None:
    for path in sorted(src.rglob("*")):
        rel = path.relative_to(src)
        if "__pycache__" in rel.parts or path.is_dir():
            continue
        if any(rel.as_posix() == ex or rel.as_posix().startswith(ex + "/") for ex in excludes):
            continue
        original = path.read_bytes()
        target = dst / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        key = f"{prefix}/{rel.as_posix()}"
        if _is_text(path):
            content = rewrite(original.decode("utf-8"))
            target.write_text(content, encoding="utf-8")
        else:
            target.write_bytes(original)
        files[key] = {"source": str(path.relative_to(src.parents[2])), "sha256": _sha256(original)}


def _apply_value_edits(files: dict[str, dict]) -> None:
    for rel, old, new, count, reason in VALUE_EDITS:
        target = EXAMPLE_DIR / rel
        text = target.read_text(encoding="utf-8")
        if text.count(old) != count:
            raise SystemExit(f"value edit for {rel} matches {text.count(old)} times, expected {count}: {old!r}")
        target.write_text(text.replace(old, new), encoding="utf-8")
        files[rel].setdefault("value_edits", []).append({"old": old, "new": new, "count": count, "reason": reason})


def _copy_tests(proto_root: Path, files: dict[str, dict]) -> None:
    text_src = proto_root / "tests" / "unit" / "prototypes"
    voice_src = text_src / "voice"
    TEST_DIR.mkdir(parents=True, exist_ok=True)

    def port(text: str) -> str:
        text = rewrite(text)
        for old, new in TEST_TEXT_REWRITES:
            text = text.replace(old, new)
        for old, new in TEST_HELPER_RENAMES:
            text = text.replace(f"from {old} import", f"from {new} import")
            text = text.replace(f"import {old}\n", f"import {new}\n")
        return text

    def name_for(path: Path) -> str:
        stem = path.name
        for old, new in TEST_HELPER_RENAMES:
            if path.stem == old:
                return f"{new}.py"
        if stem.startswith("test_voice_"):
            return "test_fbv_" + stem[len("test_") :]
        if stem.startswith("test_"):
            return "test_fbv_" + stem[len("test_") :]
        return stem

    for src_dir, sub in ((text_src, ""), (voice_src, "")):
        for path in sorted(src_dir.glob("*.py")):
            if path.name in TEXT_TEST_EXCLUDES:
                continue
            target = TEST_DIR / sub / name_for(path)
            if target.exists() and src_dir is text_src:
                raise SystemExit(f"test name collision: {target}")
            target.write_text(port(path.read_text(encoding="utf-8")), encoding="utf-8")
            files[f"tests/{target.name}"] = {
                "source": str(path.relative_to(proto_root)),
                "sha256": _sha256(path.read_bytes()),
                "kind": "test",
            }
    fixtures = voice_src / "fixtures"
    target_fixtures = TEST_DIR / "fixtures"
    if target_fixtures.exists():
        shutil.rmtree(target_fixtures)
    for path in sorted(fixtures.rglob("*")):
        if path.is_dir() or "__pycache__" in path.parts:
            continue
        rel = path.relative_to(fixtures)
        target = target_fixtures / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        data = path.read_bytes()
        if _is_text(path):
            target.write_text(rewrite(data.decode("utf-8")), encoding="utf-8")
        else:
            target.write_bytes(data)
    for src_rel, name in EXTRA_FIXTURES:
        (target_fixtures / name).write_bytes((proto_root / src_rel).read_bytes())


def main(argv: list[str] | None = None) -> None:
    """Copy the prototype and write ``provenance.json``."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--prototype-root", required=True, help="checkout of nemotron-voice-agent-smasurekar")
    args = parser.parse_args(argv)
    proto_root = Path(args.prototype_root).resolve()
    commit = subprocess.run(
        ["git", "-C", str(proto_root), "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()
    files: dict[str, dict] = {}
    # Remove the previously ported tests only; tests written for the example itself stay.
    if PROVENANCE_FILE.is_file():
        previous = json.loads(PROVENANCE_FILE.read_text(encoding="utf-8"))
        for key, entry in previous.get("files", {}).items():
            if entry.get("kind") == "test":
                (TEST_DIR / key.removeprefix("tests/")).unlink(missing_ok=True)
    for src_rel, dst_rel, excludes in TREES:
        dst = EXAMPLE_DIR / dst_rel
        if dst.exists():
            shutil.rmtree(dst)
        _copy_tree(proto_root / src_rel, dst, excludes, dst_rel, files)
    _apply_value_edits(files)
    for src_rel, dst_rel in EXTRA_FILES:
        original = (proto_root / src_rel).read_bytes()
        (EXAMPLE_DIR / dst_rel).write_text(rewrite(original.decode("utf-8")), encoding="utf-8")
        files[dst_rel] = {"source": src_rel, "sha256": _sha256(original)}
    _copy_tests(proto_root, files)
    provenance = {
        "prototype_repository": "nemotron-voice-agent-smasurekar",
        "prototype_commit": commit,
        "package_rewrites": [list(pair) for pair in PACKAGE_REWRITES],
        "test_helper_renames": [list(pair) for pair in TEST_HELPER_RENAMES],
        "files": files,
    }
    PROVENANCE_FILE.write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"copied {len(files)} files from {proto_root} @ {commit[:7]}")


if __name__ == "__main__":
    main()
