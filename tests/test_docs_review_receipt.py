# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""Tests for the documentation-writer review receipt tooling."""

# ruff: noqa: D103

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

_SCRIPT = Path(__file__).parent.parent / "scripts" / "docs-review-receipt.py"
_HEAD_SHA = "a" * 40
_AGENTS_BLOB_SHA = "b" * 40


def _receipt(**overrides: str) -> str:
    values = {
        "checked": "x",
        "result": "`docs-updated`",
        "evidence": "Updated docs/01-getting-started.md.",
        "agent": "Codex Desktop",
        "head_sha": _HEAD_SHA[:12],
        "agents_sha": _AGENTS_BLOB_SHA[:12],
        **overrides,
    }
    return f"""## Documentation Writer Review

- [{values["checked"]}] Documentation writer reviewed the completed changes
- Result: {values["result"]}
- Evidence: {values["evidence"]}
- Agent: {values["agent"]}
<!-- docs-review-head-sha: {values["head_sha"]} -->
<!-- docs-review-agents-blob-sha: {values["agents_sha"]} -->
"""


def _run_check(
    tmp_path: Path,
    body: str,
    changed_files: list[str],
    *,
    mode: str = "advisory",
    agents_blob: str = _AGENTS_BLOB_SHA,
) -> subprocess.CompletedProcess[str]:
    event_path = tmp_path / "event.json"
    files_path = tmp_path / "changed-files.txt"
    summary_path = tmp_path / "summary.md"
    event_path.write_text(
        json.dumps(
            {
                "pull_request": {
                    "number": 42,
                    "html_url": "https://github.com/NVIDIA-AI-Blueprints/nemotron-voice-agent/pull/42",
                    "body": body,
                    "head": {"sha": _HEAD_SHA},
                }
            }
        ),
        encoding="utf-8",
    )
    files_path.write_text(f"{'\n'.join(changed_files)}\n", encoding="utf-8")
    return subprocess.run(
        [
            sys.executable,
            str(_SCRIPT),
            "check",
            "--event",
            str(event_path),
            "--changed-files",
            str(files_path),
            "--agents-blob",
            agents_blob,
            "--mode",
            mode,
        ],
        capture_output=True,
        check=False,
        env={**os.environ, "GITHUB_STEP_SUMMARY": str(summary_path)},
        text=True,
        timeout=10,
    )


def test_accepts_fresh_receipt_for_code_and_documentation(tmp_path: Path) -> None:
    result = _run_check(tmp_path, _receipt(), ["src/server.py", "docs/01-getting-started.md"])

    assert result.returncode == 0
    output = json.loads(result.stdout)
    assert output["status"] == "valid"
    assert output["result"] == "docs-updated"
    assert output["headShaMatches"] is True
    assert output["agentsShaMatches"] is True
    assert output["issues"] == []


def test_reports_missing_receipt_without_blocking_advisory_mode(tmp_path: Path) -> None:
    result = _run_check(tmp_path, "## Summary\n\nChange the server.\n", ["src/server.py"])

    assert result.returncode == 0
    assert json.loads(result.stdout)["status"] == "missing"
    assert "change code or documentation must include" in result.stderr


def test_fails_required_mode_when_receipt_is_missing(tmp_path: Path) -> None:
    result = _run_check(tmp_path, "## Summary\n\nChange the server.\n", ["src/server.py"], mode="required")

    assert result.returncode == 1
    assert json.loads(result.stdout)["status"] == "missing"


def test_rejects_stale_head_and_agents_revisions(tmp_path: Path) -> None:
    body = _receipt(
        result="`no-docs-needed`",
        evidence="The change affects an internal test helper only.",
        head_sha="c" * 12,
        agents_sha="d" * 12,
    )
    result = _run_check(tmp_path, body, ["tests/test_service_catalog.py"])

    output = json.loads(result.stdout)
    assert output["status"] == "invalid"
    assert "The documentation writer review is stale after a new commit." in output["issues"]
    assert "The reviewed AGENTS.md blob SHA does not match the pull request version." in output["issues"]


def test_docs_updated_requires_a_documentation_path(tmp_path: Path) -> None:
    result = _run_check(tmp_path, _receipt(), ["src/server.py"])

    output = json.loads(result.stdout)
    assert output["status"] == "invalid"
    assert "The docs-updated result requires a changed Markdown or docs/ file." in output["issues"]


def test_rejects_duplicate_receipt_fields(tmp_path: Path) -> None:
    body = _receipt().replace(
        "- Result: `docs-updated`",
        "- Result: `docs-updated`\n- Result: `no-docs-needed`",
    )
    result = _run_check(tmp_path, body, ["src/server.py", "docs/01-getting-started.md"])

    output = json.loads(result.stdout)
    assert output["status"] == "invalid"
    assert "The Documentation Writer Review section repeats singleton fields: Result." in output["issues"]


def test_rejects_duplicate_receipt_sections(tmp_path: Path) -> None:
    result = _run_check(
        tmp_path,
        f"{_receipt()}\n{_receipt()}",
        ["src/server.py", "docs/01-getting-started.md"],
    )

    output = json.loads(result.stdout)
    assert output["status"] == "invalid"
    assert "The PR description contains more than one Documentation Writer Review section." in output["issues"]


def test_rejects_unmodified_template_fields(tmp_path: Path) -> None:
    body = _receipt(
        checked=" ",
        result="`docs-updated` | `no-docs-needed` | `blocked`",
        evidence="",
        agent="<Codex Desktop | Codex CLI | Claude Code | Cursor>",
        head_sha="",
        agents_sha="",
    )
    result = _run_check(tmp_path, body, ["docs/01-getting-started.md"])

    output = json.loads(result.stdout)
    assert output["status"] == "invalid"
    assert len(output["issues"]) == 6


def test_report_emits_summary_and_formula_safe_csv(tmp_path: Path) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    gh_path = bin_dir / "gh"
    pull_requests = [
        {
            "number": 1,
            "url": "https://github.com/NVIDIA-AI-Blueprints/nemotron-voice-agent/pull/1",
            "state": "MERGED",
            "isDraft": False,
            "author": {"login": "engineer"},
            "createdAt": "2026-06-13T00:00:00Z",
            "mergedAt": "2026-06-14T00:00:00Z",
            "headRefOid": _HEAD_SHA,
            "body": f"""## Type of Change

- [x] Code change with doc updates

{_receipt(evidence="=1+1")}""",
        },
        {
            "number": 2,
            "url": "https://github.com/NVIDIA-AI-Blueprints/nemotron-voice-agent/pull/2",
            "state": "OPEN",
            "isDraft": True,
            "author": {"login": "engineer"},
            "createdAt": "2026-06-15T00:00:00Z",
            "mergedAt": None,
            "headRefOid": "c" * 40,
            "body": "## Type of Change\n\n- [x] Code change (feature, bug fix, or refactor)\n",
        },
        {
            "number": 3,
            "url": "https://github.com/NVIDIA-AI-Blueprints/nemotron-voice-agent/pull/3",
            "state": "MERGED",
            "isDraft": False,
            "author": {"login": "writer"},
            "createdAt": "2026-06-16T00:00:00Z",
            "mergedAt": "2026-06-17T00:00:00Z",
            "headRefOid": "d" * 40,
            "body": "## Type of Change\n\n- [x] Doc only (prose changes, no code sample modifications)\n",
        },
    ]
    gh_path.write_text(
        f"#!/usr/bin/env python3\nimport json\nprint(json.dumps({pull_requests!r}))\n",
        encoding="utf-8",
    )
    gh_path.chmod(0o755)
    env = {**os.environ, "PATH": f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}"}
    command = [
        sys.executable,
        str(_SCRIPT),
        "report",
        "--since",
        "2026-06-12",
        "--until",
        "2026-06-12",
    ]

    json_result = subprocess.run(command, capture_output=True, check=False, env=env, text=True, timeout=10)
    assert json_result.returncode == 0
    metrics = json.loads(json_result.stdout)["metrics"]
    assert metrics["eligiblePrs"] == 3
    assert metrics["recordedReceipts"] == 1
    assert metrics["validReceipts"] == 1
    assert metrics["receiptCoverage"] == 0.3333

    csv_result = subprocess.run(
        [*command, "--format", "csv"],
        capture_output=True,
        check=False,
        env=env,
        text=True,
        timeout=10,
    )
    assert csv_result.returncode == 0
    assert "receipt_status" in csv_result.stdout
    assert "1,https://github.com/NVIDIA-AI-Blueprints/nemotron-voice-agent/pull/1" in csv_result.stdout
    assert "'=1+1" in csv_result.stdout
