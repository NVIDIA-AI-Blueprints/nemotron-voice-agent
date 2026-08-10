#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""Validate documentation-writer review receipts and report their adoption."""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any

_RESULTS = {"blocked", "docs-updated", "no-docs-needed"}
_SHA_PATTERN = re.compile(r"^[0-9a-f]{7,40}$")
_REPOSITORY_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
_DEFAULT_REPOSITORY = "NVIDIA-AI-Blueprints/nemotron-voice-agent"


@dataclass
class _ParsedReceipt:
    agent: str | None
    agents_blob_sha: str | None
    completed: bool
    duplicate_fields: list[str]
    duplicate_sections: bool
    evidence: str | None
    present: bool
    result: str | None
    reviewed_head_sha: str | None


def main() -> int:
    """Run the receipt check or historical report command."""
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    check_parser = subparsers.add_parser("check", help="Check one pull request event")
    check_parser.add_argument("--event", required=True, type=Path)
    check_parser.add_argument("--changed-files", required=True, type=Path)
    check_parser.add_argument("--agents-blob", required=True)
    check_parser.add_argument("--mode", choices=("advisory", "required"), default="advisory")

    report_parser = subparsers.add_parser("report", help="Report receipt adoption")
    report_parser.add_argument("--repo", default=_DEFAULT_REPOSITORY)
    report_parser.add_argument("--since", required=True)
    report_parser.add_argument("--until", default=date.today().isoformat())
    report_parser.add_argument("--format", choices=("csv", "json", "summary"), default="json")

    args = parser.parse_args()
    if args.command == "check":
        return _run_check(args)
    return _run_report(args)


def _run_check(args: argparse.Namespace) -> int:
    event = _read_json(args.event)
    pull_request = event.get("pull_request")
    head_pr_sha = _nested_string(pull_request, "head", "sha")
    if not isinstance(pull_request, dict) or not head_pr_sha or not _SHA_PATTERN.fullmatch(head_pr_sha):
        raise ValueError("The event file does not contain a valid pull request head SHA")

    body = pull_request.get("body") if isinstance(pull_request.get("body"), str) else ""
    changed_files = [line.strip() for line in args.changed_files.read_text(encoding="utf-8").splitlines() if line]
    record = _evaluate_receipt(
        body,
        _classify_changed_files(changed_files),
        head_pr_sha,
        args.agents_blob,
    )
    output = {
        "type": "documentation-writer-review-receipt",
        "pr": pull_request.get("number"),
        "url": pull_request.get("html_url"),
        "headPrSha": head_pr_sha,
        **record,
    }
    print(json.dumps(output, sort_keys=True))
    _write_step_summary(output)

    if record["status"] in {"missing", "invalid"}:
        for issue in record["issues"]:
            print(
                f"::warning title=Documentation writer review receipt::{_escape_annotation(issue)}",
                file=sys.stderr,
            )
        if args.mode == "required":
            return 1
    return 0


def _run_report(args: argparse.Namespace) -> int:
    if not _REPOSITORY_PATTERN.fullmatch(args.repo):
        raise ValueError(f"Invalid --repo value: {args.repo}")
    since = _parse_date(args.since, "--since")
    through = _parse_date(args.until, "--until")
    if since > through:
        raise ValueError("--since must not be later than --until")

    pull_requests = _list_pull_requests(args.repo, since, through)
    records = [_to_report_record(pull_request) for pull_request in pull_requests]
    report = _build_report(args.repo, args.since, args.until, records)

    if args.format == "csv":
        sys.stdout.write(_render_csv(records))
    elif args.format == "summary":
        summary = {key: value for key, value in report.items() if key != "records"}
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def _evaluate_receipt(
    body: str,
    changes: dict[str, bool | None],
    head_pr_sha: str,
    expected_agents_blob: str | None = None,
) -> dict[str, Any]:
    parsed = _parse_receipt(body)
    code_changed = changes["codeChanged"]
    docs_changed = changes["docsChanged"]
    issues: list[str] = []

    if code_changed is None or docs_changed is None:
        issues.append("The PR description does not select one code or documentation-only change type.")
        return _receipt_record(parsed, changes, issues, "unclassified")

    if not code_changed and not docs_changed:
        return _receipt_record(parsed, changes, issues, "not-required")

    if not parsed.present:
        issues.append(
            "Pull requests that change code or documentation must include the Documentation Writer Review section."
        )
    else:
        if parsed.duplicate_sections:
            issues.append("The PR description contains more than one Documentation Writer Review section.")
        if parsed.duplicate_fields:
            issues.append(
                "The Documentation Writer Review section repeats singleton fields: "
                f"{', '.join(parsed.duplicate_fields)}."
            )
        if not parsed.completed:
            issues.append("Mark the documentation writer review as completed.")
        if not parsed.result:
            issues.append("Keep exactly one result: docs-updated, no-docs-needed, or blocked.")
        if not parsed.evidence or _looks_like_placeholder(parsed.evidence):
            issues.append("Add documentation review evidence or a no-docs-needed rationale.")
        if not parsed.agent or _looks_like_placeholder(parsed.agent):
            issues.append("Record the agent surface that ran the documentation writer review.")
        if not parsed.reviewed_head_sha:
            issues.append("Refresh the hidden head SHA after the documentation writer review.")
        if not parsed.agents_blob_sha:
            issues.append("Record a valid AGENTS.md blob SHA with 7 to 40 hexadecimal characters.")
        if parsed.result == "docs-updated" and docs_changed is not True:
            issues.append("The docs-updated result requires a changed Markdown or docs/ file.")

    head_matches = bool(parsed.reviewed_head_sha and head_pr_sha.lower().startswith(parsed.reviewed_head_sha))
    if parsed.reviewed_head_sha and not head_matches:
        issues.append("The documentation writer review is stale after a new commit.")

    agents_matches: bool | None = None
    if parsed.agents_blob_sha and expected_agents_blob:
        agents_matches = expected_agents_blob.strip().lower().startswith(parsed.agents_blob_sha)
        if not agents_matches:
            issues.append("The reviewed AGENTS.md blob SHA does not match the pull request version.")

    status = "missing" if not parsed.present else ("valid" if not issues else "invalid")
    return _receipt_record(
        parsed,
        changes,
        issues,
        status,
        head_matches=head_matches if parsed.reviewed_head_sha else None,
        agents_matches=agents_matches,
    )


def _receipt_record(
    parsed: _ParsedReceipt,
    changes: dict[str, bool | None],
    issues: list[str],
    status: str,
    *,
    head_matches: bool | None = None,
    agents_matches: bool | None = None,
) -> dict[str, Any]:
    return {
        "agent": parsed.agent,
        "agentsBlobSha": parsed.agents_blob_sha,
        "agentsShaMatches": agents_matches,
        "codeChanged": changes["codeChanged"],
        "docsChanged": changes["docsChanged"],
        "evidence": parsed.evidence,
        "headShaMatches": head_matches,
        "issues": issues,
        "result": parsed.result,
        "reviewedHeadSha": parsed.reviewed_head_sha,
        "status": status,
    }


def _parse_receipt(body: str) -> _ParsedReceipt:
    matches = list(re.finditer(r"^## Documentation Writer Review\s*$", body, flags=re.MULTILINE))
    if not matches:
        return _ParsedReceipt(None, None, False, [], False, None, False, None, None)

    start = matches[0].end()
    remaining = body[start:]
    next_heading = re.search(r"^##\s+", remaining, flags=re.MULTILINE)
    section = remaining[: next_heading.start() if next_heading else len(remaining)]
    lines = [line.strip() for line in section.splitlines()]
    review_pattern = re.compile(
        r"^- \[[ xX]\] Documentation writer (?:subagent )?reviewed the completed (?:changes|implementation)$"
    )
    completion_pattern = re.compile(
        r"^- \[[xX]\] Documentation writer (?:subagent )?reviewed the completed (?:changes|implementation)$"
    )
    duplicate_fields = [name for name in ("Result", "Evidence", "Agent") if len(_field_values(lines, name)) > 1]
    for name in ("docs-review-head-sha", "docs-review-agents-blob-sha"):
        if len(_hidden_field_values(lines, name)) > 1:
            duplicate_fields.append(name)
    if sum(bool(review_pattern.fullmatch(line)) for line in lines) > 1:
        duplicate_fields.append("review completion checkbox")

    result_value = _field_value(lines, "Result")
    result_match = re.fullmatch(r"`(blocked|docs-updated|no-docs-needed)`", result_value or "")
    result = result_match.group(1) if result_match and result_match.group(1) in _RESULTS else None
    return _ParsedReceipt(
        agent=_non_empty(_field_value(lines, "Agent")),
        agents_blob_sha=_parse_sha(_hidden_field_value(lines, "docs-review-agents-blob-sha")),
        completed=any(completion_pattern.fullmatch(line) for line in lines),
        duplicate_fields=duplicate_fields,
        duplicate_sections=len(matches) > 1,
        evidence=_non_empty(_field_value(lines, "Evidence")),
        present=True,
        result=result,
        reviewed_head_sha=_parse_sha(_hidden_field_value(lines, "docs-review-head-sha")),
    )


def _field_value(lines: list[str], name: str) -> str | None:
    values = _field_values(lines, name)
    return values[0] if values else None


def _field_values(lines: list[str], name: str) -> list[str]:
    prefix = f"- {name}:"
    return [line[len(prefix) :].strip() for line in lines if line.startswith(prefix)]


def _hidden_field_value(lines: list[str], name: str) -> str | None:
    values = _hidden_field_values(lines, name)
    return values[0] if values else None


def _hidden_field_values(lines: list[str], name: str) -> list[str]:
    prefix = f"<!-- {name}:"
    suffix = "-->"
    return [
        line[len(prefix) : -len(suffix)].strip() for line in lines if line.startswith(prefix) and line.endswith(suffix)
    ]


def _parse_sha(value: str | None) -> str | None:
    normalized = value.strip().strip("`").lower() if value else ""
    return normalized if _SHA_PATTERN.fullmatch(normalized) else None


def _non_empty(value: str | None) -> str | None:
    normalized = value.strip() if value else ""
    return normalized or None


def _looks_like_placeholder(value: str) -> bool:
    return bool(re.search(r"[<>|]", value))


def _classify_changed_files(changed_files: list[str]) -> dict[str, bool]:
    return {
        "codeChanged": any(not _is_documentation_file(file) for file in changed_files),
        "docsChanged": any(_is_documentation_file(file) for file in changed_files),
    }


def _classify_pr_type(body: str) -> dict[str, bool | None]:
    checked = [line.strip() for line in body.splitlines() if re.match(r"^- \[[xX]\] ", line.strip())]
    code_only = any(line.lower().startswith("- [x] code change (") for line in checked)
    code_with_docs = any(re.fullmatch(r"- \[[xX]\] Code change with doc updates", line) for line in checked)
    docs_only = any(re.fullmatch(r"- \[[xX]\] Doc only \(.+\)", line) for line in checked)
    if (code_only or code_with_docs) and not docs_only:
        return {"codeChanged": True, "docsChanged": code_with_docs}
    if docs_only and not code_only and not code_with_docs:
        return {"codeChanged": False, "docsChanged": True}
    return {"codeChanged": None, "docsChanged": None}


def _is_documentation_file(file: str) -> bool:
    lower = file.lower()
    return file.startswith("docs/") or lower.endswith(".md") or lower.endswith(".mdx")


def _to_report_record(pull_request: dict[str, Any]) -> dict[str, Any]:
    body = pull_request.get("body") if isinstance(pull_request.get("body"), str) else ""
    head_sha = pull_request.get("headRefOid")
    if not isinstance(head_sha, str) or not _SHA_PATTERN.fullmatch(head_sha):
        raise ValueError(f"Pull request #{pull_request.get('number')} has an invalid head SHA")
    author = pull_request.get("author")
    record = _evaluate_receipt(body, _classify_pr_type(body), head_sha)
    return {
        "author": author.get("login") if isinstance(author, dict) else None,
        "createdAt": pull_request.get("createdAt"),
        "headPrSha": head_sha,
        "isDraft": pull_request.get("isDraft"),
        "mergedAt": pull_request.get("mergedAt"),
        "number": pull_request.get("number"),
        "state": pull_request.get("state"),
        "url": pull_request.get("url"),
        **record,
    }


def _list_pull_requests(repository: str, since: date, through: date) -> list[dict[str, Any]]:
    pull_requests = _query_pull_request_range(repository, since, through)
    if len(pull_requests) < 1000:
        return pull_requests
    if since == through:
        raise RuntimeError(f"The {since.isoformat()} report reached GitHub's 1000-PR search limit.")
    midpoint = since + timedelta(days=(through - since).days // 2)
    return [
        *_list_pull_requests(repository, since, midpoint),
        *_list_pull_requests(repository, midpoint + timedelta(days=1), through),
    ]


def _query_pull_request_range(repository: str, since: date, through: date) -> list[dict[str, Any]]:
    command = [
        "gh",
        "pr",
        "list",
        "--repo",
        repository,
        "--state",
        "all",
        "--limit",
        "1000",
        "--search",
        f"created:{since.isoformat()}..{through.isoformat()}",
        "--json",
        "number,url,state,isDraft,author,createdAt,mergedAt,headRefOid,body",
    ]
    try:
        completed = subprocess.run(command, check=True, capture_output=True, text=True, timeout=60)
    except subprocess.CalledProcessError as exc:
        details = exc.stderr.strip()
        raise RuntimeError(f"{' '.join(command)} failed{f': {details}' if details else ''}") from exc
    parsed = json.loads(completed.stdout)
    if not isinstance(parsed, list):
        raise RuntimeError("GitHub CLI did not return a JSON array")
    return parsed


def _parse_date(value: str, option: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"Invalid {option} value: {value}") from exc


def _build_report(repository: str, since: str, through: str, records: list[dict[str, Any]]) -> dict[str, Any]:
    eligible = [record for record in records if record["codeChanged"] is True or record["docsChanged"] is True]
    recorded = [record for record in eligible if record["status"] != "missing"]
    valid = [record for record in eligible if record["status"] == "valid"]
    fresh = [record for record in recorded if record["headShaMatches"] is True]
    result_counts = {result: 0 for result in sorted(_RESULTS)}
    agent_counts: dict[str, int] = {}
    for record in valid:
        if record["result"]:
            result_counts[record["result"]] += 1
        if record["agent"]:
            agent = record["agent"].lower()
            agent_counts[agent] = agent_counts.get(agent, 0) + 1

    return {
        "repository": repository,
        "since": since,
        "through": through,
        "metrics": {
            "totalPrs": len(records),
            "eligiblePrs": len(eligible),
            "eligibleCodePrs": sum(record["codeChanged"] is True for record in eligible),
            "eligibleDocsOnlyPrs": sum(
                record["codeChanged"] is False and record["docsChanged"] is True for record in eligible
            ),
            "unclassifiedPrs": sum(
                record["codeChanged"] is None or record["docsChanged"] is None for record in records
            ),
            "recordedReceipts": len(recorded),
            "receiptCoverage": _ratio(len(recorded), len(eligible)),
            "validReceipts": len(valid),
            "validReceiptRate": _ratio(len(valid), len(eligible)),
            "freshReceipts": len(fresh),
            "freshReceiptRate": _ratio(len(fresh), len(recorded)),
            "results": result_counts,
            "agents": agent_counts,
        },
        "records": records,
    }


def _ratio(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, 4) if denominator else None


def _render_csv(records: list[dict[str, Any]]) -> str:
    headers = [
        "number",
        "url",
        "state",
        "is_draft",
        "author",
        "created_at",
        "merged_at",
        "code_changed",
        "docs_changed",
        "receipt_status",
        "result",
        "agent",
        "reviewed_head_sha",
        "head_pr_sha",
        "head_sha_matches",
        "agents_blob_sha",
        "evidence",
        "issues",
    ]
    output = io.StringIO()
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(headers)
    for record in records:
        writer.writerow(
            _csv_cell(value)
            for value in (
                record["number"],
                record["url"],
                record["state"],
                record["isDraft"],
                record["author"],
                record["createdAt"],
                record["mergedAt"],
                record["codeChanged"],
                record["docsChanged"],
                record["status"],
                record["result"],
                record["agent"],
                record["reviewedHeadSha"],
                record["headPrSha"],
                record["headShaMatches"],
                record["agentsBlobSha"],
                record["evidence"],
                "; ".join(record["issues"]),
            )
        )
    return output.getvalue()


def _csv_cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        text = str(value).lower()
    else:
        text = str(value)
    return f"'{text}" if re.match(r"^[=+\-@\t\r]", text) else text


def _write_step_summary(output: dict[str, Any]) -> None:
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if not summary_path:
        return
    lines = [
        "## Documentation writer review receipt",
        "",
        f"- PR: #{output['pr']}" if output["pr"] is not None else "- PR: unknown",
        f"- Head PR SHA: `{output['headPrSha'][:12]}`",
        f"- Status: `{output['status']}`",
        f"- Result: `{output['result']}`" if output["result"] else "- Result: not recorded",
        f"- Agent: {output['agent'] or 'not recorded'}",
    ]
    if output["issues"]:
        lines.extend(("", "### Advisory findings", "", *(f"- {issue}" for issue in output["issues"])))
    with Path(summary_path).open("a", encoding="utf-8") as summary:
        summary.write("\n".join(lines) + "\n")


def _read_json(path: Path) -> dict[str, Any]:
    parsed = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(parsed, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return parsed


def _nested_string(value: Any, *keys: str) -> str | None:
    current = value
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current if isinstance(current, str) else None


def _escape_annotation(value: str) -> str:
    return value.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as error:
        print(str(error), file=sys.stderr)
        raise SystemExit(1) from error
