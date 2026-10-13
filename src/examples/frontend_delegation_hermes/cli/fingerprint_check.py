# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""Check that every session of an arm ran the arm's deployed fingerprint (tau3-failure-fixes-plan.md 1, 8).

The voice server logs its switches (``fdh_session_start.features``) and the hash of the
rendered frontend prompt (``frontend_prompt``); the gateway reports its prompt variants,
catalog hash, the per-session SOUL / system prompt hashes, the detected domain and the hash of the
session's tool schemas (``backend_configured``).
An arm is scorable only if every session has all of them, all sessions agree, and the
switches match the arm's profile and gateway config. Exit 1 otherwise (re-run the arm).

    PYTHONPATH=src uv run python -m examples.frontend_delegation_hermes.cli.fingerprint_check \
        logs/fdh_voice_events.jsonl \
        --profile src/examples/frontend_delegation_hermes/config/profiles/tau3_eval_baseline.yaml \
        --gateway-config src/examples/frontend_delegation_hermes/config/gateway.baseline.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

BACKEND_KEYS = (
    "backend_features",
    "backend_catalog_sha256",
    "backend_soul_sha256",
    "backend_system_sha256",
    "backend_domain",  # which domain note the session got ("" with gateway.yaml: no domains)
    "session_tools_sha256",  # the session's tool schemas (the input of the schema-derived rules)
)


def session_fingerprints(events: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """``session_id -> fingerprint`` from a voice event log (missing parts are ``None``)."""
    sessions: dict[str, dict[str, Any]] = defaultdict(dict)
    for event in events:
        session, kind = str(event.get("session_id") or ""), event.get("kind")
        if kind == "fdh_session_start":
            sessions[session]["features"] = event.get("features")
            sessions[session]["invalid_message_keys"] = event.get("invalid_message_keys")
        elif kind == "frontend_prompt":
            sessions[session]["frontend_prompt_sha256"] = event.get("frontend_prompt_sha256")
        elif kind == "backend_configured" and event.get("backend_catalog_sha256"):
            for key in BACKEND_KEYS:
                sessions[session][key] = event.get(key)
    keys = ("features", "invalid_message_keys", "frontend_prompt_sha256", *BACKEND_KEYS)
    return {sid: {key: parts.get(key) for key in keys} for sid, parts in sessions.items() if "features" in parts}


def expected(profile: str, gateway_config: str) -> dict[str, Any]:
    """The switches an arm declares: the voice profile's features and the gateway's prompt variants."""
    from examples.frontend_delegation_hermes.core.config import load_delegation_config  # noqa: PLC0415
    from examples.frontend_delegation_hermes.core.sidecar.gateway_config import load_gateway_config  # noqa: PLC0415
    from examples.frontend_delegation_hermes.core.sidecar.templates import BackendTemplates  # noqa: PLC0415

    voice = load_delegation_config(profile)
    gateway = load_gateway_config(gateway_config)
    templates = BackendTemplates(gateway.hermes.prompts_path, prompt_features=gateway.prompt_features)
    return {
        "features": voice.features,
        "backend_features": dict(gateway.prompt_features),
        "backend_catalog_sha256": templates.catalog_sha256,
    }


def check(fingerprints: dict[str, dict[str, Any]], want: dict[str, Any] | None = None) -> list[str]:
    """Problems found (empty = the arm is scorable)."""
    problems: list[str] = []
    if not fingerprints:
        return ["no fdh_session_start with features in the log (voice server older than the fingerprint change?)"]
    for session, parts in sorted(fingerprints.items()):
        missing = [key for key, value in parts.items() if value is None]
        if missing:
            problems.append(f"{session}: missing {missing}")
        for key, value in (want or {}).items():
            if parts.get(key) is not None and parts[key] != value:
                problems.append(f"{session}: {key} is {parts[key]!r}, the arm declares {value!r}")
    distinct = Counter(json.dumps(parts, sort_keys=True) for parts in fingerprints.values())
    if len(distinct) > 1:
        problems.append(f"{len(distinct)} different fingerprints across {len(fingerprints)} sessions")
    return problems


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("events", help="voice events.jsonl of one arm (one domain)")
    parser.add_argument("--profile", default="", help="the arm's voice profile (checks the voice switches)")
    parser.add_argument("--gateway-config", default="", help="the arm's gateway config (checks backend variants)")
    args = parser.parse_args(argv)
    events = [json.loads(line) for line in Path(args.events).read_text(encoding="utf-8").splitlines() if line.strip()]
    fingerprints = session_fingerprints(events)
    want: dict[str, Any] = {}
    if args.profile or args.gateway_config:
        if not (args.profile and args.gateway_config):
            parser.error("--profile and --gateway-config go together")
        want = expected(args.profile, args.gateway_config)
    problems = check(fingerprints, want)
    unique = {json.dumps(parts, sort_keys=True) for parts in fingerprints.values()}
    print(
        json.dumps({"sessions": len(fingerprints), "fingerprints": [json.loads(u) for u in sorted(unique)]}, indent=2)
    )
    for problem in problems:
        print(f"MISMATCH {problem}")
    print("fingerprint: OK" if not problems else f"fingerprint: FAILED ({len(problems)} problem(s))")
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
