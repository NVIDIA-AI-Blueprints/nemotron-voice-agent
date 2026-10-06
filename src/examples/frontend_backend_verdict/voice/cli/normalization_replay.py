# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""Offline replay of the normalization hooks over a recorded event log (plan section 9.1).

No ASR, no LLM: it reads an event log written by an earlier run and applies a
profile's ``normalization`` settings to what the run recorded, one JSONL record
per item:

* ``asr``: each ``asr_final`` transcript, raw and normalized;
* ``call``: each backend tool call, its canonical arguments, and the decision the
  tool-argument hook would take (``sent``, ``invalid``, ``already_failed``).

This is a first-order replay: a local answer would have changed what happened
next, which a log cannot show. The retry guard is simulated conservatively: a
recorded permanent failure marks a call as failed only when the recorded
arguments equal the canonical ones (otherwise the canonical lookup's outcome is
unknown). Run from the repository root::

    PYTHONPATH=src uv run python -m examples.frontend_backend_verdict.voice.cli.normalization_replay \
        --events logs/fba_voice_events.jsonl \
        --config src/prototypes/voice_frontend_backend_agent/config/profiles/tau3_eval_normalization.yaml \
        --model pine-fba-voice-paired-airline-regular-sil800 --out /tmp/replay.jsonl

A summary goes to stderr. The log must not have been written with
``logging.redact_content: true`` (the transcripts and arguments are needed).
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any, TextIO

from examples.frontend_backend_verdict.text.messages import ToolCall, canonical_json
from examples.frontend_backend_verdict.text.prompts import load_catalog
from examples.frontend_backend_verdict.voice.config import VoiceConfig, load_voice_config
from examples.frontend_backend_verdict.voice.normalization.arguments import ArgumentNormalizer, FailureKey
from examples.frontend_backend_verdict.voice.normalization.transcript import TranscriptNormalizer


def _records(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _sessions(records: Iterable[dict[str, Any]], model: str) -> dict[str, list[dict[str, Any]]]:
    """Records grouped per session, in log order; only sessions of ``model`` when given."""
    by_session: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    models: dict[str, str] = {}
    for record in records:
        session_id = str(record.get("session_id") or "")
        if record.get("kind") == "session_start":
            models[session_id] = str(record.get("model") or "")
        by_session[session_id].append(record)
    return {sid: items for sid, items in by_session.items() if not model or models.get(sid) == model}


def _argument_normalizer(config: VoiceConfig) -> ArgumentNormalizer | None:
    settings = config.normalization.tool_arguments
    if not settings.enabled:
        return None
    catalog = load_catalog(config.agent.prompts_path, config.agent.prompts.inline)
    guard = settings.retry_guard
    return ArgumentNormalizer(
        settings,
        transcript=config.normalization.transcript,
        invalid_template=catalog.get(settings.invalid_message_key),
        already_failed_template=catalog.get(guard.message_key) if guard.enabled else "",
    )


def replay(config: VoiceConfig, sessions: dict[str, list[dict[str, Any]]], out: TextIO) -> collections.Counter[str]:
    """Write one JSONL record per ASR final and tool call; return the summary counts."""
    transcript = config.normalization.transcript
    normalizer = TranscriptNormalizer(transcript) if transcript.enabled else None
    arguments = _argument_normalizer(config)
    counts: collections.Counter[str] = collections.Counter(sessions=len(sessions))
    for session_id, records in sessions.items():
        failed: frozenset[FailureKey] = frozenset()
        pending: dict[str, tuple[FailureKey, bool]] = {}  # call_id -> (key, recorded == canonical)
        for record in records:
            kind = record.get("kind")
            if kind == "asr_final":
                raw = str(record.get("transcript") or "")
                text = normalizer.normalize(raw).text if normalizer else raw
                counts["asr"] += 1
                counts["asr_rewritten"] += text != raw
                _write(out, session_id, "asr", raw=raw, normalized=text, changed=text != raw)
            elif kind == "backend_tool_calls" and arguments is not None:
                calls = [
                    ToolCall(
                        id=str(c.get("id")), name=str(c.get("name")), arguments_json=canonical_json(c.get("arguments"))
                    )
                    for c in record.get("calls") or []
                ]
                screening = arguments.screen(calls, failed)
                local = {answer.call_id: answer.reason for answer in screening.local}
                for original, canonical in zip(calls, screening.calls, strict=True):
                    decision = local.get(original.id, "sent")
                    counts["calls"] += 1
                    counts[f"calls_{decision}"] += 1
                    counts["calls_rewritten"] += canonical.arguments_json != original.arguments_json
                    if original.id in screening.keys:
                        same = canonical.arguments_json == original.arguments_json
                        pending[original.id] = (screening.keys[original.id], same)
                    _write(
                        out,
                        session_id,
                        "call",
                        call_id=original.id,
                        tool=original.name,
                        recorded=json.loads(original.arguments_json),
                        canonical=json.loads(canonical.arguments_json),
                        decision=decision,
                    )
            elif kind == "tool_output_in" and arguments is not None:
                key, same = pending.pop(str(record.get("call_id")), (None, False))
                if key is not None and same and arguments.is_permanent_failure(str(record.get("output") or "")):
                    failed = failed | {key}
    return counts


def _write(out: TextIO, session_id: str, kind: str, **data: Any) -> None:
    out.write(json.dumps({"session_id": session_id, "kind": kind, **data}, ensure_ascii=False) + "\n")


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--events", required=True, type=Path, help="event log JSONL of an earlier run")
    parser.add_argument("--config", required=True, type=Path, help="voice config or profile with normalization")
    parser.add_argument("--model", default="", help="only sessions whose session_start model equals this")
    parser.add_argument("--out", type=Path, default=None, help="output JSONL (default: stdout)")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    args = _parse_args(argv)
    config = load_voice_config(args.config)
    sessions = _sessions(_records(args.events), args.model)
    if args.out is None:
        counts = replay(config, sessions, sys.stdout)
    else:
        with args.out.open("w", encoding="utf-8") as handle:
            counts = replay(config, sessions, handle)
    print(json.dumps(dict(counts), sort_keys=True), file=sys.stderr)


if __name__ == "__main__":
    main()
