# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""Replay the spelling-hold predicate over a voice event log (tau3-failure-fixes-plan.md sections 3.2, 9).

For every ``asr_final`` it applies the configured predicate (the profile's normalization,
tool-argument rules and ``delegation.spelling_hold.complete_patterns``) and labels the turn
"continued" when the session's next ``speech_started`` came within ``--window-s``. A held turn
that continued is a merge; one that did not only adds ``hold_ms`` of latency.

    PYTHONPATH=src uv run python -m examples.frontend_delegation_hermes.cli.spelling_hold_replay \
        <events.jsonl> --config src/examples/frontend_delegation_hermes/config/profiles/tau3_arm_m3_spelling.yaml
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from examples.frontend_delegation_hermes.core.config import load_delegation_config
from examples.frontend_delegation_hermes.core.engine.spelling_hold import COMPLETE, SpellingHoldPredicate
from examples.frontend_delegation_hermes.core.runtime import DelegationRuntime


def predicate_for(config_path: str) -> SpellingHoldPredicate:
    """The predicate exactly as the voice server builds it for ``config_path`` (hold forced on)."""
    config = load_delegation_config(config_path)
    runtime = DelegationRuntime(config, chat_model=object())  # type: ignore[arg-type] - no model call is made
    arguments = runtime.parts(None).argument_normalizer  # type: ignore[arg-type] - context is unused here
    return SpellingHoldPredicate(
        config.voice.normalization.transcript,
        complete_patterns=config.delegation.spelling_hold.complete_patterns,
        arguments=arguments,
    )


def replay(events: list[dict[str, Any]], predicate: SpellingHoldPredicate, *, window_s: float) -> dict[str, Any]:
    """Counts of held / not held turns, split by continued / not continued."""
    by_session: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for event in events:
        if event.get("kind") in ("asr_final", "speech_started"):
            by_session[str(event.get("session_id"))].append(event)
    counts: Counter[str] = Counter()
    examples: dict[str, list[str]] = defaultdict(list)
    for items in by_session.values():
        items.sort(key=lambda item: item["timestamp"])
        accumulator = False
        for index, event in enumerate(items):
            if event["kind"] != "asr_final":
                continue
            text = str(event.get("transcript") or "")
            following = next((e for e in items[index + 1 :] if e["kind"] == "speech_started"), None)
            continued = following is not None and following["timestamp"] - event["timestamp"] <= window_s
            check = predicate.check(text, accumulator=accumulator)
            label = "continued" if continued else "not_continued"
            if check.hold:
                key = f"held_{label}"
            elif check.reason == COMPLETE:
                key = f"not_held_complete_{'user_id' if '_' in check.value else 'code'}_{label}"
            else:
                key = f"not_held_{check.reason}"
            counts[key] += 1
            if len(examples[key]) < 5:
                examples[key].append(text)
            accumulator = check.hold and continued
    return {
        "finals": sum(1 for e in events if e.get("kind") == "asr_final"),
        "counts": dict(counts),
        "examples": examples,
    }


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("events", help="voice events.jsonl")
    parser.add_argument("--config", required=True, help="delegation profile whose predicate is replayed")
    parser.add_argument("--window-s", type=float, default=1.5, help="next speech within this many seconds = continued")
    args = parser.parse_args(argv)
    events = [json.loads(line) for line in Path(args.events).read_text(encoding="utf-8").splitlines() if line.strip()]
    result = replay(events, predicate_for(args.config), window_s=args.window_s)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
