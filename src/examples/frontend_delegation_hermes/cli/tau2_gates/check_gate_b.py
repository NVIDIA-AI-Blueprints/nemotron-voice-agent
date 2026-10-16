# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""tau3 Gate B checker (plan section 18.0): verify a ``tau2 run`` against the stub server.

Stdlib only; point it at the run's ``results.json``::

    python check_gate_b.py <tau2 checkout>/data/simulations/<run>/results.json

Pass criteria per simulation: termination ``agent_stop`` (the scripted agent
transfers on turn 3) or ``user_stop``; at least one agent tool call and its
result on the ticks; a non-empty scored agent transcript.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def check(path: Path) -> list[str]:
    """Return the problems found in one results file."""
    data = json.loads(path.read_text(encoding="utf-8"))
    simulations = data.get("simulations") or []
    if not simulations:
        return ["no simulations in results.json"]
    problems: list[str] = []
    for sim in simulations:
        sid = sim.get("id", "?")
        reason = sim.get("termination_reason")
        if reason not in ("agent_stop", "user_stop"):
            problems.append(f"{sid}: termination_reason={reason}")
        ticks = sim.get("ticks") or []
        calls = [call for tick in ticks for call in tick.get("agent_tool_calls") or []]
        results = [result for tick in ticks for result in tick.get("agent_tool_results") or []]
        if not calls:
            problems.append(f"{sid}: no agent tool call on the ticks")
        if calls and not results:
            problems.append(f"{sid}: tool calls without tool results")
        spoken = "".join(str((tick.get("agent_chunk") or {}).get("content") or "") for tick in ticks)
        if not spoken.strip():
            problems.append(f"{sid}: empty scored agent transcript")
        print(f"{sid}: termination={reason} ticks={len(ticks)} tool_calls={len(calls)} transcript_chars={len(spoken)}")
    return problems


def main() -> int:
    """Check every path given on the command line."""
    problems = [problem for arg in sys.argv[1:] for problem in check(Path(arg))]
    if problems:
        print("GATE B FAILED:", *problems, sep="\n  ")
        return 1
    print("GATE B PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
