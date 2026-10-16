# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D103

"""G2: the Pipecat adapter makes v1's decisions on every golden scenario.

Plan ``misc/frontend-backend-verdict/shared-stack-plan.md`` §10.1 G2. Each scenario of
``_fbv_golden.py`` runs on :class:`PipecatHarness` (Pipecat stub speech, 20 ms input frames,
no output pacing) and is compared with v1's golden capture (``fixtures/v1_golden``) by
``_fbv_parity.compare``: events per delivery class, JSONL records, interim transcripts and
audio per item, with audio-clock fields inside the B1 window. What else a scenario may show
is listed below, case by case; anything not listed fails.
"""

from __future__ import annotations

import json
import unittest

from _fbv_golden import GOLDEN_DIR, SCENARIOS, pinned_environment, run_scenario
from _fbv_parity import Allowances, compare, decisions
from _fbv_pipecat_fakes import PipecatHarness

#: Per scenario: what the B1/B9 windows may move (see ``Allowances``).
ALLOWANCES: dict[str, Allowances] = {
    # The barge-in lands one start confirmation (start_secs) later on Pipecat's VAD, so the
    # caller has heard a little more of the answer, and the answer is cut at another point.
    "barge_in_after_answer": Allowances(cut_items=frozenset({"<ITEM3>"}), heard_text=frozenset({"history_repaired"})),
    "barge_in_during_answer": Allowances(cut_items=frozenset({"<ITEM3>"}), heard_text=frozenset({"history_repaired"})),
    "barge_in_filler": Allowances(cut_items=frozenset({"<ITEM3>"}), heard_text=frozenset({"history_repaired"})),
    "write_gate_cut": Allowances(
        cut_items=frozenset({"<ITEM3>", "<ITEM4>"}), heard_text=frozenset({"history_repaired"})
    ),
    # The cancel arrives after the first audio delta; how much input audio (playout) has
    # passed by then depends on when the pipeline ended the user turn.
    "response_cancel": Allowances(cut_items=frozenset({"<ITEM2>"}), heard_text=frozenset({"history_repaired"})),
    # The truncate is clamped to the audio sent; the playout cursor at that moment moves with
    # the end of the user turn (Silero's stop).
    "truncate_known_and_unknown": Allowances(heard_text=frozenset({"history_repaired"})),
}


def make_harness(**kwargs: object) -> PipecatHarness:
    return PipecatHarness(**kwargs)  # type: ignore[arg-type]


class WireParityTests(unittest.IsolatedAsyncioTestCase):
    async def test_every_golden_scenario(self) -> None:
        for name in SCENARIOS:
            with self.subTest(scenario=name), pinned_environment():
                got = await run_scenario(name, make_harness)
                want = json.loads((GOLDEN_DIR / f"scenario_{name}.json").read_text(encoding="utf-8"))
                problems = compare(decisions(want), decisions(got), ALLOWANCES.get(name, Allowances()))
                self.assertEqual(problems, [], f"{name} differs from v1 beyond the G2 tolerances")

    def test_allowances_name_known_scenarios(self) -> None:
        self.assertLessEqual(set(ALLOWANCES), set(SCENARIOS))


if __name__ == "__main__":
    unittest.main()
