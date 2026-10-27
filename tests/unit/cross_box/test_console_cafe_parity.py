# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D103

"""The Live Console's browser cafe (``cafe.js``) and the server's cafe tools give the same results.

``client/test/live/cafe-parity.json`` holds calls and the results ``tools.py`` gives them; this test runs them through
:class:`CafeTools`, and ``client/test/live/cafe.test.js`` through ``cafe.js``. After a deliberate change to
``tools.py``, record them again by running this test with ``CAFE_PARITY_RECORD=1``.
"""

import asyncio
import json
import os
import unittest
from pathlib import Path

from delegation_core.backend.tools.cafe.tools import CafeTools

FIXTURE = Path(__file__).resolve().parents[3] / "client/test/live/cafe-parity.json"


def without_clock(value):
    """``value`` with every ``placed_at`` replaced by ``"<placed_at>"``."""
    if isinstance(value, dict):
        return {key: "<placed_at>" if key == "placed_at" else without_clock(item) for key, item in value.items()}
    if isinstance(value, list):
        return [without_clock(item) for item in value]
    return value


async def play(calls: list[dict]) -> list:
    """Run ``calls`` in order on one fresh cart; return each result, parsed."""
    tools = CafeTools()
    return [without_clock(json.loads(await tools.execute(c["call_id"], c["name"], c["arguments"]))) for c in calls]


def dumps(fixture: dict) -> str:
    """The fixture as JSON with one call or one result per line, so a recorded change reads as one line."""

    def rows(values):
        return ",\n".join(f"        {json.dumps(value, ensure_ascii=False)}" for value in values)

    cases = [
        f'    {{\n      "name": {json.dumps(case["name"])},\n'
        f'      "calls": [\n{rows(case["calls"])}\n      ],\n'
        f'      "results": [\n{rows(case["results"])}\n      ]\n    }}'
        for case in fixture["cases"]
    ]
    about = json.dumps(fixture["about"], ensure_ascii=False)
    return f'{{\n  "about": {about},\n  "cases": [\n' + ",\n".join(cases) + "\n  ]\n}\n"


class CafeParityTests(unittest.TestCase):
    def test_the_server_cafe_gives_the_recorded_results(self):
        fixture = json.loads(FIXTURE.read_text())
        played = [asyncio.run(play(case["calls"])) for case in fixture["cases"]]
        if os.environ.get("CAFE_PARITY_RECORD") == "1":
            for case, results in zip(fixture["cases"], played, strict=True):
                case["results"] = results
            FIXTURE.write_text(dumps(fixture))
        for case, results in zip(fixture["cases"], played, strict=True):
            with self.subTest(case=case["name"]):
                self.assertEqual(results, case["results"])

    def test_every_case_records_one_result_per_call(self):
        for case in json.loads(FIXTURE.read_text())["cases"]:
            with self.subTest(case=case["name"]):
                self.assertEqual(len(case["results"]), len(case["calls"]))


if __name__ == "__main__":
    unittest.main()
